"""Strict privileged scorer for the RGB ordered-chain candidate preflight.

The CUDA backend is deliberately unable to read scoring-only ground truth.  It
seals two independent two-slot arms for each frozen Acrobot episode:
``real_rgb`` and the negative-control ``spatial_shuffle``.  Slot zero is the
published object-graph-v1 anchor and slot one is inferred from the current RGB
frame.  This scorer is run only after the scoring tree has been restored.

For every arm, slot zero is scored from the exact published-v1 role-mask bytes
after exact pose/validity parity checks.  Slot one is reconstructed twice from
its sealed pose and link widths: an entity-mask partition (diagnostic only) and
RGB-derived role capsules that are not clipped by the Cutie entity mask.  The
only development gate is the real-RGB capsule K=2 result.  Ground truth chooses
the best slot only for this privileged offline coverage diagnostic.  This file
can never authorize controller training or a scientific claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any

import numpy as np

from tdmpc2.common.unified_vos import (
    CONDITIONS,
    TASK_ROLES,
    canonical_json_bytes,
    file_sha256,
    load_json,
    require_sha256,
    resolve_member,
    sha256_json,
    validate_backend_inputs,
    write_json,
)
from tdmpc2.perception.ordered_chain_rgb import (
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    MAX_RENDERED_IK_CANDIDATES,
    MAX_TIP_PROPOSALS,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES as GENERATOR_SOURCE_CODES,
    render_role_capsules,
)
from tdmpc2.perception.ordered_chain_topk import role_masks_from_pose
from tdmpc2.tools.aggregate_object_graph_temporal_replay import (
    _contained_file,
    _regular_dir,
    _regular_file,
    _strict_v1_pair,
)
from tdmpc2.tools.aggregate_object_graph_tokenizer_preflight import (
    ARRAY_KEYS as V1_ARRAY_KEYS,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import _array_trace, _load_npz
from tdmpc2.tools.replay_object_graph_rgb_candidates import (
    ARMS,
    BACKEND,
    DEFAULT_DINO_INPUT_SIZE,
    DEFAULT_DINO_MODEL,
    DINO_EXTRACTOR_METADATA_KEYS,
    FORMAT as BACKEND_FORMAT,
    MANIFEST_KEYS,
    OUTPUT_ARRAY_KEYS,
    PROTOCOL as BACKEND_PROTOCOL,
    RESULT_RECORD_KEYS,
    SOURCE_CODES,
    _implementation_snapshot as _backend_implementation_snapshot,
    _python_tree_snapshot as _backend_dino_tree_snapshot,
)


TASK = "acrobot-swingup"
ROLES = TASK_ROLES[TASK]
SUMMARY_FORMAT = "object_graph_rgb_candidate_coverage_summary_v1"
ISOLATION_FORMAT = "object_graph_rgb_candidate_scoring_isolation_v1"
RENDERERS = ("rgb_partition", "rgb_capsule")
K_VALUES = (1, 2)

MIN_AVAILABILITY = 0.99
MIN_SUCCESS_AT_05 = 0.97
MAX_FAILURE_BURST = 10
MAX_PARSER_MEAN_MS = 5.0
MAX_PARSER_P95_MS = 10.0
MAX_END_TO_END_MEAN_MS = 12.0
MAX_END_TO_END_P95_MS = 15.0
MIN_SHUFFLE_SUCCESS_GAIN = 0.01
BASELINE_BURST_FRACTION = 0.50
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271828

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MODE = re.compile(r"[0-7]{3}")


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(
            f"{label} must be finite{' and positive' if positive else ''}."
        )
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be a whole-second UTC timestamp.")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):03o}"


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("A metric series is empty or non-finite.")
    return {
        "mean": float(array.mean()),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _latency(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
    ):
        raise ValueError("Latency must be a non-empty positive finite series.")
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _maximum_false_burst(success: np.ndarray) -> int:
    values = np.asarray(success, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError("Burst input must be one-dimensional.")
    current = 0
    maximum = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _iou(predicted: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    return float(intersection / union) if union else 0.0


def _candidate_quality(
    predicted: np.ndarray, gt_indexed: np.ndarray
) -> tuple[float, np.ndarray]:
    expected = (len(ROLES), *gt_indexed.shape)
    if predicted.shape != expected or predicted.dtype != np.bool_:
        raise ValueError("Candidate role-mask shape/dtype changed.")
    if gt_indexed.dtype != np.uint8:
        raise ValueError("Ground-truth mask dtype changed.")
    if np.any(predicted.sum(axis=0) > 1):
        raise ValueError("Candidate role masks overlap.")
    per_role = np.zeros(len(ROLES), dtype=np.float64)
    visible: list[int] = []
    for index in range(len(ROLES)):
        target = gt_indexed == index + 1
        if bool(target.any()):
            visible.append(index)
            per_role[index] = _iou(predicted[index], target)
    return (
        float(min(per_role[index] for index in visible)) if visible else 0.0,
        per_role,
    )


def _paired_bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, label: str
) -> dict[str, Any]:
    """Fixed episode-resampled descriptive interval (not a hypothesis test)."""
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or not lhs.size:
        raise ValueError("Paired bootstrap inputs are malformed.")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Paired bootstrap inputs are non-finite.")
    label_seed = int.from_bytes(label.encode("utf-8"), "little") % (2**32)
    seed = BOOTSTRAP_SEED ^ label_seed
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, lhs.size, size=(BOOTSTRAP_RESAMPLES, lhs.size))
    differences = (lhs[samples] - rhs[samples]).mean(axis=1)
    return {
        "unit": "episode",
        "paired": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "estimand": "mean_episode_metric_difference",
        "estimate": float((lhs - rhs).mean()),
        "descriptive_percentile_interval_95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "one_sided_lower_95": float(np.percentile(differences, 5.0)),
        "not_a_confirmatory_hypothesis_test": True,
    }


def _tree_snapshot(root: Path, label: str) -> dict[str, Any]:
    root = _regular_dir(root, label)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} contains a special member: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not rows:
        raise ValueError(f"{label} is empty.")
    return {
        "root": str(root),
        "file_count": len(rows),
        "files": rows,
        "tree_sha256": sha256_json(rows),
    }


def _require_unit_interval_array(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain one value per episode.")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} is malformed or non-finite.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{label} escaped [0,1].")
    return array


def _validate_mask_topk_baseline(
    root: Path,
    *,
    source_root: Path,
    v1_root: Path,
    episodes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _regular_dir(root, "published mask-only top-K preflight root")
    for other, name in (
        (source_root, "source"),
        (v1_root, "v1 preflight"),
    ):
        other = other.resolve(strict=True)
        if root == other or root in other.parents or other in root.parents:
            raise ValueError(f"Mask-only top-K and {name} roots must be disjoint.")
    summary_path = _regular_file(
        root / "object_graph_topk_candidate_coverage_summary.json",
        "published mask-only top-K summary",
    )
    payload = load_json(summary_path)
    if summary_path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("Published mask-only Top-K summary is not canonical JSON.")
    if (
        payload.get("format") != "object_graph_topk_candidate_coverage_summary_v1"
        or payload.get("engineering_pass") is not True
        or payload.get("controller_training_authorized") is not False
        or payload.get("scientific_go") is not False
        or payload.get("scope", {}).get("controller_training_steps") != 0
    ):
        raise ValueError("Published mask-only Top-K summary is ineligible.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Published mask-only Top-K provenance is missing.")
    source_summary = source_root / "unified_vos_summary.json"
    v1_summary = v1_root / "object_graph_tokenizer_summary.json"
    if (
        provenance.get("source_benchmark_root") != str(source_root.resolve(strict=True))
        or provenance.get("source_summary_sha256") != file_sha256(source_summary)
        or provenance.get("v1_preflight_root") != str(v1_root.resolve(strict=True))
        or provenance.get("v1_preflight_summary_sha256") != file_sha256(v1_summary)
    ):
        raise ValueError("Mask-only Top-K summary names different source/v1 roots.")
    coverage = payload.get("real_v1_entity_topk_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Mask-only Top-K real coverage is missing.")
    baseline: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_payload = coverage.get(condition)
        if not isinstance(condition_payload, dict):
            raise ValueError(f"Mask-only Top-K {condition} coverage is missing.")
        prefixes = condition_payload.get("prefixes")
        cell = prefixes.get("2") if isinstance(prefixes, dict) else None
        if not isinstance(cell, dict) or cell.get("k") != 2:
            raise ValueError(f"Mask-only Top-K {condition} K=2 cell is missing.")
        success = _require_unit_interval_array(
            cell.get("episode_success_at_0_5"),
            episodes,
            f"mask-only {condition} episode success",
        )
        quality = _require_unit_interval_array(
            cell.get("episode_best_min_role_iou_mean"),
            episodes,
            f"mask-only {condition} episode quality",
        )
        burst = cell.get("failure_burst_at_0_5")
        if not isinstance(burst, dict) or set(burst) != {
            "global_max_with_episode_resets",
            "episode_p95",
            "per_episode",
        }:
            raise ValueError(f"Mask-only Top-K {condition} burst schema changed.")
        per_episode = burst.get("per_episode")
        if (
            not isinstance(per_episode, list)
            or len(per_episode) != episodes
            or any(type(value) is not int or value < 0 for value in per_episode)
        ):
            raise ValueError(f"Mask-only Top-K {condition} burst values changed.")
        global_burst = burst.get("global_max_with_episode_resets")
        episode_p95 = _finite_number(
            burst.get("episode_p95"), f"mask-only {condition} episode-p95 burst"
        )
        if (
            type(global_burst) is not int
            or global_burst != max(per_episode)
            or not math.isclose(
                episode_p95,
                float(np.percentile(np.asarray(per_episode), 95)),
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Mask-only Top-K {condition} burst is inconsistent.")
        reported_success = _finite_number(
            cell.get("oracle_set_success_at_0_5"),
            f"mask-only {condition} success",
        )
        reported_quality = _finite_number(
            cell.get("best_min_role_iou", {}).get("mean"),
            f"mask-only {condition} quality",
        )
        if not math.isclose(reported_success, float(success.mean()), abs_tol=1e-12):
            raise ValueError(f"Mask-only Top-K {condition} success mean changed.")
        if not math.isclose(reported_quality, float(quality.mean()), abs_tol=1e-12):
            raise ValueError(f"Mask-only Top-K {condition} quality mean changed.")
        availability = _finite_number(
            cell.get("availability_rate"),
            f"mask-only {condition} availability",
        )
        if not 0.0 <= availability <= 1.0:
            raise ValueError(f"Mask-only Top-K {condition} availability escaped [0,1].")
        baseline[condition] = {
            "availability_rate": availability,
            "oracle_set_success_at_0_5": reported_success,
            "best_min_role_iou_mean": reported_quality,
            "failure_burst_at_0_5": burst,
            "episode_success_at_0_5": success,
            "episode_best_min_role_iou_mean": quality,
        }
    tree = _tree_snapshot(root, "published mask-only top-K preflight root")
    binding = {
        "root": str(root),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "tree_sha256": tree["tree_sha256"],
        "tree_file_count": tree["file_count"],
    }
    return payload, baseline, {"binding": binding, "tree": tree}


def _validate_isolation_gate(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    mask_topk_preflight_root: Path,
    summary_root: Path,
    backend_manifest_path: Path,
) -> dict[str, Any]:
    path = _regular_file(path, "RGB-candidate scoring isolation")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative_gate = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("RGB isolation gate is outside the summary root.") from exc
    if relative_gate != "provenance/scoring_isolation.json":
        raise ValueError("RGB isolation gate is not at its fixed path.")
    payload = load_json(path)
    expected = {
        "format",
        "status",
        "source_benchmark_root",
        "v1_preflight_root",
        "mask_topk_preflight_root",
        "worker_input_root",
        "scoring_root",
        "scoring_probe_relative_to_source",
        "root_mode_before",
        "root_mode_locked",
        "root_mode_restored",
        "probe_sha256_before",
        "probe_sha256_restored",
        "locked_utc",
        "backend_completed_utc",
        "restored_utc",
        "backend_completed_before_restore",
        "backend_device",
        "cuda_visible_devices",
        "BENCHMARK_GPU_UUID",
        "rgb_backend_manifest_relative_to_summary_root",
        "rgb_backend_manifest_sha256",
        "same_uid_read_probe",
    }
    if set(payload) != expected or path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("RGB scoring-isolation schema/canonical encoding changed.")
    source_root = source_root.resolve(strict=True)
    v1_preflight_root = v1_preflight_root.resolve(strict=True)
    mask_topk_preflight_root = mask_topk_preflight_root.resolve(strict=True)
    worker_root = (source_root / "worker_inputs").resolve(strict=True)
    scoring_root = (source_root / "dataset" / "scoring").resolve(strict=True)
    gpu_uuid = payload.get("BENCHMARK_GPU_UUID")
    if (
        payload.get("format") != ISOLATION_FORMAT
        or payload.get("status") != "complete"
        or payload.get("source_benchmark_root") != str(source_root)
        or payload.get("v1_preflight_root") != str(v1_preflight_root)
        or payload.get("mask_topk_preflight_root") != str(mask_topk_preflight_root)
        or payload.get("worker_input_root") != str(worker_root)
        or payload.get("scoring_root") != str(scoring_root)
        or payload.get("backend_completed_before_restore") is not True
        or payload.get("backend_device") != "cuda:0"
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or payload.get("cuda_visible_devices") != gpu_uuid
        or worker_root == scoring_root
        or worker_root in scoring_root.parents
        or scoring_root in worker_root.parents
    ):
        raise ValueError("RGB isolation identity, GPU pairing, or causality changed.")
    relative_probe = payload.get("scoring_probe_relative_to_source")
    if not isinstance(relative_probe, str) or not relative_probe or "\\" in relative_probe:
        raise ValueError("RGB scoring-probe path is malformed.")
    probe = _contained_file(source_root, relative_probe, "RGB scoring probe")
    try:
        probe.relative_to(scoring_root)
    except ValueError as exc:
        raise ValueError("RGB scoring probe is outside scoring-only data.") from exc
    for field in ("root_mode_before", "root_mode_locked", "root_mode_restored"):
        if not isinstance(payload.get(field), str) or _MODE.fullmatch(payload[field]) is None:
            raise ValueError(f"RGB isolation mode {field} is malformed.")
    if (
        payload["root_mode_locked"] != "000"
        or payload["root_mode_before"] != payload["root_mode_restored"]
        or payload["root_mode_before"] == "000"
        or int(payload["root_mode_before"], 8) & 0o500 != 0o500
        or _mode(scoring_root) != payload["root_mode_restored"]
    ):
        raise ValueError("RGB scoring permissions were not restored exactly.")
    before_sha = require_sha256(payload.get("probe_sha256_before"), "pre-lock probe SHA")
    restored_sha = require_sha256(
        payload.get("probe_sha256_restored"), "restored probe SHA"
    )
    if before_sha != restored_sha or file_sha256(probe) != restored_sha:
        raise ValueError("RGB scoring probe changed across the lock.")
    times = [
        _parse_utc(payload[field], field)
        for field in ("locked_utc", "backend_completed_utc", "restored_utc")
    ]
    if times != sorted(times):
        raise ValueError("RGB isolation timestamps are out of order.")
    backend_manifest_path = backend_manifest_path.resolve(strict=True)
    try:
        expected_relative = backend_manifest_path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("RGB backend manifest is outside the summary root.") from exc
    if (
        expected_relative
        != "backends/object_graph_rgb_candidates/backend_predictions.json"
        or payload.get("rgb_backend_manifest_relative_to_summary_root")
        != expected_relative
        or payload.get("rgb_backend_manifest_sha256")
        != file_sha256(backend_manifest_path)
    ):
        raise ValueError("RGB isolation names a different backend manifest.")
    read_probe = payload.get("same_uid_read_probe")
    if not isinstance(read_probe, dict) or set(read_probe) != {
        "exit_code",
        "error_type",
        "log_relative_to_summary_root",
        "log_sha256",
    }:
        raise ValueError("RGB same-user read-probe schema changed.")
    if (
        type(read_probe.get("exit_code")) is not int
        or read_probe.get("exit_code") != 1
        or read_probe.get("error_type") != "PermissionError"
        or read_probe.get("log_relative_to_summary_root")
        != "contracts/scoring_read_probe.log"
    ):
        raise ValueError("RGB scoring read probe did not fail closed.")
    log_path = _contained_file(
        summary_root,
        read_probe["log_relative_to_summary_root"],
        "RGB scoring read-probe log",
    )
    if (
        file_sha256(log_path)
        != require_sha256(read_probe.get("log_sha256"), "RGB read-probe log SHA")
        or b"PermissionError" not in log_path.read_bytes()
    ):
        raise ValueError("RGB read-probe log is not the bound permission failure.")
    return payload


def _validate_immutable_inputs(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    mask_topk_preflight_root: Path,
    dino_repo: Path,
    dino_checkpoint: Path,
    summary_root: Path,
) -> dict[str, Any]:
    # Lazy import avoids a constants-only import cycle: the snapshot records
    # this aggregator/backend/core contract in its own immutable inventory.
    from tdmpc2.common.object_graph_rgb_snapshot import (
        FORMAT as IMMUTABLE_INPUTS_FORMAT,
        build as build_immutable_inputs,
    )

    path = _regular_file(path, "RGB immutable-input snapshot")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("RGB immutable inputs are outside the summary root.") from exc
    if relative != "provenance/immutable_inputs.json":
        raise ValueError("RGB immutable inputs are not at their fixed path.")
    payload = load_json(path)
    rebuilt = build_immutable_inputs(
        source_root=source_root,
        v1_root=v1_preflight_root,
        mask_topk_root=mask_topk_preflight_root,
        dino_repo=dino_repo,
        dino_checkpoint=dino_checkpoint,
    )
    scope = payload.get("scope")
    if (
        payload.get("format") != IMMUTABLE_INPUTS_FORMAT
        or payload != rebuilt
        or path.read_bytes() != canonical_json_bytes(payload)
        or not isinstance(scope, dict)
        or scope.get("episode_ground_truth_available_to_backend") is not False
        or scope.get("current_rgb_backend_input") is not True
        or scope.get("fixed_support_rgb_backend_input") is not True
        or scope.get("fixed_labelled_support_masks_backend_input") is not True
        or scope.get("backend_temporal_state") is not False
        or scope.get("controller_training_steps") != 0
        or scope.get("controller_training_authorized") is not False
    ):
        raise ValueError("RGB immutable-input snapshot changed.")
    return payload


def _rgb_array_schema(
    *, frames: int
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    candidate = (frames, 2)
    expected: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "poses_xy": ((frames, 2, 3, 2), np.dtype(np.float32)),
        "link_half_widths_px": ((frames, 2, 2), np.dtype(np.float32)),
        "candidate_valid": (candidate, np.dtype(np.bool_)),
        "candidate_cost": (candidate, np.dtype(np.float32)),
        "candidate_weight": (candidate, np.dtype(np.float32)),
        "candidate_confidence": (candidate, np.dtype(np.float32)),
        "rgb_evidence_score": (candidate, np.dtype(np.float32)),
        "mask_geometry_score": (candidate, np.dtype(np.float32)),
        "candidate_source_code": (candidate, np.dtype(np.uint8)),
        "roi_xyxy": ((frames, 4), np.dtype(np.int16)),
        "parser_runtime_ms": ((frames,), np.dtype(np.float64)),
    }
    if set(expected) != set(OUTPUT_ARRAY_KEYS):
        raise RuntimeError("RGB aggregator/backend array constants diverged.")
    return expected


def _validate_decoded_rgb_arrays(
    arrays: dict[str, np.ndarray],
    *,
    frames: int,
    resolution: int,
    expected_widths: np.ndarray,
    expected_lengths: np.ndarray,
) -> None:
    schema = _rgb_array_schema(frames=frames)
    if set(arrays) != set(schema):
        raise ValueError("Decoded RGB-candidate array schema changed.")
    for name, (shape, dtype) in schema.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Decoded RGB {name} is {value.shape}/{value.dtype}, "
                f"expected {shape}/{dtype}."
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Decoded RGB {name} contains non-finite values.")
    valid = arrays["candidate_valid"]
    source = arrays["candidate_source_code"]
    poses = arrays["poses_xy"]
    widths = arrays["link_half_widths_px"]
    costs = arrays["candidate_cost"]
    weights = arrays["candidate_weight"]
    confidence = arrays["candidate_confidence"]
    rgb_score = arrays["rgb_evidence_score"]
    mask_score = arrays["mask_geometry_score"]
    frozen_widths = np.asarray(expected_widths, dtype=np.float32)
    frozen_lengths = np.asarray(expected_lengths, dtype=np.float64)
    if (
        frozen_widths.shape != (len(ROLES),)
        or frozen_lengths.shape != (len(ROLES),)
        or not np.isfinite(frozen_widths).all()
        or not np.isfinite(frozen_lengths).all()
        or np.any(frozen_widths <= 0.0)
        or np.any(frozen_lengths <= 0.0)
    ):
        raise ValueError("Frozen RGB support geometry is malformed.")
    if set(SOURCE_CODES) != {
        "padding",
        "v1_anchor",
        "rgb_ik_positive",
        "rgb_ik_negative",
    }:
        raise RuntimeError("RGB source-code ontology changed.")
    rgb_codes = np.asarray(
        [SOURCE_CODES["rgb_ik_positive"], SOURCE_CODES["rgb_ik_negative"]],
        dtype=np.uint8,
    )
    if (
        int(SOURCE_CODES["padding"]) != 0
        or np.any(valid[:, 0] & (source[:, 0] != SOURCE_CODES["v1_anchor"]))
        or np.any(valid[:, 1] & ~np.isin(source[:, 1], rgb_codes))
        or np.any(source[~valid] != SOURCE_CODES["padding"])
        or not np.isin(
            source, np.asarray(sorted(SOURCE_CODES.values()), dtype=np.uint8)
        ).all()
    ):
        raise ValueError("RGB candidate source-code/slot identity changed.")
    if np.any(poses[~valid] != 0.0) or np.any(widths[~valid] != 0.0):
        raise ValueError("Invalid RGB candidates are not pose/width zero padded.")
    if np.any(widths[valid] <= 0.0):
        raise ValueError("Valid RGB candidates require positive link half-widths.")
    expected_width_rows = np.broadcast_to(
        frozen_widths, (frames, MAX_CANDIDATES, len(ROLES))
    )
    if not np.array_equal(widths[valid], expected_width_rows[valid]):
        raise ValueError("Valid RGB capsule widths changed from fixed support.")
    link_lengths = np.linalg.norm(np.diff(poses, axis=2), axis=3)
    if np.any(link_lengths[valid] <= 1e-9):
        raise ValueError("Valid RGB candidates require non-degenerate links.")
    slot_one_valid = valid[:, 1]
    expected_length_rows = np.broadcast_to(
        frozen_lengths, (frames, len(ROLES))
    )
    if not np.allclose(
        link_lengths[:, 1][slot_one_valid],
        expected_length_rows[slot_one_valid],
        rtol=1e-5,
        atol=1e-3,
    ):
        raise ValueError("Valid RGB IK link lengths changed from fixed support.")
    f32_max = np.finfo(np.float32).max
    if np.any(costs[~valid] != f32_max) or np.any(costs[valid] < 0.0):
        raise ValueError("RGB candidate cost/padding contract changed.")
    for name, value in (
        ("candidate_weight", weights),
        ("candidate_confidence", confidence),
        ("rgb_evidence_score", rgb_score),
        ("mask_geometry_score", mask_score),
    ):
        if np.any((value < 0.0) | (value > 1.0)) or np.any(value[~valid] != 0.0):
            raise ValueError(f"RGB {name} range/padding contract changed.")
    counts = valid.sum(axis=1)
    sums = weights.sum(axis=1)
    if not np.allclose(sums[counts > 0], 1.0, atol=2e-6, rtol=0.0) or np.any(
        sums[counts == 0] != 0.0
    ):
        raise ValueError("RGB candidate weights do not sum to one on active slots.")
    if np.any(arrays["parser_runtime_ms"] <= 0.0):
        raise ValueError("RGB parser runtime must be positive.")
    roi = arrays["roi_xyxy"].astype(np.int64)
    if np.any(roi < 0) or np.any(roi > resolution):
        raise ValueError("RGB ROI escaped the frozen frame.")
    nonempty = (roi[:, 2] > roi[:, 0]) & (roi[:, 3] > roi[:, 1])
    if np.any(~nonempty):
        raise ValueError("Every RGB replay frame must have a nonempty ROI.")


def _validate_rank_zero_exact(
    *, arrays: dict[str, np.ndarray], source_v1: dict[str, np.ndarray]
) -> str:
    valid = arrays["candidate_valid"]
    source = arrays["candidate_source_code"]
    poses = arrays["poses_xy"]
    expected_valid = source_v1["role_valid"].all(axis=1)
    if not np.array_equal(valid[:, 0], expected_valid):
        raise ValueError("RGB slot-zero validity differs from published v1.")
    expected_source = np.where(
        expected_valid, SOURCE_CODES["v1_anchor"], SOURCE_CODES["padding"]
    ).astype(np.uint8)
    if not np.array_equal(source[:, 0], expected_source):
        raise ValueError("RGB slot zero is not the reserved v1 anchor.")
    expected_pose = np.zeros_like(poses[:, 0], dtype=np.float32)
    expected_pose[expected_valid, 0] = source_v1["keypoints_xy"][
        expected_valid, 0, 0
    ]
    expected_pose[expected_valid, 1:] = source_v1["keypoints_xy"][
        expected_valid, :, 1
    ]
    if not np.array_equal(poses[:, 0], expected_pose):
        raise ValueError("RGB slot-zero pose differs from published-v1 keypoints.")
    if np.any(source_v1["role_masks"].sum(axis=1) > 1):
        raise ValueError("Published-v1 role masks overlap.")
    return _array_trace(source_v1["role_masks"])


def _frame_score_rows(
    *,
    arrays: dict[str, np.ndarray],
    source_v1: dict[str, np.ndarray],
    gt_indexed: np.ndarray,
    renderer: str,
) -> dict[str, np.ndarray]:
    if renderer not in RENDERERS:
        raise ValueError(f"Unknown RGB scorer reconstruction: {renderer}")
    frames, resolution, width = gt_indexed.shape
    if resolution != width:
        raise ValueError("RGB scorer requires square frozen inputs.")
    valid = arrays["candidate_valid"]
    if valid.shape != (frames, 2):
        raise ValueError("RGB candidate validity shape changed.")
    entity_masks = source_v1["entity_masks"][:, 0]
    if entity_masks.shape != (frames, resolution, resolution):
        raise ValueError("Published-v1 entity-mask shape changed.")
    if np.any(valid & ~source_v1["entity_valid"][:, :1]):
        raise ValueError("RGB emitted a candidate while its v1 entity was unavailable.")
    quality = np.zeros((frames, 2), dtype=np.float64)
    role_iou = np.zeros((frames, 2, len(ROLES)), dtype=np.float64)
    for frame_index in range(frames):
        for slot in range(2):
            if not bool(valid[frame_index, slot]):
                continue
            if slot == 0:
                predicted = np.ascontiguousarray(
                    source_v1["role_masks"][frame_index], dtype=np.bool_
                )
            elif renderer == "rgb_partition":
                predicted = role_masks_from_pose(
                    np.ascontiguousarray(entity_masks[frame_index], dtype=np.bool_),
                    arrays["poses_xy"][frame_index, slot],
                )
                if not np.array_equal(predicted.any(axis=0), entity_masks[frame_index]):
                    raise ValueError("RGB partition does not cover its current entity mask.")
            else:
                predicted = render_role_capsules(
                    arrays["poses_xy"][frame_index, slot],
                    arrays["link_half_widths_px"][frame_index, slot],
                    (resolution, resolution),
                )
            predicted = np.ascontiguousarray(predicted, dtype=np.bool_)
            if predicted.shape != (len(ROLES), resolution, resolution):
                raise ValueError("RGB reconstructed role-mask shape changed.")
            value, per_role = _candidate_quality(predicted, gt_indexed[frame_index])
            quality[frame_index, slot] = value
            role_iou[frame_index, slot] = per_role
    return {
        "quality": quality,
        "role_iou": role_iou,
        "valid": np.ascontiguousarray(valid, dtype=np.bool_),
    }


def _score_prefixes(episodes: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("No episodes were supplied for RGB scoring.")
    output: dict[str, Any] = {}
    for k in K_VALUES:
        best_rows: list[np.ndarray] = []
        availability_rows: list[np.ndarray] = []
        per_role_rows: list[np.ndarray] = []
        episode_success: list[float] = []
        episode_quality: list[float] = []
        episode_bursts: list[int] = []
        for episode in episodes:
            quality = episode["quality"][:, :k]
            role_iou = episode["role_iou"][:, :k]
            valid = episode["valid"][:, :k]
            if quality.shape != valid.shape or role_iou.shape != (*valid.shape, 2):
                raise ValueError("RGB score matrix shapes changed.")
            available = valid.any(axis=1)
            best = np.where(valid, quality, -1.0).max(axis=1)
            best = np.where(available, best, 0.0)
            best_role = np.zeros((len(best), len(ROLES)), dtype=np.float64)
            for frame_index in range(len(best)):
                if not bool(available[frame_index]):
                    continue
                for role_index in range(len(ROLES)):
                    best_role[frame_index, role_index] = float(
                        role_iou[frame_index, :, role_index][valid[frame_index]].max()
                    )
            success = available & (best >= 0.5)
            best_rows.append(best)
            availability_rows.append(available)
            per_role_rows.append(best_role)
            episode_success.append(float(success.mean()))
            episode_quality.append(float(best.mean()))
            episode_bursts.append(_maximum_false_burst(success))
        best = np.concatenate(best_rows)
        available = np.concatenate(availability_rows)
        best_role = np.concatenate(per_role_rows)
        bursts = np.asarray(episode_bursts, dtype=np.int64)
        output[str(k)] = {
            "k": k,
            "prefix_slots": list(range(k)),
            "frames": int(best.size),
            "episodes": len(episodes),
            "availability_rate": float(available.mean()),
            "oracle_set_success_at_0_5": float((available & (best >= 0.5)).mean()),
            "best_min_role_iou": _percentile_summary(best),
            "best_per_role_iou_mean": {
                role: float(best_role[:, index].mean())
                for index, role in enumerate(ROLES)
            },
            "failure_burst_at_0_5": {
                "global_max_with_episode_resets": int(bursts.max()),
                "episode_p95": float(np.percentile(bursts, 95)),
                "per_episode": [int(value) for value in bursts],
            },
            "episode_success_at_0_5": episode_success,
            "episode_best_min_role_iou_mean": episode_quality,
        }
    return {"prefixes": output}


def _expected_backend_protocol() -> dict[str, Any]:
    return {
        "format": BACKEND_PROTOCOL,
        "diagnostic_scope": "single_task_acrobot_development_only_v1",
        "single_task_development": True,
        "source_backend": "object_graph_cutie",
        "episode_rgb_decoded": True,
        "episode_rgb_schema": "exact_rgb_uint8_T_H_W_3_only",
        "episode_rgb_loaded_before_replay": True,
        "episode_v1_arrays_loaded_before_replay": True,
        "current_rgb_generator_input": True,
        "current_entity_mask_generator_input": True,
        "current_entity_available_generator_input": True,
        "previous_frames_generator_input": False,
        "future_frames_generator_input": False,
        "episode_state": False,
        "episode_ground_truth_read": False,
        "simulator_state_read": False,
        "actions_read": False,
        "rewards_read": False,
        "frozen_support_rgb_consumed": True,
        "frozen_support_indexed_masks_consumed": True,
        "source_v1_role_masks_consumed": False,
        "source_v1_entity_mask_consumed": True,
        "source_v1_keypoints_consumed_for_slot_zero_audit_only": True,
        "source_v1_role_valid_consumed_for_slot_zero_audit_only": True,
        "source_v1_entity_confidence_decoded_not_generator_input": True,
        "source_v1_entity_lost_decoded_not_generator_input": True,
        "source_v1_entity_mask_score_decoded_not_generator_input": True,
        "source_v1_role_confidence_lost_mask_score_decoded_for_trace_only": True,
        "real_rgb_and_spatial_shuffle_same_current_frame": True,
        "spatial_shuffle_owned_by_generator": True,
        "spatial_shuffle_protocol_bound_in_generator_metadata": True,
        "slot_zero_exact_v1_in_both_arms": True,
        "candidate_weights_noncalibrated": True,
        "candidate_confidences_noncalibrated": True,
        "controller_tokens_emitted": False,
        "controller_training_eligible": False,
        "gpu_only": True,
        "runtime_ms_semantics": (
            "ordered_chain_rgb_generator_internal_per_current_frame_"
            "including_feature_inference_v1"
        ),
    }


def _expected_output_disclosure() -> dict[str, Any]:
    return {
        "array_names": sorted(OUTPUT_ARRAY_KEYS),
        "poses_xy": "float32[T,2,3,2]_xy_pixels",
        "link_half_widths_px": "float32[T,2,2]_sealed_capsule_half_widths",
        "roi_xyxy": "int16[T,4]_exclusive_upper_bounds",
        "slot_zero": "exact_published_v1_current_mask_anchor_or_padding",
        "slot_one": "current_frame_frozen_dinov2_rgb_candidate_or_padding",
        "source_codes": dict(SOURCE_CODES),
        "role_masks_materialized": False,
        "sealed_role_mask_reconstruction": (
            "two_nonoverlapping_role_capsules_from_sealed_pose_and_width_"
            "unclipped_by_entity_mask_v1"
        ),
    }


def _validate_generator_metadata(
    metadata: Any,
    *,
    graph: Any,
    support_record: dict[str, Any],
    resolution: int,
) -> None:
    expected_keys = {
        "format",
        "protocol",
        "graph_sha256",
        "entity_name",
        "role_names",
        "max_candidates",
        "maximum_tip_proposals",
        "maximum_rendered_ik_candidates",
        "candidate_slot_semantics",
        "source_codes",
        "support_rgb_trace_sha256",
        "support_indexed_mask_trace_sha256",
        "prototype_trace_sha256",
        "landmark_names",
        "support_resolution",
        "support_link_lengths_px",
        "support_link_half_widths_px",
        "rgb_reliability_threshold",
        "relative_weights_not_calibrated",
        "current_frame_only",
        "episode_state",
        "feature_extractor_stateless_contract",
        "fixed_support_feature_replay_exact",
        "future_frames",
        "action_input",
        "reward_input",
        "simulator_state_input",
        "episode_ground_truth_input",
        "fixed_labelled_support_masks_input",
        "task_name_dispatch",
        "capsule_rendering_uses_entity_mask",
        "spatial_shuffle_policy",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ValueError("RGB generator metadata schema changed.")
    if (
        metadata.get("format") != GENERATOR_FORMAT
        or metadata.get("protocol") != GENERATOR_PROTOCOL
        or metadata.get("graph_sha256") != graph.graph_sha256
        or metadata.get("entity_name") != graph.entities[0].name
        or metadata.get("role_names") != list(ROLES)
        or metadata.get("max_candidates") != MAX_CANDIDATES
        or metadata.get("maximum_tip_proposals") != MAX_TIP_PROPOSALS
        or metadata.get("maximum_rendered_ik_candidates")
        != MAX_RENDERED_IK_CANDIDATES
        or metadata.get("candidate_slot_semantics")
        != [
            "exact_current_mask_v1_anchor",
            "best_reliable_distinct_current_rgb_ik_mode_or_invalid",
        ]
        or metadata.get("source_codes") != dict(SOURCE_CODES)
        or dict(GENERATOR_SOURCE_CODES) != dict(SOURCE_CODES)
        or metadata.get("support_rgb_trace_sha256")
        != support_record.get("rgb_trace_sha256")
        or metadata.get("support_indexed_mask_trace_sha256")
        != support_record.get("indexed_masks_trace_sha256")
        or metadata.get("support_resolution") != [resolution, resolution]
        or metadata.get("relative_weights_not_calibrated") is not True
        or metadata.get("current_frame_only") is not True
        or metadata.get("episode_state") is not False
        or metadata.get("feature_extractor_stateless_contract") is not True
        or metadata.get("fixed_support_feature_replay_exact") is not True
        or metadata.get("future_frames") is not False
        or metadata.get("action_input") is not False
        or metadata.get("reward_input") is not False
        or metadata.get("simulator_state_input") is not False
        or metadata.get("episode_ground_truth_input") is not False
        or metadata.get("fixed_labelled_support_masks_input") is not True
        or metadata.get("task_name_dispatch") is not False
        or metadata.get("capsule_rendering_uses_entity_mask") is not False
        or metadata.get("spatial_shuffle_policy")
        != "deterministic_roi_pixel_permutation_preserving_rgb_histogram_v1"
    ):
        raise ValueError("RGB generator metadata identity/causality changed.")
    for field in (
        "prototype_trace_sha256",
        "support_rgb_trace_sha256",
        "support_indexed_mask_trace_sha256",
    ):
        require_sha256(metadata.get(field), f"RGB generator {field}")
    if metadata.get("landmark_names") != [
        "root",
        "first_midpoint",
        "joint",
        "second_midpoint",
        "tip",
    ]:
        raise ValueError("RGB landmark ontology changed.")
    for field in ("support_link_lengths_px", "support_link_half_widths_px"):
        values = metadata.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(_finite_number(value, field, positive=True) <= 0.0 for value in values)
        ):
            raise ValueError(f"RGB generator {field} changed.")
    reliability = _finite_number(
        metadata.get("rgb_reliability_threshold"), "RGB reliability threshold"
    )
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("RGB reliability threshold escaped [0,1].")


def _validate_rgb_backend(
    manifest_path: Path,
    *,
    dataset: dict[str, Any],
    dataset_root: Path,
    v1_payload: dict[str, Any],
    v1_paths: dict[tuple[str, str, int], Path],
    v1_manifest_path: Path,
    v1_graph: Any,
    dino_repo: Path,
    dino_checkpoint: Path,
    isolation: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Path]]:
    manifest_path = _regular_file(manifest_path, "RGB replay backend manifest")
    payload = load_json(manifest_path)
    if set(payload) != set(MANIFEST_KEYS) or manifest_path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("RGB backend manifest schema/canonical encoding changed.")
    worker_input = _regular_file(
        dataset_root.parent / "worker_inputs" / "backend_inputs.json",
        "GT-free worker input manifest",
    )
    worker, support_paths, rgb_paths = validate_backend_inputs(
        worker_input, strict_counts=True
    )
    if (
        payload.get("format") != BACKEND_FORMAT
        or payload.get("status") != "complete"
        or payload.get("backend") != BACKEND
        or payload.get("task") != TASK
        or payload.get("development_scope")
        != "single_task_acrobot_offline_candidate_preflight_v1"
        or payload.get("dataset_id") != dataset.get("dataset_id")
        or payload.get("dataset_id") != worker.get("dataset_id")
        or payload.get("input_manifest_sha256") != file_sha256(worker_input)
        or payload.get("input_manifest_sha256")
        != dataset.get("backend_inputs", {}).get("sha256")
        or payload.get("v1_backend_manifest_sha256") != file_sha256(v1_manifest_path)
        or payload.get("roles") != list(ROLES)
        or payload.get("arms") != list(ARMS)
        or payload.get("max_candidates") != 2
        or MAX_CANDIDATES != 2
    ):
        raise ValueError("RGB backend identity/source pairing changed.")
    graph_record = payload.get("graph")
    if not isinstance(graph_record, dict) or set(graph_record) != {
        "path",
        "file_sha256",
        "graph",
    } or (
        graph_record.get("path") != str(v1_graph.source_path)
        or graph_record.get("file_sha256") != file_sha256(v1_graph.source_path)
        or graph_record.get("graph") != v1_graph.metadata()
    ):
        raise ValueError("RGB backend graph provenance changed.")
    resolution = int(dataset["resolution"])
    support_record = payload.get("support")
    support_path = support_paths[TASK]
    dataset_support_path = resolve_member(
        dataset_root,
        dataset["support"][TASK]["arrays"],
        "RGB dataset support calibration arrays",
    )
    if not isinstance(support_record, dict) or set(support_record) != {
        "path",
        "file_sha256",
        "rgb_trace_sha256",
        "indexed_masks_trace_sha256",
    } or (
        support_record.get("path") != str(support_path)
        or support_record.get("file_sha256") != file_sha256(support_path)
        or support_path == dataset_support_path
        or file_sha256(support_path) != file_sha256(dataset_support_path)
        or file_sha256(support_path)
        != dataset["support"][TASK].get("arrays_sha256")
        or file_sha256(support_path)
        != worker["support"][TASK].get("arrays_sha256")
    ):
        raise ValueError("RGB backend support provenance changed.")
    support_arrays = _load_npz(support_path, {"rgb", "indexed_masks"})
    support_rgb = support_arrays["rgb"]
    support_masks = support_arrays["indexed_masks"]
    if (
        support_rgb.shape != (6, resolution, resolution, 3)
        or support_rgb.dtype != np.uint8
        or support_masks.shape != (6, resolution, resolution)
        or support_masks.dtype == np.bool_
        or not np.issubdtype(support_masks.dtype, np.integer)
        or set(int(value) for value in np.unique(support_masks)) - {0, 1, 2}
        or not {1, 2}.issubset(
            set(int(value) for value in np.unique(support_masks))
        )
        or _array_trace(support_rgb)
        != support_record.get("rgb_trace_sha256")
        or _array_trace(support_masks)
        != support_record.get("indexed_masks_trace_sha256")
    ):
        raise ValueError("RGB backend decoded support traces changed.")
    generator_metadata = payload.get("generator")
    _validate_generator_metadata(
        generator_metadata,
        graph=v1_graph,
        support_record=support_record,
        resolution=resolution,
    )
    if payload.get("output_schema") != _expected_output_disclosure():
        raise ValueError("RGB backend output-schema disclosure changed.")
    if payload.get("protocol") != _expected_backend_protocol():
        raise ValueError("RGB backend GT-free protocol changed.")
    dino = payload.get("dino")
    dino_repo = _regular_dir(dino_repo, "DINOv2 repository")
    dino_checkpoint = _regular_file(dino_checkpoint, "DINOv2 checkpoint")
    if not isinstance(dino, dict) or set(dino) != {
        "model",
        "input_size",
        "repo",
        "python_source_tree",
        "checkpoint",
        "checkpoint_sha256",
        "extractor",
    } or (
        dino.get("model") != DEFAULT_DINO_MODEL
        or dino.get("input_size") != DEFAULT_DINO_INPUT_SIZE
        or dino.get("repo") != str(dino_repo)
        or dino.get("python_source_tree") != _backend_dino_tree_snapshot(dino_repo)
        or dino.get("checkpoint") != str(dino_checkpoint)
        or dino.get("checkpoint_sha256") != file_sha256(dino_checkpoint)
    ):
        raise ValueError("RGB backend frozen DINOv2 provenance changed.")
    extractor = dino.get("extractor")
    if not isinstance(extractor, dict) or set(extractor) != DINO_EXTRACTOR_METADATA_KEYS or (
        extractor.get("format") != "frozen_dinov2_dense_feature_extractor_v1"
        or extractor.get("repo") != str(dino_repo)
        or extractor.get("checkpoint") != str(dino_checkpoint)
        or extractor.get("model_name") != DEFAULT_DINO_MODEL
        or extractor.get("input_size") != DEFAULT_DINO_INPUT_SIZE
        or extractor.get("device") != "cuda:0"
        or extractor.get("frozen_parameters") is not True
        or extractor.get("evaluation_mode") is not True
        or extractor.get("torch_hub_source_local") is not True
        or extractor.get("pretrained_constructor_download") is not False
        or extractor.get("network_isolation_enforced") is not False
    ):
        raise ValueError("RGB backend DINOv2 extractor contract changed.")
    provenance = payload.get("backend_provenance")
    expected_provenance = {
        "treatment",
        "implementation",
        "source_v1_backend_manifest",
        "source_v1_backend_manifest_sha256",
        "source_worker_inputs",
        "source_worker_inputs_sha256",
        "source_support_arrays",
        "source_support_arrays_sha256",
        "v1_graph_file_sha256",
        "v1_graph_semantic_sha256",
        "python",
        "platform",
        "numpy",
        "torch",
        "cuda_runtime",
        "cuda_visible_devices",
        "cuda_device_order",
        "gpu_uuid",
        "logical_cuda_device",
        "device_name",
        "wallclock_seconds",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("RGB backend execution provenance schema changed.")
    text_fields = ("torch", "cuda_runtime", "device_name")
    if (
        provenance.get("treatment")
        != "frozen_dinov2_current_rgb_ordered_chain_candidate_v1"
        or provenance.get("implementation") != _backend_implementation_snapshot()
        or provenance.get("source_v1_backend_manifest") != str(v1_manifest_path)
        or provenance.get("source_v1_backend_manifest_sha256")
        != file_sha256(v1_manifest_path)
        or provenance.get("source_worker_inputs") != str(worker_input)
        or provenance.get("source_worker_inputs_sha256") != file_sha256(worker_input)
        or provenance.get("source_support_arrays") != str(support_path)
        or provenance.get("source_support_arrays_sha256") != file_sha256(support_path)
        or provenance.get("v1_graph_file_sha256") != file_sha256(v1_graph.source_path)
        or provenance.get("v1_graph_semantic_sha256") != v1_graph.graph_sha256
        or provenance.get("python") != sys.version
        or provenance.get("platform") != platform.platform()
        or provenance.get("numpy") != np.__version__
        or provenance.get("cuda_visible_devices")
        != isolation.get("BENCHMARK_GPU_UUID")
        or provenance.get("gpu_uuid") != isolation.get("BENCHMARK_GPU_UUID")
        or provenance.get("cuda_device_order") != "PCI_BUS_ID"
        or provenance.get("logical_cuda_device") != 0
        or any(not isinstance(provenance.get(field), str) or not provenance[field] for field in text_fields)
        or _finite_number(
            provenance.get("wallclock_seconds"), "RGB backend wallclock", positive=True
        )
        <= 0.0
    ):
        raise ValueError("RGB backend execution provenance is malformed.")
    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(ARMS):
        raise ValueError("RGB backend treatment-arm set changed.")
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    schema = _rgb_array_schema(frames=frames)
    expected_shapes = {name: list(shape) for name, (shape, _) in schema.items()}
    expected_dtypes = {name: str(dtype) for name, (_, dtype) in schema.items()}
    output_paths: dict[tuple[str, str, int], Path] = {}
    root = manifest_path.parent
    for arm in ARMS:
        arm_results = results[arm]
        if not isinstance(arm_results, dict) or set(arm_results) != set(CONDITIONS):
            raise ValueError(f"RGB backend {arm} condition set changed.")
        for condition in CONDITIONS:
            records = arm_results[condition]
            if not isinstance(records, list) or len(records) != episodes:
                raise ValueError(f"RGB backend {arm}/{condition} episode count changed.")
            for episode_index, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != set(RESULT_RECORD_KEYS):
                    raise ValueError("RGB backend episode record schema changed.")
                v1_record = v1_payload["results"][TASK][condition][episode_index]
                v1_path = v1_paths[(TASK, condition, episode_index)]
                rgb_record = worker["episodes"][TASK][condition][episode_index]
                rgb_path = rgb_paths[(TASK, condition, episode_index)]
                expected_relative = (
                    Path("predictions")
                    / arm
                    / TASK
                    / condition
                    / f"episode_{episode_index:03d}.npz"
                ).as_posix()
                if (
                    record.get("arm") != arm
                    or type(record.get("episode_index")) is not int
                    or record.get("episode_index") != episode_index
                    or record.get("frames") != frames
                    or record.get("max_candidates") != 2
                    or record.get("role_count") != len(ROLES)
                    or record.get("keypoint_count") != 3
                    or record.get("prediction_arrays") != expected_relative
                    or record.get("array_shapes") != expected_shapes
                    or record.get("array_dtypes") != expected_dtypes
                    or record.get("source_rgb_arrays_sha256") != file_sha256(rgb_path)
                    or record.get("source_rgb_trace_sha256")
                    != rgb_record.get("rgb_trace_sha256")
                    or record.get("source_v1_prediction_arrays_sha256")
                    != v1_record.get("prediction_arrays_sha256")
                    or record.get("source_v1_prediction_arrays_sha256")
                    != file_sha256(v1_path)
                    or record.get("source_v1_entity_mask_trace_sha256")
                    != v1_record.get("traces", {}).get("entity_mask_trace_sha256")
                    or record.get("source_v1_entity_status_trace_sha256")
                    != v1_record.get("traces", {}).get("entity_status_trace_sha256")
                    or record.get("source_v1_keypoint_trace_sha256")
                    != v1_record.get("traces", {}).get("keypoint_trace_sha256")
                    or record.get("source_v1_role_status_trace_sha256")
                    != v1_record.get("traces", {}).get("role_status_trace_sha256")
                ):
                    raise ValueError("RGB backend episode source pairing changed.")
                rgb = _load_npz(rgb_path, {"rgb"})["rgb"]
                if (
                    rgb.shape != (frames, resolution, resolution, 3)
                    or rgb.dtype != np.uint8
                    or _array_trace(rgb)
                    != record.get("decoded_rgb_array_trace_sha256")
                ):
                    raise ValueError("RGB backend decoded episode RGB changed.")
                traces = record.get("array_traces_sha256")
                if not isinstance(traces, dict) or set(traces) != set(OUTPUT_ARRAY_KEYS):
                    raise ValueError("RGB backend array trace schema changed.")
                prediction = resolve_member(
                    root, record["prediction_arrays"], "RGB backend prediction arrays"
                )
                if file_sha256(prediction) != require_sha256(
                    record.get("prediction_arrays_sha256"), "RGB prediction SHA"
                ):
                    raise ValueError("RGB backend prediction artifact changed.")
                arrays = _load_npz(prediction, set(OUTPUT_ARRAY_KEYS))
                _validate_decoded_rgb_arrays(
                    arrays,
                    frames=frames,
                    resolution=resolution,
                    expected_widths=np.asarray(
                        generator_metadata["support_link_half_widths_px"]
                    ),
                    expected_lengths=np.asarray(
                        generator_metadata["support_link_lengths_px"]
                    ),
                )
                for name in OUTPUT_ARRAY_KEYS:
                    if _array_trace(arrays[name]) != require_sha256(
                        traces.get(name), f"RGB {arm}/{condition}/{episode_index}/{name} trace"
                    ):
                        raise ValueError(f"RGB decoded array trace changed for {name}.")
                output_paths[(arm, condition, episode_index)] = prediction
    return payload, output_paths


def _score_real_backend(
    *,
    rgb_paths: dict[tuple[str, str, int], Path],
    v1_payload: dict[str, Any],
    v1_paths: dict[tuple[str, str, int], Path],
    dataset: dict[str, Any],
    dataset_root: Path,
    generator_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    resolution = int(dataset["resolution"])
    scored: dict[str, dict[str, dict[str, list[dict[str, np.ndarray]]]]] = {
        arm: {
            renderer: {condition: [] for condition in CONDITIONS}
            for renderer in RENDERERS
        }
        for arm in ARMS
    }
    runtime: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {condition: [] for condition in CONDITIONS} for arm in ARMS
    }
    end_runtime: dict[str, list[np.ndarray]] = {
        condition: [] for condition in CONDITIONS
    }
    trace_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {condition: [] for condition in CONDITIONS} for arm in ARMS
    }
    for condition in CONDITIONS:
        for episode_index in range(episodes):
            source_v1 = _load_npz(
                v1_paths[(TASK, condition, episode_index)], V1_ARRAY_KEYS
            )
            v1_record = v1_payload["results"][TASK][condition][episode_index]
            expected_traces = v1_record.get("traces", {})
            if (
                _array_trace(source_v1["role_masks"])
                != expected_traces.get("role_mask_trace_sha256")
                or _array_trace(source_v1["keypoints_xy"])
                != expected_traces.get("keypoint_trace_sha256")
                or _array_trace(source_v1["entity_masks"])
                != expected_traces.get("entity_mask_trace_sha256")
            ):
                raise ValueError("Decoded published-v1 role/keypoint/entity trace changed.")
            entity_nonempty = source_v1["entity_masks"].reshape(frames, 1, -1).any(
                axis=2
            )
            if np.any(
                source_v1["entity_valid"]
                & (source_v1["entity_lost"] | ~entity_nonempty)
            ):
                raise ValueError("Published-v1 entity availability is malformed.")
            gt_path = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "RGB candidate scoring arrays",
            )
            gt = _load_npz(
                gt_path, {"gt_indexed", "actions", "physics_states"}
            )["gt_indexed"]
            if (
                gt.shape != (frames, resolution, resolution)
                or gt.dtype != np.uint8
                or set(np.unique(gt).tolist()) - set(range(len(ROLES) + 1))
            ):
                raise ValueError("RGB candidate frozen GT schema changed.")
            episode_arrays: dict[str, dict[str, np.ndarray]] = {}
            for arm in ARMS:
                arrays = _load_npz(
                    rgb_paths[(arm, condition, episode_index)], set(OUTPUT_ARRAY_KEYS)
                )
                _validate_decoded_rgb_arrays(
                    arrays,
                    frames=frames,
                    resolution=resolution,
                    expected_widths=np.asarray(
                        generator_metadata["support_link_half_widths_px"]
                    ),
                    expected_lengths=np.asarray(
                        generator_metadata["support_link_lengths_px"]
                    ),
                )
                rank_zero_trace = _validate_rank_zero_exact(
                    arrays=arrays, source_v1=source_v1
                )
                if rank_zero_trace != expected_traces.get("role_mask_trace_sha256"):
                    raise ValueError("RGB rank-zero mask bytes changed from published v1.")
                episode_arrays[arm] = arrays
                runtime[arm][condition].append(arrays["parser_runtime_ms"])
                if arm == "real_rgb":
                    combined = source_v1["cutie_runtime_ms"] + arrays["parser_runtime_ms"]
                    if combined.shape != (frames,) or not np.isfinite(combined).all():
                        raise ValueError("RGB end-to-end runtime schema changed.")
                    end_runtime[condition].append(combined)
                row: dict[str, Any] = {
                    "episode_index": episode_index,
                    "rank_zero_exact_v1_role_mask_trace_sha256": rank_zero_trace,
                    "source_v1_entity_mask_trace_sha256": _array_trace(
                        source_v1["entity_masks"]
                    ),
                    "sealed_pose_trace_sha256": _array_trace(arrays["poses_xy"]),
                    "sealed_width_trace_sha256": _array_trace(
                        arrays["link_half_widths_px"]
                    ),
                }
                for renderer in RENDERERS:
                    current = _frame_score_rows(
                        arrays=arrays,
                        source_v1=source_v1,
                        gt_indexed=gt,
                        renderer=renderer,
                    )
                    scored[arm][renderer][condition].append(current)
                    row[f"{renderer}_quality_trace_sha256"] = _array_trace(
                        current["quality"]
                    )
                    row[f"{renderer}_role_iou_trace_sha256"] = _array_trace(
                        current["role_iou"]
                    )
                trace_rows[arm][condition].append(row)
            real = episode_arrays["real_rgb"]
            shuffled = episode_arrays["spatial_shuffle"]
            for name in (
                "poses_xy",
                "link_half_widths_px",
                "candidate_valid",
                "candidate_cost",
                "candidate_confidence",
                "mask_geometry_score",
                "candidate_source_code",
            ):
                # Only the reserved anchor must be identical across treatment arms.
                if not np.array_equal(real[name][:, 0], shuffled[name][:, 0]):
                    raise ValueError(f"RGB treatment arms changed slot-zero {name}.")
            if not np.array_equal(real["roi_xyxy"], shuffled["roi_xyxy"]):
                raise ValueError("RGB treatment arms changed current-entity ROI.")
    output: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    for arm in ARMS:
        output[arm] = {}
        traces[arm] = {}
        for renderer in RENDERERS:
            output[arm][renderer] = {}
            traces[arm][renderer] = {}
            for condition in CONDITIONS:
                metrics = _score_prefixes(scored[arm][renderer][condition])
                metrics["parser_runtime"] = _latency(
                    np.concatenate(runtime[arm][condition])
                )
                if arm == "real_rgb":
                    metrics["published_v1_cutie_plus_parser_runtime"] = {
                        **_latency(np.concatenate(end_runtime[condition])),
                        "semantics": (
                            "published_v1_cutie_runtime_plus_rgb_parser_runtime_"
                            "aligned_per_frame;disk_io_capsule_render_and_privileged_"
                            "gt_scoring_excluded"
                        ),
                    }
                metrics["rank_zero"] = {
                    "slot": 0,
                    "poses_exact_published_v1_keypoints": True,
                    "validity_exact_published_v1_projector_validity": True,
                    "scored_masks_exact_published_v1_role_mask_bytes": True,
                    "not_reconstructed_for_k1": True,
                }
                output[arm][renderer][condition] = metrics
                rows = trace_rows[arm][condition]
                traces[arm][renderer][condition] = {
                    "episodes": episodes,
                    "frames_per_episode": frames,
                    "episode_trace_bundle_sha256": sha256_json(
                        [
                            {
                                key: value
                                for key, value in row.items()
                                if key in {
                                    "episode_index",
                                    "rank_zero_exact_v1_role_mask_trace_sha256",
                                    "source_v1_entity_mask_trace_sha256",
                                    "sealed_pose_trace_sha256",
                                    "sealed_width_trace_sha256",
                                    f"{renderer}_quality_trace_sha256",
                                    f"{renderer}_role_iou_trace_sha256",
                                }
                            }
                            for row in rows
                        ]
                    ),
                }
    return output, traces


def _comparisons(
    *, metrics: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        real = metrics["real_rgb"]["rgb_capsule"][condition]["prefixes"]["2"]
        shuffled = metrics["spatial_shuffle"]["rgb_capsule"][condition]["prefixes"]["2"]
        base = baseline[condition]
        real_success = np.asarray(real["episode_success_at_0_5"], dtype=np.float64)
        real_quality = np.asarray(
            real["episode_best_min_role_iou_mean"], dtype=np.float64
        )
        shuffled_success = np.asarray(
            shuffled["episode_success_at_0_5"], dtype=np.float64
        )
        output[condition] = {
            "real_rgb_capsule_k2_minus_published_mask_topk_k2": {
                "success_at_0_5": _paired_bootstrap_difference(
                    real_success,
                    base["episode_success_at_0_5"],
                    label=f"{condition}/real_rgb_capsule_k2_minus_mask_topk_k2/success",
                ),
                "best_min_role_iou": _paired_bootstrap_difference(
                    real_quality,
                    base["episode_best_min_role_iou_mean"],
                    label=f"{condition}/real_rgb_capsule_k2_minus_mask_topk_k2/iou",
                ),
            },
            "real_rgb_capsule_k2_minus_spatial_shuffle_capsule_k2": {
                "success_at_0_5": _paired_bootstrap_difference(
                    real_success,
                    shuffled_success,
                    label=f"{condition}/real_rgb_minus_spatial_shuffle/success",
                )
            },
        }
    return output


def _development_gate(
    *,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for condition in CONDITIONS:
        cell = metrics["real_rgb"]["rgb_capsule"][condition]["prefixes"]["2"]
        burst = cell["failure_burst_at_0_5"]
        baseline_burst = int(
            baseline[condition]["failure_burst_at_0_5"]
            ["global_max_with_episode_resets"]
        )
        vs_baseline = comparisons[condition][
            "real_rgb_capsule_k2_minus_published_mask_topk_k2"
        ]
        vs_shuffle = comparisons[condition][
            "real_rgb_capsule_k2_minus_spatial_shuffle_capsule_k2"
        ]["success_at_0_5"]
        parser = metrics["real_rgb"]["rgb_capsule"][condition]["parser_runtime"]
        end = metrics["real_rgb"]["rgb_capsule"][condition][
            "published_v1_cutie_plus_parser_runtime"
        ]
        checks[f"real_rgb_capsule/{condition}/k2/availability"] = (
            float(cell["availability_rate"]) >= MIN_AVAILABILITY
        )
        checks[f"real_rgb_capsule/{condition}/k2/success_at_0_5"] = (
            float(cell["oracle_set_success_at_0_5"]) >= MIN_SUCCESS_AT_05
        )
        checks[f"real_rgb_capsule/{condition}/k2/global_burst_at_0_5"] = (
            int(burst["global_max_with_episode_resets"]) <= MAX_FAILURE_BURST
        )
        checks[f"real_rgb_capsule/{condition}/k2/burst_half_mask_topk"] = (
            int(burst["global_max_with_episode_resets"])
            <= math.floor(BASELINE_BURST_FRACTION * baseline_burst)
        )
        checks[f"real_rgb_capsule/{condition}/k2/episode_p95_burst_at_0_5"] = (
            float(burst["episode_p95"]) <= MAX_FAILURE_BURST
        )
        checks[f"real_rgb_capsule/{condition}/k2/success_lcb_vs_mask_topk"] = (
            float(vs_baseline["success_at_0_5"]["one_sided_lower_95"]) > 0.0
        )
        checks[f"real_rgb_capsule/{condition}/k2/iou_lcb_vs_mask_topk"] = (
            float(vs_baseline["best_min_role_iou"]["one_sided_lower_95"]) > 0.0
        )
        checks[f"real_rgb_capsule/{condition}/k2/success_gain_vs_shuffle"] = (
            float(vs_shuffle["estimate"]) >= MIN_SHUFFLE_SUCCESS_GAIN
        )
        checks[f"real_rgb_capsule/{condition}/k2/success_lcb_vs_shuffle"] = (
            float(vs_shuffle["one_sided_lower_95"]) > 0.0
        )
        checks[f"real_rgb/{condition}/parser_mean"] = (
            float(parser["mean_ms"]) <= MAX_PARSER_MEAN_MS
        )
        checks[f"real_rgb/{condition}/parser_p95"] = (
            float(parser["p95_ms"]) <= MAX_PARSER_P95_MS
        )
        checks[f"real_rgb/{condition}/cutie_plus_parser_mean"] = (
            float(end["mean_ms"]) <= MAX_END_TO_END_MEAN_MS
        )
        checks[f"real_rgb/{condition}/cutie_plus_parser_p95"] = (
            float(end["p95_ms"]) <= MAX_END_TO_END_P95_MS
        )
    passed = bool(checks) and all(checks.values())
    return {
        "checks": checks,
        "all_pass": passed,
        "development_candidate": passed,
        "main_treatment": "real_rgb/rgb_capsule/k2",
        "diagnostic_only": [
            "real_rgb/rgb_partition",
            "spatial_shuffle/rgb_partition",
            "spatial_shuffle/rgb_capsule",
        ],
    }


def _implementation_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        relative: repo_root / relative
        for relative in (
            "tdmpc2/tools/aggregate_object_graph_rgb_candidates.py",
            "tdmpc2/tools/replay_object_graph_rgb_candidates.py",
            "tdmpc2/perception/ordered_chain_rgb.py",
            "tdmpc2/perception/ordered_chain_topk.py",
            "tdmpc2/perception/support_conditioned_object_graph.py",
            "tdmpc2/common/object_graph_rgb_snapshot.py",
            "tdmpc2/common/unified_vos.py",
        )
    }


def _artifact_hash_snapshot(
    *,
    source_root: Path,
    dataset: dict[str, Any],
    dataset_root: Path,
    cutie_manifest_path: Path,
    v1_summary_path: Path,
    v1_manifest_path: Path,
    v1_paths: dict[tuple[str, str, int], Path],
    mask_topk_summary_path: Path,
    mask_topk_tree: dict[str, Any],
    rgb_manifest_path: Path,
    rgb_paths: dict[tuple[str, str, int], Path],
    graph_path: Path,
    dino_checkpoint: Path,
    isolation_path: Path,
    immutable_inputs_path: Path,
) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "source_summary": source_root / "unified_vos_summary.json",
        "dataset_manifest": dataset_root / "dataset_manifest.json",
        "source_cutie_manifest": cutie_manifest_path,
        "v1_summary": v1_summary_path,
        "v1_manifest": v1_manifest_path,
        "mask_topk_summary": mask_topk_summary_path,
        "rgb_manifest": rgb_manifest_path,
        "graph": graph_path,
        "dino_checkpoint": dino_checkpoint,
        "isolation": isolation_path,
        "immutable_inputs": immutable_inputs_path,
        **{
            f"implementation/{relative}": path
            for relative, path in _implementation_paths().items()
        },
    }
    episodes = int(dataset["counts"]["episodes"])
    for condition in CONDITIONS:
        for episode_index in range(episodes):
            paths[f"v1/{condition}/{episode_index}"] = v1_paths[
                (TASK, condition, episode_index)
            ]
            paths[f"gt/{condition}/{episode_index}"] = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "RGB immutable scoring arrays",
            )
            for arm in ARMS:
                paths[f"rgb/{arm}/{condition}/{episode_index}"] = rgb_paths[
                    (arm, condition, episode_index)
                ]
    hashes = {
        label: file_sha256(_regular_file(path, f"RGB bound artifact {label}"))
        for label, path in sorted(paths.items())
    }
    return {
        "file_sha256": hashes,
        "mask_topk_whole_tree_sha256": require_sha256(
            mask_topk_tree.get("tree_sha256"), "mask-only Top-K whole-tree SHA"
        ),
        "mask_topk_whole_tree_file_count": mask_topk_tree.get("file_count"),
    }


def _baseline_for_summary(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        condition: {
            "availability_rate": float(cell["availability_rate"]),
            "oracle_set_success_at_0_5": float(cell["oracle_set_success_at_0_5"]),
            "best_min_role_iou_mean": float(cell["best_min_role_iou_mean"]),
            "failure_burst_at_0_5": cell["failure_burst_at_0_5"],
            "episode_success_at_0_5": cell["episode_success_at_0_5"].tolist(),
            "episode_best_min_role_iou_mean": cell[
                "episode_best_min_role_iou_mean"
            ].tolist(),
        }
        for condition, cell in baseline.items()
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    (
        source_summary,
        dataset,
        dataset_path,
        cutie_manifest_path,
        _cutie_payload,
        _cutie_paths,
        v1_payload,
        v1_paths,
        v1_graphs,
        v1_summary,
        v1_manifest_path,
    ) = _strict_v1_pair(
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        v1_graph_path=args.v1_graph,
    )
    if tuple(ARMS) != ("real_rgb", "spatial_shuffle") or MAX_CANDIDATES != 2:
        raise RuntimeError("The preregistered RGB treatment arms/K changed.")
    episodes = int(dataset["counts"]["episodes"])
    mask_topk_payload, baseline, mask_binding = _validate_mask_topk_baseline(
        args.mask_topk_preflight_root,
        source_root=args.source_benchmark_root,
        v1_root=args.v1_preflight_root,
        episodes=episodes,
    )
    isolation = _validate_isolation_gate(
        args.isolation_gate,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        mask_topk_preflight_root=args.mask_topk_preflight_root,
        summary_root=args.output.parent,
        backend_manifest_path=args.rgb_backend_manifest,
    )
    immutable_inputs = _validate_immutable_inputs(
        args.immutable_inputs,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        mask_topk_preflight_root=args.mask_topk_preflight_root,
        dino_repo=args.dino_repo,
        dino_checkpoint=args.dino_checkpoint,
        summary_root=args.output.parent,
    )
    rgb_payload, rgb_paths = _validate_rgb_backend(
        args.rgb_backend_manifest,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        v1_payload=v1_payload,
        v1_paths=v1_paths,
        v1_manifest_path=v1_manifest_path,
        v1_graph=v1_graphs[TASK],
        dino_repo=args.dino_repo,
        dino_checkpoint=args.dino_checkpoint,
        isolation=isolation,
    )
    mask_summary_path = (
        args.mask_topk_preflight_root
        / "object_graph_topk_candidate_coverage_summary.json"
    )
    v1_summary_path = args.v1_preflight_root / "object_graph_tokenizer_summary.json"
    snapshot_before = _artifact_hash_snapshot(
        source_root=args.source_benchmark_root,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        cutie_manifest_path=cutie_manifest_path,
        v1_summary_path=v1_summary_path,
        v1_manifest_path=v1_manifest_path,
        v1_paths=v1_paths,
        mask_topk_summary_path=mask_summary_path,
        mask_topk_tree=mask_binding["tree"],
        rgb_manifest_path=args.rgb_backend_manifest,
        rgb_paths=rgb_paths,
        graph_path=args.v1_graph,
        dino_checkpoint=args.dino_checkpoint,
        isolation_path=args.isolation_gate,
        immutable_inputs_path=args.immutable_inputs,
    )
    metrics, traces = _score_real_backend(
        rgb_paths=rgb_paths,
        v1_payload=v1_payload,
        v1_paths=v1_paths,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        generator_metadata=rgb_payload["generator"],
    )
    comparisons = _comparisons(metrics=metrics, baseline=baseline)
    gate = _development_gate(
        metrics=metrics, baseline=baseline, comparisons=comparisons
    )
    immutable_inputs_after = _validate_immutable_inputs(
        args.immutable_inputs,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        mask_topk_preflight_root=args.mask_topk_preflight_root,
        dino_repo=args.dino_repo,
        dino_checkpoint=args.dino_checkpoint,
        summary_root=args.output.parent,
    )
    mask_topk_after, _baseline_after, mask_binding_after = (
        _validate_mask_topk_baseline(
            args.mask_topk_preflight_root,
            source_root=args.source_benchmark_root,
            v1_root=args.v1_preflight_root,
            episodes=episodes,
        )
    )
    if (
        immutable_inputs_after != immutable_inputs
        or mask_topk_after != mask_topk_payload
        or mask_binding_after != mask_binding
    ):
        raise RuntimeError("An immutable RGB/Mask-TopK input changed during scoring.")
    snapshot_after = _artifact_hash_snapshot(
        source_root=args.source_benchmark_root,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        cutie_manifest_path=cutie_manifest_path,
        v1_summary_path=v1_summary_path,
        v1_manifest_path=v1_manifest_path,
        v1_paths=v1_paths,
        mask_topk_summary_path=mask_summary_path,
        mask_topk_tree=mask_binding_after["tree"],
        rgb_manifest_path=args.rgb_backend_manifest,
        rgb_paths=rgb_paths,
        graph_path=args.v1_graph,
        dino_checkpoint=args.dino_checkpoint,
        isolation_path=args.isolation_gate,
        immutable_inputs_path=args.immutable_inputs,
    )
    if snapshot_after != snapshot_before:
        raise RuntimeError("A bound RGB input or implementation changed during scoring.")
    development_candidate = bool(gate["development_candidate"])
    summary = {
        "format": SUMMARY_FORMAT,
        "status": (
            "object_graph_rgb_capsule_k2_development_candidate_pending_new_seed_confirmation"
            if development_candidate
            else "object_graph_rgb_candidate_development_no_go"
        ),
        "engineering_pass": True,
        "development_candidate": development_candidate,
        "selected_arm": "real_rgb" if development_candidate else None,
        "selected_renderer": "rgb_capsule" if development_candidate else None,
        "selected_k": 2 if development_candidate else None,
        "controller_training_authorized": False,
        "scientific_go": False,
        "recommendation": (
            "freeze_new_unseen_trajectory_seed_and_rerun_rgb_capsule_confirmation_before_any_controller_pilot"
            if development_candidate
            else "do_not_train_controller_rgb_capsule_failed_coverage_causal_control_or_latency_gate"
        ),
        "scope": {
            "evidence_level": (
                "post_hoc_offline_oracle_set_coverage_on_previously_inspected_frozen_trajectories"
            ),
            "task": TASK,
            "single_task_development": True,
            "same_frozen_inputs_as_v1_and_mask_topk": True,
            "backend_episode_inputs_current_rgb_and_current_v1_entity_mask_only": True,
            "backend_fixed_support_rgb_input": True,
            "backend_fixed_labelled_support_masks_input": True,
            "backend_episode_ground_truth_access": False,
            "aggregator_ground_truth_access_after_backend_exit_and_restore": True,
            "slot_zero_scored_from_exact_published_v1_role_mask_bytes": True,
            "slot_one_partition_uses_current_cutie_entity_mask": True,
            "slot_one_capsule_is_not_clipped_by_current_cutie_entity_mask": True,
            "real_rgb_and_spatial_shuffle_scored_independently": True,
            "spatial_shuffle_is_negative_control_only": True,
            "offline_oracle_selects_best_slot_for_scoring_only": True,
            "candidate_weights_not_calibrated_for_controller_use": True,
            "controller_constructed": False,
            "controller_training_steps": 0,
            "allowed_claim": "rgb_capsule_candidate_coverage_development_signal",
            "disallowed_claims": [
                "online_candidate_selection_solved",
                "controller_advantage",
                "controller_training_authorization",
                "confirmatory_perception_result",
                "paper_level_generality",
            ],
        },
        "thresholds": {
            "minimum_real_rgb_capsule_k2_availability": MIN_AVAILABILITY,
            "minimum_real_rgb_capsule_k2_success_at_0_5": MIN_SUCCESS_AT_05,
            "maximum_global_failure_burst": MAX_FAILURE_BURST,
            "maximum_episode_p95_failure_burst": MAX_FAILURE_BURST,
            "maximum_burst_fraction_vs_published_mask_topk_k2": (
                BASELINE_BURST_FRACTION
            ),
            "paired_bootstrap_unit": "episode",
            "paired_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "paired_bootstrap_one_sided_lcb": 0.95,
            "minimum_success_and_iou_gain_lcb_vs_mask_topk_k2": 0.0,
            "minimum_success_gain_vs_spatial_shuffle": MIN_SHUFFLE_SUCCESS_GAIN,
            "minimum_success_gain_lcb_vs_spatial_shuffle": 0.0,
            "maximum_parser_mean_ms": MAX_PARSER_MEAN_MS,
            "maximum_parser_p95_ms": MAX_PARSER_P95_MS,
            "maximum_published_v1_cutie_plus_parser_mean_ms": (
                MAX_END_TO_END_MEAN_MS
            ),
            "maximum_published_v1_cutie_plus_parser_p95_ms": (
                MAX_END_TO_END_P95_MS
            ),
        },
        "gate": gate,
        "coverage": metrics,
        "coverage_traces": traces,
        "paired_episode_bootstrap_comparisons": comparisons,
        "published_mask_topk_k2_baseline": _baseline_for_summary(baseline),
        "published_mask_topk_binding": mask_binding["binding"],
        "generator": rgb_payload["generator"],
        "dino": rgb_payload["dino"],
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_benchmark_root": str(args.source_benchmark_root),
            "source_status": source_summary.get("status"),
            "source_summary_sha256": file_sha256(
                args.source_benchmark_root / "unified_vos_summary.json"
            ),
            "source_dataset_id": dataset["dataset_id"],
            "source_dataset_manifest_sha256": file_sha256(dataset_path),
            "source_cutie_manifest_sha256": file_sha256(cutie_manifest_path),
            "v1_preflight_root": str(args.v1_preflight_root),
            "v1_preflight_status": v1_summary.get("status"),
            "v1_preflight_summary_sha256": file_sha256(v1_summary_path),
            "v1_backend_manifest": str(v1_manifest_path),
            "v1_backend_manifest_sha256": file_sha256(v1_manifest_path),
            "mask_topk_preflight_root": str(args.mask_topk_preflight_root),
            "mask_topk_summary_sha256": file_sha256(mask_summary_path),
            "mask_topk_whole_tree_sha256": mask_binding["tree"]["tree_sha256"],
            "rgb_backend_manifest_relative_to_summary_root": (
                args.rgb_backend_manifest.relative_to(args.output.parent).as_posix()
            ),
            "rgb_backend_manifest_sha256": file_sha256(args.rgb_backend_manifest),
            "scoring_isolation_relative_to_summary_root": (
                args.isolation_gate.relative_to(args.output.parent).as_posix()
            ),
            "scoring_isolation_sha256": file_sha256(args.isolation_gate),
            "scoring_isolation_status": isolation["status"],
            "immutable_inputs_relative_to_summary_root": (
                args.immutable_inputs.relative_to(args.output.parent).as_posix()
            ),
            "immutable_inputs_sha256": file_sha256(args.immutable_inputs),
            "immutable_inputs_format": immutable_inputs["format"],
            "v1_graph": str(args.v1_graph),
            "v1_graph_file_sha256": file_sha256(args.v1_graph),
            "v1_graph_semantic_sha256": v1_graphs[TASK].graph_sha256,
            "dino_repo": str(args.dino_repo),
            "dino_checkpoint": str(args.dino_checkpoint),
            "dino_checkpoint_sha256": file_sha256(args.dino_checkpoint),
            "backend_gpu_uuid": isolation["BENCHMARK_GPU_UUID"],
            "bound_artifact_snapshot_sha256": sha256_json(snapshot_before),
        },
    }
    write_json(args.output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--v1-preflight-root", type=Path, required=True)
    parser.add_argument("--mask-topk-preflight-root", type=Path, required=True)
    parser.add_argument("--rgb-backend-manifest", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--isolation-gate", type=Path, required=True)
    parser.add_argument("--immutable-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_benchmark_root = _regular_dir(
        args.source_benchmark_root, "frozen unified source root"
    )
    args.v1_preflight_root = _regular_dir(
        args.v1_preflight_root, "published v1 preflight root"
    )
    args.mask_topk_preflight_root = _regular_dir(
        args.mask_topk_preflight_root, "published mask-only top-K root"
    )
    args.dino_repo = _regular_dir(args.dino_repo, "DINOv2 repository")
    for name, label in (
        ("rgb_backend_manifest", "RGB backend manifest"),
        ("v1_graph", "published-v1 Acrobot graph"),
        ("dino_checkpoint", "DINOv2 checkpoint"),
        ("isolation_gate", "RGB scoring isolation gate"),
        ("immutable_inputs", "RGB immutable inputs"),
    ):
        setattr(args, name, _regular_file(getattr(args, name), label))
    output_lexical = args.output.expanduser().absolute()
    for component in reversed(output_lexical.parents):
        if component.is_symlink():
            raise ValueError(f"Output contains a symlink ancestor: {component}")
    output_parent = _regular_dir(output_lexical.parent, "RGB summary parent")
    args.output = output_parent / output_lexical.name
    if args.output.name != "object_graph_rgb_candidate_coverage_summary.json":
        raise ValueError("RGB summary output name is not the fixed publication name.")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = aggregate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "engineering_pass": result["engineering_pass"],
                "development_candidate": result["development_candidate"],
                "controller_training_authorized": False,
                "scientific_go": False,
                "summary": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
