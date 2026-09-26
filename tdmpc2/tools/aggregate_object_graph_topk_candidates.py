"""Strict privileged scorer for the stateless ordered-chain top-K preflight.

The backend being scored is deliberately GT-free.  This program is only run
after the scoring directory has been restored.  It binds the frozen unified
VOS source, the published object-graph v1 entity masks, the sealed top-K pose
artifacts, and the scoring-isolation record before decoding ground truth.

Candidate masks are not accepted from the backend.  Alternative slots are
reconstructed deterministically from each sealed pose and its exact published
v1 entity mask with :func:`role_masks_from_pose`; slot zero is scored from the
exact published v1 role-mask bytes after its pose/keypoint parity is checked.
K=1/2/4 therefore means the literal prefixes ``[:1]``, ``[:2]``, and ``[:4]``
even when a slot is invalid.  Ground truth is used only to choose the best
member of a prefix for this offline oracle-set coverage diagnostic.

This is post-hoc development evidence.  It can never authorize controller
training or support a scientific claim.
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
from tdmpc2.perception.ordered_chain_topk import (
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES,
    OrderedChainTopKGenerator,
    role_masks_from_pose,
)
from tdmpc2.tools.aggregate_object_graph_temporal_replay import (
    _contained_file,
    _regular_dir,
    _regular_file,
    _strict_v1_pair,
)
from tdmpc2.tools.aggregate_object_graph_tokenizer_preflight import (
    ARRAY_KEYS as V1_ARRAY_KEYS,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
    _array_trace,
    _load_npz,
)
from tdmpc2.tools.replay_object_graph_topk_candidates import (
    BACKEND as BACKEND_NAME,
    COMPONENT_DIAGNOSTICS,
    EXPECTED_KEYPOINT_COUNT,
    EXPECTED_MAX_CANDIDATES,
    FAILURE_CODES,
    FORMAT as BACKEND_FORMAT,
    OUTPUT_ARRAY_KEYS,
    PROTOCOL as BACKEND_PROTOCOL,
)


TASK = "acrobot-swingup"
ROLES = TASK_ROLES[TASK]
K_VALUES = (1, 2, 4)
IOU_THRESHOLDS = (0.5, 0.75)
SUMMARY_FORMAT = "object_graph_topk_candidate_coverage_summary_v1"
ISOLATION_FORMAT = "object_graph_topk_candidate_scoring_isolation_v1"

# These are intentionally strict development gates, not paper thresholds.
MIN_GT_UNION_SUCCESS_AT_05 = 0.99
MIN_REAL_SUCCESS_AT_05 = 0.97
MIN_AVAILABILITY = 0.99
MAX_FAILURE_BURST = 10
MAX_PARSER_MEAN_MS = 5.0
MAX_PARSER_P95_MS = 10.0
MAX_END_TO_END_MEAN_MS = 12.0
MAX_END_TO_END_P95_MS = 15.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271828

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MODE = re.compile(r"[0-7]{3}")
_F32_MAX = np.finfo(np.float32).max


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}.")
    return result


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):03o}"


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be a whole-second UTC timestamp.")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("A reported metric series is empty or non-finite.")
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
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Latency is empty or non-finite.")
    if np.any(array <= 0.0):
        raise ValueError("Latency must be positive.")
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
    if predicted.shape != (len(ROLES), *gt_indexed.shape):
        raise ValueError("Candidate role-mask shape changed.")
    if predicted.dtype != np.bool_ or gt_indexed.dtype != np.uint8:
        raise ValueError("Candidate/GT mask dtype changed.")
    if np.any(predicted.sum(axis=0) > 1):
        raise ValueError("Candidate role masks overlap.")
    per_role = np.zeros(len(ROLES), dtype=np.float64)
    visible: list[int] = []
    for role_index in range(len(ROLES)):
        target = gt_indexed == role_index + 1
        if bool(target.any()):
            visible.append(role_index)
            per_role[role_index] = _iou(predicted[role_index], target)
    if not visible:
        return 0.0, per_role
    return float(min(per_role[index] for index in visible)), per_role


def _paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    """Episode-resampled paired descriptive interval, never a hypothesis test."""
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or not lhs.size:
        raise ValueError("Paired bootstrap inputs are malformed.")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Paired bootstrap inputs are non-finite.")
    # Label-derived seeding prevents accidental dependence on dictionary order.
    label_seed = int.from_bytes(label.encode("utf-8"), "little") % (2**32)
    rng = np.random.default_rng(BOOTSTRAP_SEED ^ label_seed)
    samples = rng.integers(0, lhs.size, size=(BOOTSTRAP_RESAMPLES, lhs.size))
    differences = (lhs[samples] - rhs[samples]).mean(axis=1)
    return {
        "unit": "episode",
        "paired": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED ^ label_seed,
        "estimand": "mean_episode_metric_difference",
        "estimate": float((lhs - rhs).mean()),
        "descriptive_percentile_interval_95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "one_sided_lower_95": float(np.percentile(differences, 5.0)),
        "not_a_confirmatory_hypothesis_test": True,
    }


def _score_prefixes(
    episodes: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("No episodes were supplied for top-K scoring.")
    output: dict[str, Any] = {}
    episode_metrics: dict[int, dict[str, np.ndarray]] = {}
    for k in K_VALUES:
        all_best: list[np.ndarray] = []
        all_available: list[np.ndarray] = []
        all_candidate_counts: list[np.ndarray] = []
        all_mode_counts: list[np.ndarray] = []
        all_alternative_available: list[np.ndarray] = []
        all_slot_zero_unavailable: list[np.ndarray] = []
        role_best_rows: list[np.ndarray] = []
        episode_best_mean = []
        episode_success: dict[float, list[float]] = {
            threshold: [] for threshold in IOU_THRESHOLDS
        }
        episode_bursts: dict[float, list[int]] = {
            threshold: [] for threshold in IOU_THRESHOLDS
        }
        source_histogram = {
            name: 0 for name, code in SOURCE_CODES.items() if code != 0
        }
        for episode in episodes:
            quality = episode["quality"][:, :k]
            role_iou = episode["role_iou"][:, :k]
            valid = episode["valid"][:, :k]
            source = episode["source_code"][:, :k]
            mode = episode["mode_code"][:, :k]
            if (
                quality.shape != valid.shape
                or source.shape != valid.shape
                or mode.shape != valid.shape
            ):
                raise ValueError("Top-K score matrix shapes changed.")
            available = valid.any(axis=1)
            best = np.where(valid, quality, -1.0).max(axis=1)
            best = np.where(available, best, 0.0)
            candidate_count = valid.sum(axis=1).astype(np.int16)
            mode_count = np.asarray(
                [len(set(int(v) for v in row[mask])) for row, mask in zip(mode, valid)],
                dtype=np.int16,
            )
            alternative_available = (
                valid[:, 1:].any(axis=1)
                if k > 1
                else np.zeros(len(valid), dtype=np.bool_)
            )
            slot_zero_unavailable = ~valid[:, 0]
            best_role = np.zeros((len(best), len(ROLES)), dtype=np.float64)
            for frame_index in range(len(best)):
                if not bool(available[frame_index]):
                    continue
                # This is role-wise oracle IoU for diagnosis only.  Set success
                # always uses one jointly selected candidate via min-role IoU.
                for role_index in range(len(ROLES)):
                    best_role[frame_index, role_index] = float(
                        role_iou[frame_index, :, role_index][valid[frame_index]].max()
                    )
            for name, code in SOURCE_CODES.items():
                if code:
                    source_histogram[name] += int(np.logical_and(valid, source == code).sum())
            all_best.append(best)
            all_available.append(available)
            all_candidate_counts.append(candidate_count)
            all_mode_counts.append(mode_count)
            all_alternative_available.append(alternative_available)
            all_slot_zero_unavailable.append(slot_zero_unavailable)
            role_best_rows.append(best_role)
            episode_best_mean.append(float(best.mean()))
            for threshold in IOU_THRESHOLDS:
                success = available & (best >= threshold)
                episode_success[threshold].append(float(success.mean()))
                episode_bursts[threshold].append(_maximum_false_burst(success))
        best = np.concatenate(all_best)
        available = np.concatenate(all_available)
        candidate_count = np.concatenate(all_candidate_counts)
        mode_count = np.concatenate(all_mode_counts)
        alternative_available = np.concatenate(all_alternative_available)
        slot_zero_unavailable = np.concatenate(all_slot_zero_unavailable)
        role_best = np.concatenate(role_best_rows)
        rescue_denominator = int(slot_zero_unavailable.sum())
        rescue_numerator = int(
            np.logical_and(slot_zero_unavailable, alternative_available).sum()
        )
        cell: dict[str, Any] = {
            "k": k,
            "prefix_slots": list(range(k)),
            "frames": int(len(best)),
            "episodes": len(episodes),
            "availability_rate": float(available.mean()),
            "best_min_role_iou": _percentile_summary(best),
            "best_per_role_iou_mean": {
                role: float(role_best[:, index].mean())
                for index, role in enumerate(ROLES)
            },
            "candidate_count": _percentile_summary(candidate_count.astype(np.float64)),
            "distinct_bend_mode_count": _percentile_summary(
                mode_count.astype(np.float64)
            ),
            "alternative_availability_rate_unconditional": float(
                alternative_available.mean()
            ),
            "rescue_available_at_k": {
                "conditioning": "slot_zero_unavailable_current_frame_only_no_gt",
                "denominator_frames": rescue_denominator,
                "numerator_frames": rescue_numerator,
                "rate": (
                    float(rescue_numerator / rescue_denominator)
                    if rescue_denominator
                    else None
                ),
                "status": "measured" if rescue_denominator else "not_applicable",
                "gt_free_ambiguity_threshold_used": False,
            },
            "source_histogram": source_histogram,
            "episode_best_min_role_iou_mean": [float(value) for value in episode_best_mean],
        }
        for threshold in IOU_THRESHOLDS:
            suffix = str(threshold).replace(".", "_")
            success = available & (best >= threshold)
            bursts = np.asarray(episode_bursts[threshold], dtype=np.int64)
            cell[f"oracle_set_success_at_{suffix}"] = float(success.mean())
            cell[f"failure_burst_at_{suffix}"] = {
                "global_max_with_episode_resets": int(bursts.max()),
                "episode_p95": float(np.percentile(bursts, 95)),
                "per_episode": [int(value) for value in bursts],
            }
            cell[f"episode_success_at_{suffix}"] = [
                float(value) for value in episode_success[threshold]
            ]
        output[str(k)] = cell
        episode_metrics[k] = {
            "best": np.asarray(episode_best_mean, dtype=np.float64),
            **{
                f"success_{threshold}": np.asarray(
                    episode_success[threshold], dtype=np.float64
                )
                for threshold in IOU_THRESHOLDS
            },
        }
    gains: dict[str, Any] = {}
    for current, previous in ((2, 1), (4, 2), (4, 1)):
        label = f"k{current}_minus_k{previous}"
        gains[label] = {
            "best_min_role_iou": _paired_bootstrap_difference(
                episode_metrics[current]["best"],
                episode_metrics[previous]["best"],
                label=f"{label}/best",
            ),
            **{
                f"oracle_set_success_at_{str(threshold).replace('.', '_')}": (
                    _paired_bootstrap_difference(
                        episode_metrics[current][f"success_{threshold}"],
                        episode_metrics[previous][f"success_{threshold}"],
                        label=f"{label}/success/{threshold}",
                    )
                )
                for threshold in IOU_THRESHOLDS
            },
        }
    return {"prefixes": output, "paired_episode_bootstrap_gains": gains}


def _frame_score_rows(
    *,
    poses: np.ndarray,
    valid: np.ndarray,
    source_code: np.ndarray,
    entity_masks: np.ndarray,
    gt_indexed: np.ndarray,
    slot_zero_masks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    frames = int(entity_masks.shape[0])
    if slot_zero_masks is not None and (
        slot_zero_masks.shape
        != (frames, len(ROLES), entity_masks.shape[-2], entity_masks.shape[-1])
        or slot_zero_masks.dtype != np.bool_
    ):
        raise ValueError("Exact slot-zero role-mask schema changed.")
    quality = np.zeros((frames, MAX_CANDIDATES), dtype=np.float64)
    role_iou = np.zeros(
        (frames, MAX_CANDIDATES, len(ROLES)), dtype=np.float64
    )
    mode_code = np.zeros((frames, MAX_CANDIDATES), dtype=np.int8)
    for frame_index in range(frames):
        entity = np.ascontiguousarray(entity_masks[frame_index], dtype=np.bool_)
        for slot in range(MAX_CANDIDATES):
            if not bool(valid[frame_index, slot]):
                continue
            predicted = (
                np.ascontiguousarray(slot_zero_masks[frame_index], dtype=np.bool_)
                if slot == 0 and slot_zero_masks is not None
                else role_masks_from_pose(entity, poses[frame_index, slot])
            )
            if not np.array_equal(predicted.any(axis=0), entity):
                raise ValueError("A valid top-K pose does not partition its source entity mask.")
            value, per_role = _candidate_quality(predicted, gt_indexed[frame_index])
            quality[frame_index, slot] = value
            role_iou[frame_index, slot] = per_role
            first, second = np.diff(poses[frame_index, slot].astype(np.float64), axis=0)
            scale = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
            sine = float((first[0] * second[1] - first[1] * second[0]) / scale)
            mode_code[frame_index, slot] = np.int8(
                0 if abs(sine) < 0.08 else (1 if sine > 0.0 else -1)
            )
    return {
        "quality": quality,
        "role_iou": role_iou,
        "valid": np.ascontiguousarray(valid, dtype=np.bool_),
        "source_code": np.ascontiguousarray(source_code, dtype=np.uint8),
        "mode_code": mode_code,
    }


def _validate_rank_zero_exact(
    *,
    poses: np.ndarray,
    valid: np.ndarray,
    source_code: np.ndarray,
    source_v1: dict[str, np.ndarray],
) -> str:
    expected_valid = source_v1["role_valid"].all(axis=1)
    if not np.array_equal(valid[:, 0], expected_valid):
        raise ValueError("Top-K slot zero validity differs from published v1.")
    expected_codes = np.where(expected_valid, SOURCE_CODES["v1_anchor"], 0).astype(
        np.uint8
    )
    if not np.array_equal(source_code[:, 0], expected_codes):
        raise ValueError("Top-K slot zero source is not the reserved v1 anchor.")
    expected_pose = np.zeros_like(poses[:, 0], dtype=np.float32)
    expected_pose[expected_valid, 0] = source_v1["keypoints_xy"][expected_valid, 0, 0]
    expected_pose[expected_valid, 1:] = source_v1["keypoints_xy"][expected_valid, :, 1]
    if not np.array_equal(poses[:, 0], expected_pose):
        raise ValueError("Top-K slot zero pose differs from published v1 keypoints.")
    # Slot zero is scored from these exact bytes.  Pose reconstruction is not
    # used for K=1 because a nearest-segment tie policy could otherwise turn a
    # parity check into a subtly different baseline.
    if np.any(source_v1["role_masks"].sum(axis=1) > 1):
        raise ValueError("Published v1 role masks overlap.")
    return _array_trace(source_v1["role_masks"])


def _validate_isolation_gate(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    summary_root: Path,
    backend_manifest_path: Path,
) -> dict[str, Any]:
    path = _regular_file(path, "top-K scoring isolation")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative_gate = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Top-K isolation gate is outside the summary root.") from exc
    if relative_gate != "provenance/scoring_isolation.json":
        raise ValueError("Top-K isolation gate is not at its fixed path.")
    payload = load_json(path)
    expected = {
        "format",
        "status",
        "source_benchmark_root",
        "v1_preflight_root",
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
        "cpu_only",
        "cuda_visible_devices",
        "topk_backend_manifest_relative_to_summary_root",
        "topk_backend_manifest_sha256",
        "same_uid_read_probe",
    }
    if set(payload) != expected or path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("Top-K scoring-isolation schema changed.")
    source_root = source_root.resolve(strict=True)
    v1_preflight_root = v1_preflight_root.resolve(strict=True)
    worker_root = (source_root / "worker_inputs").resolve(strict=True)
    scoring_root = (source_root / "dataset" / "scoring").resolve(strict=True)
    if (
        payload.get("format") != ISOLATION_FORMAT
        or payload.get("status") != "complete"
        or payload.get("source_benchmark_root") != str(source_root)
        or payload.get("v1_preflight_root") != str(v1_preflight_root)
        or payload.get("worker_input_root") != str(worker_root)
        or payload.get("scoring_root") != str(scoring_root)
        or payload.get("backend_completed_before_restore") is not True
        or payload.get("cpu_only") is not True
        or payload.get("cuda_visible_devices") != ""
        or worker_root == scoring_root
        or worker_root in scoring_root.parents
        or scoring_root in worker_root.parents
    ):
        raise ValueError("Top-K isolation identity or causality changed.")
    relative_probe = payload.get("scoring_probe_relative_to_source")
    if not isinstance(relative_probe, str) or not relative_probe or "\\" in relative_probe:
        raise ValueError("Top-K scoring-probe path is malformed.")
    probe = _contained_file(source_root, relative_probe, "top-K scoring probe")
    try:
        probe.relative_to(scoring_root)
    except ValueError as exc:
        raise ValueError("Top-K scoring probe is outside scoring-only data.") from exc
    for field in ("root_mode_before", "root_mode_locked", "root_mode_restored"):
        if not isinstance(payload.get(field), str) or _MODE.fullmatch(payload[field]) is None:
            raise ValueError(f"Top-K isolation mode {field} is malformed.")
    if (
        payload["root_mode_locked"] != "000"
        or payload["root_mode_before"] != payload["root_mode_restored"]
        or payload["root_mode_before"] == "000"
        or int(payload["root_mode_before"], 8) & 0o500 != 0o500
        or _mode(scoring_root) != payload["root_mode_restored"]
    ):
        raise ValueError("Top-K scoring permissions were not restored exactly.")
    before_sha = require_sha256(
        payload.get("probe_sha256_before"), "top-K pre-lock probe SHA"
    )
    restored_sha = require_sha256(
        payload.get("probe_sha256_restored"), "top-K restored probe SHA"
    )
    if before_sha != restored_sha or file_sha256(probe) != restored_sha:
        raise ValueError("Top-K scoring probe changed across the lock.")
    times = [
        _parse_utc(payload[field], field)
        for field in ("locked_utc", "backend_completed_utc", "restored_utc")
    ]
    if times != sorted(times):
        raise ValueError("Top-K isolation timestamps are out of order.")
    backend_manifest_path = backend_manifest_path.resolve(strict=True)
    try:
        expected_relative = backend_manifest_path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Top-K backend manifest is outside the summary root.") from exc
    if (
        expected_relative
        != "backends/object_graph_topk_candidates/backend_predictions.json"
        or payload.get("topk_backend_manifest_relative_to_summary_root")
        != expected_relative
        or payload.get("topk_backend_manifest_sha256")
        != file_sha256(backend_manifest_path)
    ):
        raise ValueError("Top-K isolation names a different backend manifest.")
    read_probe = payload.get("same_uid_read_probe")
    if not isinstance(read_probe, dict) or set(read_probe) != {
        "exit_code",
        "error_type",
        "log_relative_to_summary_root",
        "log_sha256",
    }:
        raise ValueError("Top-K same-user read-probe schema changed.")
    if (
        type(read_probe.get("exit_code")) is not int
        or read_probe.get("exit_code") != 1
        or read_probe.get("error_type") != "PermissionError"
        or read_probe.get("log_relative_to_summary_root")
        != "contracts/scoring_read_probe.log"
    ):
        raise ValueError("Top-K scoring read probe did not fail closed.")
    log_path = _contained_file(
        summary_root,
        read_probe["log_relative_to_summary_root"],
        "top-K scoring read-probe log",
    )
    if (
        file_sha256(log_path)
        != require_sha256(read_probe.get("log_sha256"), "top-K read-probe log SHA")
        or b"PermissionError" not in log_path.read_bytes()
    ):
        raise ValueError("Top-K read-probe log is not the bound permission failure.")
    return payload


def _validate_immutable_inputs(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    summary_root: Path,
) -> dict[str, Any]:
    # Lazy import avoids a harmless constants-only import cycle: the snapshot
    # records this aggregator's published format strings in its own contract.
    from tdmpc2.common.object_graph_topk_snapshot import (
        FORMAT as IMMUTABLE_INPUTS_FORMAT,
        build as build_immutable_inputs,
    )

    path = _regular_file(path, "top-K immutable-input snapshot")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Top-K immutable inputs are outside the summary root.") from exc
    if relative != "provenance/immutable_inputs.json":
        raise ValueError("Top-K immutable inputs are not at their fixed path.")
    payload = load_json(path)
    rebuilt = build_immutable_inputs(
        source_root=source_root,
        v1_root=v1_preflight_root,
    )
    if (
        payload.get("format") != IMMUTABLE_INPUTS_FORMAT
        or payload != rebuilt
        or path.read_bytes() != canonical_json_bytes(payload)
        or payload.get("dataset_id") != payload.get("v1_backend_dataset_id")
        or payload.get("scope", {}).get("cpu_only") is not True
        or payload.get("scope", {}).get("episode_ground_truth_available_to_backend")
        is not False
        or payload.get("scope", {}).get("backend_temporal_state") is not False
        or payload.get("scope", {}).get("controller_training_steps") != 0
        or payload.get("scope", {}).get("controller_training_authorized") is not False
    ):
        raise ValueError("Top-K immutable-input snapshot changed.")
    return payload


def _generator_from_frozen_support(
    *,
    graph: Any,
    dataset: dict[str, Any],
    dataset_root: Path,
    source_root: Path,
) -> tuple[OrderedChainTopKGenerator, Path]:
    # Reproduce the backend from the exact GT-free worker view it consumed.
    # The dataset tree contains a byte-identical support file at a deliberately
    # different path; using that path here would make strict provenance reject
    # every correctly isolated run even though the calibration bytes match.
    worker_input_path = _regular_file(
        source_root / "worker_inputs" / "backend_inputs.json",
        "GT-free worker input manifest for support calibration",
    )
    worker, worker_support_paths, _ = validate_backend_inputs(
        worker_input_path, strict_counts=True
    )
    if worker.get("dataset_id") != dataset.get("dataset_id"):
        raise ValueError("GT-free worker support belongs to a different dataset.")
    support_path = worker_support_paths[TASK]
    dataset_support_path = resolve_member(
        dataset_root,
        dataset["support"][TASK]["arrays"],
        "top-K dataset support calibration arrays",
    )
    worker_support_record = worker["support"][TASK]
    dataset_support_record = dataset["support"][TASK]
    support_sha = file_sha256(support_path)
    if (
        support_path == dataset_support_path
        or support_sha != file_sha256(dataset_support_path)
        or support_sha != worker_support_record.get("arrays_sha256")
        or support_sha != dataset_support_record.get("arrays_sha256")
    ):
        raise ValueError(
            "GT-free worker and dataset support must be path-disjoint and byte-identical."
        )
    support = _load_npz(support_path, {"rgb", "indexed_masks"})
    generator = OrderedChainTopKGenerator(
        graph, support["indexed_masks"], max_candidates=MAX_CANDIDATES
    )
    metadata = generator.metadata()
    if (
        metadata.get("format") != GENERATOR_FORMAT
        or metadata.get("protocol") != GENERATOR_PROTOCOL
        or metadata.get("max_candidates") != MAX_CANDIDATES
        or metadata.get("source_codes") != dict(SOURCE_CODES)
        or metadata.get("episode_state") is not False
        or metadata.get("future_frames") is not False
        or metadata.get("ground_truth_input") is not False
        or metadata.get("task_name_dispatch") is not False
    ):
        raise ValueError("Local top-K generator metadata is not the frozen stateless API.")
    return generator, support_path


def _expected_output_schema() -> dict[str, Any]:
    return {
        "role_masks_materialized": False,
        "alternative_role_mask_reconstruction": (
            "nearest_ordered_pose_segment_partition_of_sealed_current_"
            "source_entity_mask_v1"
        ),
        "slot_zero_scoring_role_masks": (
            "exact_published_v1_role_masks_after_privileged_scoring_unlock_v1"
        ),
        "slot_zero_pose_use": (
            "strict_parity_check_against_published_v1_keypoints_v1"
        ),
        "slot_zero": "reserved_exact_published_v1_anchor",
        "alternative_slots": [1, 2, 3],
        "candidate_valid_is_prefix": False,
        "candidate_count_semantics": "sum_candidate_valid",
        "array_names": sorted(OUTPUT_ARRAY_KEYS),
        "failure_codes": FAILURE_CODES,
        "source_codes": dict(SOURCE_CODES),
    }


def _expected_backend_protocol() -> dict[str, Any]:
    return {
        "format": BACKEND_PROTOCOL,
        "diagnostic_scope": "offline_candidate_coverage_preflight_v1",
        "source_backend": "object_graph_cutie",
        "source_entity_masks_consumed": True,
        "source_entity_lost_consumed": True,
        "source_entity_valid_consumed": True,
        "source_entity_confidence_consumed": False,
        "source_entity_mask_score_consumed": False,
        "source_role_masks_consumed": False,
        "frozen_support_indexed_masks_consumed": True,
        "support_rgb_schema_validated_not_used": True,
        "episode_rgb_bytes_hashed_for_input_validation": True,
        "episode_rgb_parser_input": False,
        "episode_rgb_decoded": False,
        "episode_ground_truth_read": False,
        "simulator_state_read": False,
        "actions_read": False,
        "rewards_read": False,
        "previous_frames_parser_input": False,
        "future_frames_parser_input": False,
        "episode_entity_arrays_loaded_before_replay": True,
        "future_entity_frames_generator_input": False,
        "previous_entity_frames_generator_input": False,
        "episode_state": False,
        "max_k_generated_once": MAX_CANDIDATES,
        "k1_k2_k4_are_fixed_prefix_views": True,
        "slot_zero_reserved_for_exact_v1": True,
        "descriptors_emitted": False,
        "role_masks_materialized": False,
        "privileged_scorer_slot_zero_uses_published_v1_role_masks": True,
        "privileged_scorer_alternatives_reconstruct_role_masks": True,
        "controller_token_not_evaluated": True,
        "controller_training_eligible": False,
        "cpu_only": True,
        "runtime_ms_semantics": (
            "ordered_chain_topk_generator_internal_per_current_entity_mask_v1"
        ),
    }


def _array_schema(
    *, frames: int
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    candidate = (frames, MAX_CANDIDATES)
    result: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "poses_xy": ((frames, MAX_CANDIDATES, 3, 2), np.dtype(np.float32)),
        "candidate_valid": (candidate, np.dtype(np.bool_)),
        "candidate_cost": (candidate, np.dtype(np.float32)),
        "candidate_weight": (candidate, np.dtype(np.float32)),
        "candidate_confidence": (candidate, np.dtype(np.float32)),
        "candidate_source_code": (candidate, np.dtype(np.uint8)),
        "candidate_count": ((frames,), np.dtype(np.uint8)),
        "parser_runtime_ms": ((frames,), np.dtype(np.float64)),
        "best_second_cost_margin": ((frames,), np.dtype(np.float32)),
        "weight_entropy": ((frames,), np.dtype(np.float32)),
        "normalized_weight_entropy": ((frames,), np.dtype(np.float32)),
        "weighted_pose_dispersion_px": ((frames,), np.dtype(np.float32)),
        "fit_uncertainty": ((frames,), np.dtype(np.float32)),
        "failure_code": ((frames,), np.dtype(np.uint8)),
    }
    for name in COMPONENT_DIAGNOSTICS.values():
        result[name] = (candidate, np.dtype(np.float32))
    if set(result) != set(OUTPUT_ARRAY_KEYS):
        raise RuntimeError("Aggregator/backend output array constants diverged.")
    return result


def _validate_decoded_topk_arrays(
    arrays: dict[str, np.ndarray], *, frames: int
) -> None:
    if set(arrays) != set(OUTPUT_ARRAY_KEYS):
        raise ValueError("Decoded top-K array schema changed.")
    schema = _array_schema(frames=frames)
    for name, (shape, dtype) in schema.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Decoded top-K {name} is {value.shape}/{value.dtype}, "
                f"expected {shape}/{dtype}."
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Decoded top-K {name} is non-finite.")
    valid = arrays["candidate_valid"]
    counts = valid.sum(axis=1).astype(np.uint8)
    if not np.array_equal(counts, arrays["candidate_count"]):
        raise ValueError("candidate_count is not exactly sum(candidate_valid).")
    if np.any(valid[:, 2] & ~valid[:, 1]) or np.any(valid[:, 3] & ~valid[:, 2]):
        raise ValueError("Alternative top-K slots are not left-packed in slots 1..3.")
    source = arrays["candidate_source_code"]
    anchor_code = int(SOURCE_CODES["v1_anchor"])
    if np.any(valid[:, 0] & (source[:, 0] != anchor_code)):
        raise ValueError("A valid slot zero is not marked as the v1 anchor.")
    if np.any(valid[:, 1:] & ((source[:, 1:] == 0) | (source[:, 1:] == anchor_code))):
        raise ValueError("An alternative slot has a padding/anchor source code.")
    if np.any(source[~valid] != 0):
        raise ValueError("Invalid slots have non-padding source codes.")
    if not np.isin(source, np.asarray(sorted(set(SOURCE_CODES.values())))).all():
        raise ValueError("Unknown top-K source code.")
    poses = arrays["poses_xy"]
    costs = arrays["candidate_cost"]
    weights = arrays["candidate_weight"]
    confidence = arrays["candidate_confidence"]
    if np.any(poses[~valid] != 0.0):
        raise ValueError("Invalid top-K poses are not zero padded.")
    if np.any(costs[~valid] != _F32_MAX) or np.any(costs[valid] < 0.0):
        raise ValueError("Top-K candidate cost/padding contract changed.")
    if np.any(weights[~valid] != 0.0) or np.any(confidence[~valid] != 0.0):
        raise ValueError("Invalid top-K weight/confidence is not zero padded.")
    if np.any((weights < 0.0) | (weights > 1.0)) or np.any(
        (confidence < 0.0) | (confidence > 1.0)
    ):
        raise ValueError("Top-K weight/confidence escaped [0,1].")
    sums = weights.sum(axis=1)
    if not np.allclose(sums[counts > 0], 1.0, atol=2e-6, rtol=0.0) or np.any(
        sums[counts == 0] != 0.0
    ):
        raise ValueError("Top-K weights do not sum to one over valid slots.")
    if np.any(arrays["parser_runtime_ms"] <= 0.0):
        raise ValueError("Top-K parser runtime must be positive.")
    if np.any(arrays["best_second_cost_margin"] < 0.0):
        raise ValueError("Top-K cost margin cannot be negative.")
    if np.any(arrays["weight_entropy"] < 0.0) or np.any(
        (arrays["normalized_weight_entropy"] < 0.0)
        | (arrays["normalized_weight_entropy"] > 1.0 + 2e-6)
    ):
        raise ValueError("Top-K entropy contract changed.")
    if np.any(arrays["weighted_pose_dispersion_px"] < 0.0) or np.any(
        (arrays["fit_uncertainty"] < 0.0) | (arrays["fit_uncertainty"] > 1.0)
    ):
        raise ValueError("Top-K uncertainty contract changed.")
    allowed_failure_codes = np.asarray(sorted(set(FAILURE_CODES.values())), dtype=np.uint8)
    if not np.isin(arrays["failure_code"], allowed_failure_codes).all():
        raise ValueError("Unknown top-K failure code.")
    if np.any((counts > 0) != (arrays["failure_code"] == FAILURE_CODES["none"])):
        raise ValueError("Top-K failure code disagrees with candidate availability.")
    zero_padded = {
        "candidate_coverage",
        "candidate_render_iou",
        "candidate_branch_fraction",
        "candidate_disconnected_fraction",
    }
    for name in COMPONENT_DIAGNOSTICS.values():
        value = arrays[name]
        if name in zero_padded:
            if np.any((value < 0.0) | (value > 1.0)) or np.any(value[~valid] != 0.0):
                raise ValueError(f"Top-K {name} range/padding contract changed.")
        elif np.any(value[valid] < 0.0) or np.any(value[~valid] != _F32_MAX):
            raise ValueError(f"Top-K {name} cost/padding contract changed.")
    for frame_index, active_mask in enumerate(valid):
        active = np.flatnonzero(active_mask)
        number = len(active)
        if not number:
            if (
                arrays["best_second_cost_margin"][frame_index] != 0.0
                or arrays["weight_entropy"][frame_index] != 0.0
                or arrays["normalized_weight_entropy"][frame_index] != 0.0
                or arrays["weighted_pose_dispersion_px"][frame_index] != 0.0
                or arrays["fit_uncertainty"][frame_index] != 1.0
            ):
                raise ValueError("Empty top-K frame uncertainty convention changed.")
            continue
        probabilities = weights[frame_index, active].astype(np.float64)
        expected_uncertainty = 1.0 - float(confidence[frame_index, active].max())
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        normalized = entropy / math.log(number) if number > 1 else 0.0
        margin = 0.0
        if number > 1:
            ranked = np.sort(costs[frame_index, active].astype(np.float64))
            margin = float(ranked[1] - ranked[0])
        mean_pose = np.sum(
            probabilities[:, None, None] * poses[frame_index, active].astype(np.float64),
            axis=0,
        )
        dispersion = float(
            np.sum(
                probabilities
                * np.linalg.norm(
                    poses[frame_index, active].astype(np.float64) - mean_pose[None],
                    axis=2,
                ).mean(axis=1)
            )
        )
        for actual, expected, label, atol, rtol in (
            (arrays["fit_uncertainty"][frame_index], expected_uncertainty, "uncertainty", 2e-6, 0.0),
            (arrays["weight_entropy"][frame_index], entropy, "entropy", 3e-6, 0.0),
            (arrays["normalized_weight_entropy"][frame_index], normalized, "normalized entropy", 3e-6, 0.0),
            (arrays["best_second_cost_margin"][frame_index], margin, "cost margin", 3e-5, 2e-6),
            (arrays["weighted_pose_dispersion_px"][frame_index], dispersion, "pose dispersion", 3e-5, 2e-6),
        ):
            if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
                raise ValueError(f"Top-K {label} is inconsistent with sealed arrays.")


def _implementation_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        relative: repo_root / relative
        for relative in (
            "tdmpc2/tools/replay_object_graph_topk_candidates.py",
            "tdmpc2/perception/ordered_chain_topk.py",
            "tdmpc2/perception/support_conditioned_object_graph.py",
            "tdmpc2/tools/replay_object_graph_temporal_v2.py",
            "tdmpc2/common/unified_vos.py",
        )
    }


def _validate_topk_backend(
    manifest_path: Path,
    *,
    dataset: dict[str, Any],
    dataset_root: Path,
    v1_payload: dict[str, Any],
    v1_paths: dict[tuple[str, str, int], Path],
    v1_manifest_path: Path,
    v1_graph: Any,
    generator: OrderedChainTopKGenerator,
    support_path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], Path]]:
    manifest_path = _regular_file(manifest_path, "top-K replay manifest")
    payload = load_json(manifest_path)
    expected_top = {
        "format",
        "status",
        "backend",
        "task",
        "dataset_id",
        "input_manifest_sha256",
        "v1_backend_manifest_sha256",
        "roles",
        "max_candidates",
        "graph",
        "generator",
        "output_schema",
        "protocol",
        "backend_provenance",
        "results",
    }
    worker_input = _regular_file(
        dataset_root.parent / "worker_inputs" / "backend_inputs.json",
        "GT-free worker input manifest",
    )
    worker, support_paths, _ = validate_backend_inputs(worker_input, strict_counts=True)
    if (
        set(payload) != expected_top
        or manifest_path.read_bytes() != canonical_json_bytes(payload)
        or (
        payload.get("format") != BACKEND_FORMAT
        or BACKEND_PROTOCOL != "gt_free_stateless_current_entity_mask_topk_replay_v1"
        or payload.get("status") != "complete"
        or payload.get("backend") != BACKEND_NAME
        or payload.get("task") != TASK
        or payload.get("dataset_id") != dataset.get("dataset_id")
        or payload.get("dataset_id") != worker.get("dataset_id")
        or payload.get("input_manifest_sha256") != file_sha256(worker_input)
        or payload.get("input_manifest_sha256")
        != dataset.get("backend_inputs", {}).get("sha256")
        or payload.get("v1_backend_manifest_sha256") != file_sha256(v1_manifest_path)
        or payload.get("roles") != list(ROLES)
        or payload.get("max_candidates") != MAX_CANDIDATES
        or EXPECTED_MAX_CANDIDATES != MAX_CANDIDATES
        or EXPECTED_KEYPOINT_COUNT != 3
        )
    ):
        raise ValueError("Top-K backend identity/pairing schema changed.")
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
        raise ValueError("Top-K backend graph provenance changed.")
    if payload.get("generator") != generator.metadata():
        raise ValueError("Top-K backend generator metadata differs from local implementation.")
    if payload.get("output_schema") != _expected_output_schema():
        raise ValueError("Top-K backend output-schema disclosure changed.")
    if payload.get("protocol") != _expected_backend_protocol():
        raise ValueError("Top-K backend GT-free protocol changed.")

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
        "device",
        "cuda_visible_devices",
        "wallclock_seconds",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("Top-K backend provenance schema changed.")
    implementation = provenance.get("implementation")
    implementation_paths = _implementation_paths()
    if not isinstance(implementation, dict) or set(implementation) != set(
        implementation_paths
    ):
        raise ValueError("Top-K implementation snapshot is incomplete.")
    for relative, path in implementation_paths.items():
        path = _regular_file(path, f"top-K implementation {relative}")
        record = implementation[relative]
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"} or (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != file_sha256(path)
        ):
            raise ValueError(f"Top-K implementation changed: {relative}.")
    if (
        provenance.get("treatment")
        != "stateless_current_mask_ordered_chain_topk_replay"
        or provenance.get("source_v1_backend_manifest") != str(v1_manifest_path)
        or provenance.get("source_v1_backend_manifest_sha256")
        != file_sha256(v1_manifest_path)
        or provenance.get("source_worker_inputs") != str(worker_input)
        or provenance.get("source_worker_inputs_sha256") != file_sha256(worker_input)
        or provenance.get("source_support_arrays") != str(support_path)
        or provenance.get("source_support_arrays_sha256") != file_sha256(support_path)
        or support_paths[TASK] != support_path
        or provenance.get("v1_graph_file_sha256") != file_sha256(v1_graph.source_path)
        or provenance.get("v1_graph_semantic_sha256") != v1_graph.graph_sha256
        or provenance.get("python") != sys.version
        or provenance.get("platform") != platform.platform()
        or provenance.get("numpy") != np.__version__
        or provenance.get("device") != "cpu"
        or provenance.get("cuda_visible_devices") != ""
        or _finite_number(provenance.get("wallclock_seconds"), "wallclock", positive=True)
        <= 0.0
    ):
        raise ValueError("Top-K execution provenance is malformed.")

    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(CONDITIONS):
        raise ValueError("Top-K condition set changed.")
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    schema = _array_schema(frames=frames)
    expected_shapes = {name: list(shape) for name, (shape, _) in schema.items()}
    expected_dtypes = {name: str(dtype) for name, (_, dtype) in schema.items()}
    paths: dict[tuple[str, int], Path] = {}
    root = manifest_path.parent
    for condition in CONDITIONS:
        records = results[condition]
        if not isinstance(records, list) or len(records) != episodes:
            raise ValueError(f"Top-K episode count changed for {condition}.")
        for episode_index, record in enumerate(records):
            expected_record = {
                "episode_index",
                "frames",
                "max_candidates",
                "role_count",
                "keypoint_count",
                "prediction_arrays",
                "prediction_arrays_sha256",
                "array_shapes",
                "array_dtypes",
                "array_traces_sha256",
                "source_v1_prediction_arrays_sha256",
                "source_v1_entity_mask_trace_sha256",
                "source_v1_entity_status_trace_sha256",
            }
            v1_record = v1_payload["results"][TASK][condition][episode_index]
            v1_path = v1_paths[(TASK, condition, episode_index)]
            if not isinstance(record, dict) or set(record) != expected_record or (
                type(record.get("episode_index")) is not int
                or record.get("episode_index") != episode_index
                or record.get("frames") != frames
                or record.get("max_candidates") != MAX_CANDIDATES
                or record.get("role_count") != len(ROLES)
                or record.get("keypoint_count") != 3
                or record.get("array_shapes") != expected_shapes
                or record.get("array_dtypes") != expected_dtypes
                or record.get("source_v1_prediction_arrays_sha256")
                != v1_record.get("prediction_arrays_sha256")
                or record.get("source_v1_prediction_arrays_sha256")
                != file_sha256(v1_path)
                or record.get("source_v1_entity_mask_trace_sha256")
                != v1_record.get("traces", {}).get("entity_mask_trace_sha256")
                or record.get("source_v1_entity_status_trace_sha256")
                != v1_record.get("traces", {}).get("entity_status_trace_sha256")
            ):
                raise ValueError("Top-K episode/v1 pairing schema changed.")
            traces = record.get("array_traces_sha256")
            if not isinstance(traces, dict) or set(traces) != set(OUTPUT_ARRAY_KEYS):
                raise ValueError("Top-K episode trace schema changed.")
            for name, digest in traces.items():
                require_sha256(digest, f"top-K {condition}/{episode_index}/{name}")
            path = resolve_member(root, record.get("prediction_arrays"), "top-K arrays")
            if file_sha256(path) != require_sha256(
                record.get("prediction_arrays_sha256"), "top-K prediction SHA"
            ):
                raise ValueError("Top-K prediction artifact changed.")
            arrays = _load_npz(path, set(OUTPUT_ARRAY_KEYS))
            _validate_decoded_topk_arrays(arrays, frames=frames)
            for name in OUTPUT_ARRAY_KEYS:
                if _array_trace(arrays[name]) != traces[name]:
                    raise ValueError(f"Top-K decoded trace changed for {name}.")
            paths[(condition, episode_index)] = path
    return payload, paths



def _score_gt_union_once(
    *,
    generator: OrderedChainTopKGenerator,
    dataset: dict[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify clean/hard GT identity, then evaluate one physical trajectory once."""
    resolution = int(dataset["resolution"])
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    scored: list[dict[str, np.ndarray]] = []
    runtime_rows: list[np.ndarray] = []
    trace_rows = []
    for episode_index in range(episodes):
        decoded: dict[str, dict[str, np.ndarray]] = {}
        gt_paths: dict[str, Path] = {}
        for condition in CONDITIONS:
            gt_path = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "top-K GT-union scoring arrays",
            )
            gt_paths[condition] = gt_path
            scoring = _load_npz(
                gt_path, {"gt_indexed", "actions", "physics_states"}
            )
            gt = scoring["gt_indexed"]
            if (
                gt.shape != (frames, resolution, resolution)
                or gt.dtype != np.uint8
                or set(np.unique(gt).tolist()) - set(range(len(ROLES) + 1))
            ):
                raise ValueError("Top-K GT-union frozen mask schema changed.")
            for name, expected_rows in (
                ("actions", int(dataset["counts"]["actions_per_episode"])),
                ("physics_states", frames),
            ):
                value = scoring[name]
                if (
                    value.ndim != 2
                    or value.shape[0] != expected_rows
                    or not np.issubdtype(value.dtype, np.floating)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Top-K GT-union frozen {name} schema changed.")
            decoded[condition] = scoring
        for name in ("gt_indexed", "actions", "physics_states"):
            if not np.array_equal(decoded["clean"][name], decoded["hard"][name]):
                raise ValueError(
                    f"Acrobot clean/hard frozen {name} is not byte-identical."
                )
        gt = decoded["clean"]["gt_indexed"]
        poses = np.zeros((frames, MAX_CANDIDATES, 3, 2), dtype=np.float32)
        valid = np.zeros((frames, MAX_CANDIDATES), dtype=np.bool_)
        costs = np.full((frames, MAX_CANDIDATES), _F32_MAX, dtype=np.float32)
        weights = np.zeros((frames, MAX_CANDIDATES), dtype=np.float32)
        confidence = np.zeros((frames, MAX_CANDIDATES), dtype=np.float32)
        source_code = np.zeros((frames, MAX_CANDIDATES), dtype=np.uint8)
        slot_zero_masks = np.zeros(
            (frames, len(ROLES), resolution, resolution), dtype=np.bool_
        )
        runtime = np.zeros(frames, dtype=np.float64)
        entity_masks = gt != 0
        for frame_index in range(frames):
            frame = generator.project(
                entity_mask=entity_masks[frame_index],
                entity_available=bool(entity_masks[frame_index].any()),
            )
            poses[frame_index] = frame.poses_xy
            valid[frame_index] = frame.valid
            costs[frame_index] = frame.costs
            weights[frame_index] = frame.weights
            confidence[frame_index] = frame.confidence
            source_code[frame_index] = frame.source_codes
            slot_zero_masks[frame_index] = frame.role_masks[0]
            runtime[frame_index] = frame.runtime_ms
            if frame.candidate_count != int(frame.valid.sum()):
                raise ValueError("GT-union generator candidate_count is not sum(valid).")
            # Slot zero keeps exact v1 parser masks by contract; its sealed
            # pose is parity metadata, not the mask reconstruction semantic.
            # Only alternatives must round-trip from the published float32
            # pose byte-exactly.
            for slot in np.flatnonzero(frame.valid[1:]) + 1:
                reconstructed = role_masks_from_pose(
                    entity_masks[frame_index], frame.poses_xy[slot]
                )
                if not np.array_equal(reconstructed, frame.role_masks[slot]):
                    raise ValueError(
                        "GT-union sealed alternative pose reconstruction changed masks."
                    )
        episode_scores = _frame_score_rows(
            poses=poses,
            valid=valid,
            source_code=source_code,
            entity_masks=entity_masks,
            gt_indexed=gt,
            slot_zero_masks=slot_zero_masks,
        )
        scored.append(episode_scores)
        runtime_rows.append(runtime)
        trace_rows.append(
            {
                "episode_index": episode_index,
                "clean_gt_sha256": _array_trace(decoded["clean"]["gt_indexed"]),
                "hard_gt_sha256": _array_trace(decoded["hard"]["gt_indexed"]),
                "clean_action_sha256": _array_trace(decoded["clean"]["actions"]),
                "hard_action_sha256": _array_trace(decoded["hard"]["actions"]),
                "clean_physics_sha256": _array_trace(
                    decoded["clean"]["physics_states"]
                ),
                "hard_physics_sha256": _array_trace(
                    decoded["hard"]["physics_states"]
                ),
                "poses_sha256": _array_trace(poses),
                "valid_sha256": _array_trace(valid),
                "costs_sha256": _array_trace(costs),
                "weights_sha256": _array_trace(weights),
                "confidence_sha256": _array_trace(confidence),
                "source_code_sha256": _array_trace(source_code),
                "slot_zero_exact_v1_mask_sha256": _array_trace(slot_zero_masks),
                "candidate_bend_mode_sha256": _array_trace(
                    episode_scores["mode_code"]
                ),
            }
        )
    metrics = _score_prefixes(scored)
    metrics["parser_runtime"] = _latency(np.concatenate(runtime_rows))
    traces = {
        "condition_pairing": {
            "clean_hard_gt_byte_identical": True,
            "clean_hard_actions_byte_identical": True,
            "clean_hard_physics_byte_identical": True,
            "generator_evaluated_once_per_physical_frame": True,
            "clean_hard_not_double_counted": True,
            "unique_episodes": episodes,
            "unique_frames": episodes * frames,
        },
        "episode_trace_bundle_sha256": sha256_json(trace_rows),
        "episode_traces": trace_rows,
    }
    return metrics, traces


def _score_real_backend(
    *,
    topk_payload: dict[str, Any],
    topk_paths: dict[tuple[str, int], Path],
    v1_payload: dict[str, Any],
    v1_paths: dict[tuple[str, str, int], Path],
    dataset: dict[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolution = int(dataset["resolution"])
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    output: dict[str, Any] = {}
    trace_output: dict[str, Any] = {}
    for condition in CONDITIONS:
        scored: list[dict[str, np.ndarray]] = []
        parser_runtime: list[np.ndarray] = []
        end_runtime: list[np.ndarray] = []
        ambiguity: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "candidate_count",
                "best_second_cost_margin",
                "weight_entropy",
                "normalized_weight_entropy",
                "weighted_pose_dispersion_px",
                "fit_uncertainty",
            )
        }
        trace_rows = []
        for episode_index in range(episodes):
            arrays = _load_npz(
                topk_paths[(condition, episode_index)], set(OUTPUT_ARRAY_KEYS)
            )
            _validate_decoded_topk_arrays(arrays, frames=frames)
            source_v1 = _load_npz(
                v1_paths[(TASK, condition, episode_index)], V1_ARRAY_KEYS
            )
            entity_nonempty = source_v1["entity_masks"].reshape(frames, 1, -1).any(
                axis=2
            )
            if np.any(
                source_v1["entity_valid"]
                & (source_v1["entity_lost"] | ~entity_nonempty)
            ):
                raise ValueError("Published v1 entity availability is malformed.")
            if np.any(
                arrays["candidate_valid"]
                & ~source_v1["entity_valid"][:, :1]
            ):
                raise ValueError(
                    "Top-K emitted a candidate when the sealed v1 entity was unavailable."
                )
            rank_zero_trace = _validate_rank_zero_exact(
                poses=arrays["poses_xy"],
                valid=arrays["candidate_valid"],
                source_code=arrays["candidate_source_code"],
                source_v1=source_v1,
            )
            v1_record = v1_payload["results"][TASK][condition][episode_index]
            expected_traces = v1_record.get("traces", {})
            if (
                rank_zero_trace != expected_traces.get("role_mask_trace_sha256")
                or _array_trace(source_v1["keypoints_xy"])
                != expected_traces.get("keypoint_trace_sha256")
                or _array_trace(source_v1["entity_masks"])
                != expected_traces.get("entity_mask_trace_sha256")
            ):
                raise ValueError("Decoded v1 role/keypoint/entity trace changed.")
            gt_path = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "top-K scoring arrays",
            )
            gt = _load_npz(
                gt_path, {"gt_indexed", "actions", "physics_states"}
            )["gt_indexed"]
            if (
                gt.shape != (frames, resolution, resolution)
                or gt.dtype != np.uint8
                or set(np.unique(gt).tolist()) - set(range(len(ROLES) + 1))
            ):
                raise ValueError("Top-K frozen GT schema changed.")
            current = _frame_score_rows(
                poses=arrays["poses_xy"],
                valid=arrays["candidate_valid"],
                source_code=arrays["candidate_source_code"],
                entity_masks=source_v1["entity_masks"][:, 0],
                gt_indexed=gt,
                slot_zero_masks=source_v1["role_masks"],
            )
            scored.append(current)
            parser_runtime.append(arrays["parser_runtime_ms"])
            combined = source_v1["cutie_runtime_ms"] + arrays["parser_runtime_ms"]
            if combined.shape != (frames,) or not np.isfinite(combined).all():
                raise ValueError("Top-K combined online runtime schema changed.")
            end_runtime.append(combined)
            for name in ambiguity:
                ambiguity[name].append(arrays[name].astype(np.float64))
            trace_rows.append(
                {
                    "episode_index": episode_index,
                    "rank_zero_exact_v1_role_mask_trace_sha256": rank_zero_trace,
                    "source_v1_role_mask_trace_sha256": _array_trace(
                        source_v1["role_masks"]
                    ),
                    "source_v1_keypoint_trace_sha256": _array_trace(
                        source_v1["keypoints_xy"]
                    ),
                    "candidate_quality_trace_sha256": _array_trace(current["quality"]),
                    "candidate_role_iou_trace_sha256": _array_trace(current["role_iou"]),
                    "candidate_bend_mode_trace_sha256": _array_trace(
                        current["mode_code"]
                    ),
                }
            )
        metrics = _score_prefixes(scored)
        metrics["parser_runtime"] = _latency(np.concatenate(parser_runtime))
        metrics["estimated_online_end_to_end_runtime"] = {
            **_latency(np.concatenate(end_runtime)),
            "semantics": (
                "published_v1_cutie_runtime_plus_topk_parser_runtime_aligned_per_frame;"
                "disk_io_and_privileged_gt_scoring_excluded"
            ),
        }
        metrics["ambiguity_diagnostics"] = {
            name: _percentile_summary(np.concatenate(values))
            for name, values in ambiguity.items()
        }
        metrics["rank_zero"] = {
            "slot": 0,
            "poses_exact_published_v1_keypoints": True,
            "validity_exact_published_v1_projector_validity": True,
            "scored_masks_exact_published_v1_role_mask_bytes": True,
            "not_pose_reconstructed_for_k1": True,
        }
        output[condition] = metrics
        trace_output[condition] = {
            "episode_trace_bundle_sha256": sha256_json(trace_rows),
            "episodes": episodes,
            "frames_per_episode": frames,
        }
    macro: dict[str, Any] = {"prefixes": {}, "paired_episode_bootstrap_gains": {}}
    for k in K_VALUES:
        cells = [output[condition]["prefixes"][str(k)] for condition in CONDITIONS]
        macro["prefixes"][str(k)] = {
            "oracle_set_success_at_0_5": float(
                np.mean([cell["oracle_set_success_at_0_5"] for cell in cells])
            ),
            "oracle_set_success_at_0_75": float(
                np.mean([cell["oracle_set_success_at_0_75"] for cell in cells])
            ),
            "best_min_role_iou_mean": float(
                np.mean([cell["best_min_role_iou"]["mean"] for cell in cells])
            ),
            "episode_success_at_0_5": np.mean(
                np.stack(
                    [
                        np.asarray(cell["episode_success_at_0_5"], dtype=np.float64)
                        for cell in cells
                    ]
                ),
                axis=0,
            ).tolist(),
            "episode_best_min_role_iou_mean": np.mean(
                np.stack(
                    [
                        np.asarray(
                            cell["episode_best_min_role_iou_mean"], dtype=np.float64
                        )
                        for cell in cells
                    ]
                ),
                axis=0,
            ).tolist(),
        }
    for current, previous in ((2, 1), (4, 2), (4, 1)):
        label = f"k{current}_minus_k{previous}"
        current_cell = macro["prefixes"][str(current)]
        previous_cell = macro["prefixes"][str(previous)]
        macro["paired_episode_bootstrap_gains"][label] = {
            "oracle_set_success_at_0_5": _paired_bootstrap_difference(
                np.asarray(current_cell["episode_success_at_0_5"], dtype=np.float64),
                np.asarray(previous_cell["episode_success_at_0_5"], dtype=np.float64),
                label=f"real_macro/{label}/success/0.5",
            ),
            "best_min_role_iou": _paired_bootstrap_difference(
                np.asarray(
                    current_cell["episode_best_min_role_iou_mean"], dtype=np.float64
                ),
                np.asarray(
                    previous_cell["episode_best_min_role_iou_mean"], dtype=np.float64
                ),
                label=f"real_macro/{label}/best",
            ),
        }
    output["cross_condition_macro"] = macro
    return output, trace_output


def _quality_gate_for_k(
    *,
    k: int,
    previous_k: int,
    real: dict[str, Any],
    gt_union: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    oracle = gt_union["prefixes"][str(k)]
    oracle_previous = gt_union["prefixes"][str(previous_k)]
    gain_label = f"k{k}_minus_k{previous_k}"
    oracle_gain = gt_union["paired_episode_bootstrap_gains"][gain_label]
    minimum_cell_success_gain = 0.02 if k == 2 else 0.0
    minimum_cell_quality_gain = 0.01 if k == 2 else 0.0
    minimum_oracle_quality_gain = 0.02 if k == 2 else 0.01
    minimum_macro_gain = 0.02 if k == 2 else 0.01
    checks[f"gt_union/k{k}/availability"] = (
        float(oracle["availability_rate"]) >= MIN_AVAILABILITY
    )
    if k == 2 and oracle["rescue_available_at_k"]["denominator_frames"] > 0:
        checks["gt_union/k2/rescue_available_when_slot0_unavailable"] = (
            float(oracle["rescue_available_at_k"]["rate"]) >= 0.90
        )
    checks[f"gt_union/k{k}/success_at_0_5"] = (
        float(oracle["oracle_set_success_at_0_5"])
        >= MIN_GT_UNION_SUCCESS_AT_05
    )
    checks[f"gt_union/k{k}/burst_at_0_5"] = (
        int(oracle["failure_burst_at_0_5"]["global_max_with_episode_resets"])
        <= MAX_FAILURE_BURST
    )
    checks[f"gt_union/k{k}/episode_p95_burst_at_0_5"] = (
        float(oracle["failure_burst_at_0_5"]["episode_p95"])
        <= MAX_FAILURE_BURST
    )
    previous_oracle_burst = int(
        oracle_previous["failure_burst_at_0_5"]["global_max_with_episode_resets"]
    )
    checks[f"gt_union/k{k}/burst_20pct_shorter_than_k{previous_k}"] = (
        previous_oracle_burst == 0
        or int(oracle["failure_burst_at_0_5"]["global_max_with_episode_resets"])
        <= math.floor(0.8 * previous_oracle_burst)
    )
    checks[f"gt_union/k{k}/success_gain_vs_k{previous_k}"] = (
        float(oracle["oracle_set_success_at_0_5"])
        - float(oracle_previous["oracle_set_success_at_0_5"])
        >= (0.02 if k == 2 else 0.01)
    )
    checks[f"gt_union/k{k}/quality_gain_vs_k{previous_k}"] = (
        float(oracle["best_min_role_iou"]["mean"])
        - float(oracle_previous["best_min_role_iou"]["mean"])
        >= minimum_oracle_quality_gain
    )
    checks[f"gt_union/k{k}/bootstrap_success_lcb_positive"] = (
        float(oracle_gain["oracle_set_success_at_0_5"]["one_sided_lower_95"])
        > 0.0
    )
    checks[f"gt_union/k{k}/bootstrap_quality_lcb_positive"] = (
        float(oracle_gain["best_min_role_iou"]["one_sided_lower_95"]) > 0.0
    )
    for condition in CONDITIONS:
        cell = real[condition]["prefixes"][str(k)]
        previous = real[condition]["prefixes"][str(previous_k)]
        gain = real[condition]["paired_episode_bootstrap_gains"][gain_label]
        checks[f"real/{condition}/k{k}/availability"] = (
            float(cell["availability_rate"]) >= MIN_AVAILABILITY
        )
        if k == 2 and cell["rescue_available_at_k"]["denominator_frames"] > 0:
            checks[
                f"real/{condition}/k2/rescue_available_when_slot0_unavailable"
            ] = float(cell["rescue_available_at_k"]["rate"]) >= 0.90
        checks[f"real/{condition}/k{k}/success_at_0_5"] = (
            float(cell["oracle_set_success_at_0_5"]) >= MIN_REAL_SUCCESS_AT_05
        )
        checks[f"real/{condition}/k{k}/burst_at_0_5"] = (
            int(cell["failure_burst_at_0_5"]["global_max_with_episode_resets"])
            <= MAX_FAILURE_BURST
        )
        checks[f"real/{condition}/k{k}/episode_p95_burst_at_0_5"] = (
            float(cell["failure_burst_at_0_5"]["episode_p95"])
            <= MAX_FAILURE_BURST
        )
        previous_burst = int(
            previous["failure_burst_at_0_5"]["global_max_with_episode_resets"]
        )
        checks[
            f"real/{condition}/k{k}/burst_20pct_shorter_than_k{previous_k}"
        ] = (
            previous_burst == 0
            or int(cell["failure_burst_at_0_5"]["global_max_with_episode_resets"])
            <= math.floor(0.8 * previous_burst)
        )
        checks[f"real/{condition}/k{k}/success_gain_vs_k{previous_k}"] = (
            float(cell["oracle_set_success_at_0_5"])
            - float(previous["oracle_set_success_at_0_5"])
            >= minimum_cell_success_gain
        )
        checks[f"real/{condition}/k{k}/quality_gain_vs_k{previous_k}"] = (
            float(cell["best_min_role_iou"]["mean"])
            - float(previous["best_min_role_iou"]["mean"])
            >= minimum_cell_quality_gain
        )
        checks[f"real/{condition}/k{k}/bootstrap_success_lcb_positive"] = (
            float(gain["oracle_set_success_at_0_5"]["one_sided_lower_95"])
            > 0.0
        )
        checks[f"real/{condition}/k{k}/bootstrap_quality_lcb_positive"] = (
            float(gain["best_min_role_iou"]["one_sided_lower_95"]) > 0.0
        )
    macro = real["cross_condition_macro"]
    macro_cell = macro["prefixes"][str(k)]
    macro_previous = macro["prefixes"][str(previous_k)]
    macro_gain = macro["paired_episode_bootstrap_gains"][gain_label]
    checks[f"real/macro/k{k}/success_gain_vs_k{previous_k}"] = (
        float(macro_cell["oracle_set_success_at_0_5"])
        - float(macro_previous["oracle_set_success_at_0_5"])
        >= minimum_macro_gain
    )
    checks[f"real/macro/k{k}/quality_gain_vs_k{previous_k}"] = (
        float(macro_cell["best_min_role_iou_mean"])
        - float(macro_previous["best_min_role_iou_mean"])
        >= minimum_macro_gain
    )
    checks[f"real/macro/k{k}/bootstrap_success_lcb_at_least_0_01"] = (
        float(macro_gain["oracle_set_success_at_0_5"]["one_sided_lower_95"])
        >= 0.01
    )
    checks[f"real/macro/k{k}/bootstrap_quality_lcb_at_least_0_01"] = (
        float(macro_gain["best_min_role_iou"]["one_sided_lower_95"])
        >= 0.01
    )
    return checks


def _development_gate(
    *, real: dict[str, Any], gt_union: dict[str, Any]
) -> dict[str, Any]:
    deployment: dict[str, bool] = {
        "rank_zero_exact_all_conditions": all(
            bool(real[condition]["rank_zero"]["scored_masks_exact_published_v1_role_mask_bytes"])
            for condition in CONDITIONS
        )
    }
    for condition in CONDITIONS:
        parser = real[condition]["parser_runtime"]
        end = real[condition]["estimated_online_end_to_end_runtime"]
        deployment[f"real/{condition}/parser_mean"] = (
            float(parser["mean_ms"]) <= MAX_PARSER_MEAN_MS
        )
        deployment[f"real/{condition}/parser_p95"] = (
            float(parser["p95_ms"]) <= MAX_PARSER_P95_MS
        )
        deployment[f"real/{condition}/end_to_end_mean"] = (
            float(end["mean_ms"]) <= MAX_END_TO_END_MEAN_MS
        )
        deployment[f"real/{condition}/end_to_end_p95"] = (
            float(end["p95_ms"]) <= MAX_END_TO_END_P95_MS
        )
    oracle_runtime = gt_union["parser_runtime"]
    deployment["gt_union/parser_mean"] = (
        float(oracle_runtime["mean_ms"]) <= MAX_PARSER_MEAN_MS
    )
    deployment["gt_union/parser_p95"] = (
        float(oracle_runtime["p95_ms"]) <= MAX_PARSER_P95_MS
    )
    deployment_budget_pass = bool(deployment) and all(deployment.values())
    quality_by_k = {
        "2": _quality_gate_for_k(
            k=2, previous_k=1, real=real, gt_union=gt_union
        ),
        "4": _quality_gate_for_k(
            k=4, previous_k=2, real=real, gt_union=gt_union
        ),
    }
    quality_pass = {
        key: bool(checks) and all(checks.values())
        for key, checks in quality_by_k.items()
    }
    selected: int | None = None
    if deployment_budget_pass and quality_pass["2"]:
        selected = 2
    elif deployment_budget_pass and not quality_pass["2"] and quality_pass["4"]:
        selected = 4
    return {
        "deployment_runtime_and_parity_checks": deployment,
        "deployment_budget_pass": deployment_budget_pass,
        "quality_checks_by_k": quality_by_k,
        "quality_pass_by_k": quality_pass,
        "minimum_k_policy": "select_k2_if_pass_else_k4_if_pass_else_no_go",
        "selected_k": selected,
        "development_candidate": selected is not None,
    }


def _artifact_hash_snapshot(
    *,
    source_root: Path,
    dataset: dict[str, Any],
    dataset_root: Path,
    v1_summary_path: Path,
    v1_manifest_path: Path,
    v1_paths: dict[tuple[str, str, int], Path],
    topk_manifest_path: Path,
    topk_paths: dict[tuple[str, int], Path],
    graph_path: Path,
    support_path: Path,
    isolation_path: Path,
    immutable_inputs_path: Path,
) -> dict[str, str]:
    paths: dict[str, Path] = {
        "source_summary": source_root / "unified_vos_summary.json",
        "dataset_manifest": dataset_root / "dataset_manifest.json",
        "v1_summary": v1_summary_path,
        "v1_manifest": v1_manifest_path,
        "topk_manifest": topk_manifest_path,
        "graph": graph_path,
        "support": support_path,
        "isolation": isolation_path,
        "immutable_inputs": immutable_inputs_path,
        "aggregator": Path(__file__).resolve(),
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
            paths[f"topk/{condition}/{episode_index}"] = topk_paths[
                (condition, episode_index)
            ]
            paths[f"gt/{condition}/{episode_index}"] = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "top-K immutable scoring arrays",
            )
    output: dict[str, str] = {}
    for label, path in sorted(paths.items()):
        output[label] = file_sha256(_regular_file(path, f"bound artifact {label}"))
    return output


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
    if args.v1_backend_manifest.resolve(strict=True) != v1_manifest_path:
        raise ValueError(
            "--v1-backend-manifest is not the exact manifest published by v1 summary."
        )
    if MAX_CANDIDATES != 4 or K_VALUES != (1, 2, 4):
        raise RuntimeError("The preregistered top-K prefix set changed.")
    generator, support_path = _generator_from_frozen_support(
        graph=v1_graphs[TASK],
        dataset=dataset,
        dataset_root=dataset_path.parent,
        source_root=args.source_benchmark_root,
    )
    immutable_inputs = _validate_immutable_inputs(
        args.immutable_inputs,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        summary_root=args.output.parent,
    )
    isolation = _validate_isolation_gate(
        args.isolation_gate,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        summary_root=args.output.parent,
        backend_manifest_path=args.topk_backend,
    )
    topk_payload, topk_paths = _validate_topk_backend(
        args.topk_backend,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        v1_payload=v1_payload,
        v1_paths=v1_paths,
        v1_manifest_path=v1_manifest_path,
        v1_graph=v1_graphs[TASK],
        generator=generator,
        support_path=support_path,
    )
    snapshot_before = _artifact_hash_snapshot(
        source_root=args.source_benchmark_root,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        v1_summary_path=args.v1_preflight_root / "object_graph_tokenizer_summary.json",
        v1_manifest_path=v1_manifest_path,
        v1_paths=v1_paths,
        topk_manifest_path=args.topk_backend,
        topk_paths=topk_paths,
        graph_path=args.v1_graph,
        support_path=support_path,
        isolation_path=args.isolation_gate,
        immutable_inputs_path=args.immutable_inputs,
    )
    real_metrics, real_traces = _score_real_backend(
        topk_payload=topk_payload,
        topk_paths=topk_paths,
        v1_payload=v1_payload,
        v1_paths=v1_paths,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    gt_union_metrics, gt_union_traces = _score_gt_union_once(
        generator=generator,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    gate = _development_gate(real=real_metrics, gt_union=gt_union_metrics)
    immutable_inputs_after = _validate_immutable_inputs(
        args.immutable_inputs,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        summary_root=args.output.parent,
    )
    if immutable_inputs_after != immutable_inputs:
        raise RuntimeError("Top-K immutable inputs changed during scoring.")
    snapshot_after = _artifact_hash_snapshot(
        source_root=args.source_benchmark_root,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        v1_summary_path=args.v1_preflight_root / "object_graph_tokenizer_summary.json",
        v1_manifest_path=v1_manifest_path,
        v1_paths=v1_paths,
        topk_manifest_path=args.topk_backend,
        topk_paths=topk_paths,
        graph_path=args.v1_graph,
        support_path=support_path,
        isolation_path=args.isolation_gate,
        immutable_inputs_path=args.immutable_inputs,
    )
    if snapshot_after != snapshot_before:
        raise RuntimeError("A bound top-K input or implementation changed during scoring.")
    development_candidate = bool(gate["development_candidate"])
    selected_k = gate["selected_k"]
    summary = {
        "format": SUMMARY_FORMAT,
        "status": (
            f"object_graph_topk_k{selected_k}_development_candidate"
            if development_candidate
            else "object_graph_topk_candidate_coverage_development_no_go"
        ),
        "engineering_pass": True,
        "development_candidate": development_candidate,
        "selected_k": selected_k,
        "controller_training_authorized": False,
        "scientific_go": False,
        "recommendation": (
            "freeze_unseen_trajectory_seeds_and_run_confirmatory_candidate_coverage_only"
            if development_candidate
            else "do_not_train_controller_topk_candidates_failed_coverage_or_latency_gate"
        ),
        "scope": {
            "evidence_level": (
                "post_hoc_offline_oracle_set_coverage_on_previously_inspected_frozen_trajectories"
            ),
            "task": TASK,
            "same_frozen_inputs_as_v1": True,
            "backend_current_entity_mask_and_status_only": True,
            "backend_episode_ground_truth_access": False,
            "aggregator_episode_ground_truth_access_after_backend_and_restore": True,
            "slot_zero_scored_from_exact_published_v1_role_masks": True,
            "slot_zero_pose_used_only_for_strict_keypoint_parity": True,
            "alternative_masks_reconstructed_from_sealed_pose_and_entity_mask": True,
            "k_prefixes": list(K_VALUES),
            "max_k_generated_once": MAX_CANDIDATES,
            "valid_slots_may_be_sparse_due_to_reserved_anchor": True,
            "gt_union_clean_hard_evaluated_once_not_double_counted": True,
            "oracle_selects_candidate_only_for_privileged_offline_scoring": True,
            "candidate_weights_not_calibrated_for_controller_use": True,
            "controller_constructed": False,
            "controller_descriptors_constructed": False,
            "controller_training_steps": 0,
            "allowed_claim": "stateless_topk_candidate_coverage_development_signal",
            "disallowed_claims": [
                "online_candidate_selection_solved",
                "controller_advantage",
                "controller_training_authorization",
                "confirmatory_perception_result",
                "paper_level_generality",
            ],
        },
        "thresholds": {
            "minimum_gt_union_oracle_set_success_at_0_5": MIN_GT_UNION_SUCCESS_AT_05,
            "minimum_real_oracle_set_success_at_0_5": MIN_REAL_SUCCESS_AT_05,
            "minimum_availability": MIN_AVAILABILITY,
            "minimum_k2_rescue_availability_when_slot0_unavailable": 0.90,
            "zero_rescue_denominator_policy": "report_not_applicable_no_boolean_gate",
            "maximum_global_failure_burst": MAX_FAILURE_BURST,
            "maximum_episode_p95_failure_burst": MAX_FAILURE_BURST,
            "required_burst_reduction_fraction_vs_previous_prefix": 0.20,
            "k2_minimum_per_cell_success_gain_vs_k1": 0.02,
            "k2_minimum_per_cell_quality_gain_vs_k1": 0.01,
            "k2_minimum_real_macro_and_gt_union_quality_gain_vs_k1": 0.02,
            "k4_minimum_real_macro_and_gt_union_gain_vs_k2": 0.01,
            "paired_bootstrap_unit": "episode",
            "paired_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "paired_bootstrap_one_sided_lcb": 0.95,
            "minimum_real_macro_quality_gain_lcb": 0.01,
            "maximum_parser_mean_ms": MAX_PARSER_MEAN_MS,
            "maximum_parser_p95_ms": MAX_PARSER_P95_MS,
            "maximum_estimated_end_to_end_mean_ms": MAX_END_TO_END_MEAN_MS,
            "maximum_estimated_end_to_end_p95_ms": MAX_END_TO_END_P95_MS,
            "minimum_k_policy": "k2_then_k4_only_if_k2_quality_fails",
        },
        "gate": gate,
        "real_v1_entity_topk_coverage": real_metrics,
        "real_v1_entity_topk_traces": real_traces,
        "gt_union_topk_coverage_counted_once": gt_union_metrics,
        "gt_union_topk_traces": gt_union_traces,
        "generator": generator.metadata(),
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
            "v1_preflight_summary_sha256": file_sha256(
                args.v1_preflight_root / "object_graph_tokenizer_summary.json"
            ),
            "v1_backend_manifest": str(v1_manifest_path),
            "v1_backend_manifest_sha256": file_sha256(v1_manifest_path),
            "topk_backend_manifest_relative_to_summary_root": (
                args.topk_backend.relative_to(args.output.parent).as_posix()
            ),
            "topk_backend_manifest_sha256": file_sha256(args.topk_backend),
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
            "support_arrays": str(support_path),
            "support_arrays_sha256": file_sha256(support_path),
            "bound_artifact_snapshot_sha256": sha256_json(snapshot_before),
        },
    }
    write_json(args.output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--v1-preflight-root", type=Path, required=True)
    parser.add_argument("--v1-backend-manifest", type=Path, required=True)
    parser.add_argument("--topk-backend", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
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
    for name, label in (
        ("v1_backend_manifest", "published v1 backend manifest"),
        ("topk_backend", "top-K replay backend manifest"),
        ("v1_graph", "canonical v1 graph"),
        ("isolation_gate", "top-K scoring isolation gate"),
        ("immutable_inputs", "top-K immutable-input snapshot"),
    ):
        setattr(args, name, _regular_file(getattr(args, name), label))
    output_lexical = args.output.expanduser().absolute()
    for component in reversed(output_lexical.parents):
        if component.is_symlink():
            raise ValueError(f"Output contains a symlink ancestor: {component}")
    output_parent = _regular_dir(output_lexical.parent, "output parent")
    args.output = output_parent / output_lexical.name
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = aggregate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "engineering_pass": result["engineering_pass"],
                "development_candidate": result["development_candidate"],
                "selected_k": result["selected_k"],
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
