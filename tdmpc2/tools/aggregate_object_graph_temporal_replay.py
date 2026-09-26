"""Strict offline replay diagnostic for the temporal-v2 role projector.

This aggregation is deliberately narrower than the object-graph v1 preflight.
It reuses the *published* Acrobot entity masks/status from that run, scores the
real temporal-v2 parser replay, and then runs the same parser against exact GT
entity unions after scoring access has been restored.  The latter is a
privileged offline parser diagnostic.

No episode RGB is decoded or used as parser input; support RGB is only
schema-validated.  No raw Cutie appearance feature, controller descriptor,
controller, or controller training is part of this script.  A passing result
is only a development candidate on previously inspected frozen trajectories
and can never authorize controller training.
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

from tdmpc2.common.object_graph_temporal_replay_snapshot import (
    FORMAT as IMMUTABLE_INPUTS_FORMAT,
    build as build_immutable_inputs,
)
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
from tdmpc2.perception.support_conditioned_object_graph import (
    TEMPORAL_STATE_PROTOCOL,
    TEMPORAL_TOKENIZER_FORMAT,
    SupportConditionedObjectGraphTokenizer,
    load_object_graph,
)
from tdmpc2.tools.aggregate_object_graph_tokenizer_preflight import (
    ARRAY_KEYS as V1_ARRAY_KEYS,
    BACKEND_FORMAT as V1_BACKEND_FORMAT,
    BACKEND_NAME as V1_BACKEND_NAME,
    SUMMARY_FORMAT as V1_SUMMARY_FORMAT,
    _entity_gt,
    _graph_paths,
    _invalid_bursts,
    _score_graph_backend,
    _score_gt_union_parser_oracle as _score_v1_gt_union_parser_oracle,
    _source_artifacts,
    _validate_isolation_gate as _validate_v1_isolation_gate,
    _validate_graph_backend,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
    MaskScorer,
    _array_trace,
    _load_npz,
    _validate_backend_manifest,
)


TASK = "acrobot-swingup"
ROLES = TASK_ROLES[TASK]
V2_BACKEND_FORMAT = "object_graph_temporal_v2_replay_predictions_v1"
V2_BACKEND_NAME = "object_graph_temporal_v2_replay"
SUMMARY_FORMAT = "object_graph_temporal_v2_replay_summary_v1"
TEMPORAL_PROJECTOR = "ordered_chain_temporal_v2"
ISOLATION_FORMAT = "object_graph_temporal_replay_scoring_isolation_v1"
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MODE = re.compile(r"[0-7]{3}")

V2_ARRAY_KEYS = {
    "role_masks",
    "keypoints_xy",
    "projector_valid",
    "projector_confidence",
    "role_valid",
    "role_confidence",
    "role_lost",
    "role_mask_score",
    "parser_runtime_ms",
}
V2_TRACE_KEYS = {
    "role_mask_trace_sha256",
    "keypoint_trace_sha256",
    "projector_status_trace_sha256",
    "role_status_trace_sha256",
    "parser_runtime_trace_sha256",
    "diagnostics_trace_sha256",
}
QUALITY_METRICS = (
    "visible_recall",
    "mean_iou_on_gt_visible_frames",
    "tolerant_f1_radius_2_on_gt_visible_frames",
    "identity_accuracy_on_gt_visible_frames",
    "success_at_iou_0_5_rate_on_gt_visible_frames",
)
MAX_FAILURE_BURST = 10
MAX_INVALID_BURST = 10
MAX_PARSER_MEAN_MS = 5.0
MAX_PARSER_P95_MS = 10.0
QUALITY_REGRESSION_TOLERANCE = 0.0


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    result = candidate.resolve(strict=True)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _regular_dir(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    result = candidate.resolve(strict=True)
    if not result.is_dir():
        raise NotADirectoryError(result)
    return result


def _contained_file(root: Path, relative: Any, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
    ):
        raise ValueError(f"{label} must be a POSIX relative path.")
    lexical = (root / relative).absolute()
    for component in (*reversed(lexical.parents), lexical):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    path = lexical.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes its publication root.") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _v2_shapes(*, frames: int, roles: int, resolution: int) -> dict[str, list[int]]:
    return {
        "role_masks": [frames, roles, resolution, resolution],
        "keypoints_xy": [frames, roles, 2, 2],
        "projector_valid": [frames, roles],
        "projector_confidence": [frames, roles],
        "role_valid": [frames, roles],
        "role_confidence": [frames, roles],
        "role_lost": [frames, roles],
        "role_mask_score": [frames, roles],
        "parser_runtime_ms": [frames],
    }


def _dtype_contract(name: str, value: np.ndarray) -> None:
    expected = {
        "role_masks": np.bool_,
        "keypoints_xy": np.float32,
        "projector_valid": np.bool_,
        "projector_confidence": np.float32,
        "role_valid": np.bool_,
        "role_confidence": np.float32,
        "role_lost": np.bool_,
        "role_mask_score": np.float32,
        "parser_runtime_ms": np.float64,
    }[name]
    if value.dtype != expected:
        raise ValueError(f"Temporal-v2 {name} dtype changed: {value.dtype} != {expected}.")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"Temporal-v2 {name} contains a non-finite value.")


def _status_trace(arrays: dict[str, np.ndarray]) -> str:
    status = np.concatenate(
        (
            arrays["role_valid"].astype(np.float32)[..., None],
            arrays["role_confidence"][..., None],
            arrays["role_lost"].astype(np.float32)[..., None],
            arrays["role_mask_score"][..., None],
        ),
        axis=-1,
    )
    return _array_trace(status)


def _projector_status_trace(arrays: dict[str, np.ndarray]) -> str:
    status = np.concatenate(
        (
            arrays["projector_valid"].astype(np.float32)[..., None],
            arrays["projector_confidence"][..., None],
        ),
        axis=-1,
    )
    return _array_trace(status)


def _decoded_traces(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        "role_mask_trace_sha256": _array_trace(arrays["role_masks"]),
        "keypoint_trace_sha256": _array_trace(arrays["keypoints_xy"]),
        "projector_status_trace_sha256": _projector_status_trace(arrays),
        "role_status_trace_sha256": _status_trace(arrays),
        "parser_runtime_trace_sha256": _array_trace(arrays["parser_runtime_ms"]),
    }


def _metric(role: dict[str, Any], name: str) -> float:
    value = role.get(name)
    return 0.0 if value is None else float(value)


def _load_temporal_graph(path: Path, v1_graph: Any) -> Any:
    path = _regular_file(path, "temporal-v2 graph")
    graph = load_object_graph(path)
    if (
        graph.task != TASK
        or graph.source_roles != ROLES
        or graph.semantic_roles != ROLES
        or len(graph.entities) != 1
        or len(v1_graph.entities) != 1
    ):
        raise ValueError("Temporal-v2 graph is not the paired Acrobot one-entity graph.")
    current = graph.entities[0]
    previous = v1_graph.entities[0]
    if current.projector.type != TEMPORAL_PROJECTOR:
        raise ValueError("Temporal-v2 graph did not select the temporal projector.")
    if (
        current.name != previous.name
        or current.source_roles != previous.source_roles
        or current.projector.roles != previous.projector.roles
        or graph.relations != v1_graph.relations
        or graph.stack_frames != v1_graph.stack_frames
    ):
        raise ValueError("Temporal-v2 graph changed the v1 entity/role ontology.")
    paired_fields = (
        "minimum_entity_pixels",
        "maximum_base_distance_fraction",
        "minimum_path_length_fraction",
        "maximum_path_length_fraction",
        "maximum_disconnected_fraction",
        "maximum_branch_fraction",
        "maximum_entity_area_multiple",
        "minimum_projection_confidence",
    )
    for field in paired_fields:
        if getattr(current.projector, field) != getattr(previous.projector, field):
            raise ValueError(f"Temporal-v2 graph changed paired v1 field {field}.")
    return graph


def _strict_v1_pair(
    *,
    source_root: Path,
    v1_preflight_root: Path,
    v1_graph_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    dict[tuple[str, str, int], Path],
    dict[str, Any],
    dict[tuple[str, str, int], Path],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    source_root = _regular_dir(source_root, "frozen unified source root")
    v1_preflight_root = _regular_dir(v1_preflight_root, "published v1 preflight root")
    if (
        source_root == v1_preflight_root
        or source_root in v1_preflight_root.parents
        or v1_preflight_root in source_root.parents
    ):
        raise ValueError("Frozen source and v1 preflight roots must be disjoint.")
    source_summary, dataset, dataset_path, cutie_manifest_path = _source_artifacts(
        source_root
    )
    v1_graph_path = _regular_file(v1_graph_path, "published v1 Acrobot graph")
    graph_paths = _graph_paths(v1_graph_path.parent)
    if graph_paths[TASK] != v1_graph_path:
        raise ValueError("--v1-graph is not the canonical v1 Acrobot graph file.")
    cutie_payload, cutie_paths = _validate_backend_manifest(
        cutie_manifest_path, backend="cutie", dataset=dataset
    )
    v1_summary_path = _regular_file(
        v1_preflight_root / "object_graph_tokenizer_summary.json",
        "published v1 summary",
    )
    v1_summary = load_json(v1_summary_path)
    if (
        v1_summary.get("format") != V1_SUMMARY_FORMAT
        or v1_summary.get("engineering_pass") is not True
        or v1_summary.get("controller_training_authorized") is not False
        or v1_summary.get("scientific_go") is not False
        or v1_summary.get("scope", {}).get("controller_training_steps") != 0
    ):
        raise ValueError("Published v1 preflight is not an eligible replay source.")
    published = v1_summary.get("provenance")
    if not isinstance(published, dict):
        raise ValueError("Published v1 summary provenance is missing.")
    if (
        published.get("source_benchmark_root") != str(source_root)
        or published.get("source_summary_sha256")
        != file_sha256(source_root / "unified_vos_summary.json")
        or published.get("source_dataset_manifest_sha256") != file_sha256(dataset_path)
        or published.get("source_dataset_id") != dataset.get("dataset_id")
        or published.get("source_cutie_manifest_sha256")
        != file_sha256(cutie_manifest_path)
    ):
        raise ValueError("Published v1 preflight names a different frozen source.")
    v1_manifest_path = _contained_file(
        v1_preflight_root,
        published.get("object_graph_backend_manifest_relative_to_summary_root"),
        "published v1 graph manifest",
    )
    if file_sha256(v1_manifest_path) != require_sha256(
        published.get("object_graph_backend_manifest_sha256"),
        "published v1 graph manifest SHA",
    ):
        raise ValueError("Published v1 graph manifest changed after publication.")
    v1_isolation_path = _contained_file(
        v1_preflight_root,
        published.get("scoring_isolation_relative_to_summary_root"),
        "published v1 scoring isolation",
    )
    if file_sha256(v1_isolation_path) != require_sha256(
        published.get("scoring_isolation_sha256"), "published v1 isolation SHA"
    ):
        raise ValueError("Published v1 isolation record changed after publication.")
    _validate_v1_isolation_gate(
        v1_isolation_path,
        source_root=source_root,
        summary_root=v1_preflight_root,
        backend_manifest=v1_manifest_path,
    )
    v1_payload, v1_paths, v1_graphs = _validate_graph_backend(
        v1_manifest_path,
        dataset=dataset,
        graph_paths=graph_paths,
    )
    if (
        v1_payload.get("format") != V1_BACKEND_FORMAT
        or v1_payload.get("backend") != V1_BACKEND_NAME
    ):
        raise ValueError("The supplied source is not the published v1 graph backend.")
    v1_provenance = v1_payload["backend_provenance"]
    cutie_provenance = cutie_payload["backend_provenance"]
    for field in (
        "model_family",
        "model_size",
        "tracker_size",
        "checkpoint_sha256",
        "seed",
        "python",
        "torch",
        "cuda_runtime",
        "device_name",
        "gpu_uuid",
        "amp",
    ):
        if v1_provenance.get(field) != cutie_provenance.get(field):
            raise ValueError(f"Published v1/Cutie provenance differs for {field}.")
    if v1_provenance.get("cutie_implementation") != cutie_provenance.get(
        "implementation"
    ):
        raise ValueError("Published v1/Cutie implementation trees differ.")
    graph_record = published.get("graph_files", {}).get(TASK)
    if (
        not isinstance(graph_record, dict)
        or graph_record.get("path") != str(v1_graph_path)
        or graph_record.get("file_sha256") != file_sha256(v1_graph_path)
        or graph_record.get("semantic_sha256") != v1_graphs[TASK].graph_sha256
    ):
        raise ValueError("Published v1 summary/Acrobot graph binding changed.")
    return (
        source_summary,
        dataset,
        dataset_path,
        cutie_manifest_path,
        cutie_payload,
        cutie_paths,
        v1_payload,
        v1_paths,
        v1_graphs,
        v1_summary,
        v1_manifest_path,
    )


def _validate_protocol(protocol: Any) -> None:
    """Validate the frozen replay-only causality disclosure.

    This exact schema is shared with run_object_graph_temporal_v2_replay_backend.
    """
    expected = {
        "diagnostic_scope": "offline_causal_mask_parser_replay_v1",
        "source_backend": V1_BACKEND_NAME,
        "source_entity_masks_consumed": True,
        "source_entity_status_consumed": True,
        "frozen_support_indexed_masks_consumed": True,
        "support_rgb_schema_validated_not_used": True,
        "episode_rgb_bytes_hashed_for_input_validation": True,
        "episode_rgb_parser_input": False,
        "episode_rgb_decoded": False,
        "episode_ground_truth_read": False,
        "simulator_state_read": False,
        "actions_read": False,
        "rewards_read": False,
        "episode_entity_arrays_loaded_before_replay": True,
        "future_entity_frames_parser_input": False,
        "causal_frame_order": True,
        "episode_reset_before_frame_zero": True,
        "appearance_features_replayed": False,
        "appearance_not_replayed": True,
        "source_entity_valid_exactly_mask_lost_reproducible": True,
        "descriptors_emitted": False,
        "controller_token_not_evaluated": True,
        "controller_training_eligible": False,
        "cpu_only": True,
        "runtime_ms_semantics": (
            "project_temporal_geometry_internal_per_current_entity_mask_v1"
        ),
    }
    if protocol != expected:
        raise ValueError("Temporal-v2 replay protocol changed.")


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):03o}"


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{label} must be a whole-second UTC timestamp.")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _validate_isolation_gate(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    summary_root: Path,
    v2_manifest_path: Path,
) -> dict[str, Any]:
    path = _regular_file(path, "temporal replay scoring isolation")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative_gate = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Temporal replay isolation gate is outside the summary root.") from exc
    if relative_gate != "provenance/scoring_isolation.json":
        raise ValueError("Temporal replay isolation gate is not at its fixed path.")
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
        "v2_backend_manifest_relative_to_summary_root",
        "v2_backend_manifest_sha256",
        "same_uid_read_probe",
    }
    if set(payload) != expected:
        raise ValueError("Temporal replay isolation schema changed.")
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
        raise ValueError("Temporal replay isolation identity/causality changed.")
    relative_probe = payload.get("scoring_probe_relative_to_source")
    if (
        not isinstance(relative_probe, str)
        or not relative_probe
        or "\\" in relative_probe
    ):
        raise ValueError("Temporal replay scoring probe path is malformed.")
    probe = _contained_file(
        source_root, relative_probe, "temporal replay scoring probe"
    )
    try:
        probe.relative_to(scoring_root)
    except ValueError as exc:
        raise ValueError("Temporal replay probe is outside scoring-only data.") from exc
    if not probe.is_file():
        raise FileNotFoundError(probe)
    for field in ("root_mode_before", "root_mode_locked", "root_mode_restored"):
        if not isinstance(payload.get(field), str) or _MODE.fullmatch(payload[field]) is None:
            raise ValueError(f"Temporal replay isolation mode {field} is malformed.")
    if (
        payload["root_mode_locked"] != "000"
        or payload["root_mode_before"] != payload["root_mode_restored"]
        or payload["root_mode_before"] == "000"
        or int(payload["root_mode_before"], 8) & 0o500 != 0o500
        or _mode(scoring_root) != payload["root_mode_restored"]
    ):
        raise ValueError("Temporal replay scoring permissions were not restored exactly.")
    before_sha = require_sha256(
        payload.get("probe_sha256_before"), "temporal replay pre-lock probe SHA"
    )
    restored_sha = require_sha256(
        payload.get("probe_sha256_restored"), "temporal replay restored probe SHA"
    )
    if before_sha != restored_sha or file_sha256(probe) != restored_sha:
        raise ValueError("Temporal replay scoring probe changed across the lock.")
    times = [
        _parse_utc(payload[field], field)
        for field in ("locked_utc", "backend_completed_utc", "restored_utc")
    ]
    if times != sorted(times):
        raise ValueError("Temporal replay isolation timestamps are out of order.")
    v2_manifest_path = v2_manifest_path.resolve(strict=True)
    try:
        expected_v2_relative = v2_manifest_path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Temporal-v2 backend manifest is outside the summary root.") from exc
    if (
        expected_v2_relative
        != "backends/object_graph_temporal_v2/backend_predictions.json"
        or
        payload.get("v2_backend_manifest_relative_to_summary_root")
        != expected_v2_relative
        or payload.get("v2_backend_manifest_sha256") != file_sha256(v2_manifest_path)
    ):
        raise ValueError("Temporal replay isolation names a different backend manifest.")
    read_probe = payload.get("same_uid_read_probe")
    if not isinstance(read_probe, dict) or set(read_probe) != {
        "exit_code",
        "error_type",
        "log_relative_to_summary_root",
        "log_sha256",
    }:
        raise ValueError("Temporal replay same-user read probe schema changed.")
    if (
        read_probe.get("exit_code") != 1
        or type(read_probe.get("exit_code")) is not int
        or read_probe.get("error_type") != "PermissionError"
        or read_probe.get("log_relative_to_summary_root")
        != "contracts/scoring_read_probe.log"
    ):
        raise ValueError("Temporal replay scoring read probe did not fail closed.")
    log_path = _contained_file(
        summary_root,
        read_probe["log_relative_to_summary_root"],
        "temporal replay scoring read-probe log",
    )
    if file_sha256(log_path) != require_sha256(
        read_probe.get("log_sha256"), "temporal replay read-probe log SHA"
    ) or b"PermissionError" not in log_path.read_bytes():
        raise ValueError("Temporal replay read-probe log is not the bound permission failure.")
    return payload


def _validate_immutable_inputs(
    path: Path,
    *,
    source_root: Path,
    v1_preflight_root: Path,
    summary_root: Path,
    v1_graph_path: Path,
    temporal_graph_path: Path,
) -> dict[str, Any]:
    path = _regular_file(path, "temporal replay immutable-input snapshot")
    summary_root = summary_root.resolve(strict=True)
    try:
        relative = path.relative_to(summary_root).as_posix()
    except ValueError as exc:
        raise ValueError("Immutable-input snapshot is outside the summary root.") from exc
    if relative != "provenance/immutable_inputs.json":
        raise ValueError("Immutable-input snapshot is not at its fixed path.")
    payload = load_json(path)
    rebuilt = build_immutable_inputs(
        source_root=source_root,
        v1_root=v1_preflight_root,
    )
    if (
        payload.get("format") != IMMUTABLE_INPUTS_FORMAT
        or payload != rebuilt
        or path.read_bytes() != canonical_json_bytes(payload)
    ):
        raise ValueError("Temporal replay immutable inputs changed.")
    if (
        payload.get("source_benchmark_root") != str(source_root.resolve(strict=True))
        or payload.get("v1_preflight_root")
        != str(v1_preflight_root.resolve(strict=True))
        or payload.get("dataset_id") != payload.get("v1_backend_dataset_id")
    ):
        raise ValueError("Temporal replay immutable source/v1 pairing changed.")
    local = payload.get("local_source")
    rows = local.get("files") if isinstance(local, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Immutable local-source inventory is missing.")
    package_root = Path(__file__).resolve().parents[1]
    expected_graphs = {}
    for label, graph_path in (
        ("v1", v1_graph_path),
        ("temporal_v2", temporal_graph_path),
    ):
        graph_path = graph_path.resolve(strict=True)
        try:
            relative_graph = graph_path.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{label} graph is outside the inventoried tdmpc2 tree.") from exc
        expected_graphs[relative_graph] = file_sha256(graph_path)
    indexed = {
        row.get("path"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    for relative_graph, digest in expected_graphs.items():
        row = indexed.get(relative_graph)
        if (
            not isinstance(row, dict)
            or row.get("sha256") != digest
            or row.get("bytes")
            != (package_root / relative_graph).stat().st_size
        ):
            raise ValueError("Immutable snapshot graph binding changed.")
    return payload


def _validate_v2_manifest(
    manifest_path: Path,
    *,
    dataset: dict[str, Any],
    dataset_root: Path,
    v1_manifest_path: Path,
    v1_payload: dict[str, Any],
    v1_paths: dict[tuple[str, str, int], Path],
    v1_graph: Any,
    temporal_graph_path: Path,
    temporal_graph: Any,
) -> tuple[dict[str, Any], dict[tuple[str, int], Path]]:
    manifest_path = _regular_file(manifest_path, "temporal-v2 replay manifest")
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
        "graphs",
        "protocol",
        "backend_provenance",
        "results",
    }
    if set(payload) != expected_top:
        raise ValueError("Temporal-v2 replay manifest schema changed.")
    if (
        payload.get("format") != V2_BACKEND_FORMAT
        or payload.get("status") != "complete"
        or payload.get("backend") != V2_BACKEND_NAME
        or payload.get("task") != TASK
        or payload.get("dataset_id") != dataset.get("dataset_id")
        or payload.get("input_manifest_sha256")
        != dataset.get("backend_inputs", {}).get("sha256")
        or payload.get("v1_backend_manifest_sha256") != file_sha256(v1_manifest_path)
        or payload.get("roles") != list(ROLES)
    ):
        raise ValueError("Temporal-v2 replay identity/pairing changed.")
    _validate_protocol(payload.get("protocol"))

    graph_map = payload.get("graphs")
    if not isinstance(graph_map, dict) or set(graph_map) != {"v1", "temporal_v2"}:
        raise ValueError("Temporal-v2 replay graph-pair schema changed.")
    v1_graph_record = graph_map["v1"]
    graph_record = graph_map["temporal_v2"]
    if not isinstance(v1_graph_record, dict) or set(v1_graph_record) != {
        "path",
        "file_sha256",
        "graph",
    }:
        raise ValueError("Temporal replay v1 graph provenance schema changed.")
    if not isinstance(graph_record, dict) or set(graph_record) != {
        "path",
        "file_sha256",
        "graph",
        "tokenizer",
        "derivation",
    }:
        raise ValueError("Temporal-v2 graph provenance schema changed.")
    v1_entry = v1_payload["graphs"][TASK]
    if (
        v1_graph_record.get("path") != str(v1_graph.source_path)
        or v1_graph_record.get("file_sha256")
        != v1_entry.get("graph_file_sha256")
        or v1_graph_record.get("graph") != v1_graph.metadata()
        or graph_record.get("path") != str(temporal_graph_path)
        or graph_record.get("file_sha256") != file_sha256(temporal_graph_path)
        or graph_record.get("graph") != temporal_graph.metadata()
        or graph_record.get("derivation")
        != "same_frozen_graph_except_graph_name_and_ordered_chain_projector_version_v2"
    ):
        raise ValueError("Temporal-v2/v1 graph provenance pairing changed.")
    support_path = resolve_member(
        dataset_root,
        dataset["support"][TASK]["arrays"],
        "temporal-v2 support calibration arrays",
    )
    support = _load_npz(support_path, {"rgb", "indexed_masks"})
    expected_tokenizer = SupportConditionedObjectGraphTokenizer(
        temporal_graph, support["indexed_masks"]
    ).metadata()
    tokenizer_metadata = graph_record.get("tokenizer")
    if tokenizer_metadata != expected_tokenizer or (
        tokenizer_metadata.get("format") != TEMPORAL_TOKENIZER_FORMAT
        or tokenizer_metadata.get("temporal_state_protocol")
        != TEMPORAL_STATE_PROTOCOL
        or tokenizer_metadata.get("geometry_replay_entrypoint")
        != "project_temporal_geometry"
        or tokenizer_metadata.get("geometry_replay_protocol")
        != "mask_and_tracker_lost_only_v2"
        or tokenizer_metadata.get("geometry_replay_has_appearance_or_descriptors")
        is not False
        or tokenizer_metadata.get("forbidden_runtime_inputs")
        != ["simulator_state", "episode_ground_truth", "reward", "action", "future_rgb"]
    ):
        raise ValueError("Temporal-v2 tokenizer metadata is not exact and causally bound.")

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
        "temporal_graph_file_sha256",
        "temporal_graph_semantic_sha256",
        "python",
        "platform",
        "numpy",
        "device",
        "cuda_visible_devices",
        "wallclock_seconds",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("Temporal-v2 backend provenance schema changed.")
    project_root = Path(__file__).resolve().parents[2]
    expected_implementation_paths = {
        "tdmpc2/tools/replay_object_graph_temporal_v2.py": (
            project_root / "tdmpc2" / "tools" / "replay_object_graph_temporal_v2.py"
        ),
        "tdmpc2/perception/support_conditioned_object_graph.py": (
            project_root
            / "tdmpc2"
            / "perception"
            / "support_conditioned_object_graph.py"
        ),
        "tdmpc2/common/unified_vos.py": (
            project_root / "tdmpc2" / "common" / "unified_vos.py"
        ),
    }
    implementation = provenance.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != set(
        expected_implementation_paths
    ):
        raise ValueError("Temporal-v2 implementation snapshot is incomplete.")
    for relative, implementation_path in expected_implementation_paths.items():
        implementation_path = _regular_file(
            implementation_path, f"temporal-v2 implementation {relative}"
        )
        record = implementation[relative]
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"} or (
            record.get("bytes") != implementation_path.stat().st_size
            or record.get("sha256") != file_sha256(implementation_path)
        ):
            raise ValueError(f"Temporal-v2 implementation changed: {relative}.")
    worker_inputs_path = (dataset_root.parent / "worker_inputs" / "backend_inputs.json").resolve(
        strict=True
    )
    worker, support_paths, _ = validate_backend_inputs(
        worker_inputs_path, strict_counts=True
    )
    support_path = support_paths[TASK]
    if (
        provenance.get("treatment")
        != "ordered_chain_temporal_v2_mask_parser_replay"
        or provenance.get("source_v1_backend_manifest") != str(v1_manifest_path)
        or provenance.get("source_v1_backend_manifest_sha256")
        != file_sha256(v1_manifest_path)
        or provenance.get("source_worker_inputs") != str(worker_inputs_path)
        or provenance.get("source_worker_inputs_sha256")
        != file_sha256(worker_inputs_path)
        or provenance.get("source_worker_inputs_sha256")
        != dataset["backend_inputs"]["sha256"]
        or worker.get("dataset_id") != dataset.get("dataset_id")
        or provenance.get("source_support_arrays") != str(support_path)
        or provenance.get("source_support_arrays_sha256") != file_sha256(support_path)
        or provenance.get("v1_graph_file_sha256") != file_sha256(v1_graph.source_path)
        or provenance.get("v1_graph_semantic_sha256") != v1_graph.graph_sha256
        or provenance.get("temporal_graph_file_sha256")
        != file_sha256(temporal_graph_path)
        or provenance.get("temporal_graph_semantic_sha256")
        != temporal_graph.graph_sha256
        or provenance.get("python") != sys.version
        or provenance.get("platform") != platform.platform()
        or provenance.get("numpy") != np.__version__
        or provenance.get("device") != "cpu"
        or provenance.get("cuda_visible_devices") != ""
        or not isinstance(provenance.get("wallclock_seconds"), (int, float))
        or isinstance(provenance.get("wallclock_seconds"), bool)
        or not math.isfinite(float(provenance["wallclock_seconds"]))
        or float(provenance["wallclock_seconds"]) <= 0.0
    ):
        raise ValueError("Temporal-v2 execution provenance is malformed.")

    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(CONDITIONS):
        raise ValueError("Temporal-v2 replay condition set changed.")
    condition_map = results
    root = manifest_path.parent
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    resolution = int(dataset["resolution"])
    expected_shapes = _v2_shapes(
        frames=frames, roles=len(ROLES), resolution=resolution
    )
    paths: dict[tuple[str, int], Path] = {}
    for condition in CONDITIONS:
        records = condition_map[condition]
        if not isinstance(records, list) or len(records) != episodes:
            raise ValueError(f"Temporal-v2 episode count changed for {condition}.")
        for episode_index, record in enumerate(records):
            expected_record = {
                "episode_index",
                "frames",
                "entity_count",
                "role_count",
                "prediction_arrays",
                "prediction_arrays_sha256",
                "array_shapes",
                "traces",
                "source_v1_prediction_arrays_sha256",
                "source_v1_entity_mask_trace_sha256",
                "source_v1_entity_status_trace_sha256",
                "diagnostics_json",
                "diagnostics_json_sha256",
            }
            if not isinstance(record, dict) or set(record) != expected_record:
                raise ValueError("Temporal-v2 episode record schema changed.")
            v1_record = v1_payload["results"][TASK][condition][episode_index]
            v1_path = v1_paths[(TASK, condition, episode_index)]
            if (
                record.get("episode_index") != episode_index
                or record.get("frames") != frames
                or record.get("entity_count") != len(v1_graph.entities)
                or record.get("role_count") != len(ROLES)
                or record.get("array_shapes") != expected_shapes
                or record.get("source_v1_prediction_arrays_sha256")
                != v1_record.get("prediction_arrays_sha256")
                or record.get("source_v1_prediction_arrays_sha256")
                != file_sha256(v1_path)
                or record.get("source_v1_entity_mask_trace_sha256")
                != v1_record.get("traces", {}).get("entity_mask_trace_sha256")
                or record.get("source_v1_entity_status_trace_sha256")
                != v1_record.get("traces", {}).get("entity_status_trace_sha256")
            ):
                raise ValueError("Temporal-v2 episode is not paired to its exact v1 entity source.")
            traces = record.get("traces")
            if not isinstance(traces, dict) or set(traces) != V2_TRACE_KEYS:
                raise ValueError("Temporal-v2 episode trace schema changed.")
            for name, digest in traces.items():
                require_sha256(digest, f"temporal-v2 {condition}/{episode_index}/{name}")
            path = resolve_member(root, record.get("prediction_arrays"), "temporal-v2 arrays")
            if file_sha256(path) != require_sha256(
                record.get("prediction_arrays_sha256"), "temporal-v2 prediction SHA"
            ):
                raise ValueError("Temporal-v2 prediction artifact changed.")
            diagnostics_path = resolve_member(
                root, record.get("diagnostics_json"), "temporal-v2 diagnostics"
            )
            if file_sha256(diagnostics_path) != require_sha256(
                record.get("diagnostics_json_sha256"),
                "temporal-v2 diagnostics SHA",
            ):
                raise ValueError("Temporal-v2 diagnostics artifact changed.")
            diagnostics = load_json(diagnostics_path)
            if set(diagnostics) != {
                "format",
                "task",
                "condition",
                "episode_index",
                "frames",
                "records",
            } or (
                diagnostics.get("format")
                != "object_graph_temporal_v2_episode_diagnostics_v1"
                or diagnostics.get("task") != TASK
                or diagnostics.get("condition") != condition
                or diagnostics.get("episode_index") != episode_index
                or diagnostics.get("frames") != frames
                or not isinstance(diagnostics.get("records"), list)
                or len(diagnostics["records"]) != frames
                or diagnostics_path.read_bytes() != canonical_json_bytes(diagnostics)
            ):
                raise ValueError("Temporal-v2 decoded diagnostics schema changed.")
            if record["traces"]["diagnostics_trace_sha256"] != file_sha256(
                diagnostics_path
            ):
                raise ValueError("Temporal-v2 diagnostics trace is not the canonical JSON SHA.")
            source_arrays = _load_npz(v1_path, V1_ARRAY_KEYS)
            for frame_index, item in enumerate(diagnostics["records"]):
                if not isinstance(item, dict) or set(item) != {
                    "frame_index",
                    "source_entity_status",
                    "projector",
                }:
                    raise ValueError("Temporal-v2 per-frame diagnostics schema changed.")
                status = item["source_entity_status"]
                if (
                    item.get("frame_index") != frame_index
                    or not isinstance(status, dict)
                    or set(status) != {"valid", "confidence", "lost", "mask_score"}
                    or type(status.get("valid")) is not bool
                    or type(status.get("lost")) is not bool
                    or isinstance(status.get("confidence"), bool)
                    or not isinstance(status.get("confidence"), (int, float))
                    or not math.isfinite(float(status["confidence"]))
                    or not 0.0 <= float(status["confidence"]) <= 1.0
                    or isinstance(status.get("mask_score"), bool)
                    or not isinstance(status.get("mask_score"), (int, float))
                    or not math.isfinite(float(status["mask_score"]))
                    or not 0.0 <= float(status["mask_score"]) <= 1.0
                    or not isinstance(item.get("projector"), dict)
                    or set(item["projector"]) != {v1_graph.entity_names[0]}
                    or status["valid"]
                    is not bool(source_arrays["entity_valid"][frame_index, 0])
                    or status["lost"]
                    is not bool(source_arrays["entity_lost"][frame_index, 0])
                    or float(status.get("confidence"))
                    != float(source_arrays["entity_confidence"][frame_index, 0])
                    or float(status.get("mask_score"))
                    != float(source_arrays["entity_mask_score"][frame_index, 0])
                ):
                    raise ValueError("Temporal-v2 diagnostics/source entity pairing changed.")
            paths[(condition, episode_index)] = path
    return payload, paths


def _validate_replay_arrays(
    *,
    arrays: dict[str, np.ndarray],
    record: dict[str, Any],
    source_v1: dict[str, np.ndarray],
) -> None:
    for name, value in arrays.items():
        _dtype_contract(name, value)
        if list(value.shape) != record["array_shapes"][name]:
            raise ValueError(f"Temporal-v2 decoded shape changed for {name}.")
    decoded_traces = _decoded_traces(arrays)
    if any(record["traces"].get(name) != digest for name, digest in decoded_traces.items()):
        raise ValueError("Temporal-v2 decoded trace bundle changed.")
    if np.any(arrays["role_valid"] & arrays["role_lost"]):
        raise ValueError("A lost temporal-v2 role cannot be valid.")
    for field in ("projector_confidence", "role_confidence", "role_mask_score"):
        if np.any((arrays[field] < 0.0) | (arrays[field] > 1.0)):
            raise ValueError(f"Temporal-v2 {field} escaped [0,1].")
    if np.any(arrays["parser_runtime_ms"] <= 0.0):
        raise ValueError("Temporal-v2 parser runtime must be positive.")
    if np.any(arrays["role_masks"].sum(axis=1) > 1):
        raise ValueError("Temporal-v2 role masks overlap.")
    nonempty = arrays["role_masks"].reshape(
        *arrays["role_masks"].shape[:2], -1
    ).any(axis=-1)
    if not np.array_equal(arrays["projector_valid"], nonempty):
        raise ValueError("Temporal-v2 projector validity differs from mask availability.")
    if not np.array_equal(
        arrays["projector_valid"][:, 0], arrays["projector_valid"][:, 1]
    ):
        raise ValueError("The paired Acrobot roles have inconsistent projector validity.")
    if np.any(
        (~arrays["projector_valid"])
        & (
            (arrays["projector_confidence"] != 0.0)
            | np.any(arrays["keypoints_xy"] != 0.0, axis=(2, 3))
        )
    ):
        raise ValueError("Invalid temporal-v2 projector outputs are not fail-closed zeros.")
    source_lost = np.repeat(source_v1["entity_lost"], len(ROLES), axis=1)
    source_valid = np.repeat(source_v1["entity_valid"], len(ROLES), axis=1)
    source_nonempty = source_v1["entity_masks"].reshape(
        source_v1["entity_masks"].shape[0], source_v1["entity_masks"].shape[1], -1
    ).any(axis=-1)
    if not np.array_equal(
        source_v1["entity_valid"], (~source_v1["entity_lost"]) & source_nonempty
    ):
        raise ValueError("Published v1 entity validity is not mask/lost reproducible.")
    source_confidence = np.repeat(
        source_v1["entity_confidence"], len(ROLES), axis=1
    )
    source_mask_score = np.repeat(
        source_v1["entity_mask_score"], len(ROLES), axis=1
    )
    if not np.array_equal(arrays["role_lost"], source_lost):
        raise ValueError("Temporal-v2 role-lost trace differs from the replayed entity trace.")
    expected_valid = arrays["projector_valid"] & source_valid
    if not np.array_equal(arrays["role_valid"], expected_valid):
        raise ValueError("Temporal-v2 role validity is not projector-valid AND entity-valid.")
    expected_confidence = np.where(
        expected_valid,
        np.clip(source_confidence * arrays["projector_confidence"], 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    expected_mask_score = np.where(
        expected_valid,
        np.clip(source_mask_score * arrays["projector_confidence"], 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    if not np.array_equal(arrays["role_confidence"], expected_confidence):
        raise ValueError("Temporal-v2 role confidence is not exactly source×projector.")
    if not np.array_equal(arrays["role_mask_score"], expected_mask_score):
        raise ValueError("Temporal-v2 role mask score is not exactly source×projector.")
    valid_frames = arrays["projector_valid"].all(axis=1)
    projected_union = arrays["role_masks"].any(axis=1)
    if not np.array_equal(
        projected_union[valid_frames], source_v1["entity_masks"][valid_frames, 0]
    ):
        raise ValueError("Valid temporal-v2 role masks do not partition the v1 entity mask.")


def _score_real_replay(
    *,
    v2_payload: dict[str, Any],
    v2_paths: dict[tuple[str, int], Path],
    v1_paths: dict[tuple[str, str, int], Path],
    dataset: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    resolution = int(dataset["resolution"])
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        scorer = MaskScorer(ROLES, resolution)
        valid_count = np.zeros(len(ROLES), dtype=np.int64)
        invalid_max = np.zeros(len(ROLES), dtype=np.int64)
        projector_valid_count = np.zeros(len(ROLES), dtype=np.int64)
        projector_invalid_max = np.zeros(len(ROLES), dtype=np.int64)
        lost_count = np.zeros(len(ROLES), dtype=np.int64)
        confidence: list[np.ndarray] = []
        mask_score: list[np.ndarray] = []
        decoded_episode_traces = []
        for episode_index in range(episodes):
            record = v2_payload["results"][condition][episode_index]
            arrays = _load_npz(v2_paths[(condition, episode_index)], V2_ARRAY_KEYS)
            source_v1 = _load_npz(
                v1_paths[(TASK, condition, episode_index)],
                {
                    "role_masks",
                    "entity_masks",
                    "descriptors",
                    "keypoints_xy",
                    "role_valid",
                    "role_confidence",
                    "role_lost",
                    "role_mask_score",
                    "entity_valid",
                    "entity_confidence",
                    "entity_lost",
                    "entity_mask_score",
                    "cutie_runtime_ms",
                    "parser_runtime_ms",
                    "end_to_end_runtime_ms",
                },
            )
            _validate_replay_arrays(
                arrays=arrays, record=record, source_v1=source_v1
            )
            gt_path = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "temporal-v2 scoring arrays",
            )
            gt = _load_npz(
                gt_path, {"gt_indexed", "actions", "physics_states"}
            )["gt_indexed"]
            if gt.shape != (frames, resolution, resolution) or gt.dtype != np.uint8:
                raise ValueError("Temporal-v2 frozen GT schema changed.")
            scorer.begin_episode()
            for frame_index in range(frames):
                scorer.record(
                    predicted=arrays["role_masks"][frame_index],
                    gt_indexed=gt[frame_index],
                    runtime_ms=float(arrays["parser_runtime_ms"][frame_index]),
                )
            valid_count += arrays["role_valid"].sum(axis=0)
            projector_valid_count += arrays["projector_valid"].sum(axis=0)
            lost_count += arrays["role_lost"].sum(axis=0)
            _, burst = _invalid_bursts(arrays["role_valid"])
            invalid_max = np.maximum(invalid_max, burst)
            _, projector_burst = _invalid_bursts(arrays["projector_valid"])
            projector_invalid_max = np.maximum(
                projector_invalid_max, projector_burst
            )
            confidence.append(arrays["role_confidence"])
            mask_score.append(arrays["role_mask_score"])
            decoded_episode_traces.append(
                {
                    "episode_index": episode_index,
                    **_decoded_traces(arrays),
                }
            )
        summary = scorer.summary()
        confidence_array = np.concatenate(confidence, axis=0)
        mask_score_array = np.concatenate(mask_score, axis=0)
        summary["role_validity"] = {
            role: {
                "valid_rate": float(valid_count[index]) / (episodes * frames),
                "max_invalid_burst": int(invalid_max[index]),
                "reported_lost_rate": float(lost_count[index]) / (episodes * frames),
                "mean_confidence": float(confidence_array[:, index].mean()),
                "mean_mask_score": float(mask_score_array[:, index].mean()),
            }
            for index, role in enumerate(ROLES)
        }
        summary["projector_validity"] = {
            role: {
                "valid_rate": float(projector_valid_count[index])
                / (episodes * frames),
                "max_invalid_burst": int(projector_invalid_max[index]),
            }
            for index, role in enumerate(ROLES)
        }
        summary["decoded_episode_trace_bundle_sha256"] = sha256_json(
            decoded_episode_traces
        )
        result[condition] = summary
    return result


def _projected_arrays(projected: Any, *, resolution: int) -> dict[str, np.ndarray]:
    arrays = {
        "role_masks": np.ascontiguousarray(projected.masks),
        "keypoints_xy": np.ascontiguousarray(projected.keypoints_xy),
        "projector_valid": np.ascontiguousarray(projected.valid),
        "projector_confidence": np.ascontiguousarray(
            projected.projector_confidence
        ),
        "parser_runtime_ms": np.asarray(float(projected.runtime_ms), dtype=np.float64),
    }
    expected = {
        "role_masks": ((len(ROLES), resolution, resolution), np.bool_),
        "keypoints_xy": ((len(ROLES), 2, 2), np.float32),
        "projector_valid": ((len(ROLES),), np.bool_),
        "projector_confidence": ((len(ROLES),), np.float32),
        "parser_runtime_ms": ((), np.float64),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Temporal-v2 oracle output schema changed for {name}.")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(arrays[name]).all():
            raise ValueError(f"Temporal-v2 oracle output is non-finite for {name}.")
    if float(arrays["parser_runtime_ms"]) <= 0.0:
        raise ValueError("Temporal-v2 oracle runtime must be positive.")
    if tuple(projected.role_names) != ROLES or not isinstance(projected.diagnostics, dict):
        raise ValueError("Temporal-v2 oracle role order/diagnostics changed.")
    return arrays


def _score_gt_union_oracle(
    *,
    temporal_graph: Any,
    dataset: dict[str, Any],
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the same temporal API on privileged exact GT entity unions."""
    resolution = int(dataset["resolution"])
    if resolution != 128:
        raise ValueError("Temporal-v2 oracle is bound to the frozen 128px dataset.")
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    support_path = resolve_member(
        dataset_root, dataset["support"][TASK]["arrays"], "temporal-v2 oracle support"
    )
    support = _load_npz(support_path, {"rgb", "indexed_masks"})
    tokenizer = SupportConditionedObjectGraphTokenizer(
        temporal_graph, support["indexed_masks"]
    )
    result: dict[str, Any] = {}
    trace_result: dict[str, Any] = {}
    entity_count = len(temporal_graph.entities)
    for condition in CONDITIONS:
        scorer = MaskScorer(ROLES, resolution)
        valid_count = np.zeros(len(ROLES), dtype=np.int64)
        invalid_max = np.zeros(len(ROLES), dtype=np.int64)
        projector_valid_count = np.zeros(len(ROLES), dtype=np.int64)
        projector_invalid_max = np.zeros(len(ROLES), dtype=np.int64)
        episode_trace_rows = []
        for episode_index in range(episodes):
            gt_path = resolve_member(
                dataset_root,
                dataset["episodes"][TASK][condition][episode_index]["arrays"],
                "temporal-v2 oracle scoring arrays",
            )
            gt = _load_npz(
                gt_path, {"gt_indexed", "actions", "physics_states"}
            )["gt_indexed"]
            tokenizer.reset_episode()
            scorer.begin_episode()
            episode = {
                "role_masks": np.zeros(
                    (frames, len(ROLES), resolution, resolution), dtype=np.bool_
                ),
                "keypoints_xy": np.zeros(
                    (frames, len(ROLES), 2, 2), dtype=np.float32
                ),
                "projector_valid": np.zeros(
                    (frames, len(ROLES)), dtype=np.bool_
                ),
                "projector_confidence": np.zeros(
                    (frames, len(ROLES)), dtype=np.float32
                ),
                "role_valid": np.zeros((frames, len(ROLES)), dtype=np.bool_),
                "role_confidence": np.zeros(
                    (frames, len(ROLES)), dtype=np.float32
                ),
                "role_lost": np.zeros((frames, len(ROLES)), dtype=np.bool_),
                "role_mask_score": np.zeros(
                    (frames, len(ROLES)), dtype=np.float32
                ),
                "parser_runtime_ms": np.zeros(frames, dtype=np.float64),
            }
            diagnostics_rows = []
            for frame_index in range(frames):
                entity_gt = _entity_gt(
                    gt[frame_index], temporal_graph.role_to_entity_index
                )
                entity_masks = np.stack(
                    [
                        entity_gt == entity_index + 1
                        for entity_index in range(entity_count)
                    ]
                )
                entity_lost = ~entity_masks.reshape(entity_count, -1).any(axis=1)
                projected = tokenizer.project_temporal_geometry(
                    entity_masks=entity_masks,
                    entity_lost=np.ascontiguousarray(entity_lost, dtype=np.bool_),
                )
                current = _projected_arrays(projected, resolution=resolution)
                role_lost = np.asarray(
                    [entity_lost[index] for index in temporal_graph.role_to_entity_index],
                    dtype=np.bool_,
                )
                current.update(
                    {
                        "role_valid": current["projector_valid"].copy(),
                        "role_confidence": current[
                            "projector_confidence"
                        ].copy(),
                        "role_lost": role_lost,
                        "role_mask_score": current[
                            "projector_confidence"
                        ].copy(),
                    }
                )
                for name in V2_ARRAY_KEYS:
                    episode[name][frame_index] = current[name]
                diagnostics_rows.append(projected.diagnostics)
                scorer.record(
                    predicted=current["role_masks"],
                    gt_indexed=gt[frame_index],
                    runtime_ms=float(current["parser_runtime_ms"]),
                )
            if np.any(episode["role_valid"] & episode["role_lost"]):
                raise ValueError("Temporal-v2 oracle marked a lost role valid.")
            if np.any(episode["role_masks"].sum(axis=1) > 1):
                raise ValueError("Temporal-v2 oracle role masks overlap.")
            oracle_nonempty = episode["role_masks"].reshape(
                frames, len(ROLES), -1
            ).any(axis=-1)
            if (
                not np.array_equal(episode["projector_valid"], oracle_nonempty)
                or not np.array_equal(
                    episode["role_valid"], episode["projector_valid"]
                )
                or not np.array_equal(
                    episode["role_confidence"],
                    episode["projector_confidence"],
                )
                or not np.array_equal(
                    episode["role_mask_score"],
                    episode["projector_confidence"],
                )
            ):
                raise ValueError("Temporal-v2 oracle status derivation changed.")
            invalid = ~episode["projector_valid"]
            if np.any(
                invalid
                & (
                    (episode["projector_confidence"] != 0.0)
                    | np.any(episode["keypoints_xy"] != 0.0, axis=(2, 3))
                )
            ):
                raise ValueError("Temporal-v2 oracle invalid outputs are not zeroed.")
            oracle_union = episode["role_masks"].any(axis=1)
            valid_frames = episode["projector_valid"].all(axis=1)
            gt_entity_union = gt != 0
            if not np.array_equal(
                oracle_union[valid_frames], gt_entity_union[valid_frames]
            ):
                raise ValueError("Temporal-v2 oracle roles do not partition the GT union.")
            valid_count += episode["role_valid"].sum(axis=0)
            projector_valid_count += episode["projector_valid"].sum(axis=0)
            _, burst = _invalid_bursts(episode["role_valid"])
            invalid_max = np.maximum(invalid_max, burst)
            _, projector_burst = _invalid_bursts(episode["projector_valid"])
            projector_invalid_max = np.maximum(
                projector_invalid_max, projector_burst
            )
            episode_trace_rows.append(
                {
                    "episode_index": episode_index,
                    **_decoded_traces(episode),
                    "diagnostics_sha256": sha256_json(diagnostics_rows),
                }
            )
        summary = scorer.summary()
        summary["role_validity"] = {
            role: {
                "valid_rate": float(valid_count[index]) / (episodes * frames),
                "max_invalid_burst": int(invalid_max[index]),
            }
            for index, role in enumerate(ROLES)
        }
        summary["projector_validity"] = {
            role: {
                "valid_rate": float(projector_valid_count[index])
                / (episodes * frames),
                "max_invalid_burst": int(projector_invalid_max[index]),
            }
            for index, role in enumerate(ROLES)
        }
        result[condition] = summary
        trace_result[condition] = {
            "episode_trace_bundle_sha256": sha256_json(episode_trace_rows),
            "episodes": episodes,
            "frames_per_episode": frames,
        }
    return result, trace_result


def _development_gate(
    *,
    v1_semantic: dict[str, Any],
    v1_oracle: dict[str, Any],
    real: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for arm_name, arm, reference_arm in (
        ("real_entity_replay", real, v1_semantic),
        ("gt_union_oracle_replay", oracle, v1_oracle),
    ):
        for condition in CONDITIONS:
            candidate = arm[condition]
            reference = reference_arm[condition]
            checks[f"{arm_name}/{condition}/runtime_mean"] = (
                float(candidate["latency"]["mean_ms"]) <= MAX_PARSER_MEAN_MS
            )
            checks[f"{arm_name}/{condition}/runtime_p95"] = (
                float(candidate["latency"]["p95_ms"]) <= MAX_PARSER_P95_MS
            )
            checks[f"{arm_name}/{condition}/swaps"] = (
                candidate["role_swap_frames"] == 0
            )
            for role in ROLES:
                role_metrics = candidate["per_role"][role]
                reference_metrics = reference["per_role"][role]
                checks[f"{arm_name}/{condition}/{role}/failure_burst"] = (
                    int(role_metrics["max_failure_burst_at_iou_0_5_or_identity"])
                    <= MAX_FAILURE_BURST
                )
                checks[f"{arm_name}/{condition}/{role}/invalid_burst"] = (
                    int(candidate["projector_validity"][role]["max_invalid_burst"])
                    <= MAX_INVALID_BURST
                )
                for metric in QUALITY_METRICS:
                    checks[
                        f"{arm_name}/{condition}/{role}/no_regression/{metric}"
                    ] = (
                        _metric(role_metrics, metric)
                        >= _metric(reference_metrics, metric)
                        - QUALITY_REGRESSION_TOLERANCE
                    )
    return {"checks": checks, "pass": bool(checks) and all(checks.values())}


def _published_v1_oracle_baseline(
    v1_summary: dict[str, Any], *, expected_frames: int
) -> dict[str, Any]:
    root = v1_summary.get("gt_union_parser_oracle")
    if not isinstance(root, dict) or set(root) != {TASK}:
        raise ValueError("Published v1 GT-union oracle task schema changed.")
    cells = root[TASK]
    if not isinstance(cells, dict) or set(cells) != set(CONDITIONS):
        raise ValueError("Published v1 GT-union oracle condition schema changed.")
    for condition in CONDITIONS:
        cell = cells[condition]
        if (
            not isinstance(cell, dict)
            or cell.get("frames") != expected_frames
            or set(cell.get("per_role", {})) != set(ROLES)
            or set(cell.get("parser_validity", {})) != set(ROLES)
            or type(cell.get("role_swap_frames")) is not int
            or cell["role_swap_frames"] < 0
        ):
            raise ValueError("Published v1 GT-union oracle cell schema changed.")
        for role in ROLES:
            metrics = cell["per_role"][role]
            validity = cell["parser_validity"][role]
            if not isinstance(metrics, dict) or any(
                name not in metrics for name in QUALITY_METRICS
            ):
                raise ValueError("Published v1 oracle role metrics are incomplete.")
            for name in QUALITY_METRICS:
                value = metrics[name]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError("Published v1 oracle quality metric is malformed.")
            if (
                not isinstance(validity, dict)
                or set(validity) != {"valid_rate", "max_invalid_burst"}
                or isinstance(validity.get("valid_rate"), bool)
                or not isinstance(validity.get("valid_rate"), (int, float))
                or not math.isfinite(float(validity["valid_rate"]))
                or not 0.0 <= float(validity["valid_rate"]) <= 1.0
                or type(validity.get("max_invalid_burst")) is not int
                or not 0 <= validity["max_invalid_burst"]
            ):
                raise ValueError("Published v1 oracle validity metric is malformed.")
    return cells


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    (
        source_summary,
        dataset,
        dataset_path,
        cutie_manifest_path,
        _cutie_payload,
        cutie_paths,
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
    immutable_inputs = _validate_immutable_inputs(
        args.immutable_inputs,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        summary_root=args.output.parent,
        v1_graph_path=args.v1_graph,
        temporal_graph_path=args.graph,
    )
    temporal_graph = _load_temporal_graph(args.graph, v1_graphs[TASK])
    isolation = _validate_isolation_gate(
        args.isolation_gate,
        source_root=args.source_benchmark_root,
        v1_preflight_root=args.v1_preflight_root,
        summary_root=args.output.parent,
        v2_manifest_path=args.v2_replay_backend,
    )
    v2_payload, v2_paths = _validate_v2_manifest(
        args.v2_replay_backend,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        v1_manifest_path=v1_manifest_path,
        v1_payload=v1_payload,
        v1_paths=v1_paths,
        v1_graph=v1_graphs[TASK],
        temporal_graph_path=args.graph,
        temporal_graph=temporal_graph,
    )
    v1_metrics, direct_parity = _score_graph_backend(
        payload=v1_payload,
        paths=v1_paths,
        graphs=v1_graphs,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        baseline_paths=cutie_paths,
    )
    if not direct_parity or not all(
        value for cell in direct_parity.values() for value in cell.values()
    ):
        raise ValueError("Published v1 direct-control parity no longer holds.")
    if (
        v1_summary.get("object_graph_metrics") != v1_metrics
        or v1_summary.get("direct_parity") != direct_parity
    ):
        raise ValueError("Recomputed v1 metrics/parity differ from the publication.")
    real_metrics = _score_real_replay(
        v2_payload=v2_payload,
        v2_paths=v2_paths,
        v1_paths=v1_paths,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    oracle_metrics, oracle_traces = _score_gt_union_oracle(
        temporal_graph=temporal_graph,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    v1_semantic = {
        condition: v1_metrics[TASK][condition]["semantic_roles"]
        for condition in CONDITIONS
    }
    v1_entities = {
        condition: v1_metrics[TASK][condition]["tracking_entities"]
        for condition in CONDITIONS
    }
    published_v1_oracle = _published_v1_oracle_baseline(
        v1_summary,
        expected_frames=(
            int(dataset["counts"]["episodes"])
            * int(dataset["counts"]["frames_per_episode"])
        ),
    )
    recomputed_v1_oracle = _score_v1_gt_union_parser_oracle(
        graphs=v1_graphs,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    published_deterministic = json.loads(json.dumps(published_v1_oracle))
    recomputed_deterministic = json.loads(
        json.dumps(recomputed_v1_oracle.get(TASK, {}))
    )
    for condition in CONDITIONS:
        if isinstance(published_deterministic.get(condition), dict):
            published_deterministic[condition].pop("latency", None)
        if isinstance(recomputed_deterministic.get(condition), dict):
            recomputed_deterministic[condition].pop("latency", None)
    if (
        not isinstance(recomputed_v1_oracle, dict)
        or set(recomputed_v1_oracle) != {TASK}
        or recomputed_deterministic != published_deterministic
    ):
        raise ValueError(
            "Recomputed v1 GT-union oracle quality/status differs from publication."
        )
    v1_oracle = recomputed_v1_oracle[TASK]
    gate = _development_gate(
        v1_semantic=v1_semantic,
        v1_oracle=v1_oracle,
        real=real_metrics,
        oracle=oracle_metrics,
    )
    development_candidate = bool(gate["pass"])
    summary = {
        "format": SUMMARY_FORMAT,
        "status": (
            "object_graph_temporal_v2_development_candidate"
            if development_candidate
            else "object_graph_temporal_v2_development_no_go"
        ),
        "engineering_pass": True,
        "development_candidate": development_candidate,
        "controller_training_authorized": False,
        "scientific_go": False,
        "recommendation": (
            "freeze_unseen_trajectory_seeds_and_run_confirmatory_parser_preflight"
            if development_candidate
            else "do_not_train_controller_temporal_v2_parser_did_not_clear_development_gate"
        ),
        "scope": {
            "evidence_level": "post_hoc_offline_parser_replay_on_previously_inspected_frozen_trajectories",
            "task": TASK,
            "same_frozen_rgb_as_v1": True,
            "real_arm_uses_only_published_v1_entity_masks_and_status": True,
            "gt_union_oracle_is_offline_privileged_parser_diagnostic": True,
            "v1_oracle_revalidation_exact_for_quality_status_not_wallclock": True,
            "episode_ground_truth_backend_access": False,
            "episode_ground_truth_aggregator_access_after_backend": True,
            "support_rgb_schema_validated_not_used": True,
            "episode_rgb_bytes_hashed_for_input_validation": True,
            "episode_rgb_parser_input": False,
            "episode_rgb_decoded": False,
            "episode_entity_arrays_loaded_before_replay": True,
            "future_entity_frames_parser_input": False,
            "raw_appearance_features_consumed_by_temporal_replay": False,
            "mask_only_api_used": True,
            "appearance_feature_sentinel_used": False,
            "controller_descriptors_constructed": False,
            "controller_constructed": False,
            "controller_training_steps": 0,
            "allowed_claim": "temporal_v2_offline_role_parser_development_signal",
            "disallowed_claims": [
                "controller_advantage",
                "deployable_controller_representation",
                "confirmatory_perception_result",
                "paper_level_generality",
            ],
        },
        "thresholds": {
            "max_failure_burst": MAX_FAILURE_BURST,
            "max_invalid_burst": MAX_INVALID_BURST,
            "max_parser_mean_ms": MAX_PARSER_MEAN_MS,
            "max_parser_p95_ms": MAX_PARSER_P95_MS,
            "maximum_role_swaps": 0,
            "quality_regression_tolerance": QUALITY_REGRESSION_TOLERANCE,
            "quality_non_regression_metrics": list(QUALITY_METRICS),
            "applied_to_arms": [
                "real_entity_replay",
                "gt_union_oracle_replay",
            ],
            "serialized_in_gate_checks": True,
        },
        "gate": gate,
        "v1_entity_baseline_metrics": v1_entities,
        "v1_semantic_role_baseline_metrics": v1_semantic,
        "v1_gt_union_parser_oracle_baseline_metrics": published_v1_oracle,
        "v1_gt_union_parser_oracle_recomputed_quality_status_match": True,
        "temporal_v2_real_entity_replay": real_metrics,
        "temporal_v2_gt_union_oracle_replay": oracle_metrics,
        "temporal_v2_gt_union_oracle_traces": oracle_traces,
        "v1_direct_control_parity": direct_parity,
        "graphs": {
            "v1": v1_graphs[TASK].metadata(),
            "temporal_v2": temporal_graph.metadata(),
        },
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_benchmark_root": str(args.source_benchmark_root),
            "source_summary_sha256": file_sha256(
                args.source_benchmark_root / "unified_vos_summary.json"
            ),
            "source_dataset_id": dataset["dataset_id"],
            "source_dataset_manifest_sha256": file_sha256(dataset_path),
            "source_cutie_manifest_sha256": file_sha256(cutie_manifest_path),
            "v1_preflight_root": str(args.v1_preflight_root),
            "v1_preflight_summary_sha256": file_sha256(
                args.v1_preflight_root / "object_graph_tokenizer_summary.json"
            ),
            "v1_object_graph_backend_manifest": str(v1_manifest_path),
            "v1_object_graph_backend_manifest_sha256": file_sha256(
                v1_manifest_path
            ),
            "temporal_v2_replay_backend_manifest_relative_to_summary_root": (
                args.v2_replay_backend.relative_to(args.output.parent).as_posix()
            ),
            "temporal_v2_replay_backend_manifest_sha256": file_sha256(
                args.v2_replay_backend
            ),
            "scoring_isolation_relative_to_summary_root": (
                args.isolation_gate.relative_to(args.output.parent).as_posix()
            ),
            "scoring_isolation_sha256": file_sha256(args.isolation_gate),
            "immutable_inputs_relative_to_summary_root": (
                args.immutable_inputs.relative_to(args.output.parent).as_posix()
            ),
            "immutable_inputs_sha256": file_sha256(args.immutable_inputs),
            "immutable_inputs_format": immutable_inputs["format"],
            "v1_graph_file_sha256": v1_payload["graphs"][TASK][
                "graph_file_sha256"
            ],
            "v1_graph_sha256": v1_graphs[TASK].graph_sha256,
            "v1_graph": str(args.v1_graph),
            "temporal_v2_graph": str(args.graph),
            "temporal_v2_graph_file_sha256": file_sha256(args.graph),
            "temporal_v2_graph_sha256": temporal_graph.graph_sha256,
            "source_status": source_summary.get("status"),
            "scoring_isolation_status": isolation["status"],
        },
    }
    write_json(args.output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--v1-preflight-root", type=Path, required=True)
    parser.add_argument("--v2-replay-backend", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
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
        ("v2_replay_backend", "temporal-v2 replay manifest"),
        ("v1_graph", "published v1 graph"),
        ("graph", "temporal-v2 graph"),
        ("isolation_gate", "temporal replay isolation gate"),
        ("immutable_inputs", "temporal replay immutable-input snapshot"),
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
