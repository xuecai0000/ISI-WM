"""Strict development preflight for Cutie entity-to-role object graphs.

This scorer reuses one already-completed unified-VOS frozen dataset.  It
compares the historical separate-role Cutie baseline with a new grouped-entity
Cutie run, and deliberately separates three questions:

* did Cutie track each declared visual entity;
* did the support-conditioned projector recover the semantic roles;
* can the same projector recover roles when given an offline GT entity union.

The last arm is a privileged parser diagnostic.  This script never authorizes
controller training: the source trajectories were already inspected while the
method was designed, so a fresh confirmatory preflight is required first.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np

from tdmpc2.common.unified_vos import (
    CONDITIONS,
    TASK_ROLES,
    file_sha256,
    load_json,
    require_sha256,
    resolve_member,
    validate_backend_inputs,
    validate_dataset_files,
    write_json,
)
from tdmpc2.perception.support_conditioned_object_graph import (
    FRAME_DIM,
    QUERY_FEATURE_DIM,
    SupportConditionedObjectGraphTokenizer,
    load_object_graph,
)
from tdmpc2.tools.aggregate_unified_vos_benchmark import (
    MaskScorer,
    _array_trace,
    _load_npz,
    _score_backend,
    _validate_backend_manifest,
    _validate_decoded_dataset_artifacts,
)


BACKEND_FORMAT = "object_graph_cutie_backend_predictions_v1"
SUMMARY_FORMAT = "object_graph_tokenizer_preflight_summary_v1"
BACKEND_NAME = "object_graph_cutie"
ISOLATION_FORMAT = "object_graph_scoring_isolation_gate_v1"
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MODE = re.compile(r"[0-7]{3}")
GRAPH_FILES = {
    "reacher-visual-small": "reacher_visual_small.json",
    "cartpole-swingup": "cartpole_swingup.json",
    "acrobot-swingup": "acrobot_swingup.json",
}
ARRAY_KEYS = {
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
}
TRACE_KEYS = {
    "role_mask_trace_sha256",
    "entity_mask_trace_sha256",
    "descriptor_trace_sha256",
    "keypoint_trace_sha256",
    "role_status_trace_sha256",
    "entity_status_trace_sha256",
    "cutie_runtime_trace_sha256",
    "parser_runtime_trace_sha256",
    "end_to_end_runtime_trace_sha256",
}

MAX_DIRECT_METRIC_REGRESSION = 0.02
MAX_PARSER_MEAN_MS = 5.0
MAX_PARSER_P95_MS = 10.0
MAX_TOTAL_RATIO_VS_CUTIE = 1.75
MAX_TOTAL_DELTA_MS_VS_CUTIE = 5.0
EXPECTED_BACKEND_SEED = 2718281
MIN_ROLE_VALID_RATE = 0.90
MAX_ROLE_INVALID_BURST = 25
MIN_ORACLE_VALID_RATE = 0.95
MAX_ORACLE_INVALID_BURST = 10


def _graph_paths(graph_dir: Path) -> dict[str, Path]:
    graph_dir = graph_dir.expanduser().resolve()
    if not graph_dir.is_dir():
        raise FileNotFoundError(graph_dir)
    result = {}
    for task, filename in GRAPH_FILES.items():
        path = (graph_dir / filename).resolve()
        try:
            path.relative_to(graph_dir)
        except ValueError as exc:
            raise ValueError("Graph file escapes graph-dir.") from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        graph = load_object_graph(path)
        if graph.task != task or graph.source_roles != TASK_ROLES[task]:
            raise ValueError(f"Graph task/role contract changed for {task}.")
        result[task] = path
    return result


def _source_artifacts(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    summary_path = source_root / "unified_vos_summary.json"
    dataset_path = source_root / "dataset" / "dataset_manifest.json"
    baseline_path = source_root / "backends" / "cutie" / "backend_predictions.json"
    for path in (summary_path, dataset_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_summary = load_json(summary_path)
    if source_summary.get("engineering_pass") is not True:
        raise ValueError("Source unified-VOS benchmark did not pass engineering gates.")
    provenance = source_summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Source summary provenance is missing.")
    dataset = validate_dataset_files(dataset_path, strict_counts=True)
    _validate_decoded_dataset_artifacts(dataset, dataset_path.parent)
    if provenance.get("dataset_id") != dataset.get("dataset_id"):
        raise ValueError("Source summary/dataset identity mismatch.")
    if provenance.get("dataset_manifest_sha256") != file_sha256(dataset_path):
        raise ValueError("Source dataset manifest differs from its published summary.")
    cutie_record = provenance.get("backend_manifests", {}).get("cutie")
    if (
        not isinstance(cutie_record, dict)
        or cutie_record.get("sha256") != file_sha256(baseline_path)
    ):
        raise ValueError("Source Cutie baseline differs from its published summary.")
    backend_inputs_path = resolve_member(
        dataset_path.parent,
        dataset.get("backend_inputs", {}).get("path"),
        "source GT-free backend inputs",
    )
    if file_sha256(backend_inputs_path) != dataset["backend_inputs"]["sha256"]:
        raise ValueError("Source GT-free backend inputs changed.")
    worker_inputs_path = source_root / "worker_inputs" / "backend_inputs.json"
    worker_inputs, _, _ = validate_backend_inputs(
        worker_inputs_path, strict_counts=True
    )
    if (
        file_sha256(worker_inputs_path) != dataset["backend_inputs"]["sha256"]
        or worker_inputs.get("dataset_id") != dataset.get("dataset_id")
    ):
        raise ValueError("Source worker-view inputs differ from the dataset publication.")
    return source_summary, dataset, dataset_path, baseline_path


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):03o}"


def _validate_isolation_gate(
    path: Path,
    *,
    source_root: Path,
    summary_root: Path,
    backend_manifest: Path,
) -> dict[str, Any]:
    payload = load_json(path)
    expected = {
        "format",
        "status",
        "source_benchmark_root",
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
        "backend_manifest_sha256",
        "same_uid_read_probe",
    }
    if set(payload) != expected or payload.get("format") != ISOLATION_FORMAT:
        raise ValueError("Object-graph scoring-isolation schema changed.")
    if payload.get("status") != "complete" or payload.get(
        "backend_completed_before_restore"
    ) is not True:
        raise ValueError("Object-graph scoring isolation did not complete.")
    source_root = source_root.resolve()
    worker_root = (source_root / "worker_inputs").resolve(strict=True)
    scoring_root = (source_root / "dataset" / "scoring").resolve(strict=True)
    if (
        payload.get("source_benchmark_root") != str(source_root)
        or payload.get("worker_input_root") != str(worker_root)
        or payload.get("scoring_root") != str(scoring_root)
        or worker_root == scoring_root
        or worker_root in scoring_root.parents
        or scoring_root in worker_root.parents
    ):
        raise ValueError("Object-graph worker/scoring roots are not strictly disjoint.")
    relative = payload.get("scoring_probe_relative_to_source")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Object-graph scoring probe path is malformed.")
    probe = (source_root / relative).resolve(strict=True)
    try:
        probe.relative_to(scoring_root)
    except ValueError as exc:
        raise ValueError("Object-graph probe is outside the scoring tree.") from exc
    if not probe.is_file():
        raise FileNotFoundError(probe)
    for name in ("root_mode_before", "root_mode_locked", "root_mode_restored"):
        if not isinstance(payload.get(name), str) or _MODE.fullmatch(payload[name]) is None:
            raise ValueError(f"Object-graph isolation {name} is malformed.")
    if (
        payload["root_mode_locked"] != "000"
        or payload["root_mode_before"] != payload["root_mode_restored"]
        or _mode(scoring_root) != payload["root_mode_restored"]
    ):
        raise ValueError("Object-graph scoring permission transition is invalid.")
    before = require_sha256(payload.get("probe_sha256_before"), "isolation probe SHA")
    restored = require_sha256(
        payload.get("probe_sha256_restored"), "isolation restored probe SHA"
    )
    if before != restored or file_sha256(probe) != restored:
        raise ValueError("Object-graph scoring probe bytes changed.")
    timestamps = [
        payload.get("locked_utc"),
        payload.get("backend_completed_utc"),
        payload.get("restored_utc"),
    ]
    if any(not isinstance(value, str) or _UTC.fullmatch(value) is None for value in timestamps):
        raise ValueError("Object-graph isolation timestamps are malformed.")
    if timestamps != sorted(timestamps):
        raise ValueError("Object-graph isolation event order is impossible.")
    if payload.get("backend_manifest_sha256") != file_sha256(backend_manifest):
        raise ValueError("Isolation gate is bound to a different backend artifact.")
    read_probe = payload.get("same_uid_read_probe")
    if not isinstance(read_probe, dict) or set(read_probe) != {
        "exit_code", "error_type", "log_relative_to_summary_root", "log_sha256"
    }:
        raise ValueError("Object-graph same-UID read-probe schema changed.")
    if read_probe.get("exit_code") != 1 or read_probe.get("error_type") != "PermissionError":
        raise ValueError("Object-graph same-UID GT read did not fail closed.")
    log_relative = read_probe.get("log_relative_to_summary_root")
    if not isinstance(log_relative, str) or not log_relative or "\\" in log_relative:
        raise ValueError("Object-graph isolation log path is malformed.")
    log_path = (summary_root.resolve() / log_relative).resolve(strict=True)
    try:
        log_path.relative_to(summary_root.resolve())
    except ValueError as exc:
        raise ValueError("Object-graph isolation log escapes the output root.") from exc
    if file_sha256(log_path) != require_sha256(
        read_probe.get("log_sha256"), "isolation read-probe log SHA"
    ):
        raise ValueError("Object-graph isolation log changed.")
    if "PermissionError" not in log_path.read_text(encoding="utf-8", errors="replace"):
        raise ValueError("Object-graph isolation log lacks PermissionError evidence.")
    return payload


def _expected_shapes(
    *, frames: int, roles: int, entities: int, resolution: int
) -> dict[str, list[int]]:
    return {
        "role_masks": [frames, roles, resolution, resolution],
        "entity_masks": [frames, entities, resolution, resolution],
        "descriptors": [frames, roles, FRAME_DIM],
        "keypoints_xy": [frames, roles, 2, 2],
        "role_valid": [frames, roles],
        "role_confidence": [frames, roles],
        "role_lost": [frames, roles],
        "role_mask_score": [frames, roles],
        "entity_valid": [frames, entities],
        "entity_confidence": [frames, entities],
        "entity_lost": [frames, entities],
        "entity_mask_score": [frames, entities],
        "cutie_runtime_ms": [frames],
        "parser_runtime_ms": [frames],
        "end_to_end_runtime_ms": [frames],
    }


def _validate_graph_backend(
    manifest_path: Path,
    *,
    dataset: dict[str, Any],
    graph_paths: dict[str, Path],
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, int], Path],
    dict[str, Any],
]:
    manifest_path = manifest_path.expanduser().resolve()
    payload = load_json(manifest_path)
    expected_top = {
        "format",
        "status",
        "backend",
        "dataset_id",
        "input_manifest_sha256",
        "roles",
        "graphs",
        "protocol",
        "backend_provenance",
        "results",
    }
    if set(payload) != expected_top:
        raise ValueError("Object-graph backend manifest schema changed.")
    if (
        payload.get("format") != BACKEND_FORMAT
        or payload.get("status") != "complete"
        or payload.get("backend") != BACKEND_NAME
        or payload.get("dataset_id") != dataset.get("dataset_id")
        or payload.get("input_manifest_sha256")
        != dataset.get("backend_inputs", {}).get("sha256")
    ):
        raise ValueError("Object-graph backend identity/status mismatch.")
    if payload.get("roles") != {
        task: list(roles) for task, roles in TASK_ROLES.items()
    }:
        raise ValueError("Object-graph semantic role order changed.")
    protocol = payload.get("protocol")
    expected_protocol = {
        "base_backend_format": "unified_vos_backend_predictions_v1",
        "support_only_prompts": True,
        "support_frames_replayed_each_episode": 6,
        "episode_first_frame_gt_prompt": False,
        "mid_episode_reprompt": False,
        "episode_ground_truth_read": False,
        "causal_frame_order": True,
        "prompt_adapter": "declared_union_to_entity_indexed_support_v1",
        "role_projector": "support_conditioned_object_graph_v1",
        "model_internal_features_exported": True,
        "reported_confidence_cross_backend_comparable": False,
        "strict_source_pixel_access_gate": True,
        "online_deployment_eligible": True,
        "runtime_ms_semantics": {
            "cutie_runtime_ms": "adapter_track_internal_per_current_frame_v1",
            "parser_runtime_ms": "tokenizer_project_internal_per_current_frame_v1",
            "end_to_end_runtime_ms": (
                "episode_end_to_end_amortized_per_frame_including_reset_support_replay_"
                "excluding_npz_io_and_model_construction_v1"
            ),
        },
    }
    if protocol != expected_protocol:
        raise ValueError("Object-graph backend protocol changed.")
    provenance = payload.get("backend_provenance")
    required_provenance = {
        "model_family",
        "treatment",
        "model_size",
        "tracker_size",
        "oc_storm_repo",
        "cutie_implementation",
        "object_graph_implementation",
        "graph_tree",
        "checkpoint",
        "checkpoint_sha256",
        "seed",
        "python",
        "platform",
        "torch",
        "cuda_runtime",
        "cuda_visible_devices",
        "gpu_uuid",
        "cuda_device_order",
        "logical_cuda_device",
        "device_name",
        "amp",
        "wallclock_seconds",
        "runtime_by_task",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("Object-graph backend provenance schema changed.")
    require_sha256(provenance.get("checkpoint_sha256"), "graph checkpoint SHA")
    if (
        provenance.get("model_family") != "cutie"
        or provenance.get("treatment") != "support_conditioned_object_graph_v1"
        or provenance.get("model_size") != "small"
        or provenance.get("tracker_size") != [448, 448]
        or provenance.get("seed") != EXPECTED_BACKEND_SEED
        or provenance.get("amp") is not True
        or provenance.get("cuda_device_order") != "PCI_BUS_ID"
        or provenance.get("logical_cuda_device") != 0
        or not str(provenance.get("gpu_uuid", "")).startswith("GPU-")
    ):
        raise ValueError("Object-graph model/config/GPU binding is incomplete.")

    graph_entries = payload.get("graphs")
    if not isinstance(graph_entries, dict) or set(graph_entries) != set(TASK_ROLES):
        raise ValueError("Object-graph metadata map is incomplete.")
    compiled_graphs = {}
    for task, graph_path in graph_paths.items():
        graph = load_object_graph(graph_path)
        compiled_graphs[task] = graph
        entry = graph_entries[task]
        if not isinstance(entry, dict) or set(entry) != {
            "graph_path",
            "graph_file_sha256",
            "graph",
            "entity_support",
            "tokenizer",
        }:
            raise ValueError(f"Graph metadata schema changed for {task}.")
        if (
            entry.get("graph_file_sha256") != file_sha256(graph_path)
            or entry.get("graph") != graph.metadata()
            or entry.get("tokenizer", {}).get("graph_sha256") != graph.graph_sha256
            or entry.get("entity_support", {}).get("graph_sha256") != graph.graph_sha256
        ):
            raise ValueError(f"Graph/config/calibration provenance mismatch for {task}.")
        if entry.get("tokenizer", {}).get("forbidden_runtime_inputs") != [
            "simulator_state",
            "episode_ground_truth",
            "reward",
            "action",
            "future_rgb",
        ]:
            raise ValueError(f"Tokenizer forbidden-input contract changed for {task}.")

    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(TASK_ROLES):
        raise ValueError("Object-graph backend task results are incomplete.")
    root = manifest_path.parent
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    resolution = int(dataset["resolution"])
    paths: dict[tuple[str, str, int], Path] = {}
    for task, roles in TASK_ROLES.items():
        graph = compiled_graphs[task]
        condition_map = results[task]
        if not isinstance(condition_map, dict) or set(condition_map) != set(CONDITIONS):
            raise ValueError(f"Object-graph conditions are incomplete for {task}.")
        expected_shapes = _expected_shapes(
            frames=frames,
            roles=len(roles),
            entities=len(graph.entities),
            resolution=resolution,
        )
        for condition in CONDITIONS:
            records = condition_map[condition]
            if not isinstance(records, list) or len(records) != episodes:
                raise ValueError(f"Object-graph episode count changed for {task}/{condition}.")
            for episode_index, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != {
                    "episode_index",
                    "frames",
                    "entity_count",
                    "role_count",
                    "prediction_arrays",
                    "prediction_arrays_sha256",
                    "array_shapes",
                    "traces",
                }:
                    raise ValueError("Object-graph result record schema changed.")
                if (
                    record.get("episode_index") != episode_index
                    or record.get("frames") != frames
                    or record.get("entity_count") != len(graph.entities)
                    or record.get("role_count") != len(roles)
                    or record.get("array_shapes") != expected_shapes
                    or not isinstance(record.get("traces"), dict)
                    or set(record["traces"]) != TRACE_KEYS
                ):
                    raise ValueError("Object-graph result counts/shapes changed.")
                path = resolve_member(root, record.get("prediction_arrays"), "graph arrays")
                if file_sha256(path) != require_sha256(
                    record.get("prediction_arrays_sha256"), "graph prediction SHA"
                ):
                    raise ValueError("Object-graph prediction artifact changed.")
                for name, digest in record["traces"].items():
                    require_sha256(digest, f"object-graph {name}")
                paths[(task, condition, episode_index)] = path
    return payload, paths, compiled_graphs


def _dtype_contract(name: str, value: np.ndarray) -> None:
    expected = {
        "role_masks": np.bool_,
        "entity_masks": np.bool_,
        "descriptors": np.float32,
        "keypoints_xy": np.float32,
        "role_valid": np.bool_,
        "role_confidence": np.float32,
        "role_lost": np.bool_,
        "role_mask_score": np.float32,
        "entity_valid": np.bool_,
        "entity_confidence": np.float32,
        "entity_lost": np.bool_,
        "entity_mask_score": np.float32,
        "cutie_runtime_ms": np.float64,
        "parser_runtime_ms": np.float64,
        "end_to_end_runtime_ms": np.float64,
    }[name]
    if value.dtype != expected:
        raise ValueError(f"{name} dtype changed: {value.dtype} != {expected}.")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def _entity_gt(gt: np.ndarray, role_to_entity: tuple[int, ...]) -> np.ndarray:
    output = np.zeros_like(gt, dtype=np.uint8)
    for role_index, entity_index in enumerate(role_to_entity, start=1):
        output[gt == role_index] = entity_index + 1
    return output


def _invalid_bursts(values: np.ndarray) -> tuple[list[int], list[int]]:
    frames, roles = values.shape
    counts = [0 for _ in range(roles)]
    maximum = [0 for _ in range(roles)]
    for frame_index in range(frames):
        for role_index in range(roles):
            if bool(values[frame_index, role_index]):
                counts[role_index] = 0
            else:
                counts[role_index] += 1
                maximum[role_index] = max(maximum[role_index], counts[role_index])
    return counts, maximum


def _score_graph_backend(
    *,
    payload: dict[str, Any],
    paths: dict[tuple[str, str, int], Path],
    graphs: dict[str, Any],
    dataset: dict[str, Any],
    dataset_root: Path,
    baseline_paths: dict[tuple[str, str, int], Path],
) -> tuple[dict[str, Any], dict[str, dict[str, bool]]]:
    resolution = int(dataset["resolution"])
    frames = int(dataset["counts"]["frames_per_episode"])
    episodes = int(dataset["counts"]["episodes"])
    output: dict[str, Any] = {}
    direct_parity: dict[str, dict[str, bool]] = {}
    for task, roles in TASK_ROLES.items():
        graph = graphs[task]
        output[task] = {}
        for condition in CONDITIONS:
            role_scorer = MaskScorer(roles, resolution)
            entity_scorer = MaskScorer(graph.entity_names, resolution)
            valid_count = np.zeros(len(roles), dtype=np.int64)
            invalid_max = np.zeros(len(roles), dtype=np.int64)
            partition_mismatch_frames = 0
            role_overlap_frames = 0
            cutie_runtime: list[float] = []
            parser_runtime: list[float] = []
            end_runtime: list[float] = []
            direct_masks_identical = True
            direct_lost_identical = True
            direct_confidence_identical = True
            for episode_index in range(episodes):
                record = payload["results"][task][condition][episode_index]
                arrays = _load_npz(paths[(task, condition, episode_index)], ARRAY_KEYS)
                for name, value in arrays.items():
                    _dtype_contract(name, value)
                    if list(value.shape) != record["array_shapes"][name]:
                        raise ValueError(f"Decoded graph array shape changed: {name}.")
                trace_values = {
                    "role_mask_trace_sha256": _array_trace(arrays["role_masks"]),
                    "entity_mask_trace_sha256": _array_trace(arrays["entity_masks"]),
                    "descriptor_trace_sha256": _array_trace(arrays["descriptors"]),
                    "keypoint_trace_sha256": _array_trace(arrays["keypoints_xy"]),
                    "role_status_trace_sha256": _array_trace(np.concatenate((
                        arrays["role_valid"].astype(np.float32)[..., None],
                        arrays["role_confidence"][..., None],
                        arrays["role_lost"].astype(np.float32)[..., None],
                        arrays["role_mask_score"][..., None],
                    ), axis=-1)),
                    "entity_status_trace_sha256": _array_trace(np.concatenate((
                        arrays["entity_valid"].astype(np.float32)[..., None],
                        arrays["entity_confidence"][..., None],
                        arrays["entity_lost"].astype(np.float32)[..., None],
                        arrays["entity_mask_score"][..., None],
                    ), axis=-1)),
                    "cutie_runtime_trace_sha256": _array_trace(arrays["cutie_runtime_ms"]),
                    "parser_runtime_trace_sha256": _array_trace(arrays["parser_runtime_ms"]),
                    "end_to_end_runtime_trace_sha256": _array_trace(arrays["end_to_end_runtime_ms"]),
                }
                if trace_values != record["traces"]:
                    raise ValueError("Decoded graph trace bundle changed.")
                if np.any(arrays["role_valid"] & arrays["role_lost"]):
                    raise ValueError("A lost graph role cannot be valid.")
                if np.any(arrays["entity_valid"] & arrays["entity_lost"]):
                    raise ValueError("A lost graph entity cannot be valid.")
                if np.any(arrays["role_masks"].sum(axis=1) > 1):
                    role_overlap_frames += int(
                        np.any(arrays["role_masks"].sum(axis=1) > 1, axis=(1, 2)).sum()
                    )
                gt_path = resolve_member(
                    dataset_root,
                    dataset["episodes"][task][condition][episode_index]["arrays"],
                    "graph scoring arrays",
                )
                gt = _load_npz(
                    gt_path, {"gt_indexed", "actions", "physics_states"}
                )["gt_indexed"]
                entity_gt = _entity_gt(gt, graph.role_to_entity_index)
                role_scorer.begin_episode()
                entity_scorer.begin_episode()
                for frame_index in range(frames):
                    role_scorer.record(
                        predicted=arrays["role_masks"][frame_index],
                        gt_indexed=gt[frame_index],
                        runtime_ms=float(arrays["end_to_end_runtime_ms"][frame_index]),
                    )
                    entity_scorer.record(
                        predicted=arrays["entity_masks"][frame_index],
                        gt_indexed=entity_gt[frame_index],
                        runtime_ms=max(
                            float(arrays["cutie_runtime_ms"][frame_index]), 1e-9
                        ),
                    )
                    for entity_index in range(len(graph.entities)):
                        role_indices = [
                            index
                            for index, owner in enumerate(graph.role_to_entity_index)
                            if owner == entity_index
                        ]
                        if bool(arrays["role_valid"][frame_index, role_indices].all()):
                            union = arrays["role_masks"][frame_index, role_indices].any(axis=0)
                            partition_mismatch_frames += int(
                                not np.array_equal(
                                    union,
                                    arrays["entity_masks"][frame_index, entity_index],
                                )
                            )
                valid_count += arrays["role_valid"].sum(axis=0)
                _, episode_burst = _invalid_bursts(arrays["role_valid"])
                invalid_max = np.maximum(invalid_max, episode_burst)
                cutie_runtime.extend(arrays["cutie_runtime_ms"].tolist())
                parser_runtime.extend(arrays["parser_runtime_ms"].tolist())
                end_runtime.extend(arrays["end_to_end_runtime_ms"].tolist())
                if all(entity.projector.type == "direct_role_v1" for entity in graph.entities):
                    baseline = _load_npz(
                        baseline_paths[(task, condition, episode_index)],
                        {"predicted_masks", "reported_confidence", "reported_lost", "runtime_ms"},
                    )
                    direct_masks_identical &= np.array_equal(
                        arrays["role_masks"], baseline["predicted_masks"]
                    )
                    direct_lost_identical &= np.array_equal(
                        arrays["entity_lost"], baseline["reported_lost"]
                    )
                    direct_confidence_identical &= np.array_equal(
                        arrays["entity_confidence"], baseline["reported_confidence"]
                    )
            role_summary = role_scorer.summary()
            entity_summary = entity_scorer.summary()
            parser_summary = {
                "per_role": {
                    role: {
                        "valid_rate": float(valid_count[index]) / (episodes * frames),
                        "max_invalid_burst": int(invalid_max[index]),
                    }
                    for index, role in enumerate(roles)
                },
                "partition_mismatch_frames_when_all_roles_valid": partition_mismatch_frames,
                "role_overlap_frames": role_overlap_frames,
                "runtime": {
                    "cutie": MaskScorer._latency(cutie_runtime),
                    "parser": MaskScorer._latency(parser_runtime),
                    "end_to_end": MaskScorer._latency(end_runtime),
                },
            }
            output[task][condition] = {
                "semantic_roles": role_summary,
                "tracking_entities": entity_summary,
                "parser": parser_summary,
            }
            if all(entity.projector.type == "direct_role_v1" for entity in graph.entities):
                direct_parity[f"{task}/{condition}"] = {
                    "role_masks": direct_masks_identical,
                    "reported_lost": direct_lost_identical,
                    "reported_confidence": direct_confidence_identical,
                }
    return output, direct_parity


def _score_gt_union_parser_oracle(
    *,
    graphs: dict[str, Any],
    dataset: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Offline privileged diagnostic: parse exact GT entity unions."""
    result = {}
    resolution = int(dataset["resolution"])
    episodes = int(dataset["counts"]["episodes"])
    frames = int(dataset["counts"]["frames_per_episode"])
    for task, graph in graphs.items():
        if all(entity.projector.type == "direct_role_v1" for entity in graph.entities):
            continue
        support_path = resolve_member(
            dataset_root, dataset["support"][task]["arrays"], "oracle support arrays"
        )
        support = _load_npz(support_path, {"rgb", "indexed_masks"})
        tokenizer = SupportConditionedObjectGraphTokenizer(
            graph, support["indexed_masks"]
        )
        result[task] = {}
        for condition in CONDITIONS:
            scorer = MaskScorer(graph.semantic_roles, resolution)
            valid_count = np.zeros(len(graph.semantic_roles), dtype=np.int64)
            invalid_max = np.zeros(len(graph.semantic_roles), dtype=np.int64)
            for episode_index in range(episodes):
                gt_path = resolve_member(
                    dataset_root,
                    dataset["episodes"][task][condition][episode_index]["arrays"],
                    "oracle scoring arrays",
                )
                gt = _load_npz(
                    gt_path, {"gt_indexed", "actions", "physics_states"}
                )["gt_indexed"]
                tokenizer.reset_episode()
                scorer.begin_episode()
                episode_valid = np.zeros((frames, len(graph.semantic_roles)), dtype=bool)
                for frame_index in range(frames):
                    entity_gt = _entity_gt(
                        gt[frame_index], graph.role_to_entity_index
                    )
                    entity_masks = np.stack(
                        [
                            entity_gt == entity_index + 1
                            for entity_index in range(len(graph.entities))
                        ]
                    )
                    projected = tokenizer.project(
                        entity_masks=entity_masks,
                        entity_features=np.ones(
                            (len(graph.entities), QUERY_FEATURE_DIM), dtype=np.float32
                        ),
                        entity_lost=np.zeros(len(graph.entities), dtype=bool),
                        entity_confidence=np.ones(len(graph.entities), dtype=np.float32),
                        entity_mask_score=np.ones(len(graph.entities), dtype=np.float32),
                    )
                    scorer.record(
                        predicted=projected.masks,
                        gt_indexed=gt[frame_index],
                        runtime_ms=max(projected.runtime_ms, 1e-9),
                    )
                    episode_valid[frame_index] = projected.valid
                valid_count += episode_valid.sum(axis=0)
                _, burst = _invalid_bursts(episode_valid)
                invalid_max = np.maximum(invalid_max, burst)
            summary = scorer.summary()
            summary["parser_validity"] = {
                role: {
                    "valid_rate": float(valid_count[index]) / (episodes * frames),
                    "max_invalid_burst": int(invalid_max[index]),
                }
                for index, role in enumerate(graph.semantic_roles)
            }
            result[task][condition] = summary
    return result


def _metric(role: dict[str, Any], name: str) -> float:
    value = role.get(name)
    return 0.0 if value is None else float(value)


def _gates(
    *,
    baseline: dict[str, Any],
    graph_metrics: dict[str, Any],
    parser_oracle: dict[str, Any],
    direct_parity: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for cell, parity in direct_parity.items():
        for field, value in parity.items():
            checks[f"direct_exact/{cell}/{field}"] = value
    for task in ("reacher-visual-small", "cartpole-swingup"):
        for condition in CONDITIONS:
            candidate = graph_metrics[task][condition]["semantic_roles"]
            reference = baseline[task][condition]
            for role in TASK_ROLES[task]:
                for name in (
                    "visible_recall",
                    "mean_iou_on_gt_visible_frames",
                    "tolerant_f1_radius_2_on_gt_visible_frames",
                    "identity_accuracy_on_gt_visible_frames",
                ):
                    checks[f"direct_no_regression/{task}/{condition}/{role}/{name}"] = (
                        _metric(candidate["per_role"][role], name)
                        >= _metric(reference["per_role"][role], name)
                        - MAX_DIRECT_METRIC_REGRESSION
                    )
    task = "acrobot-swingup"
    for condition in CONDITIONS:
        candidate = graph_metrics[task][condition]
        baseline_cell = baseline[task][condition]
        entity = candidate["tracking_entities"]
        entity_role = next(iter(entity["per_role"].values()))
        for name, threshold in (
            ("visible_recall", 0.95),
            ("mean_iou_on_gt_visible_frames", 0.50),
            ("tolerant_f1_radius_2_on_gt_visible_frames", 0.80),
            ("success_at_iou_0_5_rate_on_gt_visible_frames", 0.50),
        ):
            checks[f"entity_quality/{condition}/{name}"] = _metric(entity_role, name) >= threshold
        checks[f"entity_quality/{condition}/burst"] = (
            int(entity_role["max_failure_burst_at_iou_0_5_or_identity"]) <= 10
        )
        semantic = candidate["semantic_roles"]
        parser_validity = candidate["parser"]["per_role"]
        for role in TASK_ROLES[task]:
            role_metrics = semantic["per_role"][role]
            for name, threshold in (
                ("visible_recall", 0.90),
                ("mean_iou_on_gt_visible_frames", 0.45),
                ("tolerant_f1_radius_2_on_gt_visible_frames", 0.75),
                ("identity_accuracy_on_gt_visible_frames", 0.90),
                ("success_at_iou_0_5_rate_on_gt_visible_frames", 0.45),
            ):
                checks[f"semantic_quality/{condition}/{role}/{name}"] = (
                    _metric(role_metrics, name) >= threshold
                )
            checks[f"semantic_quality/{condition}/{role}/burst"] = (
                int(role_metrics["max_failure_burst_at_iou_0_5_or_identity"]) <= 25
            )
            checks[f"parser_validity/{condition}/{role}/rate"] = (
                float(parser_validity[role]["valid_rate"]) >= MIN_ROLE_VALID_RATE
            )
            checks[f"parser_validity/{condition}/{role}/burst"] = (
                int(parser_validity[role]["max_invalid_burst"])
                <= MAX_ROLE_INVALID_BURST
            )
        checks[f"semantic_quality/{condition}/swaps"] = semantic["role_swap_frames"] == 0
        lower = semantic["per_role"]["lower_arm"]
        base_lower = baseline_cell["per_role"]["lower_arm"]
        for name, gain in (
            ("visible_recall", 0.20),
            ("mean_iou_on_gt_visible_frames", 0.10),
            ("identity_accuracy_on_gt_visible_frames", 0.20),
            ("success_at_iou_0_5_rate_on_gt_visible_frames", 0.10),
        ):
            checks[f"lower_gain/{condition}/{name}"] = (
                _metric(lower, name) >= _metric(base_lower, name) + gain
            )
        parser_runtime = candidate["parser"]["runtime"]["parser"]
        total_runtime = candidate["parser"]["runtime"]["end_to_end"]
        baseline_runtime = baseline_cell["latency"]
        checks[f"latency/{condition}/parser_mean"] = (
            parser_runtime["mean_ms"] <= MAX_PARSER_MEAN_MS
        )
        checks[f"latency/{condition}/parser_p95"] = (
            parser_runtime["p95_ms"] <= MAX_PARSER_P95_MS
        )
        checks[f"latency/{condition}/total"] = total_runtime["mean_ms"] <= min(
            baseline_runtime["mean_ms"] * MAX_TOTAL_RATIO_VS_CUTIE,
            baseline_runtime["mean_ms"] + MAX_TOTAL_DELTA_MS_VS_CUTIE,
        )
        checks[f"partition/{condition}"] = (
            candidate["parser"]["partition_mismatch_frames_when_all_roles_valid"] == 0
            and candidate["parser"]["role_overlap_frames"] == 0
        )

        oracle = parser_oracle[task][condition]
        oracle_validity = oracle["parser_validity"]
        checks[f"parser_oracle/{condition}/swaps"] = oracle["role_swap_frames"] == 0
        for role in TASK_ROLES[task]:
            role_metrics = oracle["per_role"][role]
            for name, threshold in (
                ("visible_recall", 0.95),
                ("mean_iou_on_gt_visible_frames", 0.50),
                ("tolerant_f1_radius_2_on_gt_visible_frames", 0.80),
                ("identity_accuracy_on_gt_visible_frames", 0.95),
                ("success_at_iou_0_5_rate_on_gt_visible_frames", 0.50),
            ):
                checks[f"parser_oracle/{condition}/{role}/{name}"] = (
                    _metric(role_metrics, name) >= threshold
                )
            checks[f"parser_oracle/{condition}/{role}/burst"] = (
                int(role_metrics["max_failure_burst_at_iou_0_5_or_identity"]) <= 10
            )
            checks[f"parser_oracle/{condition}/{role}/valid_rate"] = (
                float(oracle_validity[role]["valid_rate"]) >= MIN_ORACLE_VALID_RATE
            )
            checks[f"parser_oracle/{condition}/{role}/invalid_burst"] = (
                int(oracle_validity[role]["max_invalid_burst"])
                <= MAX_ORACLE_INVALID_BURST
            )
    return {"checks": checks, "pass": all(checks.values())}


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    source_summary, dataset, dataset_path, baseline_path = _source_artifacts(
        args.source_benchmark_root
    )
    graph_paths = _graph_paths(args.graph_dir)
    baseline_payload, baseline_paths = _validate_backend_manifest(
        baseline_path, backend="cutie", dataset=dataset
    )
    graph_payload, graph_prediction_paths, graphs = _validate_graph_backend(
        args.object_graph_backend,
        dataset=dataset,
        graph_paths=graph_paths,
    )
    isolation = _validate_isolation_gate(
        args.isolation_gate,
        source_root=args.source_benchmark_root,
        summary_root=args.output.parent,
        backend_manifest=args.object_graph_backend,
    )
    graph_provenance = graph_payload["backend_provenance"]
    baseline_provenance = baseline_payload["backend_provenance"]
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
        if graph_provenance.get(field) != baseline_provenance.get(field):
            raise ValueError(
                f"Baseline/graph Cutie provenance differs for {field}."
            )
    if graph_provenance.get("cutie_implementation") != baseline_provenance.get(
        "implementation"
    ):
        raise ValueError("Baseline/graph Cutie implementation trees differ.")
    baseline_metrics = _score_backend(
        backend="cutie",
        backend_payload=baseline_payload,
        prediction_paths=baseline_paths,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    graph_metrics, direct_parity = _score_graph_backend(
        payload=graph_payload,
        paths=graph_prediction_paths,
        graphs=graphs,
        dataset=dataset,
        dataset_root=dataset_path.parent,
        baseline_paths=baseline_paths,
    )
    parser_oracle = _score_gt_union_parser_oracle(
        graphs=graphs,
        dataset=dataset,
        dataset_root=dataset_path.parent,
    )
    gate = _gates(
        baseline=baseline_metrics,
        graph_metrics=graph_metrics,
        parser_oracle=parser_oracle,
        direct_parity=direct_parity,
    )
    development_candidate = bool(gate["pass"])
    payload = {
        "format": SUMMARY_FORMAT,
        "status": (
            "object_graph_development_preflight_candidate"
            if development_candidate
            else "object_graph_development_preflight_no_go"
        ),
        "engineering_pass": True,
        "development_candidate": development_candidate,
        "controller_training_authorized": False,
        "scientific_go": False,
        "recommendation": (
            "freeze_new_unseen_trajectory_seeds_and_run_confirmatory_preflight"
            if development_candidate
            else "do_not_train_controller_diagnose_entity_tracker_or_role_parser"
        ),
        "scope": {
            "evidence_level": "post_hoc_development_preflight_on_previously_inspected_random_trajectories",
            "allowed_claim": "object_graph_implementation_and_development_signal",
            "disallowed_claims": [
                "controller_advantage",
                "confirmatory_perception_result",
                "paper_level_generality",
            ],
            "controller_training_steps": 0,
            "same_frozen_rgb_as_unified_vos": True,
            "episode_ground_truth_backend_access": False,
            "gt_union_parser_oracle_is_offline_privileged_diagnostic": True,
            "task_conditioned_graphs": True,
            "core_task_name_dispatch": False,
            "task_specific_role_ontology_required": True,
            "support_labels_per_task": 6,
            "simulator_support_labels_used": True,
            "manual_support_deployability_not_proven": True,
            "fixed_camera_evaluation": True,
            "runtime_simulator_state_used": False,
            "scoring_isolation_gate_validated": isolation["status"] == "complete",
        },
        "thresholds": {
            "max_direct_metric_regression": MAX_DIRECT_METRIC_REGRESSION,
            "max_parser_mean_ms": MAX_PARSER_MEAN_MS,
            "max_parser_p95_ms": MAX_PARSER_P95_MS,
            "max_total_ratio_vs_cutie": MAX_TOTAL_RATIO_VS_CUTIE,
            "max_total_delta_ms_vs_cutie": MAX_TOTAL_DELTA_MS_VS_CUTIE,
            "min_role_valid_rate": MIN_ROLE_VALID_RATE,
            "max_role_invalid_burst": MAX_ROLE_INVALID_BURST,
            "min_oracle_valid_rate": MIN_ORACLE_VALID_RATE,
            "max_oracle_invalid_burst": MAX_ORACLE_INVALID_BURST,
            "serialized_in_gate_checks": True,
        },
        "gate": gate,
        "direct_parity": direct_parity,
        "baseline_cutie_metrics": baseline_metrics,
        "object_graph_metrics": graph_metrics,
        "gt_union_parser_oracle": parser_oracle,
        "graphs": {task: graph.metadata() for task, graph in graphs.items()},
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_benchmark_root": str(args.source_benchmark_root.resolve()),
            "source_summary_sha256": file_sha256(
                args.source_benchmark_root / "unified_vos_summary.json"
            ),
            "source_dataset_manifest_sha256": file_sha256(dataset_path),
            "source_dataset_id": dataset["dataset_id"],
            "source_cutie_manifest_sha256": file_sha256(baseline_path),
            "object_graph_backend_manifest_relative_to_summary_root": (
                args.object_graph_backend.resolve()
                .relative_to(args.output.parent.resolve())
                .as_posix()
            ),
            "object_graph_backend_manifest_sha256": file_sha256(
                args.object_graph_backend
            ),
            "scoring_isolation_relative_to_summary_root": (
                args.isolation_gate.resolve()
                .relative_to(args.output.parent.resolve())
                .as_posix()
            ),
            "scoring_isolation_sha256": file_sha256(args.isolation_gate),
            "graph_files": {
                task: {
                    "path": str(path),
                    "file_sha256": file_sha256(path),
                    "semantic_sha256": graphs[task].graph_sha256,
                }
                for task, path in graph_paths.items()
            },
            "source_status": source_summary.get("status"),
        },
    }
    write_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-benchmark-root", type=Path, required=True)
    parser.add_argument("--object-graph-backend", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--isolation-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "source_benchmark_root",
        "object_graph_backend",
        "graph_dir",
        "isolation_gate",
        "output",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    result = aggregate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "engineering_pass": result["engineering_pass"],
                "development_candidate": result["development_candidate"],
                "controller_training_authorized": False,
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
