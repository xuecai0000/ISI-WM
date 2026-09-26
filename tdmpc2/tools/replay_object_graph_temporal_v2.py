"""Causal, mask-only temporal-v2 replay over a published v1 object-graph run.

This backend never decodes episode RGB or uses it as parser input, and never
reads episode ground truth, simulator state, actions, rewards, or raw Cutie
appearance features.  The GT-free input validator hashes episode-RGB artifacts
only to bind the frozen dataset.  The parser consumes only published v1 entity
masks/status and frozen support indexed masks.  Its output is a parser
diagnostic (role masks, keypoints, reliability and runtime), not a controller
observation and not a reconstruction of the 590-D descriptor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.common.unified_vos import (  # noqa: E402
    CONDITIONS,
    TASK_ROLES,
    file_sha256,
    load_json,
    require_sha256,
    resolve_member,
    validate_backend_inputs,
    write_json,
)
from tdmpc2.perception.support_conditioned_object_graph import (  # noqa: E402
    TEMPORAL_STATE_PROTOCOL,
    TEMPORAL_TOKENIZER_FORMAT,
    SupportConditionedObjectGraphTokenizer,
    load_object_graph,
)


FORMAT = "object_graph_temporal_v2_replay_predictions_v1"
BACKEND = "object_graph_temporal_v2_replay"
DIAGNOSTICS_FORMAT = "object_graph_temporal_v2_episode_diagnostics_v1"
V1_FORMAT = "object_graph_cutie_backend_predictions_v1"
V1_BACKEND = "object_graph_cutie"
TASK = "acrobot-swingup"
EXPECTED_V1_SEED = 2718281

V1_ARRAY_KEYS = {
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
V1_TRACE_KEYS = {
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
OUTPUT_ARRAY_KEYS = {
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
OUTPUT_TRACE_KEYS = {
    "role_mask_trace_sha256",
    "keypoint_trace_sha256",
    "projector_status_trace_sha256",
    "role_status_trace_sha256",
    "parser_runtime_trace_sha256",
    "diagnostics_trace_sha256",
}


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    result = candidate.resolve(strict=True)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _array_trace(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Temporal diagnostics keys must be strings.")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Temporal diagnostics contain a non-finite float.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported temporal diagnostic value: {type(value).__name__}.")


def _implementation_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_DIR / "perception" / "support_conditioned_object_graph.py",
        PROJECT_DIR / "common" / "unified_vos.py",
    )
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Replay implementation must be a regular file: {path}")
        result[path.relative_to(REPO_DIR).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def _load_support(path: Path, *, resolution: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"rgb", "indexed_masks"}:
            raise ValueError("Frozen support archive schema changed.")
        rgb = np.ascontiguousarray(archive["rgb"])
        masks = np.ascontiguousarray(archive["indexed_masks"])
    if rgb.shape != (6, resolution, resolution, 3) or rgb.dtype != np.uint8:
        raise ValueError("Frozen support RGB schema changed.")
    if (
        masks.shape != (6, resolution, resolution)
        or masks.dtype == np.bool_
        or not np.issubdtype(masks.dtype, np.integer)
    ):
        raise ValueError("Frozen support indexed-mask schema changed.")
    return masks


def _graph_pair(v1_path: Path, temporal_path: Path):
    v1_path = _regular_file(v1_path, "v1 graph")
    temporal_path = _regular_file(temporal_path, "temporal graph")
    v1_graph = load_object_graph(v1_path)
    temporal_graph = load_object_graph(temporal_path)
    for graph in (v1_graph, temporal_graph):
        if (
            graph.task != TASK
            or graph.source_roles != TASK_ROLES[TASK]
            or graph.semantic_roles != TASK_ROLES[TASK]
            or len(graph.entities) != 1
            or graph.entities[0].source_roles != TASK_ROLES[TASK]
            or graph.entities[0].projector.roles != TASK_ROLES[TASK]
        ):
            raise ValueError("Acrobot graph task/entity/role topology changed.")
    if (
        v1_graph.graph_name != "acrobot_swingup_whole_entity_ordered_chain_v1"
        or v1_graph.entities[0].projector.type != "ordered_chain_segments_v1"
    ):
        raise ValueError("The frozen v1 Acrobot graph is not the published chain-v1 graph.")
    if (
        temporal_graph.graph_name
        != "acrobot_swingup_whole_entity_ordered_chain_temporal_v2"
        or temporal_graph.entities[0].projector.type != "ordered_chain_temporal_v2"
    ):
        raise ValueError("The replay graph is not the declared temporal-v2 graph.")
    v1_payload = json.loads(_canonical_json_bytes(v1_graph.canonical_payload))
    temporal_payload = json.loads(_canonical_json_bytes(temporal_graph.canonical_payload))
    temporal_payload["graph_name"] = v1_payload["graph_name"]
    temporal_payload["tracking_entities"][0]["projector"]["type"] = (
        "ordered_chain_segments_v1"
    )
    if temporal_payload != v1_payload:
        raise ValueError(
            "Temporal-v2 graph must differ from frozen v1 only by graph_name and projector.type."
        )
    return v1_graph, temporal_graph, v1_path, temporal_path


def _expected_v1_shapes(*, frames: int, resolution: int) -> dict[str, list[int]]:
    return {
        "role_masks": [frames, 2, resolution, resolution],
        "entity_masks": [frames, 1, resolution, resolution],
        "descriptors": [frames, 2, 590],
        "keypoints_xy": [frames, 2, 2, 2],
        "role_valid": [frames, 2],
        "role_confidence": [frames, 2],
        "role_lost": [frames, 2],
        "role_mask_score": [frames, 2],
        "entity_valid": [frames, 1],
        "entity_confidence": [frames, 1],
        "entity_lost": [frames, 1],
        "entity_mask_score": [frames, 1],
        "cutie_runtime_ms": [frames],
        "parser_runtime_ms": [frames],
        "end_to_end_runtime_ms": [frames],
    }


def _validate_v1_manifest(
    path: Path,
    *,
    inputs: dict[str, Any],
    input_sha256: str,
    v1_graph: Any,
    v1_graph_path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], tuple[Path, dict[str, Any]]]]:
    path = path.expanduser().resolve(strict=True)
    payload = load_json(path)
    if set(payload) != {
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
    }:
        raise ValueError("Published v1 object-graph manifest schema changed.")
    if (
        payload.get("format") != V1_FORMAT
        or payload.get("status") != "complete"
        or payload.get("backend") != V1_BACKEND
        or payload.get("dataset_id") != inputs["dataset_id"]
        or payload.get("input_manifest_sha256") != input_sha256
        or payload.get("roles")
        != {task: list(roles) for task, roles in TASK_ROLES.items()}
    ):
        raise ValueError("Published v1 object-graph identity/input pairing changed.")
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
    if payload.get("protocol") != expected_protocol:
        raise ValueError("Published v1 object-graph protocol changed.")
    provenance = payload.get("backend_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("model_family") != "cutie"
        or provenance.get("treatment") != "support_conditioned_object_graph_v1"
        or provenance.get("model_size") != "small"
        or provenance.get("tracker_size") != [448, 448]
        or provenance.get("seed") != EXPECTED_V1_SEED
        or provenance.get("amp") is not True
    ):
        raise ValueError("Published v1 object-graph provenance changed.")
    graph_entries = payload.get("graphs")
    if not isinstance(graph_entries, dict) or set(graph_entries) != set(TASK_ROLES):
        raise ValueError("Published v1 graph map is incomplete.")
    entry = graph_entries[TASK]
    if not isinstance(entry, dict) or set(entry) != {
        "graph_path",
        "graph_file_sha256",
        "graph",
        "entity_support",
        "tokenizer",
    }:
        raise ValueError("Published v1 Acrobot graph entry schema changed.")
    if (
        entry.get("graph_file_sha256") != file_sha256(v1_graph_path)
        or entry.get("graph") != v1_graph.metadata()
        or entry.get("entity_support", {}).get("graph_sha256")
        != v1_graph.graph_sha256
        or entry.get("tokenizer", {}).get("graph_sha256") != v1_graph.graph_sha256
    ):
        raise ValueError("Published v1 Acrobot graph no longer matches --v1-graph.")

    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != set(TASK_ROLES):
        raise ValueError("Published v1 result task set changed.")
    episodes = int(inputs["counts"]["episodes"])
    frames = int(inputs["counts"]["frames_per_episode"])
    resolution = int(inputs["resolution"])
    expected_shapes = _expected_v1_shapes(frames=frames, resolution=resolution)
    selected: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    root = path.parent
    for task in TASK_ROLES:
        condition_map = results[task]
        if not isinstance(condition_map, dict) or set(condition_map) != set(CONDITIONS):
            raise ValueError(f"Published v1 conditions changed for {task}.")
        for condition in CONDITIONS:
            records = condition_map[condition]
            if not isinstance(records, list) or len(records) != episodes:
                raise ValueError(f"Published v1 episode count changed for {task}/{condition}.")
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
                    raise ValueError("Published v1 episode record schema changed.")
                if record.get("episode_index") != episode_index or record.get("frames") != frames:
                    raise ValueError("Published v1 episode ordering/count changed.")
                if not isinstance(record.get("traces"), dict) or set(record["traces"]) != V1_TRACE_KEYS:
                    raise ValueError("Published v1 trace schema changed.")
                for name, digest in record["traces"].items():
                    require_sha256(digest, f"published v1 {name}")
                if task != TASK:
                    continue
                if (
                    record.get("entity_count") != 1
                    or record.get("role_count") != 2
                    or record.get("array_shapes") != expected_shapes
                ):
                    raise ValueError("Published v1 Acrobot array shapes changed.")
                prediction = resolve_member(
                    root, record.get("prediction_arrays"), "published v1 prediction arrays"
                )
                if file_sha256(prediction) != require_sha256(
                    record.get("prediction_arrays_sha256"), "published v1 prediction SHA"
                ):
                    raise ValueError("Published v1 prediction artifact changed.")
                selected[(condition, episode_index)] = (prediction, record)
    return payload, selected


def _load_v1_entity_arrays(
    path: Path,
    record: dict[str, Any],
    *,
    frames: int,
    resolution: int,
) -> dict[str, np.ndarray]:
    wanted = {
        "entity_masks",
        "entity_valid",
        "entity_confidence",
        "entity_lost",
        "entity_mask_score",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != V1_ARRAY_KEYS:
            raise ValueError("Published v1 prediction NPZ schema changed.")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in wanted}
    expected = {
        "entity_masks": ((frames, 1, resolution, resolution), np.bool_),
        "entity_valid": ((frames, 1), np.bool_),
        "entity_confidence": ((frames, 1), np.float32),
        "entity_lost": ((frames, 1), np.bool_),
        "entity_mask_score": ((frames, 1), np.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Published v1 {name} schema changed.")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Published v1 {name} contains non-finite values.")
    for name in ("entity_confidence", "entity_mask_score"):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise ValueError(f"Published v1 {name} escaped [0,1].")
    if np.any(arrays["entity_valid"] & arrays["entity_lost"]):
        raise ValueError("A published v1 entity cannot be both valid and lost.")
    nonempty = arrays["entity_masks"].reshape(frames, 1, -1).any(axis=2)
    expected_mask_only_valid = ~arrays["entity_lost"] & nonempty
    if not np.array_equal(arrays["entity_valid"], expected_mask_only_valid):
        raise ValueError(
            "Published v1 entity_valid is not exactly reproducible from current-mask "
            "non-emptiness and entity_lost; mask-only replay would not preserve the "
            "online availability trajectory."
        )
    if _array_trace(arrays["entity_masks"]) != record["traces"][
        "entity_mask_trace_sha256"
    ]:
        raise ValueError("Published v1 decoded entity-mask trace changed.")
    status = np.concatenate(
        (
            arrays["entity_valid"].astype(np.float32)[..., None],
            arrays["entity_confidence"][..., None],
            arrays["entity_lost"].astype(np.float32)[..., None],
            arrays["entity_mask_score"][..., None],
        ),
        axis=-1,
    )
    if _array_trace(status) != record["traces"]["entity_status_trace_sha256"]:
        raise ValueError("Published v1 decoded entity-status trace changed.")
    return arrays


def _validate_output_arrays(
    arrays: dict[str, np.ndarray], *, frames: int, resolution: int
) -> None:
    if set(arrays) != OUTPUT_ARRAY_KEYS:
        raise RuntimeError("Temporal replay output array schema changed.")
    expected = {
        "role_masks": ((frames, 2, resolution, resolution), np.bool_),
        "keypoints_xy": ((frames, 2, 2, 2), np.float32),
        "projector_valid": ((frames, 2), np.bool_),
        "projector_confidence": ((frames, 2), np.float32),
        "role_valid": ((frames, 2), np.bool_),
        "role_confidence": ((frames, 2), np.float32),
        "role_lost": ((frames, 2), np.bool_),
        "role_mask_score": ((frames, 2), np.float32),
        "parser_runtime_ms": ((frames,), np.float64),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"Temporal replay {name} is {value.shape}/{value.dtype}, expected {shape}/{dtype}."
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise RuntimeError(f"Temporal replay {name} contains non-finite values.")
    if np.any(arrays["role_masks"].astype(np.uint8).sum(axis=1) > 1):
        raise RuntimeError("Temporal replay role masks overlap.")
    nonempty = arrays["role_masks"].reshape(frames, 2, -1).any(axis=2)
    if np.any(arrays["projector_valid"] & ~nonempty):
        raise RuntimeError("A valid temporal projection cannot have an empty role mask.")
    if np.any(arrays["role_valid"] & ~arrays["projector_valid"]):
        raise RuntimeError("Role validity cannot exceed projector validity.")
    if np.any(arrays["role_valid"] & arrays["role_lost"]):
        raise RuntimeError("A lost temporal role cannot be valid.")
    for name in ("projector_confidence", "role_confidence", "role_mask_score"):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise RuntimeError(f"Temporal replay {name} escaped [0,1].")
    if np.any(arrays["parser_runtime_ms"] < 0.0):
        raise RuntimeError("Temporal replay parser runtime cannot be negative.")


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{name: np.ascontiguousarray(arrays[name]) for name in sorted(arrays)}
    )
    return file_sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "Temporal mask replay is CPU-only; set CUDA_VISIBLE_DEVICES to the empty string."
        )
    input_path = _regular_file(args.inputs, "GT-free worker input manifest")
    input_sha = file_sha256(input_path)
    inputs, support_paths, _ = validate_backend_inputs(
        input_path, strict_counts=args.strict_counts
    )
    v1_graph, temporal_graph, v1_graph_path, temporal_graph_path = _graph_pair(
        args.v1_graph, args.graph
    )
    v1_manifest_path = _regular_file(
        args.v1_backend_manifest, "published v1 backend manifest"
    )
    v1_manifest_sha = file_sha256(v1_manifest_path)
    v1_payload, v1_paths = _validate_v1_manifest(
        v1_manifest_path,
        inputs=inputs,
        input_sha256=input_sha,
        v1_graph=v1_graph,
        v1_graph_path=v1_graph_path,
    )
    resolution = int(inputs["resolution"])
    frames = int(inputs["counts"]["frames_per_episode"])
    episodes = int(inputs["counts"]["episodes"])
    support_path = support_paths[TASK]
    support_sha = file_sha256(support_path)
    support_masks = _load_support(support_path, resolution=resolution)
    implementation_before = _implementation_snapshot()
    v1_graph_sha = file_sha256(v1_graph_path)
    temporal_graph_sha = file_sha256(temporal_graph_path)

    output_root = args.output_root.expanduser().resolve()
    incomplete_root = output_root.with_name(output_root.name + ".incomplete")
    if output_root.exists() or incomplete_root.exists():
        raise FileExistsError(output_root if output_root.exists() else incomplete_root)
    if not output_root.parent.is_dir():
        raise FileNotFoundError(output_root.parent)
    incomplete_root.mkdir(exist_ok=False)

    tokenizer = SupportConditionedObjectGraphTokenizer(temporal_graph, support_masks)
    tokenizer_metadata = tokenizer.metadata()
    if (
        tokenizer_metadata.get("format") != TEMPORAL_TOKENIZER_FORMAT
        or tokenizer_metadata.get("temporal_state_protocol")
        != TEMPORAL_STATE_PROTOCOL
        or tokenizer_metadata.get("geometry_replay_entrypoint")
        != "project_temporal_geometry"
        or tokenizer_metadata.get("geometry_replay_protocol")
        != "mask_and_tracker_lost_only_v2"
        or tokenizer_metadata.get("geometry_replay_has_appearance_or_descriptors")
        is not False
    ):
        raise RuntimeError("Temporal tokenizer metadata/entrypoint contract changed.")

    started = perf_counter()
    results: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        records = []
        for episode_index in range(episodes):
            source_path, source_record = v1_paths[(condition, episode_index)]
            source = _load_v1_entity_arrays(
                source_path, source_record, frames=frames, resolution=resolution
            )
            arrays = {
                "role_masks": np.zeros((frames, 2, resolution, resolution), np.bool_),
                "keypoints_xy": np.zeros((frames, 2, 2, 2), np.float32),
                "projector_valid": np.zeros((frames, 2), np.bool_),
                "projector_confidence": np.zeros((frames, 2), np.float32),
                "role_valid": np.zeros((frames, 2), np.bool_),
                "role_confidence": np.zeros((frames, 2), np.float32),
                "role_lost": np.ones((frames, 2), np.bool_),
                "role_mask_score": np.zeros((frames, 2), np.float32),
                "parser_runtime_ms": np.zeros(frames, np.float64),
            }
            diagnostic_records = []
            tokenizer.reset_episode()
            for frame_index in range(frames):
                geometry = tokenizer.project_temporal_geometry(
                    entity_masks=source["entity_masks"][frame_index],
                    entity_lost=source["entity_lost"][frame_index],
                )
                if (
                    geometry.role_names != TASK_ROLES[TASK]
                    or geometry.masks.shape != (2, resolution, resolution)
                    or geometry.masks.dtype != np.bool_
                    or geometry.keypoints_xy.shape != (2, 2, 2)
                    or geometry.keypoints_xy.dtype != np.float32
                    or geometry.valid.shape != (2,)
                    or geometry.valid.dtype != np.bool_
                    or geometry.projector_confidence.shape != (2,)
                    or geometry.projector_confidence.dtype != np.float32
                    or not np.isfinite(geometry.keypoints_xy).all()
                    or not np.isfinite(geometry.projector_confidence).all()
                    or np.any(
                        (geometry.projector_confidence < 0.0)
                        | (geometry.projector_confidence > 1.0)
                    )
                    or not math.isfinite(float(geometry.runtime_ms))
                    or float(geometry.runtime_ms) < 0.0
                    or not isinstance(geometry.diagnostics, dict)
                ):
                    raise RuntimeError("Temporal geometry frame contract changed.")
                owner = np.asarray(temporal_graph.role_to_entity_index, dtype=np.int64)
                entity_valid = source["entity_valid"][frame_index, owner]
                entity_lost = source["entity_lost"][frame_index, owner]
                entity_confidence = source["entity_confidence"][frame_index, owner]
                entity_mask_score = source["entity_mask_score"][frame_index, owner]
                arrays["role_masks"][frame_index] = geometry.masks
                arrays["keypoints_xy"][frame_index] = geometry.keypoints_xy
                arrays["projector_valid"][frame_index] = geometry.valid
                arrays["projector_confidence"][frame_index] = (
                    geometry.projector_confidence
                )
                arrays["role_lost"][frame_index] = entity_lost
                arrays["role_valid"][frame_index] = geometry.valid & entity_valid
                arrays["role_confidence"][frame_index] = np.clip(
                    entity_confidence * geometry.projector_confidence, 0.0, 1.0
                )
                arrays["role_mask_score"][frame_index] = np.clip(
                    entity_mask_score * geometry.projector_confidence, 0.0, 1.0
                )
                arrays["parser_runtime_ms"][frame_index] = float(geometry.runtime_ms)
                diagnostic_records.append(
                    {
                        "frame_index": frame_index,
                        "source_entity_status": {
                            "valid": bool(source["entity_valid"][frame_index, 0]),
                            "confidence": float(
                                source["entity_confidence"][frame_index, 0]
                            ),
                            "lost": bool(source["entity_lost"][frame_index, 0]),
                            "mask_score": float(
                                source["entity_mask_score"][frame_index, 0]
                            ),
                        },
                        "projector": _jsonable(geometry.diagnostics),
                    }
                )
            _validate_output_arrays(arrays, frames=frames, resolution=resolution)
            relative = Path("predictions") / TASK / condition / f"episode_{episode_index:03d}.npz"
            prediction_sha = _save_npz(incomplete_root / relative, arrays)
            diagnostics_relative = (
                Path("diagnostics") / TASK / condition / f"episode_{episode_index:03d}.json"
            )
            diagnostics_payload = {
                "format": DIAGNOSTICS_FORMAT,
                "task": TASK,
                "condition": condition,
                "episode_index": episode_index,
                "frames": frames,
                "records": diagnostic_records,
            }
            diagnostics_bytes = _canonical_json_bytes(diagnostics_payload)
            diagnostics_path = incomplete_root / diagnostics_relative
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_bytes(diagnostics_bytes)
            diagnostics_sha = file_sha256(diagnostics_path)
            projector_status = np.concatenate(
                (
                    arrays["projector_valid"].astype(np.float32)[..., None],
                    arrays["projector_confidence"][..., None],
                ),
                axis=-1,
            )
            role_status = np.concatenate(
                (
                    arrays["role_valid"].astype(np.float32)[..., None],
                    arrays["role_confidence"][..., None],
                    arrays["role_lost"].astype(np.float32)[..., None],
                    arrays["role_mask_score"][..., None],
                ),
                axis=-1,
            )
            traces = {
                "role_mask_trace_sha256": _array_trace(arrays["role_masks"]),
                "keypoint_trace_sha256": _array_trace(arrays["keypoints_xy"]),
                "projector_status_trace_sha256": _array_trace(projector_status),
                "role_status_trace_sha256": _array_trace(role_status),
                "parser_runtime_trace_sha256": _array_trace(
                    arrays["parser_runtime_ms"]
                ),
                "diagnostics_trace_sha256": hashlib.sha256(
                    diagnostics_bytes
                ).hexdigest(),
            }
            if set(traces) != OUTPUT_TRACE_KEYS or traces[
                "diagnostics_trace_sha256"
            ] != diagnostics_sha:
                raise AssertionError("Internal temporal trace construction failed.")
            records.append(
                {
                    "episode_index": episode_index,
                    "frames": frames,
                    "entity_count": 1,
                    "role_count": 2,
                    "prediction_arrays": relative.as_posix(),
                    "prediction_arrays_sha256": prediction_sha,
                    "array_shapes": {
                        name: list(value.shape) for name, value in sorted(arrays.items())
                    },
                    "traces": traces,
                    "diagnostics_json": diagnostics_relative.as_posix(),
                    "diagnostics_json_sha256": diagnostics_sha,
                    "source_v1_prediction_arrays_sha256": source_record[
                        "prediction_arrays_sha256"
                    ],
                    "source_v1_entity_mask_trace_sha256": source_record["traces"][
                        "entity_mask_trace_sha256"
                    ],
                    "source_v1_entity_status_trace_sha256": source_record["traces"][
                        "entity_status_trace_sha256"
                    ],
                }
            )
            print(
                "OBJECT_GRAPH_TEMPORAL_V2_REPLAY_EPISODE",
                json.dumps(
                    {
                        "condition": condition,
                        "episode_index": episode_index,
                        "role_mask_trace_sha256": traces[
                            "role_mask_trace_sha256"
                        ],
                        "projector_status_trace_sha256": traces[
                            "projector_status_trace_sha256"
                        ],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        results[condition] = records

    inputs_after, _, _ = validate_backend_inputs(
        input_path, strict_counts=args.strict_counts
    )
    v1_after, paths_after = _validate_v1_manifest(
        v1_manifest_path,
        inputs=inputs_after,
        input_sha256=file_sha256(input_path),
        v1_graph=v1_graph,
        v1_graph_path=v1_graph_path,
    )
    if (
        inputs_after != inputs
        or file_sha256(input_path) != input_sha
        or v1_after != v1_payload
        or file_sha256(v1_manifest_path) != v1_manifest_sha
        or {
            key: (file_sha256(value[0]), value[1]["prediction_arrays_sha256"])
            for key, value in paths_after.items()
        }
        != {
            key: (file_sha256(value[0]), value[1]["prediction_arrays_sha256"])
            for key, value in v1_paths.items()
        }
        or file_sha256(v1_graph_path) != v1_graph_sha
        or file_sha256(temporal_graph_path) != temporal_graph_sha
        or file_sha256(support_path) != support_sha
        or _implementation_snapshot() != implementation_before
    ):
        raise RuntimeError("A temporal replay input or implementation changed during execution.")

    payload = {
        "format": FORMAT,
        "status": "complete",
        "backend": BACKEND,
        "task": TASK,
        "dataset_id": inputs["dataset_id"],
        "input_manifest_sha256": input_sha,
        "v1_backend_manifest_sha256": v1_manifest_sha,
        "roles": list(TASK_ROLES[TASK]),
        "graphs": {
            "v1": {
                "path": str(v1_graph_path),
                "file_sha256": v1_graph_sha,
                "graph": v1_graph.metadata(),
            },
            "temporal_v2": {
                "path": str(temporal_graph_path),
                "file_sha256": temporal_graph_sha,
                "graph": temporal_graph.metadata(),
                "tokenizer": tokenizer_metadata,
                "derivation": (
                    "same_frozen_graph_except_graph_name_and_"
                    "ordered_chain_projector_version_v2"
                ),
            },
        },
        "protocol": {
            "diagnostic_scope": "offline_causal_mask_parser_replay_v1",
            "source_backend": V1_BACKEND,
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
        },
        "backend_provenance": {
            "treatment": "ordered_chain_temporal_v2_mask_parser_replay",
            "implementation": implementation_before,
            "source_v1_backend_manifest": str(v1_manifest_path),
            "source_v1_backend_manifest_sha256": v1_manifest_sha,
            "source_worker_inputs": str(input_path),
            "source_worker_inputs_sha256": input_sha,
            "source_support_arrays": str(support_path),
            "source_support_arrays_sha256": support_sha,
            "v1_graph_file_sha256": v1_graph_sha,
            "v1_graph_semantic_sha256": v1_graph.graph_sha256,
            "temporal_graph_file_sha256": temporal_graph_sha,
            "temporal_graph_semantic_sha256": temporal_graph.graph_sha256,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wallclock_seconds": perf_counter() - started,
        },
        "results": results,
    }
    write_json(incomplete_root / "backend_predictions.json", payload)
    os.replace(incomplete_root, output_root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--v1-backend-manifest", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-counts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        "OBJECT_GRAPH_TEMPORAL_V2_REPLAY_COMPLETE",
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "manifest": str(args.output_root / "backend_predictions.json"),
                "controller_training_eligible": False,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
