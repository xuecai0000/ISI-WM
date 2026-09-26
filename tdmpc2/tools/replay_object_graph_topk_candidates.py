"""Stateless top-K ordered-chain replay over published v1 entity masks.

This diagnostic backend is intentionally narrower than the published v1
object-graph backend.  It consumes only the current Acrobot entity mask, its
current ``entity_valid``/``entity_lost`` bits, and the six frozen support
indexed masks.  It
does not decode episode RGB and does not read ground truth, simulator state,
actions, rewards, descriptors, or tracker appearance.  Episode entity arrays
are decoded up front for strict artifact validation, but the generator is
called with exactly one current mask/status and receives no previous or future
frame.

Slot zero is permanently reserved for the exact current-frame v1 parser.
Consequently K=1 is the published v1 parser even when that parser fails;
additional stateless physical hypotheses may only occupy slots 1..3.  The
backend emits compact poses and diagnostics rather than materialising large
role-mask tensors.  A privileged scorer can deterministically reconstruct a
candidate partition from a sealed pose and its sealed source entity mask.
"""

from __future__ import annotations

import argparse
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
    validate_backend_inputs,
    write_json,
)
from tdmpc2.perception.ordered_chain_topk import (  # noqa: E402
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES,
    OrderedChainTopKGenerator,
)
from tdmpc2.perception.support_conditioned_object_graph import (  # noqa: E402
    load_object_graph,
)
from tdmpc2.tools.replay_object_graph_temporal_v2 import (  # noqa: E402
    TASK,
    V1_ARRAY_KEYS,
    _array_trace,
    _load_support,
    _regular_file,
    _validate_v1_manifest,
)


FORMAT = "object_graph_ordered_chain_topk_replay_predictions_v1"
BACKEND = "object_graph_ordered_chain_topk_replay"
PROTOCOL = "gt_free_stateless_current_entity_mask_topk_replay_v1"
EXPECTED_MAX_CANDIDATES = 4
EXPECTED_ROLE_COUNT = 2
EXPECTED_KEYPOINT_COUNT = 3
PADDING_COST = np.float32(np.finfo(np.float32).max)

FAILURE_CODES = {
    "none": 0,
    "entity_unavailable": 1,
    "entity_too_small": 2,
    "entity_area_implausible": 3,
    "no_current_mask_supported_candidate": 4,
    "unknown": 255,
}

COMPONENT_DIAGNOSTICS = {
    "candidate_coverage": "candidate_coverage",
    "candidate_render_iou": "candidate_render_iou",
    "candidate_branch_fraction": "candidate_branch_fraction",
    "candidate_disconnected_fraction": "candidate_disconnected_fraction",
    "candidate_pixel_fit_cost": "candidate_pixel_fit_cost",
    "candidate_segment_fit_cost": "candidate_segment_fit_cost",
    "candidate_length_cost": "candidate_length_cost",
    "candidate_area_cost": "candidate_area_cost",
    "candidate_root_cost": "candidate_root_cost",
}

OUTPUT_ARRAY_KEYS = {
    "poses_xy",
    "candidate_valid",
    "candidate_cost",
    "candidate_weight",
    "candidate_confidence",
    "candidate_source_code",
    "candidate_count",
    "parser_runtime_ms",
    "best_second_cost_margin",
    "weight_entropy",
    "normalized_weight_entropy",
    "weighted_pose_dispersion_px",
    "fit_uncertainty",
    "failure_code",
    *COMPONENT_DIAGNOSTICS.values(),
}
OUTPUT_TRACE_KEYS = frozenset(OUTPUT_ARRAY_KEYS)
MANIFEST_KEYS = {
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
RESULT_RECORD_KEYS = {
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


def _implementation_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_DIR / "perception" / "ordered_chain_topk.py",
        PROJECT_DIR / "perception" / "support_conditioned_object_graph.py",
        PROJECT_DIR / "tools" / "replay_object_graph_temporal_v2.py",
        PROJECT_DIR / "common" / "unified_vos.py",
    )
    result: dict[str, Any] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Replay implementation must be a regular file: {path}")
        result[path.relative_to(REPO_DIR).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def _load_v1_graph(path: Path):
    graph_path = _regular_file(path, "v1 graph")
    graph = load_object_graph(graph_path)
    roles = TASK_ROLES[TASK]
    if (
        graph.task != TASK
        or graph.source_roles != roles
        or graph.semantic_roles != roles
        or graph.graph_name != "acrobot_swingup_whole_entity_ordered_chain_v1"
        or len(graph.entities) != 1
        or graph.entities[0].source_roles != roles
        or graph.entities[0].projector.roles != roles
        or graph.entities[0].projector.type != "ordered_chain_segments_v1"
    ):
        raise ValueError("The v1 graph is not the published two-link Acrobot graph.")
    return graph, graph_path


def _load_v1_mask_status(
    path: Path,
    record: dict[str, Any],
    *,
    frames: int,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode exactly three current-frame source arrays and validate their binding."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != V1_ARRAY_KEYS:
            raise ValueError("Published v1 prediction NPZ schema changed.")
        entity_masks = np.ascontiguousarray(archive["entity_masks"])
        entity_valid = np.ascontiguousarray(archive["entity_valid"])
        entity_lost = np.ascontiguousarray(archive["entity_lost"])
    if (
        entity_masks.shape != (frames, 1, resolution, resolution)
        or entity_masks.dtype != np.bool_
    ):
        raise ValueError("Published v1 entity_masks schema changed.")
    for name, value in (
        ("entity_valid", entity_valid),
        ("entity_lost", entity_lost),
    ):
        if value.shape != (frames, 1) or value.dtype != np.bool_:
            raise ValueError(f"Published v1 {name} schema changed.")
    nonempty = entity_masks.reshape(frames, 1, -1).any(axis=2)
    if np.any(entity_valid & (entity_lost | ~nonempty)):
        raise ValueError(
            "Published v1 entity_valid exceeds current non-lost, nonempty status."
        )
    if _array_trace(entity_masks) != record["traces"]["entity_mask_trace_sha256"]:
        raise ValueError("Published v1 decoded entity-mask trace changed.")
    return entity_masks, entity_valid, entity_lost


def _empty_arrays(*, frames: int) -> dict[str, np.ndarray]:
    candidate_shape = (frames, EXPECTED_MAX_CANDIDATES)
    arrays: dict[str, np.ndarray] = {
        "poses_xy": np.zeros(
            (frames, EXPECTED_MAX_CANDIDATES, EXPECTED_KEYPOINT_COUNT, 2),
            dtype=np.float32,
        ),
        "candidate_valid": np.zeros(candidate_shape, dtype=np.bool_),
        "candidate_cost": np.full(candidate_shape, PADDING_COST, dtype=np.float32),
        "candidate_weight": np.zeros(candidate_shape, dtype=np.float32),
        "candidate_confidence": np.zeros(candidate_shape, dtype=np.float32),
        "candidate_source_code": np.zeros(candidate_shape, dtype=np.uint8),
        "candidate_count": np.zeros(frames, dtype=np.uint8),
        "parser_runtime_ms": np.zeros(frames, dtype=np.float64),
        "best_second_cost_margin": np.zeros(frames, dtype=np.float32),
        "weight_entropy": np.zeros(frames, dtype=np.float32),
        "normalized_weight_entropy": np.zeros(frames, dtype=np.float32),
        "weighted_pose_dispersion_px": np.zeros(frames, dtype=np.float32),
        "fit_uncertainty": np.ones(frames, dtype=np.float32),
        "failure_code": np.zeros(frames, dtype=np.uint8),
    }
    zero_padded = {
        "candidate_coverage",
        "candidate_render_iou",
        "candidate_branch_fraction",
        "candidate_disconnected_fraction",
    }
    for name in COMPONENT_DIAGNOSTICS.values():
        fill = np.float32(0.0) if name in zero_padded else PADDING_COST
        arrays[name] = np.full(candidate_shape, fill, dtype=np.float32)
    return arrays


def _failure_code(value: Any) -> np.uint8:
    if value is None:
        return np.uint8(FAILURE_CODES["none"])
    if not isinstance(value, str):
        return np.uint8(FAILURE_CODES["unknown"])
    return np.uint8(FAILURE_CODES.get(value, FAILURE_CODES["unknown"]))


def _copy_frame(arrays: dict[str, np.ndarray], frame_index: int, frame: Any) -> None:
    expected_shapes = {
        "poses_xy": (EXPECTED_MAX_CANDIDATES, EXPECTED_KEYPOINT_COUNT, 2),
        "valid": (EXPECTED_MAX_CANDIDATES,),
        "costs": (EXPECTED_MAX_CANDIDATES,),
        "weights": (EXPECTED_MAX_CANDIDATES,),
        "confidence": (EXPECTED_MAX_CANDIDATES,),
        "source_codes": (EXPECTED_MAX_CANDIDATES,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(frame, name)
        if value.shape != shape:
            raise RuntimeError(f"Top-K frame {name} shape changed: {value.shape}.")
    if (
        frame.poses_xy.dtype != np.float32
        or frame.valid.dtype != np.bool_
        or frame.costs.dtype != np.float32
        or frame.weights.dtype != np.float32
        or frame.confidence.dtype != np.float32
        or frame.source_codes.dtype != np.uint8
        or type(frame.candidate_count) is not int
        or not isinstance(frame.diagnostics, dict)
        or not math.isfinite(float(frame.runtime_ms))
        or float(frame.runtime_ms) <= 0.0
    ):
        raise RuntimeError("Top-K frame dtype/runtime contract changed.")

    arrays["poses_xy"][frame_index] = frame.poses_xy
    arrays["candidate_valid"][frame_index] = frame.valid
    arrays["candidate_cost"][frame_index] = frame.costs
    arrays["candidate_weight"][frame_index] = frame.weights
    arrays["candidate_confidence"][frame_index] = frame.confidence
    arrays["candidate_source_code"][frame_index] = frame.source_codes
    arrays["candidate_count"][frame_index] = np.uint8(frame.candidate_count)
    arrays["parser_runtime_ms"][frame_index] = float(frame.runtime_ms)
    arrays["best_second_cost_margin"][frame_index] = np.float32(
        frame.best_second_cost_margin
    )
    arrays["weight_entropy"][frame_index] = np.float32(frame.weight_entropy)
    arrays["normalized_weight_entropy"][frame_index] = np.float32(
        frame.normalized_weight_entropy
    )
    arrays["weighted_pose_dispersion_px"][frame_index] = np.float32(
        frame.weighted_pose_dispersion_px
    )
    arrays["fit_uncertainty"][frame_index] = np.float32(frame.fit_uncertainty)
    arrays["failure_code"][frame_index] = _failure_code(
        frame.diagnostics.get("failure")
    )

    active_slots = np.flatnonzero(frame.valid)
    diagnostic_slots = frame.diagnostics.get("candidate_slots")
    if (
        not isinstance(diagnostic_slots, list)
        or diagnostic_slots != [int(value) for value in active_slots]
    ):
        raise RuntimeError("Top-K candidate slot diagnostics disagree with valid slots.")
    sources = frame.diagnostics.get("candidate_sources")
    if not isinstance(sources, list) or len(sources) != len(active_slots):
        raise RuntimeError("Top-K candidate source diagnostics lost slot alignment.")
    inverse_source_codes = {name: int(code) for name, code in SOURCE_CODES.items()}
    for item_index, slot in enumerate(active_slots):
        source = sources[item_index]
        if (
            source not in inverse_source_codes
            or int(frame.source_codes[slot]) != inverse_source_codes[source]
        ):
            raise RuntimeError("Top-K source name/code diagnostics disagree.")
    for diagnostic_name, array_name in COMPONENT_DIAGNOSTICS.items():
        values = frame.diagnostics.get(diagnostic_name)
        if not isinstance(values, list) or len(values) != len(active_slots):
            raise RuntimeError(
                f"Top-K {diagnostic_name} diagnostics lost slot alignment."
            )
        if values:
            numeric = np.asarray(values, dtype=np.float32)
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"Top-K {diagnostic_name} is non-finite.")
            arrays[array_name][frame_index, active_slots] = numeric


def _expected_schema(*, frames: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    candidate_shape = (frames, EXPECTED_MAX_CANDIDATES)
    schema: dict[str, tuple[tuple[int, ...], Any]] = {
        "poses_xy": (
            (frames, EXPECTED_MAX_CANDIDATES, EXPECTED_KEYPOINT_COUNT, 2),
            np.float32,
        ),
        "candidate_valid": (candidate_shape, np.bool_),
        "candidate_cost": (candidate_shape, np.float32),
        "candidate_weight": (candidate_shape, np.float32),
        "candidate_confidence": (candidate_shape, np.float32),
        "candidate_source_code": (candidate_shape, np.uint8),
        "candidate_count": ((frames,), np.uint8),
        "parser_runtime_ms": ((frames,), np.float64),
        "best_second_cost_margin": ((frames,), np.float32),
        "weight_entropy": ((frames,), np.float32),
        "normalized_weight_entropy": ((frames,), np.float32),
        "weighted_pose_dispersion_px": ((frames,), np.float32),
        "fit_uncertainty": ((frames,), np.float32),
        "failure_code": ((frames,), np.uint8),
    }
    for name in COMPONENT_DIAGNOSTICS.values():
        schema[name] = (candidate_shape, np.float32)
    return schema


def _validate_output_arrays(arrays: dict[str, np.ndarray], *, frames: int) -> None:
    if set(arrays) != OUTPUT_ARRAY_KEYS:
        raise RuntimeError("Top-K replay output array schema changed.")
    for name, (shape, dtype) in _expected_schema(frames=frames).items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"Top-K replay {name} is {value.shape}/{value.dtype}, "
                f"expected {shape}/{dtype}."
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise RuntimeError(f"Top-K replay {name} contains non-finite values.")

    valid = arrays["candidate_valid"]
    count = valid.sum(axis=1).astype(np.uint8)
    if not np.array_equal(count, arrays["candidate_count"]):
        raise RuntimeError("candidate_count is not exactly sum(candidate_valid).")
    if np.any(valid[:, 1:][:, :-1] < valid[:, 1:][:, 1:]):
        raise RuntimeError("Alternative slots must be left-packed within slots 1..3.")
    source = arrays["candidate_source_code"]
    anchor_code = int(SOURCE_CODES["v1_anchor"])
    if np.any(valid[:, 0] & (source[:, 0] != anchor_code)):
        raise RuntimeError("A valid reserved slot zero must be the exact v1 anchor.")
    if np.any(valid[:, 1:] & ((source[:, 1:] == 0) | (source[:, 1:] == anchor_code))):
        raise RuntimeError("Alternative slots contain padding or a duplicate v1 anchor.")
    if np.any(source[~valid] != 0):
        raise RuntimeError("Invalid candidate slots must use the padding source code.")
    allowed_codes = np.asarray(sorted(set(int(v) for v in SOURCE_CODES.values())))
    if not np.isin(source, allowed_codes).all():
        raise RuntimeError("Candidate source code is unknown.")

    poses = arrays["poses_xy"]
    costs = arrays["candidate_cost"]
    weights = arrays["candidate_weight"]
    confidence = arrays["candidate_confidence"]
    if np.any(poses[~valid] != 0.0):
        raise RuntimeError("Invalid candidate poses must be zero padded.")
    if np.any(costs[~valid] != PADDING_COST):
        raise RuntimeError("Invalid candidate costs must use float32-max padding.")
    if np.any(weights[~valid] != 0.0) or np.any(confidence[~valid] != 0.0):
        raise RuntimeError("Invalid candidate weight/confidence must be zero padded.")
    if np.any(costs[valid] < 0.0):
        raise RuntimeError("Valid candidate costs cannot be negative.")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise RuntimeError("Candidate weights escaped [0,1].")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise RuntimeError("Candidate confidence escaped [0,1].")
    sums = weights.sum(axis=1)
    if not np.allclose(sums[count > 0], 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError("Valid candidate weights do not sum to one.")
    if np.any(sums[count == 0] != 0.0):
        raise RuntimeError("Empty frames must have zero candidate weight.")

    if np.any(arrays["parser_runtime_ms"] <= 0.0):
        raise RuntimeError("Parser runtime must be finite and positive.")
    if np.any(arrays["best_second_cost_margin"] < 0.0):
        raise RuntimeError("Best/second candidate cost margin cannot be negative.")
    for name in ("weight_entropy", "normalized_weight_entropy"):
        if np.any(arrays[name] < 0.0):
            raise RuntimeError(f"{name} cannot be negative.")
    if np.any(arrays["normalized_weight_entropy"] > 1.0 + 2e-6):
        raise RuntimeError("Normalized candidate entropy escaped [0,1].")
    if np.any(arrays["weighted_pose_dispersion_px"] < 0.0):
        raise RuntimeError("Weighted pose dispersion cannot be negative.")
    if np.any(
        (arrays["fit_uncertainty"] < 0.0)
        | (arrays["fit_uncertainty"] > 1.0)
    ):
        raise RuntimeError("Fit uncertainty escaped [0,1].")

    zero_padded = {
        "candidate_coverage",
        "candidate_render_iou",
        "candidate_branch_fraction",
        "candidate_disconnected_fraction",
    }
    for name in COMPONENT_DIAGNOSTICS.values():
        value = arrays[name]
        if name in zero_padded:
            if np.any((value < 0.0) | (value > 1.0)):
                raise RuntimeError(f"{name} escaped [0,1].")
            if np.any(value[~valid] != 0.0):
                raise RuntimeError(f"Invalid {name} slots must be zero padded.")
        else:
            if np.any(value[valid] < 0.0):
                raise RuntimeError(f"Valid {name} cannot be negative.")
            if np.any(value[~valid] != PADDING_COST):
                raise RuntimeError(
                    f"Invalid {name} slots must use float32-max padding."
                )

    for frame_index in range(frames):
        active = np.flatnonzero(valid[frame_index])
        number = len(active)
        if number == 0:
            if (
                arrays["best_second_cost_margin"][frame_index] != 0.0
                or arrays["weight_entropy"][frame_index] != 0.0
                or arrays["normalized_weight_entropy"][frame_index] != 0.0
                or arrays["weighted_pose_dispersion_px"][frame_index] != 0.0
                or arrays["fit_uncertainty"][frame_index] != 1.0
            ):
                raise RuntimeError("Empty-frame uncertainty convention changed.")
            continue
        expected_uncertainty = 1.0 - float(confidence[frame_index, active].max())
        if not math.isclose(
            float(arrays["fit_uncertainty"][frame_index]),
            expected_uncertainty,
            abs_tol=2e-6,
            rel_tol=0.0,
        ):
            raise RuntimeError("Fit uncertainty is inconsistent with confidence.")
        probabilities = weights[frame_index, active].astype(np.float64)
        positive = probabilities[probabilities > 0.0]
        expected_entropy = float(-np.sum(positive * np.log(positive)))
        expected_normalized = (
            expected_entropy / math.log(number) if number > 1 else 0.0
        )
        if not math.isclose(
            float(arrays["weight_entropy"][frame_index]),
            expected_entropy,
            abs_tol=3e-6,
            rel_tol=0.0,
        ) or not math.isclose(
            float(arrays["normalized_weight_entropy"][frame_index]),
            expected_normalized,
            abs_tol=3e-6,
            rel_tol=0.0,
        ):
            raise RuntimeError("Candidate entropy is inconsistent with weights.")
        if number == 1:
            if arrays["best_second_cost_margin"][frame_index] != 0.0:
                raise RuntimeError("A singleton candidate set must use zero margin.")
        else:
            ranked = np.sort(costs[frame_index, active].astype(np.float64))
            expected_margin = float(ranked[1] - ranked[0])
            if not math.isclose(
                float(arrays["best_second_cost_margin"][frame_index]),
                expected_margin,
                abs_tol=3e-5,
                rel_tol=2e-6,
            ):
                raise RuntimeError("Best/second cost margin is inconsistent with costs.")
        active_poses = poses[frame_index, active].astype(np.float64)
        mean_pose = np.sum(probabilities[:, None, None] * active_poses, axis=0)
        expected_dispersion = float(
            np.sum(
                probabilities
                * np.linalg.norm(active_poses - mean_pose[None], axis=2).mean(axis=1)
            )
        )
        if not math.isclose(
            float(arrays["weighted_pose_dispersion_px"][frame_index]),
            expected_dispersion,
            abs_tol=3e-5,
            rel_tol=2e-6,
        ):
            raise RuntimeError("Weighted pose dispersion is inconsistent with poses.")


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{name: np.ascontiguousarray(arrays[name]) for name in sorted(arrays)}
    )
    return file_sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "Stateless ordered-chain top-K replay is CPU-only; set "
            "CUDA_VISIBLE_DEVICES to the empty string."
        )
    if EXPECTED_MAX_CANDIDATES != MAX_CANDIDATES:
        raise RuntimeError("The fixed max-K backend schema changed.")

    input_path = _regular_file(Path(args.inputs), "GT-free worker input manifest")
    input_sha = file_sha256(input_path)
    inputs, support_paths, _ = validate_backend_inputs(
        input_path, strict_counts=args.strict_counts
    )
    graph, graph_path = _load_v1_graph(Path(args.v1_graph))
    graph_file_sha = file_sha256(graph_path)
    manifest_path = _regular_file(
        Path(args.v1_backend_manifest), "published v1 backend manifest"
    )
    manifest_sha = file_sha256(manifest_path)
    v1_payload, v1_paths = _validate_v1_manifest(
        manifest_path,
        inputs=inputs,
        input_sha256=input_sha,
        v1_graph=graph,
        v1_graph_path=graph_path,
    )
    resolution = int(inputs["resolution"])
    frames = int(inputs["counts"]["frames_per_episode"])
    episodes = int(inputs["counts"]["episodes"])
    support_path = support_paths[TASK]
    support_sha = file_sha256(support_path)
    support_masks = _load_support(support_path, resolution=resolution)
    implementation_before = _implementation_snapshot()

    output_root = Path(args.output_root).expanduser().resolve()
    incomplete_root = output_root.with_name(output_root.name + ".incomplete")
    if output_root.exists() or incomplete_root.exists():
        raise FileExistsError(output_root if output_root.exists() else incomplete_root)
    if not output_root.parent.is_dir():
        raise FileNotFoundError(output_root.parent)
    incomplete_root.mkdir(exist_ok=False)

    generator = OrderedChainTopKGenerator(
        graph, support_masks, max_candidates=EXPECTED_MAX_CANDIDATES
    )
    generator_metadata = generator.metadata()
    if (
        generator_metadata.get("format") != GENERATOR_FORMAT
        or generator_metadata.get("protocol") != GENERATOR_PROTOCOL
        or generator_metadata.get("max_candidates") != EXPECTED_MAX_CANDIDATES
        or generator_metadata.get("episode_state") is not False
        or generator_metadata.get("future_frames") is not False
        or generator_metadata.get("ground_truth_input") is not False
    ):
        raise RuntimeError("Top-K generator metadata contract changed.")

    started = perf_counter()
    results: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        records = []
        for episode_index in range(episodes):
            source_path, source_record = v1_paths[(condition, episode_index)]
            entity_masks, entity_valid, entity_lost = _load_v1_mask_status(
                source_path,
                source_record,
                frames=frames,
                resolution=resolution,
            )
            arrays = _empty_arrays(frames=frames)
            for frame_index in range(frames):
                current_mask = entity_masks[frame_index, 0]
                current_available = bool(entity_valid[frame_index, 0])
                frame = generator.project(
                    entity_mask=current_mask,
                    entity_available=current_available,
                )
                if frame.role_names != TASK_ROLES[TASK]:
                    raise RuntimeError("Top-K frame semantic role order changed.")
                _copy_frame(arrays, frame_index, frame)
            _validate_output_arrays(arrays, frames=frames)

            relative = (
                Path("predictions")
                / TASK
                / condition
                / f"episode_{episode_index:03d}.npz"
            )
            prediction_sha = _save_npz(incomplete_root / relative, arrays)
            traces = {
                name: _array_trace(arrays[name]) for name in sorted(OUTPUT_ARRAY_KEYS)
            }
            if set(traces) != OUTPUT_TRACE_KEYS:
                raise AssertionError("Internal top-K trace construction failed.")
            record = {
                "episode_index": episode_index,
                "frames": frames,
                "max_candidates": EXPECTED_MAX_CANDIDATES,
                "role_count": EXPECTED_ROLE_COUNT,
                "keypoint_count": EXPECTED_KEYPOINT_COUNT,
                "prediction_arrays": relative.as_posix(),
                "prediction_arrays_sha256": prediction_sha,
                "array_shapes": {
                    name: list(arrays[name].shape)
                    for name in sorted(OUTPUT_ARRAY_KEYS)
                },
                "array_dtypes": {
                    name: str(arrays[name].dtype)
                    for name in sorted(OUTPUT_ARRAY_KEYS)
                },
                "array_traces_sha256": traces,
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
            if set(record) != RESULT_RECORD_KEYS:
                raise AssertionError("Internal top-K result record schema changed.")
            records.append(record)
            print(
                "OBJECT_GRAPH_TOPK_REPLAY_EPISODE",
                json.dumps(
                    {
                        "condition": condition,
                        "episode_index": episode_index,
                        "poses_trace_sha256": traces["poses_xy"],
                        "candidate_valid_trace_sha256": traces["candidate_valid"],
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
        manifest_path,
        inputs=inputs_after,
        input_sha256=file_sha256(input_path),
        v1_graph=graph,
        v1_graph_path=graph_path,
    )
    before_files = {
        key: (file_sha256(value[0]), value[1]["prediction_arrays_sha256"])
        for key, value in v1_paths.items()
    }
    after_files = {
        key: (file_sha256(value[0]), value[1]["prediction_arrays_sha256"])
        for key, value in paths_after.items()
    }
    if (
        inputs_after != inputs
        or file_sha256(input_path) != input_sha
        or v1_after != v1_payload
        or file_sha256(manifest_path) != manifest_sha
        or after_files != before_files
        or file_sha256(graph_path) != graph_file_sha
        or file_sha256(support_path) != support_sha
        or _implementation_snapshot() != implementation_before
    ):
        raise RuntimeError("A top-K replay input or implementation changed during execution.")

    payload = {
        "format": FORMAT,
        "status": "complete",
        "backend": BACKEND,
        "task": TASK,
        "dataset_id": inputs["dataset_id"],
        "input_manifest_sha256": input_sha,
        "v1_backend_manifest_sha256": manifest_sha,
        "roles": list(TASK_ROLES[TASK]),
        "max_candidates": EXPECTED_MAX_CANDIDATES,
        "graph": {
            "path": str(graph_path),
            "file_sha256": graph_file_sha,
            "graph": graph.metadata(),
        },
        "generator": generator_metadata,
        "output_schema": {
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
        },
        "protocol": {
            "format": PROTOCOL,
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
            "max_k_generated_once": EXPECTED_MAX_CANDIDATES,
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
        },
        "backend_provenance": {
            "treatment": "stateless_current_mask_ordered_chain_topk_replay",
            "implementation": implementation_before,
            "source_v1_backend_manifest": str(manifest_path),
            "source_v1_backend_manifest_sha256": manifest_sha,
            "source_worker_inputs": str(input_path),
            "source_worker_inputs_sha256": input_sha,
            "source_support_arrays": str(support_path),
            "source_support_arrays_sha256": support_sha,
            "v1_graph_file_sha256": graph_file_sha,
            "v1_graph_semantic_sha256": graph.graph_sha256,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wallclock_seconds": perf_counter() - started,
        },
        "results": results,
    }
    if set(payload) != MANIFEST_KEYS:
        raise AssertionError("Internal top-K manifest schema changed.")
    write_json(incomplete_root / "backend_predictions.json", payload)
    os.replace(incomplete_root, output_root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--v1-backend-manifest", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-counts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        "OBJECT_GRAPH_TOPK_REPLAY_COMPLETE",
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
