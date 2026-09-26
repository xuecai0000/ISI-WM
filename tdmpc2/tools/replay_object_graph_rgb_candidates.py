"""GPU replay for current-frame RGB ordered-chain candidate diagnostics.

This is a deliberately single-task Acrobot development backend.  It compares
two otherwise identical current-frame arms: real RGB and the RGB generator's
declared deterministic spatial-shuffle control.  In both arms slot zero is the
exact published v1 current-mask ordered-chain anchor; slot one is the frozen
DINOv2 RGB candidate.  The backend emits compact poses and sealed rendering
widths, never controller tokens or materialised role masks.

Episode inputs contain RGB only.  The generator receives one current RGB frame,
one current published-v1 entity mask, and one current availability bit.  It
never receives previous/future frames, simulator state, ground truth, actions,
or rewards.  Published v1 confidences are decoded only to verify the immutable
status trace; they are not generator inputs.  Candidate weights/confidences are
diagnostic, non-calibrated values and do not authorize controller training.
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
    validate_backend_inputs,
    write_json,
)
TASK = "acrobot-swingup"
FORMAT = "object_graph_ordered_chain_rgb_candidate_replay_predictions_v1"
BACKEND = "object_graph_ordered_chain_rgb_candidate_replay"
PROTOCOL = "gt_free_current_rgb_ordered_chain_candidate_development_v1"
ARMS = ("real_rgb", "spatial_shuffle")
MAX_CANDIDATES = 2
ROLE_COUNT = 2
KEYPOINT_COUNT = 3
DEFAULT_DINO_MODEL = "dinov2_vits14_reg"
DEFAULT_DINO_INPUT_SIZE = 224
DINO_EXTRACTOR_METADATA_KEYS = frozenset(
    {
        "format",
        "repo",
        "checkpoint",
        "model_name",
        "input_size",
        "device",
        "frozen_parameters",
        "evaluation_mode",
        "torch_hub_source_local",
        "pretrained_constructor_download",
        "network_isolation_enforced",
    }
)
PADDING_COST = np.float32(np.finfo(np.float32).max)

# Frozen independently from the core module so an accidental semantic change in
# either side fails at construction rather than silently changing old artifacts.
SOURCE_CODES = {
    "padding": 0,
    "v1_anchor": 1,
    "rgb_ik_positive": 2,
    "rgb_ik_negative": 3,
}

V1_ARRAY_KEYS = frozenset(
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
    }
)

OUTPUT_ARRAY_KEYS = frozenset(
    {
        "poses_xy",
        "link_half_widths_px",
        "candidate_valid",
        "candidate_cost",
        "candidate_weight",
        "candidate_confidence",
        "rgb_evidence_score",
        "mask_geometry_score",
        "candidate_source_code",
        "roi_xyxy",
        "parser_runtime_ms",
    }
)

GENERATOR_METADATA_KEYS = frozenset(
    {
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
)

MANIFEST_KEYS = frozenset(
    {
        "format",
        "status",
        "backend",
        "task",
        "development_scope",
        "dataset_id",
        "input_manifest_sha256",
        "v1_backend_manifest_sha256",
        "roles",
        "arms",
        "max_candidates",
        "graph",
        "support",
        "dino",
        "generator",
        "output_schema",
        "protocol",
        "backend_provenance",
        "results",
    }
)

RESULT_RECORD_KEYS = frozenset(
    {
        "arm",
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
        "source_rgb_arrays_sha256",
        "source_rgb_trace_sha256",
        "decoded_rgb_array_trace_sha256",
        "source_v1_prediction_arrays_sha256",
        "source_v1_entity_mask_trace_sha256",
        "source_v1_entity_status_trace_sha256",
        "source_v1_keypoint_trace_sha256",
        "source_v1_role_status_trace_sha256",
    }
)


def _array_trace(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    result = candidate.resolve(strict=True)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _validate_v1_manifest(*args: Any, **kwargs: Any):
    # Lazy so ``--help`` does not import the perception package or PyTorch.
    from tdmpc2.tools.replay_object_graph_temporal_v2 import (
        V1_ARRAY_KEYS as frozen_v1_keys,
        _validate_v1_manifest as validate,
    )

    if frozenset(frozen_v1_keys) != V1_ARRAY_KEYS:
        raise RuntimeError("The published v1 NPZ key contract changed.")
    return validate(*args, **kwargs)


def _regular_directory(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {component}")
    result = candidate.resolve(strict=True)
    if not result.is_dir():
        raise NotADirectoryError(result)
    return result


def _python_tree_snapshot(root: Path) -> dict[str, Any]:
    """Bind every regular Python source file below an external repository."""
    root = _regular_directory(root, "DINOv2 repository")
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(
            "DINOv2 repository tree must not contain symlinks; first is "
            f"{symlinks[0]}"
        )
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise ValueError(f"DINOv2 Python source tree is empty: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"DINOv2 Python source must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"DINOv2 Python source escaped repository: {path}")
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": records,
    }


def _implementation_snapshot() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_DIR / "perception" / "ordered_chain_rgb.py",
        PROJECT_DIR / "perception" / "ordered_chain_topk.py",
        PROJECT_DIR / "perception" / "support_conditioned_object_graph.py",
        PROJECT_DIR / "tools" / "replay_object_graph_temporal_v2.py",
        PROJECT_DIR / "common" / "unified_vos.py",
    )
    result: dict[str, Any] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"RGB replay implementation must be regular: {path}")
        result[path.relative_to(REPO_DIR).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def _load_v1_graph(path: Path):
    from tdmpc2.perception.support_conditioned_object_graph import load_object_graph

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
        raise ValueError("The selected graph is not the published two-link Acrobot v1 graph.")
    return graph, graph_path


def _load_support(
    path: Path, *, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"rgb", "indexed_masks"}:
            raise ValueError("Frozen support NPZ must contain exactly rgb/indexed_masks.")
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
    labels = set(int(value) for value in np.unique(masks))
    if not labels.issubset({0, 1, 2}) or not {1, 2}.issubset(labels):
        raise ValueError("Frozen Acrobot support labels must be background/roles 1/2.")
    return rgb, masks


def _episode_rgb_trace(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    for frame in rgb:
        value = np.ascontiguousarray(frame)
        digest.update(b"rgb")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_episode_rgb(
    path: Path,
    *,
    frames: int,
    resolution: int,
    expected_trace: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"rgb"}:
            raise ValueError("GT-free episode NPZ must contain exactly {'rgb'}.")
        rgb = np.ascontiguousarray(archive["rgb"])
    if rgb.shape != (frames, resolution, resolution, 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"GT-free episode RGB changed: {rgb.shape}/{rgb.dtype}; expected "
            f"{(frames, resolution, resolution, 3)}/uint8."
        )
    if _episode_rgb_trace(rgb) != expected_trace:
        raise ValueError("Decoded episode RGB trace disagrees with worker manifest.")
    return rgb


def _load_v1_arrays(
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
        "keypoints_xy",
        "role_valid",
        "role_confidence",
        "role_lost",
        "role_mask_score",
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
        "keypoints_xy": ((frames, ROLE_COUNT, 2, 2), np.float32),
        "role_valid": ((frames, ROLE_COUNT), np.bool_),
        "role_confidence": ((frames, ROLE_COUNT), np.float32),
        "role_lost": ((frames, ROLE_COUNT), np.bool_),
        "role_mask_score": ((frames, ROLE_COUNT), np.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Published v1 {name} schema changed.")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Published v1 {name} contains non-finite values.")
    for name in (
        "entity_confidence",
        "entity_mask_score",
        "role_confidence",
        "role_mask_score",
    ):
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise ValueError(f"Published v1 {name} escaped [0,1].")
    nonempty = arrays["entity_masks"].reshape(frames, 1, -1).any(axis=2)
    if np.any(arrays["entity_valid"] & (arrays["entity_lost"] | ~nonempty)):
        raise ValueError(
            "Published v1 entity validity exceeds current non-lost, nonempty status."
        )
    if np.any(arrays["role_valid"] & arrays["role_lost"]):
        raise ValueError("Published v1 role cannot be valid and lost simultaneously.")
    if np.any(arrays["role_valid"] & ~arrays["entity_valid"]):
        raise ValueError("Published v1 role validity exceeds entity validity.")

    entity_status = np.concatenate(
        (
            arrays["entity_valid"].astype(np.float32)[..., None],
            arrays["entity_confidence"][..., None],
            arrays["entity_lost"].astype(np.float32)[..., None],
            arrays["entity_mask_score"][..., None],
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
    expected_traces = {
        "entity_masks": record["traces"]["entity_mask_trace_sha256"],
        "keypoints_xy": record["traces"]["keypoint_trace_sha256"],
        "entity_status": record["traces"]["entity_status_trace_sha256"],
        "role_status": record["traces"]["role_status_trace_sha256"],
    }
    observed_traces = {
        "entity_masks": _array_trace(arrays["entity_masks"]),
        "keypoints_xy": _array_trace(arrays["keypoints_xy"]),
        "entity_status": _array_trace(entity_status),
        "role_status": _array_trace(role_status),
    }
    if observed_traces != expected_traces:
        raise ValueError("Published v1 decoded mask/keypoint/status traces changed.")
    return arrays


def _v1_pose_and_valid(arrays: dict[str, np.ndarray], frame_index: int):
    keypoints = arrays["keypoints_xy"][frame_index]
    pose = np.ascontiguousarray(
        np.vstack((keypoints[0, 0], keypoints[:, 1])), dtype=np.float32
    )
    valid = bool(arrays["role_valid"][frame_index].all())
    return pose, valid


def _empty_arrays(*, frames: int) -> dict[str, np.ndarray]:
    candidate_shape = (frames, MAX_CANDIDATES)
    return {
        "poses_xy": np.zeros(
            (frames, MAX_CANDIDATES, KEYPOINT_COUNT, 2), dtype=np.float32
        ),
        "link_half_widths_px": np.zeros(
            (frames, MAX_CANDIDATES, ROLE_COUNT), dtype=np.float32
        ),
        "candidate_valid": np.zeros(candidate_shape, dtype=np.bool_),
        "candidate_cost": np.full(candidate_shape, PADDING_COST, dtype=np.float32),
        "candidate_weight": np.zeros(candidate_shape, dtype=np.float32),
        "candidate_confidence": np.zeros(candidate_shape, dtype=np.float32),
        "rgb_evidence_score": np.zeros(candidate_shape, dtype=np.float32),
        "mask_geometry_score": np.zeros(candidate_shape, dtype=np.float32),
        "candidate_source_code": np.zeros(candidate_shape, dtype=np.uint8),
        "roi_xyxy": np.zeros((frames, 4), dtype=np.int16),
        "parser_runtime_ms": np.zeros(frames, dtype=np.float64),
    }


def _copy_frame(
    arrays: dict[str, np.ndarray],
    frame_index: int,
    frame: Any,
    *,
    expected_v1_pose: np.ndarray,
    expected_v1_valid: bool,
    expected_spatial_shuffle: bool,
) -> None:
    values = {
        "poses_xy": frame.poses_xy,
        "link_half_widths_px": frame.link_half_widths_px,
        "candidate_valid": frame.valid,
        "candidate_cost": frame.costs,
        "candidate_weight": frame.relative_weights,
        "candidate_confidence": frame.confidence,
        "rgb_evidence_score": frame.rgb_evidence_score,
        "mask_geometry_score": frame.mask_geometry_score,
        "candidate_source_code": frame.source_codes,
    }
    expected = {
        "poses_xy": ((MAX_CANDIDATES, KEYPOINT_COUNT, 2), np.float32),
        "link_half_widths_px": ((MAX_CANDIDATES, ROLE_COUNT), np.float32),
        "candidate_valid": ((MAX_CANDIDATES,), np.bool_),
        "candidate_cost": ((MAX_CANDIDATES,), np.float32),
        "candidate_weight": ((MAX_CANDIDATES,), np.float32),
        "candidate_confidence": ((MAX_CANDIDATES,), np.float32),
        "rgb_evidence_score": ((MAX_CANDIDATES,), np.float32),
        "mask_geometry_score": ((MAX_CANDIDATES,), np.float32),
        "candidate_source_code": ((MAX_CANDIDATES,), np.uint8),
    }
    for name, (shape, dtype) in expected.items():
        value = values[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"RGB generator {name} changed: "
                f"{getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}."
            )
    roi = frame.roi_xyxy
    if not isinstance(roi, np.ndarray) or roi.shape != (4,) or roi.dtype != np.int32:
        raise RuntimeError("RGB generator roi_xyxy must be int32[4].")
    if not isinstance(frame.diagnostics, dict):
        raise RuntimeError("RGB generator diagnostics must be a dict.")
    if not math.isfinite(float(frame.runtime_ms)) or float(frame.runtime_ms) <= 0.0:
        raise RuntimeError("RGB generator runtime must be finite and positive.")
    if frame.role_names != TASK_ROLES[TASK]:
        raise RuntimeError("RGB generator semantic role order changed.")
    active_slots = [int(value) for value in np.flatnonzero(frame.valid)]
    shuffle_trace = frame.diagnostics.get("spatial_shuffle_trace_sha256")
    if (
        frame.diagnostics.get("spatial_shuffle") is not expected_spatial_shuffle
        or frame.diagnostics.get("current_frame_only") is not True
        or frame.diagnostics.get("relative_weights_not_calibrated") is not True
        or frame.diagnostics.get("anchor_slot_reserved") is not True
        or frame.diagnostics.get("rgb_slot_reserved") is not True
        or frame.diagnostics.get("candidate_slots") != active_slots
        or frame.diagnostics.get("candidate_count") != len(active_slots)
        or frame.diagnostics.get("fail_closed") is not (not bool(active_slots))
        or (
            expected_spatial_shuffle
            and (
                not isinstance(shuffle_trace, str)
                or len(shuffle_trace) != 64
                or any(character not in "0123456789abcdef" for character in shuffle_trace)
            )
        )
        or (not expected_spatial_shuffle and shuffle_trace is not None)
    ):
        raise RuntimeError("RGB generator arm/diagnostic contract changed.")

    # Slot zero is not merely similar to v1: it is audited bit-for-bit against
    # the published keypoint/valid arrays on every frame and in both arms.
    if bool(frame.valid[0]) != expected_v1_valid:
        raise RuntimeError("RGB generator slot-zero validity diverged from published v1.")
    if expected_v1_valid:
        if not np.array_equal(frame.poses_xy[0], expected_v1_pose):
            raise RuntimeError("RGB generator slot-zero pose diverged from published v1.")
        if int(frame.source_codes[0]) != SOURCE_CODES["v1_anchor"]:
            raise RuntimeError("Valid slot zero is not tagged as the v1 anchor.")
    else:
        if np.any(frame.poses_xy[0] != 0.0) or int(frame.source_codes[0]) != 0:
            raise RuntimeError("Invalid v1 slot zero must retain canonical padding.")

    for name, value in values.items():
        arrays[name][frame_index] = value
    if np.any((roi < np.iinfo(np.int16).min) | (roi > np.iinfo(np.int16).max)):
        raise RuntimeError("RGB generator ROI does not fit the sealed int16 schema.")
    arrays["roi_xyxy"][frame_index] = roi.astype(np.int16)
    arrays["parser_runtime_ms"][frame_index] = float(frame.runtime_ms)


def _expected_output_schema(frames: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    candidates = (frames, MAX_CANDIDATES)
    return {
        "poses_xy": ((frames, MAX_CANDIDATES, KEYPOINT_COUNT, 2), np.float32),
        "link_half_widths_px": ((frames, MAX_CANDIDATES, ROLE_COUNT), np.float32),
        "candidate_valid": (candidates, np.bool_),
        "candidate_cost": (candidates, np.float32),
        "candidate_weight": (candidates, np.float32),
        "candidate_confidence": (candidates, np.float32),
        "rgb_evidence_score": (candidates, np.float32),
        "mask_geometry_score": (candidates, np.float32),
        "candidate_source_code": (candidates, np.uint8),
        "roi_xyxy": ((frames, 4), np.int16),
        "parser_runtime_ms": ((frames,), np.float64),
    }


def _validate_output_arrays(
    arrays: dict[str, np.ndarray],
    *,
    frames: int,
    resolution: int,
    expected_widths: np.ndarray,
    expected_lengths: np.ndarray,
) -> None:
    if set(arrays) != OUTPUT_ARRAY_KEYS:
        raise RuntimeError("RGB replay output NPZ schema changed.")
    for name, (shape, dtype) in _expected_output_schema(frames).items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(
                f"RGB replay {name} is {value.shape}/{value.dtype}, expected {shape}/{dtype}."
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise RuntimeError(f"RGB replay {name} contains non-finite values.")

    valid = arrays["candidate_valid"]
    poses = arrays["poses_xy"]
    widths = arrays["link_half_widths_px"]
    costs = arrays["candidate_cost"]
    weights = arrays["candidate_weight"]
    confidence = arrays["candidate_confidence"]
    rgb_score = arrays["rgb_evidence_score"]
    mask_score = arrays["mask_geometry_score"]
    source = arrays["candidate_source_code"]
    roi = arrays["roi_xyxy"].astype(np.int32)
    frozen_widths = np.asarray(expected_widths, dtype=np.float32)
    frozen_lengths = np.asarray(expected_lengths, dtype=np.float64)
    if (
        frozen_widths.shape != (ROLE_COUNT,)
        or frozen_lengths.shape != (ROLE_COUNT,)
        or not np.isfinite(frozen_widths).all()
        or not np.isfinite(frozen_lengths).all()
        or np.any(frozen_widths <= 0.0)
        or np.any(frozen_lengths <= 0.0)
    ):
        raise RuntimeError("Frozen RGB support geometry is malformed.")
    if np.any(valid[:, 0] & (source[:, 0] != SOURCE_CODES["v1_anchor"])):
        raise RuntimeError("Valid slot zero must use the v1 source code.")
    rgb_codes = np.asarray(
        [SOURCE_CODES["rgb_ik_positive"], SOURCE_CODES["rgb_ik_negative"]],
        dtype=np.uint8,
    )
    if np.any(valid[:, 1] & ~np.isin(source[:, 1], rgb_codes)):
        raise RuntimeError("Valid slot one must be an RGB inverse-kinematic candidate.")
    if np.any(source[~valid] != SOURCE_CODES["padding"]):
        raise RuntimeError("Invalid candidate source codes must be padding.")
    if not np.isin(source, np.asarray(list(SOURCE_CODES.values()), np.uint8)).all():
        raise RuntimeError("Unknown RGB candidate source code.")
    if np.any(poses[~valid] != 0.0) or np.any(widths[~valid] != 0.0):
        raise RuntimeError("Invalid candidate pose/width slots must be zero padded.")
    if np.any(costs[~valid] != PADDING_COST):
        raise RuntimeError("Invalid candidate costs must use float32-max padding.")
    for name, value in (
        ("candidate_weight", weights),
        ("candidate_confidence", confidence),
        ("rgb_evidence_score", rgb_score),
        ("mask_geometry_score", mask_score),
    ):
        if np.any((value < 0.0) | (value > 1.0)):
            raise RuntimeError(f"{name} escaped [0,1].")
        if np.any(value[~valid] != 0.0):
            raise RuntimeError(f"Invalid {name} must be zero padded.")
    if np.any(costs[valid] < 0.0) or np.any(widths[valid] <= 0.0):
        raise RuntimeError("Valid candidate cost/width must be non-negative/positive.")
    expected_width_rows = np.broadcast_to(
        frozen_widths, (frames, MAX_CANDIDATES, ROLE_COUNT)
    )
    if not np.array_equal(widths[valid], expected_width_rows[valid]):
        raise RuntimeError("Valid RGB capsule widths changed from fixed support.")
    slot_one_valid = valid[:, 1]
    slot_one_lengths = np.linalg.norm(np.diff(poses[:, 1], axis=1), axis=2)
    expected_length_rows = np.broadcast_to(frozen_lengths, (frames, ROLE_COUNT))
    if not np.allclose(
        slot_one_lengths[slot_one_valid],
        expected_length_rows[slot_one_valid],
        rtol=1e-5,
        atol=1e-3,
    ):
        raise RuntimeError("Valid RGB IK link lengths changed from fixed support.")
    counts = valid.sum(axis=1)
    sums = weights.sum(axis=1)
    if not np.allclose(sums[counts > 0], 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError("Active RGB candidate weights do not sum to one.")
    if np.any(sums[counts == 0] != 0.0):
        raise RuntimeError("Empty RGB candidate frames must have zero total weight.")
    if np.any(arrays["parser_runtime_ms"] <= 0.0):
        raise RuntimeError("RGB parser runtime must be positive.")
    if np.any(roi < 0) or np.any(roi > resolution):
        raise RuntimeError("RGB ROI bounds escaped the frame.")
    nonempty_roi = (roi[:, 2] > roi[:, 0]) & (roi[:, 3] > roi[:, 1])
    if np.any(~nonempty_roi):
        raise RuntimeError("Every replay frame must seal a nonempty RGB ROI.")


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: np.ascontiguousarray(arrays[name]) for name in sorted(arrays)},
    )
    return file_sha256(path)


def _cuda_snapshot(torch: Any) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_uuid = os.environ.get("BENCHMARK_GPU_UUID")
    order = os.environ.get("CUDA_DEVICE_ORDER")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must bind one benchmark GPU.")
    if order != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID.")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("BENCHMARK_GPU_UUID must bind the physical GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("RGB replay requires exactly one visible CUDA device.")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": visible,
        "cuda_device_order": order,
        "gpu_uuid": gpu_uuid,
        "logical_cuda_device": 0,
        "device_name": torch.cuda.get_device_name(0),
    }


def _construct_generator(
    *,
    args: argparse.Namespace,
    graph: Any,
    support_rgb: np.ndarray,
    support_masks: np.ndarray,
):
    import torch
    from tdmpc2.perception.ordered_chain_rgb import (
        FORMAT as generator_format,
        LANDMARK_NAMES,
        PROTOCOL as generator_protocol,
        MAX_RENDERED_IK_CANDIDATES as generator_max_rendered_ik_candidates,
        MAX_TIP_PROPOSALS as generator_max_tip_proposals,
        SOURCE_CODES as generator_source_codes,
        FrozenDinoV2FeatureExtractor,
        OrderedChainRGBGenerator,
    )

    if dict(generator_source_codes) != SOURCE_CODES:
        raise RuntimeError("RGB generator source-code contract changed.")
    extractor = FrozenDinoV2FeatureExtractor(
        args.dino_repo,
        args.dino_checkpoint,
        model_name=args.dino_model,
        input_size=args.dino_input_size,
        device="cuda:0",
    )
    generator = OrderedChainRGBGenerator(
        graph,
        support_rgb,
        support_masks,
        extractor,
        max_candidates=MAX_CANDIDATES,
    )
    metadata = generator.metadata()
    expected_boolean_metadata = {
        "relative_weights_not_calibrated": True,
        "current_frame_only": True,
        "episode_state": False,
        "feature_extractor_stateless_contract": True,
        "fixed_support_feature_replay_exact": True,
        "future_frames": False,
        "action_input": False,
        "reward_input": False,
        "simulator_state_input": False,
        "episode_ground_truth_input": False,
        "fixed_labelled_support_masks_input": True,
        "task_name_dispatch": False,
        "capsule_rendering_uses_entity_mask": False,
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != GENERATOR_METADATA_KEYS
        or metadata.get("format") != generator_format
        or metadata.get("protocol") != generator_protocol
        or metadata.get("graph_sha256") != graph.graph_sha256
        or metadata.get("entity_name") != graph.entities[0].name
        or metadata.get("role_names") != list(TASK_ROLES[TASK])
        or metadata.get("max_candidates") != MAX_CANDIDATES
        or metadata.get("maximum_tip_proposals") != generator_max_tip_proposals
        or metadata.get("maximum_rendered_ik_candidates")
        != generator_max_rendered_ik_candidates
        or metadata.get("candidate_slot_semantics")
        != [
            "exact_current_mask_v1_anchor",
            "best_reliable_distinct_current_rgb_ik_mode_or_invalid",
        ]
        or metadata.get("source_codes") != SOURCE_CODES
        or metadata.get("support_rgb_trace_sha256") != _array_trace(support_rgb)
        or metadata.get("support_indexed_mask_trace_sha256")
        != _array_trace(support_masks)
        or metadata.get("support_resolution") != list(support_rgb.shape[1:3])
        or metadata.get("landmark_names") != list(LANDMARK_NAMES)
        or metadata.get("spatial_shuffle_policy")
        != "deterministic_roi_pixel_permutation_preserving_rgb_histogram_v1"
        or any(metadata.get(key) is not value for key, value in expected_boolean_metadata.items())
    ):
        raise RuntimeError("RGB generator metadata identity/schema changed.")
    for key in (
        "support_link_lengths_px",
        "support_link_half_widths_px",
    ):
        value = np.asarray(metadata.get(key))
        if value.shape != (2,) or not np.isfinite(value).all() or np.any(value <= 0.0):
            raise RuntimeError(f"RGB generator {key} metadata is malformed.")
    reliability = metadata.get("rgb_reliability_threshold")
    if (
        not isinstance(reliability, (int, float))
        or not math.isfinite(float(reliability))
        or not 0.0 <= float(reliability) <= 1.0
    ):
        raise RuntimeError("RGB generator reliability threshold is malformed.")
    extractor_metadata = extractor.metadata()
    if extractor_metadata != {
        "format": "frozen_dinov2_dense_feature_extractor_v1",
        "repo": str(Path(args.dino_repo).expanduser().resolve()),
        "checkpoint": str(Path(args.dino_checkpoint).expanduser().resolve()),
        "model_name": args.dino_model,
        "input_size": args.dino_input_size,
        "device": "cuda:0",
        "frozen_parameters": True,
        "evaluation_mode": True,
        "torch_hub_source_local": True,
        "pretrained_constructor_download": False,
        "network_isolation_enforced": False,
    }:
        raise RuntimeError("Frozen DINOv2 extractor metadata contract changed.")
    return generator, metadata, extractor


def _project_one(
    generator: Any,
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    available: bool,
    spatial_shuffle: bool,
    expected_v1_pose: np.ndarray,
    expected_v1_valid: bool,
) -> dict[str, np.ndarray]:
    arrays = _empty_arrays(frames=1)
    frame = generator.project(
        current_rgb=np.ascontiguousarray(rgb),
        entity_mask=np.ascontiguousarray(mask),
        entity_available=available,
        spatial_shuffle=spatial_shuffle,
    )
    _copy_frame(
        arrays,
        0,
        frame,
        expected_v1_pose=expected_v1_pose,
        expected_v1_valid=expected_v1_valid,
        expected_spatial_shuffle=spatial_shuffle,
    )
    metadata = generator.metadata()
    _validate_output_arrays(
        arrays,
        frames=1,
        resolution=rgb.shape[0],
        expected_widths=np.asarray(metadata["support_link_half_widths_px"]),
        expected_lengths=np.asarray(metadata["support_link_lengths_px"]),
    )
    return arrays


def _record_input_files(
    *,
    inputs: dict[str, Any],
    episode_paths: dict[tuple[str, str, int], Path],
    v1_paths: dict[tuple[str, int], tuple[Path, dict[str, Any]]],
) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    episodes = int(inputs["counts"]["episodes"])
    for condition in CONDITIONS:
        for episode_index in range(episodes):
            rgb_path = episode_paths[(TASK, condition, episode_index)]
            v1_path, _ = v1_paths[(condition, episode_index)]
            key = f"{condition}/{episode_index:03d}"
            records[key] = (file_sha256(rgb_path), file_sha256(v1_path))
    return records


def _immutable_revalidation(
    *,
    args: argparse.Namespace,
    input_path: Path,
    input_sha: str,
    inputs: dict[str, Any],
    manifest_path: Path,
    manifest_sha: str,
    v1_payload: dict[str, Any],
    graph: Any,
    graph_path: Path,
    graph_sha: str,
    support_path: Path,
    support_sha: str,
    dino_repo: Path,
    dino_tree: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha: str,
    implementation: dict[str, Any],
    source_files: dict[str, tuple[str, str]],
    cuda: dict[str, Any],
) -> None:
    import torch

    inputs_after, _, episode_paths_after = validate_backend_inputs(
        input_path, strict_counts=args.strict_counts
    )
    v1_after, v1_paths_after = _validate_v1_manifest(
        manifest_path,
        inputs=inputs_after,
        input_sha256=file_sha256(input_path),
        v1_graph=graph,
        v1_graph_path=graph_path,
    )
    if (
        inputs_after != inputs
        or file_sha256(input_path) != input_sha
        or v1_after != v1_payload
        or file_sha256(manifest_path) != manifest_sha
        or file_sha256(graph_path) != graph_sha
        or file_sha256(support_path) != support_sha
        or _python_tree_snapshot(dino_repo) != dino_tree
        or file_sha256(checkpoint_path) != checkpoint_sha
        or _implementation_snapshot() != implementation
        or _cuda_snapshot(torch) != cuda
        or _record_input_files(
            inputs=inputs_after,
            episode_paths=episode_paths_after,
            v1_paths=v1_paths_after,
        )
        != source_files
    ):
        raise RuntimeError("An RGB replay input or implementation changed during execution.")


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cuda = _cuda_snapshot(torch)
    input_path = _regular_file(Path(args.inputs), "GT-free worker input manifest")
    input_sha = file_sha256(input_path)
    inputs, support_paths, episode_paths = validate_backend_inputs(
        input_path, strict_counts=args.strict_counts
    )
    graph, graph_path = _load_v1_graph(Path(args.v1_graph))
    graph_sha = file_sha256(graph_path)
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
    source_provenance = v1_payload.get("backend_provenance", {})
    if (
        source_provenance.get("gpu_uuid") != cuda["gpu_uuid"]
        or source_provenance.get("device_name") != cuda["device_name"]
    ):
        raise RuntimeError(
            "RGB development replay must use the exact physical GPU/model of the "
            "published v1 Cutie benchmark."
        )
    resolution = int(inputs["resolution"])
    support_path = _regular_file(support_paths[TASK], "Acrobot frozen support")
    support_sha = file_sha256(support_path)
    support_rgb, support_masks = _load_support(support_path, resolution=resolution)
    dino_repo = _regular_directory(Path(args.dino_repo), "DINOv2 repository")
    dino_tree = _python_tree_snapshot(dino_repo)
    checkpoint_path = _regular_file(Path(args.dino_checkpoint), "DINOv2 checkpoint")
    checkpoint_sha = file_sha256(checkpoint_path)
    implementation = _implementation_snapshot()
    source_files = _record_input_files(
        inputs=inputs, episode_paths=episode_paths, v1_paths=v1_paths
    )
    generator, generator_metadata, extractor = _construct_generator(
        args=args,
        graph=graph,
        support_rgb=support_rgb,
        support_masks=support_masks,
    )
    return {
        "torch": torch,
        "cuda": cuda,
        "input_path": input_path,
        "input_sha": input_sha,
        "inputs": inputs,
        "episode_paths": episode_paths,
        "graph": graph,
        "graph_path": graph_path,
        "graph_sha": graph_sha,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "v1_payload": v1_payload,
        "v1_paths": v1_paths,
        "support_path": support_path,
        "support_sha": support_sha,
        "support_rgb": support_rgb,
        "support_masks": support_masks,
        "dino_repo": dino_repo,
        "dino_tree": dino_tree,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha": checkpoint_sha,
        "implementation": implementation,
        "source_files": source_files,
        "generator": generator,
        "generator_metadata": generator_metadata,
        "extractor": extractor,
    }


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    context = _prepare(args)
    inputs = context["inputs"]
    frames = int(inputs["counts"]["frames_per_episode"])
    episodes = int(inputs["counts"]["episodes"])
    resolution = int(inputs["resolution"])
    selected: tuple[
        str, int, int, dict[str, np.ndarray], dict[str, Any]
    ] | None = None
    for condition in CONDITIONS:
        for episode_index in range(episodes):
            v1_path, v1_record = context["v1_paths"][(condition, episode_index)]
            v1 = _load_v1_arrays(
                v1_path, v1_record, frames=frames, resolution=resolution
            )
            usable = v1["entity_valid"][:, 0] & v1["role_valid"].all(axis=1)
            indices = np.flatnonzero(usable)
            if indices.size:
                selected = (
                    condition,
                    episode_index,
                    int(indices[0]),
                    v1,
                    v1_record,
                )
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError(
            "No deterministic published-v1 valid frame exists for the real DINO smoke."
        )
    condition, episode_index, frame_index, v1, v1_record = selected
    rgb_record = inputs["episodes"][TASK][condition][episode_index]
    rgb = _load_episode_rgb(
        context["episode_paths"][(TASK, condition, episode_index)],
        frames=frames,
        resolution=resolution,
        expected_trace=rgb_record["rgb_trace_sha256"],
    )
    expected_pose, expected_valid = _v1_pose_and_valid(v1, frame_index)
    if not bool(v1["entity_valid"][frame_index, 0]) or not expected_valid:
        raise RuntimeError("The deterministically selected DINO smoke frame changed.")
    traces = {}
    for arm in ARMS:
        arrays = _project_one(
            context["generator"],
            rgb=rgb[frame_index],
            mask=v1["entity_masks"][frame_index, 0],
            available=bool(v1["entity_valid"][frame_index, 0]),
            spatial_shuffle=arm == "spatial_shuffle",
            expected_v1_pose=expected_pose,
            expected_v1_valid=expected_valid,
        )
        traces[arm] = {
            name: _array_trace(value) for name, value in sorted(arrays.items())
        }
    _immutable_revalidation(
        args=args,
        input_path=context["input_path"],
        input_sha=context["input_sha"],
        inputs=inputs,
        manifest_path=context["manifest_path"],
        manifest_sha=context["manifest_sha"],
        v1_payload=context["v1_payload"],
        graph=context["graph"],
        graph_path=context["graph_path"],
        graph_sha=context["graph_sha"],
        support_path=context["support_path"],
        support_sha=context["support_sha"],
        dino_repo=context["dino_repo"],
        dino_tree=context["dino_tree"],
        checkpoint_path=context["checkpoint_path"],
        checkpoint_sha=context["checkpoint_sha"],
        implementation=context["implementation"],
        source_files=context["source_files"],
        cuda=context["cuda"],
    )
    return {
        "status": "preflight_complete",
        "task": TASK,
        "dataset_id": inputs["dataset_id"],
        "arms": traces,
        "probe": {
            "selection": "first_entity_and_both_roles_valid_in_condition_episode_frame_order",
            "condition": condition,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "source_v1_prediction_arrays_sha256": v1_record[
                "prediction_arrays_sha256"
            ],
        },
        "generator": context["generator_metadata"],
        "gpu_uuid": context["cuda"]["gpu_uuid"],
        "controller_training_eligible": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only:
        if args.output_root is not None:
            raise ValueError("--preflight-only forbids --output-root.")
        return _run_preflight(args)

    if args.output_root is None:
        raise ValueError("Full RGB replay requires --output-root.")

    context = _prepare(args)
    inputs = context["inputs"]
    frames = int(inputs["counts"]["frames_per_episode"])
    episodes = int(inputs["counts"]["episodes"])
    resolution = int(inputs["resolution"])
    output_root = Path(args.output_root).expanduser().resolve()
    incomplete_root = output_root.with_name(output_root.name + ".incomplete")
    if output_root.exists() or incomplete_root.exists():
        raise FileExistsError(output_root if output_root.exists() else incomplete_root)
    if not output_root.parent.is_dir():
        raise FileNotFoundError(output_root.parent)
    incomplete_root.mkdir(exist_ok=False)

    started = perf_counter()
    results: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }
    for condition in CONDITIONS:
        for episode_index in range(episodes):
            rgb_record = inputs["episodes"][TASK][condition][episode_index]
            rgb_path = context["episode_paths"][(TASK, condition, episode_index)]
            rgb = _load_episode_rgb(
                rgb_path,
                frames=frames,
                resolution=resolution,
                expected_trace=rgb_record["rgb_trace_sha256"],
            )
            v1_path, v1_record = context["v1_paths"][(condition, episode_index)]
            v1 = _load_v1_arrays(
                v1_path, v1_record, frames=frames, resolution=resolution
            )
            arm_arrays = {arm: _empty_arrays(frames=frames) for arm in ARMS}
            for frame_index in range(frames):
                expected_pose, expected_valid = _v1_pose_and_valid(v1, frame_index)
                current_rgb = np.ascontiguousarray(rgb[frame_index])
                current_mask = np.ascontiguousarray(v1["entity_masks"][frame_index, 0])
                current_available = bool(v1["entity_valid"][frame_index, 0])
                for arm in ARMS:
                    frame = context["generator"].project(
                        current_rgb=current_rgb,
                        entity_mask=current_mask,
                        entity_available=current_available,
                        spatial_shuffle=arm == "spatial_shuffle",
                    )
                    _copy_frame(
                        arm_arrays[arm],
                        frame_index,
                        frame,
                        expected_v1_pose=expected_pose,
                        expected_v1_valid=expected_valid,
                        expected_spatial_shuffle=arm == "spatial_shuffle",
                    )

            for arm in ARMS:
                arrays = arm_arrays[arm]
                _validate_output_arrays(
                    arrays,
                    frames=frames,
                    resolution=resolution,
                    expected_widths=np.asarray(
                        context["generator_metadata"]["support_link_half_widths_px"]
                    ),
                    expected_lengths=np.asarray(
                        context["generator_metadata"]["support_link_lengths_px"]
                    ),
                )
                relative = (
                    Path("predictions")
                    / arm
                    / TASK
                    / condition
                    / f"episode_{episode_index:03d}.npz"
                )
                prediction_sha = _save_npz(incomplete_root / relative, arrays)
                traces = {
                    name: _array_trace(arrays[name]) for name in sorted(arrays)
                }
                record = {
                    "arm": arm,
                    "episode_index": episode_index,
                    "frames": frames,
                    "max_candidates": MAX_CANDIDATES,
                    "role_count": ROLE_COUNT,
                    "keypoint_count": KEYPOINT_COUNT,
                    "prediction_arrays": relative.as_posix(),
                    "prediction_arrays_sha256": prediction_sha,
                    "array_shapes": {
                        name: list(arrays[name].shape) for name in sorted(arrays)
                    },
                    "array_dtypes": {
                        name: str(arrays[name].dtype) for name in sorted(arrays)
                    },
                    "array_traces_sha256": traces,
                    "source_rgb_arrays_sha256": file_sha256(rgb_path),
                    "source_rgb_trace_sha256": rgb_record["rgb_trace_sha256"],
                    "decoded_rgb_array_trace_sha256": _array_trace(rgb),
                    "source_v1_prediction_arrays_sha256": v1_record[
                        "prediction_arrays_sha256"
                    ],
                    "source_v1_entity_mask_trace_sha256": v1_record["traces"][
                        "entity_mask_trace_sha256"
                    ],
                    "source_v1_entity_status_trace_sha256": v1_record["traces"][
                        "entity_status_trace_sha256"
                    ],
                    "source_v1_keypoint_trace_sha256": v1_record["traces"][
                        "keypoint_trace_sha256"
                    ],
                    "source_v1_role_status_trace_sha256": v1_record["traces"][
                        "role_status_trace_sha256"
                    ],
                }
                if set(record) != RESULT_RECORD_KEYS:
                    raise AssertionError("Internal RGB replay result schema changed.")
                results[arm].setdefault(condition, []).append(record)
                print(
                    "OBJECT_GRAPH_RGB_REPLAY_EPISODE",
                    json.dumps(
                        {
                            "arm": arm,
                            "condition": condition,
                            "episode_index": episode_index,
                            "poses_trace_sha256": traces["poses_xy"],
                            "candidate_valid_trace_sha256": traces[
                                "candidate_valid"
                            ],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

    _immutable_revalidation(
        args=args,
        input_path=context["input_path"],
        input_sha=context["input_sha"],
        inputs=inputs,
        manifest_path=context["manifest_path"],
        manifest_sha=context["manifest_sha"],
        v1_payload=context["v1_payload"],
        graph=context["graph"],
        graph_path=context["graph_path"],
        graph_sha=context["graph_sha"],
        support_path=context["support_path"],
        support_sha=context["support_sha"],
        dino_repo=context["dino_repo"],
        dino_tree=context["dino_tree"],
        checkpoint_path=context["checkpoint_path"],
        checkpoint_sha=context["checkpoint_sha"],
        implementation=context["implementation"],
        source_files=context["source_files"],
        cuda=context["cuda"],
    )

    payload = {
        "format": FORMAT,
        "status": "complete",
        "backend": BACKEND,
        "task": TASK,
        "development_scope": "single_task_acrobot_offline_candidate_preflight_v1",
        "dataset_id": inputs["dataset_id"],
        "input_manifest_sha256": context["input_sha"],
        "v1_backend_manifest_sha256": context["manifest_sha"],
        "roles": list(TASK_ROLES[TASK]),
        "arms": list(ARMS),
        "max_candidates": MAX_CANDIDATES,
        "graph": {
            "path": str(context["graph_path"]),
            "file_sha256": context["graph_sha"],
            "graph": context["graph"].metadata(),
        },
        "support": {
            "path": str(context["support_path"]),
            "file_sha256": context["support_sha"],
            "rgb_trace_sha256": _array_trace(context["support_rgb"]),
            "indexed_masks_trace_sha256": _array_trace(context["support_masks"]),
        },
        "dino": {
            "model": args.dino_model,
            "input_size": args.dino_input_size,
            "repo": str(context["dino_repo"]),
            "python_source_tree": context["dino_tree"],
            "checkpoint": str(context["checkpoint_path"]),
            "checkpoint_sha256": context["checkpoint_sha"],
            "extractor": context["extractor"].metadata(),
        },
        "generator": context["generator_metadata"],
        "output_schema": {
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
        },
        "protocol": {
            "format": PROTOCOL,
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
        },
        "backend_provenance": {
            "treatment": "frozen_dinov2_current_rgb_ordered_chain_candidate_v1",
            "implementation": context["implementation"],
            "source_v1_backend_manifest": str(context["manifest_path"]),
            "source_v1_backend_manifest_sha256": context["manifest_sha"],
            "source_worker_inputs": str(context["input_path"]),
            "source_worker_inputs_sha256": context["input_sha"],
            "source_support_arrays": str(context["support_path"]),
            "source_support_arrays_sha256": context["support_sha"],
            "v1_graph_file_sha256": context["graph_sha"],
            "v1_graph_semantic_sha256": context["graph"].graph_sha256,
            **context["cuda"],
            "wallclock_seconds": perf_counter() - started,
        },
        "results": results,
    }
    if set(payload) != MANIFEST_KEYS:
        raise AssertionError("Internal RGB replay manifest schema changed.")
    write_json(incomplete_root / "backend_predictions.json", payload)
    os.replace(incomplete_root, output_root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--v1-backend-manifest", type=Path, required=True)
    parser.add_argument("--v1-graph", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--dino-input-size", type=int, default=DEFAULT_DINO_INPUT_SIZE)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Required for full replay and forbidden with --preflight-only.",
    )
    parser.add_argument("--strict-counts", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate every frozen input, really load DINOv2, and project the "
            "first deterministic published-v1 valid frame in both arms without "
            "creating --output-root."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dino_input_size <= 0:
        raise ValueError("--dino-input-size must be positive.")
    payload = run(args)
    marker = (
        "OBJECT_GRAPH_RGB_REPLAY_PREFLIGHT_COMPLETE"
        if args.preflight_only
        else "OBJECT_GRAPH_RGB_REPLAY_COMPLETE"
    )
    message = {
        "status": payload["status"],
        "dataset_id": payload["dataset_id"],
        "controller_training_eligible": False,
    }
    if not args.preflight_only:
        message["manifest"] = str(
            Path(args.output_root).expanduser().resolve() / "backend_predictions.json"
        )
    print(marker, json.dumps(message, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
