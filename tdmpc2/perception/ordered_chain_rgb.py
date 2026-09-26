"""Current-frame RGB evidence for a support-calibrated ordered chain.

The implementation is deliberately small and fail-closed.  It supports one
tracking entity projected into exactly two ordered roles.  The graph selects
that primitive; Python code never dispatches on a task identifier.  Fixed
support RGB and labelled support masks provide all calibration.  Projection
accepts only the current RGB image and current tracker entity mask.

Slot zero is the exact current-mask v1 anchor from
``OrderedChainTopKGenerator``.  Slot one is a distinct inverse-kinematics mode
ranked by frozen RGB feature evidence.  Candidate weights are relative scores,
not calibrated probabilities and not authorization for controller training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from tdmpc2.perception.ordered_chain_topk import (
    OrderedChainTopKGenerator,
    _circle_intersections,
)
from tdmpc2.perception.support_conditioned_object_graph import (
    CompiledObjectGraph,
    ObjectGraphContractError,
    _segment_distance_squared,
)


FORMAT = "support_conditioned_ordered_chain_rgb_v1"
PROTOCOL = "stateless_current_rgb_entity_roi_two_link_v1"
MAX_CANDIDATES = 2
LANDMARK_NAMES = ("root", "first_midpoint", "joint", "second_midpoint", "tip")
SOURCE_CODES: Mapping[str, int] = {
    "padding": 0,
    "v1_anchor": 1,
    "rgb_ik_positive": 2,
    "rgb_ik_negative": 3,
}

_LINE_SAMPLES = 17
MAX_TIP_PROPOSALS = 24
MAX_RENDERED_IK_CANDIDATES = 8
_WEIGHT_TEMPERATURE = 0.35


@runtime_checkable
class DenseFeatureExtractor(Protocol):
    """Minimal injectable interface used by :class:`OrderedChainRGBGenerator`.

    The return value is a finite floating-point ``[feature_h, feature_w, dim]``
    array.  Spatial dimensions need not equal the RGB input dimensions.
    """

    stateless: bool
    current_frame_only: bool

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """Extract a dense feature map from one exact uint8 RGB image."""


def _typed_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_rgb(value: np.ndarray, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.uint8:
        raise TypeError(f"{label} must have exact uint8 dtype.")
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"{label} must have shape [H,W,3].")
    if min(raw.shape[:2]) < 8:
        raise ValueError(f"{label} spatial dimensions are too small.")
    return np.ascontiguousarray(raw)


def _validate_feature_map(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 4 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 3 or min(raw.shape) < 1:
        raise ObjectGraphContractError(
            f"{label} must return a non-empty [feature_h,feature_w,dim] array."
        )
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.floating):
        raise ObjectGraphContractError(f"{label} must return floating-point features.")
    result = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ObjectGraphContractError(f"{label} returned non-finite features.")
    return result


def _extract_dense(extractor: Any, rgb: np.ndarray) -> np.ndarray:
    method = getattr(extractor, "extract", None)
    if callable(method):
        result = method(np.ascontiguousarray(rgb))
    elif callable(extractor):
        result = extractor(np.ascontiguousarray(rgb))
    else:
        raise ObjectGraphContractError(
            "feature_extractor must expose extract(rgb) or be callable."
        )
    return _validate_feature_map(result, "feature_extractor")


def _normalize_vectors(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(
        array,
        np.maximum(norm, np.float32(1e-12)),
        out=np.zeros_like(array),
        where=norm > np.float32(1e-12),
    )


def _bilinear_sample_feature(
    feature_map: np.ndarray,
    point_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    feature_h, feature_w = feature_map.shape[:2]
    image_h, image_w = image_shape
    x = float(point_xy[0]) * max(feature_w - 1, 0) / max(image_w - 1, 1)
    y = float(point_xy[1]) * max(feature_h - 1, 0) / max(image_h - 1, 1)
    x = float(np.clip(x, 0.0, max(feature_w - 1, 0)))
    y = float(np.clip(y, 0.0, max(feature_h - 1, 0)))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, feature_w - 1), min(y0 + 1, feature_h - 1)
    wx, wy = x - x0, y - y0
    return np.ascontiguousarray(
        (1.0 - wy)
        * ((1.0 - wx) * feature_map[y0, x0] + wx * feature_map[y0, x1])
        + wy * ((1.0 - wx) * feature_map[y1, x0] + wx * feature_map[y1, x1]),
        dtype=np.float32,
    )


def _resize_score_map(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Dependency-free deterministic bilinear resize for a scalar map."""
    source = np.asarray(value, dtype=np.float32)
    target_h, target_w = shape
    source_h, source_w = source.shape
    if (source_h, source_w) == shape:
        return np.ascontiguousarray(source)
    x = np.linspace(0.0, max(source_w - 1, 0), target_w, dtype=np.float64)
    y = np.linspace(0.0, max(source_h - 1, 0), target_h, dtype=np.float64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_w - 1)
    y1 = np.minimum(y0 + 1, source_h - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    top = source[:, x0] * (1.0 - wx)[None] + source[:, x1] * wx[None]
    bottom = source[:, x0] * (1.0 - wx)[None] + source[:, x1] * wx[None]
    result = top[y0] * (1.0 - wy)[:, None] + bottom[y1] * wy[:, None]
    return np.ascontiguousarray(result, dtype=np.float32)


def _similarity_map(
    feature_map: np.ndarray,
    prototype: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    normalized = _normalize_vectors(feature_map)
    proto = _normalize_vectors(np.asarray(prototype, dtype=np.float32)[None])[0]
    cosine = np.sum(normalized * proto[None, None], axis=-1)
    valid = np.linalg.norm(feature_map, axis=-1) > 1e-12
    score = np.where(valid, np.clip((cosine + 1.0) * 0.5, 0.0, 1.0), 0.0)
    return _resize_score_map(score.astype(np.float32), image_shape)


def _point_score(score_map: np.ndarray, point_xy: np.ndarray) -> float:
    return float(_bilinear_sample_feature(score_map[..., None], point_xy, score_map.shape)[0])


def _line_score(
    score_map: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> float:
    alpha = np.linspace(0.12, 0.88, _LINE_SAMPLES, dtype=np.float64)[:, None]
    points = start_xy[None] + alpha * (end_xy - start_xy)[None]
    height, width = score_map.shape
    x = np.clip(points[:, 0], 0.0, max(width - 1, 0))
    y = np.clip(points[:, 1], 0.0, max(height - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    values = (
        (1.0 - wy) * ((1.0 - wx) * score_map[y0, x0] + wx * score_map[y0, x1])
        + wy * ((1.0 - wx) * score_map[y1, x0] + wx * score_map[y1, x1])
    )
    return float(np.mean(values))


def _capsule_distance_maps(
    pose_xy: np.ndarray,
    widths: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float64)
    distance = np.stack(
        [
            _segment_distance_squared(xx, yy, pose_xy[index], pose_xy[index + 1])
            for index in range(2)
        ]
    )
    normalized = distance / np.maximum(widths[:, None, None] ** 2, 1e-12)
    return distance, normalized


def render_role_capsules(
    pose_xy: np.ndarray,
    widths: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Render two deterministic, non-overlapping ordered role capsules.

    Rendering is deliberately independent of the tracker mask.  A pixel in
    both capsules is assigned to the closest normalized centreline; exact ties
    go to the first ordered role through ``numpy.argmin``.
    """
    pose = np.asarray(pose_xy)
    half_widths = np.asarray(widths)
    if pose.shape != (3, 2) or pose.dtype == np.bool_ or np.iscomplexobj(pose) or not np.issubdtype(
        pose.dtype, np.number
    ):
        raise ValueError("pose_xy must be a numeric [3,2] array.")
    if half_widths.shape != (2,) or half_widths.dtype == np.bool_ or np.iscomplexobj(half_widths) or not np.issubdtype(
        half_widths.dtype, np.number
    ):
        raise ValueError("widths must be a numeric [2] array.")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("shape must be a positive (height,width) tuple.")
    pose64 = np.ascontiguousarray(pose, dtype=np.float64)
    widths64 = np.ascontiguousarray(half_widths, dtype=np.float64)
    if not np.isfinite(pose64).all() or not np.isfinite(widths64).all():
        raise ValueError("pose_xy and widths must be finite.")
    if np.any(widths64 <= 0.0) or np.any(
        np.linalg.norm(np.diff(pose64, axis=0), axis=1) <= 1e-9
    ):
        raise ValueError("Capsule widths and link lengths must be positive.")
    distance, normalized = _capsule_distance_maps(pose64, widths64, shape)
    inside = distance <= widths64[:, None, None] ** 2
    owner = np.argmin(normalized, axis=0)
    output = np.stack([inside[index] & (owner == index) for index in range(2)])
    return np.ascontiguousarray(output, dtype=np.bool_)


def _mask_iou(rendered_roles: np.ndarray, entity_mask: np.ndarray) -> float:
    rendered = rendered_roles.any(axis=0)
    intersection = int(np.logical_and(rendered, entity_mask).sum())
    union = int(np.logical_or(rendered, entity_mask).sum())
    return float(intersection / union) if union else 0.0


def _spatial_shuffle_roi(rgb: np.ndarray, roi_xyxy: np.ndarray) -> np.ndarray:
    """Deterministically permute ROI pixels while preserving its histogram."""
    x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
    result = np.ascontiguousarray(rgb.copy())
    view = np.ascontiguousarray(result[y0:y1, x0:x1])
    count = int(view.shape[0] * view.shape[1])
    if count <= 1:
        return result
    seed_material = np.asarray([x0, y0, x1, y1, count], dtype=np.int64).tobytes()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    permutation = np.random.default_rng(seed).permutation(count)
    result[y0:y1, x0:x1] = view.reshape(count, 3)[permutation].reshape(view.shape)
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class _RGBCandidate:
    pose_xy: np.ndarray
    cost: float
    confidence: float
    source: str
    source_code: int
    rgb_evidence: float
    mask_geometry: float
    bend: int


@dataclass(frozen=True)
class OrderedChainRGBFrame:
    role_names: tuple[str, str]
    poses_xy: np.ndarray
    link_half_widths_px: np.ndarray
    valid: np.ndarray
    cost: np.ndarray
    relative_weights: np.ndarray
    confidence: np.ndarray
    source_codes: np.ndarray
    rgb_evidence_score: np.ndarray
    mask_geometry_score: np.ndarray
    roi_xyxy: np.ndarray
    runtime_ms: float
    diagnostics: dict[str, Any]

    @property
    def costs(self) -> np.ndarray:
        """Compatibility alias; the frozen serialized field is ``cost``."""
        return self.cost


class FrozenDinoV2FeatureExtractor:
    """Lazy-PyTorch dense extractor backed by a local frozen DINOv2 checkout.

    Importing this module never imports PyTorch.  Construction resolves local
    repository/checkpoint paths, builds the model with ``torch.hub`` in local
    mode, loads weights, freezes parameters, and switches to evaluation mode.
    """

    stateless = True
    current_frame_only = True

    def __init__(
        self,
        repo: str | Path,
        checkpoint: str | Path,
        model_name: str = "dinov2_vits14_reg",
        input_size: int = 224,
        device: str = "cuda:0",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if type(input_size) is not int or input_size < 28:
            raise ValueError("input_size must be an integer >= 28.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string.")
        repo_path = Path(repo).expanduser().absolute()
        checkpoint_path = Path(checkpoint).expanduser().absolute()
        if repo_path.is_symlink() or checkpoint_path.is_symlink():
            raise ValueError("DINOv2 repository and checkpoint must not be symlinks.")
        repo_path = repo_path.resolve(strict=True)
        checkpoint_path = checkpoint_path.resolve(strict=True)
        if not repo_path.is_dir() or not checkpoint_path.is_file():
            raise ValueError("DINOv2 repository/checkpoint types are invalid.")

        import torch  # Lazy by contract: intentionally local to construction.

        self._torch = torch
        self.repo = repo_path
        self.checkpoint = checkpoint_path
        self.model_name = model_name
        self.input_size = input_size
        self.device = device
        model = torch.hub.load(
            str(repo_path), model_name, source="local", pretrained=False
        )
        try:
            payload = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=True
            )
        except TypeError:
            payload = torch.load(str(checkpoint_path), map_location="cpu")
        state = payload
        if isinstance(payload, Mapping):
            for key in ("model", "state_dict", "teacher", "student"):
                candidate = payload.get(key)
                if isinstance(candidate, Mapping):
                    state = candidate
                    break
        if not isinstance(state, Mapping):
            raise ValueError("DINOv2 checkpoint does not contain a state dictionary.")
        cleaned: dict[str, Any] = {}
        for key, value in state.items():
            name = str(key)
            for prefix in ("module.", "backbone.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            cleaned[name] = value
        # A partially loaded backbone would leave randomly initialised tensors
        # in the supposedly frozen scientific frontend.  Require an exact
        # model/checkpoint identity after removing only well-known wrapper
        # prefixes; incompatible exports fail closed instead of becoming a
        # seed-dependent feature extractor.
        try:
            model.load_state_dict(cleaned, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "DINOv2 checkpoint is not an exact match for model_name."
            ) from exc
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(device)

    def metadata(self) -> dict[str, Any]:
        """Return construction provenance without importing or mutating state."""
        return {
            "format": "frozen_dinov2_dense_feature_extractor_v1",
            "repo": str(self.repo),
            "checkpoint": str(self.checkpoint),
            "model_name": self.model_name,
            "input_size": self.input_size,
            "device": self.device,
            "frozen_parameters": True,
            "evaluation_mode": True,
            "torch_hub_source_local": True,
            "pretrained_constructor_download": False,
            "network_isolation_enforced": False,
        }

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        image = _validate_rgb(rgb, "rgb")
        torch = self._torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None]
        tensor = tensor.to(device=self.device, dtype=torch.float32) / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=tensor.dtype
        )[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=tensor.dtype
        )[None, :, None, None]
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            output = self._model.forward_features(tensor)
        if isinstance(output, Mapping):
            tokens = output.get("x_norm_patchtokens")
            if tokens is None:
                tokens = output.get("x_prenorm")
        else:
            tokens = output
        if tokens is None or getattr(tokens, "ndim", None) != 3:
            raise RuntimeError("DINOv2 did not return dense patch tokens.")
        token_count = int(tokens.shape[1])
        grid = int(round(math.sqrt(token_count)))
        if grid * grid != token_count:
            patch_size = getattr(getattr(self._model, "patch_embed", None), "patch_size", None)
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            if type(patch_size) is int and patch_size > 0:
                grid = self.input_size // patch_size
        if grid * grid != token_count:
            raise RuntimeError("DINOv2 patch-token grid is not square.")
        array = tokens[0].reshape(grid, grid, -1).detach().float().cpu().numpy()
        return np.ascontiguousarray(array, dtype=np.float32)


class OrderedChainRGBGenerator:
    """Support-conditioned, current-frame-only RGB two-link generator."""

    def __init__(
        self,
        graph: CompiledObjectGraph,
        support_rgb: np.ndarray,
        support_indexed_masks: np.ndarray,
        feature_extractor: DenseFeatureExtractor,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        if max_candidates != MAX_CANDIDATES or type(max_candidates) is not int:
            raise ObjectGraphContractError(
                f"This primitive has fixed max_candidates={MAX_CANDIDATES}."
            )
        if not isinstance(graph, CompiledObjectGraph):
            raise ObjectGraphContractError("graph must be a CompiledObjectGraph.")
        if (
            getattr(feature_extractor, "stateless", None) is not True
            or getattr(feature_extractor, "current_frame_only", None) is not True
        ):
            raise ObjectGraphContractError(
                "feature_extractor must explicitly declare stateless=True and "
                "current_frame_only=True."
            )
        if len(graph.entities) != 1:
            raise ObjectGraphContractError(
                "The RGB ordered-chain primitive requires exactly one entity."
            )
        entity = graph.entities[0]
        if entity.projector.type not in {
            "ordered_chain_segments_v1",
            "ordered_chain_temporal_v2",
        } or len(entity.projector.roles) != 2:
            raise ObjectGraphContractError(
                "The RGB primitive requires exactly two ordered chain roles."
            )
        matching_relations = [
            relation
            for relation in graph.relations
            if relation.parent == entity.projector.roles[0]
            and relation.child == entity.projector.roles[1]
            and relation.type == "revolute_joint_v1"
        ]
        if len(matching_relations) != 1 or len(graph.relations) != 1:
            raise ObjectGraphContractError(
                "The RGB primitive requires one declared ordered revolute relation."
            )

        support = np.asarray(support_rgb)
        indexed = np.asarray(support_indexed_masks)
        if support.dtype != np.uint8:
            raise TypeError("support_rgb must have exact uint8 dtype.")
        if support.ndim != 4 or support.shape[-1] != 3:
            raise ObjectGraphContractError(
                "support_rgb must have shape [N,H,W,3]."
            )
        if (
            indexed.ndim != 3
            or indexed.shape != support.shape[:3]
            or indexed.dtype == np.bool_
            or not np.issubdtype(indexed.dtype, np.integer)
        ):
            raise ObjectGraphContractError(
                "support_indexed_masks must be integer [N,H,W] aligned to RGB."
            )
        if support.shape[0] != 6:
            raise ObjectGraphContractError(
                "The exact v1 anchor requires six support frames."
            )
        expected_ids = set(range(len(graph.source_roles) + 1))
        observed = set(int(value) for value in np.unique(indexed))
        if not observed.issubset(expected_ids):
            raise ObjectGraphContractError("Support masks contain an unknown role id.")
        for role_id in range(1, len(graph.source_roles) + 1):
            if any(not np.any(frame == role_id) for frame in indexed):
                raise ObjectGraphContractError(
                    "Every support frame must contain both ordered roles."
                )

        self.graph = graph
        self.entity = entity
        self.max_candidates = max_candidates
        self.feature_extractor = feature_extractor
        self._support_rgb = np.ascontiguousarray(support)
        self._support_indexed_masks = np.ascontiguousarray(indexed)
        self._resolution = tuple(int(value) for value in support.shape[1:3])
        self._anchor = OrderedChainTopKGenerator(
            graph, self._support_indexed_masks, max_candidates=1
        )
        calibration = self._anchor.calibration
        self._support_poses = np.ascontiguousarray(
            calibration.support_poses_xy, dtype=np.float64
        )
        self._link_lengths = np.ascontiguousarray(
            calibration.link_lengths_px, dtype=np.float64
        )
        self._root_xy = np.ascontiguousarray(calibration.root_xy, dtype=np.float64)
        self._root_scale_px = float(calibration.root_scale_px)

        role_ids = [
            graph.source_roles.index(role) + 1 for role in entity.projector.roles
        ]
        support_widths: list[np.ndarray] = []
        for frame_index, pose in enumerate(self._support_poses):
            counts = np.asarray(
                [
                    int((self._support_indexed_masks[frame_index] == role_id).sum())
                    for role_id in role_ids
                ],
                dtype=np.float64,
            )
            lengths = np.linalg.norm(np.diff(pose, axis=0), axis=1)
            support_widths.append(np.maximum(counts / np.maximum(2.0 * lengths, 1.0), 1.0))
        self._link_half_widths = np.ascontiguousarray(
            np.median(np.stack(support_widths), axis=0), dtype=np.float64
        )

        feature_maps: list[np.ndarray] = []
        for frame in self._support_rgb:
            first = _extract_dense(self.feature_extractor, frame)
            replayed = _extract_dense(self.feature_extractor, frame)
            if not np.array_equal(first, replayed):
                raise ObjectGraphContractError(
                    "feature_extractor changed on an adjacent fixed-support replay."
                )
            feature_maps.append(first)
        dimensions = {int(value.shape[-1]) for value in feature_maps}
        if len(dimensions) != 1:
            raise ObjectGraphContractError(
                "Feature dimension changed across fixed support frames."
            )
        prototype_samples: list[list[np.ndarray]] = [[] for _ in LANDMARK_NAMES]
        for feature_map, pose in zip(feature_maps, self._support_poses):
            points = (
                pose[0],
                0.5 * (pose[0] + pose[1]),
                pose[1],
                0.5 * (pose[1] + pose[2]),
                pose[2],
            )
            for index, point in enumerate(points):
                prototype_samples[index].append(
                    _bilinear_sample_feature(feature_map, point, self._resolution)
                )
        prototypes = []
        for samples in prototype_samples:
            normalized_samples = _normalize_vectors(np.stack(samples))
            prototype = np.median(normalized_samples, axis=0)
            if float(np.linalg.norm(prototype)) <= 1e-8:
                prototype = np.mean(normalized_samples, axis=0)
            prototype = _normalize_vectors(prototype[None])[0]
            if float(np.linalg.norm(prototype)) <= 1e-8:
                raise ObjectGraphContractError(
                    "Support produced a zero RGB landmark prototype."
                )
            prototypes.append(prototype)
        self._prototypes = np.ascontiguousarray(np.stack(prototypes), dtype=np.float32)

        support_evidence = []
        for feature_map, pose in zip(feature_maps, self._support_poses):
            maps = self._score_maps(feature_map)
            support_evidence.append(self._pose_rgb_evidence(pose, maps)[0])
        self._support_evidence = np.asarray(support_evidence, dtype=np.float64)
        self._rgb_reliability_threshold = float(
            np.clip(np.quantile(self._support_evidence, 0.1) - 0.10, 0.55, 0.92)
        )
        self._support_rgb_sha256 = _typed_array_sha256(self._support_rgb)
        self._support_mask_sha256 = _typed_array_sha256(self._support_indexed_masks)
        self._prototype_sha256 = _typed_array_sha256(self._prototypes)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "protocol": PROTOCOL,
            "graph_sha256": self.graph.graph_sha256,
            "entity_name": self.entity.name,
            "role_names": list(self.entity.projector.roles),
            "max_candidates": MAX_CANDIDATES,
            "maximum_tip_proposals": MAX_TIP_PROPOSALS,
            "maximum_rendered_ik_candidates": MAX_RENDERED_IK_CANDIDATES,
            "candidate_slot_semantics": [
                "exact_current_mask_v1_anchor",
                "best_reliable_distinct_current_rgb_ik_mode_or_invalid",
            ],
            "source_codes": dict(SOURCE_CODES),
            "support_rgb_trace_sha256": self._support_rgb_sha256,
            "support_indexed_mask_trace_sha256": self._support_mask_sha256,
            "prototype_trace_sha256": self._prototype_sha256,
            "landmark_names": list(LANDMARK_NAMES),
            "support_resolution": list(self._resolution),
            "support_link_lengths_px": self._link_lengths.tolist(),
            "support_link_half_widths_px": self._link_half_widths.tolist(),
            "rgb_reliability_threshold": self._rgb_reliability_threshold,
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
            "spatial_shuffle_policy": (
                "deterministic_roi_pixel_permutation_preserving_rgb_histogram_v1"
            ),
        }

    def _score_maps(self, feature_map: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: _similarity_map(feature_map, prototype, self._resolution)
            for name, prototype in zip(LANDMARK_NAMES, self._prototypes)
        }

    def _pose_rgb_evidence(
        self,
        pose_xy: np.ndarray,
        score_maps: Mapping[str, np.ndarray],
    ) -> tuple[float, dict[str, float]]:
        pose = np.asarray(pose_xy, dtype=np.float64)
        root = _point_score(score_maps["root"], pose[0])
        joint = _point_score(score_maps["joint"], pose[1])
        tip = _point_score(score_maps["tip"], pose[2])
        first = _line_score(score_maps["first_midpoint"], pose[0], pose[1])
        second = _line_score(score_maps["second_midpoint"], pose[1], pose[2])
        first_cross = _line_score(score_maps["second_midpoint"], pose[0], pose[1])
        second_cross = _line_score(score_maps["first_midpoint"], pose[1], pose[2])
        landmark = (root + joint + tip) / 3.0
        link = (first + second) / 2.0
        contrast = float(
            np.clip(
                0.5 + 0.25 * ((first - first_cross) + (second - second_cross)),
                0.0,
                1.0,
            )
        )
        evidence = float(np.clip(0.35 * landmark + 0.50 * link + 0.15 * contrast, 0.0, 1.0))
        return evidence, {
            "root": root,
            "joint": joint,
            "tip": tip,
            "first_link": first,
            "second_link": second,
            "role_contrast": contrast,
        }

    def _roi(self, entity_mask: np.ndarray) -> np.ndarray:
        height, width = entity_mask.shape
        radius = float(self._link_lengths.sum() + 3.0 * self._link_half_widths.max())
        points = [self._root_xy]
        yx = np.argwhere(entity_mask)
        if len(yx):
            xy = yx[:, ::-1].astype(np.float64)
            points.extend((xy.min(axis=0), xy.max(axis=0)))
        point_array = np.stack(points)
        x0 = max(int(math.floor(min(float(point_array[:, 0].min()), self._root_xy[0] - radius))), 0)
        y0 = max(int(math.floor(min(float(point_array[:, 1].min()), self._root_xy[1] - radius))), 0)
        x1 = min(int(math.ceil(max(float(point_array[:, 0].max()) + 1.0, self._root_xy[0] + radius + 1.0))), width)
        y1 = min(int(math.ceil(max(float(point_array[:, 1].max()) + 1.0, self._root_xy[1] + radius + 1.0))), height)
        return np.asarray([x0, y0, x1, y1], dtype=np.int32)

    def _current_root(
        self, root_map: np.ndarray, roi_xyxy: np.ndarray
    ) -> np.ndarray:
        height, width = root_map.shape
        yy, xx = np.indices((height, width), dtype=np.float64)
        radius = max(2.5 * self._root_scale_px, 2.0 * float(self._link_half_widths.max()), 2.0)
        distance = np.sqrt((xx - self._root_xy[0]) ** 2 + (yy - self._root_xy[1]) ** 2)
        x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
        allowed = (distance <= radius)
        allowed[:y0] = False
        allowed[y1:] = False
        allowed[:, :x0] = False
        allowed[:, x1:] = False
        if not allowed.any():
            return self._root_xy.copy()
        objective = root_map.astype(np.float64) - 0.08 * distance / radius
        objective[~allowed] = -np.inf
        flat = int(np.argmax(objective))
        y, x = np.unravel_index(flat, root_map.shape)
        detected = np.asarray([float(x), float(y)], dtype=np.float64)
        # Retain subpixel support calibration unless appearance gives material
        # evidence for a small current-frame displacement.
        if _point_score(root_map, self._root_xy) >= float(root_map[y, x]) - 0.025:
            return self._root_xy.copy()
        return detected

    def _tip_points(
        self,
        tip_map: np.ndarray,
        second_map: np.ndarray,
        root_xy: np.ndarray,
        roi_xyxy: np.ndarray,
    ) -> list[np.ndarray]:
        combined = 0.70 * tip_map.astype(np.float64) + 0.30 * second_map.astype(np.float64)
        height, width = combined.shape
        x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
        allowed = np.zeros((height, width), dtype=np.bool_)
        allowed[y0:y1, x0:x1] = True
        yy, xx = np.indices((height, width), dtype=np.float64)
        radial = np.sqrt((xx - root_xy[0]) ** 2 + (yy - root_xy[1]) ** 2)
        minimum = abs(float(self._link_lengths[0] - self._link_lengths[1])) + 0.2
        maximum = float(self._link_lengths.sum()) - 0.2
        allowed &= (radial >= minimum) & (radial <= maximum)
        values = combined[allowed]
        if not len(values):
            return []
        maximum_score = float(values.max())
        threshold = max(float(np.quantile(values, 0.88)), maximum_score - 0.10)
        selected = allowed & (combined >= threshold)
        selected_yx = np.argwhere(selected)
        if not len(selected_yx):
            return []
        selected_xy = selected_yx[:, ::-1].astype(np.float64)
        selected_score = combined[selected_yx[:, 0], selected_yx[:, 1]]
        proposed: list[np.ndarray] = []

        # Principal-axis cap centres recover line endpoints from a homogeneous
        # role-colored region instead of treating an arbitrary high-score
        # pixel as the tip.
        if len(selected_xy) >= 3:
            centred = selected_xy - selected_xy.mean(axis=0)
            covariance = centred.T @ centred / max(len(centred), 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            projection = selected_xy @ axis
            span = float(projection.max() - projection.min())
            cap = max(1.0, 0.08 * span)
            for bound, comparison in (
                (float(projection.min() + cap), np.less_equal),
                (float(projection.max() - cap), np.greater_equal),
            ):
                members = selected_xy[comparison(projection, bound)]
                if len(members):
                    proposed.append(members.mean(axis=0))

        directions = np.stack(
            [
                np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
                for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
            ]
        )
        for direction in directions:
            projection = selected_xy @ direction
            proposed.append(selected_xy[int(np.argmax(projection))])

        order = np.lexsort((selected_yx[:, 1], selected_yx[:, 0], -selected_score))
        for index in order[: min(len(order), MAX_TIP_PROPOSALS)]:
            proposed.append(selected_xy[int(index)])

        deduplicated: list[np.ndarray] = []
        minimum_separation = max(1.0, 0.35 * float(self._link_half_widths.min()))
        for point in proposed:
            delta = np.asarray(point, dtype=np.float64) - root_xy
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                continue
            clamped = root_xy + delta * (float(np.clip(distance, minimum, maximum)) / distance)
            if any(float(np.linalg.norm(clamped - previous)) < minimum_separation for previous in deduplicated):
                continue
            deduplicated.append(np.ascontiguousarray(clamped))
            if len(deduplicated) >= MAX_TIP_PROPOSALS:
                break
        return deduplicated

    @staticmethod
    def _bend(pose_xy: np.ndarray) -> int:
        first, second = np.diff(np.asarray(pose_xy, dtype=np.float64), axis=0)
        scale = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
        sine = float((first[0] * second[1] - first[1] * second[0]) / scale)
        return 0 if abs(sine) < 0.08 else (1 if sine > 0.0 else -1)

    def _different_mode(self, left: np.ndarray, right: np.ndarray) -> bool:
        if self._bend(left) != self._bend(right):
            return True
        distance = float(np.linalg.norm(left - right, axis=1).mean())
        return distance > 0.05 * float(self._link_lengths.sum())

    def _enumerate_rgb(
        self,
        score_maps: Mapping[str, np.ndarray],
        entity_mask: np.ndarray,
        roi_xyxy: np.ndarray,
    ) -> tuple[list[_RGBCandidate], dict[str, Any]]:
        root_xy = self._current_root(score_maps["root"], roi_xyxy)
        tip_points = self._tip_points(
            score_maps["tip"], score_maps["second_midpoint"], root_xy, roi_xyxy
        )
        proposals: list[tuple[np.ndarray, float, str, int, int]] = []
        for tip_xy in tip_points:
            elbows = _circle_intersections(
                root_xy,
                tip_xy,
                float(self._link_lengths[0]),
                float(self._link_lengths[1]),
            )
            for elbow_index, elbow_xy in enumerate(elbows):
                source = "rgb_ik_positive" if elbow_index == 0 else "rgb_ik_negative"
                pose = np.ascontiguousarray(
                    np.stack((root_xy, elbow_xy, tip_xy)), dtype=np.float64
                )
                evidence, _ = self._pose_rgb_evidence(pose, score_maps)
                proposals.append(
                    (
                        pose,
                        evidence,
                        source,
                        SOURCE_CODES[source],
                        self._bend(pose),
                    )
                )
        ordered_proposals = sorted(
            proposals,
            key=lambda item: (
                -item[1],
                item[3],
                tuple(float(value) for value in item[0].reshape(-1)),
            ),
        )
        # Rendering full-resolution capsules dominates the lightweight
        # structure step.  The output needs one alternate mode, so preserve a
        # deterministic bend-diverse evidence beam and render at most eight.
        shortlist: list[tuple[np.ndarray, float, str, int, int]] = []
        selected_ids: set[int] = set()
        per_bend = {-1: 0, 0: 0, 1: 0}
        for proposal in ordered_proposals:
            bend = proposal[4]
            if per_bend[bend] >= 2:
                continue
            shortlist.append(proposal)
            selected_ids.add(id(proposal))
            per_bend[bend] += 1
        for proposal in ordered_proposals:
            if len(shortlist) >= MAX_RENDERED_IK_CANDIDATES:
                break
            if id(proposal) not in selected_ids:
                shortlist.append(proposal)
                selected_ids.add(id(proposal))
        shortlist = sorted(
            shortlist[:MAX_RENDERED_IK_CANDIDATES],
            key=lambda item: (
                -item[1],
                item[3],
                tuple(float(value) for value in item[0].reshape(-1)),
            ),
        )
        root_cost = float(
            np.linalg.norm(root_xy - self._root_xy)
            / max(self._root_scale_px, 1.0)
        )
        candidates: list[_RGBCandidate] = []
        for pose, evidence, source, source_code, bend in shortlist:
            rendered = render_role_capsules(
                pose, self._link_half_widths, entity_mask.shape
            )
            mask_geometry = _mask_iou(rendered, entity_mask)
            cost = float(
                2.0 * (1.0 - evidence)
                + 0.12 * (1.0 - mask_geometry)
                + 0.03 * root_cost
            )
            confidence = float(
                np.clip(
                    (evidence - self._rgb_reliability_threshold)
                    / max(1.0 - self._rgb_reliability_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                _RGBCandidate(
                    pose_xy=pose,
                    cost=cost,
                    confidence=confidence,
                    source=source,
                    source_code=source_code,
                    rgb_evidence=evidence,
                    mask_geometry=mask_geometry,
                    bend=bend,
                )
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.rgb_evidence,
                item.cost,
                item.source_code,
                tuple(float(value) for value in item.pose_xy.reshape(-1)),
            ),
        )
        return ordered, {
            "detected_root_xy": root_xy.tolist(),
            "tip_proposals": len(tip_points),
            "ik_candidates": len(proposals),
            "rendered_ik_candidates": len(candidates),
        }

    def project(
        self,
        *,
        current_rgb: np.ndarray,
        entity_mask: np.ndarray,
        entity_available: bool,
        spatial_shuffle: bool = False,
    ) -> OrderedChainRGBFrame:
        start = perf_counter()
        rgb = _validate_rgb(current_rgb, "current_rgb")
        if tuple(rgb.shape[:2]) != self._resolution:
            raise ValueError("current_rgb resolution differs from support calibration.")
        raw_mask = np.asarray(entity_mask)
        if raw_mask.dtype != np.bool_:
            raise TypeError("entity_mask must have exact boolean dtype.")
        if raw_mask.ndim != 2:
            raise ValueError("entity_mask must have shape [H,W].")
        if tuple(raw_mask.shape) != self._resolution:
            raise ValueError("entity_mask resolution differs from support calibration.")
        if type(entity_available) is not bool:
            raise TypeError("entity_available must be bool.")
        if type(spatial_shuffle) is not bool:
            raise TypeError("spatial_shuffle must be bool.")
        mask = np.ascontiguousarray(raw_mask)
        roi_xyxy = self._roi(mask)

        poses = np.zeros((MAX_CANDIDATES, 3, 2), dtype=np.float32)
        widths = np.zeros((MAX_CANDIDATES, 2), dtype=np.float32)
        valid = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
        cost = np.full(MAX_CANDIDATES, np.finfo(np.float32).max, dtype=np.float32)
        weights = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        confidence = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        source_codes = np.zeros(MAX_CANDIDATES, dtype=np.uint8)
        rgb_scores = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        mask_scores = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        diagnostics: dict[str, Any] = {
            "format": FORMAT,
            "protocol": PROTOCOL,
            "failure": None,
            "spatial_shuffle": spatial_shuffle,
            "current_frame_only": True,
            "relative_weights_not_calibrated": True,
            "rgb_reliability_threshold": self._rgb_reliability_threshold,
            "anchor_slot_reserved": True,
            "rgb_slot_reserved": True,
        }
        active: list[int] = []
        if not entity_available:
            diagnostics["failure"] = "entity_unavailable"
        else:
            anchor = self._anchor.project(
                entity_mask=mask, entity_available=True
            )
            feature_rgb = _spatial_shuffle_roi(rgb, roi_xyxy) if spatial_shuffle else rgb
            feature_map = _extract_dense(self.feature_extractor, feature_rgb)
            if int(feature_map.shape[-1]) != int(self._prototypes.shape[-1]):
                raise ObjectGraphContractError(
                    "Current feature dimension differs from support prototypes."
                )
            score_maps = self._score_maps(feature_map)
            if bool(anchor.valid[0]):
                anchor_pose = np.ascontiguousarray(anchor.poses_xy[0], dtype=np.float64)
                anchor_evidence, anchor_detail = self._pose_rgb_evidence(
                    anchor_pose, score_maps
                )
                anchor_render = render_role_capsules(
                    anchor_pose, self._link_half_widths, mask.shape
                )
                poses[0] = anchor.poses_xy[0]
                widths[0] = self._link_half_widths.astype(np.float32)
                valid[0] = True
                cost[0] = anchor.costs[0]
                confidence[0] = anchor.confidence[0]
                source_codes[0] = np.uint8(SOURCE_CODES["v1_anchor"])
                rgb_scores[0] = np.float32(anchor_evidence)
                mask_scores[0] = np.float32(_mask_iou(anchor_render, mask))
                active.append(0)
                diagnostics["anchor_rgb_components"] = anchor_detail
            candidates, rgb_diagnostics = self._enumerate_rgb(
                score_maps, mask, roi_xyxy
            )
            diagnostics.update(rgb_diagnostics)
            selected: _RGBCandidate | None = None
            rejected_unreliable = 0
            rejected_same_mode = 0
            for candidate in candidates:
                if candidate.rgb_evidence + 1e-12 < self._rgb_reliability_threshold:
                    rejected_unreliable += 1
                    continue
                if valid[0] and not self._different_mode(
                    candidate.pose_xy, poses[0].astype(np.float64)
                ):
                    rejected_same_mode += 1
                    continue
                selected = candidate
                break
            diagnostics["rgb_rejected_unreliable"] = rejected_unreliable
            diagnostics["rgb_rejected_same_mode"] = rejected_same_mode
            if selected is not None:
                poses[1] = selected.pose_xy.astype(np.float32)
                widths[1] = self._link_half_widths.astype(np.float32)
                valid[1] = True
                cost[1] = np.float32(selected.cost)
                confidence[1] = np.float32(selected.confidence)
                source_codes[1] = np.uint8(selected.source_code)
                rgb_scores[1] = np.float32(selected.rgb_evidence)
                mask_scores[1] = np.float32(selected.mask_geometry)
                active.append(1)
                diagnostics["rgb_selected_source"] = selected.source
                diagnostics["rgb_selected_bend"] = selected.bend
            else:
                diagnostics["rgb_selected_source"] = None
                if not active:
                    diagnostics["failure"] = "no_reliable_current_frame_candidate"

        if active:
            logits = -cost[np.asarray(active, dtype=np.int64)].astype(np.float64)
            logits = logits / _WEIGHT_TEMPERATURE
            logits -= float(logits.max())
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            for index, probability in zip(active, probabilities):
                weights[index] = np.float32(probability)
        diagnostics["candidate_slots"] = active
        diagnostics["candidate_count"] = len(active)
        diagnostics["fail_closed"] = not bool(active)
        diagnostics["spatial_shuffle_trace_sha256"] = (
            _typed_array_sha256(_spatial_shuffle_roi(rgb, roi_xyxy))
            if spatial_shuffle
            else None
        )
        runtime_ms = max((perf_counter() - start) * 1000.0, 1e-9)
        return OrderedChainRGBFrame(
            role_names=(
                self.entity.projector.roles[0],
                self.entity.projector.roles[1],
            ),
            poses_xy=np.ascontiguousarray(poses),
            link_half_widths_px=np.ascontiguousarray(widths),
            valid=np.ascontiguousarray(valid),
            cost=np.ascontiguousarray(cost),
            relative_weights=np.ascontiguousarray(weights),
            confidence=np.ascontiguousarray(confidence),
            source_codes=np.ascontiguousarray(source_codes),
            rgb_evidence_score=np.ascontiguousarray(rgb_scores),
            mask_geometry_score=np.ascontiguousarray(mask_scores),
            roi_xyxy=np.ascontiguousarray(roi_xyxy),
            runtime_ms=float(runtime_ms),
            diagnostics=diagnostics,
        )


__all__ = [
    "DenseFeatureExtractor",
    "FORMAT",
    "FrozenDinoV2FeatureExtractor",
    "LANDMARK_NAMES",
    "MAX_CANDIDATES",
    "MAX_RENDERED_IK_CANDIDATES",
    "MAX_TIP_PROPOSALS",
    "OrderedChainRGBFrame",
    "OrderedChainRGBGenerator",
    "PROTOCOL",
    "SOURCE_CODES",
    "render_role_capsules",
]
