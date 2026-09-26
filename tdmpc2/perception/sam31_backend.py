"""SAM 3.1 fixed-support backend for frozen perception evaluation.

This module intentionally has two layers:

* dependency-light validation and deterministic conversion of the six fixed
  RGB+indexed-mask support exemplars into point prompts; and
* a lazy official-SAM-3.1 runtime used only by the dedicated worker process.

The public SAM 3.1 video predictor does not expose a full-mask prompt in its
``handle_request`` API.  It does expose instance point prompts with stable
``obj_id`` values.  Consequently this backend uses the explicitly named
``fixed_support_point_replay_v1`` protocol: every fixed support mask is
converted into deterministic positive and negative points, all support frames
are prepended to the RGB-only episode, and the episode is propagated forward
one frame per call.  It is *not* equivalent to a mask-prompt benchmark.

Episode ground-truth masks are deliberately absent from every function in this
module.  They belong only in the parent evaluator after the worker has written
its predictions.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BACKEND_FORMAT = "sam31_fixed_support_point_replay_v1"
PROMPT_PROTOCOL = "fixed_support_point_replay_v1"
PROMPT_SAMPLER = "mask_centroid_farthest_points_relcoords_v1"
CAUSAL_PROTOCOL = "forward_one_effective_frame_per_call_v2"
EXPECTED_SUPPORT_FRAMES = 6


class Sam31BackendError(RuntimeError):
    """Base error for an invalid SAM 3.1 backend run."""


class Sam31ContractError(Sam31BackendError):
    """Raised when frozen inputs or backend outputs violate the contract."""


class Sam31DependencyError(Sam31BackendError):
    """Raised when the isolated official SAM 3.1 environment is incomplete."""


@dataclass(frozen=True)
class FrozenSupport:
    rgb: np.ndarray
    indexed_masks: np.ndarray
    role_names: tuple[str, ...]
    source_path: Path
    rgb_sha256: str
    masks_sha256: str


@dataclass(frozen=True)
class SupportPointPrompt:
    frame_index: int
    role_index: int
    role_name: str
    object_id: int
    points_xy: np.ndarray
    labels: np.ndarray

    def as_json(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "role_index": int(self.role_index),
            "role_name": self.role_name,
            "object_id": int(self.object_id),
            "points_xy": np.round(self.points_xy, decimals=9).tolist(),
            "labels": self.labels.astype(np.int32, copy=False).tolist(),
        }


@dataclass(frozen=True)
class OfficialSam31Config:
    repo_path: Path
    checkpoint_path: Path
    bpe_path: Path
    role_names: tuple[str, ...]
    device: str = "cuda:0"
    use_fa3: bool = False
    use_rope_real: bool = False
    compile_model: bool = False
    output_probability_threshold: float = 0.5

    def validated(self) -> "OfficialSam31Config":
        roles = validate_role_names(self.role_names)
        if roles != self.role_names:
            raise Sam31ContractError("role_names must already be a tuple of strings")
        if not re.fullmatch(r"cuda(?::\d+)?", self.device):
            raise Sam31ContractError(
                f"SAM 3.1 requires an explicit CUDA device, got {self.device!r}."
            )
        if not 0.0 <= float(self.output_probability_threshold) <= 1.0:
            raise Sam31ContractError("output_probability_threshold must be in [0,1]")
        return self


def validate_role_names(role_names: Iterable[str]) -> tuple[str, ...]:
    roles = tuple(role_names)
    if not roles or any(not isinstance(role, str) or not role.strip() for role in roles):
        raise Sam31ContractError("role names must be non-empty strings")
    if len(set(roles)) != len(roles):
        raise Sam31ContractError(f"role names must be unique, got {roles!r}")
    if len(roles) > 255:
        raise Sam31ContractError("indexed uint8 output supports at most 255 roles")
    return roles


def _typed_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def typed_array_sha256(array: np.ndarray) -> str:
    """Hash an array together with its dtype and shape.

    This is the trace convention used by the unified VOS scorer.  Keeping it
    here lets the isolated worker bind the scientific arrays independently of
    the compressed-NPZ byte hash.
    """
    return _typed_sha256(array)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_rgb(value: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise Sam31ContractError(
            f"{name} must be uint8 [T,H,W,3], got {value.dtype} {value.shape}."
        )
    if value.shape[0] < 1 or min(value.shape[1:3]) < 1:
        raise Sam31ContractError(f"{name} must contain non-empty frames")
    return np.ascontiguousarray(value)


def load_frozen_support(
    support_path: str | Path,
    *,
    role_names: Iterable[str],
    expected_frames: int = EXPECTED_SUPPORT_FRAMES,
) -> FrozenSupport:
    """Load the fixed support pack; no episode file is accepted here."""
    roles = validate_role_names(role_names)
    path = Path(support_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise Sam31ContractError(f"support NPZ is missing or empty: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "rgb" not in archive.files or "indexed_masks" not in archive.files:
                raise Sam31ContractError(
                    "support NPZ requires rgb and indexed_masks members"
                )
            rgb = _require_rgb(np.array(archive["rgb"], copy=True), name="support.rgb")
            indexed = np.array(archive["indexed_masks"], copy=True)
    except Sam31ContractError:
        raise
    except Exception as exc:
        raise Sam31ContractError(f"failed to decode support NPZ {path}: {exc}") from exc

    if type(expected_frames) is not int or expected_frames < 1:
        raise Sam31ContractError("expected_frames must be a positive integer")
    if rgb.shape[0] != expected_frames:
        raise Sam31ContractError(
            f"support must contain exactly {expected_frames} frames, got {rgb.shape[0]}"
        )
    if indexed.dtype != np.uint8 or indexed.shape != rgb.shape[:3]:
        raise Sam31ContractError(
            "support.indexed_masks must be uint8 [T,H,W] aligned with support.rgb; "
            f"got {indexed.dtype} {indexed.shape} vs {rgb.shape[:3]}"
        )
    indexed = np.ascontiguousarray(indexed)
    values = set(np.unique(indexed).tolist())
    allowed = set(range(len(roles) + 1))
    if not values.issubset(allowed):
        raise Sam31ContractError(
            f"support indexed IDs must be within {sorted(allowed)}, got {sorted(values)}"
        )
    for frame_index, mask in enumerate(indexed):
        missing = [
            role
            for role_index, role in enumerate(roles, start=1)
            if not np.any(mask == role_index)
        ]
        if missing:
            raise Sam31ContractError(
                f"support frame {frame_index} has no pixels for roles {missing!r}"
            )
    return FrozenSupport(
        rgb=rgb,
        indexed_masks=indexed,
        role_names=roles,
        source_path=path,
        rgb_sha256=_typed_sha256(rgb),
        masks_sha256=_typed_sha256(indexed),
    )


def load_episode_rgb(episode_path: str | Path) -> tuple[np.ndarray, Path, str]:
    """Read only the ``rgb`` member of a frozen episode NPZ.

    The scorer owns any ground-truth members.  This loader neither accepts nor
    indexes a ground-truth key.
    """
    path = Path(episode_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise Sam31ContractError(f"episode NPZ is missing or empty: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "rgb" not in archive.files:
                raise Sam31ContractError("episode NPZ requires an rgb member")
            rgb = _require_rgb(np.array(archive["rgb"], copy=True), name="episode.rgb")
    except Sam31ContractError:
        raise
    except Exception as exc:
        raise Sam31ContractError(f"failed to decode episode RGB {path}: {exc}") from exc
    return rgb, path, _typed_sha256(rgb)


def _farthest_pixel_sample(mask: np.ndarray, count: int) -> np.ndarray:
    """Select deterministic (x,y) pixel centres, starting nearest the centroid."""
    if type(count) is not int or count < 0:
        raise Sam31ContractError("point counts must be non-negative integers")
    coordinates_yx = np.argwhere(np.asarray(mask, dtype=bool))
    if count == 0 or coordinates_yx.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    count = min(count, len(coordinates_yx))
    values = coordinates_yx.astype(np.float64)
    centroid = values.mean(axis=0)
    first = int(np.argmin(np.square(values - centroid).sum(axis=1)))
    chosen = [first]
    min_distance = np.square(values - values[first]).sum(axis=1)
    min_distance[first] = -1.0
    while len(chosen) < count:
        next_index = int(np.argmax(min_distance))
        if min_distance[next_index] < 0:
            break
        chosen.append(next_index)
        distance = np.square(values - values[next_index]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[np.asarray(chosen, dtype=np.int64)] = -1.0
    # Convert y,x integer pixels to x,y pixel centres.  Relative conversion is
    # applied separately so the sampling hash also binds the original raster.
    sampled_yx = values[np.asarray(chosen, dtype=np.int64)]
    return np.stack((sampled_yx[:, 1] + 0.5, sampled_yx[:, 0] + 0.5), axis=1).astype(
        np.float32
    )


def mask_to_relative_point_prompt(
    role_mask: np.ndarray,
    *,
    positive_points: int = 4,
    negative_points: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one support role mask into fixed positive/negative point prompts."""
    mask = np.asarray(role_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise Sam31ContractError(
            f"role mask must be bool [H,W], got {mask.dtype} {mask.shape}"
        )
    if not mask.any():
        raise Sam31ContractError("cannot prompt an empty support role mask")
    height, width = mask.shape
    positive = _farthest_pixel_sample(mask, positive_points)
    negative = _farthest_pixel_sample(~mask, negative_points)
    points = np.concatenate((positive, negative), axis=0)
    if points.shape[0] < 1:
        raise Sam31ContractError("support prompt unexpectedly contains no points")
    points[:, 0] /= float(width)
    points[:, 1] /= float(height)
    if not np.isfinite(points).all() or np.any(points <= 0) or np.any(points >= 1):
        raise Sam31ContractError("relative support points must be finite and inside (0,1)")
    labels = np.concatenate(
        (
            np.ones(len(positive), dtype=np.int32),
            np.zeros(len(negative), dtype=np.int32),
        )
    )
    return np.ascontiguousarray(points, dtype=np.float32), labels


def build_support_point_prompts(
    support: FrozenSupport,
    *,
    positive_points: int = 4,
    negative_points: int = 4,
) -> tuple[tuple[SupportPointPrompt, ...], str]:
    prompts: list[SupportPointPrompt] = []
    for frame_index, indexed in enumerate(support.indexed_masks):
        for role_index, role_name in enumerate(support.role_names, start=1):
            points, labels = mask_to_relative_point_prompt(
                indexed == role_index,
                positive_points=positive_points,
                negative_points=negative_points,
            )
            prompts.append(
                SupportPointPrompt(
                    frame_index=frame_index,
                    role_index=role_index - 1,
                    role_name=role_name,
                    object_id=role_index,
                    points_xy=points,
                    labels=labels,
                )
            )
    encoded = json.dumps(
        [prompt.as_json() for prompt in prompts],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return tuple(prompts), hashlib.sha256(encoded).hexdigest()


def decode_official_frame_output(
    outputs: Mapping[str, Any],
    *,
    role_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Map sparse official object IDs to a dense, role-ordered result."""
    if not isinstance(outputs, Mapping):
        raise Sam31ContractError("official SAM 3.1 frame output must be a mapping")
    try:
        object_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int64).reshape(-1)
        masks = np.asarray(outputs["out_binary_masks"])
        probabilities = np.asarray(outputs["out_probs"], dtype=np.float32).reshape(-1)
    except KeyError as exc:
        raise Sam31ContractError(f"official output lacks {exc.args[0]!r}") from exc
    if masks.dtype != np.bool_:
        masks = masks.astype(bool, copy=False)
    if masks.shape != (len(object_ids), height, width):
        raise Sam31ContractError(
            f"official masks must be [N,{height},{width}], got {masks.shape}"
        )
    if probabilities.shape != object_ids.shape:
        raise Sam31ContractError("official out_probs and out_obj_ids are not aligned")
    if not np.isfinite(probabilities).all():
        raise Sam31ContractError("official out_probs contains non-finite values")
    if len(set(object_ids.tolist())) != len(object_ids):
        raise Sam31ContractError("official output contains duplicate object IDs")
    if any(object_id < 1 or object_id > role_count for object_id in object_ids):
        raise Sam31ContractError(
            f"official object IDs must be within [1,{role_count}], got {object_ids.tolist()}"
        )

    dense_masks = np.zeros((role_count, height, width), dtype=bool)
    confidence = np.zeros(role_count, dtype=np.float32)
    for object_id, mask, probability in zip(object_ids, masks, probabilities):
        role_index = int(object_id) - 1
        dense_masks[role_index] = mask
        confidence[role_index] = np.clip(probability, 0.0, 1.0)

    overlap_pixels = int((dense_masks.sum(axis=0) > 1).sum())
    if overlap_pixels:
        scores = np.where(
            dense_masks,
            confidence[:, None, None],
            np.float32(-1.0),
        )
        winner = scores.argmax(axis=0)
        covered = dense_masks.any(axis=0)
        dense_masks[...] = False
        for role_index in range(role_count):
            dense_masks[role_index] = covered & (winner == role_index)
    valid = dense_masks.reshape(role_count, -1).any(axis=1)
    return dense_masks, confidence, valid, overlap_pixels


def _parse_version_prefix(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value))
    if match is None:
        raise Sam31DependencyError(f"cannot parse version {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def _official_source_tree_snapshot(repo: Path) -> dict[str, Any]:
    """Bind every official Python/config source that can affect inference."""
    suffixes = {".py", ".json", ".toml", ".yaml", ".yml"}
    roots = [repo / "sam3", repo / "configs"]
    paths = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    )
    project_file = repo / "pyproject.toml"
    if project_file.is_file() and project_file not in paths:
        paths.append(project_file)
        paths.sort()
    if not paths:
        raise Sam31DependencyError(
            f"official SAM 3.1 source/config tree is empty: {repo}"
        )
    records = [
        {
            "path": path.relative_to(repo).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def inspect_official_installation(config: OfficialSam31Config) -> dict[str, Any]:
    """Fail-fast preflight without downloading repositories or weights."""
    config = config.validated()
    repo = config.repo_path.expanduser().resolve()
    checkpoint = config.checkpoint_path.expanduser().resolve()
    bpe = config.bpe_path.expanduser().resolve()
    if not repo.is_dir():
        raise Sam31DependencyError(f"official SAM 3.1 repo is missing: {repo}")
    required = {
        "model_builder": repo / "sam3" / "model_builder.py",
        "multiplex_predictor": repo
        / "sam3"
        / "model"
        / "sam3_multiplex_video_predictor.py",
        "base_predictor": repo / "sam3" / "model" / "sam3_base_predictor.py",
        "multiplex_tracking": repo
        / "sam3"
        / "model"
        / "sam3_multiplex_tracking.py",
        "multiplex_base": repo / "sam3" / "model" / "sam3_multiplex_base.py",
        "io_utils": repo / "sam3" / "model" / "io_utils.py",
        "checkpoint": checkpoint,
        "bpe": bpe,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    empty = [name for name, path in required.items() if path.is_file() and path.stat().st_size <= 0]
    if missing or empty:
        raise Sam31DependencyError(
            f"incomplete official SAM 3.1 installation; missing={missing}, empty={empty}"
        )
    if sys.version_info < (3, 12):
        raise Sam31DependencyError(
            "official SAM 3.1 requires Python >=3.12; use its isolated environment"
        )
    try:
        import torch
    except Exception as exc:
        raise Sam31DependencyError(f"PyTorch import failed: {exc}") from exc
    if _parse_version_prefix(torch.__version__) < (2, 7, 0):
        raise Sam31DependencyError(
            f"official SAM 3.1 requires torch>=2.7, got {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise Sam31DependencyError("official SAM 3.1 requires a CUDA GPU")
    if torch.version.cuda is None or _parse_version_prefix(torch.version.cuda) < (12, 6, 0):
        raise Sam31DependencyError(
            f"official SAM 3.1 requires CUDA>=12.6, got {torch.version.cuda!r}"
        )
    device_index = int(config.device.split(":", 1)[1]) if ":" in config.device else 0
    if device_index >= torch.cuda.device_count():
        raise Sam31DependencyError(
            f"configured {config.device} but only {torch.cuda.device_count()} logical GPUs exist"
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": config.device,
        "device_name": torch.cuda.get_device_name(device_index),
        "repo": str(repo),
        "source_tree": _official_source_tree_snapshot(repo),
        "model_builder_sha256": sha256_file(required["model_builder"]),
        "multiplex_predictor_sha256": sha256_file(
            required["multiplex_predictor"]
        ),
        "base_predictor_sha256": sha256_file(required["base_predictor"]),
        "multiplex_tracking_sha256": sha256_file(
            required["multiplex_tracking"]
        ),
        "multiplex_base_sha256": sha256_file(required["multiplex_base"]),
        "io_utils_sha256": sha256_file(required["io_utils"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "bpe": str(bpe),
        "bpe_sha256": sha256_file(bpe),
        "official_api_mask_prompt": False,
        "prompt_api_not_equivalent": True,
    }


class OfficialSam31Runtime:
    """Thin wrapper around the official SAM 3.1 multiplex predictor.

    This class is constructed inside the isolated worker, never in TD-MPC2's
    training process.
    """

    def __init__(self, config: OfficialSam31Config):
        self.config = config.validated()
        self.installation = inspect_official_installation(self.config)
        import torch

        self._torch = torch
        device_index = (
            int(self.config.device.split(":", 1)[1]) if ":" in self.config.device else 0
        )
        torch.cuda.set_device(device_index)
        self._predictor = self._build_predictor()
        self.installation["checkpoint_contract"] = self._checkpoint_contract()
        self._session_id: str | None = None
        self.start_session_dispatch: str | None = None

    def _build_predictor(self):
        repo = self.config.repo_path.expanduser().resolve()
        repo_string = str(repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        try:
            builder_module = importlib.import_module("sam3.model_builder")
        except Exception as exc:
            raise Sam31DependencyError(
                f"failed to import official sam3.model_builder from {repo}: {exc}"
            ) from exc
        module_path = Path(builder_module.__file__).resolve()
        try:
            module_path.relative_to(repo)
        except ValueError as exc:
            raise Sam31DependencyError(
                f"imported sam3 from {module_path}, not configured repo {repo}"
            ) from exc
        builder = getattr(builder_module, "build_sam3_multiplex_video_predictor", None)
        if not callable(builder):
            raise Sam31DependencyError(
                "official repo lacks build_sam3_multiplex_video_predictor"
            )
        kwargs = {
            "checkpoint_path": str(self.config.checkpoint_path.expanduser().resolve()),
            "bpe_path": str(self.config.bpe_path.expanduser().resolve()),
            "max_num_objects": max(16, len(self.config.role_names)),
            "multiplex_count": 16,
            "use_fa3": bool(self.config.use_fa3),
            "use_rope_real": bool(self.config.use_rope_real),
            "compile": bool(self.config.compile_model),
            "warm_up": False,
            "default_output_prob_thresh": float(
                self.config.output_probability_threshold
            ),
            "async_loading_frames": False,
        }
        signature = inspect.signature(builder)
        unsupported = sorted(set(kwargs) - set(signature.parameters))
        if unsupported:
            raise Sam31DependencyError(
                f"official SAM 3.1 builder API is incompatible; missing kwargs {unsupported}"
            )
        try:
            predictor = builder(**kwargs)
        except Exception as exc:
            raise Sam31DependencyError(
                f"official SAM 3.1 model construction failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not callable(getattr(predictor, "handle_request", None)) or not callable(
            getattr(predictor, "handle_stream_request", None)
        ):
            raise Sam31DependencyError("official predictor lacks request/stream APIs")
        return predictor

    def _checkpoint_contract(self) -> dict[str, Any]:
        """Require the supplied checkpoint to exactly cover the built model.

        The official multiplex builder currently calls ``load_state_dict`` with
        ``strict=False``.  That is convenient for development checkpoints, but
        unsafe for a benchmark: a SAM 3 (rather than SAM 3.1) checkpoint can
        otherwise construct a partly random model and still produce outputs.
        Mirror the official key remapping and fail closed on every missing,
        unexpected, non-tensor, or shape-incompatible entry.
        """
        checkpoint_path = self.config.checkpoint_path.expanduser().resolve()
        try:
            payload = self._torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise Sam31DependencyError(
                f"failed to inspect SAM 3.1 checkpoint: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
            payload = payload["model"]
        if not isinstance(payload, Mapping) or not payload:
            raise Sam31DependencyError("SAM 3.1 checkpoint state must be a non-empty mapping")
        needs_remap = any(
            isinstance(key, str)
            and (key.startswith("sam3_model.") or key.startswith("sam2_predictor."))
            for key in payload
        )
        checkpoint_state: dict[str, Any] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not key:
                raise Sam31DependencyError("SAM 3.1 checkpoint contains a non-string key")
            remapped = key
            if needs_remap and key.startswith("sam3_model."):
                remapped = "detector." + key[len("sam3_model.") :]
            elif needs_remap and key.startswith("sam2_predictor."):
                remapped = "tracker." + key[len("sam2_predictor.") :]
            if remapped in checkpoint_state:
                raise Sam31DependencyError(
                    f"SAM 3.1 checkpoint remapping collides at {remapped!r}"
                )
            checkpoint_state[remapped] = value

        model_state = self._predictor.model.state_dict()
        checkpoint_keys = set(checkpoint_state)
        model_keys = set(model_state)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        non_tensors = sorted(
            key
            for key, value in checkpoint_state.items()
            if not self._torch.is_tensor(value)
        )
        shape_mismatches = sorted(
            key
            for key in model_keys & checkpoint_keys
            if self._torch.is_tensor(checkpoint_state[key])
            and tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
        )
        if missing or unexpected or non_tensors or shape_mismatches:
            raise Sam31DependencyError(
                "SAM 3.1 checkpoint does not exactly match the official multiplex model; "
                f"missing={missing[:10]} ({len(missing)}), "
                f"unexpected={unexpected[:10]} ({len(unexpected)}), "
                f"non_tensors={non_tensors[:10]} ({len(non_tensors)}), "
                f"shape_mismatches={shape_mismatches[:10]} ({len(shape_mismatches)})"
            )
        key_shapes = [
            [key, list(checkpoint_state[key].shape)] for key in sorted(checkpoint_state)
        ]
        encoded = json.dumps(key_shapes, separators=(",", ":")).encode("utf-8")
        return {
            "format": "sam31_exact_checkpoint_coverage_v1",
            "official_key_remap_applied": needs_remap,
            "state_entry_count": len(key_shapes),
            "key_shape_sha256": hashlib.sha256(encoded).hexdigest(),
            "missing_keys": 0,
            "unexpected_keys": 0,
            "non_tensor_entries": 0,
            "shape_mismatches": 0,
            "exact_coverage": True,
        }

    def recheck_installation(self) -> dict[str, Any]:
        """Rehash sources/assets and repeat exact checkpoint compatibility."""
        report = inspect_official_installation(self.config)
        report["checkpoint_contract"] = self._checkpoint_contract()
        return report

    def _synchronize(self) -> None:
        self._torch.cuda.synchronize(self._torch.cuda.current_device())

    def start_session(self, resource_path: Path) -> None:
        if self._session_id is not None:
            raise Sam31ContractError("SAM 3.1 session is already open")
        model = self._predictor.model
        signature = inspect.signature(model.init_state)
        init_candidates = {
            "resource_path": str(resource_path),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
            "async_loading_frames": False,
        }
        init_kwargs = {
            name: value
            for name, value in init_candidates.items()
            if name in signature.parameters
        }
        if "resource_path" not in init_kwargs:
            raise Sam31DependencyError("official init_state lacks resource_path")
        try:
            state = model.init_state(**init_kwargs)
        except Exception as exc:
            raise Sam31DependencyError(
                f"official start_session failed: {type(exc).__name__}: {exc}"
            ) from exc
        session_id = str(uuid.uuid4())
        now = time.time()
        self._predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }
        self._session_id = session_id
        # The May-2026 official base wrapper still forwards an unsupported
        # offload_state_to_cpu kwarg to the multiplex init_state on some commits.
        # Signature filtering mirrors the upstream proposed compatibility fix.
        self.start_session_dispatch = "signature_filtered_init_state_v1"

    def add_support_prompt(self, prompt: SupportPointPrompt) -> None:
        if self._session_id is None:
            raise Sam31ContractError("start_session must precede support replay")
        request = {
            "type": "add_prompt",
            "session_id": self._session_id,
            "frame_index": int(prompt.frame_index),
            "points": prompt.points_xy,
            "point_labels": prompt.labels,
            "clear_old_points": True,
            "obj_id": int(prompt.object_id),
            "rel_coordinates": True,
            "output_prob_thresh": float(self.config.output_probability_threshold),
        }
        try:
            response = self._predictor.handle_request(request)
        except Exception as exc:
            raise Sam31DependencyError(
                "official point-prompt replay failed on "
                f"frame={prompt.frame_index} role={prompt.role_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(response, Mapping) or response.get("frame_index") != int(
            prompt.frame_index
        ):
            raise Sam31ContractError("official support-prompt response is malformed")

    def propagate_one(self, frame_index: int) -> tuple[Mapping[str, Any], float]:
        if self._session_id is None:
            raise Sam31ContractError("start_session must precede propagation")
        request = {
            "type": "propagate_in_video",
            "session_id": self._session_id,
            "propagation_direction": "forward",
            "start_frame_index": int(frame_index),
            # The official multiplex implementation treats this argument as
            # the number of frames *after* start_frame_index and constructs an
            # inclusive range.  Zero therefore means exactly the requested
            # frame.  We verify the observed response count below so a source
            # revision with different semantics fails closed.
            "max_frame_num_to_track": 0,
            "output_prob_thresh": float(self.config.output_probability_threshold),
        }
        self._synchronize()
        started = time.perf_counter()
        try:
            responses = list(self._predictor.handle_stream_request(request))
        except Exception as exc:
            raise Sam31DependencyError(
                f"official forward propagation failed at frame {frame_index}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if (
            len(responses) != 1
            or not isinstance(responses[0], Mapping)
            or int(responses[0].get("frame_index", -1)) != int(frame_index)
            or not isinstance(responses[0].get("outputs"), Mapping)
        ):
            observed = [
                response.get("frame_index")
                for response in responses
                if isinstance(response, Mapping)
            ]
            raise Sam31ContractError(
                f"expected exactly one output for frame {frame_index}, observed {observed}"
            )
        return responses[0]["outputs"], float(latency_ms)

    def close(self) -> None:
        if self._session_id is None:
            return
        session_id = self._session_id
        try:
            self._predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": True,
                }
            )
        except Exception as exc:
            raise Sam31DependencyError(
                f"official close_session failed for {session_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if session_id in self._predictor._all_inference_states:
            raise Sam31ContractError(
                f"official close_session retained inference state {session_id}"
            )
        self._session_id = None


def write_lossless_frame_directory(
    directory: str | Path,
    frames: np.ndarray,
) -> tuple[Path, str]:
    """Write exact PNG frames in the numeric order required by official SAM 3.1."""
    value = _require_rgb(frames, name="combined_rgb")
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        from PIL import Image
    except Exception as exc:
        raise Sam31DependencyError(f"Pillow is required to stage PNG frames: {exc}") from exc
    width = max(6, len(str(len(value) - 1)))
    for frame_index, frame in enumerate(value):
        output = destination / f"{frame_index:0{width}d}.png"
        Image.fromarray(frame, mode="RGB").save(output, format="PNG", optimize=False)
    return destination, _typed_sha256(value)


def run_official_sam31_episode(
    *,
    config: OfficialSam31Config,
    support: FrozenSupport,
    episode_rgb: np.ndarray,
    frame_directory: str | Path,
    positive_points: int = 4,
    negative_points: int = 4,
    runtime: OfficialSam31Runtime | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one frozen RGB-only episode with fixed support and no episode labels."""
    config = config.validated()
    if support.role_names != config.role_names:
        raise Sam31ContractError("support/config role order mismatch")
    episode = _require_rgb(episode_rgb, name="episode.rgb")
    if support.rgb.shape[1:] != episode.shape[1:]:
        raise Sam31ContractError(
            f"support and episode resolutions differ: {support.rgb.shape[1:]} vs {episode.shape[1:]}"
        )
    prompts, prompt_trace_sha256 = build_support_point_prompts(
        support,
        positive_points=positive_points,
        negative_points=negative_points,
    )
    combined = np.concatenate((support.rgb, episode), axis=0)
    frame_dir, combined_rgb_sha256 = write_lossless_frame_directory(
        frame_directory, combined
    )

    runtime_instance = runtime if runtime is not None else OfficialSam31Runtime(config)
    if len(runtime_instance.config.role_names) != len(config.role_names):
        raise Sam31ContractError(
            "shared SAM 3.1 runtime object capacity does not match this task"
        )
    support_started = time.perf_counter()
    runtime_instance.start_session(frame_dir)
    try:
        for prompt in prompts:
            runtime_instance.add_support_prompt(prompt)
        runtime_instance._synchronize()
        support_runtime_ms = (time.perf_counter() - support_started) * 1000.0

        time_steps, height, width = episode.shape[:3]
        role_count = len(config.role_names)
        role_masks = np.zeros((time_steps, role_count, height, width), dtype=bool)
        confidence = np.zeros((time_steps, role_count), dtype=np.float32)
        valid = np.zeros((time_steps, role_count), dtype=bool)
        latency_ms = np.zeros(time_steps, dtype=np.float64)
        overlap_pixels = np.zeros(time_steps, dtype=np.int64)
        for episode_index in range(time_steps):
            absolute_index = support.rgb.shape[0] + episode_index
            runtime_instance._synchronize()
            frame_started = time.perf_counter()
            outputs, _model_latency_ms = runtime_instance.propagate_one(
                absolute_index
            )
            (
                role_masks[episode_index],
                confidence[episode_index],
                valid[episode_index],
                overlap_pixels[episode_index],
            ) = decode_official_frame_output(
                outputs,
                role_count=role_count,
                height=height,
                width=width,
            )
            runtime_instance._synchronize()
            latency_ms[episode_index] = (
                time.perf_counter() - frame_started
            ) * 1000.0
    finally:
        runtime_instance.close()

    arrays = {
        "predicted_masks": np.ascontiguousarray(role_masks, dtype=np.bool_),
        "reported_confidence": np.ascontiguousarray(confidence, dtype=np.float32),
        "reported_lost": np.ascontiguousarray(~valid, dtype=np.bool_),
        "runtime_ms": np.ascontiguousarray(latency_ms, dtype=np.float64),
    }
    metadata = {
        "format": BACKEND_FORMAT,
        "backend": "sam3.1_multiplex",
        "roles": list(config.role_names),
        "support_frames": EXPECTED_SUPPORT_FRAMES,
        "episode_frames": int(time_steps),
        "frame_height": int(height),
        "frame_width": int(width),
        "prompt_protocol": PROMPT_PROTOCOL,
        "prompt_sampler": PROMPT_SAMPLER,
        "positive_points_per_role_frame_requested": int(positive_points),
        "negative_points_per_role_frame_requested": int(negative_points),
        "prompt_records": len(prompts),
        "prompt_trace_sha256": prompt_trace_sha256,
        "support_rgb_sha256": support.rgb_sha256,
        "support_masks_sha256": support.masks_sha256,
        "combined_rgb_sha256": combined_rgb_sha256,
        "support_runtime_ms": float(support_runtime_ms),
        "causal_protocol": CAUSAL_PROTOCOL,
        "propagation_direction": "forward",
        "official_max_frame_num_to_track_argument": 0,
        "effective_frames_per_propagation_call": 1,
        "episode_first_frame_prompt_used": False,
        "episode_ground_truth_read": False,
        "official_api_mask_prompt": False,
        "prompt_api_not_equivalent": True,
        "comparison_class": "offline_fixed_video_backend_diagnostic",
        "session_rgb_staging": "support_plus_full_episode_png_directory_v1",
        "future_frame_outputs_requested": False,
        "future_frame_model_compute_not_proven_by_adapter": True,
        "strict_source_pixel_access_gate": False,
        "source_pixel_access_limitation": (
            "official init_state preprocesses the full staged sequence; all model "
            "propagation and outputs are nevertheless requested in strict forward "
            "one-effective-frame order"
        ),
        "overlap_pixels_before_resolution_total": int(overlap_pixels.sum()),
        "start_session_dispatch": runtime_instance.start_session_dispatch,
        "installation": runtime_instance.installation,
    }
    return arrays, metadata
