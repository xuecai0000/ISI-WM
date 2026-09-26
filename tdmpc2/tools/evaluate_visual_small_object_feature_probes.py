"""Leakage-safe probes for Visual-Small Cutie object features.

This evaluator consumes two *development-only* rollout manifests (train and
validation) and their corresponding Cutie feature manifests.  It deliberately
does not import a task, simulator, point decoder, or perception backend.  The
only learning code is deterministic, train-fitted PCA/scaling followed by
fixed-regularization ridge regression implemented with NumPy.

The validation split is transformed and evaluated once after every fitted
quantity has been learned from train.  Test and support inputs are rejected.

Expected command::

    python tdmpc2/tools/evaluate_visual_small_object_feature_probes.py \
        --train-rollout train_rollout.json \
        --train-features train_cutie_features.json \
        --validation-rollout validation_rollout.json \
        --validation-features validation_cutie_features.json \
        --output object_feature_probe_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    from .collect_visual_small_object_rollouts import (
        load_and_validate_rollout_manifest as _collector_load_rollout,
    )
    from .extract_visual_small_cutie_object_features import (
        validate_feature_arrays as _extractor_validate_feature_arrays,
    )
except ImportError:  # Direct ``python tdmpc2/tools/...py`` execution.
    _TOOLS_DIR = Path(__file__).resolve().parent
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    from collect_visual_small_object_rollouts import (
        load_and_validate_rollout_manifest as _collector_load_rollout,
    )
    from extract_visual_small_cutie_object_features import (
        validate_feature_arrays as _extractor_validate_feature_arrays,
    )


REPORT_FORMAT = "visual_small_object_feature_probe_report_v1"
ROLLOUT_FORMAT = "visual_small_object_rollout_v1"
FEATURE_FORMAT = "visual_small_cutie_object_features_v1"
TASK = "reacher-visual-small"
ROLES = ("whole_arm", "goal")
ALLOWED_SPLITS = ("train", "validation")
FORBIDDEN_SPLITS = ("test", "support")

FRAME_KEYS = {"t", "image", "image_sha256", "source_frame_index"}
TRANSITION_ARRAY_KEYS = {"actions", "rewards", "terminated", "truncated"}
FEATURE_ARRAY_KEYS = {
    "features",
    "masks",
    "centroid_xy",
    "confidence",
    "lost",
    "valid",
    "mask_score",
    "runtime_ms",
}
FEATURE_TOP_KEYS = {
    "format",
    "rollout_manifest_sha256",
    "perception_provenance",
    "roles",
    "object_schema",
    "collection",
    "feature_contract",
    "provenance",
    "runtime",
    "sequences",
}
FEATURE_COLLECTION_KEYS = {
    "task",
    "split",
    "num_sequences",
    "num_frames",
    "native_input_size",
    "tracker_size",
    "no_test",
    "no_support_trajectories",
    "causal",
    "episode_memory_reset",
    "permanent_support_loaded_once",
    "transitions_passed_to_cutie",
    "rollout_labels_passed_to_cutie",
}
FEATURE_SEQUENCE_KEYS = {
    "sequence_id",
    "split",
    "episode",
    "source",
    "env_seed",
    "background_seed",
    "action_seed",
    "source_selection_reset_attempt",
    "ended_by_environment",
    "num_transitions",
    "frames",
    "rollout_transitions_asset",
    "npz_asset",
    "coverage",
    "runtime_ms",
}
PERCEPTION_PROVENANCE_KEYS = {
    "algorithm",
    "object_schema",
    "roles",
    "native_input_size",
    "tracker_size",
    "foreground_queries",
    "feature_dim",
    "query_order",
    "model_size",
    "amp",
    "prompt_radius",
    "checkpoint_file_sha256",
    "support_annotations_file_sha256",
    "support_manifest_sha256",
    "combined_manifest_sha256",
    "config_tree_sha256",
    "cutie_code_tree_sha256",
    "adapter_file_sha256",
    "extractor_file_sha256",
}
PERCEPTION_SHA_KEYS = {
    "checkpoint_file_sha256",
    "support_annotations_file_sha256",
    "support_manifest_sha256",
    "combined_manifest_sha256",
    "config_tree_sha256",
    "cutie_code_tree_sha256",
    "adapter_file_sha256",
    "extractor_file_sha256",
}

IMAGE_SIZE = 64
RGB_POOL_SIZE = 8
MASK_POOL_SIZE = 8
MASK_DILATION_RADIUS = 3
QUERY_SLOTS = 8
QUERY_DIM = 256
QUERY_FEATURE_DIM = QUERY_SLOTS * QUERY_DIM
QUERY_POOL_DIM = 2 * QUERY_DIM
SPATIAL_DIM_PER_ROLE = 64 + 2 + 1 + 4 + 3
STATUS_DIM_PER_ROLE = 4  # confidence, lost, valid, mask_score

PCA_COMPONENTS = 64
PCA_OVERSAMPLE = 8
PCA_POWER_ITERATIONS = 1
RIDGE_ALPHA = 1e-2
RANDOM_SEED = 20260825
BOOTSTRAP_SAMPLES = 5000

MIN_TRAIN_POSITIVES = 200
MIN_VALIDATION_POSITIVES = 80
MIN_VALIDATION_POSITIVE_SOURCES = 8
MIN_VALIDATION_SOURCES = 10
MIN_VALIDATION_EPISODES = 40
MIN_CURRENT_VALID_COVERAGE = 0.90
MIN_DYNAMICS_VALID_COVERAGE = 0.85
MAX_INVALID_BURST = 5

INFORMATION_MIN_RELATIVE_IMPROVEMENT = 0.10
ACTION_MIN_RELATIVE_IMPROVEMENT = 0.05
RGB_NONINFERIORITY_MARGIN = 0.05
NEGATIVE_CONTROL_MAX_INCREMENT = 0.02

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_FEATURE_KEY_TOKENS = {
    "point",
    "points",
    "decoder",
    "physics",
    "simulator",
    "groundtruth",
    "ground_truth",
    "manualpoint",
    "manualpoints",
    "manual_point",
    "manual_points",
}


class ContractError(ValueError):
    """Raised when an input violates the frozen leakage-safe contract."""


def _duplicate_safe_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_duplicate_safe_object
        )
    except UnicodeDecodeError as exc:
        raise ContractError(f"Manifest is not UTF-8: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Manifest root must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{location} must be a lowercase SHA-256.")
    return value


def _normal_key_tokens(key: str) -> set[str]:
    lowered = str(key).lower()
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    tokens.add(lowered)
    tokens.add(lowered.replace("_", ""))
    return tokens


def _reject_privileged_feature_keys(value: Any, location: str = "features") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = _normal_key_tokens(str(key))
            if tokens & _FORBIDDEN_FEATURE_KEY_TOKENS:
                raise ContractError(
                    f"{location}.{key} is forbidden: probes cannot consume points, "
                    "decoders, manual labels, or physics/simulator data."
                )
            _reject_privileged_feature_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_privileged_feature_keys(child, f"{location}[{index}]")


def _manifest_value(data: Mapping[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    collection = data.get("collection")
    if isinstance(collection, Mapping) and key in collection:
        return collection[key]
    return None


def _manifest_split(data: Mapping[str, Any], location: str) -> str:
    split = _manifest_value(data, "split")
    if split not in ALLOWED_SPLITS:
        raise ContractError(
            f"{location}.split must be one of {ALLOWED_SPLITS}, got {split!r}; "
            "test/support are never accepted."
        )
    return str(split)


def _manifest_task(data: Mapping[str, Any], location: str) -> str:
    task = _manifest_value(data, "task")
    if task != TASK:
        raise ContractError(f"{location}.task must be {TASK!r}, got {task!r}.")
    return str(task)


def _safe_relative_path(root: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} must be a non-empty relative path.")
    relative = Path(value)
    if relative.is_absolute():
        raise ContractError(f"{location} must be relative to its manifest.")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{location} escapes its manifest directory.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _asset_path(
    manifest_path: Path,
    sequence: Mapping[str, Any],
    candidates: Sequence[str],
    location: str,
) -> tuple[Path, str, str]:
    present = [key for key in candidates if key in sequence]
    if len(present) != 1:
        raise ContractError(
            f"{location} must contain exactly one asset key from {tuple(candidates)!r}; "
            f"found {present!r}."
        )
    key = present[0]
    entry = sequence[key]
    if isinstance(entry, Mapping):
        path_value = entry.get("path")
        sha = entry.get("file_sha256")
    else:
        path_value = entry
        sha = sequence.get(f"{key}_sha256")
    sha = _require_sha256(sha, f"{location}.{key}.file_sha256")
    path = _safe_relative_path(
        manifest_path.parent, path_value, f"{location}.{key}.path"
    )
    actual = _file_sha256(path)
    if actual != sha:
        raise ContractError(
            f"{location}.{key} SHA mismatch: expected {sha}, actual {actual}."
        )
    return path, sha, key


def _seed_value(
    sequence: Mapping[str, Any], aliases: Sequence[str], location: str
) -> int:
    found = [(key, sequence[key]) for key in aliases if key in sequence]
    if len(found) != 1:
        raise ContractError(
            f"{location} must contain exactly one seed key from {tuple(aliases)!r}."
        )
    key, value = found[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{location}.{key} must be an integer seed.")
    return int(value)


def _sequence_id(sequence: Mapping[str, Any], location: str) -> str:
    value = sequence.get("sequence_id")
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location}.sequence_id must be a non-empty string.")
    return value


@dataclass(frozen=True)
class RolloutManifest:
    path: Path
    sha256: str
    split: str
    data: dict[str, Any]
    sequences: tuple[dict[str, Any], ...]
    sequence_ids: frozenset[str]
    sources: frozenset[str]
    env_seeds: frozenset[int]
    background_seeds: frozenset[int]
    action_seeds: frozenset[int]
    image_sha256s: frozenset[str]


@dataclass(frozen=True)
class FeatureManifest:
    path: Path
    sha256: str
    split: str
    data: dict[str, Any]
    sequences: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    provenance_sha256: str


def _load_rollout_manifest(path: Path, expected_split: str) -> RolloutManifest:
    # The collector owns the exact rollout schema, seed-domain declarations,
    # allowlists, decoded-image hashes, transition dtypes/alignment, and every
    # referenced asset hash.  Reusing its dependency-light public validator
    # prevents this evaluator from accepting a subtly different contract.
    try:
        validated = _collector_load_rollout(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError(f"Collector rollout validation failed for {path}: {exc}") from exc
    data = validated.payload
    sha = validated.file_sha256
    resolved_path = validated.path
    collection = data["collection"]
    split = collection["split"]
    if split != expected_split:
        raise ContractError(f"Expected {expected_split!r} rollout, got {split!r}.")
    sequences = tuple(data["sequences"])
    sequence_ids = frozenset(sequence["sequence_id"] for sequence in sequences)
    sources = frozenset(sequence["source"] for sequence in sequences)
    env_seeds = frozenset(int(sequence["env_seed"]) for sequence in sequences)
    background_seeds = frozenset(
        int(sequence["background_seed"]) for sequence in sequences
    )
    action_seeds = frozenset(int(sequence["action_seed"]) for sequence in sequences)
    image_hashes = frozenset(
        frame["image_sha256"]
        for sequence in sequences
        for frame in sequence["frames"]
    )
    return RolloutManifest(
        path=resolved_path,
        sha256=sha,
        split=split,
        data=data,
        sequences=sequences,
        sequence_ids=sequence_ids,
        sources=sources,
        env_seeds=env_seeds,
        background_seeds=background_seeds,
        action_seeds=action_seeds,
        image_sha256s=image_hashes,
    )


def _perception_provenance(data: Mapping[str, Any], location: str) -> dict[str, Any]:
    value = data.get("perception_provenance")
    if value is None and isinstance(data.get("metadata"), Mapping):
        value = data["metadata"].get("perception_provenance")
    if not isinstance(value, dict) or not value:
        raise ContractError(f"{location}.perception_provenance must be a non-empty object.")
    if set(value) != PERCEPTION_PROVENANCE_KEYS:
        raise ContractError(
            f"{location}.perception_provenance differs from the frozen schema: "
            f"missing={sorted(PERCEPTION_PROVENANCE_KEYS - set(value))}, "
            f"extra={sorted(set(value) - PERCEPTION_PROVENANCE_KEYS)}."
        )
    expected = {
        "algorithm": "visual_small_cutie_object_features_v1",
        "object_schema": "whole_arm_goal_v1",
        "roles": list(ROLES),
        "native_input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "tracker_size": [448, 448],
        "foreground_queries": QUERY_SLOTS,
        "feature_dim": QUERY_FEATURE_DIM,
        "query_order": "official_query_post_process_cache_first_8_flattened",
        "model_size": "small",
        "amp": True,
        "prompt_radius": 2.0,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ContractError(
                f"{location}.perception_provenance.{key} must be "
                f"{expected_value!r}, got {value.get(key)!r}."
            )
    for key in PERCEPTION_SHA_KEYS:
        _require_sha256(value.get(key), f"{location}.perception_provenance.{key}")
    return value


def _load_feature_manifest(
    path: Path,
    expected_split: str,
    rollout: RolloutManifest,
) -> FeatureManifest:
    data, sha = _read_json(path)
    _reject_privileged_feature_keys(data, str(path))
    if set(data) != FEATURE_TOP_KEYS:
        raise ContractError(
            f"{path} feature top-level keys differ from the frozen contract: "
            f"missing={sorted(FEATURE_TOP_KEYS - set(data))}, "
            f"extra={sorted(set(data) - FEATURE_TOP_KEYS)}."
        )
    if data.get("format") != FEATURE_FORMAT:
        raise ContractError(
            f"{path}.format must be {FEATURE_FORMAT!r}, got {data.get('format')!r}."
        )
    split = _manifest_split(data, str(path))
    _manifest_task(data, str(path))
    if split != expected_split or split != rollout.split:
        raise ContractError(f"Feature split {split!r} does not match rollout split.")
    if tuple(data.get("roles", ())) != ROLES:
        raise ContractError(f"{path}.roles must be ordered exactly as {ROLES!r}.")
    if data.get("object_schema") != "whole_arm_goal_v1":
        raise ContractError(f"{path}.object_schema must be 'whole_arm_goal_v1'.")
    collection = data.get("collection")
    if not isinstance(collection, dict) or set(collection) != FEATURE_COLLECTION_KEYS:
        raise ContractError(f"{path}.collection differs from the frozen feature contract.")
    if (
        collection.get("native_input_size") != [64, 64]
        or collection.get("tracker_size") != [448, 448]
        or collection.get("no_test") is not True
        or collection.get("no_support_trajectories") is not True
        or collection.get("causal") is not True
        or collection.get("episode_memory_reset") is not True
        or collection.get("permanent_support_loaded_once") is not True
        or collection.get("transitions_passed_to_cutie") is not False
        or collection.get("rollout_labels_passed_to_cutie") is not False
    ):
        raise ContractError(f"{path}.collection violates causal/no-held-out feature policy.")
    if collection.get("num_sequences") != len(rollout.sequences) or collection.get(
        "num_frames"
    ) != sum(len(sequence["frames"]) for sequence in rollout.sequences):
        raise ContractError(f"{path}.collection sequence/frame counts mismatch rollout.")
    feature_contract = data.get("feature_contract")
    expected_feature_contract_keys = {
        "feature_dim",
        "foreground_queries",
        "query_dim",
        "query_order",
        "valid_definition",
        "missing_policy",
        "mask_values",
        "centroid_convention",
    }
    if not isinstance(feature_contract, dict) or set(
        feature_contract
    ) != expected_feature_contract_keys:
        raise ContractError(f"{path}.feature_contract has a non-frozen schema.")
    if (
        feature_contract.get("feature_dim") != QUERY_FEATURE_DIM
        or feature_contract.get("foreground_queries") != QUERY_SLOTS
        or feature_contract.get("query_dim") != QUERY_DIM
        or feature_contract.get("query_order")
        != "official_query_post_process_cache_first_8_flattened"
        or feature_contract.get("valid_definition")
        != "(~official_lost) & mask_nonempty & feature_finite"
        or feature_contract.get("missing_policy")
        != "No future fill and no success-only filtering; consumers must mask valid=false."
        or feature_contract.get("mask_values") != [0, 1]
        or feature_contract.get("centroid_convention")
        != "[x,y] in native 64x64 RGB; NaN is allowed when invalid"
    ):
        raise ContractError(f"{path}.feature_contract values are incompatible.")
    bound_rollout_sha = _require_sha256(
        data.get("rollout_manifest_sha256"),
        f"{path}.rollout_manifest_sha256",
    )
    if bound_rollout_sha != rollout.sha256:
        raise ContractError(
            f"{path} is bound to rollout SHA {bound_rollout_sha}, expected "
            f"{rollout.sha256}."
        )
    provenance = _perception_provenance(data, str(path))

    raw_sequences = data.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ContractError(f"{path}.sequences must be a non-empty list.")
    sequences: dict[str, dict[str, Any]] = {}
    rollout_by_id = {
        sequence["sequence_id"]: sequence for sequence in rollout.sequences
    }
    for index, raw_sequence in enumerate(raw_sequences):
        location = f"{path}.sequences[{index}]"
        if not isinstance(raw_sequence, dict):
            raise ContractError(f"{location} must be an object.")
        if set(raw_sequence) != FEATURE_SEQUENCE_KEYS:
            raise ContractError(
                f"{location} keys differ from the frozen feature sequence contract."
            )
        sequence_id = _sequence_id(raw_sequence, location)
        if sequence_id in sequences:
            raise ContractError(f"Duplicate feature sequence {sequence_id!r}.")
        if raw_sequence.get("split", split) != split:
            raise ContractError(f"{location}.split does not match manifest split.")
        if sequence_id not in rollout_by_id:
            raise ContractError(f"{location} is absent from its bound rollout.")
        rollout_sequence = rollout_by_id[sequence_id]
        for field in (
            "episode",
            "source",
            "env_seed",
            "background_seed",
            "action_seed",
            "num_transitions",
        ):
            if raw_sequence.get(field) != rollout_sequence.get(field):
                raise ContractError(
                    f"{location}.{field} does not match the bound rollout sequence."
                )
        feature_frames = raw_sequence.get("frames")
        rollout_frames = rollout_sequence["frames"]
        if not isinstance(feature_frames, list) or len(feature_frames) != len(
            rollout_frames
        ):
            raise ContractError(f"{location}.frames does not match bound N+1 frames.")
        for frame_index, (feature_frame, rollout_frame) in enumerate(
            zip(feature_frames, rollout_frames)
        ):
            if not isinstance(feature_frame, dict) or set(feature_frame) != {
                "t",
                "image_sha256",
                "source_frame_index",
            }:
                raise ContractError(
                    f"{location}.frames[{frame_index}] has a non-frozen schema."
                )
            expected = {
                "t": rollout_frame["t"],
                "image_sha256": rollout_frame["image_sha256"],
                "source_frame_index": rollout_frame["source_frame_index"],
            }
            if feature_frame != expected:
                raise ContractError(
                    f"{location}.frames[{frame_index}] does not match its rollout frame."
                )
        if raw_sequence.get("rollout_transitions_asset") != rollout_sequence.get(
            "transitions_asset"
        ):
            raise ContractError(
                f"{location}.rollout_transitions_asset is not the bound rollout asset."
            )
        npz_asset = raw_sequence.get("npz_asset")
        if not isinstance(npz_asset, dict) or set(npz_asset) != {
            "path",
            "file_sha256",
            "arrays",
        }:
            raise ContractError(
                f"{location}.npz_asset must contain path/file_sha256/arrays exactly."
            )
        _asset_path(
            path,
            raw_sequence,
            ("npz_asset",),
            location,
        )
        sequences[sequence_id] = raw_sequence
    if frozenset(sequences) != rollout.sequence_ids:
        missing = sorted(rollout.sequence_ids - frozenset(sequences))
        extra = sorted(frozenset(sequences) - rollout.sequence_ids)
        raise ContractError(
            f"Feature/rollout sequence mismatch: missing={missing}, extra={extra}."
        )
    return FeatureManifest(
        path=path.expanduser().resolve(),
        sha256=sha,
        split=split,
        data=data,
        sequences=sequences,
        provenance=provenance,
        provenance_sha256=_canonical_sha256(provenance),
    )


def _validate_cross_split(
    train_rollout: RolloutManifest,
    validation_rollout: RolloutManifest,
    train_features: FeatureManifest,
    validation_features: FeatureManifest,
) -> None:
    if train_rollout.sha256 == validation_rollout.sha256:
        raise ContractError("Train and validation rollout manifests are identical.")
    overlaps = {
        "sequence_id": train_rollout.sequence_ids & validation_rollout.sequence_ids,
        "source": train_rollout.sources & validation_rollout.sources,
        "env_seed": train_rollout.env_seeds & validation_rollout.env_seeds,
        "background_seed": (
            train_rollout.background_seeds & validation_rollout.background_seeds
        ),
        "action_seed": train_rollout.action_seeds & validation_rollout.action_seeds,
        "decoded_image_sha256": (
            train_rollout.image_sha256s & validation_rollout.image_sha256s
        ),
    }
    nonempty = {key: sorted(value) for key, value in overlaps.items() if value}
    if nonempty:
        raise ContractError(f"Train/validation isolation failure: {nonempty!r}.")
    if train_features.provenance != validation_features.provenance:
        raise ContractError(
            "Train and validation must use byte-equivalent canonical perception "
            "provenance (same frozen Cutie implementation/config/checkpoint)."
        )


def _load_npz_exact(path: Path, expected_keys: set[str], location: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != expected_keys:
                raise ContractError(
                    f"{location} must contain exactly {sorted(expected_keys)!r}, "
                    f"got {sorted(keys)!r}."
                )
            return {key: np.asarray(archive[key]) for key in sorted(keys)}
    except ValueError as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Could not read safe NumPy asset {path}: {exc}") from exc


def _decode_rgb(path: Path, expected_sha256: str, location: str) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ContractError(
            f"{location} must decode to native 64x64 RGB, got {value.shape}."
        )
    actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if actual != expected_sha256:
        raise ContractError(
            f"{location} decoded-image SHA mismatch: expected {expected_sha256}, "
            f"actual {actual}."
        )
    return value


def _query_pool(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Accumulate in float64 so a permutation of the eight query slots rounds to
    # the same exported float32 pooled representation in practice.
    safe = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )
    slots = safe.reshape(*safe.shape[:-1], QUERY_SLOTS, QUERY_DIM)
    pooled = np.concatenate(
        [slots.mean(axis=-2), slots.std(axis=-2, ddof=0)], axis=-1
    ).astype(np.float32)
    pooled[~valid] = 0.0
    return pooled


def _mask_spatial_one(mask: np.ndarray) -> np.ndarray:
    mask_float = np.asarray(mask, dtype=np.float32)
    occupancy = mask_float.reshape(
        MASK_POOL_SIZE,
        IMAGE_SIZE // MASK_POOL_SIZE,
        MASK_POOL_SIZE,
        IMAGE_SIZE // MASK_POOL_SIZE,
    ).mean(axis=(1, 3)).reshape(-1)
    yx = np.argwhere(mask_float > 0.5)
    if not len(yx):
        summary = np.zeros(2 + 1 + 4 + 3, dtype=np.float32)
        return np.concatenate([occupancy, summary]).astype(np.float32)
    y = yx[:, 0].astype(np.float64) / (IMAGE_SIZE - 1)
    x = yx[:, 1].astype(np.float64) / (IMAGE_SIZE - 1)
    cx, cy = float(x.mean()), float(y.mean())
    bbox = np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)
    dx, dy = x - cx, y - cy
    moments = np.asarray(
        [(dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()],
        dtype=np.float32,
    )
    summary = np.concatenate(
        [
            np.asarray([cx, cy, mask_float.mean()], dtype=np.float32),
            bbox,
            moments,
        ]
    )
    return np.concatenate([occupancy, summary]).astype(np.float32)


def _mask_spatial(masks: np.ndarray) -> np.ndarray:
    frames, roles = masks.shape[:2]
    result = np.empty(
        (frames, roles, SPATIAL_DIM_PER_ROLE), dtype=np.float32
    )
    for frame in range(frames):
        for role in range(roles):
            result[frame, role] = _mask_spatial_one(masks[frame, role])
    return result


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            [
                padded[dy : dy + IMAGE_SIZE, dx : dx + IMAGE_SIZE]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return result


def _pool_rgb_frame(frame: np.ndarray) -> np.ndarray:
    return (
        frame.astype(np.float32)
        .reshape(
            RGB_POOL_SIZE,
            IMAGE_SIZE // RGB_POOL_SIZE,
            RGB_POOL_SIZE,
            IMAGE_SIZE // RGB_POOL_SIZE,
            3,
        )
        .mean(axis=(1, 3))
        .reshape(-1)
        .astype(np.float32)
        / 255.0
    )


def _rgb_stack_features(frames: np.ndarray) -> np.ndarray:
    transitions = len(frames) - 1
    pooled_frames = np.asarray(
        [_pool_rgb_frame(frame) for frame in frames], dtype=np.float32
    )
    result = []
    for t in range(transitions):
        indices = (max(0, t - 2), max(0, t - 1), t)
        result.append(np.concatenate([pooled_frames[index] for index in indices]))
    return np.asarray(result, dtype=np.float32)


def _temporal_stack_features(values: np.ndarray) -> np.ndarray:
    """Give object modalities the same three-observation context as RGB."""
    array = np.asarray(values, dtype=np.float32)
    transitions = len(array) - 1
    result = []
    for t in range(transitions):
        indices = (max(0, t - 2), max(0, t - 1), t)
        result.append(np.concatenate([array[index] for index in indices], axis=-1))
    return np.asarray(result, dtype=np.float32)


def _temporal_stack_valid(valid_frames: np.ndarray) -> np.ndarray:
    values = np.asarray(valid_frames, dtype=bool)
    transitions = len(values) - 1
    return np.asarray(
        [
            bool(values[max(0, t - 2)] and values[max(0, t - 1)] and values[t])
            for t in range(transitions)
        ],
        dtype=bool,
    )


def _background_frame_statistics(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return non-spatial color statistics outside a dilated object union.

    Replacing foreground pixels with a constant would leave a silhouette whose
    location and shape directly encode the object.  This control discards all
    spatial arrangement and never reports the masked fraction.
    """
    union = np.asarray(masks, dtype=bool).any(axis=0)
    background = np.asarray(frame, dtype=np.float32)[
        ~_dilate_mask(union, MASK_DILATION_RADIUS)
    ] / 255.0
    if not len(background):
        raise ContractError("Dilated object mask leaves no background pixels.")
    quantiles = np.quantile(
        background, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0
    )
    return np.concatenate(
        [background.mean(axis=0), background.std(axis=0), quantiles.reshape(-1)]
    ).astype(np.float32)


def _background_stack_statistics(
    frames: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    transitions = len(frames) - 1
    frame_statistics = np.asarray(
        [
            _background_frame_statistics(frame, frame_masks)
            for frame, frame_masks in zip(frames, masks)
        ],
        dtype=np.float32,
    )
    result = []
    for t in range(transitions):
        indices = (max(0, t - 2), max(0, t - 1), t)
        result.append(
            np.concatenate(
                [
                    frame_statistics[index]
                    for index in indices
                ]
            )
        )
    return np.asarray(result, dtype=np.float32)


def _invalid_runs(valid_frames: np.ndarray) -> list[int]:
    invalid = ~np.asarray(valid_frames, dtype=bool)
    runs: list[int] = []
    current = 0
    for value in invalid:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


@dataclass
class ProbeDataset:
    split: str
    actions: np.ndarray
    rewards: np.ndarray
    query: np.ndarray
    mask: np.ndarray
    object: np.ndarray
    spatial: np.ndarray
    next_spatial: np.ndarray
    rgb: np.ndarray
    background_rgb: np.ndarray
    reward_valid: np.ndarray
    dynamics_valid: np.ndarray
    sources: np.ndarray
    episodes: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    coverage: dict[str, Any]


def _validate_feature_arrays(
    arrays: Mapping[str, np.ndarray], frames: int, location: str
) -> tuple[np.ndarray, ...]:
    try:
        official_metadata = _extractor_validate_feature_arrays(
            arrays, num_frames=frames
        )
    except (ValueError, RuntimeError) as exc:
        raise ContractError(
            f"{location} fails the extractor's official array contract: {exc}"
        ) from exc
    features = arrays["features"]
    masks = arrays["masks"]
    centroid = arrays["centroid_xy"]
    confidence = arrays["confidence"]
    lost = arrays["lost"]
    valid = arrays["valid"]
    mask_score = arrays["mask_score"]
    runtime = arrays["runtime_ms"]
    if features.shape != (frames, len(ROLES), QUERY_FEATURE_DIM):
        raise ContractError(
            f"{location}.features must have shape {(frames, len(ROLES), QUERY_FEATURE_DIM)}, "
            f"got {features.shape}."
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ContractError(f"{location}.features must be floating point.")
    if masks.shape != (frames, len(ROLES), IMAGE_SIZE, IMAGE_SIZE):
        raise ContractError(f"{location}.masks has wrong shape {masks.shape}.")
    if not np.all((masks == 0) | (masks == 1)):
        raise ContractError(f"{location}.masks must be binary.")
    if centroid.shape != (frames, len(ROLES), 2):
        raise ContractError(f"{location}.centroid_xy has wrong shape {centroid.shape}.")
    for name, value in (("confidence", confidence), ("mask_score", mask_score)):
        if value.shape != (frames, len(ROLES)):
            raise ContractError(f"{location}.{name} has wrong shape {value.shape}.")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ContractError(f"{location}.{name} must be finite floating point.")
        if np.any((value < 0.0) | (value > 1.0)):
            raise ContractError(f"{location}.{name} must lie in [0, 1].")
    for name, value in (("lost", lost), ("valid", valid)):
        if value.shape != (frames, len(ROLES)) or value.dtype != np.bool_:
            raise ContractError(f"{location}.{name} must be bool[{frames},2].")
    if runtime.shape not in ((frames,), (frames, len(ROLES))):
        raise ContractError(f"{location}.runtime_ms has wrong shape {runtime.shape}.")
    if not np.issubdtype(runtime.dtype, np.floating) or not np.isfinite(runtime).all():
        raise ContractError(f"{location}.runtime_ms must be finite floating point.")
    if np.any(runtime < 0):
        raise ContractError(f"{location}.runtime_ms must be non-negative.")

    feature_finite = np.isfinite(features).all(axis=-1)
    mask_nonempty = masks.astype(bool).any(axis=(-1, -2))
    expected_valid = (~lost) & mask_nonempty & feature_finite
    if not np.array_equal(valid, expected_valid):
        mismatches = int(np.count_nonzero(valid != expected_valid))
        raise ContractError(
            f"{location}.valid violates official not-lost + nonempty-mask + "
            f"finite-feature policy at {mismatches} role-frames."
        )
    if np.any(valid & ~np.isfinite(centroid).all(axis=-1)):
        raise ContractError(f"{location}.centroid_xy must be finite for valid roles.")
    return (
        features,
        masks.astype(bool),
        confidence,
        lost,
        valid,
        mask_score,
        official_metadata,
    )


def _load_dataset(
    rollout: RolloutManifest,
    feature_manifest: FeatureManifest,
) -> ProbeDataset:
    actions_all: list[np.ndarray] = []
    rewards_all: list[np.ndarray] = []
    query_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    object_all: list[np.ndarray] = []
    spatial_all: list[np.ndarray] = []
    next_spatial_all: list[np.ndarray] = []
    rgb_all: list[np.ndarray] = []
    background_all: list[np.ndarray] = []
    reward_valid_all: list[np.ndarray] = []
    dynamics_valid_all: list[np.ndarray] = []
    sources_all: list[np.ndarray] = []
    episodes_all: list[np.ndarray] = []
    terminated_all: list[np.ndarray] = []
    truncated_all: list[np.ndarray] = []

    role_valid_counts = np.zeros(len(ROLES), dtype=np.int64)
    role_lost_counts = np.zeros(len(ROLES), dtype=np.int64)
    total_frames = 0
    invalid_runs: list[int] = []
    action_dim: int | None = None

    for sequence_index, sequence in enumerate(rollout.sequences):
        sequence_id = _sequence_id(sequence, f"rollout.sequences[{sequence_index}]")
        frames_meta = sequence["frames"]
        frame_count = len(frames_meta)
        transition_count = frame_count - 1
        transition_path, _, _ = _asset_path(
            rollout.path,
            sequence,
            ("transitions_asset",),
            f"rollout.{sequence_id}",
        )
        transitions = _load_npz_exact(
            transition_path,
            TRANSITION_ARRAY_KEYS,
            f"rollout.{sequence_id}.transitions",
        )
        actions = transitions["actions"]
        rewards = transitions["rewards"]
        terminated = transitions["terminated"]
        truncated = transitions["truncated"]
        if actions.ndim != 2 or actions.shape[0] != transition_count:
            raise ContractError(
                f"rollout.{sequence_id}.actions must be [N,A] with N={transition_count}."
            )
        if actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ContractError(f"rollout.{sequence_id}.actions must be finite float32.")
        if action_dim is None:
            action_dim = int(actions.shape[1])
        elif action_dim != int(actions.shape[1]):
            raise ContractError("Action dimension changes between sequences.")
        if rewards.shape != (transition_count,) or rewards.dtype != np.float32:
            raise ContractError(f"rollout.{sequence_id}.rewards must be float32[N].")
        if not np.isfinite(rewards).all():
            raise ContractError(f"rollout.{sequence_id}.rewards must be finite.")
        for name, values in (("terminated", terminated), ("truncated", truncated)):
            if values.shape != (transition_count,) or values.dtype != np.bool_:
                raise ContractError(f"rollout.{sequence_id}.{name} must be bool[N].")

        feature_sequence = feature_manifest.sequences[sequence_id]
        if feature_sequence.get("source", sequence["source"]) != sequence["source"]:
            raise ContractError(f"Feature source mismatch for {sequence_id}.")
        feature_path, _, _ = _asset_path(
            feature_manifest.path,
            feature_sequence,
            ("npz_asset",),
            f"features.{sequence_id}",
        )
        feature_arrays = _load_npz_exact(
            feature_path,
            FEATURE_ARRAY_KEYS,
            f"features.{sequence_id}",
        )
        (
            features,
            masks,
            confidence,
            lost,
            valid,
            mask_score,
            official_metadata,
        ) = _validate_feature_arrays(
            feature_arrays, frame_count, f"features.{sequence_id}"
        )
        declared_arrays = feature_sequence["npz_asset"].get("arrays")
        expected_arrays = {
            name: value
            for name, value in official_metadata.items()
            if name != "coverage"
        }
        if declared_arrays != expected_arrays:
            raise ContractError(
                f"features.{sequence_id}.npz_asset.arrays does not describe its NPZ."
            )
        if feature_sequence.get("coverage") != official_metadata["coverage"]:
            raise ContractError(
                f"features.{sequence_id}.coverage does not match its NPZ."
            )

        images = []
        for frame_index, frame in enumerate(frames_meta):
            image_path = _safe_relative_path(
                rollout.path.parent,
                frame["image"],
                f"rollout.{sequence_id}.frames[{frame_index}].image",
            )
            images.append(
                _decode_rgb(
                    image_path,
                    frame["image_sha256"],
                    f"rollout.{sequence_id}.frames[{frame_index}].image",
                )
            )
        image_array = np.asarray(images, dtype=np.uint8)

        query_roles = _query_pool(features, valid)
        spatial_roles = _mask_spatial(masks)
        status_roles = np.stack(
            [confidence, lost.astype(np.float32), valid.astype(np.float32), mask_score],
            axis=-1,
        ).astype(np.float32)
        mask_roles = np.concatenate([spatial_roles, status_roles], axis=-1)
        object_roles = np.concatenate([query_roles, mask_roles], axis=-1)
        query_flat = query_roles.reshape(frame_count, -1)
        mask_flat = mask_roles.reshape(frame_count, -1)
        object_flat = object_roles.reshape(frame_count, -1)
        spatial_flat = spatial_roles.reshape(frame_count, -1)

        frame_valid = valid.all(axis=-1)
        history_valid = _temporal_stack_valid(frame_valid)
        reward_valid = history_valid
        dynamics_valid = history_valid & frame_valid[1:]

        actions_all.append(actions)
        rewards_all.append(rewards)
        query_all.append(_temporal_stack_features(query_flat))
        mask_all.append(_temporal_stack_features(mask_flat))
        object_all.append(_temporal_stack_features(object_flat))
        spatial_all.append(spatial_flat[:-1])
        next_spatial_all.append(spatial_flat[1:])
        rgb_all.append(_rgb_stack_features(image_array))
        background_all.append(_background_stack_statistics(image_array, masks))
        reward_valid_all.append(reward_valid)
        dynamics_valid_all.append(dynamics_valid)
        sources_all.append(
            np.full(transition_count, sequence["source"], dtype=object)
        )
        episodes_all.append(np.full(transition_count, sequence_id, dtype=object))
        terminated_all.append(terminated)
        truncated_all.append(truncated)

        role_valid_counts += valid.sum(axis=0, dtype=np.int64)
        role_lost_counts += lost.sum(axis=0, dtype=np.int64)
        total_frames += frame_count
        invalid_runs.extend(_invalid_runs(frame_valid))

    concatenate = lambda values: np.concatenate(values, axis=0)
    reward_valid_array = concatenate(reward_valid_all).astype(bool)
    dynamics_valid_array = concatenate(dynamics_valid_all).astype(bool)
    transitions_total = int(len(reward_valid_array))
    coverage = {
        "frames_total": int(total_frames),
        "transitions_total": transitions_total,
        "per_role": {
            role: {
                "valid_frames": int(role_valid_counts[index]),
                "valid_fraction": float(role_valid_counts[index] / total_frames),
                "lost_frames": int(role_lost_counts[index]),
                "lost_fraction": float(role_lost_counts[index] / total_frames),
            }
            for index, role in enumerate(ROLES)
        },
        "reward_history_all_roles_valid": int(reward_valid_array.sum()),
        "reward_history_all_roles_valid_fraction": float(reward_valid_array.mean()),
        "dynamics_history_and_next_all_roles_valid": int(dynamics_valid_array.sum()),
        "dynamics_history_and_next_all_roles_valid_fraction": float(
            dynamics_valid_array.mean()
        ),
        "invalid_bursts": {
            "count": len(invalid_runs),
            "maximum_frames": max(invalid_runs, default=0),
            "mean_frames": float(np.mean(invalid_runs)) if invalid_runs else 0.0,
        },
        "missing_value_policy": (
            "No future filling. Models use the explicit common valid subset; "
            "coverage and invalid bursts are independent hard results."
        ),
    }
    return ProbeDataset(
        split=rollout.split,
        actions=concatenate(actions_all).astype(np.float32),
        rewards=concatenate(rewards_all).astype(np.float32),
        query=concatenate(query_all).astype(np.float32),
        mask=concatenate(mask_all).astype(np.float32),
        object=concatenate(object_all).astype(np.float32),
        spatial=concatenate(spatial_all).astype(np.float32),
        next_spatial=concatenate(next_spatial_all).astype(np.float32),
        rgb=concatenate(rgb_all).astype(np.float32),
        background_rgb=concatenate(background_all).astype(np.float32),
        reward_valid=reward_valid_array,
        dynamics_valid=dynamics_valid_array,
        sources=concatenate(sources_all),
        episodes=concatenate(episodes_all),
        terminated=concatenate(terminated_all).astype(bool),
        truncated=concatenate(truncated_all).astype(bool),
        coverage=coverage,
    )


def _stable_component_signs(components: np.ndarray) -> np.ndarray:
    result = components.copy()
    for index, row in enumerate(result):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            result[index] *= -1.0
    return result


@dataclass
class PCAProjector:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    singular_values: np.ndarray
    method: str

    @classmethod
    def fit(cls, values: np.ndarray, name: str) -> "PCAProjector":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ContractError(f"Cannot fit {name} PCA on shape {x.shape}.")
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale[scale < 1e-8] = 1.0
        z = (x - mean) / scale
        components_count = min(PCA_COMPONENTS, len(z) - 1, z.shape[1])
        if components_count < 1:
            raise ContractError(f"{name} PCA has no estimable components.")
        if min(z.shape) <= 256:
            _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
            components = vt[:components_count]
            singular_values = singular_values[:components_count]
            method = "exact_svd"
        else:
            seed_bytes = hashlib.sha256(
                f"{RANDOM_SEED}:{name}".encode("utf-8")
            ).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
            sketch_dim = min(
                components_count + PCA_OVERSAMPLE, z.shape[0], z.shape[1]
            )
            omega = rng.standard_normal((z.shape[1], sketch_dim))
            q, _ = np.linalg.qr(z @ omega, mode="reduced")
            for _ in range(PCA_POWER_ITERATIONS):
                q, _ = np.linalg.qr(z @ (z.T @ q), mode="reduced")
            _, singular_values, vt = np.linalg.svd(q.T @ z, full_matrices=False)
            components = vt[:components_count]
            singular_values = singular_values[:components_count]
            method = "deterministic_randomized_svd"
        components = _stable_component_signs(components)
        projected = z @ components.T
        output_mean = projected.mean(axis=0)
        output_scale = projected.std(axis=0, ddof=0)
        output_scale[output_scale < 1e-8] = 1.0
        return cls(
            mean=mean,
            scale=scale,
            components=components,
            output_mean=output_mean,
            output_scale=output_scale,
            singular_values=np.asarray(singular_values, dtype=np.float64),
            method=method,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ContractError(
                f"PCA input width {x.shape[1] if x.ndim == 2 else None} does not "
                f"match fitted width {len(self.mean)}."
            )
        projected = ((x - self.mean) / self.scale) @ self.components.T
        return ((projected - self.output_mean) / self.output_scale).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        fit_digest = hashlib.sha256()
        for value in (
            self.mean,
            self.scale,
            self.components,
            self.output_mean,
            self.output_scale,
        ):
            fit_digest.update(np.ascontiguousarray(value).tobytes())
        return {
            "method": self.method,
            "input_dim": int(len(self.mean)),
            "output_dim": int(len(self.components)),
            "singular_values": [float(value) for value in self.singular_values],
            "train_fit_sha256": fit_digest.hexdigest(),
        }


@dataclass
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        if y64.ndim == 1:
            y64 = y64[:, None]
        if x64.ndim != 2 or y64.ndim != 2 or len(x64) != len(y64):
            raise ContractError("Ridge inputs have incompatible shapes.")
        if len(x64) < 2 or not np.isfinite(x64).all() or not np.isfinite(y64).all():
            raise ContractError("Ridge inputs must contain at least two finite rows.")
        x_mean = x64.mean(axis=0)
        x_scale = x64.std(axis=0, ddof=0)
        x_scale[x_scale < 1e-8] = 1.0
        y_mean = y64.mean(axis=0)
        y_scale = y64.std(axis=0, ddof=0)
        y_scale[y_scale < 1e-8] = 1.0
        zx = (x64 - x_mean) / x_scale
        zy = (y64 - y_mean) / y_scale
        gram = zx.T @ zx / len(zx)
        cross = zx.T @ zy / len(zx)
        coefficients = np.linalg.solve(
            gram + RIDGE_ALPHA * np.eye(gram.shape[0], dtype=np.float64), cross
        )
        return cls(x_mean, x_scale, y_mean, y_scale, coefficients)

    def predict(self, x: np.ndarray) -> np.ndarray:
        zx = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        prediction = (zx @ self.coefficients) * self.y_scale + self.y_mean
        return prediction.astype(np.float64)

    def fit_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.x_mean,
            self.x_scale,
            self.y_mean,
            self.y_scale,
            self.coefficients,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def _shuffle_within_episode(
    values: np.ndarray, episodes: np.ndarray, label: str
) -> np.ndarray:
    result = np.asarray(values).copy()
    seed_bytes = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    for episode in sorted(set(episodes.tolist())):
        indices = np.flatnonzero(episodes == episode)
        if len(indices) > 1:
            result[indices] = result[indices[rng.permutation(len(indices))]]
    return result


def _shuffle_episode_blocks(
    values: np.ndarray, episodes: np.ndarray, label: str
) -> np.ndarray:
    """Exchange complete episode targets while preserving within-episode order."""
    episode_values = sorted(set(episodes.tolist()))
    if len(episode_values) < 2:
        raise ContractError("Episode-block shuffle requires at least two episodes.")
    indices = [np.flatnonzero(episodes == episode) for episode in episode_values]
    lengths = {len(value) for value in indices}
    if len(lengths) != 1:
        raise ContractError(
            "Frozen rollout contract must have equal episode lengths for block shuffle."
        )
    seed_bytes = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    shift = int(rng.integers(1, len(indices)))
    result = np.asarray(values).copy()
    for destination, source in zip(indices, indices[shift:] + indices[:shift]):
        result[destination] = np.asarray(values)[source]
    return result


def _reward_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    squared = np.square(prediction - target)
    absolute = np.abs(prediction - target)
    denominator = float(np.square(target - target.mean()).sum())
    r2 = 1.0 - float(squared.sum()) / denominator if denominator > 1e-12 else None
    return (
        {
            "rmse": float(np.sqrt(squared.mean())),
            "mae": float(absolute.mean()),
            "r2": r2,
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
        },
        squared,
    )


def _dynamics_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    train_target_scale: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    raw_squared = np.square(prediction - target)
    normalized_squared = np.square(
        (prediction - target) / np.asarray(train_target_scale, dtype=np.float64)
    )
    row_loss = normalized_squared.mean(axis=1)
    return (
        {
            "rmse": float(np.sqrt(raw_squared.mean())),
            "normalized_rmse": float(np.sqrt(row_loss.mean())),
            "mae": float(np.abs(prediction - target).mean()),
        },
        row_loss,
    )


def _hierarchical_bootstrap_comparison(
    candidate_loss: np.ndarray,
    baseline_loss: np.ndarray,
    sources: np.ndarray,
    episodes: np.ndarray,
    label: str,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_loss, dtype=np.float64)
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    source_values = sorted(set(sources.tolist()))
    episode_by_source = {
        source: sorted(set(episodes[sources == source].tolist()))
        for source in source_values
    }
    episode_stats: dict[tuple[str, str], tuple[float, float, int]] = {}
    for source in source_values:
        for episode in episode_by_source[source]:
            indices = np.flatnonzero((sources == source) & (episodes == episode))
            episode_stats[(source, episode)] = (
                float(candidate[indices].sum()),
                float(baseline[indices].sum()),
                int(len(indices)),
            )
    observed_baseline = float(baseline.mean())
    observed_candidate = float(candidate.mean())
    observed = (
        (observed_baseline - observed_candidate) / observed_baseline
        if observed_baseline > 1e-12
        else 0.0
    )
    seed_bytes = hashlib.sha256(f"{RANDOM_SEED}:{label}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled_sources = rng.choice(source_values, size=len(source_values), replace=True)
        candidate_sum = 0.0
        baseline_sum = 0.0
        count = 0
        for source in sampled_sources:
            source_episodes = episode_by_source[str(source)]
            sampled_episodes = rng.choice(
                source_episodes, size=len(source_episodes), replace=True
            )
            for episode in sampled_episodes:
                c_sum, b_sum, n = episode_stats[(str(source), str(episode))]
                candidate_sum += c_sum
                baseline_sum += b_sum
                count += n
        candidate_mean = candidate_sum / max(count, 1)
        baseline_mean = baseline_sum / max(count, 1)
        samples[sample_index] = (
            (baseline_mean - candidate_mean) / baseline_mean
            if baseline_mean > 1e-12
            else 0.0
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "meaning": "positive values favor candidate (relative loss reduction)",
        "relative_improvement": float(observed),
        "ci95": [float(lower), float(upper)],
        "candidate_mean_loss": observed_candidate,
        "baseline_mean_loss": observed_baseline,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "grouping": "resample sources, then episodes within sampled source",
    }


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, str]:
    model = RidgeModel.fit(train_x, train_y)
    return model.predict(validation_x), model.fit_sha256()


def _synthetic_reward_contract() -> dict[str, Any]:
    """Prove the reward information gate isolates object from action signal."""
    rng = np.random.default_rng(910247)
    train_action = rng.normal(size=(1024, 2))
    validation_action = rng.normal(size=(512, 2))
    train_object = rng.normal(size=(1024, 4))
    validation_object = rng.normal(size=(512, 4))

    def relative_object_increment(
        train_target: np.ndarray, validation_target: np.ndarray
    ) -> float:
        action_model = RidgeModel.fit(train_action, train_target)
        object_action_model = RidgeModel.fit(
            np.concatenate([train_object, train_action], axis=1), train_target
        )
        action_error = np.square(
            action_model.predict(validation_action).reshape(-1) - validation_target
        ).mean()
        object_action_error = np.square(
            object_action_model.predict(
                np.concatenate([validation_object, validation_action], axis=1)
            ).reshape(-1)
            - validation_target
        ).mean()
        return float((action_error - object_action_error) / max(action_error, 1e-12))

    train_action_only_target = 1.5 * train_action[:, 0] - 0.75 * train_action[:, 1]
    validation_action_only_target = (
        1.5 * validation_action[:, 0] - 0.75 * validation_action[:, 1]
    )
    noise_only_increment = relative_object_increment(
        train_action_only_target, validation_action_only_target
    )
    train_object_signal_target = train_action_only_target + 2.5 * train_object[:, 0]
    validation_object_signal_target = (
        validation_action_only_target + 2.5 * validation_object[:, 0]
    )
    signal_increment = relative_object_increment(
        train_object_signal_target, validation_object_signal_target
    )
    noise_rejected = noise_only_increment < INFORMATION_MIN_RELATIVE_IMPROVEMENT
    signal_accepted = signal_increment >= INFORMATION_MIN_RELATIVE_IMPROVEMENT
    if not noise_rejected or not signal_accepted:
        raise RuntimeError(
            "Synthetic reward contract failed to distinguish action-only from "
            "genuine object information."
        )
    return {
        "status": "pass",
        "comparison": "object+action versus action-only",
        "action_only_with_noise_object": {
            "relative_object_increment": noise_only_increment,
            "information_gate_pass": False,
        },
        "object_contains_signal": {
            "relative_object_increment": signal_increment,
            "information_gate_pass": True,
        },
        "threshold": INFORMATION_MIN_RELATIVE_IMPROVEMENT,
    }


def _fit_projectors(train: ProbeDataset) -> dict[str, PCAProjector]:
    indices = np.flatnonzero(train.reward_valid)
    if len(indices) < 2:
        raise ContractError("Train has fewer than two current-valid transitions.")
    modalities = {
        "query": train.query,
        "mask": train.mask,
        "object": train.object,
        "rgb": train.rgb,
        "background_rgb": train.background_rgb,
    }
    return {
        name: PCAProjector.fit(values[indices], name)
        for name, values in modalities.items()
    }


def _transform_modalities(
    dataset: ProbeDataset, projectors: Mapping[str, PCAProjector]
) -> dict[str, np.ndarray]:
    return {
        "query": projectors["query"].transform(dataset.query),
        "mask": projectors["mask"].transform(dataset.mask),
        "object": projectors["object"].transform(dataset.object),
        "rgb": projectors["rgb"].transform(dataset.rgb),
        "background_rgb": projectors["background_rgb"].transform(
            dataset.background_rgb
        ),
    }


def _design(modality: np.ndarray | None, action: np.ndarray | None) -> np.ndarray:
    values = [value for value in (modality, action) if value is not None]
    if not values:
        raise ContractError("A learned probe needs at least one input.")
    return np.concatenate(values, axis=1) if len(values) > 1 else values[0]


def _probe_reward(
    train: ProbeDataset,
    validation: ProbeDataset,
    train_z: Mapping[str, np.ndarray],
    validation_z: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    ti = np.flatnonzero(train.reward_valid)
    vi = np.flatnonzero(validation.reward_valid)
    train_y = train.rewards[ti]
    validation_y = validation.rewards[vi]
    train_shuffled_action = _shuffle_within_episode(
        train.actions, train.episodes, "reward_train_action"
    )
    validation_shuffled_action = _shuffle_within_episode(
        validation.actions, validation.episodes, "reward_validation_action"
    )
    shuffled_target = _shuffle_episode_blocks(
        train.rewards, train.episodes, "reward_train_target"
    )

    train_designs = {
        "action_only": _design(None, train.actions),
        "query_only": _design(train_z["query"], None),
        "query_only_plus_action": _design(train_z["query"], train.actions),
        "mask_only": _design(train_z["mask"], None),
        "mask_only_plus_action": _design(train_z["mask"], train.actions),
        "object_only": _design(train_z["object"], None),
        "object_plus_action": _design(train_z["object"], train.actions),
        "rgb_pca_only": _design(train_z["rgb"], None),
        "rgb_pca_plus_action": _design(train_z["rgb"], train.actions),
        "object_plus_rgb": _design(
            np.concatenate([train_z["object"], train_z["rgb"]], axis=1), None
        ),
        "object_plus_rgb_plus_action": _design(
            np.concatenate([train_z["object"], train_z["rgb"]], axis=1),
            train.actions,
        ),
        "background_only_plus_action": _design(
            train_z["background_rgb"], train.actions
        ),
        "object_plus_shuffled_action": _design(
            train_z["object"], train_shuffled_action
        ),
        "object_plus_action_shuffled_target": _design(
            train_z["object"], train.actions
        ),
    }
    validation_designs = {
        "action_only": _design(None, validation.actions),
        "query_only": _design(validation_z["query"], None),
        "query_only_plus_action": _design(validation_z["query"], validation.actions),
        "mask_only": _design(validation_z["mask"], None),
        "mask_only_plus_action": _design(validation_z["mask"], validation.actions),
        "object_only": _design(validation_z["object"], None),
        "object_plus_action": _design(validation_z["object"], validation.actions),
        "rgb_pca_only": _design(validation_z["rgb"], None),
        "rgb_pca_plus_action": _design(validation_z["rgb"], validation.actions),
        "object_plus_rgb": _design(
            np.concatenate([validation_z["object"], validation_z["rgb"]], axis=1),
            None,
        ),
        "object_plus_rgb_plus_action": _design(
            np.concatenate([validation_z["object"], validation_z["rgb"]], axis=1),
            validation.actions,
        ),
        "background_only_plus_action": _design(
            validation_z["background_rgb"], validation.actions
        ),
        "object_plus_shuffled_action": _design(
            validation_z["object"], validation_shuffled_action
        ),
        "object_plus_action_shuffled_target": _design(
            validation_z["object"], validation.actions
        ),
    }

    predictions: dict[str, np.ndarray] = {
        "constant": np.full(
            len(vi), float(train_y.mean()), dtype=np.float64
        )
    }
    fit_hashes: dict[str, str | None] = {"constant": None}
    for name in train_designs:
        target = shuffled_target[ti] if name.endswith("shuffled_target") else train_y
        predictions[name], fit_hashes[name] = _fit_predict(
            train_designs[name][ti], target, validation_designs[name][vi]
        )

    metrics: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        metric, loss = _reward_metrics(validation_y, prediction)
        metric["fit_sha256"] = fit_hashes[name]
        metrics[name] = metric
        losses[name] = loss

    sources = validation.sources[vi]
    episodes = validation.episodes[vi]
    comparisons = {
        "object_vs_constant": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["constant"], sources, episodes,
            "reward_object_vs_constant",
        ),
        "object_increment_over_action": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["action_only"], sources, episodes,
            "reward_object_increment_over_action",
        ),
        "action_increment": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["object_only"], sources, episodes,
            "reward_action_increment",
        ),
        "object_vs_rgb": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["rgb_pca_plus_action"],
            sources, episodes, "reward_object_vs_rgb",
        ),
        "shuffled_action_increment": _hierarchical_bootstrap_comparison(
            losses["object_plus_shuffled_action"], losses["object_only"],
            sources, episodes, "reward_shuffled_action_increment",
        ),
        "background_increment_over_action": _hierarchical_bootstrap_comparison(
            losses["background_only_plus_action"], losses["action_only"],
            sources, episodes, "reward_background_increment",
        ),
        "shuffled_target_vs_constant": _hierarchical_bootstrap_comparison(
            losses["object_plus_action_shuffled_target"], losses["constant"],
            sources, episodes, "reward_shuffled_target",
        ),
        "shuffled_target_vs_action": _hierarchical_bootstrap_comparison(
            losses["object_plus_action_shuffled_target"], losses["action_only"],
            sources, episodes, "reward_shuffled_target_vs_action",
        ),
    }
    positive_train = int(np.count_nonzero(train_y > 0))
    positive_validation = int(np.count_nonzero(validation_y > 0))
    positive_sources = len(set(sources[validation_y > 0].tolist()))
    power_sufficient = (
        positive_train >= MIN_TRAIN_POSITIVES
        and positive_validation >= MIN_VALIDATION_POSITIVES
        and positive_sources >= MIN_VALIDATION_POSITIVE_SOURCES
    )
    return {
        "target": "real environment reward for action_t (two simulator substeps)",
        "sample_subset": (
            "whole_arm and goal explicitly valid throughout the causal "
            "three-observation history"
        ),
        "train_samples": int(len(ti)),
        "validation_samples": int(len(vi)),
        "validation_sources": len(set(sources.tolist())),
        "validation_episodes": len(set(episodes.tolist())),
        "positive_support": {
            "train_reward_gt_zero": positive_train,
            "validation_reward_gt_zero": positive_validation,
            "validation_sources_with_positive": positive_sources,
            "minimums": {
                "train": MIN_TRAIN_POSITIVES,
                "validation": MIN_VALIDATION_POSITIVES,
                "validation_sources": MIN_VALIDATION_POSITIVE_SOURCES,
            },
            "sufficient": bool(power_sufficient),
        },
        "models": metrics,
        "comparisons": comparisons,
    }


def _probe_dynamics(
    train: ProbeDataset,
    validation: ProbeDataset,
    train_z: Mapping[str, np.ndarray],
    validation_z: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    ti = np.flatnonzero(train.dynamics_valid)
    vi = np.flatnonzero(validation.dynamics_valid)
    if len(ti) < 2 or len(vi) < 1:
        raise ContractError("Insufficient common valid transitions for dynamics probe.")
    # The goal is static within an episode.  It is intentionally excluded from
    # the primary target so it cannot dilute whole-arm dynamics with an easy
    # persistence signal.
    whole_arm_slice = slice(0, SPATIAL_DIM_PER_ROLE)
    train_y = train.next_spatial[ti, whole_arm_slice]
    validation_y = validation.next_spatial[vi, whole_arm_slice]
    target_scale = train_y.std(axis=0, ddof=0).astype(np.float64)
    target_scale[target_scale < 1e-6] = 1.0
    train_shuffled_action = _shuffle_within_episode(
        train.actions, train.episodes, "dynamics_train_action"
    )
    validation_shuffled_action = _shuffle_within_episode(
        validation.actions, validation.episodes, "dynamics_validation_action"
    )
    train_designs = {
        "object_only": _design(train_z["object"], None),
        "object_plus_action": _design(train_z["object"], train.actions),
        "rgb_pca_plus_action": _design(train_z["rgb"], train.actions),
        "object_plus_shuffled_action": _design(
            train_z["object"], train_shuffled_action
        ),
    }
    validation_designs = {
        "object_only": _design(validation_z["object"], None),
        "object_plus_action": _design(validation_z["object"], validation.actions),
        "rgb_pca_plus_action": _design(validation_z["rgb"], validation.actions),
        "object_plus_shuffled_action": _design(
            validation_z["object"], validation_shuffled_action
        ),
    }
    predictions: dict[str, np.ndarray] = {
        "persistence": validation.spatial[vi, whole_arm_slice]
    }
    fit_hashes: dict[str, str | None] = {"persistence": None}
    for name in train_designs:
        predictions[name], fit_hashes[name] = _fit_predict(
            train_designs[name][ti], train_y, validation_designs[name][vi]
        )
    metrics: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        metric, loss = _dynamics_metrics(validation_y, prediction, target_scale)
        metric["fit_sha256"] = fit_hashes[name]
        metrics[name] = metric
        losses[name] = loss
    sources = validation.sources[vi]
    episodes = validation.episodes[vi]
    comparisons = {
        "object_vs_persistence": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["persistence"], sources, episodes,
            "dynamics_object_vs_persistence",
        ),
        "action_increment": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["object_only"], sources, episodes,
            "dynamics_action_increment",
        ),
        "object_vs_rgb": _hierarchical_bootstrap_comparison(
            losses["object_plus_action"], losses["rgb_pca_plus_action"],
            sources, episodes, "dynamics_object_vs_rgb",
        ),
        "shuffled_action_increment": _hierarchical_bootstrap_comparison(
            losses["object_plus_shuffled_action"], losses["object_only"],
            sources, episodes, "dynamics_shuffled_action_increment",
        ),
    }
    return {
        "target": (
            "next whole_arm generic mask spatial state: 8x8 occupancy, "
            "centroid, area, bounding box, and central moments"
        ),
        "target_dim": int(train_y.shape[1]),
        "sample_subset": (
            "whole_arm and goal explicitly valid throughout the causal "
            "three-observation history and at the next observation"
        ),
        "train_samples": int(len(ti)),
        "validation_samples": int(len(vi)),
        "validation_sources": len(set(sources.tolist())),
        "validation_episodes": len(set(episodes.tolist())),
        "models": metrics,
        "comparisons": comparisons,
        "goal_policy": (
            "goal remains in current object conditioning but is excluded from "
            "the next-state target and every primary dynamics metric"
        ),
    }


def _comparison_pass(value: Mapping[str, Any], threshold: float) -> bool:
    return bool(
        value["relative_improvement"] >= threshold and value["ci95"][0] > 0.0
    )


def _coverage_status(train: ProbeDataset, validation: ProbeDataset) -> dict[str, Any]:
    train_current = train.coverage["reward_history_all_roles_valid_fraction"]
    validation_current = validation.coverage["reward_history_all_roles_valid_fraction"]
    train_dynamics = train.coverage["dynamics_history_and_next_all_roles_valid_fraction"]
    validation_dynamics = validation.coverage[
        "dynamics_history_and_next_all_roles_valid_fraction"
    ]
    maximum_burst = max(
        train.coverage["invalid_bursts"]["maximum_frames"],
        validation.coverage["invalid_bursts"]["maximum_frames"],
    )
    passed = bool(
        train_current >= MIN_CURRENT_VALID_COVERAGE
        and validation_current >= MIN_CURRENT_VALID_COVERAGE
        and train_dynamics >= MIN_DYNAMICS_VALID_COVERAGE
        and validation_dynamics >= MIN_DYNAMICS_VALID_COVERAGE
        and maximum_burst <= MAX_INVALID_BURST
    )
    return {
        "status": "pass" if passed else "fail",
        "hard_gate": True,
        "thresholds": {
            "reward_history_valid_fraction_min": MIN_CURRENT_VALID_COVERAGE,
            "dynamics_history_and_next_valid_fraction_min": MIN_DYNAMICS_VALID_COVERAGE,
            "invalid_burst_max_frames": MAX_INVALID_BURST,
        },
        "observed": {
            "train_reward_history_valid_fraction": train_current,
            "validation_reward_history_valid_fraction": validation_current,
            "train_dynamics_history_and_next_valid_fraction": train_dynamics,
            "validation_dynamics_history_and_next_valid_fraction": validation_dynamics,
            "maximum_invalid_burst_frames": maximum_burst,
        },
    }


def _conclusion_status(
    coverage: Mapping[str, Any],
    reward: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    validation: ProbeDataset,
) -> dict[str, Any]:
    # The full probe reports subset-specific counts.  Keep this conclusion
    # helper conservative for reduced/underpowered audit fixtures: absent
    # counts fall back to the validation statistical units, while an absent
    # required comparison makes only the affected conclusion inconclusive.
    validation_source_count = len(set(np.asarray(validation.sources).tolist()))
    validation_episode_count = len(set(np.asarray(validation.episodes).tolist()))
    reward_source_count = int(
        reward.get("validation_sources", validation_source_count)
    )
    reward_episode_count = int(
        reward.get("validation_episodes", validation_episode_count)
    )
    dynamics_source_count = int(
        dynamics.get("validation_sources", validation_source_count)
    )
    dynamics_episode_count = int(
        dynamics.get("validation_episodes", validation_episode_count)
    )
    reward_comparisons = reward.get("comparisons", {})
    dynamics_comparisons = dynamics.get("comparisons", {})
    if not isinstance(reward_comparisons, Mapping) or not isinstance(
        dynamics_comparisons, Mapping
    ):
        raise ContractError("Probe comparisons must be mappings.")

    neutral_comparison = {"relative_improvement": 0.0, "ci95": [0.0, 0.0]}

    def comparison(
        comparisons: Mapping[str, Any], key: str
    ) -> Mapping[str, Any]:
        value = comparisons.get(key)
        return value if isinstance(value, Mapping) else neutral_comparison

    reward_grouped_power = (
        reward_source_count >= MIN_VALIDATION_SOURCES
        and reward_episode_count >= MIN_VALIDATION_EPISODES
    )
    dynamics_grouped_power = (
        dynamics_source_count >= MIN_VALIDATION_SOURCES
        and dynamics_episode_count >= MIN_VALIDATION_EPISODES
    )
    reward_power = bool(reward["positive_support"]["sufficient"])
    reward_information_contract_complete = all(
        key in reward_comparisons
        for key in (
            "object_increment_over_action",
            "shuffled_target_vs_constant",
            "shuffled_target_vs_action",
        )
    )
    reward_base_inconclusive = (
        coverage["status"] != "pass"
        or not reward_grouped_power
        or not reward_information_contract_complete
    )
    dynamics_base_inconclusive = (
        coverage["status"] != "pass" or not dynamics_grouped_power
    )

    reward_information = _comparison_pass(
        comparison(reward_comparisons, "object_increment_over_action"),
        INFORMATION_MIN_RELATIVE_IMPROVEMENT,
    )
    dynamics_information = _comparison_pass(
        comparison(dynamics_comparisons, "object_vs_persistence"),
        INFORMATION_MIN_RELATIVE_IMPROVEMENT,
    )
    shuffled_target_ok = (
        comparison(reward_comparisons, "shuffled_target_vs_constant")[
            "relative_improvement"
        ]
        <= NEGATIVE_CONTROL_MAX_INCREMENT
        and comparison(reward_comparisons, "shuffled_target_vs_constant")["ci95"][0]
        <= 0.0
        and comparison(reward_comparisons, "shuffled_target_vs_action")[
            "relative_improvement"
        ]
        <= NEGATIVE_CONTROL_MAX_INCREMENT
        and comparison(reward_comparisons, "shuffled_target_vs_action")["ci95"][0]
        <= 0.0
    )
    reward_information_status = (
        "inconclusive"
        if reward_base_inconclusive or not reward_power
        else ("pass" if reward_information and shuffled_target_ok else "fail")
    )
    dynamics_information_status = (
        "inconclusive"
        if dynamics_base_inconclusive
        else ("pass" if dynamics_information else "fail")
    )

    reward_action = _comparison_pass(
        comparison(reward_comparisons, "action_increment"),
        ACTION_MIN_RELATIVE_IMPROVEMENT,
    )
    dynamics_action = _comparison_pass(
        comparison(dynamics_comparisons, "action_increment"),
        ACTION_MIN_RELATIVE_IMPROVEMENT,
    )
    shuffled_reward_ok = (
        comparison(reward_comparisons, "shuffled_action_increment")[
            "relative_improvement"
        ]
        <= NEGATIVE_CONTROL_MAX_INCREMENT
        and comparison(reward_comparisons, "shuffled_action_increment")["ci95"][0]
        <= 0.0
    )
    shuffled_dynamics_ok = (
        comparison(dynamics_comparisons, "shuffled_action_increment")[
            "relative_improvement"
        ]
        <= NEGATIVE_CONTROL_MAX_INCREMENT
        and comparison(dynamics_comparisons, "shuffled_action_increment")["ci95"][0]
        <= 0.0
    )
    reward_action_status = (
        "inconclusive"
        if reward_base_inconclusive or not reward_power
        else ("pass" if reward_action and shuffled_reward_ok else "fail")
    )
    dynamics_action_status = (
        "inconclusive"
        if dynamics_base_inconclusive
        else ("pass" if dynamics_action and shuffled_dynamics_ok else "fail")
    )

    reward_rgb = comparison(reward_comparisons, "object_vs_rgb")
    dynamics_rgb = comparison(dynamics_comparisons, "object_vs_rgb")
    reward_noninferior = reward_rgb["ci95"][0] >= -RGB_NONINFERIORITY_MARGIN
    dynamics_noninferior = dynamics_rgb["ci95"][0] >= -RGB_NONINFERIORITY_MARGIN
    reward_rgb_status = (
        "inconclusive"
        if reward_base_inconclusive or not reward_power
        else ("pass" if reward_noninferior else "fail")
    )
    dynamics_rgb_status = (
        "inconclusive"
        if dynamics_base_inconclusive
        else ("pass" if dynamics_noninferior else "fail")
    )

    background_control = comparison(
        reward_comparisons, "background_increment_over_action"
    )
    background_control_ok = bool(
        background_control["relative_improvement"]
        <= NEGATIVE_CONTROL_MAX_INCREMENT
        and background_control["ci95"][0] <= 0.0
    )
    if not background_control_ok:
        reward_information_status = "inconclusive"
        dynamics_information_status = "inconclusive"
        reward_action_status = "inconclusive"
        dynamics_action_status = "inconclusive"
        reward_rgb_status = "inconclusive"
        dynamics_rgb_status = "inconclusive"

    def joint_status(*values: str) -> str:
        if "fail" in values:
            return "fail"
        if all(value == "pass" for value in values):
            return "pass"
        return "inconclusive"

    return {
        "data_quality": coverage,
        "object_information": {
            "status": joint_status(
                reward_information_status, dynamics_information_status
            ),
            "reward_status": reward_information_status,
            "dynamics_status": dynamics_information_status,
            "requires": (
                "reward object+a improves action-only and dynamics object+a improves "
                "persistence by >=10%, paired CI lower bound >0, with shuffled-target "
                "controls no better than constant or action-only"
            ),
            "reward_pass": bool(reward_information),
            "dynamics_pass": bool(dynamics_information),
            "shuffled_target_control_pass": bool(shuffled_target_ok),
        },
        "action_increment": {
            # A state-defined reward can be conditionally independent of the
            # current action even when the transition model correctly needs it.
            # Dynamics is therefore the hard action-sufficiency gate; reward is
            # retained as a transparent task-specific diagnostic.
            "status": dynamics_action_status,
            "hard_gate_modality": "dynamics",
            "reward_status": reward_action_status,
            "dynamics_status": dynamics_action_status,
            "requires": (
                "dynamics object+a improves object-only by >=5%, paired CI lower "
                "bound >0, and shuffled action removes the gain; reward action "
                "increment is diagnostic because rewards may be state-defined"
            ),
            "reward_pass": bool(reward_action),
            "dynamics_pass": bool(dynamics_action),
            "reward_shuffle_control_pass": bool(shuffled_reward_ok),
            "dynamics_shuffle_control_pass": bool(shuffled_dynamics_ok),
        },
        "RGB_noninferiority": {
            "status": joint_status(reward_rgb_status, dynamics_rgb_status),
            "reward_status": reward_rgb_status,
            "dynamics_status": dynamics_rgb_status,
            "margin": RGB_NONINFERIORITY_MARGIN,
            "requires": "paired CI lower bound for object-vs-RGB improvement >= -5%",
            "reward_pass": bool(reward_noninferior),
            "dynamics_pass": bool(dynamics_noninferior),
        },
        "background_leakage_control": {
            "status": "pass" if background_control_ok else "fail",
            "comparison": "background-only RGB + action versus action-only",
            "maximum_increment": NEGATIVE_CONTROL_MAX_INCREMENT,
        },
        "statistical_units": {
            "minimum_sources": MIN_VALIDATION_SOURCES,
            "minimum_episodes": MIN_VALIDATION_EPISODES,
            "reward": {
                "validation_sources": reward_source_count,
                "validation_episodes": reward_episode_count,
                "sufficient": bool(reward_grouped_power),
            },
            "dynamics": {
                "validation_sources": dynamics_source_count,
                "validation_episodes": dynamics_episode_count,
                "sufficient": bool(dynamics_grouped_power),
            },
        },
    }


def _frozen_configuration(
    manifest_hashes: Mapping[str, str],
    perception_provenance_sha256: str,
    script_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": {
            "report": REPORT_FORMAT,
            "rollout": ROLLOUT_FORMAT,
            "features": FEATURE_FORMAT,
            "task": TASK,
            "roles": list(ROLES),
        },
        "inputs": dict(manifest_hashes),
        "perception_provenance_sha256": perception_provenance_sha256,
        "script_sha256": script_sha256,
        "object_representation": {
            "query_pool": "reshape 2048 to 8x256, concatenate mean and population std",
            "query_slot_permutation_invariant": True,
            "stack": (
                "causal 3-observation stack: t=0 [0,0,0], "
                "t=1 [0,0,1], otherwise [t-2,t-1,t]"
            ),
            "mask_spatial": (
                "8x8 occupancy + normalized centroid/area/bbox + normalized central moments"
            ),
            "status": ["confidence", "lost", "valid", "mask_score"],
            "forbidden": ["manual points", "point decoder", "physics/simulator state"],
        },
        "rgb_representation": {
            "stack": (
                "causal 3-observation stack: t=0 [0,0,0], t=1 [0,0,1], "
                "otherwise [t-2,t-1,t]"
            ),
            "native_frame_shape": [64, 64, 3],
            "deterministic_area_pool": [8, 8],
            "background_control_mask_dilation_radius": MASK_DILATION_RADIUS,
            "background_control": (
                "per-frame RGB channel mean/std/q10/q25/q50/q75/q90 over only "
                "non-object pixels after mask dilation; no spatial layout or mask area"
            ),
        },
        "learner": {
            "implementation": "numpy deterministic train-only PCA/scaler + ridge",
            "pca_components_max": PCA_COMPONENTS,
            "pca_oversample": PCA_OVERSAMPLE,
            "pca_power_iterations": PCA_POWER_ITERATIONS,
            "ridge_alpha": RIDGE_ALPHA,
            "seed": RANDOM_SEED,
            "validation_policy": "single frozen transform/evaluation; never fit or tune",
            "synthetic_reward_contract": (
                "action-only signal plus noise object must fail object-information; "
                "added object signal must pass"
            ),
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "method": "source then episode hierarchical paired bootstrap",
        },
        "thresholds": {
            "minimum_train_positive_rewards": MIN_TRAIN_POSITIVES,
            "minimum_validation_positive_rewards": MIN_VALIDATION_POSITIVES,
            "minimum_validation_positive_sources": MIN_VALIDATION_POSITIVE_SOURCES,
            "minimum_validation_sources": MIN_VALIDATION_SOURCES,
            "minimum_validation_episodes": MIN_VALIDATION_EPISODES,
            "minimum_reward_history_valid_coverage": MIN_CURRENT_VALID_COVERAGE,
            "minimum_dynamics_history_and_next_valid_coverage": (
                MIN_DYNAMICS_VALID_COVERAGE
            ),
            "maximum_invalid_burst": MAX_INVALID_BURST,
            "object_information_relative_improvement": (
                INFORMATION_MIN_RELATIVE_IMPROVEMENT
            ),
            "object_information_reward_baseline": "action_only",
            "object_information_dynamics_baseline": "persistence",
            "action_increment_relative_improvement": ACTION_MIN_RELATIVE_IMPROVEMENT,
            "rgb_noninferiority_margin": RGB_NONINFERIORITY_MARGIN,
            "negative_control_max_increment": NEGATIVE_CONTROL_MAX_INCREMENT,
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    synthetic_contract = _synthetic_reward_contract()

    train_rollout = _load_rollout_manifest(args.train_rollout, "train")
    validation_rollout = _load_rollout_manifest(
        args.validation_rollout, "validation"
    )
    train_features = _load_feature_manifest(
        args.train_features, "train", train_rollout
    )
    validation_features = _load_feature_manifest(
        args.validation_features, "validation", validation_rollout
    )
    _validate_cross_split(
        train_rollout, validation_rollout, train_features, validation_features
    )

    # All fitting, including PCA/scaling, happens before validation arrays are
    # materialized.  Reading validation manifest metadata above is only contract
    # validation and never supplies a fitted value.
    train = _load_dataset(train_rollout, train_features)
    projectors = _fit_projectors(train)
    train_z = _transform_modalities(train, projectors)

    # Single frozen validation materialization and transform.
    validation = _load_dataset(validation_rollout, validation_features)
    if train.actions.shape[1] != validation.actions.shape[1]:
        raise ContractError("Train and validation action dimensions differ.")
    validation_z = _transform_modalities(validation, projectors)

    reward = _probe_reward(train, validation, train_z, validation_z)
    dynamics = _probe_dynamics(train, validation, train_z, validation_z)
    coverage_status = _coverage_status(train, validation)
    status = _conclusion_status(coverage_status, reward, dynamics, validation)

    script_sha256 = _file_sha256(Path(__file__).resolve())
    manifest_hashes = {
        "train_rollout_sha256": train_rollout.sha256,
        "train_features_sha256": train_features.sha256,
        "validation_rollout_sha256": validation_rollout.sha256,
        "validation_features_sha256": validation_features.sha256,
    }
    configuration = _frozen_configuration(
        manifest_hashes,
        train_features.provenance_sha256,
        script_sha256,
    )
    configuration_sha256 = _canonical_sha256(configuration)
    report = {
        "format": REPORT_FORMAT,
        "status": status,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "inputs": {
            **manifest_hashes,
            "train_sources": sorted(train_rollout.sources),
            "validation_sources": sorted(validation_rollout.sources),
            "train_env_seeds": sorted(train_rollout.env_seeds),
            "validation_env_seeds": sorted(validation_rollout.env_seeds),
            "train_action_seeds": sorted(train_rollout.action_seeds),
            "validation_action_seeds": sorted(validation_rollout.action_seeds),
            "train_background_seeds": sorted(train_rollout.background_seeds),
            "validation_background_seeds": sorted(
                validation_rollout.background_seeds
            ),
            "perception_provenance_sha256": train_features.provenance_sha256,
        },
        "isolation": {
            "accepted_splits": list(ALLOWED_SPLITS),
            "test_trajectory_pixels_read": False,
            "support_trajectory_pixels_read": False,
            "verified_support_prompt_used_by_cutie": True,
            "sequence_source_seed_image_hash_overlap": False,
            "perception_provenance_equal": True,
            "validation_fit_or_tuning": False,
        },
        "scope": {
            "decision_level": "eligibility_for_small_matched_rl_pilot",
            "causal_background_invariance_established": False,
            "paired_background_counterfactual_included": False,
            "limitation": (
                "Disjoint validation backgrounds and the background-only negative "
                "control test transfer and obvious shortcuts, but do not constitute "
                "a same-physics paired causal background intervention."
            ),
        },
        "coverage": {"train": train.coverage, "validation": validation.coverage},
        "projectors": {
            name: projector.metadata() for name, projector in projectors.items()
        },
        "synthetic_contract": synthetic_contract,
        "reward_probe": reward,
        "dynamics_probe": dynamics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-rollout", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print(
        json.dumps(
            {
                "format": report["format"],
                "configuration_sha256": report["configuration_sha256"],
                "status": report["status"],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
