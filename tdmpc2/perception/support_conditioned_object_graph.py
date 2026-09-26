"""Support-conditioned visual entities and semantic object-role tokens.

The tracker-facing object identity and the controller-facing semantic role are
deliberately separate in this module.  A declarative graph can ask Cutie to
track one or more visually stable *entities*, then project each entity into one
or more ordered semantic roles.  The execution core never branches on a task
name: all task-specific choices live in an immutable JSON graph.

Only fixed support RGB/masks and the current causal tracker result are accepted.
Episode ground truth, simulator state, rewards, actions, and future frames are
not part of any public method in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


GRAPH_FORMAT = "support_conditioned_object_graph_v1"
DESCRIPTOR_FORMAT = "shared_entity_appearance_role_mask_geometry_status_590_v1"
ENTITY_SUPPORT_FORMAT = "object_graph_entity_support_v1"
TOKENIZER_FORMAT = "support_conditioned_entity_role_tokenizer_v1"
TEMPORAL_TOKENIZER_FORMAT = "support_conditioned_entity_role_tokenizer_temporal_v2"
TEMPORAL_STATE_PROTOCOL = "causal_current_mask_supported_ordered_chain_state_v2"
TEMPORAL_MAX_SCORED_CANDIDATES = 24
TEMPORAL_MAX_RETAINED_CANDIDATES = 12

QUERY_SLOTS = 8
QUERY_DIM = 256
QUERY_FEATURE_DIM = QUERY_SLOTS * QUERY_DIM
APPEARANCE_DIM = QUERY_DIM * 2
MASK_POOL_SIZE = 8
GEOMETRY_DIM = MASK_POOL_SIZE * MASK_POOL_SIZE + 10
STATUS_DIM = 4
FRAME_DIM = APPEARANCE_DIM + GEOMETRY_DIM + STATUS_DIM

_PROJECTORS = {
    "direct_role_v1",
    "ordered_chain_segments_v1",
    "ordered_chain_temporal_v2",
}
_RELATIONS = {
    "independent_v1",
    "revolute_joint_v1",
    "prismatic_joint_v1",
    "spatial_target_v1",
}


class ObjectGraphContractError(ValueError):
    """Raised when a graph, support pack, or causal frame violates the contract."""


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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ObjectGraphContractError(f"{label} must be a non-empty trimmed string.")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ObjectGraphContractError(f"{label} contains a forbidden character.")
    return value


def _strict_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObjectGraphContractError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ObjectGraphContractError(f"{label} must be finite.")
    if minimum is not None and result < minimum:
        raise ObjectGraphContractError(f"{label} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ObjectGraphContractError(f"{label} must be <= {maximum}.")
    return result


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ObjectGraphContractError(f"{label} must be an integer >= {minimum}.")
    return value


@dataclass(frozen=True)
class ProjectorSpec:
    type: str
    roles: tuple[str, ...]
    minimum_entity_pixels: int
    maximum_base_distance_fraction: float
    minimum_path_length_fraction: float
    maximum_path_length_fraction: float
    maximum_disconnected_fraction: float
    maximum_branch_fraction: float
    maximum_entity_area_multiple: float
    minimum_projection_confidence: float


@dataclass(frozen=True)
class EntitySpec:
    name: str
    source_roles: tuple[str, ...]
    projector: ProjectorSpec


@dataclass(frozen=True)
class RelationSpec:
    type: str
    parent: str
    child: str


@dataclass(frozen=True)
class CompiledObjectGraph:
    """Strict, path-independent compiled representation of one graph JSON."""

    graph_name: str
    task: str
    source_roles: tuple[str, ...]
    semantic_roles: tuple[str, ...]
    entities: tuple[EntitySpec, ...]
    relations: tuple[RelationSpec, ...]
    stack_frames: int
    canonical_payload: dict[str, Any]
    graph_sha256: str
    source_path: Path | None

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.entities)

    @property
    def role_to_entity_index(self) -> tuple[int, ...]:
        owner: dict[str, int] = {}
        for entity_index, entity in enumerate(self.entities):
            for role in entity.projector.roles:
                owner[role] = entity_index
        return tuple(owner[role] for role in self.semantic_roles)

    def metadata(self) -> dict[str, Any]:
        return {
            "format": GRAPH_FORMAT,
            "graph_name": self.graph_name,
            "task": self.task,
            "source_roles": list(self.source_roles),
            "semantic_roles": list(self.semantic_roles),
            "entity_names": list(self.entity_names),
            "role_to_entity_index": list(self.role_to_entity_index),
            "relations": [
                {
                    "type": relation.type,
                    "parent": relation.parent,
                    "child": relation.child,
                }
                for relation in self.relations
            ],
            "relation_policy": (
                "declarative_provenance_only_not_an_explicit_descriptor_field_v1"
            ),
            "descriptor_format": DESCRIPTOR_FORMAT,
            "frame_shape": [len(self.semantic_roles), FRAME_DIM],
            "stack_shape": [len(self.semantic_roles), FRAME_DIM * self.stack_frames],
            "graph_sha256": self.graph_sha256,
        }


def compile_object_graph(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> CompiledObjectGraph:
    """Validate and compile a graph without consulting any task registry."""
    if not isinstance(payload, Mapping):
        raise ObjectGraphContractError("Object graph must be a JSON object.")
    expected_top = {
        "format",
        "graph_name",
        "task",
        "source_roles",
        "semantic_roles",
        "tracking_entities",
        "relations",
        "descriptor",
    }
    if set(payload) != expected_top:
        raise ObjectGraphContractError(
            f"Object graph fields must be exactly {sorted(expected_top)!r}."
        )
    if payload.get("format") != GRAPH_FORMAT:
        raise ObjectGraphContractError(f"Graph format must be {GRAPH_FORMAT!r}.")
    graph_name = _strict_name(payload.get("graph_name"), "graph_name")
    task = _strict_name(payload.get("task"), "task")

    def ordered_names(value: Any, label: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise ObjectGraphContractError(f"{label} must be a non-empty list.")
        result = tuple(_strict_name(item, f"{label}[]") for item in value)
        if len(set(result)) != len(result):
            raise ObjectGraphContractError(f"{label} must be unique and ordered.")
        return result

    source_roles = ordered_names(payload.get("source_roles"), "source_roles")
    semantic_roles = ordered_names(payload.get("semantic_roles"), "semantic_roles")
    if set(source_roles) != set(semantic_roles):
        raise ObjectGraphContractError(
            "Version 1 requires source_roles and semantic_roles to have the same set."
        )

    raw_entities = payload.get("tracking_entities")
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ObjectGraphContractError("tracking_entities must be a non-empty list.")
    entities: list[EntitySpec] = []
    assigned_sources: list[str] = []
    projected_roles: list[str] = []
    for entity_index, raw in enumerate(raw_entities):
        label = f"tracking_entities[{entity_index}]"
        if not isinstance(raw, Mapping) or set(raw) != {
            "name", "source_roles", "projector"
        }:
            raise ObjectGraphContractError(f"{label} fields are malformed.")
        name = _strict_name(raw.get("name"), f"{label}.name")
        entity_source_roles = ordered_names(
            raw.get("source_roles"), f"{label}.source_roles"
        )
        if not set(entity_source_roles).issubset(source_roles):
            raise ObjectGraphContractError(f"{label} references an unknown source role.")
        assigned_sources.extend(entity_source_roles)
        projector = raw.get("projector")
        if not isinstance(projector, Mapping) or "type" not in projector:
            raise ObjectGraphContractError(f"{label}.projector must be an object.")
        projector_type = projector.get("type")
        if projector_type not in _PROJECTORS:
            raise ObjectGraphContractError(
                f"{label}.projector.type must be one of {sorted(_PROJECTORS)!r}."
            )
        if projector_type == "direct_role_v1":
            if set(projector) != {"type", "role"}:
                raise ObjectGraphContractError(
                    f"{label} direct projector fields are malformed."
                )
            projector_roles = (
                _strict_name(projector.get("role"), f"{label}.projector.role"),
            )
            if entity_source_roles != projector_roles:
                raise ObjectGraphContractError(
                    f"{label} direct projector must preserve its one source role."
                )
            spec = ProjectorSpec(
                type=projector_type,
                roles=projector_roles,
                minimum_entity_pixels=1,
                maximum_base_distance_fraction=1.0,
                minimum_path_length_fraction=0.0,
                maximum_path_length_fraction=float("inf"),
                maximum_disconnected_fraction=1.0,
                maximum_branch_fraction=1.0,
                maximum_entity_area_multiple=float("inf"),
                minimum_projection_confidence=0.0,
            )
        else:
            expected = {
                "type",
                "ordered_roles",
                "minimum_entity_pixels",
                "maximum_base_distance_fraction",
                "minimum_path_length_fraction",
                "maximum_path_length_fraction",
                "maximum_disconnected_fraction",
                "maximum_branch_fraction",
                "maximum_entity_area_multiple",
                "minimum_projection_confidence",
            }
            if set(projector) != expected:
                raise ObjectGraphContractError(
                    f"{label} ordered-chain projector fields are malformed."
                )
            projector_roles = ordered_names(
                projector.get("ordered_roles"),
                f"{label}.projector.ordered_roles",
            )
            if len(projector_roles) < 2 or projector_roles != entity_source_roles:
                raise ObjectGraphContractError(
                    f"{label} ordered roles must equal its source roles in order."
                )
            if (
                projector_type == "ordered_chain_temporal_v2"
                and len(projector_roles) != 2
            ):
                raise ObjectGraphContractError(
                    f"{label} ordered_chain_temporal_v2 currently requires exactly "
                    "two ordered roles."
                )
            minimum_path = _strict_float(
                projector.get("minimum_path_length_fraction"),
                f"{label}.projector.minimum_path_length_fraction",
                minimum=0.05,
                maximum=1.0,
            )
            maximum_path = _strict_float(
                projector.get("maximum_path_length_fraction"),
                f"{label}.projector.maximum_path_length_fraction",
                minimum=1.0,
            )
            if minimum_path >= maximum_path:
                raise ObjectGraphContractError(
                    f"{label} path-length bounds must be increasing."
                )
            spec = ProjectorSpec(
                type=projector_type,
                roles=projector_roles,
                minimum_entity_pixels=_strict_int(
                    projector.get("minimum_entity_pixels"),
                    f"{label}.projector.minimum_entity_pixels",
                    minimum=2,
                ),
                maximum_base_distance_fraction=_strict_float(
                    projector.get("maximum_base_distance_fraction"),
                    f"{label}.projector.maximum_base_distance_fraction",
                    minimum=0.01,
                    maximum=1.0,
                ),
                minimum_path_length_fraction=minimum_path,
                maximum_path_length_fraction=maximum_path,
                maximum_disconnected_fraction=_strict_float(
                    projector.get("maximum_disconnected_fraction"),
                    f"{label}.projector.maximum_disconnected_fraction",
                    minimum=0.0,
                    maximum=0.75,
                ),
                maximum_branch_fraction=_strict_float(
                    projector.get("maximum_branch_fraction"),
                    f"{label}.projector.maximum_branch_fraction",
                    minimum=0.0,
                    maximum=0.75,
                ),
                maximum_entity_area_multiple=_strict_float(
                    projector.get("maximum_entity_area_multiple"),
                    f"{label}.projector.maximum_entity_area_multiple",
                    minimum=1.0,
                    maximum=10.0,
                ),
                minimum_projection_confidence=_strict_float(
                    projector.get("minimum_projection_confidence"),
                    f"{label}.projector.minimum_projection_confidence",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        projected_roles.extend(projector_roles)
        entities.append(EntitySpec(name, entity_source_roles, spec))

    if len({entity.name for entity in entities}) != len(entities):
        raise ObjectGraphContractError("Tracking entity names must be unique.")
    if assigned_sources != list(source_roles):
        raise ObjectGraphContractError(
            "Tracking entities must partition source_roles exactly in source order."
        )
    if projected_roles != list(semantic_roles):
        raise ObjectGraphContractError(
            "Projectors must emit every semantic role exactly in semantic order."
        )

    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        raise ObjectGraphContractError("relations must be a list.")
    relations: list[RelationSpec] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relation_index, raw in enumerate(raw_relations):
        label = f"relations[{relation_index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"type", "parent", "child"}:
            raise ObjectGraphContractError(f"{label} fields are malformed.")
        relation_type = raw.get("type")
        if relation_type not in _RELATIONS:
            raise ObjectGraphContractError(
                f"{label}.type must be one of {sorted(_RELATIONS)!r}."
            )
        parent = _strict_name(raw.get("parent"), f"{label}.parent")
        child = _strict_name(raw.get("child"), f"{label}.child")
        if parent not in semantic_roles or child not in semantic_roles or parent == child:
            raise ObjectGraphContractError(f"{label} references invalid semantic roles.")
        identity = (relation_type, parent, child)
        if identity in seen_relations:
            raise ObjectGraphContractError(f"{label} duplicates a relation.")
        seen_relations.add(identity)
        relations.append(RelationSpec(*identity))

    descriptor = payload.get("descriptor")
    expected_descriptor = {
        "format": DESCRIPTOR_FORMAT,
        "appearance_dim": APPEARANCE_DIM,
        "geometry_dim": GEOMETRY_DIM,
        "status_dim": STATUS_DIM,
        "frame_dim": FRAME_DIM,
        "stack_frames": 3,
        "appearance_policy": "shared_within_tracking_entity_v1",
        "geometry_policy": "role_mask_8x8_and_moments_v1",
        "status_policy": "entity_times_projector_reliability_v1",
    }
    if descriptor != expected_descriptor:
        raise ObjectGraphContractError("Descriptor contract changed.")

    canonical_payload = json.loads(_canonical_json_bytes(dict(payload)))
    return CompiledObjectGraph(
        graph_name=graph_name,
        task=task,
        source_roles=source_roles,
        semantic_roles=semantic_roles,
        entities=tuple(entities),
        relations=tuple(relations),
        stack_frames=3,
        canonical_payload=canonical_payload,
        graph_sha256=_sha256_json(canonical_payload),
        source_path=source_path,
    )


def load_object_graph(path: str | Path) -> CompiledObjectGraph:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink():
        raise ObjectGraphContractError(
            f"Graph must be a regular non-symlink file: {candidate}"
        )
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ObjectGraphContractError(f"Graph must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectGraphContractError(f"Invalid graph JSON: {path}") from exc
    return compile_object_graph(payload, source_path=path)


@dataclass(frozen=True)
class EntitySupportArrays:
    rgb: np.ndarray
    indexed_masks: np.ndarray
    source_roles: tuple[str, ...]
    entity_names: tuple[str, ...]
    metadata: dict[str, Any]


def _typed_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def project_support_to_entities(
    rgb: np.ndarray,
    indexed_masks: np.ndarray,
    graph: CompiledObjectGraph,
) -> EntitySupportArrays:
    """Merge fixed semantic-role support masks into tracker entity IDs."""
    rgb = np.asarray(rgb)
    indexed_masks = np.asarray(indexed_masks)
    if (
        rgb.ndim != 4
        or rgb.shape[-1] != 3
        or rgb.dtype != np.uint8
        or indexed_masks.shape != rgb.shape[:3]
    ):
        raise ObjectGraphContractError(
            "Support must be uint8 RGB [N,H,W,3] and indexed masks [N,H,W]."
        )
    if rgb.shape[0] < 1 or min(rgb.shape[1:3]) < 8:
        raise ObjectGraphContractError("Support dimensions are too small.")
    if indexed_masks.dtype == np.bool_ or not np.issubdtype(
        indexed_masks.dtype, np.integer
    ):
        raise ObjectGraphContractError("Support indexed masks must use an integer dtype.")
    expected_ids = set(range(len(graph.source_roles) + 1))
    observed_ids = set(int(value) for value in np.unique(indexed_masks))
    if not observed_ids.issubset(expected_ids):
        raise ObjectGraphContractError(
            f"Support contains unexpected source IDs: {sorted(observed_ids)!r}."
        )
    role_counts: dict[str, list[int]] = {}
    for role_index, role in enumerate(graph.source_roles, start=1):
        counts = [int((frame == role_index).sum()) for frame in indexed_masks]
        if any(count <= 0 for count in counts):
            raise ObjectGraphContractError(
                f"Source role {role!r} must be non-empty in every support frame."
            )
        role_counts[role] = counts

    entity_masks = np.zeros(indexed_masks.shape, dtype=np.uint8)
    entity_counts: dict[str, list[int]] = {}
    source_index = {role: index + 1 for index, role in enumerate(graph.source_roles)}
    for entity_index, entity in enumerate(graph.entities, start=1):
        selected = np.zeros(indexed_masks.shape, dtype=bool)
        for role in entity.source_roles:
            selected |= indexed_masks == source_index[role]
        if np.any(selected & (entity_masks != 0)):
            raise ObjectGraphContractError("Entity support masks overlap after projection.")
        entity_masks[selected] = entity_index
        counts = [int(frame.sum()) for frame in selected]
        if any(count <= 0 for count in counts):
            raise ObjectGraphContractError(
                f"Entity {entity.name!r} must be non-empty in every support frame."
            )
        entity_counts[entity.name] = counts
    if not np.array_equal(entity_masks > 0, indexed_masks > 0):
        raise ObjectGraphContractError("Entity projection changed support foreground pixels.")
    rgb_copy = np.array(rgb, dtype=np.uint8, order="C", copy=True)
    mask_copy = np.array(entity_masks, dtype=np.uint8, order="C", copy=True)
    metadata = {
        "format": ENTITY_SUPPORT_FORMAT,
        "graph_sha256": graph.graph_sha256,
        "source_roles": list(graph.source_roles),
        "entity_names": list(graph.entity_names),
        "records": int(rgb.shape[0]),
        "resolution": [int(rgb.shape[1]), int(rgb.shape[2])],
        "source_rgb_trace_sha256": _typed_array_sha256(rgb_copy),
        "source_indexed_mask_trace_sha256": _typed_array_sha256(indexed_masks),
        "entity_indexed_mask_trace_sha256": _typed_array_sha256(mask_copy),
        "source_role_pixel_counts": role_counts,
        "entity_pixel_counts": entity_counts,
        "projection": "ordered_union_of_declared_source_role_ids_v1",
    }
    return EntitySupportArrays(
        rgb=rgb_copy,
        indexed_masks=mask_copy,
        source_roles=graph.source_roles,
        entity_names=graph.entity_names,
        metadata=metadata,
    )


def _mask_geometry(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2 or min(mask.shape) < MASK_POOL_SIZE:
        raise ObjectGraphContractError("Role mask must be a sufficiently large 2-D array.")
    height, width = mask.shape
    if height % MASK_POOL_SIZE == 0 and width % MASK_POOL_SIZE == 0:
        occupancy = mask.reshape(
            MASK_POOL_SIZE,
            height // MASK_POOL_SIZE,
            MASK_POOL_SIZE,
            width // MASK_POOL_SIZE,
        ).mean(axis=(1, 3), dtype=np.float32)
    else:
        y_edges = np.linspace(0, height, MASK_POOL_SIZE + 1, dtype=np.int64)
        x_edges = np.linspace(0, width, MASK_POOL_SIZE + 1, dtype=np.int64)
        occupancy = np.empty((MASK_POOL_SIZE, MASK_POOL_SIZE), dtype=np.float32)
        for y_index in range(MASK_POOL_SIZE):
            for x_index in range(MASK_POOL_SIZE):
                cell = mask[
                    y_edges[y_index]:y_edges[y_index + 1],
                    x_edges[x_index]:x_edges[x_index + 1],
                ]
                occupancy[y_index, x_index] = (
                    float(cell.mean()) if cell.size else 0.0
                )
    yx = np.argwhere(mask > 0.5)
    if not len(yx):
        return np.concatenate((occupancy.reshape(-1), np.zeros(10, np.float32)))
    y = yx[:, 0].astype(np.float64) / max(height - 1, 1)
    x = yx[:, 1].astype(np.float64) / max(width - 1, 1)
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    summary = np.asarray(
        [
            cx,
            cy,
            float(mask.mean()),
            float(x.min()),
            float(y.min()),
            float(x.max()),
            float(y.max()),
            float((dx * dx).mean()),
            float((dy * dy).mean()),
            float((dx * dy).mean()),
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(
        np.concatenate((occupancy.reshape(-1), summary)), dtype=np.float32
    )


def _appearance_feature(features: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(features)
    if value.shape != (QUERY_FEATURE_DIM,):
        raise ObjectGraphContractError(
            f"Each entity feature must have shape ({QUERY_FEATURE_DIM},)."
        )
    finite = bool(np.isfinite(value).all())
    safe = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )
    queries = safe.reshape(QUERY_SLOTS, QUERY_DIM)
    pooled = np.concatenate((queries.mean(0), queries.std(0, ddof=0))).astype(
        np.float32
    )
    return np.ascontiguousarray(pooled), finite


def _closest_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_xy = np.argwhere(left)[:, ::-1].astype(np.float64)
    right_xy = np.argwhere(right)[:, ::-1].astype(np.float64)
    if not len(left_xy) or not len(right_xy):
        raise ObjectGraphContractError("Chain support roles cannot be empty.")
    best_distance = float("inf")
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    for start in range(0, len(left_xy), 512):
        block = left_xy[start:start + 512]
        distances = ((block[:, None, :] - right_xy[None, :, :]) ** 2).sum(axis=-1)
        flat = int(np.argmin(distances))
        row, column = np.unravel_index(flat, distances.shape)
        candidate = float(distances[row, column])
        pair = (block[row], right_xy[column])
        key = (candidate, float(pair[0][1]), float(pair[0][0]), float(pair[1][1]), float(pair[1][0]))
        if best_pair is None:
            best_distance, best_pair, best_key = candidate, pair, key
        elif key < best_key:
            best_distance, best_pair, best_key = candidate, pair, key
    assert best_pair is not None and math.isfinite(best_distance)
    return best_pair


def _cap_farthest_from(mask: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    xy = np.argwhere(mask)[:, ::-1].astype(np.float64)
    if not len(xy):
        raise ObjectGraphContractError("Cannot find an endpoint in an empty mask.")
    distance = np.linalg.norm(xy - origin_xy, axis=1)
    threshold = max(float(distance.max()) - 1.5, float(np.quantile(distance, 0.95)))
    cap = xy[distance >= threshold]
    return cap.mean(axis=0)


@dataclass(frozen=True)
class ChainCalibration:
    entity_name: str
    roles: tuple[str, ...]
    base_xy: tuple[float, float]
    link_fractions: tuple[float, ...]
    expected_total_length_px: float
    support_total_lengths_px: tuple[float, ...]
    expected_entity_pixels: float
    support_entity_pixels: tuple[int, ...]
    support_joint_trace_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "roles": list(self.roles),
            "base_xy": list(self.base_xy),
            "link_fractions": list(self.link_fractions),
            "expected_total_length_px": self.expected_total_length_px,
            "support_total_lengths_px": list(self.support_total_lengths_px),
            "expected_entity_pixels": self.expected_entity_pixels,
            "support_entity_pixels": list(self.support_entity_pixels),
            "support_joint_trace_sha256": self.support_joint_trace_sha256,
            "source": "fixed_indexed_support_masks_only_v1",
        }


def _calibrate_chain(
    graph: CompiledObjectGraph,
    entity: EntitySpec,
    source_indexed_masks: np.ndarray,
) -> ChainCalibration:
    role_index = {role: index + 1 for index, role in enumerate(graph.source_roles)}
    frame_points: list[np.ndarray] = []
    frame_lengths: list[np.ndarray] = []
    for indexed in source_indexed_masks:
        masks = [indexed == role_index[role] for role in entity.projector.roles]
        joints: list[np.ndarray] = []
        for left, right in zip(masks[:-1], masks[1:]):
            left_point, right_point = _closest_pair(left, right)
            joints.append((left_point + right_point) * 0.5)
        base = _cap_farthest_from(masks[0], joints[0])
        tip = _cap_farthest_from(masks[-1], joints[-1])
        points = np.vstack((base, *joints, tip)).astype(np.float64)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if (
            len(lengths) != len(entity.projector.roles)
            or not np.isfinite(points).all()
            or not np.isfinite(lengths).all()
            or np.any(lengths <= 1.0)
        ):
            raise ObjectGraphContractError(
                f"Support cannot calibrate chain entity {entity.name!r}."
            )
        frame_points.append(points)
        frame_lengths.append(lengths)
    point_array = np.stack(frame_points)
    length_array = np.stack(frame_lengths)
    entity_role_ids = np.asarray(
        [role_index[role] for role in entity.source_roles], dtype=source_indexed_masks.dtype
    )
    support_entity_pixels = tuple(
        int(np.isin(indexed, entity_role_ids).sum())
        for indexed in source_indexed_masks
    )
    if any(value <= 0 for value in support_entity_pixels):
        raise ObjectGraphContractError("Calibrated chain entity support is empty.")
    base_xy = np.median(point_array[:, 0], axis=0)
    median_lengths = np.median(length_array, axis=0)
    total = float(median_lengths.sum())
    fractions = tuple(float(value / total) for value in median_lengths)
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ObjectGraphContractError("Calibrated chain fractions do not sum to one.")
    return ChainCalibration(
        entity_name=entity.name,
        roles=entity.projector.roles,
        base_xy=(float(base_xy[0]), float(base_xy[1])),
        link_fractions=fractions,
        expected_total_length_px=total,
        support_total_lengths_px=tuple(float(row.sum()) for row in length_array),
        expected_entity_pixels=float(np.median(support_entity_pixels)),
        support_entity_pixels=support_entity_pixels,
        support_joint_trace_sha256=_typed_array_sha256(point_array),
    )


@dataclass(frozen=True)
class TemporalChainCalibration:
    """Support-only robust scales used by the causal chain projector.

    The labelled masks are fixed support data.  No episode observation is added
    to this calibration and none of these statistics is updated online.
    """

    entity_name: str
    roles: tuple[str, ...]
    support_poses_xy: np.ndarray
    root_xy: tuple[float, float]
    root_scale_px: float
    link_lengths_px: tuple[float, ...]
    link_scales_px: tuple[float, ...]
    role_area_fractions: tuple[float, ...]
    role_area_scales: tuple[float, ...]
    expected_total_length_px: float
    expected_entity_pixels: float
    entity_pixel_scale: float
    half_width_px: float
    support_pose_trace_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "roles": list(self.roles),
            "root_xy": list(self.root_xy),
            "root_scale_px": self.root_scale_px,
            "link_lengths_px": list(self.link_lengths_px),
            "link_scales_px": list(self.link_scales_px),
            "role_area_fractions": list(self.role_area_fractions),
            "role_area_scales": list(self.role_area_scales),
            "expected_total_length_px": self.expected_total_length_px,
            "expected_entity_pixels": self.expected_entity_pixels,
            "entity_pixel_scale": self.entity_pixel_scale,
            "half_width_px": self.half_width_px,
            "support_pose_trace_sha256": self.support_pose_trace_sha256,
            "source": "fixed_indexed_support_role_masks_robust_statistics_v2",
        }


def _robust_scale(values: np.ndarray, *, floor: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, float(floor))


def _calibrate_temporal_chain(
    graph: CompiledObjectGraph,
    entity: EntitySpec,
    source_indexed_masks: np.ndarray,
) -> TemporalChainCalibration:
    role_index = {role: index + 1 for index, role in enumerate(graph.source_roles)}
    poses: list[np.ndarray] = []
    role_pixels: list[np.ndarray] = []
    entity_pixels: list[int] = []
    for indexed in source_indexed_masks:
        masks = [indexed == role_index[role] for role in entity.projector.roles]
        joints: list[np.ndarray] = []
        for left, right in zip(masks[:-1], masks[1:]):
            left_point, right_point = _closest_pair(left, right)
            joints.append((left_point + right_point) * 0.5)
        base = _cap_farthest_from(masks[0], joints[0])
        tip = _cap_farthest_from(masks[-1], joints[-1])
        pose = np.vstack((base, *joints, tip)).astype(np.float64)
        lengths = np.linalg.norm(np.diff(pose, axis=0), axis=1)
        if (
            len(lengths) != len(entity.projector.roles)
            or not np.isfinite(pose).all()
            or np.any(lengths <= 1.0)
        ):
            raise ObjectGraphContractError(
                f"Support cannot calibrate temporal chain entity {entity.name!r}."
            )
        counts = np.asarray([int(mask.sum()) for mask in masks], dtype=np.float64)
        poses.append(pose)
        role_pixels.append(counts)
        entity_pixels.append(int(counts.sum()))

    pose_array = np.ascontiguousarray(np.stack(poses), dtype=np.float64)
    pose_array.setflags(write=False)
    length_array = np.linalg.norm(np.diff(pose_array, axis=1), axis=2)
    role_pixel_array = np.stack(role_pixels)
    entity_pixel_array = np.asarray(entity_pixels, dtype=np.float64)
    area_fractions = role_pixel_array / entity_pixel_array[:, None]
    median_lengths = np.median(length_array, axis=0)
    median_area_fractions = np.median(area_fractions, axis=0)
    total_length = float(median_lengths.sum())
    expected_pixels = float(np.median(entity_pixel_array))
    # The union of thick line segments has area approximately 2*r*length.
    # This gives a resolution-scaled tolerance without a task-specific constant.
    half_width = max(expected_pixels / max(2.0 * total_length, 1.0), 1.0)
    root_values = pose_array[:, 0]
    root_median = np.median(root_values, axis=0)
    root_radius = np.linalg.norm(root_values - root_median, axis=1)
    root_scale = _robust_scale(root_radius, floor=half_width)
    link_scales = tuple(
        _robust_scale(length_array[:, index], floor=max(1.0, half_width * 0.5))
        for index in range(length_array.shape[1])
    )
    fraction_floor = max(1.0 / max(math.sqrt(expected_pixels), 1.0), 0.01)
    area_scales = tuple(
        _robust_scale(area_fractions[:, index], floor=fraction_floor)
        for index in range(area_fractions.shape[1])
    )
    return TemporalChainCalibration(
        entity_name=entity.name,
        roles=entity.projector.roles,
        support_poses_xy=pose_array,
        root_xy=(float(root_median[0]), float(root_median[1])),
        root_scale_px=root_scale,
        link_lengths_px=tuple(float(value) for value in median_lengths),
        link_scales_px=link_scales,
        role_area_fractions=tuple(float(value) for value in median_area_fractions),
        role_area_scales=area_scales,
        expected_total_length_px=total_length,
        expected_entity_pixels=expected_pixels,
        entity_pixel_scale=_robust_scale(
            entity_pixel_array, floor=max(math.sqrt(expected_pixels), 1.0)
        ),
        half_width_px=half_width,
        support_pose_trace_sha256=_typed_array_sha256(pose_array),
    )


@dataclass
class _TemporalChainState:
    """Mutable causal state owned by one tokenizer episode and one entity."""

    last_pose_xy: np.ndarray | None = None
    previous_pose_xy: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    uncertainty_px: float = 0.0
    invalid_age: int = 0
    branch_signature: tuple[int, ...] | None = None
    accepted_frames: int = 0

    @property
    def initialized(self) -> bool:
        return self.last_pose_xy is not None


@dataclass(frozen=True)
class _Component:
    mask: np.ndarray
    size: int
    minimum_base_distance_px: float
    first_yx: tuple[int, int]


def _components(mask: np.ndarray, base_xy: np.ndarray) -> list[_Component]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    result: list[_Component] = []
    for y_value, x_value in np.argwhere(mask):
        start = (int(y_value), int(x_value))
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidate = (y + dy, x + dx)
                    if (
                        (dx or dy)
                        and 0 <= candidate[0] < height
                        and 0 <= candidate[1] < width
                        and mask[candidate]
                        and not visited[candidate]
                    ):
                        visited[candidate] = True
                        stack.append(candidate)
        component = np.zeros_like(mask)
        yx = np.asarray(pixels, dtype=np.int64)
        component[yx[:, 0], yx[:, 1]] = True
        xy = yx[:, ::-1].astype(np.float64)
        result.append(
            _Component(
                component,
                len(pixels),
                float(np.linalg.norm(xy - base_xy, axis=1).min()),
                min(pixels),
            )
        )
    return result


def _binary_close(mask: np.ndarray) -> np.ndarray:
    """A deterministic 3x3 closing used only for centreline extraction."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    dilated = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    padded = np.pad(dilated, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return eroded


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Vectorized deterministic Zhang-Suen thinning."""
    image = np.asarray(mask, dtype=bool).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1, constant_values=False)
            height, width = image.shape
            p2 = padded[0:height, 1:width + 1]
            p3 = padded[0:height, 2:width + 2]
            p4 = padded[1:height + 1, 2:width + 2]
            p5 = padded[2:height + 2, 2:width + 2]
            p6 = padded[2:height + 2, 1:width + 1]
            p7 = padded[2:height + 2, 0:width]
            p8 = padded[1:height + 1, 0:width]
            p9 = padded[0:height, 0:width]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = np.stack(neighbors, axis=0).sum(axis=0, dtype=np.uint8)
            transitions = np.stack(
                [
                    (~neighbors[index]) & neighbors[(index + 1) % 8]
                    for index in range(8)
                ],
                axis=0,
            ).sum(axis=0, dtype=np.uint8)
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                remove &= ~(p2 & p4 & p6)
                remove &= ~(p4 & p6 & p8)
            else:
                remove &= ~(p2 & p4 & p8)
                remove &= ~(p2 & p6 & p8)
            if bool(remove.any()):
                changed = True
                image[remove] = False
    return image


Node = tuple[int, int]


def _skeleton_graph(mask: np.ndarray) -> dict[Node, tuple[tuple[Node, float], ...]]:
    nodes = {tuple(int(item) for item in yx) for yx in np.argwhere(mask)}
    result: dict[Node, tuple[tuple[Node, float], ...]] = {}
    for y, x in sorted(nodes):
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate = (y + dy, x + dx)
                if (dx or dy) and candidate in nodes:
                    neighbors.append((candidate, math.hypot(dx, dy)))
        result[(y, x)] = tuple(sorted(neighbors))
    return result


def _dijkstra(
    graph: Mapping[Node, Sequence[tuple[Node, float]]], start: Node
) -> tuple[dict[Node, float], dict[Node, Node]]:
    distance = {node: float("inf") for node in graph}
    previous: dict[Node, Node] = {}
    distance[start] = 0.0
    queue: list[tuple[float, Node]] = [(0.0, start)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distance[node] + 1e-12:
            continue
        for neighbor, weight in graph[node]:
            candidate = current_distance + weight
            if candidate < distance[neighbor] - 1e-12:
                distance[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
            elif math.isclose(candidate, distance[neighbor], abs_tol=1e-12):
                old = previous.get(neighbor)
                if old is None or node < old:
                    previous[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
    return distance, previous


def _path(previous: Mapping[Node, Node], start: Node, end: Node) -> list[Node]:
    result = [end]
    while result[-1] != start:
        if result[-1] not in previous:
            return []
        result.append(previous[result[-1]])
    result.reverse()
    return result


def _sample_polyline(points: np.ndarray, fraction: float) -> np.ndarray:
    if len(points) < 2:
        raise ObjectGraphContractError("Cannot sample a degenerate centreline.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(np.clip(fraction, 0.0, 1.0)) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
    alpha = (target - cumulative[index]) / max(float(lengths[index]), 1e-12)
    return points[index] + alpha * (points[index + 1] - points[index])


def _segment_distance_squared(
    xx: np.ndarray, yy: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return (xx - start[0]) ** 2 + (yy - start[1]) ** 2
    position = np.clip(
        ((xx - start[0]) * dx + (yy - start[1]) * dy) / denominator,
        0.0,
        1.0,
    )
    return (xx - (start[0] + position * dx)) ** 2 + (
        yy - (start[1] + position * dy)
    ) ** 2


@dataclass(frozen=True)
class _Projection:
    masks: np.ndarray
    keypoints: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    diagnostics: dict[str, Any]


def _direct_projection(mask: np.ndarray) -> _Projection:
    mask = np.ascontiguousarray(mask, dtype=bool)
    yx = np.argwhere(mask)
    valid = bool(len(yx))
    if valid:
        xy = yx[:, ::-1].astype(np.float64)
        centroid = xy.mean(axis=0)
        keypoints = np.stack((centroid, centroid))[None]
    else:
        keypoints = np.zeros((1, 2, 2), dtype=np.float64)
    return _Projection(
        masks=mask[None],
        keypoints=keypoints,
        valid=np.asarray([valid], dtype=bool),
        confidence=np.asarray([1.0 if valid else 0.0], dtype=np.float32),
        diagnostics={"projector": "direct_role_v1", "failure_reasons": [] if valid else ["entity_empty"]},
    )


def _chain_projection(
    mask: np.ndarray,
    spec: ProjectorSpec,
    calibration: ChainCalibration,
) -> _Projection:
    mask = np.ascontiguousarray(mask, dtype=bool)
    failure: list[str] = []
    pixel_count = int(mask.sum())
    role_count = len(spec.roles)
    keypoints = np.zeros((role_count, 2, 2), dtype=np.float64)
    role_masks = np.zeros((role_count, *mask.shape), dtype=bool)
    if pixel_count < spec.minimum_entity_pixels:
        failure.append("entity_too_small")
    maximum_entity_pixels = (
        spec.maximum_entity_area_multiple * calibration.expected_entity_pixels
    )
    if pixel_count > maximum_entity_pixels:
        failure.append("entity_area_implausible")
    if failure:
        return _Projection(
            masks=role_masks,
            keypoints=keypoints,
            valid=np.zeros(role_count, dtype=bool),
            confidence=np.zeros(role_count, dtype=np.float32),
            diagnostics={
                "projector": "ordered_chain_segments_v1",
                "failure_reasons": failure,
                "entity_pixels": pixel_count,
                "expected_entity_pixels": calibration.expected_entity_pixels,
                "maximum_entity_pixels": maximum_entity_pixels,
                "early_rejection": True,
            },
        )
    base = np.asarray(calibration.base_xy, dtype=np.float64)
    components = _components(mask, base)
    selected = (
        min(
            components,
            key=lambda item: (
                item.minimum_base_distance_px,
                -item.size,
                item.first_yx,
            ),
        )
        if components
        else None
    )
    selected_mask = selected.mask if selected is not None else np.zeros_like(mask)
    disconnected_fraction = (
        1.0 - float(selected.size) / max(pixel_count, 1)
        if selected is not None
        else 1.0
    )
    if selected is None:
        failure.append("entity_empty")
    else:
        if (
            selected.minimum_base_distance_px
            > spec.maximum_base_distance_fraction * calibration.expected_total_length_px
        ):
            failure.append("entity_not_support_base_attached")
        if disconnected_fraction > spec.maximum_disconnected_fraction:
            failure.append("entity_disconnected")
    centreline_mask = _binary_close(selected_mask)
    skeleton = _zhang_suen(centreline_mask)
    graph = _skeleton_graph(skeleton)
    path_nodes: list[Node] = []
    branch_fraction = 1.0
    if graph:
        start = min(
            graph,
            key=lambda node: (
                float(np.linalg.norm(np.asarray([node[1], node[0]]) - base)),
                node,
            ),
        )
        distance, previous = _dijkstra(graph, start)
        reachable = [node for node, value in distance.items() if math.isfinite(value)]
        end = max(reachable, key=lambda node: (distance[node], node))
        path_nodes = _path(previous, start, end)
        branch_fraction = 1.0 - float(len(path_nodes)) / max(len(graph), 1)
    if len(path_nodes) < 2:
        failure.append("centreline_unavailable")
        path_xy = np.empty((0, 2), dtype=np.float64)
        path_length = 0.0
    else:
        path_xy = np.asarray([[node[1], node[0]] for node in path_nodes], np.float64)
        if float(np.linalg.norm(path_xy[0] - base)) > 1e-12:
            path_xy = np.vstack((base, path_xy))
        path_length = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())
        ratio = path_length / calibration.expected_total_length_px
        if ratio < spec.minimum_path_length_fraction:
            failure.append("centreline_too_short")
        if ratio > spec.maximum_path_length_fraction:
            failure.append("centreline_too_long")
        if branch_fraction > spec.maximum_branch_fraction:
            failure.append("centreline_excessively_branched")
    valid_geometry = not failure
    if valid_geometry:
        boundaries = [0.0]
        total = 0.0
        for fraction in calibration.link_fractions:
            total += fraction
            boundaries.append(total)
        points = np.stack([_sample_polyline(path_xy, value) for value in boundaries])
        keypoints = np.stack((points[:-1], points[1:]), axis=1)
        yy, xx = np.indices(mask.shape, dtype=np.float64)
        distances = np.stack(
            [
                _segment_distance_squared(xx, yy, segment[0], segment[1])
                for segment in keypoints
            ]
        )
        assignments = np.argmin(distances, axis=0)
        for role_index in range(role_count):
            # The selected component supplies the centreline, but every accepted
            # entity pixel is assigned to exactly one semantic role.  This keeps
            # the projection information-preserving whenever it is valid.
            role_masks[role_index] = mask & (assignments == role_index)
        if any(not role_mask.any() for role_mask in role_masks):
            failure.append("projected_role_empty")
            valid_geometry = False
            role_masks.fill(False)
            keypoints.fill(0.0)
    length_score = math.exp(
        -abs(path_length - calibration.expected_total_length_px)
        / max(0.2 * calibration.expected_total_length_px, 1.0)
    )
    confidence_value = (
        float(np.clip(length_score * (1.0 - disconnected_fraction) * (1.0 - branch_fraction), 0.0, 1.0))
        if valid_geometry
        else 0.0
    )
    if valid_geometry and confidence_value < spec.minimum_projection_confidence:
        failure.append("projection_confidence_too_low")
        valid_geometry = False
        confidence_value = 0.0
        role_masks.fill(False)
        keypoints.fill(0.0)
    return _Projection(
        masks=role_masks,
        keypoints=keypoints,
        valid=np.full(role_count, valid_geometry, dtype=bool),
        confidence=np.full(role_count, confidence_value, dtype=np.float32),
        diagnostics={
            "projector": "ordered_chain_segments_v1",
            "failure_reasons": failure,
            "entity_pixels": pixel_count,
            "expected_entity_pixels": calibration.expected_entity_pixels,
            "maximum_entity_pixels": maximum_entity_pixels,
            "components": len(components),
            "selected_pixels": 0 if selected is None else selected.size,
            "disconnected_fraction": disconnected_fraction,
            "minimum_base_distance_px": None if selected is None else selected.minimum_base_distance_px,
            "skeleton_pixels": len(graph),
            "path_pixels": len(path_nodes),
            "path_length_px": path_length,
            "expected_total_length_px": calibration.expected_total_length_px,
            "branch_fraction": branch_fraction,
            "minimum_projection_confidence": spec.minimum_projection_confidence,
            "early_rejection": False,
        },
    )


@dataclass(frozen=True)
class _TemporalCandidate:
    pose_xy: np.ndarray
    assignments: np.ndarray
    total_cost: float
    length_cost: float
    pixel_fit_cost: float
    segment_fit_cost: float
    area_cost: float
    root_cost: float
    temporal_cost: float
    switch_cost: float
    coverage: float
    path_length_px: float
    branch_fraction: float
    disconnected_fraction: float
    signature: tuple[int, ...]
    source: str


def _turn_signature(pose_xy: np.ndarray) -> tuple[int, ...]:
    segments = np.diff(np.asarray(pose_xy, dtype=np.float64), axis=0)
    if len(segments) < 2:
        return ()
    result: list[int] = []
    for left, right in zip(segments[:-1], segments[1:]):
        scale = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
        sine = float((left[0] * right[1] - left[1] * right[0]) / scale)
        result.append(0 if abs(sine) < 0.12 else (1 if sine > 0.0 else -1))
    return tuple(result)


def _predicted_pose(
    state: _TemporalChainState,
) -> np.ndarray | None:
    if state.last_pose_xy is None:
        return None
    if state.velocity_xy is None:
        return np.array(state.last_pose_xy, dtype=np.float64, copy=True)
    # A missing observation increases uncertainty and damps extrapolation.  The
    # prediction remains a candidate prior; it is never emitted without support
    # from the current mask.
    damping = 1.0 / (1.0 + 0.35 * float(state.invalid_age))
    return np.asarray(state.last_pose_xy) + damping * np.asarray(state.velocity_xy)


def _record_temporal_miss(
    state: _TemporalChainState,
    calibration: TemporalChainCalibration,
) -> None:
    state.invalid_age += 1
    if state.last_pose_xy is None:
        return
    state.uncertainty_px = min(
        calibration.expected_total_length_px,
        max(state.uncertainty_px, calibration.half_width_px)
        + calibration.half_width_px * (0.5 + 0.1 * min(state.invalid_age, 5)),
    )
    if state.velocity_xy is not None:
        state.velocity_xy = np.ascontiguousarray(
            np.asarray(state.velocity_xy) * 0.8, dtype=np.float64
        )


def _record_temporal_accept(
    state: _TemporalChainState,
    pose_xy: np.ndarray,
    calibration: TemporalChainCalibration,
) -> None:
    pose = np.ascontiguousarray(pose_xy, dtype=np.float64)
    old_last = state.last_pose_xy
    predicted = _predicted_pose(state)
    if old_last is None:
        velocity = np.zeros_like(pose)
        residual = calibration.half_width_px
    else:
        observed_velocity = pose - old_last
        if state.velocity_xy is None:
            velocity = observed_velocity
        else:
            velocity = 0.55 * np.asarray(state.velocity_xy) + 0.45 * observed_velocity
        residual = float(
            np.linalg.norm(pose - predicted, axis=1).mean()
            if predicted is not None
            else np.linalg.norm(observed_velocity, axis=1).mean()
        )
    previous_uncertainty = max(state.uncertainty_px, calibration.half_width_px)
    state.previous_pose_xy = (
        None if old_last is None else np.ascontiguousarray(old_last, dtype=np.float64)
    )
    state.last_pose_xy = pose
    state.velocity_xy = np.ascontiguousarray(velocity, dtype=np.float64)
    state.uncertainty_px = max(
        calibration.half_width_px,
        0.7 * previous_uncertainty + 0.3 * residual,
    )
    state.invalid_age = 0
    state.branch_signature = _turn_signature(pose)
    state.accepted_frames += 1


def _score_temporal_pose(
    pose_xy: np.ndarray,
    mask_xy: np.ndarray,
    calibration: TemporalChainCalibration,
    state: _TemporalChainState,
    predicted_pose: np.ndarray | None,
    *,
    branch_fraction: float,
    disconnected_fraction: float,
    source: str,
) -> _TemporalCandidate | None:
    pose = np.ascontiguousarray(pose_xy, dtype=np.float64)
    role_count = len(calibration.roles)
    if pose.shape != (role_count + 1, 2) or not np.isfinite(pose).all():
        return None
    segments = np.stack((pose[:-1], pose[1:]), axis=1)
    lengths = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    if np.any(lengths <= 1.0):
        return None

    point_distances = np.stack(
        [
            _segment_distance_squared(
                mask_xy[:, 0], mask_xy[:, 1], segment[0], segment[1]
            )
            for segment in segments
        ]
    )
    assignments = np.argmin(point_distances, axis=0).astype(np.int16, copy=False)
    counts = np.bincount(assignments, minlength=role_count).astype(np.float64)
    if np.any(counts <= 0.0):
        return None
    fractions = counts / max(float(len(mask_xy)), 1.0)
    minimum_distance = np.sqrt(np.min(point_distances, axis=0))
    pixel_fit = float(
        np.sqrt(np.mean(minimum_distance * minimum_distance))
        / calibration.half_width_px
    )

    samples: list[np.ndarray] = []
    for segment in segments:
        alpha = np.linspace(0.0, 1.0, 12, dtype=np.float64)[:, None]
        samples.append(segment[0] + alpha * (segment[1] - segment[0]))
    sample_xy = np.concatenate(samples, axis=0)
    sample_distance_squared = (
        (sample_xy[:, None, :] - mask_xy[None, :, :]) ** 2
    ).sum(axis=2)
    sample_distance = np.sqrt(np.min(sample_distance_squared, axis=1))
    segment_fit = float(np.sqrt(np.mean(sample_distance ** 2)) / calibration.half_width_px)
    coverage = float(np.mean(sample_distance <= 2.25 * calibration.half_width_px))

    expected_lengths = np.asarray(calibration.link_lengths_px)
    length_scales = np.asarray(calibration.link_scales_px)
    length_cost = float(np.mean(np.abs(lengths - expected_lengths) / length_scales))
    expected_fractions = np.asarray(calibration.role_area_fractions)
    area_scales = np.asarray(calibration.role_area_scales)
    area_cost = float(np.mean(np.abs(fractions - expected_fractions) / area_scales))

    root_reference = (
        predicted_pose[0]
        if predicted_pose is not None
        else np.asarray(calibration.root_xy, dtype=np.float64)
    )
    root_scale = (
        max(state.uncertainty_px, calibration.root_scale_px)
        if predicted_pose is not None
        else calibration.root_scale_px
    )
    root_cost = float(np.linalg.norm(pose[0] - root_reference) / root_scale)
    if predicted_pose is None:
        temporal_cost = 0.0
    else:
        temporal_scale = max(state.uncertainty_px, 1.5 * calibration.half_width_px)
        temporal_cost = float(
            np.linalg.norm(pose - predicted_pose, axis=1).mean() / temporal_scale
        )

    signature = _turn_signature(pose)
    switch_cost = 0.0
    if state.branch_signature is not None and signature:
        compared = [
            left != 0 and right != 0 and left != right
            for left, right in zip(state.branch_signature, signature)
        ]
        if any(compared):
            switch_cost = float(sum(compared)) / max(len(compared), 1)
            switch_cost /= 1.0 + state.uncertainty_px / max(
                calibration.half_width_px, 1.0
            )

    # All terms are dimensionless and support-normalized.  These weights are
    # protocol constants shared by every graph; there is no task dispatch.
    total_cost = (
        0.55 * length_cost
        + 0.70 * pixel_fit
        + 0.70 * segment_fit
        + 0.25 * area_cost
        + 0.25 * root_cost
        + (0.90 * temporal_cost if predicted_pose is not None else 0.0)
        + 1.50 * (1.0 - coverage)
        + 0.60 * max(branch_fraction, 0.0)
        + 1.50 * max(disconnected_fraction, 0.0)
        + 0.80 * switch_cost
    )
    return _TemporalCandidate(
        pose_xy=pose,
        assignments=np.ascontiguousarray(assignments),
        total_cost=float(total_cost),
        length_cost=length_cost,
        pixel_fit_cost=pixel_fit,
        segment_fit_cost=segment_fit,
        area_cost=area_cost,
        root_cost=root_cost,
        temporal_cost=temporal_cost,
        switch_cost=switch_cost,
        coverage=coverage,
        path_length_px=float(lengths.sum()),
        branch_fraction=float(branch_fraction),
        disconnected_fraction=float(disconnected_fraction),
        signature=signature,
        source=source,
    )


def _deduplicate_temporal_candidates(
    candidates: Sequence[_TemporalCandidate],
) -> list[_TemporalCandidate]:
    result: list[_TemporalCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.total_cost,
            item.source,
            tuple(float(value) for value in item.pose_xy.reshape(-1)),
        ),
    ):
        duplicate = any(
            np.array_equal(candidate.assignments, previous.assignments)
            and float(
                np.linalg.norm(candidate.pose_xy - previous.pose_xy, axis=1).mean()
            ) < 0.75
            for previous in result
        )
        if not duplicate:
            result.append(candidate)
        if len(result) >= TEMPORAL_MAX_RETAINED_CANDIDATES:
            break
    return result


def _empty_temporal_projection(
    role_count: int,
    resolution: tuple[int, int],
    state: _TemporalChainState,
    failure: Sequence[str],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> _Projection:
    payload: dict[str, Any] = {
        "projector": "ordered_chain_temporal_v2",
        "failure_reasons": list(failure),
        "candidate_count": 0,
        "state_initialized": state.initialized,
        "invalid_age": state.invalid_age,
        "accepted_frames": state.accepted_frames,
        "fail_closed": True,
    }
    if diagnostics:
        payload.update(diagnostics)
    return _Projection(
        masks=np.zeros((role_count, *resolution), dtype=bool),
        keypoints=np.zeros((role_count, 2, 2), dtype=np.float64),
        valid=np.zeros(role_count, dtype=bool),
        confidence=np.zeros(role_count, dtype=np.float32),
        diagnostics=payload,
    )


def _temporal_chain_projection(
    mask: np.ndarray,
    spec: ProjectorSpec,
    calibration: TemporalChainCalibration,
    state: _TemporalChainState,
    *,
    entity_available: bool,
) -> _Projection:
    mask = np.ascontiguousarray(mask, dtype=bool)
    role_count = len(spec.roles)
    resolution = tuple(int(value) for value in mask.shape)
    initialized_before = state.initialized
    invalid_age_before = state.invalid_age
    pixel_count = int(mask.sum())
    maximum_entity_pixels = (
        spec.maximum_entity_area_multiple * calibration.expected_entity_pixels
    )
    failure: list[str] = []
    if not entity_available:
        failure.append("entity_unavailable")
    if pixel_count < spec.minimum_entity_pixels:
        failure.append("entity_too_small")
    if pixel_count > maximum_entity_pixels:
        failure.append("entity_area_implausible")
    if failure:
        _record_temporal_miss(state, calibration)
        return _empty_temporal_projection(
            role_count,
            resolution,
            state,
            failure,
            diagnostics={
                "entity_pixels": pixel_count,
                "expected_entity_pixels": calibration.expected_entity_pixels,
                "maximum_entity_pixels": maximum_entity_pixels,
                "state_initialized_before": initialized_before,
                "invalid_age_before": invalid_age_before,
                "early_rejection": True,
            },
        )

    predicted = _predicted_pose(state)
    base_reference = (
        predicted[0]
        if predicted is not None
        else np.asarray(calibration.root_xy, dtype=np.float64)
    )
    components = sorted(
        _components(mask, base_reference),
        key=lambda item: (
            item.minimum_base_distance_px,
            -item.size,
            item.first_yx,
        ),
    )
    mask_xy = np.argwhere(mask)[:, ::-1].astype(np.float64)
    candidates: list[_TemporalCandidate] = []
    scored_candidate_count = 0
    topology_records: list[dict[str, Any]] = []
    causal_branch_fractions: list[float] = []
    for component_index, component in enumerate(components[:3]):
        if scored_candidate_count >= TEMPORAL_MAX_SCORED_CANDIDATES:
            break
        disconnected_fraction = 1.0 - float(component.size) / max(pixel_count, 1)
        if (
            component.minimum_base_distance_px
            > spec.maximum_base_distance_fraction * calibration.expected_total_length_px
            or disconnected_fraction > spec.maximum_disconnected_fraction
        ):
            topology_records.append(
                {
                    "component": component_index,
                    "skipped": True,
                    "minimum_base_distance_px": component.minimum_base_distance_px,
                    "disconnected_fraction": disconnected_fraction,
                }
            )
            continue
        variants = [("raw", component.mask)]
        closed = _binary_close(component.mask)
        if not np.array_equal(closed, component.mask):
            variants.append(("closed", closed))
        for variant_name, centreline_mask in variants:
            if scored_candidate_count >= TEMPORAL_MAX_SCORED_CANDIDATES:
                break
            graph = _skeleton_graph(_zhang_suen(centreline_mask))
            if len(graph) < 2:
                continue
            nodes = sorted(graph)
            endpoint_nodes = [node for node in nodes if len(graph[node]) <= 1]
            root_references = [base_reference]
            support_root = np.asarray(calibration.root_xy, dtype=np.float64)
            if float(np.linalg.norm(base_reference - support_root)) > 0.5:
                root_references.append(support_root)
            root_nodes: list[Node] = []
            for reference in root_references:
                ranked = sorted(
                    nodes,
                    key=lambda node: (
                        float(
                            np.linalg.norm(
                                np.asarray([node[1], node[0]], dtype=np.float64)
                                - reference
                            )
                        ),
                        node,
                    ),
                )
                root_nodes.extend(ranked[:2])
            root_nodes = list(dict.fromkeys(root_nodes))[:4]
            variant_candidates: list[_TemporalCandidate] = []
            variant_branch_fractions: list[float] = []
            maximum_branch = 0.0
            for root in root_nodes:
                if scored_candidate_count >= TEMPORAL_MAX_SCORED_CANDIDATES:
                    break
                distance, previous = _dijkstra(graph, root)
                reachable = [
                    node for node, value in distance.items() if math.isfinite(value)
                ]
                preferred_ends = [node for node in endpoint_nodes if node != root]
                if not preferred_ends:
                    preferred_ends = [node for node in reachable if node != root]
                ends = sorted(
                    preferred_ends,
                    key=lambda node: (-distance[node], node),
                )[:6]
                for end in ends:
                    if scored_candidate_count >= TEMPORAL_MAX_SCORED_CANDIDATES:
                        break
                    path_nodes = _path(previous, root, end)
                    if len(path_nodes) < 2:
                        continue
                    path_xy = np.asarray(
                        [[node[1], node[0]] for node in path_nodes], dtype=np.float64
                    )
                    path_length = float(
                        np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
                    )
                    ratio = path_length / calibration.expected_total_length_px
                    if (
                        ratio < spec.minimum_path_length_fraction
                        or ratio > spec.maximum_path_length_fraction
                    ):
                        continue
                    branch_fraction = 1.0 - float(len(path_nodes)) / max(len(graph), 1)
                    maximum_branch = max(maximum_branch, branch_fraction)
                    if branch_fraction > spec.maximum_branch_fraction:
                        continue
                    cumulative = np.cumsum(
                        np.asarray(calibration.link_lengths_px, dtype=np.float64)
                    )
                    fractions = np.concatenate(
                        ([0.0], cumulative / max(float(cumulative[-1]), 1e-12))
                    )
                    pose = np.stack(
                        [_sample_polyline(path_xy, value) for value in fractions]
                    )
                    candidate = _score_temporal_pose(
                        pose,
                        mask_xy,
                        calibration,
                        state,
                        predicted,
                        branch_fraction=branch_fraction,
                        disconnected_fraction=disconnected_fraction,
                        source=f"skeleton_{variant_name}",
                    )
                    scored_candidate_count += 1
                    if candidate is not None:
                        variant_candidates.append(candidate)
                        variant_branch_fractions.append(branch_fraction)
            # ``maximum_branch_fraction`` is a topology maximum, not merely a
            # soft candidate score.  Reject the complete skeleton variant when
            # any plausible root-to-end path exposes too much off-path mass;
            # otherwise a temporal prediction could reinterpret a large spur as
            # a link and bypass the current-frame topology contract.
            excessive_branching = maximum_branch > spec.maximum_branch_fraction
            if not excessive_branching:
                candidates.extend(variant_candidates)
                if component_index == 0:
                    causal_branch_fractions.extend(variant_branch_fractions)
            generated = 0 if excessive_branching else len(variant_candidates)
            topology_records.append(
                {
                    "component": component_index,
                    "skipped": False,
                    "variant": variant_name,
                    "skeleton_pixels": len(graph),
                    "endpoints": len(endpoint_nodes),
                    "root_candidates": len(root_nodes),
                    "generated_candidates": generated,
                    "maximum_branch_fraction": maximum_branch,
                    "excessive_branching": excessive_branching,
                    "minimum_base_distance_px": component.minimum_base_distance_px,
                    "disconnected_fraction": disconnected_fraction,
                }
            )

    # A causal continuation is scored against the current pixels just like any
    # skeleton proposal.  It bridges unstable skeleton topology, not missing
    # observations: poor current-mask support rejects it below.
    if predicted is not None and causal_branch_fractions and components:
        best_disconnected = (
            1.0 - float(components[0].size) / max(pixel_count, 1)
            if components
            else 1.0
        )
        if (
            components[0].minimum_base_distance_px
            <= spec.maximum_base_distance_fraction
            * calibration.expected_total_length_px
            and best_disconnected <= spec.maximum_disconnected_fraction
        ):
            predicted_candidate = _score_temporal_pose(
                predicted,
                mask_xy,
                calibration,
                state,
                predicted,
                branch_fraction=min(causal_branch_fractions),
                disconnected_fraction=best_disconnected,
                source="causal_prediction_current_mask_fit",
            )
            scored_candidate_count += 1
            if predicted_candidate is not None:
                candidates.append(predicted_candidate)

    candidates = _deduplicate_temporal_candidates(candidates)
    if not candidates:
        _record_temporal_miss(state, calibration)
        return _empty_temporal_projection(
            role_count,
            resolution,
            state,
            ["no_current_mask_supported_candidate"],
            diagnostics={
                "entity_pixels": pixel_count,
                "components": len(components),
                "topology": topology_records,
                "scored_candidate_count": scored_candidate_count,
                "state_initialized_before": initialized_before,
                "invalid_age_before": invalid_age_before,
                "early_rejection": False,
            },
        )

    best = candidates[0]
    distinct = [
        candidate
        for candidate in candidates[1:]
        if float(np.mean(candidate.assignments != best.assignments)) >= 0.02
    ]
    second_cost = distinct[0].total_cost if distinct else float("inf")
    margin = (
        second_cost - best.total_cost if math.isfinite(second_cost) else float("inf")
    )
    ambiguity_factor = (
        1.0
        if not math.isfinite(second_cost)
        else 0.55 + 0.45 * (1.0 - math.exp(-max(margin, 0.0)))
    )
    absolute_fit = math.exp(-best.total_cost / 4.0)
    confidence_value = float(
        np.clip(
            absolute_fit
            * best.coverage
            * ambiguity_factor
            * (1.0 - best.disconnected_fraction),
            0.0,
            1.0,
        )
    )
    if best.coverage < 0.55:
        failure.append("insufficient_current_mask_support")
    if math.isfinite(second_cost) and margin < 0.01:
        failure.append("candidate_identity_ambiguous")
    if confidence_value < spec.minimum_projection_confidence:
        failure.append("projection_confidence_too_low")
    if failure:
        _record_temporal_miss(state, calibration)
        return _empty_temporal_projection(
            role_count,
            resolution,
            state,
            failure,
            diagnostics={
                "entity_pixels": pixel_count,
                "components": len(components),
                "topology": topology_records,
                "candidate_count": len(candidates),
                "scored_candidate_count": scored_candidate_count,
                "best_cost": best.total_cost,
                "second_distinct_cost": (
                    second_cost if math.isfinite(second_cost) else None
                ),
                "candidate_margin": margin if math.isfinite(margin) else None,
                "best_coverage": best.coverage,
                "best_source": best.source,
                "projector_confidence": confidence_value,
                "state_initialized_before": initialized_before,
                "invalid_age_before": invalid_age_before,
                "early_rejection": False,
            },
        )

    role_masks = np.zeros((role_count, *resolution), dtype=bool)
    mask_yx = mask_xy[:, ::-1].astype(np.int64, copy=False)
    for role_index in range(role_count):
        selected = mask_yx[best.assignments == role_index]
        role_masks[role_index, selected[:, 0], selected[:, 1]] = True
    if (
        any(not role_mask.any() for role_mask in role_masks)
        or np.any(role_masks.astype(np.uint8).sum(axis=0) > 1)
        or not np.array_equal(role_masks.any(axis=0), mask)
    ):
        _record_temporal_miss(state, calibration)
        return _empty_temporal_projection(
            role_count,
            resolution,
            state,
            ["current_mask_partition_invariant_failed"],
            diagnostics={
                "candidate_count": len(candidates),
                "scored_candidate_count": scored_candidate_count,
                "early_rejection": False,
            },
        )

    keypoints = np.stack((best.pose_xy[:-1], best.pose_xy[1:]), axis=1)
    _record_temporal_accept(state, best.pose_xy, calibration)
    return _Projection(
        masks=np.ascontiguousarray(role_masks),
        keypoints=np.ascontiguousarray(keypoints),
        valid=np.ones(role_count, dtype=bool),
        confidence=np.full(role_count, confidence_value, dtype=np.float32),
        diagnostics={
            "projector": "ordered_chain_temporal_v2",
            "failure_reasons": [],
            "entity_pixels": pixel_count,
            "components": len(components),
            "topology": topology_records,
            "candidate_count": len(candidates),
            "scored_candidate_count": scored_candidate_count,
            "best_cost": best.total_cost,
            "second_distinct_cost": second_cost if math.isfinite(second_cost) else None,
            "candidate_margin": margin if math.isfinite(margin) else None,
            "best_coverage": best.coverage,
            "best_source": best.source,
            "path_length_px": best.path_length_px,
            "length_cost": best.length_cost,
            "pixel_fit_cost": best.pixel_fit_cost,
            "segment_fit_cost": best.segment_fit_cost,
            "area_cost": best.area_cost,
            "root_cost": best.root_cost,
            "temporal_cost": best.temporal_cost,
            "switch_cost": best.switch_cost,
            "projector_confidence": confidence_value,
            "state_initialized_before": initialized_before,
            "state_initialized": state.initialized,
            "invalid_age_before": invalid_age_before,
            "invalid_age": state.invalid_age,
            "accepted_frames": state.accepted_frames,
            "branch_signature": list(state.branch_signature or ()),
            "current_mask_partition_exact": True,
            "fail_closed": False,
            "early_rejection": False,
        },
    )


@dataclass(frozen=True)
class RoleTokenFrame:
    role_names: tuple[str, ...]
    masks: np.ndarray
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    lost: np.ndarray
    confidence: np.ndarray
    mask_score: np.ndarray
    valid: np.ndarray
    runtime_ms: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RoleGeometryFrame:
    """Mask-only parser output for honest offline temporal replay.

    This deliberately has no appearance or descriptor field.  Calling it once
    advances the same episode state as one real-time ``project`` call.
    """

    role_names: tuple[str, ...]
    masks: np.ndarray
    keypoints_xy: np.ndarray
    valid: np.ndarray
    projector_confidence: np.ndarray
    runtime_ms: float
    diagnostics: dict[str, Any]


class SupportConditionedObjectGraphTokenizer:
    """Project current Cutie entity outputs into ordered role tokens."""

    def __init__(
        self,
        graph: CompiledObjectGraph,
        source_support_indexed_masks: np.ndarray,
    ):
        masks = np.asarray(source_support_indexed_masks)
        if masks.ndim != 3 or masks.dtype == np.bool_ or not np.issubdtype(
            masks.dtype, np.integer
        ):
            raise ObjectGraphContractError("Source support masks must be indexed [N,H,W].")
        project_support_to_entities(
            np.zeros((*masks.shape, 3), dtype=np.uint8), masks, graph
        )
        self.graph = graph
        self._resolution = tuple(int(value) for value in masks.shape[1:])
        calibrations: dict[str, ChainCalibration] = {}
        temporal_calibrations: dict[str, TemporalChainCalibration] = {}
        for entity in graph.entities:
            if entity.projector.type == "ordered_chain_segments_v1":
                calibrations[entity.name] = _calibrate_chain(graph, entity, masks)
            elif entity.projector.type == "ordered_chain_temporal_v2":
                temporal_calibrations[entity.name] = _calibrate_temporal_chain(
                    graph, entity, masks
                )
        self._calibrations = calibrations
        self._temporal_calibrations = temporal_calibrations
        self._temporal_states = {
            name: _TemporalChainState()
            for name in sorted(self._temporal_calibrations)
        }
        self._has_temporal = bool(self._temporal_calibrations)
        self._frames = 0
        self._resets = 0

    def reset_episode(self) -> None:
        self._frames = 0
        self._temporal_states = {
            name: _TemporalChainState()
            for name in sorted(self._temporal_calibrations)
        }
        self._resets += 1

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "format": TOKENIZER_FORMAT,
            **self.graph.metadata(),
            "source_resolution": list(self._resolution),
            "chain_calibrations": {
                name: calibration.metadata()
                for name, calibration in sorted(self._calibrations.items())
            },
            "episode_state": "reset_only_frame_counter_no_future_memory_v1",
            "runtime_inputs": [
                "current_entity_masks",
                "current_entity_features",
                "current_entity_lost",
                "current_entity_confidence",
                "current_entity_mask_score",
            ],
            "forbidden_runtime_inputs": [
                "simulator_state",
                "episode_ground_truth",
                "reward",
                "action",
                "future_rgb",
            ],
        }
        if self._has_temporal:
            metadata.update(
                {
                    "format": TEMPORAL_TOKENIZER_FORMAT,
                    "temporal_state_protocol": TEMPORAL_STATE_PROTOCOL,
                    "temporal_chain_calibrations": {
                        name: calibration.metadata()
                        for name, calibration in sorted(
                            self._temporal_calibrations.items()
                        )
                    },
                    "episode_state": (
                        "reset_clears_pose_velocity_uncertainty_invalid_age_"
                        "and_branch_signature_v2"
                    ),
                    "temporal_state_fields": [
                        "last_pose_xy",
                        "previous_pose_xy",
                        "velocity_xy",
                        "uncertainty_px",
                        "invalid_age",
                        "branch_signature",
                        "accepted_frames",
                    ],
                    "geometry_replay_entrypoint": "project_temporal_geometry",
                    "geometry_replay_protocol": "mask_and_tracker_lost_only_v2",
                    "geometry_replay_has_appearance_or_descriptors": False,
                }
            )
        return metadata

    def _validate_geometry_inputs(
        self,
        entity_masks: np.ndarray,
        entity_lost: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks = np.asarray(entity_masks)
        lost = np.asarray(entity_lost)
        entity_count = len(self.graph.entities)
        if masks.shape != (entity_count, *self._resolution):
            raise ObjectGraphContractError("Entity mask shape changed.")
        if not np.issubdtype(masks.dtype, np.bool_):
            if not (
                np.issubdtype(masks.dtype, np.number)
                and np.isfinite(masks).all()
                and np.logical_or(masks == 0, masks == 1).all()
            ):
                raise ObjectGraphContractError("Entity masks must be binary.")
        if lost.shape != (entity_count,):
            raise ObjectGraphContractError("Entity lost shape changed.")
        if lost.dtype != np.bool_:
            raise ObjectGraphContractError("Entity lost flags must be boolean.")
        if np.any(masks.astype(np.uint8).sum(axis=0) > 1):
            raise ObjectGraphContractError("Tracker entity masks overlap.")
        return masks, lost

    def _geometry_step(
        self,
        entity_masks: np.ndarray,
        entity_available: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        role_count = len(self.graph.semantic_roles)
        role_masks = np.zeros((role_count, *self._resolution), dtype=bool)
        role_keypoints = np.zeros((role_count, 2, 2), dtype=np.float32)
        valid = np.zeros(role_count, dtype=bool)
        projector_confidence = np.zeros(role_count, dtype=np.float32)
        diagnostics: dict[str, Any] = {}
        role_offset = 0
        for entity_index, entity in enumerate(self.graph.entities):
            entity_mask = np.ascontiguousarray(entity_masks[entity_index], dtype=bool)
            available = bool(entity_available[entity_index] and entity_mask.any())
            if entity.projector.type == "direct_role_v1":
                projection = _direct_projection(entity_mask)
            elif entity.projector.type == "ordered_chain_segments_v1":
                projection = _chain_projection(
                    entity_mask,
                    entity.projector,
                    self._calibrations[entity.name],
                )
            else:
                projection = _temporal_chain_projection(
                    entity_mask,
                    entity.projector,
                    self._temporal_calibrations[entity.name],
                    self._temporal_states[entity.name],
                    entity_available=available,
                )
            count = len(entity.projector.roles)
            for local_index in range(count):
                role_index = role_offset + local_index
                geometry_valid = bool(
                    available
                    and projection.valid[local_index]
                    and projection.masks[local_index].any()
                )
                if geometry_valid:
                    role_masks[role_index] = projection.masks[local_index]
                    role_keypoints[role_index] = projection.keypoints[local_index]
                    valid[role_index] = True
                    projector_confidence[role_index] = projection.confidence[local_index]
            diagnostics[entity.name] = projection.diagnostics
            role_offset += count
        if role_offset != role_count:
            raise ObjectGraphContractError("Role geometry construction failed.")
        if np.any(role_masks.astype(np.uint8).sum(axis=0) > 1):
            raise ObjectGraphContractError("Projected semantic role masks overlap.")
        self._frames += 1
        return (
            np.ascontiguousarray(role_masks),
            np.ascontiguousarray(role_keypoints),
            np.ascontiguousarray(valid),
            np.ascontiguousarray(projector_confidence),
            diagnostics,
        )

    def project_temporal_geometry(
        self,
        *,
        entity_masks: np.ndarray,
        entity_lost: np.ndarray,
    ) -> RoleGeometryFrame:
        """Advance geometry once using masks/status only, without fake appearance.

        Offline replay must call either this method or :meth:`project` for one
        frame, never both.  They share the same causal episode state and reset.
        """

        started = perf_counter()
        masks, lost = self._validate_geometry_inputs(entity_masks, entity_lost)
        available = (~lost) & masks.reshape(len(self.graph.entities), -1).any(axis=1)
        (
            role_masks,
            keypoints,
            valid,
            projector_confidence,
            diagnostics,
        ) = self._geometry_step(masks, available)
        return RoleGeometryFrame(
            role_names=self.graph.semantic_roles,
            masks=role_masks,
            keypoints_xy=keypoints,
            valid=valid,
            projector_confidence=projector_confidence,
            runtime_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )

    def _assemble_temporal_tokens(
        self,
        *,
        started: float,
        entity_masks: np.ndarray,
        entity_features: np.ndarray,
        entity_lost: np.ndarray,
        entity_confidence: np.ndarray,
        entity_mask_score: np.ndarray,
    ) -> RoleTokenFrame:
        appearances: list[np.ndarray] = []
        appearance_finite = np.zeros(len(self.graph.entities), dtype=bool)
        for entity_index in range(len(self.graph.entities)):
            appearance, finite = _appearance_feature(entity_features[entity_index])
            appearances.append(appearance)
            appearance_finite[entity_index] = finite
        available = (
            (~entity_lost)
            & appearance_finite
            & entity_masks.reshape(len(self.graph.entities), -1).any(axis=1)
        )
        (
            role_masks,
            role_keypoints,
            valid,
            projector_confidence,
            diagnostics,
        ) = self._geometry_step(entity_masks, available)

        role_count = len(self.graph.semantic_roles)
        descriptors = np.zeros((role_count, FRAME_DIM), dtype=np.float32)
        lost = np.zeros(role_count, dtype=bool)
        confidence = np.zeros(role_count, dtype=np.float32)
        mask_score = np.zeros(role_count, dtype=np.float32)
        for role_index, entity_index in enumerate(self.graph.role_to_entity_index):
            lost[role_index] = bool(entity_lost[entity_index])
            if valid[role_index]:
                confidence[role_index] = float(
                    np.clip(
                        entity_confidence[entity_index]
                        * projector_confidence[role_index],
                        0.0,
                        1.0,
                    )
                )
                mask_score[role_index] = float(
                    np.clip(
                        entity_mask_score[entity_index]
                        * projector_confidence[role_index],
                        0.0,
                        1.0,
                    )
                )
                descriptors[role_index, :APPEARANCE_DIM] = appearances[entity_index]
                descriptors[
                    role_index,
                    APPEARANCE_DIM:APPEARANCE_DIM + GEOMETRY_DIM,
                ] = _mask_geometry(role_masks[role_index])
            descriptors[role_index, -STATUS_DIM:] = np.asarray(
                [
                    confidence[role_index],
                    float(lost[role_index]),
                    float(valid[role_index]),
                    mask_score[role_index],
                ],
                dtype=np.float32,
            )
        if not np.isfinite(descriptors).all():
            raise ObjectGraphContractError("Role token construction failed.")
        return RoleTokenFrame(
            role_names=self.graph.semantic_roles,
            masks=role_masks,
            keypoints_xy=role_keypoints,
            descriptors=np.ascontiguousarray(descriptors),
            lost=np.ascontiguousarray(lost),
            confidence=np.ascontiguousarray(confidence),
            mask_score=np.ascontiguousarray(mask_score),
            valid=np.ascontiguousarray(valid),
            runtime_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )

    def project(
        self,
        *,
        entity_masks: np.ndarray,
        entity_features: np.ndarray,
        entity_lost: np.ndarray,
        entity_confidence: np.ndarray,
        entity_mask_score: np.ndarray,
    ) -> RoleTokenFrame:
        started = perf_counter()
        entity_masks = np.asarray(entity_masks)
        entity_features = np.asarray(entity_features)
        entity_lost = np.asarray(entity_lost)
        entity_confidence = np.asarray(entity_confidence)
        entity_mask_score = np.asarray(entity_mask_score)
        entity_count = len(self.graph.entities)
        if entity_masks.shape != (entity_count, *self._resolution):
            raise ObjectGraphContractError("Entity mask shape changed.")
        if entity_features.shape != (entity_count, QUERY_FEATURE_DIM):
            raise ObjectGraphContractError("Entity feature shape changed.")
        if not np.issubdtype(entity_features.dtype, np.floating):
            raise ObjectGraphContractError("Entity features must be real floating point.")
        for label, value in (
            ("lost", entity_lost),
            ("confidence", entity_confidence),
            ("mask_score", entity_mask_score),
        ):
            if value.shape != (entity_count,):
                raise ObjectGraphContractError(f"Entity {label} shape changed.")
        if not np.issubdtype(entity_masks.dtype, np.bool_):
            if not (
                np.issubdtype(entity_masks.dtype, np.number)
                and np.isfinite(entity_masks).all()
                and np.logical_or(entity_masks == 0, entity_masks == 1).all()
            ):
                raise ObjectGraphContractError("Entity masks must be binary.")
        if entity_lost.dtype != np.bool_:
            raise ObjectGraphContractError("Entity lost flags must be boolean.")
        for label, value in (
            ("confidence", entity_confidence),
            ("mask_score", entity_mask_score),
        ):
            if not np.issubdtype(value.dtype, np.floating):
                raise ObjectGraphContractError(
                    f"Entity {label} must be real floating point."
                )
            if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
                raise ObjectGraphContractError(
                    f"Entity {label} must be finite and remain in [0,1]."
                )
        if np.any(entity_masks.astype(np.uint8).sum(axis=0) > 1):
            raise ObjectGraphContractError("Tracker entity masks overlap.")

        # Pure v1 graphs deliberately stay on the historical loop below.  This
        # preserves bitwise direct-role behavior, including its old distinction
        # between tracker ``lost`` and geometry availability.  A graph containing
        # temporal v2 takes exactly one shared causal geometry step here.
        if self._has_temporal:
            return self._assemble_temporal_tokens(
                started=started,
                entity_masks=entity_masks,
                entity_features=entity_features,
                entity_lost=entity_lost,
                entity_confidence=entity_confidence,
                entity_mask_score=entity_mask_score,
            )

        role_count = len(self.graph.semantic_roles)
        role_masks = np.zeros((role_count, *self._resolution), dtype=bool)
        role_keypoints = np.zeros((role_count, 2, 2), dtype=np.float32)
        descriptors = np.zeros((role_count, FRAME_DIM), dtype=np.float32)
        lost = np.ones(role_count, dtype=bool)
        confidence = np.zeros(role_count, dtype=np.float32)
        mask_score = np.zeros(role_count, dtype=np.float32)
        valid = np.zeros(role_count, dtype=bool)
        diagnostics: dict[str, Any] = {}
        role_offset = 0
        for entity_index, entity in enumerate(self.graph.entities):
            entity_mask = np.ascontiguousarray(entity_masks[entity_index], dtype=bool)
            if entity.projector.type == "direct_role_v1":
                projection = _direct_projection(entity_mask)
            else:
                projection = _chain_projection(
                    entity_mask,
                    entity.projector,
                    self._calibrations[entity.name],
                )
            appearance, appearance_finite = _appearance_feature(
                entity_features[entity_index]
            )
            count = len(entity.projector.roles)
            entity_valid = (
                not bool(entity_lost[entity_index])
                and appearance_finite
                and bool(entity_mask.any())
            )
            for local_index in range(count):
                role_index = role_offset + local_index
                role_mask = projection.masks[local_index]
                geometry_valid = bool(
                    projection.valid[local_index] and role_mask.any()
                )
                role_valid = bool(entity_valid and geometry_valid)
                # Masks remain a model-neutral geometric prediction even when
                # Cutie's backend-specific lost flag invalidates the appearance
                # token.  This preserves direct-projector mask parity and keeps
                # cross-backend scoring independent of confidence calibration.
                role_masks[role_index] = role_mask if geometry_valid else False
                if geometry_valid:
                    role_keypoints[role_index] = projection.keypoints[local_index]
                projector_confidence = (
                    1.0
                    if entity.projector.type == "direct_role_v1"
                    else float(projection.confidence[local_index])
                )
                confidence[role_index] = float(
                    np.clip(
                        entity_confidence[entity_index]
                        * projector_confidence,
                        0.0,
                        1.0,
                    )
                )
                mask_score[role_index] = float(
                    np.clip(
                        entity_mask_score[entity_index]
                        * projector_confidence,
                        0.0,
                        1.0,
                    )
                )
                valid[role_index] = role_valid
                # Preserve the historical descriptor contract: ``lost`` is the
                # tracker's reported state, while ``valid`` additionally gates
                # empty masks, non-finite appearance, and parser geometry.
                # They are deliberately not logical complements.
                lost[role_index] = bool(entity_lost[entity_index])
                if role_valid:
                    descriptors[role_index, :APPEARANCE_DIM] = appearance
                descriptors[
                    role_index,
                    APPEARANCE_DIM:APPEARANCE_DIM + GEOMETRY_DIM,
                ] = _mask_geometry(role_masks[role_index])
                descriptors[role_index, -STATUS_DIM:] = np.asarray(
                    [
                        confidence[role_index],
                        float(lost[role_index]),
                        float(valid[role_index]),
                        mask_score[role_index],
                    ],
                    dtype=np.float32,
                )
            diagnostics[entity.name] = projection.diagnostics
            role_offset += count

        if role_offset != role_count or not np.isfinite(descriptors).all():
            raise ObjectGraphContractError("Role token construction failed.")
        if np.any(role_masks.astype(np.uint8).sum(axis=0) > 1):
            raise ObjectGraphContractError("Projected semantic role masks overlap.")
        self._frames += 1
        return RoleTokenFrame(
            role_names=self.graph.semantic_roles,
            masks=np.ascontiguousarray(role_masks),
            keypoints_xy=np.ascontiguousarray(role_keypoints),
            descriptors=np.ascontiguousarray(descriptors),
            lost=np.ascontiguousarray(lost),
            confidence=np.ascontiguousarray(confidence),
            mask_score=np.ascontiguousarray(mask_score),
            valid=np.ascontiguousarray(valid),
            runtime_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )
