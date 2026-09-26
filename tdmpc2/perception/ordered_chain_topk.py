"""Current-frame top-K hypotheses for a support-calibrated planar chain.

This module is deliberately stateless.  It consumes one current entity mask
and fixed labelled support masks, then returns a small ordered set of
kinematically valid pose hypotheses.  Episode ground truth, actions, rewards,
simulator state, appearance features, and future frames are not accepted by
the public API.

The first implementation primitive is a two-link planar ordered chain.  The
primitive is selected by graph structure rather than by task name.  It keeps
the published object-graph v1/v2 projectors unchanged and is intended only for
an offline candidate-coverage preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from tdmpc2.perception.support_conditioned_object_graph import (
    CompiledObjectGraph,
    ObjectGraphContractError,
    TemporalChainCalibration,
    _binary_close,
    _calibrate_chain,
    _calibrate_temporal_chain,
    _chain_projection,
    _components,
    _dijkstra,
    _path,
    _sample_polyline,
    _segment_distance_squared,
    _skeleton_graph,
    _zhang_suen,
)


FORMAT = "support_conditioned_ordered_chain_topk_v1"
PROTOCOL = "stateless_current_entity_mask_topk_two_link_v1"
MAX_CANDIDATES = 4
MAX_COMPONENTS = 3
MAX_ROOTS = 1
MAX_ENDPOINTS = 4
MAX_SCORED_CANDIDATES = 32
SAMPLES_PER_LINK = 16
WEIGHT_TEMPERATURE = 0.5

SOURCE_CODES: Mapping[str, int] = {
    "padding": 0,
    "v1_anchor": 1,
    "skeleton_raw": 2,
    "skeleton_closed": 3,
    "circle_raw_positive": 4,
    "circle_raw_negative": 5,
    "circle_closed_positive": 6,
    "circle_closed_negative": 7,
    "folded_raw": 8,
    "folded_closed": 9,
}


def _typed_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _circle_intersections(
    root_xy: np.ndarray,
    tip_xy: np.ndarray,
    first_length: float,
    second_length: float,
) -> tuple[np.ndarray, ...]:
    """Return both possible elbows joining a fixed root and proposed tip."""
    root = np.asarray(root_xy, dtype=np.float64)
    tip = np.asarray(tip_xy, dtype=np.float64)
    delta = tip - root
    distance = float(np.linalg.norm(delta))
    if (
        not math.isfinite(distance)
        or distance <= 1e-9
        or distance > first_length + second_length + 1e-6
        or distance < abs(first_length - second_length) - 1e-6
    ):
        return ()
    along = (
        first_length * first_length
        - second_length * second_length
        + distance * distance
    ) / (2.0 * distance)
    height_squared = max(first_length * first_length - along * along, 0.0)
    midpoint = root + (along / distance) * delta
    perpendicular = np.asarray([-delta[1], delta[0]], dtype=np.float64) / distance
    height = math.sqrt(height_squared)
    if height <= 1e-8:
        return (midpoint,)
    left = midpoint + height * perpendicular
    right = midpoint - height * perpendicular
    # ``left`` has positive signed bend relative to root->tip because it uses
    # the positive perpendicular; ``right`` has the negative bend.  Preserve
    # that semantic order instead of relabelling a y/x sort as a bend sign.
    return (
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
    )


def role_masks_from_pose(entity_mask: np.ndarray, pose_xy: np.ndarray) -> np.ndarray:
    """Partition current entity pixels by their nearest ordered chain link."""
    mask = np.ascontiguousarray(entity_mask, dtype=np.bool_)
    pose = np.ascontiguousarray(pose_xy, dtype=np.float64)
    if mask.ndim != 2 or pose.ndim != 2 or pose.shape[1:] != (2,):
        raise ValueError("Entity mask or ordered-chain pose shape is malformed.")
    role_count = pose.shape[0] - 1
    if role_count < 1 or not np.isfinite(pose).all():
        raise ValueError("Ordered-chain pose is invalid.")
    mask_yx = np.argwhere(mask)
    output = np.zeros((role_count, *mask.shape), dtype=np.bool_)
    if not len(mask_yx):
        return output
    mask_xy = mask_yx[:, ::-1].astype(np.float64)
    distances = np.stack(
        [
            _segment_distance_squared(
                mask_xy[:, 0], mask_xy[:, 1], pose[index], pose[index + 1]
            )
            for index in range(role_count)
        ]
    )
    assignment = np.argmin(distances, axis=0)
    for role_index in range(role_count):
        selected = mask_yx[assignment == role_index]
        if len(selected):
            output[role_index, selected[:, 0], selected[:, 1]] = True
    return np.ascontiguousarray(output)


def _render_union(
    pose_xy: np.ndarray,
    shape: tuple[int, int],
    half_width_px: float,
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.bool_)
    margin = int(math.ceil(float(half_width_px))) + 1
    for index in range(pose_xy.shape[0] - 1):
        start, end = pose_xy[index], pose_xy[index + 1]
        x0 = max(int(math.floor(min(start[0], end[0]))) - margin, 0)
        x1 = min(int(math.ceil(max(start[0], end[0]))) + margin + 1, shape[1])
        y0 = max(int(math.floor(min(start[1], end[1]))) - margin, 0)
        y1 = min(int(math.ceil(max(start[1], end[1]))) + margin + 1, shape[0])
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
        xx += x0
        yy += y0
        distance = _segment_distance_squared(xx, yy, start, end)
        output[y0:y1, x0:x1] |= distance <= float(half_width_px) ** 2
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class _Candidate:
    pose_xy: np.ndarray
    role_masks: np.ndarray
    cost: float
    confidence: float
    coverage: float
    render_iou: float
    source: str
    source_code: int
    branch_fraction: float
    disconnected_fraction: float
    pixel_fit_cost: float
    segment_fit_cost: float
    length_cost: float
    area_cost: float
    root_cost: float


@dataclass(frozen=True)
class OrderedChainTopKFrame:
    role_names: tuple[str, ...]
    poses_xy: np.ndarray
    role_masks: np.ndarray
    valid: np.ndarray
    costs: np.ndarray
    weights: np.ndarray
    confidence: np.ndarray
    source_codes: np.ndarray
    candidate_count: int
    best_second_cost_margin: float
    weight_entropy: float
    weighted_pose_dispersion_px: float
    fit_uncertainty: float
    normalized_weight_entropy: float
    runtime_ms: float
    diagnostics: dict[str, Any]


class OrderedChainTopKGenerator:
    """Support-calibrated, task-neutral two-link candidate generator."""

    def __init__(
        self,
        graph: CompiledObjectGraph,
        source_indexed_masks: np.ndarray,
        *,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
            raise ObjectGraphContractError(
                f"max_candidates must be in [1,{MAX_CANDIDATES}]."
            )
        if len(graph.entities) != 1:
            raise ObjectGraphContractError(
                "The top-K primitive currently requires one grouped tracking entity."
            )
        entity = graph.entities[0]
        if entity.projector.type not in {
            "ordered_chain_segments_v1",
            "ordered_chain_temporal_v2",
        }:
            raise ObjectGraphContractError(
                "The top-K primitive requires an ordered-chain projector."
            )
        if len(entity.projector.roles) != 2:
            raise ObjectGraphContractError(
                "The current top-K primitive is explicitly limited to two links."
            )
        masks = np.ascontiguousarray(source_indexed_masks)
        if (
            masks.ndim != 3
            or masks.shape[0] != 6
            or masks.dtype == np.bool_
            or not np.issubdtype(masks.dtype, np.integer)
        ):
            raise ObjectGraphContractError(
                "Top-K calibration requires six indexed support masks."
            )
        expected_ids = set(range(len(graph.source_roles) + 1))
        if not set(int(value) for value in np.unique(masks)).issubset(expected_ids):
            raise ObjectGraphContractError("Support masks contain an unknown role id.")
        for role_index in range(1, len(graph.source_roles) + 1):
            if any(not np.any(frame == role_index) for frame in masks):
                raise ObjectGraphContractError(
                    "Every support frame must contain every source role."
                )
        self.graph = graph
        self.entity = entity
        self.max_candidates = max_candidates
        self.v1_calibration = _calibrate_chain(graph, entity, masks)
        self.calibration = _calibrate_temporal_chain(graph, entity, masks)
        self._support_trace_sha256 = _typed_array_sha256(masks)
        self._resolution = tuple(int(value) for value in masks.shape[1:])

    def metadata(self) -> dict[str, Any]:
        payload = {
            "format": FORMAT,
            "protocol": PROTOCOL,
            "graph_sha256": self.graph.graph_sha256,
            "entity_name": self.entity.name,
            "role_names": list(self.entity.projector.roles),
            "max_candidates": self.max_candidates,
            "source_codes": dict(SOURCE_CODES),
            "support_indexed_mask_trace_sha256": self._support_trace_sha256,
            "calibration": self.calibration.metadata(),
            "ranking": (
                "current_mask_render_iou_centerline_bidirectional_fit_"
                "length_area_root_topology_cost_v1"
            ),
            "candidate_weights": {
                "type": "softmax_negative_cost_v1",
                "temperature": WEIGHT_TEMPERATURE,
                "not_calibrated_for_controller_use": True,
            },
            "candidate_slot_semantics": (
                "slot_0_exact_v1_anchor_or_invalid_alternatives_start_at_slot_1_v1"
            ),
            "prefix_semantics": "K_uses_fixed_slots_0_through_K_minus_1_v1",
            "episode_state": False,
            "future_frames": False,
            "task_name_dispatch": False,
            "ground_truth_input": False,
            "support_resolution": list(self._resolution),
            "branch_threshold_policy": (
                "coverage_diagnostic_soft_penalty_not_online_acceptance_gate_v1"
            ),
            "alternative_pose_semantics": (
                "seal_float32_then_score_partition_and_publish_v1"
            ),
            "anchor_mask_semantics": (
                "exact_published_v1_masks_pose_used_for_parity_only_v1"
            ),
        }
        payload["metadata_sha256"] = _json_sha256(payload)
        return payload

    def _score(
        self,
        pose_xy: np.ndarray,
        mask: np.ndarray,
        mask_xy: np.ndarray,
        *,
        branch_fraction: float,
        disconnected_fraction: float,
        source: str,
        role_masks_override: np.ndarray | None = None,
    ) -> _Candidate | None:
        pose = np.ascontiguousarray(pose_xy, dtype=np.float64)
        # Alternative masks are reconstructed later from the compact backend
        # artifact.  Make the published float32 pose the *only* semantic pose
        # before any scoring or partitioning so that reconstruction is byte
        # exact rather than a second, slightly rounded geometric method.
        # Slot zero is different by design: its exact v1 masks are preserved
        # through role_masks_override and the pose is parity metadata only.
        if role_masks_override is None:
            pose = np.ascontiguousarray(
                np.ascontiguousarray(pose, dtype=np.float32), dtype=np.float64
            )
        calibration = self.calibration
        if pose.shape != (3, 2) or not np.isfinite(pose).all():
            return None
        lengths = np.linalg.norm(np.diff(pose, axis=0), axis=1)
        if np.any(lengths <= 1.0):
            return None
        role_masks = (
            role_masks_from_pose(mask, pose)
            if role_masks_override is None
            else np.ascontiguousarray(role_masks_override, dtype=np.bool_)
        )
        if role_masks.shape != (2, *mask.shape):
            return None
        if np.any(role_masks.astype(np.uint8).sum(axis=0) > 1):
            return None
        if not np.array_equal(role_masks.any(axis=0), mask):
            return None
        counts = role_masks.reshape(2, -1).sum(axis=1).astype(np.float64)
        if np.any(counts <= 0.0):
            return None
        fractions = counts / max(float(mask.sum()), 1.0)

        point_distances = np.stack(
            [
                _segment_distance_squared(
                    mask_xy[:, 0], mask_xy[:, 1], pose[index], pose[index + 1]
                )
                for index in range(2)
            ]
        )
        minimum_distance = np.sqrt(np.min(point_distances, axis=0))
        pixel_fit = float(
            np.sqrt(np.mean(minimum_distance * minimum_distance))
            / calibration.half_width_px
        )
        samples = []
        alpha = np.linspace(0.0, 1.0, SAMPLES_PER_LINK, dtype=np.float64)[:, None]
        for index in range(2):
            samples.append(pose[index] + alpha * (pose[index + 1] - pose[index]))
        sample_xy = np.concatenate(samples, axis=0)
        sample_distance = np.sqrt(
            np.min(
                ((sample_xy[:, None, :] - mask_xy[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
        )
        segment_fit = float(
            np.sqrt(np.mean(sample_distance * sample_distance))
            / calibration.half_width_px
        )
        coverage = float(np.mean(sample_distance <= 2.25 * calibration.half_width_px))
        length_cost = float(
            np.mean(
                np.abs(lengths - np.asarray(calibration.link_lengths_px))
                / np.asarray(calibration.link_scales_px)
            )
        )
        area_cost = float(
            np.mean(
                np.abs(fractions - np.asarray(calibration.role_area_fractions))
                / np.asarray(calibration.role_area_scales)
            )
        )
        root_cost = float(
            np.linalg.norm(pose[0] - np.asarray(calibration.root_xy))
            / calibration.root_scale_px
        )
        rendered = _render_union(pose, mask.shape, calibration.half_width_px)
        intersection = int(np.logical_and(rendered, mask).sum())
        union = int(np.logical_or(rendered, mask).sum())
        render_iou = float(intersection / union) if union else 0.0
        total_cost = (
            0.55 * length_cost
            + 0.70 * pixel_fit
            + 0.70 * segment_fit
            + 0.25 * area_cost
            + 0.25 * root_cost
            + 1.50 * (1.0 - coverage)
            + 1.00 * (1.0 - render_iou)
            + 0.60 * max(branch_fraction, 0.0)
            + 1.50 * max(disconnected_fraction, 0.0)
        )
        confidence = float(
            np.clip(
                math.exp(-total_cost / 4.0)
                * coverage
                * render_iou
                * (1.0 - disconnected_fraction),
                0.0,
                1.0,
            )
        )
        return _Candidate(
            pose_xy=pose,
            role_masks=role_masks,
            cost=float(total_cost),
            confidence=confidence,
            coverage=coverage,
            render_iou=render_iou,
            source=source,
            source_code=SOURCE_CODES[source],
            branch_fraction=float(branch_fraction),
            disconnected_fraction=float(disconnected_fraction),
            pixel_fit_cost=pixel_fit,
            segment_fit_cost=segment_fit,
            length_cost=length_cost,
            area_cost=area_cost,
            root_cost=root_cost,
        )

    def _enumerate(
        self, mask: np.ndarray
    ) -> tuple[list[_Candidate | None], dict[str, Any]]:
        calibration = self.calibration
        spec = self.entity.projector
        pixel_count = int(mask.sum())
        maximum_pixels = spec.maximum_entity_area_multiple * calibration.expected_entity_pixels
        if pixel_count < spec.minimum_entity_pixels:
            return [None] * self.max_candidates, {
                "failure": "entity_too_small",
                "entity_pixels": pixel_count,
                "v1_anchor_valid": False,
                "anchor_slot_reserved": True,
            }
        if pixel_count > maximum_pixels:
            return [None] * self.max_candidates, {
                "failure": "entity_area_implausible",
                "entity_pixels": pixel_count,
                "maximum_entity_pixels": maximum_pixels,
                "v1_anchor_valid": False,
                "anchor_slot_reserved": True,
            }
        base = np.asarray(calibration.root_xy, dtype=np.float64)
        components = sorted(
            _components(mask, base),
            key=lambda item: (item.minimum_base_distance_px, -item.size, item.first_yx),
        )
        mask_xy = np.argwhere(mask)[:, ::-1].astype(np.float64)
        candidates: list[_Candidate] = []
        scored = 0
        topology: list[dict[str, Any]] = []

        # Rank zero is the exact published v1 current-frame parser whenever it
        # is valid.  This makes K=1 a real baseline and ensures extra hypotheses
        # can only add alternatives rather than silently replacing v1.
        anchor_projection = _chain_projection(
            mask, self.entity.projector, self.v1_calibration
        )
        anchor: _Candidate | None = None
        if bool(anchor_projection.valid.all()):
            anchor_pose = np.vstack(
                (
                    anchor_projection.keypoints[0, 0],
                    anchor_projection.keypoints[:, 1],
                )
            )
            anchor = self._score(
                anchor_pose,
                mask,
                mask_xy,
                branch_fraction=float(
                    anchor_projection.diagnostics.get("branch_fraction", 0.0)
                ),
                disconnected_fraction=float(
                    anchor_projection.diagnostics.get("disconnected_fraction", 0.0)
                ),
                source="v1_anchor",
                role_masks_override=anchor_projection.masks,
            )
            if anchor is not None:
                candidates.append(anchor)
                scored += 1
        for component_index, component in enumerate(components[:MAX_COMPONENTS]):
            disconnected = 1.0 - float(component.size) / max(pixel_count, 1)
            if (
                component.minimum_base_distance_px
                > spec.maximum_base_distance_fraction * calibration.expected_total_length_px
                or disconnected > spec.maximum_disconnected_fraction
            ):
                topology.append(
                    {
                        "component": component_index,
                        "skipped": True,
                        "disconnected_fraction": disconnected,
                        "minimum_base_distance_px": component.minimum_base_distance_px,
                    }
                )
                continue
            variants = [("raw", component.mask)]
            closed = _binary_close(component.mask)
            if not np.array_equal(closed, component.mask):
                variants.append(("closed", closed))
            for variant_name, centreline_mask in variants:
                graph = _skeleton_graph(_zhang_suen(centreline_mask))
                if len(graph) < 2:
                    continue
                nodes = sorted(graph)
                endpoints = [node for node in nodes if len(graph[node]) <= 1]
                ranked_roots = sorted(
                    nodes,
                    key=lambda node: (
                        float(
                            np.linalg.norm(
                                np.asarray([node[1], node[0]], dtype=np.float64) - base
                            )
                        ),
                        node,
                    ),
                )[:MAX_ROOTS]
                generated = 0
                maximum_branch = 0.0
                for root_node in ranked_roots:
                    distance, previous = _dijkstra(graph, root_node)
                    usable_ends = [node for node in endpoints if node != root_node]
                    if not usable_ends:
                        usable_ends = [
                            node
                            for node, value in distance.items()
                            if node != root_node and math.isfinite(value)
                        ]
                    usable_ends = sorted(
                        usable_ends,
                        key=lambda node: (-distance[node], node),
                    )[:MAX_ENDPOINTS]
                    for end_node in usable_ends:
                        if scored >= MAX_SCORED_CANDIDATES:
                            break
                        path_nodes = _path(previous, root_node, end_node)
                        if len(path_nodes) < 2:
                            continue
                        path_xy = np.asarray(
                            [[node[1], node[0]] for node in path_nodes],
                            dtype=np.float64,
                        )
                        path_length = float(
                            np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
                        )
                        ratio = path_length / calibration.expected_total_length_px
                        branch = 1.0 - float(len(path_nodes)) / max(len(graph), 1)
                        maximum_branch = max(maximum_branch, branch)

                        # Short root-to-end paths are precisely where a folded
                        # two-link chain can make the visible endpoint be the
                        # elbow.  Generate that physical mode before applying
                        # the ordinary root-to-tip path-length gate.
                        if ratio < spec.minimum_path_length_fraction:
                            root_xy = np.asarray(
                                calibration.root_xy, dtype=np.float64
                            )
                            visible_end = np.asarray(
                                [end_node[1], end_node[0]], dtype=np.float64
                            )
                            direction = visible_end - root_xy
                            direction_norm = float(np.linalg.norm(direction))
                            if direction_norm > 1e-6:
                                direction /= direction_norm
                                elbow_xy = root_xy + float(
                                    calibration.link_lengths_px[0]
                                ) * direction
                                folded_tip = elbow_xy - float(
                                    calibration.link_lengths_px[1]
                                ) * direction
                                folded_source = f"folded_{variant_name}"
                                current = self._score(
                                    np.stack((root_xy, elbow_xy, folded_tip)),
                                    mask,
                                    mask_xy,
                                    branch_fraction=branch,
                                    disconnected_fraction=disconnected,
                                    source=folded_source,
                                )
                                scored += 1
                                if current is not None:
                                    candidates.append(current)
                                    generated += 1
                        if (
                            ratio < spec.minimum_path_length_fraction
                            or ratio > spec.maximum_path_length_fraction
                        ):
                            continue
                        cumulative = np.cumsum(
                            np.asarray(calibration.link_lengths_px, dtype=np.float64)
                        )
                        fractions = np.concatenate(
                            ([0.0], cumulative / max(float(cumulative[-1]), 1e-12))
                        )
                        skeleton_pose = np.stack(
                            [_sample_polyline(path_xy, value) for value in fractions]
                        )
                        skeleton_source = f"skeleton_{variant_name}"
                        current = self._score(
                            skeleton_pose,
                            mask,
                            mask_xy,
                            branch_fraction=branch,
                            disconnected_fraction=disconnected,
                            source=skeleton_source,
                        )
                        scored += 1
                        if current is not None:
                            candidates.append(current)
                            generated += 1

                        root_xy = np.asarray(calibration.root_xy, dtype=np.float64)
                        tip_xy = np.asarray([end_node[1], end_node[0]], dtype=np.float64)
                        elbows = _circle_intersections(
                            root_xy,
                            tip_xy,
                            float(calibration.link_lengths_px[0]),
                            float(calibration.link_lengths_px[1]),
                        )
                        for elbow_index, elbow_xy in enumerate(elbows):
                            if scored >= MAX_SCORED_CANDIDATES:
                                break
                            sign = "positive" if elbow_index == 0 else "negative"
                            circle_source = f"circle_{variant_name}_{sign}"
                            circle_pose = np.stack((root_xy, elbow_xy, tip_xy))
                            current = self._score(
                                circle_pose,
                                mask,
                                mask_xy,
                                branch_fraction=branch,
                                disconnected_fraction=disconnected,
                                source=circle_source,
                            )
                            scored += 1
                            if current is not None:
                                candidates.append(current)
                                generated += 1

                topology.append(
                    {
                        "component": component_index,
                        "skipped": False,
                        "variant": variant_name,
                        "skeleton_pixels": len(graph),
                        "endpoints": len(endpoints),
                        "generated_candidates": generated,
                        "maximum_branch_fraction": maximum_branch,
                        "disconnected_fraction": disconnected,
                    }
                )
                if scored >= MAX_SCORED_CANDIDATES:
                    break
            if scored >= MAX_SCORED_CANDIDATES:
                break

        alternatives = [candidate for candidate in candidates if candidate is not anchor]
        ordered_alternatives = sorted(
            alternatives,
            key=lambda item: (
                item.cost,
                item.source_code,
                tuple(float(value) for value in item.pose_xy.reshape(-1)),
            ),
        )
        # Slot zero is permanently reserved for the exact published v1
        # current-frame result.  If v1 fails, it remains invalid: an
        # alternative must never silently turn K=1 into a different method.
        retained: list[_Candidate] = []
        for candidate in ordered_alternatives:
            if len(retained) >= self.max_candidates - 1:
                break
            duplicate = any(
                self._same_mode(candidate, previous)
                for previous in ([anchor] if anchor is not None else []) + retained
            )
            if not duplicate:
                retained.append(candidate)
            if len(retained) >= self.max_candidates - 1:
                break
        slots: list[_Candidate | None] = [anchor, *retained]
        slots.extend([None] * (self.max_candidates - len(slots)))
        valid_count = sum(candidate is not None for candidate in slots)
        return slots, {
            "failure": (
                None if valid_count else "no_current_mask_supported_candidate"
            ),
            "entity_pixels": pixel_count,
            "components": len(components),
            "scored_candidates": scored,
            "unique_candidates": valid_count,
            "v1_anchor_valid": anchor is not None,
            "anchor_slot_reserved": True,
            "topology": topology,
        }

    def _same_mode(self, left: _Candidate, right: _Candidate) -> bool:
        def bend(pose: np.ndarray) -> int:
            first, second = np.diff(pose, axis=0)
            scale = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
            sine = float((first[0] * second[1] - first[1] * second[0]) / scale)
            return 0 if abs(sine) < 0.08 else (1 if sine > 0.0 else -1)

        if bend(left.pose_xy) != bend(right.pose_xy):
            return False
        pose_distance = float(
            np.linalg.norm(left.pose_xy - right.pose_xy, axis=1).mean()
            / self.calibration.expected_total_length_px
        )
        disagreement = float(np.mean(left.role_masks != right.role_masks))
        return pose_distance <= 0.05 and disagreement <= 0.05

    def project(
        self,
        *,
        entity_mask: np.ndarray,
        entity_available: bool,
    ) -> OrderedChainTopKFrame:
        start = perf_counter()
        raw_mask = np.asarray(entity_mask)
        if raw_mask.dtype != np.bool_:
            raise TypeError("entity_mask must have exact boolean dtype.")
        mask = np.ascontiguousarray(raw_mask)
        if mask.ndim != 2:
            raise ValueError("entity_mask must be a 2-D boolean-compatible array.")
        if tuple(mask.shape) != self._resolution:
            raise ValueError(
                "entity_mask resolution differs from support calibration."
            )
        if type(entity_available) is not bool:
            raise TypeError("entity_available must be bool.")
        if entity_available:
            candidates, diagnostics = self._enumerate(mask)
        else:
            candidates, diagnostics = (
                [None] * self.max_candidates,
                {
                    "failure": "entity_unavailable",
                    "v1_anchor_valid": False,
                    "anchor_slot_reserved": True,
                },
            )

        role_count = len(self.entity.projector.roles)
        poses = np.zeros(
            (self.max_candidates, role_count + 1, 2), dtype=np.float32
        )
        role_masks = np.zeros(
            (self.max_candidates, role_count, *mask.shape), dtype=np.bool_
        )
        valid = np.zeros(self.max_candidates, dtype=np.bool_)
        costs = np.full(
            self.max_candidates, np.finfo(np.float32).max, dtype=np.float32
        )
        weights = np.zeros(self.max_candidates, dtype=np.float32)
        confidence = np.zeros(self.max_candidates, dtype=np.float32)
        source_codes = np.zeros(self.max_candidates, dtype=np.uint8)
        active: list[tuple[int, _Candidate]] = []
        for index, candidate in enumerate(candidates):
            if candidate is None:
                continue
            active.append((index, candidate))
            poses[index] = candidate.pose_xy.astype(np.float32)
            role_masks[index] = candidate.role_masks
            valid[index] = True
            costs[index] = np.float32(candidate.cost)
            confidence[index] = np.float32(candidate.confidence)
            source_codes[index] = np.uint8(candidate.source_code)
        if active:
            logits = -np.asarray([candidate.cost for _, candidate in active])
            logits = logits / WEIGHT_TEMPERATURE
            logits -= float(logits.max())
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            for probability, (index, _) in zip(probabilities, active):
                weights[index] = np.float32(probability)
            ordered_costs = sorted(candidate.cost for _, candidate in active)
            margin = (
                float(ordered_costs[1] - ordered_costs[0])
                if len(active) > 1
                else 0.0
            )
            positive = probabilities[probabilities > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)))
            normalized_entropy = (
                float(entropy / math.log(len(active)))
                if len(active) > 1
                else 0.0
            )
            poses64 = np.stack([candidate.pose_xy for _, candidate in active])
            mean_pose = np.sum(probabilities[:, None, None] * poses64, axis=0)
            dispersion = float(
                np.sum(
                    probabilities
                    * np.linalg.norm(poses64 - mean_pose[None], axis=2).mean(axis=1)
                )
            )
            fit_uncertainty = float(
                1.0 - max(candidate.confidence for _, candidate in active)
            )
        else:
            margin = 0.0
            entropy = 0.0
            normalized_entropy = 0.0
            dispersion = 0.0
            fit_uncertainty = 1.0
        runtime_ms = max((perf_counter() - start) * 1000.0, 1e-9)
        diagnostics = {
            **diagnostics,
            "format": FORMAT,
            "protocol": PROTOCOL,
            "candidate_slots": [index for index, _ in active],
            "candidate_sources": [candidate.source for _, candidate in active],
            "candidate_coverage": [candidate.coverage for _, candidate in active],
            "candidate_render_iou": [candidate.render_iou for _, candidate in active],
            "candidate_branch_fraction": [
                candidate.branch_fraction for _, candidate in active
            ],
            "candidate_disconnected_fraction": [
                candidate.disconnected_fraction for _, candidate in active
            ],
            "candidate_pixel_fit_cost": [
                candidate.pixel_fit_cost for _, candidate in active
            ],
            "candidate_segment_fit_cost": [
                candidate.segment_fit_cost for _, candidate in active
            ],
            "candidate_length_cost": [
                candidate.length_cost for _, candidate in active
            ],
            "candidate_area_cost": [
                candidate.area_cost for _, candidate in active
            ],
            "candidate_root_cost": [
                candidate.root_cost for _, candidate in active
            ],
            "best_second_cost_margin": margin,
            "weight_entropy": entropy,
            "normalized_weight_entropy": normalized_entropy,
            "weighted_pose_dispersion_px": dispersion,
            "fit_uncertainty": fit_uncertainty,
            "weight_entropy_semantics": (
                "relative_entropy_over_retained_candidates_not_absolute_certainty_v1"
            ),
            "current_frame_only": True,
            "fail_closed": not bool(active),
        }
        return OrderedChainTopKFrame(
            role_names=self.entity.projector.roles,
            poses_xy=np.ascontiguousarray(poses),
            role_masks=np.ascontiguousarray(role_masks),
            valid=np.ascontiguousarray(valid),
            costs=np.ascontiguousarray(costs),
            weights=np.ascontiguousarray(weights),
            confidence=np.ascontiguousarray(confidence),
            source_codes=np.ascontiguousarray(source_codes),
            candidate_count=len(active),
            best_second_cost_margin=margin,
            weight_entropy=entropy,
            weighted_pose_dispersion_px=dispersion,
            fit_uncertainty=fit_uncertainty,
            normalized_weight_entropy=normalized_entropy,
            runtime_ms=float(runtime_ms),
            diagnostics=diagnostics,
        )
