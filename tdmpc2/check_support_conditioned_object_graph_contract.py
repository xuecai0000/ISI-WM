"""Dependency-light contracts for support-conditioned object graph tokens."""

from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdmpc2.perception.support_conditioned_object_graph import (
    APPEARANCE_DIM,
    FRAME_DIM,
    ObjectGraphContractError,
    SupportConditionedObjectGraphTokenizer,
    compile_object_graph,
    load_object_graph,
    project_support_to_entities,
)
from tdmpc2.common.object_graph_preflight_snapshot import _file as snapshot_file


ROOT = Path(__file__).resolve().parent
GRAPH_ROOT = ROOT / "object_graphs"
TEMPORAL_GRAPH_ROOT = ROOT / "object_graphs_temporal_v2"


def _segment_mask(
    size: int,
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 2.4,
) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    delta = end - start
    denominator = float(delta @ delta)
    position = np.clip(
        ((xx - start[0]) * delta[0] + (yy - start[1]) * delta[1])
        / denominator,
        0.0,
        1.0,
    )
    distance_squared = (
        xx - (start[0] + position * delta[0])
    ) ** 2 + (yy - (start[1] + position * delta[1])) ** 2
    return distance_squared <= radius * radius


def _chain_frame(size: int, theta1: float, theta2: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([size * 0.5, size * 0.35], dtype=np.float64)
    joint = base + 28.0 * np.asarray([math.sin(theta1), math.cos(theta1)])
    tip = joint + 25.0 * np.asarray([math.sin(theta2), math.cos(theta2)])
    upper = _segment_mask(size, base, joint)
    lower = _segment_mask(size, joint, tip)
    # Indexed support has one owner at the joint; this preserves the exact union.
    lower &= ~upper
    indexed = np.zeros((size, size), dtype=np.uint8)
    indexed[upper] = 1
    indexed[lower] = 2
    return indexed, np.stack((base, joint, tip))


def _legacy_mask_geometry_64(mask: np.ndarray) -> np.ndarray:
    mask_float = np.asarray(mask, dtype=np.float32)
    occupancy = mask_float.reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
    yx = np.argwhere(mask_float > 0.5)
    if not len(yx):
        return np.concatenate((occupancy, np.zeros(10, np.float32))).astype(np.float32)
    y = yx[:, 0].astype(np.float64) / 63.0
    x = yx[:, 1].astype(np.float64) / 63.0
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    summary = np.concatenate((
        np.asarray([cx, cy, mask_float.mean()], dtype=np.float32),
        np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float32),
        np.asarray([(dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()], np.float32),
    ))
    return np.concatenate((occupancy, summary)).astype(np.float32)


def _legacy_frame_reference(
    features: np.ndarray,
    masks: np.ndarray,
    lost: np.ndarray,
    confidence: np.ndarray,
    mask_score: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(features).all(axis=-1)
    nonempty = masks.reshape(2, -1).any(axis=-1)
    valid = (~lost) & finite & nonempty
    queries = np.nan_to_num(features).astype(np.float64).reshape(2, 8, 256)
    appearance = np.concatenate((queries.mean(1), queries.std(1, ddof=0)), axis=-1).astype(np.float32)
    appearance[~valid] = 0.0
    geometry = np.stack([_legacy_mask_geometry_64(mask) for mask in masks])
    status = np.stack((confidence, lost.astype(np.float32), valid.astype(np.float32), mask_score), axis=-1)
    return np.ascontiguousarray(np.concatenate((appearance, geometry, status), axis=-1), np.float32)


class ObjectGraphContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = {
            path.stem: load_object_graph(path)
            for path in sorted(GRAPH_ROOT.glob("*.json"))
        }
        self.graphs["acrobot_swingup_temporal_v2"] = load_object_graph(
            TEMPORAL_GRAPH_ROOT / "acrobot_swingup_temporal_v2.json"
        )

    def test_frozen_v1_graphs_additive_v2_and_no_task_dispatch_in_core(self) -> None:
        self.assertEqual(
            set(self.graphs),
            {
                "acrobot_swingup",
                "acrobot_swingup_temporal_v2",
                "cartpole_swingup",
                "reacher_visual_small",
            },
        )
        self.assertEqual(
            {path.name for path in GRAPH_ROOT.glob("*.json")},
            {
                "acrobot_swingup.json",
                "cartpole_swingup.json",
                "reacher_visual_small.json",
            },
        )
        self.assertEqual(self.graphs["acrobot_swingup"].entity_names, ("whole_acrobot",))
        self.assertEqual(
            self.graphs["acrobot_swingup"].graph_sha256,
            "a4a01168fa0ab9ad831b6c4b5741747a407cc78ec7ba60fd49b767be98597857",
        )
        self.assertEqual(
            self.graphs["acrobot_swingup"].entities[0].projector.type,
            "ordered_chain_segments_v1",
        )
        self.assertEqual(
            self.graphs["acrobot_swingup_temporal_v2"].entities[0].projector.type,
            "ordered_chain_temporal_v2",
        )
        self.assertEqual(self.graphs["cartpole_swingup"].entity_names, ("cart", "pole"))
        self.assertEqual(
            self.graphs["reacher_visual_small"].entity_names,
            ("whole_arm", "goal"),
        )
        import tdmpc2.perception.support_conditioned_object_graph as module

        source = inspect.getsource(module).lower()
        for task_literal in (
            "acrobot-swingup",
            "cartpole-swingup",
            "reacher-visual-small",
        ):
            self.assertNotIn(task_literal, source)

    def test_cutie_entity_schema_is_additive_and_single_entity_capable(self) -> None:
        from tdmpc2.perception.cutie_oc_adapter import CutieOCConfig

        one_entity = CutieOCConfig(
            repo_path="unused",
            checkpoint_path="unused",
            role_names=("entity",),
            object_schema="generic_entity_indexed_v1",
        ).validated()
        self.assertEqual(one_entity.role_names, ("entity",))
        with self.assertRaises(ValueError):
            CutieOCConfig(
                repo_path="unused",
                checkpoint_path="unused",
                role_names=("entity",),
                object_schema="generic_indexed_v1",
            ).validated()

    def test_direct_projection_is_exact_and_renaming_invariant(self) -> None:
        graph = self.graphs["cartpole_swingup"]
        rgb = np.zeros((6, 32, 32, 3), dtype=np.uint8)
        masks = np.zeros((6, 32, 32), dtype=np.uint8)
        masks[:, 4:10, 3:15] = 1
        masks[:, 9:28, 17:20] = 2
        projected = project_support_to_entities(rgb, masks, graph)
        np.testing.assert_array_equal(projected.indexed_masks, masks)

        payload = copy.deepcopy(graph.canonical_payload)
        payload["graph_name"] = "renamed_graph"
        payload["task"] = "arbitrary-task-name"
        payload["source_roles"] = ["r0", "r1"]
        payload["semantic_roles"] = ["r0", "r1"]
        payload["tracking_entities"][0]["name"] = "e0"
        payload["tracking_entities"][0]["source_roles"] = ["r0"]
        payload["tracking_entities"][0]["projector"]["role"] = "r0"
        payload["tracking_entities"][1]["name"] = "e1"
        payload["tracking_entities"][1]["source_roles"] = ["r1"]
        payload["tracking_entities"][1]["projector"]["role"] = "r1"
        payload["relations"][0]["parent"] = "r0"
        payload["relations"][0]["child"] = "r1"
        renamed = compile_object_graph(payload)
        renamed_projected = project_support_to_entities(rgb, masks, renamed)
        np.testing.assert_array_equal(renamed_projected.indexed_masks, masks)

        tokenizer = SupportConditionedObjectGraphTokenizer(renamed, masks)
        runtime_masks = np.stack((masks[0] == 1, masks[0] == 2))
        result = tokenizer.project(
            entity_masks=runtime_masks,
            entity_features=np.zeros((2, 2048), dtype=np.float32),
            entity_lost=np.asarray([True, False]),
            entity_confidence=np.asarray([0.0, 1.0], dtype=np.float32),
            entity_mask_score=np.asarray([0.2, 0.9], dtype=np.float32),
        )
        np.testing.assert_array_equal(result.masks, runtime_masks)
        self.assertFalse(result.valid[0])
        self.assertTrue(result.valid[1])

    def test_direct_token_is_bitwise_legacy_parity_at_64(self) -> None:
        graph = self.graphs["cartpole_swingup"]
        rng = np.random.default_rng(20260901)
        rgb = np.zeros((6, 64, 64, 3), dtype=np.uint8)
        support_masks = np.zeros((6, 64, 64), dtype=np.uint8)
        support_masks[:, 4:14, 5:23] = 1
        support_masks[:, 20:58, 37:42] = 2
        tokenizer = SupportConditionedObjectGraphTokenizer(graph, support_masks)
        features = rng.standard_normal((2, 2048)).astype(np.float32)
        features[1, 17] = np.nan
        masks = np.stack((support_masks[0] == 1, support_masks[0] == 2))
        # ``lost`` and ``valid`` are intentionally not complements: the second
        # role is not reported lost, but its non-finite query makes it invalid.
        lost = np.asarray([False, False])
        confidence = np.asarray([1.0, 0.7], dtype=np.float32)
        mask_score = np.asarray([0.8, 0.2], dtype=np.float32)
        token = tokenizer.project(
            entity_masks=masks,
            entity_features=features,
            entity_lost=lost,
            entity_confidence=confidence,
            entity_mask_score=mask_score,
        )
        legacy = _legacy_frame_reference(
            features, masks, lost, confidence, mask_score
        )
        np.testing.assert_array_equal(token.descriptors, legacy)

    def test_union_chain_projection_and_descriptor_contract(self) -> None:
        size = 96
        source_frames = []
        for theta1, theta2 in (
            (-0.8, 0.4), (-0.45, 0.85), (-0.1, 1.1),
            (0.25, -0.65), (0.55, -0.25), (0.9, 0.3),
        ):
            source_frames.append(_chain_frame(size, theta1, theta2)[0])
        source = np.stack(source_frames)
        rgb = np.zeros((6, size, size, 3), dtype=np.uint8)
        graph = self.graphs["acrobot_swingup"]
        support = project_support_to_entities(rgb, source, graph)
        self.assertEqual(set(np.unique(support.indexed_masks)), {0, 1})
        np.testing.assert_array_equal(support.indexed_masks > 0, source > 0)
        tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)

        runtime_source, _ = _chain_frame(size, 0.35, 1.0)
        entity_mask = (runtime_source > 0)[None]
        feature = np.linspace(-2.0, 2.0, 2048, dtype=np.float32)[None]
        tokenizer.reset_episode()
        result = tokenizer.project(
            entity_masks=entity_mask,
            entity_features=feature,
            entity_lost=np.asarray([False]),
            entity_confidence=np.asarray([1.0], dtype=np.float32),
            entity_mask_score=np.asarray([0.9], dtype=np.float32),
        )
        self.assertEqual(result.role_names, ("upper_arm", "lower_arm"))
        self.assertEqual(result.masks.shape, (2, size, size))
        self.assertEqual(result.descriptors.shape, (2, FRAME_DIM))
        self.assertTrue(result.valid.all(), result.diagnostics)
        self.assertFalse(np.logical_and(result.masks[0], result.masks[1]).any())
        np.testing.assert_array_equal(result.masks.any(axis=0), entity_mask[0])
        np.testing.assert_array_equal(
            result.descriptors[0, :APPEARANCE_DIM],
            result.descriptors[1, :APPEARANCE_DIM],
        )
        self.assertTrue(np.isfinite(result.descriptors).all())
        self.assertTrue((result.descriptors[:, -2] == 1.0).all())

        tokenizer.reset_episode()
        repeated = tokenizer.project(
            entity_masks=entity_mask.copy(),
            entity_features=feature.copy(),
            entity_lost=np.asarray([False]),
            entity_confidence=np.asarray([1.0], dtype=np.float32),
            entity_mask_score=np.asarray([0.9], dtype=np.float32),
        )
        np.testing.assert_array_equal(repeated.masks, result.masks)
        np.testing.assert_array_equal(repeated.descriptors, result.descriptors)
        np.testing.assert_array_equal(repeated.keypoints_xy, result.keypoints_xy)

    def test_empty_or_lost_entity_fails_closed(self) -> None:
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.7)[0]
            for angle in np.linspace(-0.8, 0.8, 6)
        ])
        graph = self.graphs["acrobot_swingup"]
        tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        result = tokenizer.project(
            entity_masks=np.zeros((1, size, size), dtype=bool),
            entity_features=np.zeros((1, 2048), dtype=np.float32),
            entity_lost=np.asarray([True]),
            entity_confidence=np.asarray([0.0], dtype=np.float32),
            entity_mask_score=np.asarray([0.0], dtype=np.float32),
        )
        self.assertFalse(result.valid.any())
        self.assertTrue(result.lost.all())
        self.assertFalse(result.masks.any())
        self.assertTrue((result.descriptors[:, :APPEARANCE_DIM] == 0).all())

        dense = tokenizer.project(
            entity_masks=np.ones((1, size, size), dtype=bool),
            entity_features=np.zeros((1, 2048), dtype=np.float32),
            entity_lost=np.asarray([False]),
            entity_confidence=np.asarray([1.0], dtype=np.float32),
            entity_mask_score=np.asarray([1.0], dtype=np.float32),
        )
        self.assertFalse(dense.valid.any())
        self.assertFalse(dense.masks.any())
        self.assertEqual(
            dense.diagnostics["whole_acrobot"]["failure_reasons"],
            ["entity_area_implausible"],
        )
        self.assertTrue(dense.diagnostics["whole_acrobot"]["early_rejection"])

    def test_ambiguous_folded_chain_is_rejected(self) -> None:
        size = 128
        source = np.stack([
            _chain_frame(size, angle, angle + 0.7)[0]
            for angle in np.linspace(-0.8, 0.8, 6)
        ])
        tokenizer = SupportConditionedObjectGraphTokenizer(
            self.graphs["acrobot_swingup"], source
        )
        folded, _ = _chain_frame(size, -2.89, 0.13)
        result = tokenizer.project(
            entity_masks=(folded > 0)[None],
            entity_features=np.zeros((1, 2048), dtype=np.float32),
            entity_lost=np.asarray([False]),
            entity_confidence=np.asarray([1.0], dtype=np.float32),
            entity_mask_score=np.asarray([1.0], dtype=np.float32),
        )
        self.assertFalse(result.valid.any(), result.diagnostics)
        self.assertFalse(result.masks.any())
        self.assertTrue(
            result.diagnostics["whole_acrobot"]["failure_reasons"]
        )

    def test_runtime_is_causal_and_does_not_mutate_inputs_or_rng(self) -> None:
        graph = self.graphs["acrobot_swingup"]
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.65)[0]
            for angle in np.linspace(-0.7, 0.7, 6)
        ])
        runtime_source, _ = _chain_frame(size, 0.2, 0.95)
        masks = np.ascontiguousarray((runtime_source > 0)[None])
        features = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)[None]
        lost = np.asarray([False])
        confidence = np.asarray([0.8], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        originals = tuple(
            np.array(value, copy=True)
            for value in (source, masks, features, lost, confidence, score)
        )
        public_parameters = set(
            inspect.signature(SupportConditionedObjectGraphTokenizer.project).parameters
        )
        self.assertEqual(
            public_parameters,
            {
                "self",
                "entity_masks",
                "entity_features",
                "entity_lost",
                "entity_confidence",
                "entity_mask_score",
            },
        )

        saved_global_rng = np.random.get_state()
        try:
            np.random.seed(20260901)
            rng_before = np.random.get_state()
            tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
            tokenizer.reset_episode()
            first = tokenizer.project(
                entity_masks=masks,
                entity_features=features,
                entity_lost=lost,
                entity_confidence=confidence,
                entity_mask_score=score,
            )
            rng_after = np.random.get_state()
        finally:
            np.random.set_state(saved_global_rng)
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        for value, original in zip(
            (source, masks, features, lost, confidence, score), originals
        ):
            np.testing.assert_array_equal(value, original)

        tokenizer.reset_episode()
        repeated = tokenizer.project(
            entity_masks=masks,
            entity_features=features,
            entity_lost=lost,
            entity_confidence=confidence,
            entity_mask_score=score,
        )
        np.testing.assert_array_equal(first.masks, repeated.masks)
        np.testing.assert_array_equal(first.descriptors, repeated.descriptors)

    def test_temporal_v2_rotating_chain_keeps_role_identity_and_partition(self) -> None:
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.65)[0]
            for angle in np.linspace(-0.7, 0.7, 6)
        ])
        graph = self.graphs["acrobot_swingup_temporal_v2"]
        tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        previous_pose = None
        for angle in np.linspace(-0.7, 0.7, 15):
            indexed, _ = _chain_frame(size, float(angle), float(angle + 0.65))
            result = tokenizer.project_temporal_geometry(
                entity_masks=(indexed > 0)[None],
                entity_lost=np.asarray([False]),
            )
            self.assertTrue(result.valid.all(), result.diagnostics)
            np.testing.assert_array_equal(result.masks.any(axis=0), indexed > 0)
            self.assertFalse(np.logical_and(result.masks[0], result.masks[1]).any())

            def iou(left: np.ndarray, right: np.ndarray) -> float:
                return float(np.logical_and(left, right).sum()) / max(
                    int(np.logical_or(left, right).sum()), 1
                )

            correct = iou(result.masks[0], indexed == 1) + iou(
                result.masks[1], indexed == 2
            )
            swapped = iou(result.masks[0], indexed == 2) + iou(
                result.masks[1], indexed == 1
            )
            self.assertGreater(correct, swapped + 0.5)
            pose = np.vstack((
                result.keypoints_xy[0, 0],
                result.keypoints_xy[:, 1],
            ))
            if previous_pose is not None:
                self.assertLess(
                    float(np.linalg.norm(pose - previous_pose, axis=1).max()),
                    12.0,
                )
            previous_pose = pose
            diagnostics = result.diagnostics["whole_acrobot"]
            self.assertLessEqual(diagnostics["candidate_count"], 12)
            self.assertTrue(diagnostics["current_mask_partition_exact"])

    def test_temporal_v2_folded_ambiguity_fails_closed_then_reacquires(self) -> None:
        size = 128
        source = np.stack([
            _chain_frame(size, angle, angle + 0.7)[0]
            for angle in np.linspace(-0.8, 0.8, 6)
        ])
        tokenizer = SupportConditionedObjectGraphTokenizer(
            self.graphs["acrobot_swingup_temporal_v2"], source
        )
        for theta1, theta2 in ((-0.5, 0.2), (-0.4, 0.3), (-0.3, 0.4)):
            indexed, _ = _chain_frame(size, theta1, theta2)
            warm = tokenizer.project_temporal_geometry(
                entity_masks=(indexed > 0)[None],
                entity_lost=np.asarray([False]),
            )
            self.assertTrue(warm.valid.all(), warm.diagnostics)

        folded, _ = _chain_frame(size, -2.89, 0.13)
        rejected = tokenizer.project_temporal_geometry(
            entity_masks=(folded > 0)[None],
            entity_lost=np.asarray([False]),
        )
        self.assertFalse(rejected.valid.any(), rejected.diagnostics)
        self.assertFalse(rejected.masks.any())
        self.assertTrue((rejected.keypoints_xy == 0).all())
        rejected_diagnostics = rejected.diagnostics["whole_acrobot"]
        self.assertTrue(rejected_diagnostics["fail_closed"])
        self.assertEqual(rejected_diagnostics["invalid_age"], 1)

        recovered_indexed, _ = _chain_frame(size, -0.1, 0.6)
        recovered = tokenizer.project_temporal_geometry(
            entity_masks=(recovered_indexed > 0)[None],
            entity_lost=np.asarray([False]),
        )
        self.assertTrue(recovered.valid.all(), recovered.diagnostics)
        np.testing.assert_array_equal(
            recovered.masks.any(axis=0), recovered_indexed > 0
        )
        self.assertEqual(
            recovered.diagnostics["whole_acrobot"]["invalid_age"], 0
        )

    def test_temporal_v2_rejects_excessively_branched_current_mask(self) -> None:
        size = 128
        source = np.stack([
            _chain_frame(size, angle, angle + 0.7)[0]
            for angle in np.linspace(-2.4, 2.4, 13)
        ])
        tokenizer = SupportConditionedObjectGraphTokenizer(
            self.graphs["acrobot_swingup_temporal_v2"], source
        )
        # Establish a stable causal state at the clean pose.  The corruption
        # below is then a strict test that temporal prediction cannot bypass
        # the current-frame topology gate.
        for theta1, theta2 in ((-0.25, 0.45),) * 3:
            indexed, _ = _chain_frame(size, theta1, theta2)
            accepted = tokenizer.project_temporal_geometry(
                entity_masks=(indexed > 0)[None],
                entity_lost=np.asarray([False]),
            )
            self.assertTrue(accepted.valid.all(), accepted.diagnostics)

        indexed, points = _chain_frame(size, -0.25, 0.45)
        branch_centre = points[1] + 9.0 * np.asarray(
            [math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)]
        )
        yy, xx = np.indices((size, size), dtype=np.float64)
        branch = (
            (xx - branch_centre[0]) ** 2 + (yy - branch_centre[1]) ** 2
            <= 10.0**2
        )
        corrupted = (indexed > 0) | branch
        rejected = tokenizer.project_temporal_geometry(
            entity_masks=corrupted[None],
            entity_lost=np.asarray([False]),
        )
        self.assertFalse(rejected.valid.any(), rejected.diagnostics)
        self.assertFalse(rejected.masks.any())
        diagnostics = rejected.diagnostics["whole_acrobot"]
        self.assertTrue(diagnostics["fail_closed"])
        self.assertEqual(diagnostics["candidate_count"], 0)
        observed_branch = max(
            record.get("maximum_branch_fraction", 0.0)
            for record in diagnostics["topology"]
        )
        self.assertGreater(observed_branch, 0.35)

    def test_temporal_v2_reset_clears_history_and_future_suffix_is_causal(self) -> None:
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.65)[0]
            for angle in np.linspace(-0.7, 0.7, 6)
        ])
        graph = self.graphs["acrobot_swingup_temporal_v2"]
        target, _ = _chain_frame(size, 0.25, 0.9)
        history_tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        for theta1, theta2 in ((-0.6, 0.05), (-0.45, 0.2), (-0.3, 0.35)):
            indexed, _ = _chain_frame(size, theta1, theta2)
            history_tokenizer.project_temporal_geometry(
                entity_masks=(indexed > 0)[None],
                entity_lost=np.asarray([False]),
            )
        history_tokenizer.reset_episode()
        after_reset = history_tokenizer.project_temporal_geometry(
            entity_masks=(target > 0)[None], entity_lost=np.asarray([False])
        )
        fresh = SupportConditionedObjectGraphTokenizer(graph, source)
        fresh_target = fresh.project_temporal_geometry(
            entity_masks=(target > 0)[None], entity_lost=np.asarray([False])
        )
        np.testing.assert_array_equal(after_reset.masks, fresh_target.masks)
        np.testing.assert_array_equal(after_reset.keypoints_xy, fresh_target.keypoints_xy)
        np.testing.assert_array_equal(after_reset.valid, fresh_target.valid)
        self.assertFalse(
            after_reset.diagnostics["whole_acrobot"]["state_initialized_before"]
        )

        left = SupportConditionedObjectGraphTokenizer(graph, source)
        right = SupportConditionedObjectGraphTokenizer(graph, source)
        prefix_outputs = []
        for theta1, theta2 in ((-0.5, 0.15), (-0.35, 0.3), (-0.2, 0.45)):
            indexed, _ = _chain_frame(size, theta1, theta2)
            first = left.project_temporal_geometry(
                entity_masks=(indexed > 0)[None], entity_lost=np.asarray([False])
            )
            second = right.project_temporal_geometry(
                entity_masks=(indexed > 0)[None], entity_lost=np.asarray([False])
            )
            prefix_outputs.append((first, second))
        suffix_left, _ = _chain_frame(size, 0.6, 1.25)
        suffix_right, _ = _chain_frame(size, -0.9, -0.25)
        left.project_temporal_geometry(
            entity_masks=(suffix_left > 0)[None], entity_lost=np.asarray([False])
        )
        right.project_temporal_geometry(
            entity_masks=(suffix_right > 0)[None], entity_lost=np.asarray([False])
        )
        for first, second in prefix_outputs:
            np.testing.assert_array_equal(first.masks, second.masks)
            np.testing.assert_array_equal(first.keypoints_xy, second.keypoints_xy)
            np.testing.assert_array_equal(first.valid, second.valid)

    def test_mask_only_geometry_matches_realtime_temporal_step_without_features(self) -> None:
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.65)[0]
            for angle in np.linspace(-0.7, 0.7, 6)
        ])
        graph = self.graphs["acrobot_swingup_temporal_v2"]
        geometry_tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        realtime_tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        feature = np.zeros((1, 2048), dtype=np.float32)
        for angle in np.linspace(-0.5, 0.5, 7):
            indexed, _ = _chain_frame(size, float(angle), float(angle + 0.65))
            entity_masks = (indexed > 0)[None]
            geometry = geometry_tokenizer.project_temporal_geometry(
                entity_masks=entity_masks,
                entity_lost=np.asarray([False]),
            )
            realtime = realtime_tokenizer.project(
                entity_masks=entity_masks,
                entity_features=feature,
                entity_lost=np.asarray([False]),
                entity_confidence=np.asarray([1.0], dtype=np.float32),
                entity_mask_score=np.asarray([1.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(geometry.masks, realtime.masks)
            np.testing.assert_array_equal(geometry.keypoints_xy, realtime.keypoints_xy)
            np.testing.assert_array_equal(geometry.valid, realtime.valid)
            np.testing.assert_array_equal(
                geometry.projector_confidence, realtime.confidence
            )
        self.assertFalse(hasattr(geometry, "descriptors"))

        direct = self.graphs["cartpole_swingup"]
        direct_support = np.zeros((2, 32, 32), dtype=np.uint8)
        direct_support[:, 3:8, 3:12] = 1
        direct_support[:, 10:28, 18:21] = 2
        direct_geometry = SupportConditionedObjectGraphTokenizer(
            direct, direct_support
        ).project_temporal_geometry(
            entity_masks=np.stack((
                direct_support[0] == 1,
                direct_support[0] == 2,
            )),
            entity_lost=np.asarray([False, False]),
        )
        np.testing.assert_array_equal(
            direct_geometry.masks,
            np.stack((direct_support[0] == 1, direct_support[0] == 2)),
        )
        self.assertTrue(direct_geometry.valid.all())

    def test_temporal_v2_atomic_fail_closed_status_and_speed_budget(self) -> None:
        size = 96
        source = np.stack([
            _chain_frame(size, angle, angle + 0.65)[0]
            for angle in np.linspace(-0.7, 0.7, 6)
        ])
        graph = self.graphs["acrobot_swingup_temporal_v2"]
        indexed, _ = _chain_frame(size, 0.1, 0.75)
        failed = SupportConditionedObjectGraphTokenizer(graph, source).project(
            entity_masks=(indexed > 0)[None],
            entity_features=np.zeros((1, 2048), dtype=np.float32),
            entity_lost=np.asarray([True]),
            entity_confidence=np.asarray([1.0], dtype=np.float32),
            entity_mask_score=np.asarray([1.0], dtype=np.float32),
        )
        self.assertFalse(failed.valid.any())
        self.assertFalse(failed.masks.any())
        self.assertTrue((failed.keypoints_xy == 0).all())
        self.assertTrue((failed.descriptors[:, :APPEARANCE_DIM + 74] == 0).all())
        np.testing.assert_array_equal(
            failed.descriptors[:, -4:],
            np.asarray([[0.0, 1.0, 0.0, 0.0]] * 2, dtype=np.float32),
        )

        tokenizer = SupportConditionedObjectGraphTokenizer(graph, source)
        runtimes = []
        for angle in np.linspace(-0.65, 0.65, 18):
            runtime_indexed, _ = _chain_frame(
                size, float(angle), float(angle + 0.65)
            )
            result = tokenizer.project_temporal_geometry(
                entity_masks=(runtime_indexed > 0)[None],
                entity_lost=np.asarray([False]),
            )
            self.assertTrue(result.valid.all(), result.diagnostics)
            self.assertLessEqual(
                result.diagnostics["whole_acrobot"]["candidate_count"], 12
            )
            self.assertLessEqual(
                result.diagnostics["whole_acrobot"]["scored_candidate_count"], 25
            )
            runtimes.append(result.runtime_ms)
        # This is a dependency-light CPU smoke budget.  The server preflight
        # keeps the stricter mean<=5ms and p95<=10ms release gate.
        self.assertLess(float(np.mean(runtimes)), 10.0)
        self.assertLess(float(np.quantile(runtimes, 0.95)), 20.0)

        geometry_parameters = set(
            inspect.signature(
                SupportConditionedObjectGraphTokenizer.project_temporal_geometry
            ).parameters
        )
        self.assertEqual(
            geometry_parameters,
            {"self", "entity_masks", "entity_lost"},
        )
        metadata = tokenizer.metadata()
        self.assertEqual(
            metadata["geometry_replay_entrypoint"], "project_temporal_geometry"
        )
        self.assertFalse(metadata["geometry_replay_has_appearance_or_descriptors"])

    def test_runner_is_controller_free_and_locks_gt_before_backend(self) -> None:
        runner = (
            ROOT / "tools" / "run_object_graph_tokenizer_preflight.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tdmpc2/train.py", runner)
        self.assertNotIn("chmod -R", runner)
        source_claim = runner.index('if ! mkdir -- "$SOURCE_LOCK"')
        lock = runner.index('chmod 000 -- "$SCORING_ROOT"')
        backend = runner.index('BACKEND_START backend=object_graph_cutie')
        restore = runner.index(
            'chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"', backend
        )
        self.assertLess(source_claim, lock)
        self.assertLess(lock, backend)
        self.assertLess(backend, restore)
        release = runner.index('rmdir -- "$SOURCE_LOCK"', restore)
        promotion = runner.index('mv -T -- "$STAGE" "$BASE"', restore)
        self.assertLess(restore, release)
        self.assertLess(promotion, release)
        self.assertIn(
            'LOCK_PARENT="$(dirname -- "$SOURCE_ROOT_CANONICAL")/'
            '.object_graph_tokenizer_locks"',
            runner,
        )
        self.assertNotIn(
            'LOCK_PARENT="$REPO_ROOT/logs/_diagnostic/', runner
        )
        self.assertIn(
            'export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tdmpc2"', runner
        )
        self.assertNotIn('${PYTHONPATH:+:$PYTHONPATH}', runner)
        self.assertIn(
            'PY_CORE and PY_CUTIE must resolve to the same frozen environment',
            runner,
        )
        self.assertNotIn("sed '/^[[:space:]]*$/d' || true", runner)
        self.assertIn('"controller_training_authorized":False', runner)

        snapshot = (
            ROOT / "common" / "object_graph_preflight_snapshot.py"
        ).read_text(encoding="utf-8")
        for required in (
            "validate_dataset_files(",
            "_validate_decoded_dataset_artifacts(",
            "validate_backend_inputs(",
            "_validate_backend_manifest(",
            '"baseline_predictions_rehashed": True',
        ):
            self.assertIn(required, snapshot)

    def test_snapshot_rejects_escape_and_symlink_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "immutable"
            root.mkdir()
            inside = root / "inside.json"
            inside.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                snapshot_file(
                    inside, "inside fixture", containment_root=root
                )["bytes"],
                inside.stat().st_size,
            )
            outside = Path(directory) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                snapshot_file(
                    outside, "escape fixture", containment_root=root
                )

            real = root / "real"
            real.mkdir()
            linked_file = real / "linked.json"
            linked_file.write_text("{}\n", encoding="utf-8")
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                return
            with self.assertRaises(ValueError):
                snapshot_file(
                    link / "linked.json",
                    "symlink ancestor fixture",
                    containment_root=root,
                )

    def test_graph_and_support_tampering_fail_closed(self) -> None:
        base = self.graphs["acrobot_swingup"].canonical_payload
        for mutate in (
            lambda value: value["tracking_entities"][0]["source_roles"].reverse(),
            lambda value: value["semantic_roles"].append("extra"),
            lambda value: value["tracking_entities"][0]["projector"].update(
                {"maximum_branch_fraction": 2.0}
            ),
            lambda value: value["descriptor"].update({"frame_dim": 589}),
        ):
            changed = copy.deepcopy(base)
            mutate(changed)
            with self.assertRaises(ObjectGraphContractError):
                compile_object_graph(changed)
        rgb = np.zeros((6, 32, 32, 3), dtype=np.uint8)
        masks = np.zeros((6, 32, 32), dtype=np.uint8)
        masks[:, 2:5, 2:5] = 1
        masks[:5, 8:12, 8:12] = 2
        with self.assertRaises(ObjectGraphContractError):
            project_support_to_entities(rgb, masks, self.graphs["acrobot_swingup"])

        temporal = copy.deepcopy(
            self.graphs["acrobot_swingup_temporal_v2"].canonical_payload
        )
        temporal["source_roles"].append("third_link")
        temporal["semantic_roles"].append("third_link")
        temporal["tracking_entities"][0]["source_roles"].append("third_link")
        temporal["tracking_entities"][0]["projector"]["ordered_roles"].append(
            "third_link"
        )
        temporal["relations"].append(
            {"type": "revolute_joint_v1", "parent": "lower_arm", "child": "third_link"}
        )
        with self.assertRaisesRegex(ObjectGraphContractError, "exactly two"):
            compile_object_graph(temporal)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    if not program.result.wasSuccessful():
        raise SystemExit(1)
    print("SUPPORT_CONDITIONED_OBJECT_GRAPH_CONTRACT_OK", flush=True)
