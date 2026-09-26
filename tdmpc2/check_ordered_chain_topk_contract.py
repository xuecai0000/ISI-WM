"""Dependency-light contracts for current-frame ordered-chain top-K hypotheses."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

import numpy as np

from tdmpc2.perception.ordered_chain_topk import (
    FORMAT,
    MAX_CANDIDATES,
    PROTOCOL,
    SOURCE_CODES,
    OrderedChainTopKGenerator,
    role_masks_from_pose,
)
from tdmpc2.perception.support_conditioned_object_graph import (
    QUERY_FEATURE_DIM,
    ObjectGraphContractError,
    SupportConditionedObjectGraphTokenizer,
    load_object_graph,
)


ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "object_graphs" / "acrobot_swingup.json"


def _segment_mask(
    size: int,
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 2.4,
) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    delta = end - start
    denominator = max(float(delta @ delta), 1e-12)
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


def _chain_frame(
    size: int, theta1: float, theta2: float
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([size * 0.5, size * 0.35], dtype=np.float64)
    joint = base + 28.0 * np.asarray([math.sin(theta1), math.cos(theta1)])
    tip = joint + 25.0 * np.asarray([math.sin(theta2), math.cos(theta2)])
    upper = _segment_mask(size, base, joint)
    lower = _segment_mask(size, joint, tip)
    lower &= ~upper
    indexed = np.zeros((size, size), dtype=np.uint8)
    indexed[upper] = 1
    indexed[lower] = 2
    return indexed, np.stack((base, joint, tip))


def _support(size: int = 128) -> np.ndarray:
    angles = [(-0.6, 0.3), (-0.4, 0.5), (-0.2, 0.7), (0.0, 0.9), (0.2, 1.1), (0.4, 1.3)]
    return np.stack([_chain_frame(size, left, right)[0] for left, right in angles])


class OrderedChainTopKContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = load_object_graph(GRAPH)
        self.support = _support()

    def test_metadata_is_structural_current_only_and_task_neutral(self) -> None:
        generator = OrderedChainTopKGenerator(self.graph, self.support)
        metadata = generator.metadata()
        self.assertEqual(metadata["format"], FORMAT)
        self.assertEqual(metadata["protocol"], PROTOCOL)
        self.assertFalse(metadata["episode_state"])
        self.assertFalse(metadata["future_frames"])
        self.assertFalse(metadata["task_name_dispatch"])
        self.assertFalse(metadata["ground_truth_input"])
        self.assertEqual(metadata["support_resolution"], [128, 128])
        source = inspect.getsource(
            __import__(
                "tdmpc2.perception.ordered_chain_topk", fromlist=["unused"]
            )
        ).lower()
        for literal in (
            "acrobot-swingup",
            "cartpole-swingup",
            "reacher-visual-small",
        ):
            self.assertNotIn(literal, source)

    def test_rank_zero_is_exact_published_v1_current_frame_anchor(self) -> None:
        indexed, _ = _chain_frame(128, -0.3, 0.7)
        entity = indexed > 0
        generator = OrderedChainTopKGenerator(self.graph, self.support)
        frame = generator.project(entity_mask=entity, entity_available=True)
        tokenizer = SupportConditionedObjectGraphTokenizer(self.graph, self.support)
        legacy = tokenizer.project(
            entity_masks=entity[None],
            entity_features=np.ones((1, QUERY_FEATURE_DIM), dtype=np.float32),
            entity_lost=np.zeros(1, dtype=np.bool_),
            entity_confidence=np.ones(1, dtype=np.float32),
            entity_mask_score=np.ones(1, dtype=np.float32),
        )
        self.assertTrue(frame.valid[0])
        self.assertEqual(int(frame.source_codes[0]), SOURCE_CODES["v1_anchor"])
        np.testing.assert_array_equal(frame.role_masks[0], legacy.masks)

    def test_k_one_two_four_are_deterministic_prefixes(self) -> None:
        indexed, _ = _chain_frame(128, -1.2, 1.2)
        mask = indexed > 0
        outputs = [
            OrderedChainTopKGenerator(
                self.graph, self.support, max_candidates=value
            ).project(entity_mask=mask, entity_available=True)
            for value in (1, 2, 4)
        ]
        for smaller, larger in zip(outputs, outputs[1:]):
            count = smaller.poses_xy.shape[0]
            np.testing.assert_array_equal(smaller.poses_xy, larger.poses_xy[:count])
            np.testing.assert_array_equal(
                smaller.role_masks, larger.role_masks[:count]
            )
            np.testing.assert_array_equal(smaller.valid, larger.valid[:count])
            np.testing.assert_array_equal(
                smaller.source_codes, larger.source_codes[:count]
            )
            np.testing.assert_allclose(smaller.costs, larger.costs[:count])
        repeat = OrderedChainTopKGenerator(self.graph, self.support).project(
            entity_mask=mask, entity_available=True
        )
        np.testing.assert_array_equal(outputs[-1].poses_xy, repeat.poses_xy)
        np.testing.assert_array_equal(outputs[-1].role_masks, repeat.role_masks)
        np.testing.assert_array_equal(outputs[-1].source_codes, repeat.source_codes)
        np.testing.assert_allclose(outputs[-1].costs, repeat.costs)

    def test_candidates_partition_current_mask_and_padding_is_safe(self) -> None:
        generator = OrderedChainTopKGenerator(self.graph, self.support)
        alternative_count = 0
        # Exercise near-straight, folded, and both bend orientations.  The
        # alternative mask must be reconstructible byte-for-byte from the
        # exact float32 pose that the compact backend seals.  Slot zero is not
        # subject to this assertion because its semantic mask is exact v1.
        angles = (-2.8, -1.8, -0.9, 0.0, 0.9, 1.8, 2.8)
        for theta1 in angles:
            for theta2 in angles:
                indexed, _ = _chain_frame(128, theta1, theta2)
                mask = indexed > 0
                frame = generator.project(
                    entity_mask=mask, entity_available=True
                )
                self.assertEqual(frame.candidate_count, int(frame.valid.sum()))
                for index in np.flatnonzero(frame.valid):
                    candidate = frame.role_masks[index]
                    self.assertFalse(
                        np.any(candidate.astype(np.uint8).sum(axis=0) > 1)
                    )
                    np.testing.assert_array_equal(candidate.any(axis=0), mask)
                    if index > 0:
                        alternative_count += 1
                        np.testing.assert_array_equal(
                            candidate,
                            role_masks_from_pose(mask, frame.poses_xy[index]),
                        )
                self.assertTrue(
                    np.all(
                        frame.costs[~frame.valid] == np.finfo(np.float32).max
                    )
                )
                self.assertTrue(np.all(frame.weights[~frame.valid] == 0.0))
                self.assertTrue(np.all(frame.source_codes[~frame.valid] == 0))
                expected_weight = 1.0 if frame.candidate_count else 0.0
                self.assertAlmostEqual(
                    float(frame.weights.sum()), expected_weight, places=6
                )
        self.assertGreater(alternative_count, 20)

    def test_top_two_adds_a_distinct_current_frame_mode(self) -> None:
        indexed, truth = _chain_frame(128, -1.2, 1.2)
        frame = OrderedChainTopKGenerator(self.graph, self.support).project(
            entity_mask=indexed > 0, entity_available=True
        )
        self.assertGreaterEqual(frame.candidate_count, 2)
        errors = np.asarray(
            [
                np.linalg.norm(frame.poses_xy[index] - truth, axis=1).mean()
                for index in range(2)
            ]
        )
        self.assertLessEqual(float(errors.min()), float(errors[0]) + 1e-6)
        self.assertGreater(
            float(np.linalg.norm(frame.poses_xy[0] - frame.poses_xy[1], axis=1).mean()),
            0.05 * 53.0,
        )

    def test_folded_chain_has_a_physical_candidate_instead_of_empty_output(self) -> None:
        for theta1, theta2, maximum_error in (
            (-2.89, 0.13, 5.0),
            (-0.6, math.pi - 0.6, 3.0),
        ):
            indexed, truth = _chain_frame(128, theta1, theta2)
            frame = OrderedChainTopKGenerator(self.graph, self.support).project(
                entity_mask=indexed > 0, entity_available=True
            )
            self.assertGreaterEqual(frame.candidate_count, 1)
            best = min(
                float(
                    np.linalg.norm(frame.poses_xy[index] - truth, axis=1).mean()
                )
                for index in np.flatnonzero(frame.valid)
            )
            self.assertLess(best, maximum_error)

            baseline = OrderedChainTopKGenerator(
                self.graph, self.support, max_candidates=1
            ).project(entity_mask=indexed > 0, entity_available=True)
            expanded = OrderedChainTopKGenerator(
                self.graph, self.support, max_candidates=2
            ).project(entity_mask=indexed > 0, entity_available=True)
            np.testing.assert_array_equal(baseline.poses_xy, expanded.poses_xy[:1])
            np.testing.assert_array_equal(baseline.valid, expanded.valid[:1])
            np.testing.assert_array_equal(
                baseline.source_codes, expanded.source_codes[:1]
            )
            if not baseline.valid[0]:
                self.assertFalse(expanded.valid[0])
                self.assertTrue(expanded.valid[1])

    def test_unavailable_empty_wrong_resolution_and_wrong_structure_fail_closed(self) -> None:
        generator = OrderedChainTopKGenerator(self.graph, self.support)
        unavailable = generator.project(
            entity_mask=np.ones((128, 128), dtype=np.bool_),
            entity_available=False,
        )
        self.assertEqual(unavailable.candidate_count, 0)
        self.assertFalse(unavailable.valid.any())
        empty = generator.project(
            entity_mask=np.zeros((128, 128), dtype=np.bool_),
            entity_available=True,
        )
        self.assertEqual(empty.candidate_count, 0)
        with self.assertRaises(ValueError):
            generator.project(
                entity_mask=np.zeros((64, 64), dtype=np.bool_),
                entity_available=True,
            )
        with self.assertRaises(TypeError):
            generator.project(
                entity_mask=np.zeros((128, 128), dtype=np.uint8),
                entity_available=True,
            )
        direct = load_object_graph(
            ROOT / "object_graphs" / "cartpole_swingup.json"
        )
        with self.assertRaises(ObjectGraphContractError):
            OrderedChainTopKGenerator(direct, self.support)

    def test_public_projection_api_cannot_receive_privileged_or_temporal_inputs(self) -> None:
        parameters = set(inspect.signature(OrderedChainTopKGenerator.project).parameters)
        self.assertEqual(parameters, {"self", "entity_mask", "entity_available"})
        forbidden = {
            "gt",
            "ground_truth",
            "action",
            "reward",
            "physics",
            "rgb",
            "future",
            "history",
            "state",
            "features",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
