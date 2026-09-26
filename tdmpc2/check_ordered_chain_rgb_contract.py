"""Contracts for current-frame RGB ordered-chain evidence and frozen DINO."""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdmpc2.perception.ordered_chain_rgb import (
    FORMAT,
    MAX_CANDIDATES,
    MAX_RENDERED_IK_CANDIDATES,
    MAX_TIP_PROPOSALS,
    PROTOCOL,
    SOURCE_CODES,
    FrozenDinoV2FeatureExtractor,
    OrderedChainRGBGenerator,
    render_role_capsules,
)
from tdmpc2.perception.ordered_chain_topk import OrderedChainTopKGenerator
from tdmpc2.perception.support_conditioned_object_graph import (
    ObjectGraphContractError,
    load_object_graph,
)


ROOT = Path(__file__).resolve().parent


class FakeDenseFeatureExtractor:
    """Color-semantic dense extractor with no learned or temporal state."""

    stateless = True
    current_frame_only = True

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_rgb: np.ndarray | None = None

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(rgb)
        self.last_rgb = image.copy()
        self.calls.append(hashlib.sha256(image.tobytes()).hexdigest())
        value = image.astype(np.float32) / 255.0
        maximum = value.max(axis=-1)
        minimum = value.min(axis=-1)
        chroma = maximum - minimum
        first = np.maximum(value[..., 0] - value[..., 2], 0.0)
        second = np.maximum(value[..., 2] - value[..., 0], 0.0)
        neutral = np.maximum(1.0 - 2.0 * chroma, 0.0)
        result = np.concatenate(
            (
                first[..., None],
                second[..., None],
                neutral[..., None],
                value,
            ),
            axis=-1,
        )
        return np.ascontiguousarray(result, dtype=np.float32)


def _ordered_graph_and_other():
    compiled = [
        load_object_graph(path)
        for path in sorted((ROOT / "object_graphs").glob("*.json"))
    ]
    ordered = [
        graph
        for graph in compiled
        if len(graph.entities) == 1
        and graph.entities[0].projector.type.startswith("ordered_chain_")
        and len(graph.entities[0].projector.roles) == 2
    ]
    other = [graph for graph in compiled if graph not in ordered]
    if len(ordered) != 1 or not other:
        raise AssertionError("The checked graph fixtures changed unexpectedly.")
    return ordered[0], other[0]


def _pose(size: int, first_angle: float, second_angle: float) -> np.ndarray:
    root = np.asarray([size * 0.5, size * 0.33], dtype=np.float64)
    joint = root + 26.0 * np.asarray(
        [math.sin(first_angle), math.cos(first_angle)], dtype=np.float64
    )
    tip = joint + 23.0 * np.asarray(
        [math.sin(second_angle), math.cos(second_angle)], dtype=np.float64
    )
    return np.stack((root, joint, tip))


def _indexed_and_rgb(
    size: int, first_angle: float, second_angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = _pose(size, first_angle, second_angle)
    roles = render_role_capsules(
        pose,
        np.asarray([3.0, 2.5], dtype=np.float64),
        (size, size),
    )
    indexed = np.zeros((size, size), dtype=np.uint8)
    indexed[roles[0]] = 1
    indexed[roles[1]] = 2
    rgb = np.full((size, size, 3), 18, dtype=np.uint8)
    rgb[roles[0]] = np.asarray([245, 24, 24], dtype=np.uint8)
    rgb[roles[1]] = np.asarray([24, 24, 245], dtype=np.uint8)
    return indexed, rgb, pose


def _support(size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    angles = (
        (-0.65, 0.25),
        (-0.45, 0.45),
        (-0.25, 0.65),
        (0.00, 0.85),
        (0.20, 1.05),
        (0.40, 1.25),
    )
    frames = [_indexed_and_rgb(size, left, right) for left, right in angles]
    return (
        np.stack([frame[1] for frame in frames]),
        np.stack([frame[0] for frame in frames]),
    )


class OrderedChainRGBContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph, cls.other_graph = _ordered_graph_and_other()
        cls.support_rgb, cls.support_masks = _support()

    def _generator(self):
        extractor = FakeDenseFeatureExtractor()
        generator = OrderedChainRGBGenerator(
            self.graph,
            self.support_rgb,
            self.support_masks,
            extractor,
        )
        return generator, extractor

    def test_contract_is_graph_driven_current_only_and_fixed_k(self) -> None:
        generator, _ = self._generator()
        metadata = generator.metadata()
        self.assertEqual(metadata["format"], FORMAT)
        self.assertEqual(metadata["protocol"], PROTOCOL)
        self.assertEqual(metadata["max_candidates"], MAX_CANDIDATES)
        self.assertEqual(metadata["maximum_tip_proposals"], MAX_TIP_PROPOSALS)
        self.assertEqual(
            metadata["maximum_rendered_ik_candidates"],
            MAX_RENDERED_IK_CANDIDATES,
        )
        self.assertTrue(metadata["current_frame_only"])
        self.assertFalse(metadata["episode_state"])
        self.assertTrue(metadata["feature_extractor_stateless_contract"])
        self.assertTrue(metadata["fixed_support_feature_replay_exact"])
        self.assertFalse(metadata["future_frames"])
        self.assertFalse(metadata["action_input"])
        self.assertFalse(metadata["reward_input"])
        self.assertFalse(metadata["episode_ground_truth_input"])
        self.assertTrue(metadata["fixed_labelled_support_masks_input"])
        self.assertFalse(metadata["task_name_dispatch"])
        self.assertTrue(metadata["relative_weights_not_calibrated"])
        source = inspect.getsource(
            __import__("tdmpc2.perception.ordered_chain_rgb", fromlist=["unused"])
        ).lower()
        forbidden_literals = (
            "acrobot-swingup",
            "cartpole-swingup",
            "reacher-visual-small",
        )
        for literal in forbidden_literals:
            self.assertNotIn(literal, source)
        with self.assertRaises(ObjectGraphContractError):
            OrderedChainRGBGenerator(
                self.other_graph,
                self.support_rgb,
                self.support_masks,
                FakeDenseFeatureExtractor(),
            )
        with self.assertRaises(ObjectGraphContractError):
            OrderedChainRGBGenerator(
                self.graph,
                self.support_rgb,
                self.support_masks,
                FakeDenseFeatureExtractor(),
                max_candidates=1,
            )
        class UndeclaredStatefulExtractor:
            def extract(self, rgb: np.ndarray) -> np.ndarray:
                return rgb.astype(np.float32)

        with self.assertRaises(ObjectGraphContractError):
            OrderedChainRGBGenerator(
                self.graph,
                self.support_rgb,
                self.support_masks,
                UndeclaredStatefulExtractor(),
            )

        class DeclaredStatefulExtractor:
            stateless = True
            current_frame_only = True

            def __init__(self) -> None:
                self.calls = 0

            def extract(self, rgb: np.ndarray) -> np.ndarray:
                self.calls += 1
                return np.full(
                    (*rgb.shape[:2], 2), self.calls % 2, dtype=np.float32
                )

        with self.assertRaises(ObjectGraphContractError):
            OrderedChainRGBGenerator(
                self.graph,
                self.support_rgb,
                self.support_masks,
                DeclaredStatefulExtractor(),
            )

    def test_slot_zero_is_exact_v1_anchor_and_frame_schema_is_frozen(self) -> None:
        generator, _ = self._generator()
        indexed, rgb, _ = _indexed_and_rgb(96, -0.35, 0.75)
        mask = indexed > 0
        frame = generator.project(
            current_rgb=rgb,
            entity_mask=mask,
            entity_available=True,
        )
        anchor = OrderedChainTopKGenerator(
            self.graph, self.support_masks, max_candidates=1
        ).project(entity_mask=mask, entity_available=True)
        np.testing.assert_array_equal(frame.valid[:1], anchor.valid)
        np.testing.assert_array_equal(frame.poses_xy[:1], anchor.poses_xy)
        np.testing.assert_allclose(frame.cost[:1], anchor.costs)
        if frame.valid[0]:
            self.assertEqual(int(frame.source_codes[0]), SOURCE_CODES["v1_anchor"])
        self.assertEqual(frame.poses_xy.shape, (2, 3, 2))
        self.assertEqual(frame.poses_xy.dtype, np.float32)
        self.assertEqual(frame.link_half_widths_px.shape, (2, 2))
        self.assertEqual(frame.link_half_widths_px.dtype, np.float32)
        self.assertEqual(frame.valid.dtype, np.bool_)
        self.assertEqual(frame.cost.dtype, np.float32)
        self.assertEqual(frame.relative_weights.dtype, np.float32)
        self.assertEqual(frame.confidence.dtype, np.float32)
        self.assertEqual(frame.source_codes.dtype, np.uint8)
        self.assertEqual(frame.rgb_evidence_score.dtype, np.float32)
        self.assertEqual(frame.mask_geometry_score.dtype, np.float32)
        self.assertEqual(frame.roi_xyxy.shape, (4,))
        self.assertEqual(frame.roi_xyxy.dtype, np.int32)
        self.assertAlmostEqual(float(frame.relative_weights.sum()), 1.0, places=6)
        np.testing.assert_array_equal(frame.costs, frame.cost)

    def test_rgb_candidate_recovers_chain_when_tracker_mask_loses_second_role(self) -> None:
        generator, _ = self._generator()
        indexed, rgb, truth = _indexed_and_rgb(96, -0.95, 0.80)
        damaged = indexed == 1
        frame = generator.project(
            current_rgb=rgb,
            entity_mask=damaged,
            entity_available=True,
        )
        self.assertTrue(frame.valid[1], frame.diagnostics)
        error = float(np.linalg.norm(frame.poses_xy[1] - truth, axis=1).mean())
        self.assertLess(error, 4.5, (error, frame.diagnostics, frame.poses_xy[1], truth))
        recovered = render_role_capsules(
            frame.poses_xy[1], frame.link_half_widths_px[1], damaged.shape
        )
        truth_roles = np.stack((indexed == 1, indexed == 2))
        role_ious = []
        for predicted, target in zip(recovered, truth_roles):
            union = np.logical_or(predicted, target).sum()
            role_ious.append(float(np.logical_and(predicted, target).sum() / union))
        self.assertGreater(min(role_ious), 0.55, role_ious)
        self.assertGreater(
            float(frame.rgb_evidence_score[1]),
            generator.metadata()["rgb_reliability_threshold"],
        )
        self.assertFalse(np.any(recovered.astype(np.uint8).sum(axis=0) > 1))
        self.assertTrue(np.any(recovered[1] & ~damaged))

    def test_spatial_shuffle_preserves_histogram_but_destroys_evidence(self) -> None:
        generator, extractor = self._generator()
        indexed, rgb, _ = _indexed_and_rgb(96, -0.95, 0.80)
        damaged = indexed == 1
        real = generator.project(
            current_rgb=rgb,
            entity_mask=damaged,
            entity_available=True,
        )
        shuffled = generator.project(
            current_rgb=rgb,
            entity_mask=damaged,
            entity_available=True,
            spatial_shuffle=True,
        )
        self.assertIsNotNone(extractor.last_rgb)
        x0, y0, x1, y1 = (int(value) for value in shuffled.roi_xyxy)
        original_pixels = rgb[y0:y1, x0:x1].reshape(-1, 3)
        shuffled_pixels = extractor.last_rgb[y0:y1, x0:x1].reshape(-1, 3)
        original_order = np.lexsort(original_pixels.T[::-1])
        shuffled_order = np.lexsort(shuffled_pixels.T[::-1])
        np.testing.assert_array_equal(
            original_pixels[original_order], shuffled_pixels[shuffled_order]
        )
        self.assertFalse(np.array_equal(original_pixels, shuffled_pixels))
        real_score = float(real.rgb_evidence_score[1]) if real.valid[1] else 0.0
        shuffled_score = (
            float(shuffled.rgb_evidence_score[1]) if shuffled.valid[1] else 0.0
        )
        self.assertGreater(real_score, shuffled_score + 0.08, (real_score, shuffled_score))

    def test_capsules_are_deterministic_nonoverlapping_and_mask_independent(self) -> None:
        pose = _pose(96, -0.7, 0.9).astype(np.float32)
        widths = np.asarray([3.2, 2.7], dtype=np.float32)
        first = render_role_capsules(pose, widths, (96, 96))
        second = render_role_capsules(pose, widths, (96, 96))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 96, 96))
        self.assertEqual(first.dtype, np.bool_)
        self.assertFalse(np.any(first[0] & first[1]))
        with self.assertRaises(ValueError):
            render_role_capsules(
                pose.astype(np.complex64) + 1j,
                widths,
                (96, 96),
            )
        with self.assertRaises(ValueError):
            render_role_capsules(
                pose,
                widths.astype(np.complex64) + 1j,
                (96, 96),
            )
        restrictive_tracker_mask = np.zeros((96, 96), dtype=np.bool_)
        restrictive_tracker_mask[30:35, 45:51] = True
        self.assertTrue(np.any(first.any(axis=0) & ~restrictive_tracker_mask))

    def test_resolution_dtype_determinism_and_current_frame_causality(self) -> None:
        generator, extractor = self._generator()
        indexed, rgb, _ = _indexed_and_rgb(96, -0.95, 0.80)
        mask = indexed > 0
        first = generator.project(
            current_rgb=rgb, entity_mask=mask, entity_available=True
        )
        second = generator.project(
            current_rgb=rgb, entity_mask=mask, entity_available=True
        )
        for name in (
            "poses_xy",
            "link_half_widths_px",
            "valid",
            "cost",
            "relative_weights",
            "confidence",
            "source_codes",
            "rgb_evidence_score",
            "mask_geometry_score",
            "roi_xyxy",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertEqual(extractor.calls[-1], extractor.calls[-2])
        unavailable = generator.project(
            current_rgb=rgb, entity_mask=mask, entity_available=False
        )
        self.assertFalse(unavailable.valid.any())
        self.assertEqual(float(unavailable.relative_weights.sum()), 0.0)
        with self.assertRaises(TypeError):
            generator.project(
                current_rgb=rgb.astype(np.float32),
                entity_mask=mask,
                entity_available=True,
            )
        with self.assertRaises(TypeError):
            generator.project(
                current_rgb=rgb,
                entity_mask=mask.astype(np.uint8),
                entity_available=True,
            )
        with self.assertRaises(ValueError):
            generator.project(
                current_rgb=rgb[:64, :64],
                entity_mask=mask[:64, :64],
                entity_available=True,
            )
        parameters = set(inspect.signature(OrderedChainRGBGenerator.project).parameters)
        self.assertEqual(
            parameters,
            {
                "self",
                "current_rgb",
                "entity_mask",
                "entity_available",
                "spatial_shuffle",
            },
        )
        forbidden = {
            "history",
            "future",
            "action",
            "reward",
            "state",
            "physics",
            "ground_truth",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_frozen_dino_local_exact_checkpoint_and_patch_tokens(self) -> None:
        import torch

        hub_source = '''import torch
dependencies = []

class TinyDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward_features(self, value):
        tokens = torch.ones((value.shape[0], 4, 3), device=value.device)
        return {"x_norm_patchtokens": tokens * self.weight + self.bias}

def dinov2_vits14_reg(pretrained=False):
    if pretrained is not False:
        raise RuntimeError("network-backed pretrained loading is forbidden")
    return TinyDino()
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "dinov2"
            repo.mkdir()
            (repo / "hubconf.py").write_text(hub_source, encoding="utf-8")
            checkpoint = root / "weights.pth"
            torch.save(
                {"weight": torch.ones(1), "bias": torch.zeros(1)}, checkpoint
            )
            extractor = FrozenDinoV2FeatureExtractor(
                repo,
                checkpoint,
                input_size=28,
                device="cpu",
            )
            features = extractor.extract(np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertEqual(features.shape, (2, 2, 3))
            self.assertEqual(features.dtype, np.float32)
            self.assertFalse(extractor._model.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in extractor._model.parameters())
            )
            metadata = extractor.metadata()
            self.assertTrue(metadata["frozen_parameters"])
            self.assertTrue(metadata["evaluation_mode"])
            self.assertTrue(metadata["torch_hub_source_local"])
            self.assertFalse(metadata["pretrained_constructor_download"])
            self.assertFalse(metadata["network_isolation_enforced"])

            bad_checkpoint = root / "bad_weights.pth"
            torch.save({"weight": torch.ones(1)}, bad_checkpoint)
            with self.assertRaises(ValueError):
                FrozenDinoV2FeatureExtractor(
                    repo,
                    bad_checkpoint,
                    input_size=28,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
