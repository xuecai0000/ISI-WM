"""Dependency-light contracts for the RGB ordered-chain candidate preflight."""

from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
import unittest

import numpy as np

from tdmpc2.common.object_graph_rgb_snapshot import FORMAT as SNAPSHOT_FORMAT
from tdmpc2.common.unified_vos import CONDITIONS
from tdmpc2.perception.ordered_chain_rgb import (
    FORMAT as GENERATOR_FORMAT,
    MAX_CANDIDATES,
    PROTOCOL as GENERATOR_PROTOCOL,
    SOURCE_CODES,
    render_role_capsules,
)
from tdmpc2.tools.aggregate_object_graph_rgb_candidates import (
    ISOLATION_FORMAT,
    SUMMARY_FORMAT,
    _development_gate,
    _frame_score_rows,
    _rgb_array_schema,
    _validate_decoded_rgb_arrays,
    _validate_isolation_gate,
    _validate_rank_zero_exact,
    build_parser,
)
from tdmpc2.tools.replay_object_graph_rgb_candidates import (
    ARMS,
    BACKEND,
    DINO_EXTRACTOR_METADATA_KEYS,
    FORMAT as BACKEND_FORMAT,
    OUTPUT_ARRAY_KEYS,
    PROTOCOL as BACKEND_PROTOCOL,
)


ROOT = Path(__file__).resolve().parent
RUNNER_PATH = ROOT / "tools" / "run_object_graph_rgb_candidate_preflight.sh"
BACKEND_PATH = ROOT / "tools" / "replay_object_graph_rgb_candidates.py"
AGGREGATOR_PATH = ROOT / "tools" / "aggregate_object_graph_rgb_candidates.py"


def _valid_arrays(frames: int = 2, resolution: int = 64) -> dict[str, np.ndarray]:
    schema = _rgb_array_schema(frames=frames)
    arrays = {
        name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()
    }
    arrays["candidate_valid"][:] = True
    arrays["poses_xy"][:] = np.asarray(
        [[[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]]], dtype=np.float32
    )
    arrays["link_half_widths_px"][:] = 3.0
    arrays["candidate_cost"][:] = 1.0
    arrays["candidate_weight"][:] = 0.5
    arrays["candidate_confidence"][:] = 0.75
    arrays["rgb_evidence_score"][:] = 0.75
    arrays["mask_geometry_score"][:] = 0.75
    arrays["candidate_source_code"][:, 0] = SOURCE_CODES["v1_anchor"]
    arrays["candidate_source_code"][:, 1] = SOURCE_CODES["rgb_ik_positive"]
    arrays["roi_xyxy"][:] = np.asarray([0, 0, resolution - 1, resolution - 1])
    arrays["parser_runtime_ms"][:] = 2.0
    return arrays


def _gate_fixture() -> tuple[dict, dict, dict]:
    metrics: dict = {
        arm: {"rgb_capsule": {}, "rgb_partition": {}} for arm in ARMS
    }
    baseline: dict = {}
    comparisons: dict = {}
    for condition in CONDITIONS:
        real_cell = {
            "availability_rate": 1.0,
            "oracle_set_success_at_0_5": 0.99,
            "failure_burst_at_0_5": {
                "global_max_with_episode_resets": 5,
                "episode_p95": 5.0,
                "per_episode": [5],
            },
        }
        shuffle_cell = {
            "availability_rate": 1.0,
            "oracle_set_success_at_0_5": 0.95,
            "failure_burst_at_0_5": {
                "global_max_with_episode_resets": 8,
                "episode_p95": 8.0,
                "per_episode": [8],
            },
        }
        metrics["real_rgb"]["rgb_capsule"][condition] = {
            "prefixes": {"2": real_cell},
            "parser_runtime": {"mean_ms": 2.0, "p95_ms": 3.0},
            "published_v1_cutie_plus_parser_runtime": {
                "mean_ms": 9.0,
                "p95_ms": 11.0,
            },
        }
        metrics["spatial_shuffle"]["rgb_capsule"][condition] = {
            "prefixes": {"2": shuffle_cell}
        }
        baseline[condition] = {
            "failure_burst_at_0_5": {"global_max_with_episode_resets": 20}
        }
        positive = {"estimate": 0.02, "one_sided_lower_95": 0.001}
        comparisons[condition] = {
            "real_rgb_capsule_k2_minus_published_mask_topk_k2": {
                "success_at_0_5": dict(positive),
                "best_min_role_iou": dict(positive),
            },
            "real_rgb_capsule_k2_minus_spatial_shuffle_capsule_k2": {
                "success_at_0_5": dict(positive)
            },
        }
    return metrics, baseline, comparisons


class ObjectGraphRGBPreflightContractTests(unittest.TestCase):
    def test_frozen_formats_and_two_arm_schema(self) -> None:
        self.assertEqual(SNAPSHOT_FORMAT, "object_graph_rgb_candidate_inputs_v1")
        self.assertEqual(
            GENERATOR_FORMAT, "support_conditioned_ordered_chain_rgb_v1"
        )
        self.assertEqual(
            GENERATOR_PROTOCOL, "stateless_current_rgb_entity_roi_two_link_v1"
        )
        self.assertEqual(MAX_CANDIDATES, 2)
        self.assertEqual(tuple(ARMS), ("real_rgb", "spatial_shuffle"))
        self.assertEqual(
            SUMMARY_FORMAT, "object_graph_rgb_candidate_coverage_summary_v1"
        )
        self.assertEqual(
            ISOLATION_FORMAT,
            "object_graph_rgb_candidate_scoring_isolation_v1",
        )
        self.assertEqual(
            set(SOURCE_CODES),
            {"padding", "v1_anchor", "rgb_ik_positive", "rgb_ik_negative"},
        )
        self.assertEqual(set(_rgb_array_schema(frames=3)), set(OUTPUT_ARRAY_KEYS))
        self.assertIsInstance(BACKEND, str)
        self.assertIsInstance(BACKEND_FORMAT, str)
        self.assertIsInstance(BACKEND_PROTOCOL, str)
        self.assertEqual(
            DINO_EXTRACTOR_METADATA_KEYS,
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
            },
        )

    def test_exact_cli_is_frozen(self) -> None:
        parser = build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        self.assertEqual(
            options,
            {
                "--help",
                "--source-benchmark-root",
                "--v1-preflight-root",
                "--mask-topk-preflight-root",
                "--rgb-backend-manifest",
                "--v1-graph",
                "--dino-repo",
                "--dino-checkpoint",
                "--isolation-gate",
                "--immutable-inputs",
                "--output",
            },
        )

    def test_backend_array_schema_fails_closed(self) -> None:
        arrays = _valid_arrays()
        geometry = {
            "expected_widths": np.asarray([3.0, 3.0], dtype=np.float32),
            "expected_lengths": np.asarray([10.0, 10.0], dtype=np.float64),
        }
        _validate_decoded_rgb_arrays(
            arrays, frames=2, resolution=64, **geometry
        )
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["link_half_widths_px"][0, 1, 0] = 0.0
        with self.assertRaises(ValueError):
            _validate_decoded_rgb_arrays(
                tampered, frames=2, resolution=64, **geometry
            )
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["link_half_widths_px"][0, 1] = 31.0
        with self.assertRaises(ValueError):
            _validate_decoded_rgb_arrays(
                tampered, frames=2, resolution=64, **geometry
            )
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["candidate_source_code"][0, 1] = SOURCE_CODES["v1_anchor"]
        with self.assertRaises(ValueError):
            _validate_decoded_rgb_arrays(
                tampered, frames=2, resolution=64, **geometry
            )
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["candidate_valid"][0] = False
        tampered["poses_xy"][0] = 0.0
        tampered["link_half_widths_px"][0] = 0.0
        tampered["candidate_cost"][0] = np.finfo(np.float32).max
        tampered["candidate_weight"][0] = 0.0
        tampered["candidate_confidence"][0] = 0.0
        tampered["rgb_evidence_score"][0] = 0.0
        tampered["mask_geometry_score"][0] = 0.0
        tampered["candidate_source_code"][0] = SOURCE_CODES["padding"]
        tampered["roi_xyxy"][0] = np.asarray([0, 0, -5, -5], dtype=np.int16)
        with self.assertRaises(ValueError):
            _validate_decoded_rgb_arrays(
                tampered, frames=2, resolution=64, **geometry
            )

    def test_slot_zero_is_exact_published_v1_bytes_and_pose(self) -> None:
        arrays = _valid_arrays(frames=1)
        pose = arrays["poses_xy"][0, 0]
        role_masks = render_role_capsules(pose, np.asarray([3.0, 3.0]), (64, 64))
        source_v1 = {
            "role_valid": np.asarray([[True, True]], dtype=np.bool_),
            "keypoints_xy": np.asarray(
                [[[[pose[0, 0], pose[0, 1]], [pose[1, 0], pose[1, 1]]],
                  [[pose[1, 0], pose[1, 1]], [pose[2, 0], pose[2, 1]]]]],
                dtype=np.float32,
            ),
            "role_masks": role_masks[None],
        }
        trace = _validate_rank_zero_exact(arrays=arrays, source_v1=source_v1)
        self.assertEqual(len(trace), 64)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["poses_xy"][0, 0, 1, 0] += 1.0
        with self.assertRaises(ValueError):
            _validate_rank_zero_exact(arrays=changed, source_v1=source_v1)

    def test_capsule_reconstruction_can_recover_beyond_entity_partition(self) -> None:
        arrays = _valid_arrays(frames=1)
        pose = arrays["poses_xy"][0, 1]
        capsules = render_role_capsules(pose, arrays["link_half_widths_px"][0, 1], (64, 64))
        yy, _ = np.indices((64, 64))
        clipped_entity = capsules.any(axis=0) & ((yy % 4) == 0)
        gt = np.zeros((1, 64, 64), dtype=np.uint8)
        gt[0][capsules[0]] = 1
        gt[0][capsules[1]] = 2
        source_v1 = {
            "entity_masks": clipped_entity[None, None],
            "entity_valid": np.asarray([[True]], dtype=np.bool_),
            "role_masks": np.zeros((1, 2, 64, 64), dtype=np.bool_),
        }
        partition = _frame_score_rows(
            arrays=arrays,
            source_v1=source_v1,
            gt_indexed=gt,
            renderer="rgb_partition",
        )
        capsule = _frame_score_rows(
            arrays=arrays,
            source_v1=source_v1,
            gt_indexed=gt,
            renderer="rgb_capsule",
        )
        self.assertEqual(float(capsule["quality"][0, 1]), 1.0)
        self.assertGreater(
            float(capsule["quality"][0, 1]),
            float(partition["quality"][0, 1]),
        )

    def test_gate_is_only_real_rgb_capsule_k2_and_fails_closed(self) -> None:
        metrics, baseline, comparisons = _gate_fixture()
        self.assertTrue(
            _development_gate(
                metrics=metrics, baseline=baseline, comparisons=comparisons
            )["development_candidate"]
        )
        broken = copy.deepcopy(metrics)
        broken["real_rgb"]["rgb_capsule"][CONDITIONS[0]]["prefixes"]["2"][
            "failure_burst_at_0_5"
        ]["global_max_with_episode_resets"] = 11
        gate = _development_gate(
            metrics=broken, baseline=baseline, comparisons=comparisons
        )
        self.assertFalse(gate["development_candidate"])
        self.assertEqual(gate["main_treatment"], "real_rgb/rgb_capsule/k2")
        self.assertIn("spatial_shuffle/rgb_capsule", gate["diagnostic_only"])

        condition = CONDITIONS[0]
        mutations = []
        mutations.append(
            ("availability", lambda m, b, c: m["real_rgb"]["rgb_capsule"][condition]["prefixes"]["2"].__setitem__("availability_rate", 0.98))
        )
        mutations.append(
            ("success", lambda m, b, c: m["real_rgb"]["rgb_capsule"][condition]["prefixes"]["2"].__setitem__("oracle_set_success_at_0_5", 0.96))
        )
        mutations.append(
            ("half_baseline_burst", lambda m, b, c: b[condition]["failure_burst_at_0_5"].__setitem__("global_max_with_episode_resets", 8))
        )
        mutations.append(
            ("episode_p95", lambda m, b, c: m["real_rgb"]["rgb_capsule"][condition]["prefixes"]["2"]["failure_burst_at_0_5"].__setitem__("episode_p95", 11.0))
        )
        mutations.append(
            ("mask_lcb", lambda m, b, c: c[condition]["real_rgb_capsule_k2_minus_published_mask_topk_k2"]["best_min_role_iou"].__setitem__("one_sided_lower_95", 0.0))
        )
        mutations.append(
            ("shuffle_gain", lambda m, b, c: c[condition]["real_rgb_capsule_k2_minus_spatial_shuffle_capsule_k2"]["success_at_0_5"].__setitem__("estimate", 0.009))
        )
        mutations.append(
            ("shuffle_lcb", lambda m, b, c: c[condition]["real_rgb_capsule_k2_minus_spatial_shuffle_capsule_k2"]["success_at_0_5"].__setitem__("one_sided_lower_95", 0.0))
        )
        mutations.append(
            ("parser_latency", lambda m, b, c: m["real_rgb"]["rgb_capsule"][condition]["parser_runtime"].__setitem__("mean_ms", 5.1))
        )
        mutations.append(
            ("end_latency", lambda m, b, c: m["real_rgb"]["rgb_capsule"][condition]["published_v1_cutie_plus_parser_runtime"].__setitem__("p95_ms", 15.1))
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                current_metrics = copy.deepcopy(metrics)
                current_baseline = copy.deepcopy(baseline)
                current_comparisons = copy.deepcopy(comparisons)
                mutate(current_metrics, current_baseline, current_comparisons)
                self.assertFalse(
                    _development_gate(
                        metrics=current_metrics,
                        baseline=current_baseline,
                        comparisons=current_comparisons,
                    )["development_candidate"]
                )

    def test_backend_and_runner_never_train_or_read_scoring_gt(self) -> None:
        backend_source = BACKEND_PATH.read_text(encoding="utf-8")
        run_source = inspect.getsource(
            __import__(
                "tdmpc2.tools.replay_object_graph_rgb_candidates",
                fromlist=["run"],
            ).run
        )
        self.assertNotIn('archive["gt_indexed"]', run_source)
        self.assertNotIn('archive["physics_states"]', run_source)
        self.assertNotIn('archive["actions"]', run_source)
        self.assertNotIn('archive["rewards"]', run_source)
        self.assertIn("spatial_shuffle", backend_source)
        self.assertIn("np.flatnonzero(usable)", backend_source)
        self.assertNotIn("fixed clean episode-0 frame-0", backend_source)
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        lowered = runner.lower()
        self.assertNotIn("tdmpc2.train", lowered)
        self.assertNotIn("train.py", lowered)
        lock = runner.index('chmod 000 -- "$SCORING_ROOT"')
        backend = runner.index("REPLAY_START backend=object_graph_rgb_candidates")
        restore = runner.index(
            'chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"', backend
        )
        aggregate = runner.index(
            "-m tdmpc2.tools.aggregate_object_graph_rgb_candidates", restore
        )
        self.assertLess(lock, backend)
        self.assertLess(backend, restore)
        self.assertLess(restore, aggregate)
        self.assertIn('names=("source","v1","mask_topk","dino_repo","output")', runner)

    def test_isolation_schema_and_privileged_scoring_order_are_frozen(self) -> None:
        isolation_source = inspect.getsource(_validate_isolation_gate)
        fields = {
            "format",
            "status",
            "source_benchmark_root",
            "v1_preflight_root",
            "mask_topk_preflight_root",
            "worker_input_root",
            "scoring_root",
            "scoring_probe_relative_to_source",
            "root_mode_before",
            "root_mode_locked",
            "root_mode_restored",
            "probe_sha256_before",
            "probe_sha256_restored",
            "locked_utc",
            "backend_completed_utc",
            "restored_utc",
            "backend_completed_before_restore",
            "backend_device",
            "cuda_visible_devices",
            "BENCHMARK_GPU_UUID",
            "rgb_backend_manifest_relative_to_summary_root",
            "rgb_backend_manifest_sha256",
            "same_uid_read_probe",
        }
        for field in fields:
            self.assertIn(f'"{field}"', isolation_source)
        self.assertIn('payload.get("backend_device") != "cuda:0"', isolation_source)
        self.assertIn(
            'payload.get("cuda_visible_devices") != gpu_uuid', isolation_source
        )
        aggregate_source = inspect.getsource(
            __import__(
                "tdmpc2.tools.aggregate_object_graph_rgb_candidates",
                fromlist=["aggregate"],
            ).aggregate
        )
        isolation = aggregate_source.index("_validate_isolation_gate(")
        backend = aggregate_source.index("_validate_rgb_backend(")
        scoring = aggregate_source.index("_score_real_backend(")
        self.assertLess(isolation, backend)
        self.assertLess(backend, scoring)
        self.assertIn("_tree_snapshot(root", AGGREGATOR_PATH.read_text(encoding="utf-8"))

    def test_aggregate_internal_backend_validator_call_matches_signature(self) -> None:
        source = AGGREGATOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(AGGREGATOR_PATH))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_rgb_backend"
        ]
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_rgb_backend"
        ]
        self.assertEqual(len(definitions), 1)
        self.assertEqual(len(calls), 1)
        expected_keywords = {
            argument.arg for argument in definitions[0].args.kwonlyargs
        }
        actual_keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertNotIn(None, actual_keywords)
        self.assertEqual(actual_keywords, expected_keywords)
        self.assertEqual(len(calls[0].args), 1)

    def test_aggregator_never_authorizes_controller_training(self) -> None:
        source = AGGREGATOR_PATH.read_text(encoding="utf-8")
        self.assertIn('"controller_training_authorized": False', source)
        self.assertIn('"scientific_go": False', source)
        self.assertIn('"backend_fixed_labelled_support_masks_input": True', source)
        self.assertIn("BOOTSTRAP_RESAMPLES = 10_000", source)
        self.assertIn("real_rgb_capsule_k2_minus_spatial_shuffle", source)


if __name__ == "__main__":
    unittest.main()
