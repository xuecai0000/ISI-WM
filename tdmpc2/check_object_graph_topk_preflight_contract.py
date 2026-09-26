"""Dependency-light integration contracts for the top-K coverage preflight."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

import numpy as np

from tdmpc2.check_ordered_chain_topk_contract import _chain_frame, _support
from tdmpc2.common.object_graph_topk_snapshot import (
    FORMAT as SNAPSHOT_FORMAT,
    _graph_contract,
)
from tdmpc2.common.unified_vos import TASK_ROLES, file_sha256
from tdmpc2.perception.ordered_chain_topk import (
    MAX_CANDIDATES,
    SOURCE_CODES,
    OrderedChainTopKGenerator,
)
from tdmpc2.perception.support_conditioned_object_graph import load_object_graph
from tdmpc2.tools.aggregate_object_graph_topk_candidates import (
    ISOLATION_FORMAT,
    K_VALUES,
    SUMMARY_FORMAT,
    _candidate_quality,
    _frame_score_rows,
    _generator_from_frozen_support,
    _score_prefixes,
    _validate_rank_zero_exact,
)
from tdmpc2.tools.replay_object_graph_topk_candidates import (
    BACKEND,
    FORMAT as BACKEND_FORMAT,
    OUTPUT_ARRAY_KEYS,
    _copy_frame,
    _empty_arrays,
    _validate_output_arrays,
)


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "object_graphs" / "acrobot_swingup.json"
RUNNER_PATH = ROOT / "tools" / "run_object_graph_topk_preflight.sh"
BACKEND_PATH = ROOT / "tools" / "replay_object_graph_topk_candidates.py"
AGGREGATOR_PATH = ROOT / "tools" / "aggregate_object_graph_topk_candidates.py"


class ObjectGraphTopKPreflightContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = load_object_graph(GRAPH_PATH)
        self.support = _support()

    def test_formats_and_structural_ontology_are_frozen(self) -> None:
        self.assertEqual(SNAPSHOT_FORMAT, "object_graph_topk_candidate_inputs_v1")
        self.assertEqual(
            BACKEND_FORMAT,
            "object_graph_ordered_chain_topk_replay_predictions_v1",
        )
        self.assertEqual(BACKEND, "object_graph_ordered_chain_topk_replay")
        self.assertEqual(
            SUMMARY_FORMAT, "object_graph_topk_candidate_coverage_summary_v1"
        )
        self.assertEqual(
            ISOLATION_FORMAT, "object_graph_topk_candidate_scoring_isolation_v1"
        )
        self.assertEqual(K_VALUES, (1, 2, 4))
        self.assertEqual(MAX_CANDIDATES, 4)
        self.assertEqual(
            self.graph.source_roles, TASK_ROLES["acrobot-swingup"]
        )
        self.assertEqual(
            self.graph.semantic_roles, TASK_ROLES["acrobot-swingup"]
        )

    def test_backend_array_contract_preserves_sparse_reserved_anchor(self) -> None:
        generator = OrderedChainTopKGenerator(
            self.graph, self.support, max_candidates=MAX_CANDIDATES
        )
        normal, _ = _chain_frame(128, -0.3, 0.7)
        folded, _ = _chain_frame(128, -2.89, 0.13)
        normal_frame = generator.project(
            entity_mask=normal > 0, entity_available=True
        )
        folded_frame = generator.project(
            entity_mask=folded > 0, entity_available=True
        )
        arrays = _empty_arrays(frames=2)
        _copy_frame(arrays, 0, normal_frame)
        _copy_frame(arrays, 1, folded_frame)
        _validate_output_arrays(arrays, frames=2)
        self.assertEqual(set(arrays), set(OUTPUT_ARRAY_KEYS))
        np.testing.assert_array_equal(
            arrays["candidate_valid"][0], [True, True, False, False]
        )
        np.testing.assert_array_equal(
            arrays["candidate_valid"][1], [False, True, False, False]
        )
        self.assertEqual(
            int(arrays["candidate_source_code"][0, 0]),
            SOURCE_CODES["v1_anchor"],
        )
        self.assertEqual(int(arrays["candidate_source_code"][1, 0]), 0)
        self.assertNotEqual(int(arrays["candidate_source_code"][1, 1]), 0)

        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["candidate_count"][1] += 1
        with self.assertRaises(RuntimeError):
            _validate_output_arrays(tampered, frames=2)

    def test_backend_generator_call_is_current_mask_status_only(self) -> None:
        signature = set(
            inspect.signature(OrderedChainTopKGenerator.project).parameters
        )
        self.assertEqual(signature, {"self", "entity_mask", "entity_available"})
        source = BACKEND_PATH.read_text(encoding="utf-8")
        run_source = inspect.getsource(
            __import__(
                "tdmpc2.tools.replay_object_graph_topk_candidates",
                fromlist=["run"],
            ).run
        )
        # Disclosure keys such as ``actions_read=False`` are expected; actual
        # privileged archive indexing is forbidden in the backend path.
        self.assertNotIn('archive["gt_indexed"]', run_source)
        self.assertNotIn('archive["physics_states"]', run_source)
        self.assertNotIn('archive["actions"]', run_source)
        self.assertNotIn('archive["rewards"]', run_source)
        self.assertIn("episode_entity_arrays_loaded_before_replay", source)
        self.assertIn('"future_entity_frames_generator_input": False', source)

    def test_aggregator_replays_the_disjoint_gt_free_support_path(self) -> None:
        signature = set(
            inspect.signature(_generator_from_frozen_support).parameters
        )
        self.assertEqual(
            signature, {"graph", "dataset", "dataset_root", "source_root"}
        )
        source = inspect.getsource(_generator_from_frozen_support)
        self.assertIn(
            'source_root / "worker_inputs" / "backend_inputs.json"', source
        )
        self.assertIn("support_path = worker_support_paths[TASK]", source)
        self.assertIn("support_path == dataset_support_path", source)
        self.assertIn("file_sha256(dataset_support_path)", source)

    def test_fixed_role_identity_cannot_be_permuted_by_the_scorer(self) -> None:
        indexed, _ = _chain_frame(128, -0.3, 0.7)
        frame = OrderedChainTopKGenerator(self.graph, self.support).project(
            entity_mask=indexed > 0, entity_available=True
        )
        correct, per_role = _candidate_quality(frame.role_masks[0], indexed)
        swapped, swapped_per_role = _candidate_quality(
            frame.role_masks[0][::-1].copy(), indexed
        )
        self.assertGreater(correct, 0.75)
        self.assertGreater(float(per_role.min()), 0.75)
        self.assertLess(swapped, correct)
        self.assertLess(float(swapped_per_role.min()), float(per_role.min()))

    def test_rank_zero_exact_allows_nonzero_masks_on_invalid_v1_frames(self) -> None:
        indexed, _ = _chain_frame(128, -0.3, 0.7)
        frame = OrderedChainTopKGenerator(self.graph, self.support).project(
            entity_mask=indexed > 0, entity_available=True
        )
        poses = np.zeros((2, MAX_CANDIDATES, 3, 2), dtype=np.float32)
        valid = np.zeros((2, MAX_CANDIDATES), dtype=np.bool_)
        source = np.zeros((2, MAX_CANDIDATES), dtype=np.uint8)
        valid[0, 0] = True
        source[0, 0] = SOURCE_CODES["v1_anchor"]
        poses[0, 0] = frame.poses_xy[0]
        role_masks = np.stack((frame.role_masks[0], frame.role_masks[0]))
        keypoints = np.zeros((2, 2, 2, 2), dtype=np.float32)
        keypoints[0, 0, 0] = frame.poses_xy[0, 0]
        keypoints[0, 0, 1] = frame.poses_xy[0, 1]
        keypoints[0, 1, 0] = frame.poses_xy[0, 1]
        keypoints[0, 1, 1] = frame.poses_xy[0, 2]
        trace = _validate_rank_zero_exact(
            poses=poses,
            valid=valid,
            source_code=source,
            source_v1={
                "role_valid": np.asarray(
                    [[True, True], [False, False]], dtype=np.bool_
                ),
                "keypoints_xy": keypoints,
                # The published v1 contract deliberately permits a geometric
                # mask even if appearance/status makes the token invalid.
                "role_masks": role_masks,
            },
        )
        self.assertEqual(len(trace), 64)

        # The privileged scorer must use the exact v1 slot-zero mask bytes,
        # not silently reconstruct another partition from its compact pose.
        scoring_poses = np.zeros((1, MAX_CANDIDATES, 3, 2), dtype=np.float32)
        scoring_poses[0, 0] = frame.poses_xy[0][::-1]
        scoring_valid = np.zeros((1, MAX_CANDIDATES), dtype=np.bool_)
        scoring_valid[0, 0] = True
        scoring_source = np.zeros((1, MAX_CANDIDATES), dtype=np.uint8)
        scoring_source[0, 0] = SOURCE_CODES["v1_anchor"]
        scored = _frame_score_rows(
            poses=scoring_poses,
            valid=scoring_valid,
            source_code=scoring_source,
            entity_masks=(indexed > 0)[None],
            gt_indexed=indexed[None],
            slot_zero_masks=frame.role_masks[0][None],
        )
        expected_quality, _ = _candidate_quality(frame.role_masks[0], indexed)
        self.assertEqual(float(scored["quality"][0, 0]), expected_quality)

    def test_prefix_metrics_are_monotonic_and_rescue_is_gt_free(self) -> None:
        episodes = []
        for _ in range(2):
            quality = np.zeros((4, MAX_CANDIDATES), dtype=np.float64)
            role_iou = np.zeros((4, MAX_CANDIDATES, 2), dtype=np.float64)
            valid = np.zeros((4, MAX_CANDIDATES), dtype=np.bool_)
            source = np.zeros((4, MAX_CANDIDATES), dtype=np.uint8)
            valid[:, 0] = True
            valid[0, 0] = False
            valid[:, 1] = True
            source[valid[:, 0], 0] = SOURCE_CODES["v1_anchor"]
            source[:, 1] = SOURCE_CODES["circle_raw_positive"]
            quality[:, 0] = 0.4
            quality[:, 1] = 0.8
            role_iou[:, 0] = 0.4
            role_iou[:, 1] = 0.8
            episodes.append(
                {
                    "quality": quality,
                    "role_iou": role_iou,
                    "valid": valid,
                    "source_code": source,
                    "mode_code": np.where(valid, 1, 0).astype(np.int8),
                }
            )
        scored = _score_prefixes(episodes)
        prefixes = scored["prefixes"]
        self.assertLessEqual(
            prefixes["1"]["oracle_set_success_at_0_5"],
            prefixes["2"]["oracle_set_success_at_0_5"],
        )
        self.assertLessEqual(
            prefixes["2"]["oracle_set_success_at_0_5"],
            prefixes["4"]["oracle_set_success_at_0_5"],
        )
        rescue = prefixes["2"]["rescue_available_at_k"]
        self.assertEqual(rescue["conditioning"], "slot_zero_unavailable_current_frame_only_no_gt")
        self.assertEqual(rescue["denominator_frames"], 2)
        self.assertEqual(rescue["rate"], 1.0)

    def test_snapshot_binds_canonical_role_order_and_published_graph(self) -> None:
        published = {
            "graphs": {
                "acrobot-swingup": {
                    "graph_file_sha256": file_sha256(GRAPH_PATH),
                    "graph": self.graph.metadata(),
                    "tokenizer": {"graph_sha256": self.graph.graph_sha256},
                }
            }
        }
        contract = _graph_contract(package_root=ROOT, v1_backend=published)
        self.assertEqual(
            contract["source_roles"], list(TASK_ROLES["acrobot-swingup"])
        )
        self.assertEqual(contract["projector_types"], ["ordered_chain_segments_v1"])

    def test_runner_locks_gt_before_backend_and_never_trains(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        trap_index = source.index("trap archive_on_exit EXIT")
        stage_index = source.index('if ! mkdir -- "$STAGE"')
        lock_index = source.index('chmod 000 -- "$SCORING_ROOT"')
        backend_index = source.index("REPLAY_START backend=object_graph_topk_candidates")
        restore_index = source.index(
            'chmod "$SCORING_ROOT_MODE_BEFORE" -- "$SCORING_ROOT"',
            backend_index,
        )
        aggregate_index = source.index(
            "-m tdmpc2.tools.aggregate_object_graph_topk_candidates",
            restore_index,
        )
        final_verify_index = source.rindex('--verify "$INPUTS"')
        promote_index = source.index('mv -T -- "$STAGE" "$BASE"')
        self.assertLess(trap_index, stage_index)
        self.assertLess(lock_index, backend_index)
        self.assertLess(backend_index, restore_index)
        self.assertLess(restore_index, aggregate_index)
        self.assertLess(aggregate_index, final_verify_index)
        self.assertLess(final_verify_index, promote_index)
        self.assertNotIn("chmod -R", source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        lowered = source.lower()
        self.assertNotIn("tdmpc2.train", lowered)
        self.assertNotIn("train.py", lowered)
        self.assertIn("controller_training_authorized", source)

    def test_aggregator_discloses_privileged_oracle_and_never_authorizes(self) -> None:
        source = AGGREGATOR_PATH.read_text(encoding="utf-8")
        self.assertIn("gt_union_clean_hard_evaluated_once_not_double_counted", source)
        self.assertIn("oracle_selects_candidate_only_for_privileged_offline_scoring", source)
        self.assertIn('"controller_training_authorized": False', source)
        self.assertIn('"scientific_go": False', source)
        self.assertIn("paired_bootstrap_unit", source)
        self.assertIn("maximum_global_failure_burst", source)


if __name__ == "__main__":
    unittest.main()
