"""Dependency-light contract for the native-resolution Cutie preflight."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np

from tdmpc2.tools.evaluate_cutie_native_resolution_tracker import MetricAccumulator


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tdmpc2" / "tools" / "evaluate_cutie_native_resolution_tracker.py"
AGGREGATOR = ROOT / "tdmpc2" / "tools" / "aggregate_cutie_native_resolution_tracker.py"
RUNNER = ROOT / "tdmpc2" / "tools" / "run_cutie_native_resolution_tracker_preflight.sh"
PAIRED_SCHEMA = ROOT / "tdmpc2" / "common" / "cutie_paired_support.py"
PAIRED_EXTERNAL = ROOT / "tdmpc2" / "common" / "cutie_external_snapshot.py"
PAIRED_COLLECTOR = (
	ROOT / "tdmpc2" / "tools" / "collect_cutie_paired_native_support.py"
)
PAIRED_AGGREGATOR = (
	ROOT / "tdmpc2" / "tools" / "aggregate_cutie_native_support_three_arm.py"
)
PAIRED_RUNNER = (
	ROOT / "tdmpc2" / "tools" / "run_cutie_native_support_three_arm_preflight.sh"
)


def _source_contract() -> dict:
	for path in (
		EVALUATOR, AGGREGATOR, RUNNER, PAIRED_SCHEMA, PAIRED_EXTERNAL,
		PAIRED_COLLECTOR,
		PAIRED_AGGREGATOR, PAIRED_RUNNER,
	):
		if not path.is_file():
			raise FileNotFoundError(path)
	evaluator = EVALUATOR.read_text(encoding="utf-8")
	aggregator = AGGREGATOR.read_text(encoding="utf-8")
	runner = RUNNER.read_text(encoding="utf-8")
	paired_schema = PAIRED_SCHEMA.read_text(encoding="utf-8")
	paired_external = PAIRED_EXTERNAL.read_text(encoding="utf-8")
	paired_collector = PAIRED_COLLECTOR.read_text(encoding="utf-8")
	paired_aggregator = PAIRED_AGGREGATOR.read_text(encoding="utf-8")
	paired_runner = PAIRED_RUNNER.read_text(encoding="utf-8")
	tree = ast.parse(evaluator, filename=str(EVALUATOR))
	defined_functions = {
		node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
	}
	called_names = {
		node.func.id for node in ast.walk(tree)
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
	}
	track_index = evaluator.index("result = adapter.track(frame)")
	gt_index = evaluator.index("gt_native = _role_gt_masks(")
	checks = {
		"resolution_choices_64_128_256": "RESOLUTIONS = (64, 128, 256)" in evaluator,
		"cutie_precedes_gt_query": track_index < gt_index,
		# Legacy v1 remains source64-only and defaults to S64. The adapter input
		# is now intentionally dynamic because paired v2 adds an explicit S128
		# arm; canonical tracker output/GT scoring remain fixed at 64.
		"legacy_source64_support": (
			"none_byte_exact_source64" in evaluator
			and "same_frozen_64x64_bytes_all_resolutions" in evaluator
			and "Legacy cutie_indexed_mask_support_v1 can only provide source64 support" in evaluator
			and '"--support-resolution", type=int, choices=(64, 128), default=64' in evaluator
			and "adapter_source64_direct_to_fixed_tracker448_once" in evaluator
			and "mask_output_size=(64, 64)" in evaluator
		),
		"paired_v2_support_is_explicit": all(
			value in evaluator for value in (
				'PAIRED_FORMAT = "cutie_native_support_tracker_evaluation_v2"',
				"load_paired_native_support_prompts(",
				"args.support_resolution, args.support_resolution",
				"same_paired_support_states_native_assets_selected_by_arm",
				"native_high_resolution_support",
				"native_foreground_support",
				"native_mask_support",
			)
		),
		"paired_three_arms_exact": all(
			value in evaluator for value in (
				"(64, 64), (128, 64), (128, 128)",
				"runtime{args.resolution}_support{args.support_resolution}",
				"matched_native64_runtime_and_support_control",
				"runtime_cutie_input_resolution_only",
				"runtime_and_support_native_resolution",
			)
		),
		"paired_pack_is_native_and_same_state": all(
			value in paired_schema + paired_collector for value in (
				"cutie_paired_native_support_v2",
				"single_process_same_accepted_state_no_step_between_renders",
				"native_mujoco_segmentation_per_resolution_no_resize",
				"physics_exact",
				"background_exact",
				"rng_exact",
				"cross_resolution_evidence",
			)
		),
		"support_loader_call_resolves": (
			"_load_frozen_support" in called_names
			and "_load_frozen_support" in defined_functions
			and "load_paired_native_support_prompts" in called_names
			and "_resample_support" not in called_names
		),
		"core_background_path": (
			"background.cutie_same_state_rgb(" in evaluator
			and "policy64 = _latest_policy_rgb(observation)" in evaluator
		),
		"fixed_gt_scoring": (
			"args.gt_render_resolution" in evaluator
			and "gt_masks = _scoring_masks(gt_native)" in evaluator
			and "gt_scoring64_sequence_sha256" in evaluator
		),
		"separate_action_rng": "np.random.default_rng(args.action_seed)" in evaluator,
		"trajectory_hash": "physics_trajectory_sha256" in evaluator,
		"runtime_health": all(
			name in evaluator for name in ("adapter_errors", "timeouts", "worker_restarts")
		),
		"per_role_metrics": all(
			name in evaluator for name in (
				"valid_frame_rate", "empty_mask_frame_rate", "max_invalid_burst",
				"mean_mask_area_pixels", "mean_iou_on_gt_visible_frames",
				"identity_accuracy_on_gt_visible_frames",
			)
		),
		"strict_pairing": all(
			name in aggregator for name in (
				"physics_trajectory_sha256", "action_sequence_sha256",
				"background_sequence_sha256", "support_source",
			)
		),
		"effective_improvement_required": (
			"lower_arm_valid_gain" in aggregator
			and "lower_arm_invalid_relative_reduction" in aggregator
			and "lower_arm_burst_relative_reduction" in aggregator
			and "lower_arm_iou_gain" in aggregator
		),
		"empirical_extra_sensor_evidence": (
			"policy64_pil_bilinear_to_runtime_resolution" in evaluator
			and "distinct_frames" in evaluator
			and "absolute_difference_trace_sha256" in evaluator
		),
		"scientific_claim_labels": all(
			label in evaluator and label in aggregator for label in (
				"extra_sensor_information", "raw_sensor_information_parity",
				"fair_representation_comparison", "agent_observation_unchanged",
				"runtime_cutie_input_resolution_only", "perception_diagnostic",
				"representation_advantage_vs_rgb",
			)
		),
		"native256_excluded": (
			"optional_exploratory_only_not_in_gates" in aggregator
			and 'RUN_256="${RUN_256:-0}"' in runner
		),
		"cartpole_health_default": 'RUN_CARTPOLE="${RUN_CARTPOLE:-1}"' in runner,
		"two_gpu_pair": (
			'GPU_64="${GPU_64:-0}"' in runner
			and 'GPU_128="${GPU_128:-1}"' in runner
			and '[[ "$GPU_64" != "$GPU_128" ]]' in runner
		),
		"runner_never_trains": (
			"-m tdmpc2.train" not in runner
			and "python tdmpc2/train.py" not in runner
			and "No controller training was launched" in runner
		),
		"paired_aggregator_binds_v2_support": all(
			value in paired_aggregator for value in (
				'cutie_native_support_tracker_evaluation_v2',
				'"A": {', '"B": {', '"C": {',
				"paired_support_id", "physics_state_trace_sha256",
				"action_sequence_trace_sha256", "reset_ordinal_trace_sha256",
				"support_input_size", "support_replay_total_ms",
			)
		),
		"paired_external_inputs_are_rehashed": (
			"external_inputs.json" in paired_runner
			and "tdmpc2.common.cutie_external_snapshot" in paired_runner
			and "validate_snapshot" in paired_aggregator
			and "local_python" in paired_external
			and "video_tree" in paired_external
			and "external_cutie_python" in paired_external
			and "external_cutie_config" in paired_external
			and "Cutie checkpoint changed during the run" in paired_external
		),
		"paired_scientific_counts_are_fixed": (
			'[[ "$EPISODES" == 20 && "$STEPS" == 500 ]]' in paired_runner
			and "(episodes, actions, frames_per_episode) != (20, 500, 501)"
			in paired_aggregator
		),
		"paired_support_and_evaluation_seeds_are_disjoint": (
			"ALL_SEEDS=(" in paired_runner
			and "support-collection and evaluation seed domains overlap"
			in paired_aggregator
		),
		"paired_runner_invokes_contract_and_never_trains": (
			"tdmpc2.check_cutie_native_resolution_tracker_contract" in paired_runner
			and "arm_pair()" in paired_runner
			and "A) printf '64 64'" in paired_runner
			and "B) printf '128 64'" in paired_runner
			and "C) printf '128 128'" in paired_runner
			and "--support-resolution" in paired_runner
			and "-m tdmpc2.train" not in paired_runner
			and "python tdmpc2/train.py" not in paired_runner
			and "No controller training was launched" in paired_runner
		),
	}
	if not all(checks.values()):
		raise AssertionError({key: value for key, value in checks.items() if not value})
	return checks


def _metric_contract() -> dict:
	accumulator = MetricAccumulator(("upper_arm", "lower_arm"), 2)
	gt = np.asarray([
		[[1, 0], [0, 0]],
		[[0, 0], [0, 1]],
	], dtype=np.bool_)
	accumulator.begin_episode(0)
	accumulator.record(
		step_index=0,
		predicted=gt.copy(),
		gt=gt,
		lost=np.asarray([False, False]),
		feature_finite=np.asarray([True, True]),
		tracker_ms=1.0,
		native_rgb_ms=2.0,
		gt_scoring_ms=3.0,
	)
	invalid = gt.copy()
	invalid[1] = False
	accumulator.record(
		step_index=1,
		predicted=invalid,
		gt=gt,
		lost=np.asarray([False, True]),
		feature_finite=np.asarray([True, True]),
		tracker_ms=1.5,
		native_rgb_ms=2.5,
		gt_scoring_ms=3.5,
	)
	# Burst state must reset at an episode boundary rather than joining episodes.
	accumulator.begin_episode(1)
	accumulator.record(
		step_index=0,
		predicted=invalid,
		gt=gt,
		lost=np.asarray([False, True]),
		feature_finite=np.asarray([True, True]),
		tracker_ms=2.0,
		native_rgb_ms=3.0,
		gt_scoring_ms=4.0,
	)
	summary = accumulator.summary()
	lower = summary["per_role"]["lower_arm"]
	checks = {
		"frames": summary["frames"] == 3,
		"valid_rate": lower["valid_frame_rate"] == 1 / 3,
		"empty_rate": lower["empty_mask_frame_rate"] == 2 / 3,
		"burst_resets_per_episode": lower["max_invalid_burst"] == 1,
		"perfect_first_frame_iou": lower["mean_iou_on_gt_visible_frames"] == 1 / 3,
		"identity_denominator_is_gt_visible": (
			lower["identity_accuracy_on_gt_visible_frames"] == 1 / 3
		),
		"latency": summary["latency"]["cutie_tracker"]["mean_ms"] == 1.5,
	}
	if not all(checks.values()):
		raise AssertionError({key: value for key, value in checks.items() if not value})
	return checks


def main() -> int:
	report = {
		"source": _source_contract(),
		"metrics": _metric_contract(),
		"metric_accumulator_source": inspect.getsourcefile(MetricAccumulator),
	}
	print("CUTIE_NATIVE_RESOLUTION_TRACKER_CONTRACT_OK", json.dumps(report, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
