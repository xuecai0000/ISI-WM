"""Strict cross-resolution aggregator for the Cutie tracker-only preflight.

The 128 candidate is eligible for GO only when exact physical/background/action
pairing holds, its absolute quality gates pass, and it materially improves the
task's small-role tracking over native64 without regressing the other role.
Optional native256 results are reported as exploratory and never participate in
the default engineering or scientific decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


FORMAT = "cutie_native_resolution_tracker_preflight_summary_v1"
EVALUATION_FORMAT = "cutie_native_resolution_tracker_evaluation_v1"
PRIMARY_ROLE = {
	"acrobot-swingup": "lower_arm",
	"cartpole-swingup": "pole",
}
EXPECTED_ROLES = {
	"acrobot-swingup": ("upper_arm", "lower_arm"),
	"cartpole-swingup": ("cart", "pole"),
}
REQUIRED_RESOLUTIONS = (64, 128)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError(f"Expected JSON object: {path}")
	return payload


def _validate_evaluation(
	payload: dict,
	*,
	task: str,
	resolution: int,
	expected_physical_gpu: str,
) -> None:
	if payload.get("format") != EVALUATION_FORMAT:
		raise ValueError(f"{task}/{resolution}: wrong format.")
	if payload.get("status") != "tracker_evaluation_complete":
		raise ValueError(f"{task}/{resolution}: evaluation is incomplete.")
	if payload.get("task") != task or payload.get("resolution") != resolution:
		raise ValueError(f"{task}/{resolution}: task/resolution mismatch.")
	if tuple(payload.get("roles", ())) != EXPECTED_ROLES[task]:
		raise ValueError(f"{task}/{resolution}: role order mismatch.")
	protocol = payload.get("protocol", {})
	for key, expected in {
		"tracker_only": True,
		"policy_constructed": False,
		"controller_training_steps": 0,
		"native_simulator_rgb": True,
		"dynamic_background_split": "validation",
		"gt_use": "post_track_offline_scoring_only",
		"gt_query_order": "after_cutie_track_at_same_physics_state",
		"gt_drives_tracker": False,
		"cross_resolution_support": "same_frozen_64x64_bytes_all_resolutions",
		"high_resolution_support_sensor": False,
		"extra_sensor_information": resolution > 64,
		"raw_sensor_information_parity": resolution == 64,
		"fair_representation_comparison": False,
		"agent_observation_unchanged": True,
		"treatment": "runtime_cutie_input_resolution_only",
		"allowed_claim": "perception_diagnostic",
		"disallowed_claim": "representation_advantage_vs_rgb",
	}.items():
		if protocol.get(key) != expected:
			raise ValueError(
				f"{task}/{resolution}: protocol {key}={protocol.get(key)!r}."
			)
	support = payload.get("provenance", {}).get("support", {})
	if (
		support.get("source_resolution") != [64, 64]
		or support.get("adapter_support_input_size") != [64, 64]
		or support.get("adapter_mask_output_size") != [64, 64]
		or support.get("transform_before_adapter") != "none_byte_exact_source64"
		or support.get("source_bytes_reused_exactly") is not True
		or support.get("native_high_resolution_support") is not False
		or not isinstance(support.get("source_sha256"), str)
		or len(support.get("records", ())) != 6
	):
		raise ValueError(f"{task}/{resolution}: invalid frozen-support provenance.")
	evaluation = payload.get("evaluation", {})
	expected_frames = int(evaluation.get("episodes", 0)) * int(
		evaluation.get("frames_per_episode", 0)
	)
	metrics = payload.get("metrics", {})
	if expected_frames <= 0 or metrics.get("frames") != expected_frames:
		raise ValueError(f"{task}/{resolution}: frame count mismatch.")
	if len(payload.get("episodes", ())) != evaluation.get("episodes"):
		raise ValueError(f"{task}/{resolution}: episode record count mismatch.")
	per_role = metrics.get("per_role", {})
	if tuple(per_role) != EXPECTED_ROLES[task]:
		raise ValueError(f"{task}/{resolution}: per-role metrics are missing/reordered.")
	for role, values in per_role.items():
		for key in (
			"valid_frame_rate",
			"empty_mask_frame_rate",
			"mean_mask_area_pixels",
			"mean_mask_area_fraction",
			"max_invalid_burst",
			"mean_iou_on_gt_visible_frames",
			"identity_accuracy_on_gt_visible_frames",
		):
			value = values.get(key)
			if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
				raise ValueError(f"{task}/{resolution}/{role}: invalid metric {key}.")
	health = payload.get("runtime_health", {})
	if any(health.get(key) != 0 for key in (
		"adapter_errors", "timeouts", "worker_restarts"
	)):
		raise ValueError(f"{task}/{resolution}: unhealthy Cutie execution.")
	if health.get("successful_frames") != expected_frames:
		raise ValueError(f"{task}/{resolution}: runtime successful-frame mismatch.")
	adapter = health.get("adapter", {})
	adapter_checks = {
		"input_size": adapter.get("input_size") == [resolution, resolution],
		"support_input_size": adapter.get("support_input_size") == [64, 64],
		"mask_output_size": adapter.get("mask_output_size") == [64, 64],
		"tracker_size": adapter.get("tracker_size") == [448, 448],
		"frames": adapter.get("frames") == expected_frames,
		"episode_hard_resets": adapter.get("episode_hard_resets")
		== evaluation.get("episodes"),
		"reset_strategy": adapter.get("episode_reset_strategy")
		== "fresh_inference_core_support_replay_v1",
	}
	if not all(adapter_checks.values()):
		raise ValueError(
			f"{task}/{resolution}: adapter contract failures "
			f"{[key for key, value in adapter_checks.items() if not value]}."
		)
	tracker_mean = metrics.get("latency", {}).get("cutie_tracker", {}).get("mean_ms")
	if (
		not isinstance(tracker_mean, (int, float))
		or not math.isfinite(float(tracker_mean))
		or not 0 < float(tracker_mean) <= 800
	):
		raise ValueError(f"{task}/{resolution}: implausible tracker mean {tracker_mean!r}.")
	provenance = payload.get("provenance", {})
	if provenance.get("logical_cuda_device") != 0:
		raise ValueError(f"{task}/{resolution}: logical CUDA device must be zero.")
	if provenance.get("cuda_visible_devices") != expected_physical_gpu:
		raise ValueError(
			f"{task}/{resolution}: CUDA_VISIBLE_DEVICES "
			f"{provenance.get('cuda_visible_devices')!r} != {expected_physical_gpu!r}."
		)
	if not isinstance(provenance.get("device_name"), str) or not provenance["device_name"]:
		raise ValueError(f"{task}/{resolution}: missing CUDA device name.")
	evidence = payload.get("extra_sensor_evidence", {})
	if resolution == 64:
		if (
			evidence.get("applicable") is not False
			or evidence.get("frames") != expected_frames
			or any(evidence.get(key) is not None for key in (
				"distinct_frames", "distinct_frame_fraction",
				"mean_absolute_uint8_difference", "absolute_difference_trace_sha256",
			))
		):
			raise ValueError(f"{task}/64: extra-sensor evidence must be explicit N/A.")
	else:
		trace = evidence.get("absolute_difference_trace_sha256")
		if (
			evidence.get("applicable") is not True
			or evidence.get("reference")
			!= "policy64_pil_bilinear_to_runtime_resolution"
			or evidence.get("frames") != expected_frames
			or not isinstance(evidence.get("distinct_frames"), int)
			or evidence["distinct_frames"] <= 0
			or not isinstance(evidence.get("distinct_frame_fraction"), (int, float))
			or not 0 < evidence["distinct_frame_fraction"] <= 1
			or not isinstance(evidence.get("mean_absolute_uint8_difference"), (int, float))
			or not math.isfinite(float(evidence["mean_absolute_uint8_difference"]))
			or evidence["mean_absolute_uint8_difference"] <= 0
			or not isinstance(trace, str) or len(trace) != 64
		):
			raise ValueError(
				f"{task}/{resolution}: no empirical evidence of extra native pixels."
			)


def _pairing(base: dict, candidate: dict) -> dict[str, Any]:
	checks = {
		"task": base["task"] == candidate["task"],
		"roles": base["roles"] == candidate["roles"],
		"seeds": base["seeds"] == candidate["seeds"],
		"evaluation_counts": base["evaluation"] == candidate["evaluation"],
		"cutie_checkpoint": (
			base["provenance"]["cutie_checkpoint_sha256"]
			== candidate["provenance"]["cutie_checkpoint_sha256"]
		),
		"support_source": (
			base["provenance"]["support"]["source_sha256"]
			== candidate["provenance"]["support"]["source_sha256"]
			and base["provenance"]["support"]["records"]
			== candidate["provenance"]["support"]["records"]
		),
		"manifests": (
			base["provenance"]["validation_manifest_sha256"]
			== candidate["provenance"]["validation_manifest_sha256"]
			and base["provenance"]["combined_manifest_sha256"]
			== candidate["provenance"]["combined_manifest_sha256"]
		),
		"matched_gpu_model": (
			base["provenance"]["device_name"]
			== candidate["provenance"]["device_name"]
		),
	}
	episode_mismatches = []
	paired_fields = (
		"episode_index",
		"frames",
		"actions",
		"initial_state_sha256",
		"final_state_sha256",
		"physics_trajectory_sha256",
		"action_sequence_sha256",
		"background_sequence_sha256",
		"policy64_input_sequence_sha256",
		"gt_scoring64_sequence_sha256",
		"background_source",
		"background_start_frame_index",
		"background_end_frame_index",
		"random_policy_reward",
	)
	if len(base["episodes"]) != len(candidate["episodes"]):
		episode_mismatches.append({"field": "episode_count"})
	else:
		for index, (left, right) in enumerate(zip(base["episodes"], candidate["episodes"])):
			bad = [field for field in paired_fields if left.get(field) != right.get(field)]
			if bad:
				episode_mismatches.append({"episode_index": index, "fields": bad})
	checks["episode_trajectories"] = not episode_mismatches
	return {
		"checks": checks,
		"episode_mismatches": episode_mismatches,
		"pass": all(checks.values()),
		"intentionally_unpaired_fields": [
			"native_input_sequence_sha256",
		],
	}


def _candidate_decision(base: dict, candidate: dict, args) -> dict[str, Any]:
	task = base["task"]
	primary = PRIMARY_ROLE[task]
	other_roles = [role for role in EXPECTED_ROLES[task] if role != primary]
	base_role = base["metrics"]["per_role"][primary]
	candidate_role = candidate["metrics"]["per_role"][primary]
	valid_gain = candidate_role["valid_frame_rate"] - base_role["valid_frame_rate"]
	iou_gain = (
		candidate_role["mean_iou_on_gt_visible_frames"]
		- base_role["mean_iou_on_gt_visible_frames"]
	)
	burst_reduction = (
		base_role["max_invalid_burst"] - candidate_role["max_invalid_burst"]
	)
	base_invalid_rate = 1.0 - base_role["valid_frame_rate"]
	candidate_invalid_rate = 1.0 - candidate_role["valid_frame_rate"]
	invalid_relative_reduction = (
		(base_invalid_rate - candidate_invalid_rate) / base_invalid_rate
		if base_invalid_rate > 0 else
		(0.0 if candidate_invalid_rate <= 0 else -math.inf)
	)
	burst_relative_reduction = (
		burst_reduction / base_role["max_invalid_burst"]
		if base_role["max_invalid_burst"] > 0 else
		(0.0 if candidate_role["max_invalid_burst"] <= 0 else -math.inf)
	)
	all_role_valid_deltas = {
		role: (
			candidate["metrics"]["per_role"][role]["valid_frame_rate"]
			- base["metrics"]["per_role"][role]["valid_frame_rate"]
		)
		for role in EXPECTED_ROLES[task]
	}
	all_role_iou_deltas = {
		role: (
			candidate["metrics"]["per_role"][role]["mean_iou_on_gt_visible_frames"]
			- base["metrics"]["per_role"][role]["mean_iou_on_gt_visible_frames"]
		)
		for role in EXPECTED_ROLES[task]
	}
	all_role_identity_deltas = {
		role: (
			candidate["metrics"]["per_role"][role][
				"identity_accuracy_on_gt_visible_frames"
			]
			- base["metrics"]["per_role"][role][
				"identity_accuracy_on_gt_visible_frames"
			]
		)
		for role in EXPECTED_ROLES[task]
	}
	tracker_base = base["metrics"]["latency"]["cutie_tracker"]["mean_ms"]
	tracker_candidate = candidate["metrics"]["latency"]["cutie_tracker"]["mean_ms"]
	tracker_latency_ratio = (
		tracker_candidate / tracker_base if tracker_base > 0 else math.inf
	)
	preparation_base = base["metrics"]["latency"][
		"native_rgb_render_and_composite"
	]["mean_ms"]
	preparation_candidate = candidate["metrics"]["latency"][
		"native_rgb_render_and_composite"
	]["mean_ms"]
	perception_base = tracker_base + preparation_base
	perception_candidate = tracker_candidate + preparation_candidate
	perception_latency_ratio = (
		perception_candidate / perception_base if perception_base > 0 else math.inf
	)
	common_checks = {
		"candidate_absolute_quality": candidate["absolute_quality_gates"]["quality_pass"] is True,
		"total_perception_latency_ratio": (
			perception_latency_ratio <= args.max_perception_latency_ratio
		),
		"zero_runtime_errors": all(
			candidate["runtime_health"].get(key) == 0
			for key in ("adapter_errors", "timeouts", "worker_restarts")
		),
	}
	if task == "acrobot-swingup":
		checks = {
			**common_checks,
			"lower_arm_valid_gain": valid_gain >= args.min_acrobot_valid_gain,
			"lower_arm_invalid_relative_reduction": (
				invalid_relative_reduction
				>= args.min_acrobot_invalid_relative_reduction
			),
			"lower_arm_burst_relative_reduction": (
				burst_relative_reduction
				>= args.min_acrobot_burst_relative_reduction
			),
			"lower_arm_iou_gain": (
				iou_gain >= args.min_acrobot_iou_gain
			),
			"upper_arm_valid_not_regressed": all(
				all_role_valid_deltas[role] >= -args.max_control_valid_regression
				for role in other_roles
			),
			"upper_arm_iou_not_regressed": all(
				all_role_iou_deltas[role]
				>= -args.max_control_iou_identity_regression
				for role in other_roles
			),
			"upper_arm_identity_not_regressed": all(
				all_role_identity_deltas[role]
				>= -args.max_control_iou_identity_regression
				for role in other_roles
			),
		}
		decision_kind = "small_target_effective_improvement"
	else:
		checks = {
			**common_checks,
			"both_roles_valid_not_regressed": all(
				delta >= -args.max_control_valid_regression
				for delta in all_role_valid_deltas.values()
			),
			"both_roles_iou_not_regressed": all(
				delta >= -args.max_control_iou_identity_regression
				for delta in all_role_iou_deltas.values()
			),
			"both_roles_identity_not_regressed": all(
				delta >= -args.max_control_iou_identity_regression
				for delta in all_role_identity_deltas.values()
			),
		}
		decision_kind = "health_control_non_regression"
	return {
		"candidate_resolution": candidate["resolution"],
		"primary_role": primary,
		"decision_kind": decision_kind,
		"thresholds": {
			"min_acrobot_valid_gain": args.min_acrobot_valid_gain,
			"min_acrobot_invalid_relative_reduction": (
				args.min_acrobot_invalid_relative_reduction
			),
			"min_acrobot_burst_relative_reduction": (
				args.min_acrobot_burst_relative_reduction
			),
			"min_acrobot_iou_gain": args.min_acrobot_iou_gain,
			"max_control_valid_regression": args.max_control_valid_regression,
			"max_control_iou_identity_regression": (
				args.max_control_iou_identity_regression
			),
			"max_total_perception_latency_ratio": args.max_perception_latency_ratio,
		},
		"deltas": {
			"primary_valid_frame_rate": valid_gain,
			"primary_invalid_rate_relative_reduction": invalid_relative_reduction,
			"primary_mean_iou": iou_gain,
			"primary_max_invalid_burst_reduction": burst_reduction,
			"primary_max_invalid_burst_relative_reduction": burst_relative_reduction,
			"per_role_valid_frame_rate": all_role_valid_deltas,
			"per_role_mean_iou": all_role_iou_deltas,
			"per_role_identity_accuracy": all_role_identity_deltas,
			"tracker_only_latency_ratio": tracker_latency_ratio,
			"total_perception_latency_ratio": perception_latency_ratio,
			"baseline_total_perception_ms": perception_base,
			"candidate_total_perception_ms": perception_candidate,
		},
		"checks": checks,
		"go": all(checks.values()),
	}


def aggregate(args) -> dict[str, Any]:
	root = args.input_root.expanduser().resolve()
	loaded: dict[str, dict[int, dict]] = {}
	inputs = []
	for task in args.tasks:
		loaded[task] = {}
		for resolution in REQUIRED_RESOLUTIONS:
			path = root / task / f"resolution_{resolution}.json"
			if not path.is_file():
				raise FileNotFoundError(path)
			payload = _load(path)
			expected_gpu = (
				args.expected_gpu_64 if resolution == 64 else args.expected_gpu_128
			)
			_validate_evaluation(
				payload,
				task=task,
				resolution=resolution,
				expected_physical_gpu=expected_gpu,
			)
			loaded[task][resolution] = payload
			inputs.append({
				"task": task,
				"resolution": resolution,
				"relative_to_summary_root": str(path.relative_to(root)),
				"execution_time_staging_path": str(path),
				"absolute_path_lifetime": "execution_only_invalid_after_atomic_promotion",
				"sha256": _file_sha256(path),
			})
		optional = root / task / "resolution_256.json"
		if optional.exists():
			payload = _load(optional)
			_validate_evaluation(
				payload,
				task=task,
				resolution=256,
				expected_physical_gpu=args.expected_gpu_256,
			)
			loaded[task][256] = payload
			inputs.append({
				"task": task,
				"resolution": 256,
				"relative_to_summary_root": str(optional.relative_to(root)),
				"execution_time_staging_path": str(optional),
				"absolute_path_lifetime": "execution_only_invalid_after_atomic_promotion",
				"sha256": _file_sha256(optional),
				"decision_scope": "exploratory_only",
			})

	tasks = {}
	engineering_pass = True
	all_go = True
	for task, values in loaded.items():
		pairing = _pairing(values[64], values[128])
		decision = _candidate_decision(values[64], values[128], args)
		engineering_pass &= pairing["pass"]
		task_go = pairing["pass"] and decision["go"]
		all_go &= task_go
		resolution_metrics = {}
		for resolution, payload in values.items():
			resolution_metrics[str(resolution)] = {
				"decision_scope": (
					"baseline" if resolution == 64 else
					"candidate" if resolution == 128 else "exploratory_only"
				),
				"absolute_quality_pass": payload["absolute_quality_gates"]["quality_pass"],
				"per_role": payload["metrics"]["per_role"],
				"role_swap_frames": payload["metrics"]["role_swap_frames"],
				"latency": payload["metrics"]["latency"],
				"runtime_health": {
					key: payload["runtime_health"][key]
					for key in ("adapter_errors", "timeouts", "worker_restarts")
				},
			}
		tasks[task] = {
			"pairing_64_vs_128": pairing,
			"candidate_128": decision,
			"go": task_go,
			"resolutions": resolution_metrics,
			"resolution_256_excluded_from_all_gates": True,
		}

	full_scope = set(args.tasks) == set(PRIMARY_ROLE)
	if not engineering_pass:
		status = "tracker_preflight_engineering_fail"
		recommendation = "stop_pairing_or_runtime_invalid_do_not_train"
	elif full_scope and all_go:
		status = "native128_tracker_science_go"
		recommendation = "eligible_for_separate_controller_training_protocol"
	elif not full_scope and all_go:
		status = "native128_acrobot_candidate_pass_health_control_missing"
		recommendation = "run_cartpole_health_control_before_any_training_go"
	else:
		status = "native128_tracker_no_go"
		recommendation = "do_not_train_controller_try_next_perception_fix"
	return {
		"format": FORMAT,
		"status": status,
		"scope": {
			"tracker_only": True,
			"tasks": list(args.tasks),
			"baseline_resolution": 64,
			"candidate_resolution": 128,
			"resolution_256": "optional_exploratory_only_not_in_gates",
			"same_frozen_source64_support": True,
			"automatic_controller_training_launched": False,
			"extra_sensor_information_for_candidate": True,
			"raw_sensor_information_parity": False,
			"fair_representation_comparison": False,
			"agent_observation_unchanged": True,
			"treatment": "runtime_cutie_input_resolution_only",
			"allowed_claim": "perception_diagnostic",
			"disallowed_claim": "representation_advantage_vs_rgb",
		},
		"engineering_pass": engineering_pass,
		"full_scientific_scope_present": full_scope,
		"all_required_tasks_go": engineering_pass and full_scope and all_go,
		"recommendation": recommendation,
		"inputs": inputs,
		"tasks": tasks,
		"provenance": {
			"aggregator": str(Path(__file__).resolve()),
			"aggregator_sha256": _file_sha256(Path(__file__).resolve()),
		},
	}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
	path = path.expanduser().resolve()
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + f".incomplete.{os.getpid()}")
	if temporary.exists():
		raise FileExistsError(temporary)
	try:
		with temporary.open("x", encoding="utf-8", newline="\n") as handle:
			json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
			handle.write("\n")
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-root", type=Path, required=True)
	parser.add_argument("--expected-gpu-64", required=True)
	parser.add_argument("--expected-gpu-128", required=True)
	parser.add_argument("--expected-gpu-256", default="0")
	parser.add_argument(
		"--tasks", nargs="+", choices=tuple(PRIMARY_ROLE), default=["acrobot-swingup"]
	)
	parser.add_argument("--min-acrobot-valid-gain", type=float, default=0.05)
	parser.add_argument(
		"--min-acrobot-invalid-relative-reduction", type=float, default=0.20
	)
	parser.add_argument(
		"--min-acrobot-burst-relative-reduction", type=float, default=0.20
	)
	parser.add_argument("--min-acrobot-iou-gain", type=float, default=0.05)
	parser.add_argument("--max-control-valid-regression", type=float, default=0.02)
	parser.add_argument(
		"--max-control-iou-identity-regression", type=float, default=0.02
	)
	parser.add_argument("--max-perception-latency-ratio", type=float, default=1.5)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args(argv)
	if len(args.tasks) != len(set(args.tasks)):
		parser.error("--tasks must be unique")
	for name in ("expected_gpu_64", "expected_gpu_128", "expected_gpu_256"):
		if not getattr(args, name).isdigit():
			parser.error(f"--{name.replace('_', '-')} must be a physical GPU index")
	if args.expected_gpu_64 == args.expected_gpu_128:
		parser.error("native64 and native128 must use different physical GPUs")
	for name in (
		"min_acrobot_valid_gain", "min_acrobot_invalid_relative_reduction",
		"min_acrobot_burst_relative_reduction", "max_control_valid_regression",
		"min_acrobot_iou_gain", "max_control_iou_identity_regression",
	):
		if not 0 <= getattr(args, name) <= 1:
			parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
	if args.max_perception_latency_ratio <= 0:
		parser.error("latency ratio must be positive")
	return args


def main(argv=None) -> int:
	args = parse_args(argv)
	if args.output.exists():
		raise FileExistsError(args.output)
	payload = aggregate(args)
	_atomic_write(args.output, payload)
	print("CUTIE_NATIVE_RESOLUTION_PREFLIGHT_COMPLETE", json.dumps({
		"status": payload["status"],
		"engineering_pass": payload["engineering_pass"],
		"go": payload["all_required_tasks_go"],
		"output": str(args.output.expanduser().resolve()),
	}, allow_nan=False), flush=True)
	# A scientifically negative result is a completed experiment, not a shell failure.
	return 0 if payload["engineering_pass"] else 4


if __name__ == "__main__":
	raise SystemExit(main())
