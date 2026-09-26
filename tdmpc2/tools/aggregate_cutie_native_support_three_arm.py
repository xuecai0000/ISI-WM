"""Strict three-arm native-runtime/native-support Cutie preflight aggregator.

The pre-registered arms are:

* A: runtime64 / support64,
* B: runtime128 / support64, and
* C: runtime128 / support128.

Only C-B isolates removal of the runtime/support resolution mismatch.  C-A is
the joint runtime-resolution plus support-resolution intervention.  This is a
tracker-only diagnostic with privileged simulator masks in the frozen support
pack; it cannot establish a representation advantage over RGB and it never
launches controller training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from tdmpc2.common.cutie_external_snapshot import validate_snapshot
from tdmpc2.common.cutie_paired_support import (
	FORMAT as SUPPORT_FORMAT,
	MASK_GENERATION,
	PAIRING,
	RESOLUTIONS as SUPPORT_RESOLUTIONS,
	RGB_GENERATION,
	SUPPORT_RECORDS,
	SUPPORT_SCHEMA,
	compute_paired_support_id,
	sha256_json,
)


FORMAT = "cutie_native_support_three_arm_preflight_summary_v2"
EVALUATION_FORMAT = "cutie_native_support_tracker_evaluation_v2"
RESET_STRATEGY = "fresh_inference_core_support_replay_v1"
TASKS = ("acrobot-swingup", "cartpole-swingup")
EXPECTED_ROLES = {
	"acrobot-swingup": ("upper_arm", "lower_arm"),
	"cartpole-swingup": ("cart", "pole"),
}
PRIMARY_ROLE = {
	"acrobot-swingup": "lower_arm",
	"cartpole-swingup": "pole",
}
ARM_SPECS = {
	"A": {
		"arm": "runtime64_support64",
		"runtime_resolution": 64,
		"support_resolution": 64,
		"filename": "arm_a_runtime64_support64.json",
		"gpu_group": "a_c",
	},
	"B": {
		"arm": "runtime128_support64",
		"runtime_resolution": 128,
		"support_resolution": 64,
		"filename": "arm_b_runtime128_support64.json",
		"gpu_group": "b",
	},
	"C": {
		"arm": "runtime128_support128",
		"runtime_resolution": 128,
		"support_resolution": 128,
		"filename": "arm_c_runtime128_support128.json",
		"gpu_group": "a_c",
	},
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
IMPLEMENTATION_FILES = (
	"tdmpc2/config.yaml",
	"tdmpc2/common/cutie_external_snapshot.py",
	"tdmpc2/common/cutie_paired_support.py",
	"tdmpc2/check_cutie_object_wrapper_contract.py",
	"tdmpc2/check_cutie_native_highres_contract.py",
	"tdmpc2/check_cutie_native_resolution_tracker_contract.py",
	"tdmpc2/check_cutie_native_resolution_wrapper_contract.py",
	"tdmpc2/check_cutie_paired_native_support_contract.py",
	"tdmpc2/perception/cutie_oc_adapter.py",
	"tdmpc2/tools/collect_cutie_multitask_support.py",
	"tdmpc2/tools/collect_cutie_paired_native_support.py",
	"tdmpc2/tools/evaluate_cutie_native_resolution_tracker.py",
	"tdmpc2/tools/aggregate_cutie_native_resolution_tracker.py",
	"tdmpc2/tools/run_cutie_native_resolution_tracker_preflight.sh",
	"tdmpc2/tools/aggregate_cutie_native_support_three_arm.py",
	"tdmpc2/tools/run_cutie_native_support_three_arm_preflight.sh",
	"tdmpc2/envs/dmcontrol.py",
	"tdmpc2/envs/wrappers/cutie_object.py",
	"tdmpc2/envs/wrappers/foreground_stress.py",
	"tdmpc2/envs/wrappers/tensor.py",
	"tdmpc2/envs/wrappers/video_background.py",
)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[str, list[dict[str, Any]]]:
	records = []
	for path in sorted(value for value in root.rglob("*") if value.is_file()):
		records.append({
			"path": path.relative_to(root).as_posix(),
			"bytes": path.stat().st_size,
			"sha256": _file_sha256(path),
		})
	return sha256_json(records), records


def _load(path: Path) -> dict[str, Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError(f"Expected one JSON object: {path}")
	return payload


def _validate_implementation_manifest(path: Path) -> dict[str, Any]:
	payload = _load(path)
	if payload.get("format") != "cutie_native_support_three_arm_implementation_v2":
		raise ValueError("Implementation manifest format mismatch.")
	rows = payload.get("files")
	if not isinstance(rows, list) or tuple(row.get("path") for row in rows) != IMPLEMENTATION_FILES:
		raise ValueError("Implementation manifest file order/set mismatch.")
	repo_root = Path(__file__).resolve().parents[2]
	for row in rows:
		local = (repo_root / row["path"]).resolve()
		try:
			local.relative_to(repo_root)
		except ValueError as exc:
			raise ValueError(f"Implementation path escapes repository: {local}") from exc
		if not local.is_file():
			raise FileNotFoundError(local)
		if row.get("bytes") != local.stat().st_size:
			raise ValueError(f"Implementation size changed: {row['path']}")
		if row.get("sha256") != _file_sha256(local):
			raise ValueError(f"Implementation hash changed: {row['path']}")
	return {
		"format": payload["format"],
		"relative_path": "provenance/implementation_files.json",
		"sha256": _file_sha256(path),
		"files": rows,
	}


def _require_sha(value: Any, label: str) -> str:
	if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
		raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
	return value


def _number(value: Any, label: str, *, minimum=None, maximum=None) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise ValueError(f"{label} must be numeric, got {value!r}.")
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f"{label} must be finite, got {value!r}.")
	if minimum is not None and result < minimum:
		raise ValueError(f"{label}={result} is below {minimum}.")
	if maximum is not None and result > maximum:
		raise ValueError(f"{label}={result} exceeds {maximum}.")
	return result


def _safe_asset(pack_root: Path, relative: Any, label: str) -> Path:
	if not isinstance(relative, str) or not relative:
		raise ValueError(f"{label} must be one non-empty relative path.")
	path = (pack_root / relative).resolve()
	try:
		path.relative_to(pack_root)
	except ValueError as exc:
		raise ValueError(f"{label} escapes paired support root: {path}") from exc
	if not path.is_file():
		raise FileNotFoundError(path)
	return path


def _typed_array_sha256(array: np.ndarray) -> str:
	array = np.ascontiguousarray(array)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _validate_stored_array(value: Any, label: str) -> str:
	if not isinstance(value, dict):
		raise ValueError(f"{label} must be an object.")
	dtype = value.get("dtype")
	shape = value.get("shape")
	if not isinstance(dtype, str) or not isinstance(shape, list):
		raise ValueError(f"{label} lacks dtype/shape.")
	try:
		array = np.asarray(value.get("values"), dtype=np.dtype(dtype))
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{label} values cannot be reconstructed.") from exc
	if list(array.shape) != shape or not np.isfinite(array).all():
		raise ValueError(f"{label} values/shape are invalid.")
	actual = _typed_array_sha256(array)
	if actual != _require_sha(value.get("sha256"), f"{label}.sha256"):
		raise ValueError(f"{label} typed-array hash mismatch.")
	return actual


def _validate_support_pack(
	path: Path, task: str, *, expected_video_root: Path,
) -> dict[str, Any]:
	"""Validate both native support assets and their same-state anchors."""
	path = path.expanduser().resolve()
	pack_root = path.parent
	payload = _load(path)
	if payload.get("format") != SUPPORT_FORMAT:
		raise ValueError(f"{task}: paired support format mismatch.")
	roles = tuple(payload.get("roles", ()))
	if roles != EXPECTED_ROLES[task]:
		raise ValueError(f"{task}: paired support role order mismatch: {roles!r}.")
	collection = payload.get("collection")
	if not isinstance(collection, dict):
		raise ValueError(f"{task}: paired support collection must be an object.")
	expected_collection = {
		"support_schema": SUPPORT_SCHEMA,
		"task": task,
		"observation": "rgb",
		"split": "support",
		"episodes": SUPPORT_RECORDS,
		"camera_id": 0,
		"resolutions": list(SUPPORT_RESOLUTIONS),
		"label_policy": "simulator_segmentation_support_only",
		"diagnostic_support": True,
		"pairing": PAIRING,
		"rgb_generation": RGB_GENERATION,
		"mask_generation": MASK_GENERATION,
		"full_composed_rgb_native_high_resolution": False,
		"native_foreground_and_mask_not_resized_from_64": True,
	}
	bad = {
		key: {"expected": expected, "actual": collection.get(key)}
		for key, expected in expected_collection.items()
		if collection.get(key) != expected
	}
	if bad:
		raise ValueError(f"{task}: paired support collection mismatch: {bad}.")
	for key in ("environment_seed", "background_seed", "action_seed"):
		if not isinstance(collection.get(key), int):
			raise ValueError(f"{task}: paired support {key} must be an integer.")
	for key in ("manifest_sha256", "combined_manifest_sha256"):
		_require_sha(collection.get(key), f"{task} support {key}")
	expected_videos = [f"video{index}.mp4" for index in range(85, 90)]
	if collection.get("allowed_videos") != expected_videos:
		raise ValueError(f"{task}: paired support video allowlist changed.")
	if sorted(collection.get("covered_videos", ())) != expected_videos:
		raise ValueError(f"{task}: paired support did not cover every support video.")
	paired_support_id = _require_sha(
		collection.get("paired_support_id"), f"{task} paired_support_id"
	)
	if compute_paired_support_id(payload) != paired_support_id:
		raise ValueError(f"{task}: paired_support_id does not match support content.")
	collector_path = Path(__file__).resolve().with_name(
		"collect_cutie_paired_native_support.py"
	)
	if (
		collection.get("collector") != collector_path.name
		or collection.get("collector_sha256") != _file_sha256(collector_path)
	):
		raise ValueError(f"{task}: paired-support collector provenance changed.")
	geom = _safe_asset(pack_root, collection.get("geom_catalog"), "geom_catalog")
	if _file_sha256(geom) != _require_sha(
		collection.get("geom_catalog_sha256"), "geom_catalog_sha256"
	):
		raise ValueError(f"{task}: geom catalog hash mismatch.")

	records = payload.get("records")
	if not isinstance(records, list) or len(records) != SUPPORT_RECORDS:
		raise ValueError(f"{task}: paired support must contain six records.")
	physics_hashes, action_hashes, reset_ordinals = [], [], []
	asset_trace = hashlib.sha256()
	asset_records = []
	previous_reset = 0
	for index, record in enumerate(records):
		if not isinstance(record, dict) or record.get("index") != index:
			raise ValueError(f"{task}: support record indices are not ordered.")
		if record.get("accepted_state_ordinal") != index:
			raise ValueError(f"{task}: accepted-state ordinals are not ordered.")
		reset = record.get("reset_ordinal")
		prefix = record.get("random_prefix_steps")
		if not isinstance(reset, int) or reset <= previous_reset:
			raise ValueError(f"{task}/{index}: invalid reset ordinal.")
		if not isinstance(prefix, int) or prefix != 4 + 3 * index:
			raise ValueError(f"{task}/{index}: invalid action prefix.")
		previous_reset = reset
		if record.get("active_video") not in expected_videos:
			raise ValueError(f"{task}/{index}: out-of-split support video.")
		if set(record.get("selected_names", {})) != set(roles):
			raise ValueError(f"{task}/{index}: selected role provenance mismatch.")
		state_sha = _validate_stored_array(
			record.get("physics_state"), f"{task}/{index}/physics_state"
		)
		action_sha = _validate_stored_array(
			record.get("actions"), f"{task}/{index}/actions"
		)
		action_shape = record["actions"].get("shape")
		if not action_shape or action_shape[0] != prefix:
			raise ValueError(f"{task}/{index}: action trace length mismatch.")
		guard = record.get("same_state_guard")
		if not isinstance(guard, dict):
			raise ValueError(f"{task}/{index}: same-state guard missing.")
		if guard.get("pass") is not True:
			raise ValueError(f"{task}/{index}: same-state guard did not pass.")
		for flag in ("physics_exact", "background_exact", "rng_exact"):
			if guard.get(flag) is not True:
				raise ValueError(f"{task}/{index}: {flag} did not pass.")
		if (
			guard.get("physics_before_sha256") != state_sha
			or guard.get("physics_after_sha256") != state_sha
		):
			raise ValueError(f"{task}/{index}: physics guard/state mismatch.")
		background_state = guard.get("background_state", {})
		active_source = Path(str(background_state.get("active_source", ""))).resolve()
		try:
			active_source.relative_to(expected_video_root.resolve())
		except ValueError as exc:
			raise ValueError(
				f"{task}/{index}: support background source escapes video_hard."
			) from exc
		if active_source.name != record.get("active_video"):
			raise ValueError(f"{task}/{index}: support background source mismatch.")
		for prefix_name in ("background", "rng"):
			before = _require_sha(
				guard.get(f"{prefix_name}_before_sha256"),
				f"{task}/{index}/{prefix_name}_before_sha256",
			)
			after = _require_sha(
				guard.get(f"{prefix_name}_after_sha256"),
				f"{task}/{index}/{prefix_name}_after_sha256",
			)
			if before != after:
				raise ValueError(f"{task}/{index}: {prefix_name} guard changed.")
		resolution_payload = record.get("resolutions")
		if (
			not isinstance(resolution_payload, dict)
			or set(resolution_payload) != {str(value) for value in SUPPORT_RESOLUTIONS}
		):
			raise ValueError(f"{task}/{index}: resolutions missing.")
		assets = {}
		decoded = {}
		for resolution in SUPPORT_RESOLUTIONS:
			entry = resolution_payload.get(str(resolution))
			if not isinstance(entry, dict):
				raise ValueError(f"{task}/{index}: support{resolution} missing.")
			image_path = _safe_asset(
				pack_root, entry.get("image"), f"{task}/{index}/image{resolution}"
			)
			mask_path = _safe_asset(
				pack_root,
				entry.get("indexed_mask"),
				f"{task}/{index}/mask{resolution}",
			)
			image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
			with Image.open(mask_path) as handle:
				mask = np.asarray(handle)
			if image.shape != (resolution, resolution, 3):
				raise ValueError(f"{task}/{index}: bad support{resolution} RGB shape.")
			if mask.shape != (resolution, resolution) or mask.dtype != np.uint8:
				raise ValueError(f"{task}/{index}: bad support{resolution} mask.")
			image_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
			mask_sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
			if image_sha != _require_sha(
				entry.get("image_sha256"), f"{task}/{index}/image{resolution} hash"
			):
				raise ValueError(f"{task}/{index}: support{resolution} RGB hash mismatch.")
			if mask_sha != _require_sha(
				entry.get("indexed_mask_sha256"),
				f"{task}/{index}/mask{resolution} hash",
			):
				raise ValueError(f"{task}/{index}: support{resolution} mask hash mismatch.")
			_require_sha(
				entry.get("native_clean_rgb_sha256"),
				f"{task}/{index}/native_clean_rgb{resolution}",
			)
			values = set(np.unique(mask).tolist())
			if not values.issubset({0, 1, 2}) or not {1, 2}.issubset(values):
				raise ValueError(f"{task}/{index}: support{resolution} lost a role.")
			counts = entry.get("role_pixel_counts")
			expected_counts = {
				role: int((mask == role_id).sum())
				for role_id, role in enumerate(roles, start=1)
			}
			if counts != expected_counts or any(value <= 0 for value in counts.values()):
				raise ValueError(f"{task}/{index}: role-pixel accounting mismatch.")
			asset_trace.update(image_sha.encode("ascii"))
			asset_trace.update(mask_sha.encode("ascii"))
			decoded[resolution] = (image, mask)
			assets[str(resolution)] = {
				"image_sha256": image_sha,
				"indexed_mask_sha256": mask_sha,
				"role_pixel_counts": counts,
			}
		resampling = getattr(Image, "Resampling", Image)
		up_rgb = np.asarray(Image.fromarray(decoded[64][0], mode="RGB").resize(
			(128, 128), resample=resampling.BILINEAR
		))
		up_mask = np.asarray(Image.fromarray(decoded[64][1], mode="L").resize(
			(128, 128), resample=resampling.NEAREST
		))
		if np.array_equal(up_rgb, decoded[128][0]) or np.array_equal(up_mask, decoded[128][1]):
			raise ValueError(f"{task}/{index}: native128 support equals resized source64.")
		evidence = record.get("cross_resolution_evidence")
		if not isinstance(evidence, dict) or not all(evidence.get(name) is True for name in (
			"clean128_distinct_from_bilinear64",
			"composed128_distinct_from_bilinear64",
			"mask128_distinct_from_nearest64",
		)):
			raise ValueError(f"{task}/{index}: cross-resolution evidence failed.")
		physics_hashes.append(state_sha)
		action_hashes.append(action_sha)
		reset_ordinals.append(reset)
		asset_records.append({
			"index": index,
			"physics_state_sha256": state_sha,
			"action_sequence_sha256": action_sha,
			"reset_ordinal": reset,
			"assets": assets,
		})
	if previous_reset != collection.get("reset_attempts"):
		raise ValueError(f"{task}: final reset ordinal/reset-attempt count mismatch.")
	tree_sha, tree_files = _tree_sha256(pack_root)
	return {
		"format": SUPPORT_FORMAT,
		"support_schema": SUPPORT_SCHEMA,
		"paired_support_id": paired_support_id,
		"roles": list(roles),
		"records": SUPPORT_RECORDS,
		"relative_annotations": f"support/{task}/annotations.json",
		"annotations_sha256": _file_sha256(path),
		"tree_sha256": tree_sha,
		"tree_files": tree_files,
		"physics_state_trace_sha256": sha256_json(physics_hashes),
		"action_sequence_trace_sha256": sha256_json(action_hashes),
		"reset_ordinal_trace_sha256": sha256_json(reset_ordinals),
		"asset_trace_sha256": asset_trace.hexdigest(),
		"record_identities": asset_records,
		"collection": {
			key: collection[key] for key in (
				"environment_seed", "background_seed", "action_seed", "camera_id",
				"resolutions", "pairing", "rgb_generation", "mask_generation",
				"full_composed_rgb_native_high_resolution",
				"native_foreground_and_mask_not_resized_from_64",
				"manifest_sha256", "combined_manifest_sha256",
			)
		},
	}


def _expected_gpu(spec: dict, args) -> str:
	return args.expected_gpu_a_c if spec["gpu_group"] == "a_c" else args.expected_gpu_b


def _validate_evaluation(
	payload: dict[str, Any], *, task: str, arm_key: str, pack: dict, args
) -> None:
	spec = ARM_SPECS[arm_key]
	label = f"{task}/{arm_key}"
	if payload.get("format") != EVALUATION_FORMAT:
		raise ValueError(f"{label}: wrong evaluation format.")
	if payload.get("status") != "tracker_evaluation_complete":
		raise ValueError(f"{label}: evaluation is incomplete.")
	if payload.get("task") != task or tuple(payload.get("roles", ())) != EXPECTED_ROLES[task]:
		raise ValueError(f"{label}: task/roles mismatch.")
	if (
		payload.get("arm") != spec["arm"]
		or payload.get("resolution") != spec["runtime_resolution"]
		or payload.get("support_resolution") != spec["support_resolution"]
	):
		raise ValueError(f"{label}: arm resolution contract mismatch.")
	protocol = payload.get("protocol", {})
	expected_treatment = {
		"A": "matched_native64_runtime_and_support_control",
		"B": "runtime_cutie_input_resolution_only",
		"C": "runtime_and_support_native_resolution",
	}[arm_key]
	for key, expected in {
		"tracker_only": True,
		"policy_constructed": False,
		"controller_training_steps": 0,
		"native_simulator_rgb": True,
		"dynamic_background_split": "validation",
		"gt_use": "post_track_offline_scoring_only",
		"gt_query_order": "after_cutie_track_at_same_physics_state",
		"gt_drives_tracker": False,
		"agent_observation_unchanged": True,
		"fair_representation_comparison": False,
		"allowed_claim": "perception_diagnostic",
		"disallowed_claim": "representation_advantage_vs_rgb",
		"treatment": expected_treatment,
		"cross_resolution_support": "same_paired_support_states_native_assets_selected_by_arm",
		"support_mapping": f"adapter_native{spec['support_resolution']}_direct_to_fixed_tracker448_once",
		"support_resolution": [spec["support_resolution"]] * 2,
		"high_resolution_support_sensor": spec["support_resolution"] == 128,
		"cutie_internal_tracker_size": [448, 448],
		"action_policy": "fixed_seed_uniform_action_sequence",
		"gt_render_resolution": [128, 128],
		"gt_scoring_resolution": [64, 64],
		"highres_background_semantics": (
			"policy64_selects_and_advances_background;current_frame_resized_"
			"without_clock_advance_for_highres_perception"
		),
	}.items():
		if protocol.get(key) != expected:
			raise ValueError(f"{label}: protocol {key}={protocol.get(key)!r}.")
	if protocol.get("policy_rgb_size") != [64, 64] or protocol.get("mask_output_size") != [64, 64]:
		raise ValueError(f"{label}: policy/mask canonical resolution changed.")
	if protocol.get("extra_sensor_information") is not (spec["runtime_resolution"] > 64):
		raise ValueError(f"{label}: runtime extra-sensor disclosure mismatch.")
	if protocol.get("raw_sensor_information_parity") is not (spec["runtime_resolution"] == 64):
		raise ValueError(f"{label}: raw-sensor parity disclosure mismatch.")
	evidence = payload.get("extra_sensor_evidence", {})
	if spec["runtime_resolution"] == 64:
		if evidence.get("applicable") is not False or evidence.get("frames") is None:
			raise ValueError(f"{label}: source64 extra-sensor evidence must be explicit N/A.")
		for key in (
			"distinct_frames", "distinct_frame_fraction",
			"mean_absolute_uint8_difference", "absolute_difference_trace_sha256",
		):
			if evidence.get(key) is not None:
				raise ValueError(f"{label}: source64 evidence {key} must be null.")
	else:
		if (
			evidence.get("applicable") is not True
			or evidence.get("reference") != "policy64_pil_bilinear_to_runtime_resolution"
			or not isinstance(evidence.get("distinct_frames"), int)
			or evidence["distinct_frames"] <= 0
		):
			raise ValueError(f"{label}: empirical native128 evidence missing.")
		_number(evidence.get("distinct_frame_fraction"), f"{label}/distinct_fraction", minimum=0, maximum=1)
		if evidence["distinct_frame_fraction"] <= 0:
			raise ValueError(f"{label}: no native128 frame differs from bilinear source64.")
		if _number(evidence.get("mean_absolute_uint8_difference"), f"{label}/native_difference", minimum=0) <= 0:
			raise ValueError(f"{label}: native128 has no measured extra information.")
		_require_sha(evidence.get("absolute_difference_trace_sha256"), f"{label}/native_difference_trace")

	evaluation = payload.get("evaluation", {})
	episodes = evaluation.get("episodes")
	actions = evaluation.get("actions_per_episode")
	frames_per_episode = evaluation.get("frames_per_episode")
	if not all(isinstance(value, int) and value > 0 for value in (
		episodes, actions, frames_per_episode
	)) or frames_per_episode != actions + 1:
		raise ValueError(f"{label}: invalid evaluation counts.")
	if (episodes, actions, frames_per_episode) != (20, 500, 501):
		raise ValueError(
			f"{label}: scientific preflight must be exactly 20x500 actions."
		)
	if evaluation.get("camera_id") != 0 or evaluation.get("gt_render_resolution") != 128:
		raise ValueError(f"{label}: camera/GT render protocol changed.")
	seeds = payload.get("seeds", {})
	if seeds.get("support") != pack["collection"]["environment_seed"]:
		raise ValueError(f"{label}: evaluator support seed is not bound to the pack.")
	evaluation_seed_values = {
		seeds.get("environment"), seeds.get("background"),
		seeds.get("action"), seeds.get("cutie"),
	}
	support_seed_values = {
		pack["collection"]["environment_seed"],
		pack["collection"]["background_seed"],
		pack["collection"]["action_seed"],
	}
	if (
		len(evaluation_seed_values) != 4
		or len(support_seed_values) != 3
		or evaluation_seed_values.intersection(support_seed_values)
	):
		raise ValueError(
			f"{label}: support-collection and evaluation seed domains overlap."
		)
	expected_frames = episodes * frames_per_episode
	if evidence.get("frames") != expected_frames:
		raise ValueError(f"{label}: extra-sensor frame accounting mismatch.")
	if spec["runtime_resolution"] > 64 and evidence["distinct_frames"] > expected_frames:
		raise ValueError(f"{label}: distinct native frames exceed evaluated frames.")
	if len(payload.get("episodes", ())) != episodes:
		raise ValueError(f"{label}: episode record count mismatch.")
	for index, record in enumerate(payload["episodes"]):
		if (
			record.get("episode_index") != index
			or record.get("frames") != frames_per_episode
			or record.get("actions") != actions
		):
			raise ValueError(f"{label}: episode {index} accounting mismatch.")
		for key in (
			"initial_state_sha256", "final_state_sha256", "physics_trajectory_sha256",
			"action_sequence_sha256", "background_sequence_sha256",
			"policy64_input_sequence_sha256", "native_input_sequence_sha256",
			"gt_scoring64_sequence_sha256",
		):
			_require_sha(record.get(key), f"{label}/episode{index}/{key}")
		_number(record.get("random_policy_reward"), f"{label}/episode{index}/reward")

	metrics = payload.get("metrics", {})
	if metrics.get("frames") != expected_frames:
		raise ValueError(f"{label}: metric frame count mismatch.")
	per_role = metrics.get("per_role", {})
	if tuple(per_role) != EXPECTED_ROLES[task]:
		raise ValueError(f"{label}: per-role order mismatch.")
	for role, values in per_role.items():
		if values.get("frames") != expected_frames:
			raise ValueError(f"{label}/{role}: frame count mismatch.")
		for key in (
			"valid_frame_rate", "lost_frame_rate", "empty_mask_frame_rate",
			"mean_mask_area_fraction", "gt_visible_frame_rate",
			"mean_iou_on_gt_visible_frames",
			"identity_accuracy_on_gt_visible_frames",
		):
			_number(values.get(key), f"{label}/{role}/{key}", minimum=0, maximum=1)
		for key in (
			"valid_frames", "lost_frames", "empty_mask_frames",
			"nonfinite_feature_frames", "gt_visible_frames", "identity_correct_frames",
		):
			_number(values.get(key), f"{label}/{role}/{key}", minimum=0, maximum=expected_frames)
		for count_key, rate_key in (
			("valid_frames", "valid_frame_rate"),
			("lost_frames", "lost_frame_rate"),
			("empty_mask_frames", "empty_mask_frame_rate"),
			("gt_visible_frames", "gt_visible_frame_rate"),
		):
			if not math.isclose(
				values[count_key] / expected_frames, values[rate_key],
				rel_tol=0, abs_tol=1e-12,
			):
				raise ValueError(f"{label}/{role}: {count_key}/{rate_key} disagree.")
		if values["identity_correct_frames"] > values["gt_visible_frames"]:
			raise ValueError(f"{label}/{role}: identity count exceeds visible GT.")
		if not math.isclose(
			values["identity_correct_frames"] / values["gt_visible_frames"],
			values["identity_accuracy_on_gt_visible_frames"],
			rel_tol=0, abs_tol=1e-12,
		):
			raise ValueError(f"{label}/{role}: identity count/rate disagree.")
		_number(
			values.get("max_invalid_burst"), f"{label}/{role}/max_invalid_burst",
			minimum=0, maximum=frames_per_episode,
		)
		if values.get("nonfinite_feature_frames") != 0:
			raise ValueError(f"{label}/{role}: non-finite object features.")
	role_swaps = _number(
		metrics.get("role_swap_frames"), f"{label}/role_swap_frames",
		minimum=0, maximum=expected_frames,
	)
	role_swap_rate = _number(
		metrics.get("role_swap_frame_rate"), f"{label}/role_swap_rate",
		minimum=0, maximum=1,
	)
	if not math.isclose(role_swaps / expected_frames, role_swap_rate, rel_tol=0, abs_tol=1e-12):
		raise ValueError(f"{label}: role-swap count/rate disagree.")
	latency = metrics.get("latency", {})
	for component in ("cutie_tracker", "native_rgb_render_and_composite"):
		values = latency.get(component, {})
		for key in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms"):
			_number(values.get(key), f"{label}/{component}/{key}", minimum=0)
	tracker_mean = latency["cutie_tracker"]["mean_ms"]
	if not 0 < tracker_mean <= args.max_tracker_ms_per_frame:
		raise ValueError(f"{label}: implausible tracker mean {tracker_mean}.")

	health = payload.get("runtime_health", {})
	for key in ("adapter_errors", "timeouts", "worker_restarts"):
		if health.get(key) != 0:
			raise ValueError(f"{label}: unhealthy runtime {key}={health.get(key)!r}.")
	if health.get("successful_frames") != expected_frames:
		raise ValueError(f"{label}: successful frame accounting mismatch.")
	adapter = health.get("adapter", {})
	adapter_checks = {
		"input_size": adapter.get("input_size") == [spec["runtime_resolution"]] * 2,
		"support_input_size": adapter.get("support_input_size") == [spec["support_resolution"]] * 2,
		"mask_output_size": adapter.get("mask_output_size") == [64, 64],
		"tracker_size": adapter.get("tracker_size") == [448, 448],
		"frames": adapter.get("frames") == expected_frames,
		"permanent_prompts": adapter.get("permanent_prompts") == SUPPORT_RECORDS,
		"prompt_frames": adapter.get("prompt_frames") == SUPPORT_RECORDS,
		"episode_hard_resets": adapter.get("episode_hard_resets") == episodes,
		"support_replay_frames": adapter.get("support_replay_frames") == episodes * SUPPORT_RECORDS,
		"reset_strategy": adapter.get("episode_reset_strategy") == RESET_STRATEGY,
	}
	if not all(adapter_checks.values()):
		raise ValueError(
			f"{label}: adapter contract failures "
			f"{[key for key, passed in adapter_checks.items() if not passed]}."
		)
	for key in (
		"total_ms", "ms_per_frame", "prompt_total_ms", "episode_hard_reset_total_ms",
		"support_replay_total_ms",
	):
		_number(adapter.get(key), f"{label}/adapter/{key}", minimum=0)
	if not 0 < adapter["ms_per_frame"] <= args.max_tracker_ms_per_frame:
		raise ValueError(f"{label}: adapter ms_per_frame is implausible.")
	if any(adapter[key] <= 0 for key in (
		"prompt_total_ms", "episode_hard_reset_total_ms", "support_replay_total_ms"
	)):
		raise ValueError(f"{label}: support/reset latency must be positive.")
	if not math.isclose(
		adapter["total_ms"] / expected_frames, adapter["ms_per_frame"],
		rel_tol=1e-12, abs_tol=1e-9,
	):
		raise ValueError(f"{label}: adapter total/per-frame latency disagree.")
	if not math.isclose(adapter["ms_per_frame"], tracker_mean, rel_tol=1e-5, abs_tol=1e-5):
		raise ValueError(f"{label}: tracker latency accounting disagrees.")
	_number(health.get("elapsed_seconds"), f"{label}/elapsed_seconds", minimum=0)

	provenance = payload.get("provenance", {})
	evaluator_path = Path(__file__).resolve().with_name(
		"evaluate_cutie_native_resolution_tracker.py"
	)
	if (
		not isinstance(provenance.get("evaluator"), str)
		or Path(provenance["evaluator"]).expanduser().resolve() != evaluator_path
		or provenance.get("evaluator_sha256") != _file_sha256(evaluator_path)
	):
		raise ValueError(f"{label}: evaluator implementation provenance changed.")
	if provenance.get("logical_cuda_device") != 0:
		raise ValueError(f"{label}: logical CUDA device must be zero.")
	if provenance.get("cuda_visible_devices") != _expected_gpu(spec, args):
		raise ValueError(f"{label}: assigned physical GPU mismatch.")
	if not isinstance(provenance.get("device_name"), str) or not provenance["device_name"]:
		raise ValueError(f"{label}: CUDA device name missing.")
	if provenance["device_name"] != args.expected_device_name:
		raise ValueError(f"{label}: evaluator GPU model differs from the preflight probe.")
	external = args.external_inputs
	expected_paths = {
		"video_root": external["paths"]["video_root"],
		"manifest_dir": external["paths"]["manifest_dir"],
		"oc_storm_repo": external["paths"]["oc_repo"],
		"cutie_checkpoint": external["paths"]["cutie_checkpoint"],
	}
	for key, expected in expected_paths.items():
		actual = provenance.get(key)
		if not isinstance(actual, str) or Path(actual).expanduser().resolve() != Path(
			expected
		).expanduser().resolve():
			raise ValueError(f"{label}: external path {key} is not bound to inputs.")
	checkpoint_sha = _require_sha(
		provenance.get("cutie_checkpoint_sha256"), f"{label}/checkpoint"
	)
	if checkpoint_sha != external["cutie_checkpoint"]["sha256"]:
		raise ValueError(f"{label}: Cutie checkpoint differs from external snapshot.")
	for key in ("validation_manifest_sha256", "combined_manifest_sha256"):
		_require_sha(provenance.get(key), f"{label}/{key}")
	support = provenance.get("support", {})
	expected_support = {
		"source_sha256": pack["annotations_sha256"],
		"format": SUPPORT_FORMAT,
		"support_schema": SUPPORT_SCHEMA,
		"paired_support_id": pack["paired_support_id"],
		"task": task,
		"roles": list(EXPECTED_ROLES[task]),
		"source_resolution": [spec["support_resolution"]] * 2,
		"adapter_support_input_size": [spec["support_resolution"]] * 2,
		"adapter_mask_output_size": [64, 64],
		"transform_before_adapter": "none_native_resolution_bytes",
		"support_to_tracker_mapping": f"single_adapter_resize_native{spec['support_resolution']}_to_fixed_tracker448",
		"source_bytes_reused_exactly": True,
		"native_high_resolution_support": spec["support_resolution"] == 128,
		"native_foreground_support": True,
		"native_mask_support": True,
		"full_composed_rgb_native_high_resolution": False,
		"pairing": PAIRING,
		"rgb_generation": pack["collection"]["rgb_generation"],
		"mask_generation": pack["collection"]["mask_generation"],
	}
	bad_support = {
		key: {"expected": expected, "actual": support.get(key)}
		for key, expected in expected_support.items()
		if support.get(key) != expected
	}
	if bad_support:
		raise ValueError(f"{label}: support provenance mismatch: {bad_support}.")
	for key in (
		"physics_state_trace_sha256", "action_sequence_trace_sha256",
		"reset_ordinal_trace_sha256", "asset_trace_sha256",
	):
		_require_sha(support.get(key), f"{label}/support/{key}")
	for key in (
		"physics_state_trace_sha256", "action_sequence_trace_sha256",
		"reset_ordinal_trace_sha256", "asset_trace_sha256",
	):
		if support[key] != pack[key]:
			raise ValueError(f"{label}: evaluator support {key} is not bound to the pack.")


def _pairing(left: dict, right: dict, *, compare_native_input: bool) -> dict[str, Any]:
	left_support = left["provenance"]["support"]
	right_support = right["provenance"]["support"]
	checks = {
		"task": left["task"] == right["task"],
		"roles": left["roles"] == right["roles"],
		"seeds": left["seeds"] == right["seeds"],
		"evaluation_counts": left["evaluation"] == right["evaluation"],
		"cutie_checkpoint": left["provenance"]["cutie_checkpoint_sha256"] == right["provenance"]["cutie_checkpoint_sha256"],
		"validation_manifest": left["provenance"]["validation_manifest_sha256"] == right["provenance"]["validation_manifest_sha256"],
		"combined_manifest": left["provenance"]["combined_manifest_sha256"] == right["provenance"]["combined_manifest_sha256"],
		"matched_gpu_model": left["provenance"]["device_name"] == right["provenance"]["device_name"],
		"paired_support_source": left_support["source_sha256"] == right_support["source_sha256"],
		"paired_support_id": left_support["paired_support_id"] == right_support["paired_support_id"],
		"support_physics_states": left_support["physics_state_trace_sha256"] == right_support["physics_state_trace_sha256"],
		"support_actions": left_support["action_sequence_trace_sha256"] == right_support["action_sequence_trace_sha256"],
		"support_reset_ordinals": left_support["reset_ordinal_trace_sha256"] == right_support["reset_ordinal_trace_sha256"],
		"support_asset_pair": left_support["asset_trace_sha256"] == right_support["asset_trace_sha256"],
	}
	if compare_native_input:
		checks["native128_extra_sensor_evidence"] = left.get("extra_sensor_evidence") == right.get("extra_sensor_evidence")
	common_fields = (
		"episode_index", "frames", "actions", "initial_state_sha256",
		"final_state_sha256", "physics_trajectory_sha256", "action_sequence_sha256",
		"background_sequence_sha256", "policy64_input_sequence_sha256",
		"gt_scoring64_sequence_sha256", "background_source",
		"background_start_frame_index", "background_end_frame_index",
		"random_policy_reward",
	)
	mismatches = []
	native_input_mismatches = []
	if len(left["episodes"]) != len(right["episodes"]):
		mismatches.append({"field": "episode_count"})
		if compare_native_input:
			native_input_mismatches.append({"field": "episode_count"})
	else:
		for index, (left_episode, right_episode) in enumerate(zip(left["episodes"], right["episodes"])):
			bad = [field for field in common_fields if left_episode.get(field) != right_episode.get(field)]
			if bad:
				mismatches.append({"episode_index": index, "fields": bad})
			if (
				compare_native_input
				and left_episode.get("native_input_sequence_sha256")
				!= right_episode.get("native_input_sequence_sha256")
			):
				native_input_mismatches.append({
					"episode_index": index,
					"field": "native_input_sequence_sha256",
				})
	checks["episode_common_trajectories"] = not mismatches
	if compare_native_input:
		checks["native128_input_sequences_exact"] = not native_input_mismatches
	return {
		"checks": checks,
		"episode_mismatches": mismatches,
		"native_input_mismatches": native_input_mismatches,
		"native_input_required_exact": compare_native_input,
		"intentionally_unpaired_fields": (
			[] if compare_native_input else ["native_input_sequence_sha256"]
		),
		"pass": all(checks.values()),
	}


def _relative_reduction(base: float, candidate: float) -> float:
	if base > 0:
		return (base - candidate) / base
	# A zero denominator is not evidence of improvement. Equal zero maps to
	# neutral 0; a newly introduced failure maps to finite -1 so the gate fails
	# closed while the immutable JSON remains standards-compliant.
	return 0.0 if candidate <= 0 else -1.0


def _role_contrast(base: dict, candidate: dict, role: str) -> dict[str, Any]:
	left = base["metrics"]["per_role"][role]
	right = candidate["metrics"]["per_role"][role]
	base_invalid = 1.0 - left["valid_frame_rate"]
	candidate_invalid = 1.0 - right["valid_frame_rate"]
	return {
		"base": {
			"valid_frame_rate": left["valid_frame_rate"],
			"invalid_frame_rate": base_invalid,
			"mean_iou_on_gt_visible_frames": left["mean_iou_on_gt_visible_frames"],
			"identity_accuracy_on_gt_visible_frames": left["identity_accuracy_on_gt_visible_frames"],
			"max_invalid_burst": left["max_invalid_burst"],
		},
		"candidate": {
			"valid_frame_rate": right["valid_frame_rate"],
			"invalid_frame_rate": candidate_invalid,
			"mean_iou_on_gt_visible_frames": right["mean_iou_on_gt_visible_frames"],
			"identity_accuracy_on_gt_visible_frames": right["identity_accuracy_on_gt_visible_frames"],
			"max_invalid_burst": right["max_invalid_burst"],
		},
		"delta": {
			"valid_frame_rate": right["valid_frame_rate"] - left["valid_frame_rate"],
			"invalid_rate_relative_reduction": _relative_reduction(base_invalid, candidate_invalid),
			"mean_iou_on_gt_visible_frames": right["mean_iou_on_gt_visible_frames"] - left["mean_iou_on_gt_visible_frames"],
			"identity_accuracy_on_gt_visible_frames": right["identity_accuracy_on_gt_visible_frames"] - left["identity_accuracy_on_gt_visible_frames"],
			"max_invalid_burst": left["max_invalid_burst"] - right["max_invalid_burst"],
			"max_invalid_burst_relative_reduction": _relative_reduction(
				left["max_invalid_burst"], right["max_invalid_burst"]
			),
		},
	}


def _absolute_quality(payload: dict, args) -> dict[str, Any]:
	roles = {}
	for role, values in payload["metrics"]["per_role"].items():
		checks = {
			"valid_frame_rate": values["valid_frame_rate"] >= args.min_candidate_valid_rate,
			"empty_mask_frame_rate": values["empty_mask_frame_rate"] <= args.max_candidate_empty_rate,
			"max_invalid_burst": values["max_invalid_burst"] <= args.max_candidate_invalid_burst,
			"mean_iou": values["mean_iou_on_gt_visible_frames"] >= args.min_candidate_iou,
			"identity_accuracy": values["identity_accuracy_on_gt_visible_frames"] >= args.min_candidate_identity,
			"features_finite": values["nonfinite_feature_frames"] == 0,
		}
		roles[role] = {"checks": checks, "pass": all(checks.values())}
	return {"per_role": roles, "pass": all(value["pass"] for value in roles.values())}


def _latency(payload: dict) -> dict[str, float]:
	metrics = payload["metrics"]["latency"]
	adapter = payload["runtime_health"]["adapter"]
	episodes = payload["evaluation"]["episodes"]
	prompts = adapter["permanent_prompts"]
	return {
		"tracker_ms_per_frame": adapter["ms_per_frame"],
		"same_state_render_ms_per_frame": metrics["native_rgb_render_and_composite"]["mean_ms"],
		"steady_perception_ms_per_frame": adapter["ms_per_frame"] + metrics["native_rgb_render_and_composite"]["mean_ms"],
		"initial_support_prompt_ms_per_prompt": adapter["prompt_total_ms"] / prompts,
		"episode_hard_reset_ms_per_episode": adapter["episode_hard_reset_total_ms"] / episodes,
		"support_replay_ms_per_episode": adapter["support_replay_total_ms"] / episodes,
	}


def _ratio(candidate: float, base: float) -> float | None:
	if base > 0:
		return candidate / base
	return 1.0 if candidate <= 0 else None


def _latency_contrasts(base: dict, candidate: dict, max_steady: float, args) -> dict[str, Any]:
	left, right = _latency(base), _latency(candidate)
	ratios = {
		key: _ratio(right[key], left[key]) for key in left
	}
	checks = {
		"steady_perception_ratio": ratios["steady_perception_ms_per_frame"] is not None and ratios["steady_perception_ms_per_frame"] <= max_steady,
		"initial_support_prompt_ratio": ratios["initial_support_prompt_ms_per_prompt"] is not None and ratios["initial_support_prompt_ms_per_prompt"] <= args.max_support_latency_ratio,
		"episode_hard_reset_ratio": ratios["episode_hard_reset_ms_per_episode"] is not None and ratios["episode_hard_reset_ms_per_episode"] <= args.max_support_latency_ratio,
		"support_replay_ratio": ratios["support_replay_ms_per_episode"] is not None and ratios["support_replay_ms_per_episode"] <= args.max_support_latency_ratio,
	}
	return {"base": left, "candidate": right, "ratios": ratios, "checks": checks, "pass": all(checks.values())}


def _acrobot_contrast(base: dict, candidate: dict, *, joint: bool, args) -> dict[str, Any]:
	primary = _role_contrast(base, candidate, "lower_arm")
	control = _role_contrast(base, candidate, "upper_arm")
	thresholds = {
		"min_valid_gain": args.min_joint_valid_gain if joint else args.min_support_valid_gain,
		"min_invalid_relative_reduction": args.min_joint_invalid_reduction if joint else args.min_support_invalid_reduction,
		"min_iou_gain": args.min_joint_iou_gain if joint else args.min_support_iou_gain,
		"min_burst_relative_reduction": args.min_joint_burst_reduction if joint else args.min_support_burst_reduction,
	}
	checks = {
		"lower_arm_valid_gain": primary["delta"]["valid_frame_rate"] >= thresholds["min_valid_gain"],
		"lower_arm_invalid_relative_reduction": primary["delta"]["invalid_rate_relative_reduction"] >= thresholds["min_invalid_relative_reduction"],
		"lower_arm_iou_gain": primary["delta"]["mean_iou_on_gt_visible_frames"] >= thresholds["min_iou_gain"],
		"lower_arm_burst_relative_reduction": primary["delta"]["max_invalid_burst_relative_reduction"] >= thresholds["min_burst_relative_reduction"],
		"upper_arm_valid_not_regressed": control["delta"]["valid_frame_rate"] >= -args.max_control_regression,
		"upper_arm_iou_not_regressed": control["delta"]["mean_iou_on_gt_visible_frames"] >= -args.max_control_regression,
		"upper_arm_identity_not_regressed": control["delta"]["identity_accuracy_on_gt_visible_frames"] >= -args.max_control_regression,
		"upper_arm_burst_not_regressed": -control["delta"]["max_invalid_burst"] <= args.max_control_burst_increase,
	}
	return {"primary_lower_arm": primary, "control_upper_arm": control, "thresholds": thresholds, "checks": checks, "pass": all(checks.values())}


def _cartpole_health(base: dict, candidate: dict, args) -> dict[str, Any]:
	roles = {role: _role_contrast(base, candidate, role) for role in EXPECTED_ROLES["cartpole-swingup"]}
	checks = {}
	for role, contrast in roles.items():
		checks[f"{role}_valid_not_regressed"] = contrast["delta"]["valid_frame_rate"] >= -args.max_control_regression
		checks[f"{role}_iou_not_regressed"] = contrast["delta"]["mean_iou_on_gt_visible_frames"] >= -args.max_control_regression
		checks[f"{role}_identity_not_regressed"] = contrast["delta"]["identity_accuracy_on_gt_visible_frames"] >= -args.max_control_regression
		checks[f"{role}_burst_not_regressed"] = -contrast["delta"]["max_invalid_burst"] <= args.max_control_burst_increase
	role_swap_delta = candidate["metrics"]["role_swap_frame_rate"] - base["metrics"]["role_swap_frame_rate"]
	checks["role_swap_rate_not_regressed"] = role_swap_delta <= args.max_role_swap_rate_increase
	return {"roles": roles, "role_swap_rate_delta": role_swap_delta, "checks": checks, "pass": all(checks.values())}


def _arm_summary(payload: dict, absolute: dict) -> dict[str, Any]:
	return {
		"arm": payload["arm"],
		"runtime_resolution": payload["resolution"],
		"support_resolution": payload["support_resolution"],
		"absolute_quality": absolute,
		"per_role": payload["metrics"]["per_role"],
		"role_swap_frame_rate": payload["metrics"]["role_swap_frame_rate"],
		"latency": _latency(payload),
		"runtime_health": {
			"adapter_errors": payload["runtime_health"]["adapter_errors"],
			"timeouts": payload["runtime_health"]["timeouts"],
			"worker_restarts": payload["runtime_health"]["worker_restarts"],
			"successful_frames": payload["runtime_health"]["successful_frames"],
			"adapter": payload["runtime_health"]["adapter"],
		},
		"provenance": {
			"checkpoint_sha256": payload["provenance"]["cutie_checkpoint_sha256"],
			"paired_support_id": payload["provenance"]["support"]["paired_support_id"],
			"support_source_resolution": payload["provenance"]["support"]["source_resolution"],
			"cuda_visible_devices": payload["provenance"]["cuda_visible_devices"],
			"logical_cuda_device": payload["provenance"]["logical_cuda_device"],
			"device_name": payload["provenance"]["device_name"],
		},
	}


def aggregate(args) -> dict[str, Any]:
	root = args.input_root.expanduser().resolve()
	gpu_paths = {
		"B": root / "contracts" / "gpu_b.json",
		"A_C": root / "contracts" / "gpu_a_c.json",
	}
	gpu_preflight = {}
	for key, path in gpu_paths.items():
		if not path.is_file():
			raise FileNotFoundError(path)
		value = _load(path)
		if (
			set(value) != {"logical_cuda_device", "device_name"}
			or value.get("logical_cuda_device") != 0
			or not isinstance(value.get("device_name"), str)
			or not value["device_name"]
		):
			raise ValueError(f"GPU preflight {key} is malformed.")
		gpu_preflight[key] = {
			**value,
			"physical_cuda_visible_devices": (
				args.expected_gpu_b if key == "B" else args.expected_gpu_a_c
			),
			"relative_path": path.relative_to(root).as_posix(),
			"sha256": _file_sha256(path),
		}
	if gpu_preflight["B"]["device_name"] != gpu_preflight["A_C"]["device_name"]:
		raise ValueError("B and A/C physical GPUs are not the same model.")
	args.expected_device_name = gpu_preflight["B"]["device_name"]
	implementation_path = root / "provenance" / "implementation_files.json"
	if not implementation_path.is_file():
		raise FileNotFoundError(implementation_path)
	implementation = _validate_implementation_manifest(implementation_path)
	external_path = root / "provenance" / "external_inputs.json"
	if not external_path.is_file():
		raise FileNotFoundError(external_path)
	external_inputs = validate_snapshot(_load(external_path))
	repo_root = Path(__file__).resolve().parents[2]
	if Path(external_inputs["paths"]["repo_root"]).resolve() != repo_root:
		raise ValueError("External snapshot local repository root mismatch.")
	external_inputs["relative_path"] = external_path.relative_to(root).as_posix()
	external_inputs["sha256"] = _file_sha256(external_path)
	args.external_inputs = external_inputs
	loaded: dict[str, dict[str, dict[str, Any]]] = {}
	packs, inputs = {}, [{
		"kind": "implementation_manifest",
		"relative_path": implementation["relative_path"],
		"sha256": implementation["sha256"],
	}, {
		"kind": "external_input_snapshot",
		"relative_path": external_inputs["relative_path"],
		"sha256": external_inputs["sha256"],
		"snapshot_id": external_inputs["snapshot_id"],
	}] + [
		{
			"kind": "gpu_preflight",
			"arm_group": key,
			"relative_path": value["relative_path"],
			"sha256": value["sha256"],
		} for key, value in gpu_preflight.items()
	]
	for task in args.tasks:
		pack_path = root / "support" / task / "annotations.json"
		if not pack_path.is_file():
			raise FileNotFoundError(pack_path)
		pack = _validate_support_pack(
			pack_path,
			task,
			expected_video_root=Path(
				external_inputs["paths"]["video_root"]
			).resolve(),
		)
		packs[task] = pack
		inputs.append({
			"kind": "paired_support",
			"task": task,
			"relative_path": f"support/{task}/annotations.json",
			"sha256": pack["annotations_sha256"],
			"tree_sha256": pack["tree_sha256"],
		})
		loaded[task] = {}
		for arm_key, spec in ARM_SPECS.items():
			path = root / task / spec["filename"]
			if not path.is_file():
				raise FileNotFoundError(path)
			payload = _load(path)
			_validate_evaluation(payload, task=task, arm_key=arm_key, pack=pack, args=args)
			loaded[task][arm_key] = payload
			inputs.append({
				"kind": "tracker_evaluation",
				"task": task,
				"arm_key": arm_key,
				"arm": spec["arm"],
				"relative_path": f"{task}/{spec['filename']}",
				"sha256": _file_sha256(path),
			})

	thresholds = {
		"candidate_absolute_quality": {
			"min_valid_frame_rate": args.min_candidate_valid_rate,
			"max_empty_mask_frame_rate": args.max_candidate_empty_rate,
			"max_invalid_burst": args.max_candidate_invalid_burst,
			"min_mean_iou_on_gt_visible_frames": args.min_candidate_iou,
			"min_identity_accuracy_on_gt_visible_frames": args.min_candidate_identity,
		},
		"support_effect_C_minus_B": {
			"min_lower_valid_gain": args.min_support_valid_gain,
			"min_lower_invalid_rate_relative_reduction": args.min_support_invalid_reduction,
			"min_lower_iou_gain": args.min_support_iou_gain,
			"min_lower_burst_relative_reduction": args.min_support_burst_reduction,
		},
		"joint_effect_C_minus_A": {
			"min_lower_valid_gain": args.min_joint_valid_gain,
			"min_lower_invalid_rate_relative_reduction": args.min_joint_invalid_reduction,
			"min_lower_iou_gain": args.min_joint_iou_gain,
			"min_lower_burst_relative_reduction": args.min_joint_burst_reduction,
		},
		"control_health": {
			"max_valid_iou_identity_regression": args.max_control_regression,
			"max_invalid_burst_increase_frames": args.max_control_burst_increase,
			"max_role_swap_rate_increase": args.max_role_swap_rate_increase,
		},
		"latency": {
			"max_C_over_B_steady_perception_ratio": args.max_support_steady_latency_ratio,
			"max_C_over_B_support_or_reset_ratio": args.max_support_latency_ratio,
			"max_C_over_A_steady_perception_ratio": args.max_joint_steady_latency_ratio,
			"max_tracker_ms_per_frame": args.max_tracker_ms_per_frame,
		},
		"zero_denominator_convention": (
			"relative reduction is 0 when both rates are zero and -1 when the "
			"candidate introduces failures; undefined descriptive latency ratios are null"
		),
	}

	results = {}
	engineering_pass = True
	for task, arms in loaded.items():
		pair_ab = _pairing(arms["A"], arms["B"], compare_native_input=False)
		pair_ac = _pairing(arms["A"], arms["C"], compare_native_input=False)
		pair_bc = _pairing(arms["B"], arms["C"], compare_native_input=True)
		pair_ab["intentionally_different_treatment_fields"] = [
			"runtime_resolution", "native_input_sequence_sha256",
			"protocol.treatment", "protocol.extra_sensor_information",
		]
		pair_ac["intentionally_different_treatment_fields"] = [
			"runtime_resolution", "support_resolution",
			"native_input_sequence_sha256", "selected_support_asset_hashes",
			"protocol.treatment", "protocol.extra_sensor_information",
			"protocol.high_resolution_support_sensor",
		]
		pair_bc["intentionally_different_treatment_fields"] = [
			"support_resolution", "selected_support_asset_hashes",
			"protocol.treatment", "protocol.high_resolution_support_sensor",
		]
		pairing_pass = pair_ab["pass"] and pair_ac["pass"] and pair_bc["pass"]
		engineering_pass &= pairing_pass
		absolute = {key: _absolute_quality(payload, args) for key, payload in arms.items()}
		latency_cb = _latency_contrasts(
			arms["B"], arms["C"], args.max_support_steady_latency_ratio, args
		)
		latency_ca = _latency_contrasts(
			arms["A"], arms["C"], args.max_joint_steady_latency_ratio, args
		)
		# For C-A only the steady-state ceiling is scientifically relevant.  The
		# support/reset ratios are reported, but their stricter C-B checks isolate
		# the support intervention.
		latency_ca_gate = latency_ca["checks"]["steady_perception_ratio"]
		if task == "acrobot-swingup":
			contrast_cb = _acrobot_contrast(arms["B"], arms["C"], joint=False, args=args)
			contrast_ca = _acrobot_contrast(arms["A"], arms["C"], joint=True, args=args)
		else:
			contrast_cb = _cartpole_health(arms["B"], arms["C"], args)
			contrast_ca = _cartpole_health(arms["A"], arms["C"], args)
		candidate_quality = absolute["C"]["pass"]
		task_go = (
			pairing_pass and candidate_quality and contrast_cb["pass"]
			and contrast_ca["pass"] and latency_cb["pass"] and latency_ca_gate
		)
		results[task] = {
			"paired_support": packs[task],
			"pairing": {
				"A_vs_B": pair_ab,
				"A_vs_C": pair_ac,
				"B_vs_C": pair_bc,
				"pass": pairing_pass,
			},
			"arms": {
				key: _arm_summary(payload, absolute[key]) for key, payload in arms.items()
			},
			"contrasts": {
				"C_minus_B_support_mismatch_causal": {
					"estimand": "support64_to_support128_at_fixed_runtime128",
					"quality": contrast_cb,
					"latency": latency_cb,
					"candidate_absolute_quality_pass": candidate_quality,
					"go": pairing_pass and candidate_quality and contrast_cb["pass"] and latency_cb["pass"],
				},
				"C_minus_A_joint_runtime_and_support": {
					"estimand": "joint_runtime64_support64_to_runtime128_support128",
					"quality": contrast_ca,
					"latency": latency_ca,
					"only_steady_latency_is_gated_here": True,
					"candidate_absolute_quality_pass": candidate_quality,
					"go": pairing_pass and candidate_quality and contrast_ca["pass"] and latency_ca_gate,
				},
				"B_minus_A_runtime_only_report": {
					"estimand": "runtime64_to_runtime128_at_fixed_support64",
					"decision_scope": "descriptive_not_an_independent_overall_gate",
					"per_role": {
						role: _role_contrast(arms["A"], arms["B"], role)
						for role in EXPECTED_ROLES[task]
					},
					"latency": _latency_contrasts(
						arms["A"], arms["B"], args.max_joint_steady_latency_ratio, args
					),
				},
			},
			"go": task_go,
		}

	full_scope = set(args.tasks) == set(TASKS)
	all_task_go = all(value["go"] for value in results.values())
	if not engineering_pass:
		status = "native_support_three_arm_engineering_fail"
		recommendation = "stop_pairing_or_runtime_invalid_do_not_interpret_or_train"
	elif full_scope and all_task_go:
		status = "native_support_three_arm_science_go"
		recommendation = "eligible_for_separately_preregistered_controller_training_pilot"
	elif not full_scope and all_task_go:
		status = "native_support_candidate_pass_cartpole_health_missing"
		recommendation = "run_cartpole_health_control_before_any_training_go"
	else:
		status = "native_support_three_arm_no_go"
		recommendation = "do_not_train_controller_from_this_perception_treatment"
	return {
		"format": FORMAT,
		"status": status,
		"engineering_pass": engineering_pass,
		"full_scientific_scope_present": full_scope,
		"all_required_tasks_go": engineering_pass and full_scope and all_task_go,
		"recommendation": recommendation,
		"scope": {
			"tracker_only": True,
			"tasks": list(args.tasks),
			"arms": {
				key: {
					"runtime_resolution": spec["runtime_resolution"],
					"support_resolution": spec["support_resolution"],
				} for key, spec in ARM_SPECS.items()
			},
			"C_minus_B_causal_estimand": "support_resolution_at_fixed_runtime128",
			"C_minus_A_estimand": "joint_runtime_and_support_resolution",
			"B_minus_A_estimand": "runtime_resolution_at_fixed_support64_report_only",
			"conditional_on_one_cutie_checkpoint_and_one_paired_support_pack": True,
			"episode_uncertainty_not_claimed": True,
			"privileged_simulator_masks_in_support": True,
			"native128_support_semantics": (
				"native simulator foreground and native segmentation mask at 128; "
				"the already-selected native64 video background is resized without clock advance"
			),
			"full_composed_rgb_native_high_resolution": False,
			"runtime_candidate_has_extra_sensor_information": True,
			"raw_sensor_information_parity": False,
			"fair_representation_comparison": False,
			"agent_observation_unchanged": True,
			"allowed_claim": "perception_diagnostic_and_conditional_support_mismatch_effect",
			"disallowed_claims": [
				"representation_advantage_vs_rgb",
				"training_seed_generalization",
				"support_seed_generalization",
				"controller_reward_improvement",
			],
			"automatic_controller_training_launched": False,
		},
		"thresholds": thresholds,
		"inputs": inputs,
		"implementation": implementation,
		"external_inputs": external_inputs,
		"gpu_preflight": gpu_preflight,
		"tasks": results,
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
	parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
	parser.add_argument("--expected-gpu-b", required=True)
	parser.add_argument("--expected-gpu-a-c", required=True)
	parser.add_argument("--min-support-valid-gain", type=float, default=0.03)
	parser.add_argument("--min-support-invalid-reduction", type=float, default=0.10)
	parser.add_argument("--min-support-iou-gain", type=float, default=0.03)
	parser.add_argument("--min-support-burst-reduction", type=float, default=0.10)
	parser.add_argument("--min-joint-valid-gain", type=float, default=0.05)
	parser.add_argument("--min-joint-invalid-reduction", type=float, default=0.20)
	parser.add_argument("--min-joint-iou-gain", type=float, default=0.05)
	parser.add_argument("--min-joint-burst-reduction", type=float, default=0.20)
	parser.add_argument("--max-control-regression", type=float, default=0.02)
	parser.add_argument("--max-control-burst-increase", type=int, default=25)
	parser.add_argument("--max-role-swap-rate-increase", type=float, default=0.005)
	parser.add_argument("--min-candidate-valid-rate", type=float, default=0.50)
	parser.add_argument("--max-candidate-empty-rate", type=float, default=0.50)
	parser.add_argument("--max-candidate-invalid-burst", type=int, default=250)
	parser.add_argument("--min-candidate-iou", type=float, default=0.05)
	parser.add_argument("--min-candidate-identity", type=float, default=0.50)
	parser.add_argument("--max-support-steady-latency-ratio", type=float, default=1.10)
	parser.add_argument("--max-support-latency-ratio", type=float, default=1.50)
	parser.add_argument("--max-joint-steady-latency-ratio", type=float, default=1.50)
	parser.add_argument("--max-tracker-ms-per-frame", type=float, default=800.0)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args(argv)
	if len(args.tasks) != len(set(args.tasks)):
		parser.error("--tasks must be unique")
	for name in ("expected_gpu_b", "expected_gpu_a_c"):
		if not getattr(args, name).isdigit():
			parser.error(f"--{name.replace('_', '-')} must be a physical GPU index")
	if args.expected_gpu_b == args.expected_gpu_a_c:
		parser.error("B and A/C must use different physical GPUs for paired B/C timing")
	unit_interval = (
		"min_support_valid_gain", "min_support_invalid_reduction",
		"min_support_iou_gain", "min_support_burst_reduction",
		"min_joint_valid_gain", "min_joint_invalid_reduction",
		"min_joint_iou_gain", "min_joint_burst_reduction",
		"max_control_regression", "max_role_swap_rate_increase",
		"min_candidate_valid_rate", "max_candidate_empty_rate",
		"min_candidate_iou", "min_candidate_identity",
	)
	for name in unit_interval:
		if not 0 <= getattr(args, name) <= 1:
			parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
	if args.max_candidate_invalid_burst < 0:
		parser.error("--max-candidate-invalid-burst must be non-negative")
	if args.max_control_burst_increase < 0:
		parser.error("--max-control-burst-increase must be non-negative")
	for name in (
		"max_support_steady_latency_ratio", "max_support_latency_ratio",
		"max_joint_steady_latency_ratio", "max_tracker_ms_per_frame",
	):
		if getattr(args, name) <= 0:
			parser.error(f"--{name.replace('_', '-')} must be positive")
	return args


def main(argv=None) -> int:
	args = parse_args(argv)
	if args.output.exists():
		raise FileExistsError(args.output)
	payload = aggregate(args)
	_atomic_write(args.output, payload)
	print("CUTIE_NATIVE_SUPPORT_THREE_ARM_COMPLETE", json.dumps({
		"status": payload["status"],
		"engineering_pass": payload["engineering_pass"],
		"go": payload["all_required_tasks_go"],
		"output": str(args.output.expanduser().resolve()),
	}, sort_keys=True), flush=True)
	return 0 if payload["engineering_pass"] else 4


if __name__ == "__main__":
	raise SystemExit(main())
