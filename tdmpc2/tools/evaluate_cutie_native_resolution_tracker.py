"""Tracker-only native-resolution preflight for Cutie on articulated DMC tasks.

This evaluator never constructs a policy or trains a controller. One process
evaluates one explicit runtime/support-resolution arm. Legacy v1 evaluations
retain the frozen source64 support pack. Paired-support v2 evaluations select
native64 or native128 prompts generated from the same accepted simulator
states. The strict aggregator verifies physical trajectories and, for the two
runtime128 arms, exact input-frame digests before interpreting any difference.

Cutie maps each selected support image directly to its fixed 448x448 internal
grid exactly once. Runtime masks are always decoded onto the fixed 64x64 policy
grid. Runtime MuJoCo segmentation is queried *after* each Cutie step and is
used only for offline IoU/identity scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


os.environ.setdefault("MUJOCO_GL", "egl")
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


FORMAT = "cutie_native_resolution_tracker_evaluation_v1"
PAIRED_FORMAT = "cutie_native_support_tracker_evaluation_v2"
TASKS = ("acrobot-swingup", "cartpole-swingup")
ROLE_NAMES = {
	"acrobot-swingup": ("upper_arm", "lower_arm"),
	"cartpole-swingup": ("cart", "pole"),
}
PRIMARY_ROLE = {
	"acrobot-swingup": "lower_arm",
	"cartpole-swingup": "pole",
}
RESOLUTIONS = (64, 128, 256)
SPLIT = "validation"
CAMERA_ID = 0
SOURCE_SUPPORT_SIZE = 64
SUPPORT_PROMPTS = 6


class _Config(SimpleNamespace):
	def get(self, name, default=None):
		return getattr(self, name, default)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _digest_update_array(digest, label: bytes, value: np.ndarray) -> None:
	array = np.ascontiguousarray(value)
	digest.update(label)
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())


def _find_physics(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		physics = getattr(current, "physics", None)
		if physics is not None and callable(getattr(physics, "render", None)):
			return physics
		current = getattr(current, "env", None)
	raise RuntimeError("Could not find dm_control physics in the environment chain.")


def _find_wrapper(env, wrapper_type):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		if isinstance(current, wrapper_type):
			return current
		current = getattr(current, "env", None)
	raise RuntimeError(f"{wrapper_type.__name__} is missing from the environment chain.")


def _latest_policy_rgb(observation) -> np.ndarray:
	try:
		value = observation[-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy()
	except AttributeError:
		value = np.moveaxis(np.asarray(observation)[-3:], 0, -1)
	if value.shape != (64, 64, 3) or value.dtype != np.uint8:
		raise RuntimeError(
			f"Policy observation must expose newest uint8 [64,64,3], got "
			f"{value.shape} {value.dtype}."
		)
	return np.array(value, dtype=np.uint8, order="C", copy=True)


def _render_clean_rgb(physics, resolution: int) -> np.ndarray:
	frame = np.asarray(
		physics.render(
			height=resolution,
			width=resolution,
			camera_id=CAMERA_ID,
		)
	)
	if frame.shape != (resolution, resolution, 3) or frame.dtype != np.uint8:
		raise RuntimeError(
			"Native MuJoCo RGB render must be uint8 HWC, got "
			f"{frame.shape} {frame.dtype}."
		)
	return np.array(frame, dtype=np.uint8, order="C", copy=True)


def _role_gt_masks(physics, resolution: int, selections) -> np.ndarray:
	"""Render same-state GT after tracking; never return it to Cutie."""
	from dm_control.mujoco.wrapper.mjbindings import enums

	segmentation = np.asarray(
		physics.render(
			height=resolution,
			width=resolution,
			camera_id=CAMERA_ID,
			segmentation=True,
		)
	)
	if segmentation.shape != (resolution, resolution, 2):
		raise RuntimeError(
			"MuJoCo segmentation must be [H,W,2], got "
			f"{segmentation.shape}."
		)
	object_types = {
		"geom": int(enums.mjtObj.mjOBJ_GEOM),
		"site": int(enums.mjtObj.mjOBJ_SITE),
	}
	masks = []
	for selected in selections:
		mask = np.zeros((resolution, resolution), dtype=np.bool_)
		for object_type, object_id in selected:
			mask |= (
				(segmentation[..., 0] == int(object_id))
				& (segmentation[..., 1] == object_types[object_type])
			)
		masks.append(mask)
	output = np.stack(masks, axis=0)
	if np.any(output.sum(axis=0) > 1):
		raise RuntimeError("Configured GT role masks overlap.")
	return output


def _load_frozen_support(
	annotation_path: Path,
	*,
	task: str,
	expected_seed: int,
	support_resolution: int = 64,
):
	"""Load legacy source64 or one arm of a same-state paired native pack."""
	from tdmpc2.perception.cutie_oc_adapter import (
		load_indexed_support_prompts,
		load_paired_native_support_prompts,
	)
	from tdmpc2.common.cutie_paired_support import FORMAT as PAIRED_SUPPORT_FORMAT

	path = annotation_path.expanduser().resolve()
	payload = json.loads(path.read_text(encoding="utf-8"))
	if payload.get("format") == PAIRED_SUPPORT_FORMAT:
		source = load_paired_native_support_prompts(
			path,
			role_names=ROLE_NAMES[task],
			expected_task=task,
			support_resolution=support_resolution,
			expected_seed=expected_seed,
			allow_simulator_support=True,
		)
		metadata = source.metadata
		return source, {
			"source": str(path),
			"source_sha256": _file_sha256(path),
			"format": metadata["format"],
			"support_schema": metadata["support_schema"],
			"paired_support_id": metadata["paired_support_id"],
			"task": metadata["task"],
			"roles": list(source.role_names),
			"source_resolution": metadata["source_resolution"],
			"adapter_support_input_size": metadata["source_resolution"],
			"adapter_mask_output_size": [64, 64],
			"transform_before_adapter": "none_native_resolution_bytes",
			"support_to_tracker_mapping": (
				f"single_adapter_resize_native{support_resolution}_to_fixed_tracker448"
			),
			"source_bytes_reused_exactly": True,
			"native_high_resolution_support": support_resolution == 128,
			"native_foreground_support": metadata["native_foreground_support"],
			"native_mask_support": metadata["native_mask_support"],
			"full_composed_rgb_native_high_resolution": metadata[
				"full_composed_rgb_native_high_resolution"
			],
			"pairing": metadata["pairing"],
			"rgb_generation": metadata["rgb_generation"],
			"mask_generation": metadata["mask_generation"],
			"physics_state_trace_sha256": metadata[
				"physics_state_trace_sha256"
			],
			"action_sequence_trace_sha256": metadata[
				"action_sequence_trace_sha256"
			],
			"reset_ordinal_trace_sha256": metadata[
				"reset_ordinal_trace_sha256"
			],
			"asset_trace_sha256": metadata["asset_trace_sha256"],
		}
	if support_resolution != 64:
		raise ValueError(
			"Legacy cutie_indexed_mask_support_v1 can only provide source64 support."
		)
	collection = payload.get("collection", {})
	if collection.get("image_size") != [64, 64]:
		raise ValueError("Cross-resolution support source must declare image_size=[64,64].")
	if int(collection.get("seed", -1)) != int(expected_seed):
		raise ValueError(
			f"Support seed {collection.get('seed')!r} != expected {expected_seed}."
		)
	roles = ROLE_NAMES[task]
	source = load_indexed_support_prompts(
		path,
		role_names=roles,
		expected_task=task,
		expected_records=SUPPORT_PROMPTS,
		allow_simulator_support=True,
	)
	records = []
	for index, (source_frame, source_mask) in enumerate(
		zip(source.frames, source.masks)
	):
		values = set(np.unique(source_mask).tolist())
		if not values.issubset({0, 1, 2}) or not {1, 2}.issubset(values):
			raise RuntimeError(
				f"Source support mask {index} lost a role: values={sorted(values)}."
			)
		records.append({
			"index": index,
			"source_rgb_sha256": _array_sha256(source_frame),
			"source_mask_sha256": _array_sha256(source_mask),
			"consumed_rgb_sha256": _array_sha256(source_frame),
			"consumed_mask_sha256": _array_sha256(source_mask),
			"bytes_reused_exactly": True,
			"role_pixel_counts": {
				role: int((source_mask == role_id).sum())
				for role_id, role in enumerate(roles, start=1)
			},
		})
	return source, {
		"source": str(path),
		"source_sha256": _file_sha256(path),
		"format": payload.get("format"),
		"support_schema": collection.get("support_schema"),
		"paired_support_id": None,
		"task": task,
		"roles": list(source.role_names),
		"source_resolution": [64, 64],
		"adapter_support_input_size": [64, 64],
		"adapter_mask_output_size": [64, 64],
		"transform_before_adapter": "none_byte_exact_source64",
		"support_to_tracker_mapping": "single_adapter_resize_source64_to_fixed_tracker448",
		"source_bytes_reused_exactly": True,
		"native_high_resolution_support": False,
		"native_foreground_support": False,
		"native_mask_support": False,
		"full_composed_rgb_native_high_resolution": False,
		"records": records,
	}


def _scoring_masks(native_masks: np.ndarray) -> np.ndarray:
	"""Map evaluation-only native GT to the fixed 64x64 policy mask grid."""
	if native_masks.ndim != 3:
		raise ValueError(f"Native GT masks must be [K,H,W], got {native_masks.shape}.")
	if tuple(native_masks.shape[1:]) == (64, 64):
		return np.array(native_masks, dtype=np.bool_, order="C", copy=True)
	resampling = getattr(Image, "Resampling", Image).NEAREST
	resized = []
	for mask in native_masks:
		value = np.asarray(
			Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
				(64, 64), resample=resampling
			),
			dtype=np.uint8,
		)
		resized.append(value > 0)
	output = np.stack(resized, axis=0)
	if np.any(output.sum(axis=0) > 1):
		raise RuntimeError("Downsampled scoring GT role masks overlap.")
	return np.ascontiguousarray(output)


@dataclass
class _RoleTotals:
	frames: int = 0
	valid: int = 0
	lost: int = 0
	empty: int = 0
	nonfinite: int = 0
	mask_area: int = 0
	gt_visible: int = 0
	gt_area: int = 0
	iou_sum_visible: float = 0.0
	identity_correct: int = 0
	current_invalid_burst: int = 0
	max_invalid_burst: int = 0
	max_invalid_event: dict[str, Any] | None = None


class MetricAccumulator:
	"""Pure metric accumulator shared with the dependency-light contract test."""

	def __init__(self, role_names: tuple[str, ...], resolution: int):
		self.role_names = tuple(role_names)
		self.resolution = int(resolution)
		self.roles = [_RoleTotals() for _ in role_names]
		self.frames = 0
		self.all_roles_valid_frames = 0
		self.gt_all_roles_visible_frames = 0
		self.role_swap_frames = 0
		self.current_any_invalid_burst = 0
		self.max_any_invalid_burst = 0
		self.max_any_invalid_event = None
		self.tracker_ms: list[float] = []
		self.native_rgb_ms: list[float] = []
		self.gt_scoring_ms: list[float] = []
		self._episode_index = -1

	def begin_episode(self, episode_index: int) -> None:
		self._episode_index = int(episode_index)
		self.current_any_invalid_burst = 0
		for role in self.roles:
			role.current_invalid_burst = 0

	@staticmethod
	def _iou_matrix(predicted: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		role_count = predicted.shape[0]
		iou = np.zeros((role_count, role_count), dtype=np.float64)
		intersections = np.zeros((role_count, role_count), dtype=np.int64)
		for pred_index in range(role_count):
			for gt_index in range(role_count):
				intersection = int(np.logical_and(
					predicted[pred_index], gt[gt_index]
				).sum())
				union = int(np.logical_or(
					predicted[pred_index], gt[gt_index]
				).sum())
				intersections[pred_index, gt_index] = intersection
				iou[pred_index, gt_index] = intersection / union if union else 0.0
		return iou, intersections

	def record(
		self,
		*,
		step_index: int,
		predicted: np.ndarray,
		gt: np.ndarray,
		lost: np.ndarray,
		feature_finite: np.ndarray,
		tracker_ms: float,
		native_rgb_ms: float,
		gt_scoring_ms: float,
	) -> None:
		role_count = len(self.role_names)
		if predicted.shape != (role_count, self.resolution, self.resolution):
			raise ValueError(f"Unexpected predicted mask shape {predicted.shape}.")
		if gt.shape != predicted.shape:
			raise ValueError(f"GT/predicted mask shape mismatch: {gt.shape}.")
		for value, name in (
			(lost, "lost"), (feature_finite, "feature_finite")
		):
			if np.asarray(value).shape != (role_count,):
				raise ValueError(f"{name} must have shape [{role_count}].")
		if any(not math.isfinite(value) or value < 0 for value in (
			tracker_ms, native_rgb_ms, gt_scoring_ms
		)):
			raise ValueError("All latency measurements must be finite and non-negative.")
		predicted = np.asarray(predicted, dtype=np.bool_)
		gt = np.asarray(gt, dtype=np.bool_)
		lost = np.asarray(lost, dtype=np.bool_)
		feature_finite = np.asarray(feature_finite, dtype=np.bool_)
		nonempty = predicted.reshape(role_count, -1).any(axis=1)
		valid = (~lost) & nonempty & feature_finite
		gt_visible = gt.reshape(role_count, -1).any(axis=1)
		iou, intersections = self._iou_matrix(predicted, gt)

		self.frames += 1
		self.all_roles_valid_frames += int(bool(valid.all()))
		self.gt_all_roles_visible_frames += int(bool(gt_visible.all()))
		self.tracker_ms.append(float(tracker_ms))
		self.native_rgb_ms.append(float(native_rgb_ms))
		self.gt_scoring_ms.append(float(gt_scoring_ms))
		if bool(valid.all()):
			self.current_any_invalid_burst = 0
		else:
			self.current_any_invalid_burst += 1
			if self.current_any_invalid_burst > self.max_any_invalid_burst:
				self.max_any_invalid_burst = self.current_any_invalid_burst
				self.max_any_invalid_event = {
					"episode_index": self._episode_index,
					"start_step": int(step_index - self.current_any_invalid_burst + 1),
					"end_step": int(step_index),
					"length": int(self.current_any_invalid_burst),
					"invalid_roles": [
						name for name, present in zip(self.role_names, valid) if not present
					],
				}

		if role_count == 2 and bool(gt_visible.all()):
			diagonal = int(intersections[0, 0] + intersections[1, 1])
			cross = int(intersections[0, 1] + intersections[1, 0])
			self.role_swap_frames += int(cross > 0 and cross > diagonal)

		for index, totals in enumerate(self.roles):
			totals.frames += 1
			totals.valid += int(valid[index])
			totals.lost += int(lost[index])
			totals.empty += int(not nonempty[index])
			totals.nonfinite += int(not feature_finite[index])
			totals.mask_area += int(predicted[index].sum())
			if gt_visible[index]:
				totals.gt_visible += 1
				totals.gt_area += int(gt[index].sum())
				totals.iou_sum_visible += float(iou[index, index])
				other = np.delete(intersections[index], index)
				own = int(intersections[index, index])
				totals.identity_correct += int(
					own > 0 and (other.size == 0 or own > int(other.max()))
				)
			if valid[index]:
				totals.current_invalid_burst = 0
			else:
				totals.current_invalid_burst += 1
				if totals.current_invalid_burst > totals.max_invalid_burst:
					totals.max_invalid_burst = totals.current_invalid_burst
					totals.max_invalid_event = {
						"episode_index": self._episode_index,
						"start_step": int(
							step_index - totals.current_invalid_burst + 1
						),
						"end_step": int(step_index),
						"length": int(totals.current_invalid_burst),
						"reasons": [
							name for name, failed in (
								("lost", lost[index]),
								("empty_mask", not nonempty[index]),
								("nonfinite_feature", not feature_finite[index]),
							) if failed
						],
					}

	@staticmethod
	def _latency(values: list[float]) -> dict[str, float]:
		array = np.asarray(values, dtype=np.float64)
		if array.size == 0 or not np.isfinite(array).all():
			raise RuntimeError("Latency series is empty or non-finite.")
		return {
			"mean_ms": float(array.mean()),
			"median_ms": float(np.median(array)),
			"p95_ms": float(np.percentile(array, 95)),
			"p99_ms": float(np.percentile(array, 99)),
			"max_ms": float(array.max()),
		}

	def summary(self) -> dict[str, Any]:
		if self.frames <= 0:
			raise RuntimeError("No tracker frames were recorded.")
		pixels = self.resolution * self.resolution
		per_role = {}
		for name, totals in zip(self.role_names, self.roles):
			per_role[name] = {
				"frames": totals.frames,
				"valid_frames": totals.valid,
				"valid_frame_rate": totals.valid / totals.frames,
				"lost_frames": totals.lost,
				"lost_frame_rate": totals.lost / totals.frames,
				"empty_mask_frames": totals.empty,
				"empty_mask_frame_rate": totals.empty / totals.frames,
				"nonfinite_feature_frames": totals.nonfinite,
				"mean_mask_area_pixels": totals.mask_area / totals.frames,
				"mean_mask_area_fraction": totals.mask_area / totals.frames / pixels,
				"gt_visible_frames": totals.gt_visible,
				"gt_visible_frame_rate": totals.gt_visible / totals.frames,
				"mean_gt_area_pixels_when_visible": (
					totals.gt_area / totals.gt_visible if totals.gt_visible else None
				),
				"mean_iou_on_gt_visible_frames": (
					totals.iou_sum_visible / totals.gt_visible
					if totals.gt_visible else None
				),
				"identity_correct_frames": totals.identity_correct,
				"identity_accuracy_on_gt_visible_frames": (
					totals.identity_correct / totals.gt_visible
					if totals.gt_visible else None
				),
				"max_invalid_burst": totals.max_invalid_burst,
				"max_invalid_burst_event": totals.max_invalid_event,
			}
		return {
			"frames": self.frames,
			"all_roles_valid_frame_rate": self.all_roles_valid_frames / self.frames,
			"all_roles_gt_visible_frame_rate": (
				self.gt_all_roles_visible_frames / self.frames
			),
			"max_invalid_burst_any_role": self.max_any_invalid_burst,
			"max_invalid_burst_any_role_event": self.max_any_invalid_event,
			"role_swap_frames": self.role_swap_frames,
			"role_swap_frame_rate": self.role_swap_frames / self.frames,
			"per_role": per_role,
			"latency": {
				"cutie_tracker": self._latency(self.tracker_ms),
				"native_rgb_render_and_composite": self._latency(self.native_rgb_ms),
				"post_track_gt_render_and_scoring": self._latency(self.gt_scoring_ms),
			},
		}


def _quality_gates(summary: dict, args) -> dict[str, Any]:
	role_gates = {}
	for role, metrics in summary["per_role"].items():
		iou = metrics["mean_iou_on_gt_visible_frames"]
		identity = metrics["identity_accuracy_on_gt_visible_frames"]
		checks = {
			"valid_frame_rate": metrics["valid_frame_rate"] >= args.min_valid_rate,
			"empty_mask_frame_rate": (
				metrics["empty_mask_frame_rate"] <= args.max_empty_rate
			),
			"max_invalid_burst": (
				metrics["max_invalid_burst"] <= args.max_invalid_burst
			),
			"gt_visibility_available": metrics["gt_visible_frames"] > 0,
			"mean_iou": iou is not None and iou >= args.min_mean_iou,
			"identity_accuracy": (
				identity is not None and identity >= args.min_identity_accuracy
			),
			"features_finite": metrics["nonfinite_feature_frames"] == 0,
		}
		role_gates[role] = {"checks": checks, "pass": all(checks.values())}
	checks = {
		"expected_frames": summary["frames"] == args.episodes * (args.steps + 1),
		"all_roles": all(value["pass"] for value in role_gates.values()),
	}
	return {
		"thresholds": {
			"min_valid_rate": args.min_valid_rate,
			"max_empty_rate": args.max_empty_rate,
			"max_invalid_burst": args.max_invalid_burst,
			"min_mean_iou_on_gt_visible_frames": args.min_mean_iou,
			"min_identity_accuracy_on_gt_visible_frames": args.min_identity_accuracy,
		},
		"per_role": role_gates,
		"checks": checks,
		"quality_pass": all(checks.values()),
	}


def _validate_args(args) -> None:
	if args.resolution not in RESOLUTIONS:
		raise ValueError(f"resolution must be one of {RESOLUTIONS}.")
	if args.episodes < 1 or args.steps < 1:
		raise ValueError("episodes and steps must be positive.")
	if args.tracker_size < 1:
		raise ValueError("tracker-size must be positive.")
	if args.support_resolution not in (64, 128):
		raise ValueError("support-resolution must be 64 or 128.")
	if args.support_resolution == 128 and args.resolution != 128:
		raise ValueError(
			"Native128 support is restricted to the runtime128/support128 arm."
		)
	if len({args.env_seed, args.background_seed, args.action_seed, args.cutie_seed}) != 4:
		raise ValueError("Environment/background/action/Cutie seed domains must differ.")
	for path in (args.oc_storm_repo, args.cutie_checkpoint, args.support_annotations):
		if not path.expanduser().exists():
			raise FileNotFoundError(path)
	if not args.video_root.expanduser().is_dir():
		raise FileNotFoundError(args.video_root)
	if args.output.exists():
		raise FileExistsError(args.output)
	for value, name in (
		(args.min_valid_rate, "min-valid-rate"),
		(args.max_empty_rate, "max-empty-rate"),
		(args.min_mean_iou, "min-mean-iou"),
		(args.min_identity_accuracy, "min-identity-accuracy"),
	):
		if not 0 <= value <= 1:
			raise ValueError(f"{name} must be in [0,1].")
	if args.max_invalid_burst < 0:
		raise ValueError("max-invalid-burst must be non-negative.")


def _prepare_tracking_environment(
	args, roles, dmcontrol_env, task_specs, catalog_fn, selected_objects_fn,
):
	"""Construct and validate the environment, closing it on setup failure."""
	env_config = _Config(
		task=args.task,
		obs="rgb",
		seed=args.env_seed,
		multitask=False,
		video_background_enabled=True,
		video_background_root=str(args.video_root.expanduser().resolve()),
		video_background_manifest_dir=(
			None if args.manifest_dir is None
			else str(args.manifest_dir.expanduser().resolve())
		),
		video_background_split=SPLIT,
		video_background_strength=1.0,
		video_background_total_frames=args.background_total_frames,
		video_background_source_cache_size=args.background_cache_size,
		video_background_seed=args.background_seed,
		flat_anchor=False,
	)
	env = dmcontrol_env.make_env(env_config)
	try:
		background = _find_wrapper(
			env, dmcontrol_env.ColorMultiVideoBackgroundWrapper
		)
		if background.active_split != SPLIT:
			raise RuntimeError(
				f"Background split {background.active_split!r} != {SPLIT!r}."
			)
		physics = _find_physics(env)
		spec = task_specs[args.task]
		if tuple(spec.roles) != roles:
			raise RuntimeError(
				f"Task role contract changed: {spec.roles} != {roles}."
			)
		catalog = catalog_fn(physics)
		selections = tuple(
			selected_objects_fn(catalog, selector_spec)
			for selector_spec in spec.selectors
		)
		if set(selections[0]).intersection(selections[1]):
			raise RuntimeError("Configured task role selectors overlap.")
		return env, background, physics, selections
	except BaseException:
		close = getattr(env, "close", None)
		if callable(close):
			try:
				close()
			except Exception:
				pass
		raise


def evaluate(args) -> dict[str, Any]:
	import torch
	# ``envs.dmcontrol`` constructs wrappers imported through the top-level
	# ``envs`` namespace.  Import the factory module itself and use the exact
	# wrapper class object bound there: loading the same source once as
	# ``envs...`` and again as ``tdmpc2.envs...`` creates distinct Python class
	# identities, making a strict ``isinstance`` traversal falsely report that
	# the wrapper is absent.
	import envs.dmcontrol as dmcontrol_env
	from tdmpc2.perception.cutie_oc_adapter import CutieOCAdapter, CutieOCConfig
	from tdmpc2.tools.collect_cutie_multitask_support import (
		TASK_BY_NAME,
		_catalog,
		_selected_objects,
	)

	_validate_args(args)
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required for the official Cutie preflight.")
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.cutie_seed)
	torch.cuda.manual_seed_all(args.cutie_seed)
	roles = ROLE_NAMES[args.task]
	support, support_provenance = _load_frozen_support(
		args.support_annotations,
		task=args.task,
		expected_seed=args.expected_support_seed,
		support_resolution=args.support_resolution,
	)
	paired_support = support_provenance.get("paired_support_id") is not None
	if paired_support and (args.resolution, args.support_resolution) not in {
		(64, 64), (128, 64), (128, 128)
	}:
		raise ValueError(
			"Paired native support v2 permits only A=R64/S64, B=R128/S64, "
			"or C=R128/S128."
		)
	adapter_config = CutieOCConfig(
		repo_path=args.oc_storm_repo.expanduser().resolve(),
		checkpoint_path=args.cutie_checkpoint.expanduser().resolve(),
		role_names=roles,
		model_size=args.model_size,
		device="cuda:0",
		output_device="cpu",
		expected_input_size=(args.resolution, args.resolution),
		support_input_size=(
			args.support_resolution, args.support_resolution
		),
		mask_output_size=(64, 64),
		tracker_size=(args.tracker_size, args.tracker_size),
		foreground_queries=8,
		amp=not args.disable_amp,
		return_object_features=True,
		object_schema="generic_indexed_v1",
	)
	adapter = CutieOCAdapter(adapter_config)
	adapter.add_support_prompts(support)

	env, background, physics, selections = _prepare_tracking_environment(
		args, roles, dmcontrol_env, TASK_BY_NAME, _catalog, _selected_objects
	)

	action_rng = np.random.default_rng(args.action_seed)
	# The production world-model contract always receives 64x64 decoded masks,
	# regardless of the RGB perception input resolution.
	accumulator = MetricAccumulator(roles, SOURCE_SUPPORT_SIZE)
	episode_records = []
	extra_sensor_distinct_frames = 0
	extra_sensor_abs_diff_sum = 0
	extra_sensor_scalar_count = 0
	extra_sensor_max_abs_diff = 0
	extra_sensor_trace = hashlib.sha256()
	started = perf_counter()
	try:
		for episode_index in range(args.episodes):
			observation = env.reset()
			adapter.reset_episode()
			accumulator.begin_episode(episode_index)
			trajectory_digest = hashlib.sha256()
			action_digest = hashlib.sha256()
			background_digest = hashlib.sha256()
			input_digest = hashlib.sha256()
			policy64_digest = hashlib.sha256()
			gt_scoring_digest = hashlib.sha256()
			reward_sum = 0.0
			initial_state = np.asarray(physics.get_state())
			initial_state_sha = _array_sha256(initial_state)
			background_source = None
			background_start_index = None

			for step_index in range(args.steps + 1):
				state = np.asarray(physics.get_state())
				_digest_update_array(trajectory_digest, b"state", state)
				policy64 = _latest_policy_rgb(observation)
				_digest_update_array(policy64_digest, b"policy64", policy64)
				rgb_started = perf_counter()
				frame = (
					policy64
					if args.resolution == 64
					else background.cutie_same_state_rgb(
						height=args.resolution, width=args.resolution
					)
				)
				native_rgb_ms = (perf_counter() - rgb_started) * 1000.0
				if args.resolution > 64:
					resampling = getattr(Image, "Resampling", Image).BILINEAR
					policy64_bilinear = np.asarray(
						Image.fromarray(policy64, mode="RGB").resize(
							(args.resolution, args.resolution), resample=resampling
						),
						dtype=np.uint8,
					)
					difference = np.abs(
						frame.astype(np.int16) - policy64_bilinear.astype(np.int16)
					).astype(np.uint8)
					distinct = bool(difference.any())
					extra_sensor_distinct_frames += int(distinct)
					extra_sensor_abs_diff_sum += int(difference.sum(dtype=np.int64))
					extra_sensor_scalar_count += int(difference.size)
					extra_sensor_max_abs_diff = max(
						extra_sensor_max_abs_diff, int(difference.max())
					)
					_digest_update_array(extra_sensor_trace, b"absdiff", difference)
				if step_index == 0:
					background_source = Path(background.active_source).name
					background_start_index = int(background.frame_index)
				background_digest.update(Path(background.active_source).name.encode("utf-8"))
				background_digest.update(
					np.asarray([background.frame_index], dtype=np.int64).tobytes()
				)
				_digest_update_array(input_digest, b"rgb", frame)

				# Critical causal ordering: Cutie completes before GT is rendered.
				result = adapter.track(frame)
				gt_started = perf_counter()
				gt_native = _role_gt_masks(
					physics, args.gt_render_resolution, selections
				)
				gt_masks = _scoring_masks(gt_native)
				_digest_update_array(gt_scoring_digest, b"gt64", gt_masks)
				predicted = result.masks.detach().cpu().numpy().astype(np.bool_, copy=False)
				if tuple(result.mask_output_size) != (64, 64):
					raise RuntimeError(
						f"Cutie mask output {result.mask_output_size} != fixed (64,64)."
					)
				features = result.object_features
				if features is None or tuple(features.shape) != (len(roles), 2048):
					raise RuntimeError(
						"Cutie did not return the expected [K,2048] object features."
					)
				feature_finite = np.isfinite(
					features.detach().cpu().numpy()
				).all(axis=1)
				lost = result.lost.detach().cpu().numpy().astype(np.bool_, copy=False)
				gt_scoring_ms = (perf_counter() - gt_started) * 1000.0
				accumulator.record(
					step_index=step_index,
					predicted=predicted,
					gt=gt_masks,
					lost=lost,
					feature_finite=feature_finite,
					tracker_ms=result.runtime_ms,
					native_rgb_ms=native_rgb_ms,
					gt_scoring_ms=gt_scoring_ms,
				)

				if step_index == args.steps:
					break
				action = action_rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				_digest_update_array(action_digest, b"action", action)
				observation, reward, done, _ = env.step(action)
				reward_sum += float(reward)
				if done and step_index + 1 != args.steps:
					raise RuntimeError(
						f"Environment terminated early at action {step_index + 1}."
					)

			final_state = np.asarray(physics.get_state())
			record = {
				"episode_index": episode_index,
				"frames": args.steps + 1,
				"actions": args.steps,
				"initial_state_sha256": initial_state_sha,
				"final_state_sha256": _array_sha256(final_state),
				"physics_trajectory_sha256": trajectory_digest.hexdigest(),
				"action_sequence_sha256": action_digest.hexdigest(),
				"background_sequence_sha256": background_digest.hexdigest(),
				"policy64_input_sequence_sha256": policy64_digest.hexdigest(),
				"native_input_sequence_sha256": input_digest.hexdigest(),
				"gt_scoring64_sequence_sha256": gt_scoring_digest.hexdigest(),
				"background_source": background_source,
				"background_start_frame_index": background_start_index,
				"background_end_frame_index": int(background.frame_index),
				"random_policy_reward": reward_sum,
			}
			episode_records.append(record)
			print(
				"CUTIE_NATIVE_RESOLUTION_EPISODE",
				json.dumps({
					"task": args.task,
					"resolution": args.resolution,
					**record,
				}, allow_nan=False),
				flush=True,
			)
	finally:
		close = getattr(env, "close", None)
		if callable(close):
			close()

	summary = accumulator.summary()
	gates = _quality_gates(summary, args)
	runtime = adapter.runtime_summary()
	expected_frames = args.episodes * (args.steps + 1)
	if int(runtime.get("frames", -1)) != expected_frames:
		raise RuntimeError(
			f"Adapter frame count {runtime.get('frames')} != expected {expected_frames}."
		)
	if int(runtime.get("episode_hard_resets", -1)) != args.episodes:
		raise RuntimeError("Adapter did not perform one hard reset per episode.")
	arm = f"runtime{args.resolution}_support{args.support_resolution}"
	if paired_support:
		treatment = {
			(64, 64): "matched_native64_runtime_and_support_control",
			(128, 64): "runtime_cutie_input_resolution_only",
			(128, 128): "runtime_and_support_native_resolution",
		}.get((args.resolution, args.support_resolution))
		if treatment is None:
			raise RuntimeError(f"Unsupported paired native arm {arm}.")
	else:
		treatment = "runtime_cutie_input_resolution_only"
	return {
		"format": PAIRED_FORMAT if paired_support else FORMAT,
		"status": "tracker_evaluation_complete",
		"task": args.task,
		"roles": list(roles),
		"primary_small_role": PRIMARY_ROLE[args.task],
		"resolution": args.resolution,
		"support_resolution": args.support_resolution,
		"arm": arm,
		"protocol": {
			"tracker_only": True,
			"policy_constructed": False,
			"controller_training_steps": 0,
			"native_simulator_rgb": True,
			"extra_sensor_information": args.resolution > 64,
			"raw_sensor_information_parity": args.resolution == 64,
			"fair_representation_comparison": False,
			"agent_observation_unchanged": True,
			"treatment": treatment,
			"allowed_claim": "perception_diagnostic",
			"disallowed_claim": "representation_advantage_vs_rgb",
			"policy_rgb_size": [64, 64],
			"mask_output_size": [64, 64],
			"gt_render_resolution": [
				args.gt_render_resolution, args.gt_render_resolution
			],
			"gt_scoring_resolution": [64, 64],
			"dynamic_background_split": SPLIT,
			"gt_use": "post_track_offline_scoring_only",
			"gt_query_order": "after_cutie_track_at_same_physics_state",
			"gt_drives_tracker": False,
			"action_policy": "fixed_seed_uniform_action_sequence",
			"cross_resolution_support": (
				"same_paired_support_states_native_assets_selected_by_arm"
				if paired_support else
				"same_frozen_64x64_bytes_all_resolutions"
			),
			"support_mapping": (
				f"adapter_native{args.support_resolution}_direct_to_fixed_tracker448_once"
				if paired_support else
				"adapter_source64_direct_to_fixed_tracker448_once"
			),
			"support_resolution": [
				args.support_resolution, args.support_resolution
			],
			"high_resolution_support_sensor": args.support_resolution == 128,
			"highres_background_semantics": (
				"policy64_selects_and_advances_background;current_frame_resized_"
				"without_clock_advance_for_highres_perception"
			),
			"cutie_internal_tracker_size": [args.tracker_size, args.tracker_size],
		},
		"extra_sensor_evidence": (
			{
				"applicable": True,
				"reference": "policy64_pil_bilinear_to_runtime_resolution",
				"frames": expected_frames,
				"distinct_frames": extra_sensor_distinct_frames,
				"distinct_frame_fraction": extra_sensor_distinct_frames / expected_frames,
				"mean_absolute_uint8_difference": (
					extra_sensor_abs_diff_sum / extra_sensor_scalar_count
				),
				"max_absolute_uint8_difference": extra_sensor_max_abs_diff,
				"absolute_difference_trace_sha256": extra_sensor_trace.hexdigest(),
			}
			if args.resolution > 64 else
			{
				"applicable": False,
				"reference": None,
				"frames": expected_frames,
				"distinct_frames": None,
				"distinct_frame_fraction": None,
				"mean_absolute_uint8_difference": None,
				"max_absolute_uint8_difference": None,
				"absolute_difference_trace_sha256": None,
			}
		),
		"seeds": {
			"environment": args.env_seed,
			"background": args.background_seed,
			"action": args.action_seed,
			"cutie": args.cutie_seed,
			"support": args.expected_support_seed,
		},
		"evaluation": {
			"episodes": args.episodes,
			"actions_per_episode": args.steps,
			"frames_per_episode": args.steps + 1,
			"camera_id": CAMERA_ID,
			"gt_render_resolution": args.gt_render_resolution,
			"background_total_frames": args.background_total_frames,
			"background_source_cache_size": args.background_cache_size,
		},
		"provenance": {
			"evaluator": str(Path(__file__).resolve()),
			"evaluator_sha256": _file_sha256(Path(__file__).resolve()),
			"cutie_checkpoint": str(args.cutie_checkpoint.expanduser().resolve()),
			"cutie_checkpoint_sha256": _file_sha256(
				args.cutie_checkpoint.expanduser().resolve()
			),
			"oc_storm_repo": str(args.oc_storm_repo.expanduser().resolve()),
			"video_root": str(args.video_root.expanduser().resolve()),
			"manifest_dir": str(
				args.manifest_dir.expanduser().resolve()
				if args.manifest_dir is not None
				else Path(__file__).resolve().parents[1] / "envs" / "background_manifests"
			),
			"validation_manifest_sha256": background.manifest_sha256,
			"combined_manifest_sha256": background.combined_manifest_sha256,
			"support": support_provenance,
			"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
			"logical_cuda_device": 0,
			"device_name": torch.cuda.get_device_name(0),
		},
		"episodes": episode_records,
		"metrics": summary,
		"absolute_quality_gates": gates,
		"runtime_health": {
			"execution_mode": "in_process_official_cutie_adapter",
			"adapter_errors": 0,
			"timeouts": 0,
			"worker_restarts": 0,
			"successful_frames": expected_frames,
			"adapter": runtime,
			"elapsed_seconds": perf_counter() - started,
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
	parser.add_argument("--task", choices=TASKS, default="acrobot-swingup")
	parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
	parser.add_argument("--oc-storm-repo", type=Path, required=True)
	parser.add_argument("--cutie-checkpoint", type=Path, required=True)
	parser.add_argument("--support-annotations", type=Path, required=True)
	parser.add_argument(
		"--support-resolution", type=int, choices=(64, 128), default=64,
		help="Select native64 or native128 prompts from a paired v2 support pack.",
	)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path)
	parser.add_argument("--model-size", choices=("small", "base"), default="small")
	parser.add_argument("--tracker-size", type=int, default=448)
	parser.add_argument("--disable-amp", action="store_true")
	parser.add_argument("--episodes", type=int, default=20)
	parser.add_argument("--steps", type=int, default=500)
	parser.add_argument("--env-seed", type=int, default=424243)
	parser.add_argument("--background-seed", type=int, default=1618034)
	parser.add_argument("--action-seed", type=int, default=8675400)
	parser.add_argument("--cutie-seed", type=int, default=2718281)
	parser.add_argument("--expected-support-seed", type=int, default=314159)
	parser.add_argument("--background-total-frames", type=int, default=1000)
	parser.add_argument("--background-cache-size", type=int, default=8)
	parser.add_argument(
		"--gt-render-resolution", type=int, choices=(128,), default=128,
		help="Frozen across all arms; GT is then nearest-downsampled to canonical64.",
	)
	# These are engineering plausibility gates, not the cross-resolution science
	# decision. The aggregator applies the stricter improvement/non-regression
	# criteria after exact pairing is proven.
	parser.add_argument("--min-valid-rate", type=float, default=0.50)
	parser.add_argument("--max-empty-rate", type=float, default=0.50)
	# A candidate that loses a role for more than half of a 500-action episode
	# is not useful enough to justify a controller-training pilot, even when it
	# improves relatively over the native64 baseline.
	parser.add_argument("--max-invalid-burst", type=int, default=250)
	parser.add_argument("--min-mean-iou", type=float, default=0.0)
	parser.add_argument("--min-identity-accuracy", type=float, default=0.0)
	parser.add_argument("--output", type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None) -> int:
	args = parse_args(argv)
	payload = evaluate(args)
	_atomic_write(args.output, payload)
	print("CUTIE_NATIVE_RESOLUTION_TRACKER_OK", json.dumps({
		"task": args.task,
		"resolution": args.resolution,
		"support_resolution": args.support_resolution,
		"arm": payload["arm"],
		"quality_pass": payload["absolute_quality_gates"]["quality_pass"],
		"output": str(args.output.expanduser().resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
