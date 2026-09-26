"""Freeze one task-general RGB/GT dataset for VOS backend selection.

This collector is the only component allowed to query MuJoCo segmentation.
It writes RGB-only episode archives and scoring-only GT archives separately,
then publishes a sanitized ``backend_inputs.json`` that contains no GT paths.
Cutie/SAM workers must receive the sanitized manifest, never the full dataset
manifest.  No policy is constructed and no controller is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


os.environ.setdefault("MUJOCO_GL", "egl")
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.common.unified_vos import (  # noqa: E402
	BACKEND_INPUT_FORMAT,
	CONDITIONS,
	DATASET_FORMAT,
	STRICT_ACTIONS_PER_EPISODE,
	STRICT_EPISODES,
	STRICT_RESOLUTION,
	SUPPORT_FORMAT,
	SUPPORT_RECORDS,
	TASK_ROLES,
	compute_dataset_id,
	file_sha256,
	write_json,
)


SPLIT = "validation"
CAMERA_ID = 0


class Config(SimpleNamespace):
	def get(self, name, default=None):
		return getattr(self, name, default)


def _array_trace(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())
	return digest.hexdigest()


def _digest_array(digest, label: bytes, value: np.ndarray) -> None:
	array = np.ascontiguousarray(value)
	digest.update(label)
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	digest.update(array.tobytes())


def _step_environment(env, action: np.ndarray):
	"""Step the collector's raw DMControl wrapper with a copied NumPy action.

	Unlike the training path, this collector does not install a tensor-action
	wrapper.  ``DMControlWrapper.step`` therefore receives NumPy directly and
	uses ``astype`` on it.  Validate the transport boundary here so a future
	wrapper change fails before silently changing the frozen action trajectory.
	"""
	if not isinstance(action, np.ndarray):
		raise TypeError("Frozen dataset actions must be NumPy arrays.")
	action_space = getattr(env, "action_space", None)
	if action_space is None:
		raise TypeError("Frozen dataset environment has no action_space.")
	expected_shape = tuple(action_space.shape)
	expected_dtype = np.dtype(action_space.dtype)
	if action.shape != expected_shape:
		raise ValueError(
			f"Frozen dataset action shape changed: {action.shape} != {expected_shape}."
		)
	if action.dtype != expected_dtype:
		raise ValueError(
			f"Frozen dataset action dtype changed: {action.dtype} != {expected_dtype}."
		)
	return env.step(np.array(action, dtype=expected_dtype, order="C", copy=True))


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
	raise RuntimeError(f"{wrapper_type.__name__} is absent from the environment chain.")


def _background_state(background) -> tuple[str, int, int]:
	"""Return the compositor identity/clock without advancing it."""
	source = getattr(background, "active_source", None)
	frame_index = getattr(background, "frame_index", None)
	compositor = getattr(background, "compositor", None)
	clock = getattr(compositor, "_frame", None)
	if source is None or frame_index is None or type(clock) is not int:
		raise RuntimeError("Background compositor has no active frame/clock state.")
	return str(Path(source).resolve()), int(frame_index), int(clock)


def _render_rgb(physics, resolution: int) -> np.ndarray:
	frame = np.asarray(physics.render(
		height=resolution, width=resolution, camera_id=CAMERA_ID
	))
	if frame.shape != (resolution, resolution, 3) or frame.dtype != np.uint8:
		raise RuntimeError(f"Unexpected native RGB schema {frame.shape} {frame.dtype}.")
	return np.array(frame, dtype=np.uint8, order="C", copy=True)


def _render_indexed_gt(physics, resolution: int, selections) -> np.ndarray:
	from dm_control.mujoco.wrapper.mjbindings import enums

	rendered = np.asarray(physics.render(
		height=resolution,
		width=resolution,
		camera_id=CAMERA_ID,
		segmentation=True,
	))
	if rendered.shape != (resolution, resolution, 2):
		raise RuntimeError(f"Unexpected MuJoCo segmentation schema {rendered.shape}.")
	types = {
		"geom": int(enums.mjtObj.mjOBJ_GEOM),
		"site": int(enums.mjtObj.mjOBJ_SITE),
	}
	indexed = np.zeros((resolution, resolution), dtype=np.uint8)
	for role_id, selected in enumerate(selections, start=1):
		mask = np.zeros(indexed.shape, dtype=np.bool_)
		for object_type, object_id in selected:
			mask |= (
				(rendered[..., 0] == int(object_id))
				& (rendered[..., 1] == types[object_type])
			)
		if np.any(indexed[mask] != 0):
			raise RuntimeError("Configured GT role masks overlap.")
		indexed[mask] = role_id
	return indexed


def _parse_support_arguments(values: list[str]) -> dict[str, Path]:
	result: dict[str, Path] = {}
	for value in values:
		if "=" not in value:
			raise ValueError("--support entries must be TASK=/absolute/annotations.json.")
		task, raw_path = value.split("=", 1)
		if task not in TASK_ROLES or task in result:
			raise ValueError(f"Invalid or duplicate support task {task!r}.")
		path = Path(raw_path).expanduser().resolve()
		if not path.is_file():
			raise FileNotFoundError(path)
		result[task] = path
	if set(result) != set(TASK_ROLES):
		raise ValueError(f"Exactly one support pack is required for {list(TASK_ROLES)}.")
	return result


def _load_support(path: Path, *, task: str, resolution: int, seed: int):
	from tdmpc2.common.cutie_paired_support import FORMAT as PAIRED_FORMAT
	from tdmpc2.perception.cutie_oc_adapter import (
		load_indexed_support_prompts,
		load_paired_native_support_prompts,
	)

	payload = json.loads(path.read_text(encoding="utf-8"))
	if payload.get("format") == PAIRED_FORMAT:
		prompts = load_paired_native_support_prompts(
			path,
			role_names=TASK_ROLES[task],
			expected_task=task,
			support_resolution=resolution,
			expected_seed=seed,
			allow_simulator_support=True,
		)
	else:
		if resolution != 64:
			raise ValueError(
				f"{task}: scientific runtime128 comparison requires a paired native128 "
				"support pack, not a resized legacy64 pack."
			)
		prompts = load_indexed_support_prompts(
			path,
			role_names=TASK_ROLES[task],
			expected_task=task,
			expected_records=SUPPORT_RECORDS,
			allow_simulator_support=True,
		)
	if len(prompts.frames) != SUPPORT_RECORDS or len(prompts.masks) != SUPPORT_RECORDS:
		raise RuntimeError(f"{task}: support prompt count changed.")
	frames = np.stack(prompts.frames).astype(np.uint8, copy=False)
	masks = np.stack(prompts.masks).astype(np.uint8, copy=False)
	if frames.shape != (SUPPORT_RECORDS, resolution, resolution, 3):
		raise RuntimeError(f"{task}: support RGB shape {frames.shape} is invalid.")
	if masks.shape != (SUPPORT_RECORDS, resolution, resolution):
		raise RuntimeError(f"{task}: support mask shape {masks.shape} is invalid.")
	for index, mask in enumerate(masks):
		values = set(np.unique(mask).tolist())
		if not values.issubset({0, 1, 2}) or not {1, 2}.issubset(values):
			raise RuntimeError(f"{task}: support mask {index} has IDs {sorted(values)}.")
	return np.ascontiguousarray(frames), np.ascontiguousarray(masks), prompts.metadata


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
	return file_sha256(path)


def _make_env(args, task: str, condition: str, dmcontrol_env):
	hard = condition == "hard"
	config = Config(
		task=task,
		obs="rgb",
		seed=args.env_seed,
		multitask=False,
		video_background_enabled=hard,
		video_background_root=(str(args.video_root) if hard else None),
		video_background_manifest_dir=(str(args.manifest_dir) if hard else None),
		video_background_split=SPLIT,
		video_background_strength=1.0,
		video_background_total_frames=args.background_total_frames,
		video_background_source_cache_size=args.background_cache_size,
		video_background_seed=args.background_seed,
		flat_anchor=False,
	)
	return dmcontrol_env.make_env(config)


def _collect_condition(
	args,
	*,
	root: Path,
	task: str,
	condition: str,
	dmcontrol_env,
	task_specs,
	catalog_fn,
	selected_objects_fn,
) -> list[dict[str, Any]]:
	env = _make_env(args, task, condition, dmcontrol_env)
	try:
		physics = _find_physics(env)
		background = None
		if condition == "hard":
			background = _find_wrapper(
				env, dmcontrol_env.ColorMultiVideoBackgroundWrapper
			)
			if background.active_split != SPLIT:
				raise RuntimeError("Hard-background split changed.")
		spec = task_specs[task]
		if tuple(spec.roles) != TASK_ROLES[task]:
			raise RuntimeError(f"Task role order changed for {task}.")
		catalog = catalog_fn(physics)
		selections = tuple(
			selected_objects_fn(catalog, selector) for selector in spec.selectors
		)
		if set(selections[0]).intersection(selections[1]):
			raise RuntimeError(f"Task selectors overlap for {task}.")

		action_rng = np.random.default_rng(args.action_seed)
		records = []
		for episode_index in range(args.episodes):
			observation = env.reset()
			del observation
			rgb_frames = np.empty(
				(args.steps + 1, args.resolution, args.resolution, 3), dtype=np.uint8
			)
			gt_frames = np.empty(
				(args.steps + 1, args.resolution, args.resolution), dtype=np.uint8
			)
			actions = np.empty((args.steps,) + env.action_space.shape, dtype=env.action_space.dtype)
			rgb_digest = hashlib.sha256()
			gt_digest = hashlib.sha256()
			action_digest = hashlib.sha256()
			physics_digest = hashlib.sha256()
			background_digest = hashlib.sha256()
			reward_sum = 0.0
			for frame_index in range(args.steps + 1):
				state = np.asarray(physics.get_state())
				_digest_array(physics_digest, b"state", state)
				if condition == "hard":
					frame = background.cutie_same_state_rgb(
						height=args.resolution, width=args.resolution
					)
					background_digest.update(Path(background.active_source).name.encode("utf-8"))
					background_digest.update(
						np.asarray([background.frame_index], dtype=np.int64).tobytes()
					)
				else:
					frame = _render_rgb(physics, args.resolution)
				gt = _render_indexed_gt(physics, args.resolution, selections)
				rgb_frames[frame_index] = frame
				gt_frames[frame_index] = gt
				_digest_array(rgb_digest, b"rgb", frame)
				_digest_array(gt_digest, b"gt", gt)
				if frame_index == args.steps:
					break
				action = action_rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				actions[frame_index] = action
				_digest_array(action_digest, b"action", action)
				_, reward, done, _ = _step_environment(env, action)
				reward_sum += float(reward)
				if done and frame_index + 1 != args.steps:
					raise RuntimeError(
						f"{task}/{condition}/{episode_index} terminated early."
					)

			relative_rgb = Path("inputs") / task / condition / f"episode_{episode_index:03d}.npz"
			relative_gt = Path("scoring") / task / condition / f"episode_{episode_index:03d}.npz"
			rgb_path = root / relative_rgb
			gt_path = root / relative_gt
			rgb_file_sha = _save_npz(rgb_path, rgb=rgb_frames)
			gt_file_sha = _save_npz(gt_path, gt_indexed=gt_frames, actions=actions)
			record = {
				"episode_index": episode_index,
				"frames": args.steps + 1,
				"actions": args.steps,
				"arrays": relative_gt.as_posix(),
				"arrays_sha256": gt_file_sha,
				"rgb_arrays": relative_rgb.as_posix(),
				"rgb_arrays_sha256": rgb_file_sha,
				"rgb_trace_sha256": rgb_digest.hexdigest(),
				"gt_trace_sha256": gt_digest.hexdigest(),
				"action_trace_sha256": action_digest.hexdigest(),
				"physics_trace_sha256": physics_digest.hexdigest(),
				"background_trace_sha256": (
					background_digest.hexdigest() if condition == "hard" else None
				),
				"random_policy_reward": reward_sum,
				"role_gt_visible_frames": {
					role: int((gt_frames == role_id).reshape(args.steps + 1, -1).any(axis=1).sum())
					for role_id, role in enumerate(TASK_ROLES[task], start=1)
				},
			}
			records.append(record)
			print(
				"UNIFIED_VOS_DATASET_EPISODE",
				json.dumps({
					"task": task,
					"condition": condition,
					"episode_index": episode_index,
					"rgb_trace_sha256": record["rgb_trace_sha256"],
					"gt_trace_sha256": record["gt_trace_sha256"],
				}, allow_nan=False),
				flush=True,
			)
		return records
	finally:
		close = getattr(env, "close", None)
		if callable(close):
			close()


def _collect_task_paired(
	args,
	*,
	root: Path,
	task: str,
	dmcontrol_env,
	task_specs,
	catalog_fn,
	selected_objects_fn,
) -> dict[str, list[dict[str, Any]]]:
	"""Collect clean and hard pixels from each exact same simulator state."""
	env = _make_env(args, task, "hard", dmcontrol_env)
	try:
		physics = _find_physics(env)
		background = _find_wrapper(
			env, dmcontrol_env.ColorMultiVideoBackgroundWrapper
		)
		if background.active_split != SPLIT:
			raise RuntimeError("Hard-background split changed.")
		spec = task_specs[task]
		if tuple(spec.roles) != TASK_ROLES[task]:
			raise RuntimeError(f"Task role order changed for {task}.")
		catalog = catalog_fn(physics)
		selections = tuple(
			selected_objects_fn(catalog, selector) for selector in spec.selectors
		)
		if set(selections[0]).intersection(selections[1]):
			raise RuntimeError(f"Task selectors overlap for {task}.")

		action_rng = np.random.default_rng(args.action_seed)
		all_records: dict[str, list[dict[str, Any]]] = {
			"clean": [], "hard": [],
		}
		for episode_index in range(args.episodes):
			env.reset()
			physics_states: list[np.ndarray] = []
			frames = {
				condition: np.empty(
					(args.steps + 1, args.resolution, args.resolution, 3),
					dtype=np.uint8,
				)
				for condition in CONDITIONS
			}
			gt_frames = np.empty(
				(args.steps + 1, args.resolution, args.resolution), dtype=np.uint8
			)
			actions = np.empty(
				(args.steps,) + env.action_space.shape, dtype=env.action_space.dtype
			)
			rgb_digests = {condition: hashlib.sha256() for condition in CONDITIONS}
			gt_digest = hashlib.sha256()
			action_digest = hashlib.sha256()
			physics_digest = hashlib.sha256()
			background_digest = hashlib.sha256()
			reward_sum = 0.0
			for frame_index in range(args.steps + 1):
				state = np.asarray(physics.get_state()).copy()
				physics_states.append(np.ascontiguousarray(state))
				physics_time = float(physics.time())
				background_before = _background_state(background)
				_digest_array(physics_digest, b"state", state)
				clean = _render_rgb(physics, args.resolution)
				hard = background.cutie_same_state_rgb(
					height=args.resolution, width=args.resolution
				)
				if np.array_equal(clean, hard):
					raise RuntimeError(
						f"Hard compositor did not alter {task} episode {episode_index} "
						f"frame {frame_index}."
					)
				gt = _render_indexed_gt(physics, args.resolution, selections)
				state_after = np.asarray(physics.get_state())
				if (
					not np.array_equal(state_after, state)
					or float(physics.time()) != physics_time
					or _background_state(background) != background_before
				):
					raise RuntimeError(
						"Same-state clean/hard/GT rendering changed physics or the "
						f"background clock for {task}/{episode_index}/{frame_index}."
					)
				frames["clean"][frame_index] = clean
				frames["hard"][frame_index] = hard
				gt_frames[frame_index] = gt
				_digest_array(rgb_digests["clean"], b"rgb", clean)
				_digest_array(rgb_digests["hard"], b"rgb", hard)
				_digest_array(gt_digest, b"gt", gt)
				background_digest.update(Path(background.active_source).name.encode("utf-8"))
				background_digest.update(
					np.asarray([background.frame_index], dtype=np.int64).tobytes()
				)
				if frame_index == args.steps:
					break
				action = action_rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				actions[frame_index] = action
				_digest_array(action_digest, b"action", action)
				_, reward, done, _ = _step_environment(env, action)
				reward_sum += float(reward)
				if done and frame_index + 1 != args.steps:
					raise RuntimeError(f"{task}/{episode_index} terminated early.")

			physics_state_array = np.stack(physics_states, axis=0)
			for condition in CONDITIONS:
				relative_rgb = (
					Path("inputs") / task / condition / f"episode_{episode_index:03d}.npz"
				)
				relative_gt = (
					Path("scoring") / task / condition / f"episode_{episode_index:03d}.npz"
				)
				rgb_file_sha = _save_npz(root / relative_rgb, rgb=frames[condition])
				gt_file_sha = _save_npz(
					root / relative_gt,
					gt_indexed=gt_frames,
					actions=actions,
					physics_states=physics_state_array,
				)
				record = {
					"episode_index": episode_index,
					"frames": args.steps + 1,
					"actions": args.steps,
					"arrays": relative_gt.as_posix(),
					"arrays_sha256": gt_file_sha,
					"rgb_arrays": relative_rgb.as_posix(),
					"rgb_arrays_sha256": rgb_file_sha,
					"rgb_trace_sha256": rgb_digests[condition].hexdigest(),
					"gt_trace_sha256": gt_digest.hexdigest(),
					"action_trace_sha256": action_digest.hexdigest(),
					"physics_trace_sha256": physics_digest.hexdigest(),
					"background_trace_sha256": (
						background_digest.hexdigest() if condition == "hard" else None
					),
					"random_policy_reward": reward_sum,
					"role_gt_visible_frames": {
						role: int(
							(gt_frames == role_id)
							.reshape(args.steps + 1, -1)
							.any(axis=1)
							.sum()
						)
						for role_id, role in enumerate(TASK_ROLES[task], start=1)
					},
				}
				all_records[condition].append(record)
				print(
					"UNIFIED_VOS_DATASET_EPISODE",
					json.dumps({
						"task": task,
						"condition": condition,
						"episode_index": episode_index,
						"rgb_trace_sha256": record["rgb_trace_sha256"],
						"gt_trace_sha256": record["gt_trace_sha256"],
					}, allow_nan=False),
					flush=True,
				)
		return all_records
	finally:
		close = getattr(env, "close", None)
		if callable(close):
			close()


def collect(args) -> dict[str, Any]:
	import envs.dmcontrol as dmcontrol_env
	from tdmpc2.tools.collect_cutie_multitask_support import (
		TASK_BY_NAME,
		_catalog,
		_selected_objects,
	)

	if args.output.exists():
		raise FileExistsError(args.output)
	if args.worker_output.exists():
		raise FileExistsError(args.worker_output)
	if args.resolution not in (64, 128):
		raise ValueError("Frozen benchmark resolution must be 64 or 128.")
	if args.episodes < 1 or args.steps < 1:
		raise ValueError("Episode/action counts must be positive.")
	if len({args.env_seed, args.background_seed, args.action_seed, args.support_seed}) != 4:
		raise ValueError("Environment/background/action/support seeds must differ.")
	if not args.video_root.is_dir() or args.video_root.name != "video_hard":
		raise FileNotFoundError("--video-root must be the frozen video_hard directory.")
	if not args.manifest_dir.is_dir():
		raise FileNotFoundError(args.manifest_dir)
	support_paths = _parse_support_arguments(args.support)
	args.output.mkdir(parents=True, exist_ok=False)
	root = args.output.resolve()
	worker_root = args.worker_output.resolve()
	if root == worker_root or root in worker_root.parents or worker_root in root.parents:
		raise ValueError("--output and --worker-output must be disjoint directories.")

	support_manifest = {}
	for task in TASK_ROLES:
		frames, masks, metadata = _load_support(
			support_paths[task],
			task=task,
			resolution=args.resolution,
			seed=args.support_seed,
		)
		relative = Path("support") / task / "support.npz"
		arrays_sha = _save_npz(root / relative, rgb=frames, indexed_masks=masks)
		trace = hashlib.sha256()
		_digest_array(trace, b"support_rgb", frames)
		_digest_array(trace, b"support_masks", masks)
		support_manifest[task] = {
			"format": SUPPORT_FORMAT,
			"roles": list(TASK_ROLES[task]),
			"records": SUPPORT_RECORDS,
			"resolution": args.resolution,
			"arrays": relative.as_posix(),
			"arrays_sha256": arrays_sha,
			"asset_trace_sha256": trace.hexdigest(),
			"source_relative_to_benchmark_root": support_paths[task]
			.relative_to(root.parent)
			.as_posix(),
			"source_format": metadata.get("format"),
			"source_sha256": file_sha256(support_paths[task]),
			"source_metadata": metadata,
		}

	episodes: dict[str, dict[str, list[dict[str, Any]]]] = {}
	for task in TASK_ROLES:
		episodes[task] = _collect_task_paired(
			args,
			root=root,
			task=task,
			dmcontrol_env=dmcontrol_env,
			task_specs=TASK_BY_NAME,
			catalog_fn=_catalog,
			selected_objects_fn=_selected_objects,
		)

	# Clean/hard must differ only in pixels/background, never in actions/physics.
	for task in TASK_ROLES:
		for clean, hard in zip(episodes[task]["clean"], episodes[task]["hard"]):
			for key in (
				"episode_index", "frames", "actions", "gt_trace_sha256",
				"action_trace_sha256", "physics_trace_sha256",
				"random_policy_reward", "role_gt_visible_frames",
			):
				if clean[key] != hard[key]:
					raise RuntimeError(f"Clean/hard trajectory pairing failed: {task}/{key}.")
			if clean["rgb_trace_sha256"] == hard["rgb_trace_sha256"]:
				raise RuntimeError(f"Hard background did not change RGB for {task}.")

	protocol = {
		"tracker_only": True,
		"controller_constructed": False,
		"controller_training_steps": 0,
		"episode_gt_disclosed_in_backend_inputs": False,
		"episode_gt_use": "offline_scoring_only",
		"prompt_source": "fixed_six_frame_support_pack_only",
		"episode_first_frame_gt_prompt": False,
		"mid_episode_reprompt": False,
		"causal_frame_order": True,
		"same_frames_all_backends": True,
		"allowed_claim": "video_segmentation_frontend_selection",
		"disallowed_claim": "controller_or_representation_advantage",
	}
	payload = {
		"format": DATASET_FORMAT,
		"status": "complete",
		"tasks": list(TASK_ROLES),
		"conditions": list(CONDITIONS),
		"roles": {task: list(roles) for task, roles in TASK_ROLES.items()},
		"resolution": args.resolution,
		"counts": {
			"episodes": args.episodes,
			"actions_per_episode": args.steps,
			"frames_per_episode": args.steps + 1,
		},
		"protocol": protocol,
		"seeds": {
			"environment": args.env_seed,
			"background": args.background_seed,
			"action": args.action_seed,
			"support": args.support_seed,
		},
		"render": {
			"camera_id": CAMERA_ID,
			"hard_background_split": SPLIT,
			"hard_background_root": str(args.video_root),
			"manifest_dir": str(args.manifest_dir),
			"background_total_frames": args.background_total_frames,
			"background_cache_size": args.background_cache_size,
			"native_resolution": True,
			"extra_sensor_information_vs_policy64": args.resolution > 64,
			"fair_representation_comparison": False,
		},
		"support": support_manifest,
		"episodes": episodes,
	}
	payload["dataset_id"] = compute_dataset_id(payload)

	backend_inputs = {
		"format": BACKEND_INPUT_FORMAT,
		"status": "complete",
		"dataset_id": payload["dataset_id"],
		"tasks": list(TASK_ROLES),
		"conditions": list(CONDITIONS),
		"roles": payload["roles"],
		"resolution": args.resolution,
		"counts": payload["counts"],
		"protocol": {
			"support_only_prompts": True,
			"episode_ground_truth_present": False,
			"episode_first_frame_gt_prompt": False,
			"mid_episode_reprompt": False,
			"causal_frame_order": True,
		},
		"support": {
			task: {
				"roles": entry["roles"],
				"records": entry["records"],
				"resolution": entry["resolution"],
				"arrays": entry["arrays"],
				"arrays_sha256": entry["arrays_sha256"],
				"asset_trace_sha256": entry["asset_trace_sha256"],
			}
			for task, entry in support_manifest.items()
		},
		"episodes": {
			task: {
				condition: [{
					"episode_index": record["episode_index"],
					"frames": record["frames"],
					"rgb_arrays": record["rgb_arrays"],
					"rgb_arrays_sha256": record["rgb_arrays_sha256"],
					"rgb_trace_sha256": record["rgb_trace_sha256"],
				} for record in records]
				for condition, records in condition_map.items()
			}
			for task, condition_map in episodes.items()
		},
	}
	backend_inputs_path = root / "backend_inputs.json"
	write_json(backend_inputs_path, backend_inputs)
	payload["backend_inputs"] = {
		"path": "backend_inputs.json",
		"sha256": file_sha256(backend_inputs_path),
		"gt_paths_disclosed": False,
	}
	write_json(root / "dataset_manifest.json", payload)

	# Expose a separate, exact allowlisted filesystem view to model workers.
	# It contains support and RGB only; GT masks/actions remain under the
	# disjoint dataset scoring tree.  The runner also removes read permissions
	# from that scoring tree while any external backend is alive.
	worker_root.mkdir(parents=True, exist_ok=False)
	worker_members = [Path("backend_inputs.json")]
	worker_members.extend(Path(entry["arrays"]) for entry in support_manifest.values())
	worker_members.extend(
		Path(record["rgb_arrays"])
		for condition_map in episodes.values()
		for records in condition_map.values()
		for record in records
	)
	for relative in worker_members:
		if relative.is_absolute() or ".." in relative.parts:
			raise RuntimeError(f"Unsafe worker-view member: {relative}")
		source = (root / relative).resolve()
		source.relative_to(root)
		destination = worker_root / relative
		destination.parent.mkdir(parents=True, exist_ok=True)
		try:
			os.link(source, destination)
		except OSError:
			shutil.copy2(source, destination)
		if file_sha256(destination) != file_sha256(source):
			raise RuntimeError(f"Worker-view copy changed bytes: {relative}")
	actual_members = sorted(
		path.relative_to(worker_root).as_posix()
		for path in worker_root.rglob("*")
		if path.is_file()
	)
	expected_members = sorted(relative.as_posix() for relative in worker_members)
	if actual_members != expected_members:
		raise RuntimeError("GT-free worker-view allowlist changed.")
	return payload


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument(
		"--worker-output",
		type=Path,
		required=True,
		help="Disjoint GT-free filesystem root exposed to model workers.",
	)
	parser.add_argument(
		"--support",
		action="append",
		default=[],
		help="Repeat exactly once per task: TASK=/absolute/annotations.json",
	)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path, required=True)
	parser.add_argument("--resolution", type=int, default=STRICT_RESOLUTION)
	parser.add_argument("--episodes", type=int, default=STRICT_EPISODES)
	parser.add_argument("--steps", type=int, default=STRICT_ACTIONS_PER_EPISODE)
	parser.add_argument("--env-seed", type=int, default=424243)
	parser.add_argument("--background-seed", type=int, default=1618034)
	parser.add_argument("--action-seed", type=int, default=8675400)
	parser.add_argument("--support-seed", type=int, default=314159)
	parser.add_argument("--background-total-frames", type=int, default=1000)
	parser.add_argument("--background-cache-size", type=int, default=8)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	args.output = args.output.expanduser().resolve()
	args.worker_output = args.worker_output.expanduser().resolve()
	args.video_root = args.video_root.expanduser().resolve()
	args.manifest_dir = args.manifest_dir.expanduser().resolve()
	payload = collect(args)
	print("UNIFIED_VOS_DATASET_COMPLETE", json.dumps({
		"dataset_id": payload["dataset_id"],
		"manifest": str(args.output / "dataset_manifest.json"),
	}, allow_nan=False), flush=True)


if __name__ == "__main__":
	main()
