"""Collect exact variable-role Cutie support packs for DMC tasks.

This is an offline, privileged support-data tool.  It saves composed RGB frames
from the immutable ``support`` video split and obtains labels from MuJoCo's
segmentation renderer at the same simulator state.  Simulator segmentation is
never returned to an environment or learner.  Each canonical task directory is
published atomically only after six images and indexed masks pass the strict
schema checks.

Example::

    python -m tdmpc2.tools.collect_cutie_multitask_support \
        --task reacher-visual-small \
        --output /path/to/support_root \
        --video-root /path/to/video_hard
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from PIL import Image


os.environ.setdefault("MUJOCO_GL", "egl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dm_control.mujoco.wrapper.mjbindings import enums  # noqa: E402
from common.support_camera_contract import (  # noqa: E402
	build_same_camera_contract,
	validate_same_camera_contract,
)
from envs.dmcontrol import DMControlWrapper, make_env  # noqa: E402
from envs.wrappers.video_background import (  # noqa: E402
	ColorMultiVideoBackgroundWrapper,
)


FORMAT = "cutie_indexed_mask_support_v1"
SUPPORT_SCHEMA = "generic_indexed_v1"
SPLIT = "support"
SUPPORT_EPISODES = 6
SUPPORT_VIDEOS = tuple(f"video{index}.mp4" for index in range(85, 90))
IMAGE_SIZE = 64


@dataclass(frozen=True)
class RoleSelector:
	"""Full-match regex selectors over rendered geom/site names and bodies."""

	geom: tuple[str, ...] = ()
	body: tuple[str, ...] = ()
	site: tuple[str, ...] = ()
	site_body: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
	task: str
	roles: tuple[str, ...]
	selectors: tuple[RoleSelector, ...]


TASK_SPECS = (
	TaskSpec(
		task="reacher-visual-small",
		roles=("whole_arm", "goal"),
		selectors=(
			RoleSelector(
				geom=(r"arm", r"hand", r"finger"),
				body=(r"arm", r"hand", r"finger"),
			),
			RoleSelector(geom=(r"target",)),
		),
	),
	TaskSpec(
		task="cup-catch",
		roles=("cup", "ball"),
		selectors=(
			RoleSelector(geom=(r"cup_part_[0-4]",), body=(r"cup",)),
			RoleSelector(geom=(r"ball",), body=(r"ball",)),
		),
	),
	TaskSpec(
		task="cartpole-swingup",
		roles=("cart", "pole"),
		selectors=(
			RoleSelector(geom=(r"cart",), body=(r"cart",)),
			RoleSelector(geom=(r"pole_1",), body=(r"pole_1",)),
		),
	),
	TaskSpec(
		task="finger-spin",
		roles=("finger", "spinner"),
		selectors=(
			RoleSelector(
				geom=(r"proximal", r"proximal_decoration", r"distal", r"fingertip"),
				body=(r"proximal", r"distal"),
			),
			RoleSelector(
				geom=(r"cap1", r"cap2", r"spinner_decoration"),
				body=(r"spinner",),
			),
		),
	),
	TaskSpec(
		task="acrobot-swingup",
		# Track the articulated mechanism as one entity. Its spatial occupancy
		# retains folded shape without requiring unstable link identities.
		roles=("whole_acrobot",),
		selectors=(
			RoleSelector(
				geom=(r"upper_arm", r"upper_arm_decoration", r"lower_arm"),
				body=(r"upper_arm", r"lower_arm"),
			),
		),
	),
	*(TaskSpec(
		task=task,
		roles=("whole_arm", "goal"),
		selectors=(
			RoleSelector(
				geom=(r"arm", r"hand", r"finger"),
				body=(r"arm", r"hand", r"finger"),
			),
			RoleSelector(geom=(r"target",)),
		),
	) for task in ("reacher-easy", "reacher-hard")),
	*(TaskSpec(
		task=task,
		roles=("cart", "pole"),
		selectors=(
			RoleSelector(geom=(r"cart",), body=(r"cart",)),
			RoleSelector(geom=(r"pole_1",), body=(r"pole_1",)),
		),
	) for task in (
		"cartpole-balance", "cartpole-balance-sparse", "cartpole-swingup-sparse",
	)),
	*(TaskSpec(
		task=task,
		roles=("finger", "spinner", "target"),
		selectors=(
			RoleSelector(
				geom=(r"proximal", r"proximal_decoration", r"distal", r"fingertip"),
				body=(r"proximal", r"distal"),
			),
			RoleSelector(
				geom=(r"cap1", r"cap2", r"spinner_decoration"),
				body=(r"spinner",),
			),
			# DMC Finger Turn moves this visible site to the sampled goal angle
			# during reset. MuJoCo segmentation labels sites independently from
			# geoms, so the support-only privileged collector can label it exactly.
			RoleSelector(site=(r"target",)),
		),
	) for task in ("finger-turn-easy", "finger-turn-hard")),
	TaskSpec(
		task="pendulum-swingup",
		roles=("base", "pendulum"),
		selectors=(
			RoleSelector(geom=(r"base",)),
			RoleSelector(geom=(r"pole", r"mass")),
		),
	),
	*(TaskSpec(
		task=task,
		roles=("torso", "leg", "foot"),
		selectors=(
			RoleSelector(geom=(r"torso", r"nose", r"pelvis")),
			RoleSelector(geom=(r"thigh", r"calf")),
			RoleSelector(geom=(r"foot",)),
		),
	) for task in ("hopper-stand", "hopper-hop")),
	*(TaskSpec(
		task=task,
		roles=("torso", "right_leg", "left_leg"),
		selectors=(
			RoleSelector(geom=(r"torso",)),
			RoleSelector(geom=(r"right_thigh", r"right_leg", r"right_foot")),
			RoleSelector(geom=(r"left_thigh", r"left_leg", r"left_foot")),
		),
	) for task in ("walker-stand", "walker-walk", "walker-run")),
	TaskSpec(
		task="cheetah-run",
		roles=("torso", "back_leg", "front_leg"),
		selectors=(
			RoleSelector(geom=(r"torso", r"head"), body=(r"torso",)),
			RoleSelector(geom=(r"bthigh", r"bshin", r"bfoot")),
			RoleSelector(geom=(r"fthigh", r"fshin", r"ffoot")),
		),
	),
	*(TaskSpec(
		task=task,
		roles=("torso", "front_legs", "back_legs"),
		selectors=(
			RoleSelector(geom=(r"torso", r"eye_r", r"eye_l")),
			RoleSelector(geom=(
				r"thigh_front_left", r"shin_front_left", r"foot_front_left", r"toe_front_left",
				r"thigh_front_right", r"shin_front_right", r"foot_front_right", r"toe_front_right",
			)),
			RoleSelector(geom=(
				r"thigh_back_left", r"shin_back_left", r"foot_back_left", r"toe_back_left",
				r"thigh_back_right", r"shin_back_right", r"foot_back_right", r"toe_back_right",
			)),
		),
	) for task in ("quadruped-run", "quadruped-walk")),
)
TASK_BY_NAME = {spec.task: spec for spec in TASK_SPECS}


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


class RoleVisibilityError(ValueError):
	"""A valid rendered frame in which one configured role is not visible."""

	def __init__(self, role_index: int, role_name: str):
		self.role_index = int(role_index)
		self.role_name = str(role_name)
		super().__init__(
			f"Role {self.role_index} ({self.role_name}) has no visible "
			"segmentation pixels in this frame."
		)


def _decoded_sha256(array: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(payload) -> bytes:
	return (
		json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
		+ "\n"
	).encode("utf-8")


def _write_json(path: Path, payload) -> None:
	path.write_bytes(_canonical_json_bytes(payload))


def _fullmatch_any(patterns: Iterable[str], value: str | None) -> bool:
	return value is not None and any(re.fullmatch(pattern, value) for pattern in patterns)


def _model_name(model, object_id: int, object_type: str) -> str | None:
	name = model.id2name(int(object_id), object_type)
	return None if name is None else str(name)


def _body_name(model, body_id: int) -> str | None:
	return _model_name(model, int(body_id), "body")


def _catalog(physics) -> dict:
	"""Return deterministic renderable-object provenance for one model."""
	model = physics.model
	geoms = []
	for object_id in range(int(model.ngeom)):
		body_id = int(model.geom_bodyid[object_id])
		geoms.append({
			"id": object_id,
			"name": _model_name(model, object_id, "geom"),
			"body_id": body_id,
			"body_name": _body_name(model, body_id),
		})
	sites = []
	for object_id in range(int(model.nsite)):
		body_id = int(model.site_bodyid[object_id])
		sites.append({
			"id": object_id,
			"name": _model_name(model, object_id, "site"),
			"body_id": body_id,
			"body_name": _body_name(model, body_id),
		})
	return {"geoms": geoms, "sites": sites}


def _selector_payload(selector: RoleSelector) -> dict:
	return {
		"geom_fullmatch_regex": list(selector.geom),
		"geom_body_fullmatch_regex": list(selector.body),
		"site_fullmatch_regex": list(selector.site),
		"site_body_fullmatch_regex": list(selector.site_body),
	}


def _selected_objects(catalog: dict, selector: RoleSelector) -> tuple[tuple[str, int], ...]:
	selected = []
	for item in catalog["geoms"]:
		if _fullmatch_any(selector.geom, item["name"]) or _fullmatch_any(
			selector.body, item["body_name"]
		):
			selected.append(("geom", int(item["id"])))
	for item in catalog["sites"]:
		if _fullmatch_any(selector.site, item["name"]) or _fullmatch_any(
			selector.site_body, item["body_name"]
		):
			selected.append(("site", int(item["id"])))
	if not selected:
		raise ValueError(
			"Role selector matched no model object: "
			f"{_selector_payload(selector)!r}."
		)
	return tuple(selected)


def _named_selection(catalog: dict, selected: Iterable[tuple[str, int]]) -> list[dict]:
	items = []
	for object_type, object_id in selected:
		entry = catalog["geoms" if object_type == "geom" else "sites"][object_id]
		items.append({"object_type": object_type, **entry})
	return items


def _segmentation_constants() -> dict[str, int]:
	return {
		"geom": int(enums.mjtObj.mjOBJ_GEOM),
		"site": int(enums.mjtObj.mjOBJ_SITE),
	}


def _indexed_mask(
	segmentation: np.ndarray,
	selections,
	*,
	role_names: Iterable[str] | None = None,
) -> np.ndarray:
	segmentation = np.asarray(segmentation)
	if segmentation.shape != (IMAGE_SIZE, IMAGE_SIZE, 2):
		raise ValueError(
			"MuJoCo segmentation must be [64,64,2] object-id/object-type, got "
			f"{segmentation.shape}."
		)
	constants = _segmentation_constants()
	role_names = tuple(role_names or (
		f"role_{index}" for index in range(1, len(selections) + 1)
	))
	if len(role_names) != len(selections):
		raise ValueError(
			f"role_names/selections length mismatch: {len(role_names)} vs "
			f"{len(selections)}."
		)
	output = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
	for role_id, selected in enumerate(selections, start=1):
		role_mask = np.zeros_like(output, dtype=np.bool_)
		for object_type, object_id in selected:
			role_mask |= (
				(segmentation[..., 0] == int(object_id))
				& (segmentation[..., 1] == constants[object_type])
			)
		if not role_mask.any():
			raise RoleVisibilityError(role_id, role_names[role_id - 1])
		if np.any(output[role_mask]):
			raise ValueError("Task-role selections overlap in rendered pixels.")
		output[role_mask] = role_id
	allowed = set(range(len(selections) + 1))
	if not set(np.unique(output)).issubset(allowed):
		raise AssertionError(f"Indexed support mask escaped IDs {sorted(allowed)}.")
	return output


def _find_wrapper(env, cls):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		if isinstance(current, cls):
			return current
		current = getattr(current, "env", None)
	raise RuntimeError(f"{cls.__name__} is missing from the environment chain.")


def _find_physics(env):
	current, seen = env, set()
	while current is not None and id(current) not in seen:
		seen.add(id(current))
		physics = getattr(current, "physics", None)
		if physics is not None and callable(getattr(physics, "render", None)):
			return physics
		current = getattr(current, "env", None)
	raise RuntimeError("Could not find dm_control physics in the environment chain.")


def _latest_rgb(observation) -> np.ndarray:
	try:
		array = observation[-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy()
	except AttributeError:
		array = np.moveaxis(np.asarray(observation)[-3:], 0, -1)
	if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or array.dtype != np.uint8:
		raise ValueError(
			f"Expected newest uint8 RGB [64,64,3], got {array.shape} {array.dtype}."
		)
	return np.ascontiguousarray(array)


def _collector_config(args, task: str, seed: int):
	return Config(
		task=task,
		obs="rgb",
		seed=seed,
		multitask=False,
		video_background_enabled=True,
		video_background_root=str(args.video_root.expanduser().resolve()),
		video_background_manifest_dir=(
			None
			if args.manifest_dir is None
			else str(args.manifest_dir.expanduser().resolve())
		),
		video_background_split=SPLIT,
		video_background_strength=1.0,
		video_background_total_frames=1000,
		video_background_source_cache_size=8,
		flat_anchor=False,
	)


def _validate_task_pack(
	task_dir: Path,
	spec: TaskSpec,
	*,
	expected_camera_id: int,
) -> None:
	annotations = task_dir / "annotations.json"
	data = json.loads(annotations.read_text(encoding="utf-8"))
	if set(data) != {"format", "roles", "collection", "records"}:
		raise AssertionError(f"Unexpected top-level support fields for {spec.task}.")
	if data["format"] != FORMAT or tuple(data["roles"]) != spec.roles:
		raise AssertionError((spec.task, data["format"], data["roles"]))
	collection = data["collection"]
	if (
		collection.get("support_schema") != SUPPORT_SCHEMA
		or collection.get("task") != spec.task
		or collection.get("split") != SPLIT
		or collection.get("label_policy") != "simulator_segmentation_support_only"
		or collection.get("diagnostic_support") is not True
	):
		raise AssertionError((spec.task, "collection provenance", collection))
	try:
		camera_id = validate_same_camera_contract(
			collection, expected_camera_id=expected_camera_id
		)
	except ValueError as exc:
		raise AssertionError((spec.task, "camera contract", str(exc))) from exc
	catalog_path = task_dir / str(collection.get("geom_catalog", ""))
	if not catalog_path.is_file():
		raise AssertionError((spec.task, "geometry catalog missing", catalog_path))
	if _file_sha256(catalog_path) != collection.get("geom_catalog_sha256"):
		raise AssertionError((spec.task, "geometry catalog hash"))
	catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
	try:
		catalog_camera_id = validate_same_camera_contract(
			catalog, expected_camera_id=camera_id
		)
	except ValueError as exc:
		raise AssertionError((spec.task, "catalog camera contract", str(exc))) from exc
	if catalog_camera_id != camera_id:
		raise AssertionError((spec.task, "collection/catalog camera mismatch"))
	if len(data["records"]) != SUPPORT_EPISODES:
		raise AssertionError((spec.task, "record count", len(data["records"])))
	for index, record in enumerate(data["records"]):
		if record["index"] != index or record["episode"] != index:
			raise AssertionError((spec.task, "record order", record))
		image = np.asarray(Image.open(task_dir / record["image"]).convert("RGB"))
		mask = np.asarray(Image.open(task_dir / record["indexed_mask"]), dtype=np.uint8)
		if image.shape != (64, 64, 3) or mask.shape != (64, 64):
			raise AssertionError((spec.task, image.shape, mask.shape))
		if _decoded_sha256(image) != record["image_sha256"]:
			raise AssertionError((spec.task, index, "RGB hash"))
		if _decoded_sha256(mask) != record["indexed_mask_sha256"]:
			raise AssertionError((spec.task, index, "mask hash"))
		values = set(map(int, np.unique(mask)))
		expected_values = set(range(len(spec.roles) + 1))
		if not values.issubset(expected_values) or not (
			expected_values - {0}
		).issubset(values):
			raise AssertionError((spec.task, index, "mask IDs", values))


def _collect_task(args, task_dir: Path, spec: TaskSpec) -> dict:
	task_seed = int(args.seed)
	images_dir = task_dir / "support_frames"
	masks_dir = task_dir / "indexed_masks"
	images_dir.mkdir(parents=True)
	masks_dir.mkdir(parents=True)
	env = make_env(_collector_config(args, spec.task, task_seed))
	try:
		background = _find_wrapper(env, ColorMultiVideoBackgroundWrapper)
		physics = _find_physics(env)
		# Pixels obtains RGB through this renderer's default camera. Use that
		# exact view for labels too (Quadruped defaults to camera 2, not 0).
		camera_id = _find_wrapper(env, DMControlWrapper).camera_id
		camera_contract = build_same_camera_contract(camera_id)
		if background.active_split != SPLIT:
			raise RuntimeError((spec.task, "background split", background.active_split))
		if tuple(background.source_names) != SUPPORT_VIDEOS:
			raise RuntimeError((spec.task, "support videos", background.source_names))
		catalog = _catalog(physics)
		selections = tuple(
			_selected_objects(catalog, selector) for selector in spec.selectors
		)
		for left in range(len(selections)):
			for right in range(left + 1, len(selections)):
				if set(selections[left]).intersection(selections[right]):
					raise ValueError(
						f"{spec.task} role selectors {left}/{right} overlap "
						"in the model catalog."
					)
		matched = {
			role: _named_selection(catalog, selected)
			for role, selected in zip(spec.roles, selections)
		}
		catalog_payload = {
			"task": spec.task,
			"camera_id": camera_id,
			"camera_contract": camera_contract,
			"segmentation_object_types": _segmentation_constants(),
			"objects": catalog,
			"role_selectors": {
				role: _selector_payload(selector)
				for role, selector in zip(spec.roles, spec.selectors)
			},
			"matched_objects": matched,
		}
		catalog_path = task_dir / "geom_catalog.json"
		_write_json(catalog_path, catalog_payload)

		rng = np.random.default_rng(task_seed)
		records = []
		covered_videos = set()
		reset_attempts = 0
		visibility_rejections = {role: 0 for role in spec.roles}
		while len(records) < SUPPORT_EPISODES:
			if reset_attempts >= int(args.max_reset_attempts):
				raise RuntimeError(
					f"{spec.task}: failed to collect six valid visible-role frames after "
					f"{reset_attempts} resets; videos={sorted(covered_videos)}; "
					f"visibility_rejections={visibility_rejections}."
				)
			observation = env.reset()
			reset_attempts += 1
			active_video = Path(background.active_source).name
			if active_video not in SUPPORT_VIDEOS:
				raise RuntimeError((spec.task, "out-of-split video", active_video))
			if len(covered_videos) < len(SUPPORT_VIDEOS) and active_video in covered_videos:
				continue
			index = len(records)
			prefix_steps = 4 + 3 * index
			for _ in range(prefix_steps):
				action = rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				observation = env.step(action)[0]
			segmentation = physics.render(
				height=IMAGE_SIZE,
				width=IMAGE_SIZE,
				camera_id=camera_id,
				segmentation=True,
			)
			try:
				mask = _indexed_mask(
					segmentation, selections, role_names=spec.roles
				)
			except RoleVisibilityError as exc:
				# A role can be momentarily outside the camera.  Reject the frame;
				# never publish an empty or guessed support role.
				visibility_rejections[exc.role_name] += 1
				continue
			image = _latest_rgb(observation)
			frame_index = background.frame_index
			if not isinstance(frame_index, int) or frame_index < 0:
				raise RuntimeError((spec.task, "background frame", frame_index))
			image_name = f"support_{index:02d}.png"
			mask_name = f"support_{index:02d}.png"
			Image.fromarray(image, mode="RGB").save(images_dir / image_name)
			Image.fromarray(mask, mode="L").save(masks_dir / mask_name)
			counts = {role: int((mask == role_id).sum()) for role_id, role in enumerate(spec.roles, 1)}
			records.append({
				"index": index,
				"episode": index,
				"image": f"support_frames/{image_name}",
				"image_sha256": _decoded_sha256(image),
				"indexed_mask": f"indexed_masks/{mask_name}",
				"indexed_mask_sha256": _decoded_sha256(mask),
				"active_video": active_video,
				"frame_index": frame_index,
				"random_prefix_steps": prefix_steps,
				"selected_names": matched,
				"role_pixel_counts": counts,
			})
			covered_videos.add(active_video)
		if covered_videos != set(SUPPORT_VIDEOS):
			raise RuntimeError((spec.task, "support video coverage", covered_videos))

		annotations = {
			"format": FORMAT,
			"roles": list(spec.roles),
			"collection": {
				"support_schema": SUPPORT_SCHEMA,
				"task": spec.task,
				"observation": "rgb",
				"split": SPLIT,
				"seed": task_seed,
				"episodes": SUPPORT_EPISODES,
				"camera_id": camera_id,
				"camera_contract": camera_contract,
				"image_size": [IMAGE_SIZE, IMAGE_SIZE],
				"label_policy": "simulator_segmentation_support_only",
				"diagnostic_support": True,
				"mask_policy": "exact_k_role_indexed_uint8_0_through_k",
				"allowed_videos": list(SUPPORT_VIDEOS),
				"covered_videos": sorted(covered_videos),
				"selection_reset_attempts": reset_attempts,
				"visibility_rejections": visibility_rejections,
				"manifest_sha256": background.manifest_sha256,
				"combined_manifest_sha256": background.combined_manifest_sha256,
				"geom_catalog": "geom_catalog.json",
				"geom_catalog_sha256": _file_sha256(catalog_path),
				"role_selectors": {
					role: _selector_payload(selector)
					for role, selector in zip(spec.roles, spec.selectors)
				},
				"collector": Path(__file__).name,
				"collector_sha256": _file_sha256(Path(__file__).resolve()),
			},
			"records": records,
		}
		_write_json(task_dir / "annotations.json", annotations)
		_validate_task_pack(task_dir, spec, expected_camera_id=camera_id)
		return {
			"task": spec.task,
			"roles": list(spec.roles),
			"records": len(records),
			"annotations_sha256": _file_sha256(task_dir / "annotations.json"),
			"geom_catalog_sha256": _file_sha256(catalog_path),
		}
	finally:
		close = getattr(env, "close", None)
		if callable(close):
			close()


def _parse_args():
	parser = argparse.ArgumentParser(
		description="Collect exact indexed-mask support frames for DMC tasks."
	)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--task", choices=tuple(TASK_BY_NAME), required=True)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path)
	parser.add_argument("--seed", type=int, default=314159)
	parser.add_argument("--max-reset-attempts", type=int, default=1000)
	args = parser.parse_args()
	if args.max_reset_attempts < SUPPORT_EPISODES:
		parser.error("--max-reset-attempts must be at least six")
	if not args.video_root.expanduser().is_dir():
		parser.error(f"--video-root is not a directory: {args.video_root}")
	return args


def main() -> None:
	args = _parse_args()
	output = args.output.expanduser().resolve()
	if output.exists():
		raise FileExistsError(f"Refusing to overwrite support root: {output}")
	stage = output.with_name(f"{output.name}.incomplete.{os.getpid()}")
	if stage.exists():
		raise FileExistsError(f"Refusing to reuse staging path: {stage}")
	stage.mkdir(parents=True)
	try:
		print(f"collecting {args.task}", flush=True)
		result = _collect_task(args, stage, TASK_BY_NAME[args.task])
		stage.replace(output)
	except Exception:
		failed = output.with_name(f"{output.name}.failed.{os.getpid()}")
		if stage.exists() and not failed.exists():
			stage.replace(failed)
		raise
	print("CUTIE_MULTITASK_SUPPORT_OK", json.dumps(result, ensure_ascii=False))
	print(f"OUTPUT={output}")


if __name__ == "__main__":
	main()
