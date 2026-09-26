"""Dependency-light contracts for the unified video-segmentation benchmark.

The benchmark deliberately separates trajectory collection from model
inference.  A single trusted TD-MPC2 environment process freezes RGB frames,
offline-only MuJoCo masks, actions, and provenance.  Independent backend
processes then receive only RGB frames plus the same fixed support pack; they
never receive episode ground truth.  This allows Cutie, SAM 2.1, and SAM 3.1
to run in mutually incompatible Python/CUDA environments without changing the
scientific inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


DATASET_FORMAT = "unified_vos_frozen_dataset_v1"
BACKEND_INPUT_FORMAT = "unified_vos_backend_inputs_v1"
BACKEND_FORMAT = "unified_vos_backend_predictions_v1"
SUMMARY_FORMAT = "unified_vos_benchmark_summary_v1"
SUPPORT_FORMAT = "unified_vos_support_arrays_v1"

TASK_ROLES = {
	"reacher-visual-small": ("whole_arm", "goal"),
	"cartpole-swingup": ("cart", "pole"),
	"acrobot-swingup": ("upper_arm", "lower_arm"),
}
CONDITIONS = ("clean", "hard")
BACKENDS = ("cutie", "sam21", "sam31")

STRICT_EPISODES = 20
STRICT_ACTIONS_PER_EPISODE = 500
STRICT_FRAMES_PER_EPISODE = STRICT_ACTIONS_PER_EPISODE + 1
STRICT_RESOLUTION = 128
SUPPORT_RECORDS = 6

_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any) -> bytes:
	return (
		json.dumps(
			value,
			ensure_ascii=False,
			indent=2,
			sort_keys=True,
			allow_nan=False,
		)
		+ "\n"
	).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
	path.write_bytes(canonical_json_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
	value = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(value, dict):
		raise ValueError(f"Expected a JSON object: {path}")
	return value


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def sha256_json(value: Any) -> str:
	return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, label: str) -> str:
	if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
		raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
	return value


def strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
	if type(value) is not int or value < minimum:
		raise ValueError(f"{label} must be an integer >= {minimum}.")
	return value


def resolve_member(root: Path, relative: Any, label: str) -> Path:
	if not isinstance(relative, str) or not relative or "\\" in relative:
		raise ValueError(f"{label} must be a non-empty POSIX relative path.")
	candidate = (root / relative).resolve()
	try:
		candidate.relative_to(root.resolve())
	except ValueError as exc:
		raise ValueError(f"{label} escapes the benchmark root: {relative!r}.") from exc
	if not candidate.is_file():
		raise FileNotFoundError(candidate)
	return candidate


def dataset_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
	"""Return the path-independent immutable identity of one frozen dataset."""
	if payload.get("format") != DATASET_FORMAT:
		raise ValueError(f"Dataset format must be {DATASET_FORMAT!r}.")
	if payload.get("status") != "complete":
		raise ValueError("Frozen dataset is incomplete.")
	tasks = payload.get("tasks")
	conditions = payload.get("conditions")
	roles = payload.get("roles")
	if tasks != list(TASK_ROLES) or conditions != list(CONDITIONS):
		raise ValueError("Dataset task/condition order is not the frozen protocol.")
	if not isinstance(roles, dict) or {
		key: tuple(value) for key, value in roles.items()
	} != TASK_ROLES:
		raise ValueError("Dataset ordered role contract changed.")
	resolution = strict_int(payload.get("resolution"), "dataset.resolution", minimum=1)
	counts = payload.get("counts")
	if not isinstance(counts, dict):
		raise ValueError("Dataset counts must be an object.")
	episodes = strict_int(counts.get("episodes"), "counts.episodes", minimum=1)
	actions = strict_int(
		counts.get("actions_per_episode"),
		"counts.actions_per_episode",
		minimum=1,
	)
	frames = strict_int(
		counts.get("frames_per_episode"),
		"counts.frames_per_episode",
		minimum=2,
	)
	if frames != actions + 1:
		raise ValueError("Dataset frame/action count is not causal T+1/T.")
	protocol = payload.get("protocol")
	if not isinstance(protocol, dict):
		raise ValueError("Dataset protocol must be an object.")
	expected_protocol = {
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
	for key, expected in expected_protocol.items():
		if protocol.get(key) != expected:
			raise ValueError(f"Dataset protocol {key} changed.")
	seeds = payload.get("seeds")
	if not isinstance(seeds, dict) or set(seeds) != {
		"environment", "background", "action", "support"
	}:
		raise ValueError("Dataset seed domains are incomplete.")
	seed_values = [strict_int(value, f"seeds.{key}") for key, value in seeds.items()]
	if len(set(seed_values)) != len(seed_values):
		raise ValueError("Dataset seed domains must be pairwise distinct.")
	support = payload.get("support")
	episode_map = payload.get("episodes")
	if not isinstance(support, dict) or set(support) != set(TASK_ROLES):
		raise ValueError("Dataset support map is incomplete.")
	if not isinstance(episode_map, dict) or set(episode_map) != set(TASK_ROLES):
		raise ValueError("Dataset episode map is incomplete.")

	identity_support = {}
	identity_episodes = {}
	for task, expected_roles in TASK_ROLES.items():
		entry = support[task]
		if not isinstance(entry, dict):
			raise ValueError(f"Support entry for {task} must be an object.")
		if entry.get("format") != SUPPORT_FORMAT:
			raise ValueError(f"Support format changed for {task}.")
		if tuple(entry.get("roles", ())) != expected_roles:
			raise ValueError(f"Support roles changed for {task}.")
		if entry.get("records") != SUPPORT_RECORDS:
			raise ValueError(f"Support record count changed for {task}.")
		if entry.get("resolution") != resolution:
			raise ValueError(f"Support/runtime resolution mismatch for {task}.")
		identity_support[task] = {
			"format": SUPPORT_FORMAT,
			"roles": list(expected_roles),
			"records": SUPPORT_RECORDS,
			"resolution": resolution,
			"source_format": entry.get("source_format"),
			"source_sha256": require_sha256(
				entry.get("source_sha256"), f"support.{task}.source_sha256"
			),
			"arrays_sha256": require_sha256(
				entry.get("arrays_sha256"), f"support.{task}.arrays_sha256"
			),
			"asset_trace_sha256": require_sha256(
				entry.get("asset_trace_sha256"),
				f"support.{task}.asset_trace_sha256",
			),
		}

		conditions_payload = episode_map[task]
		if not isinstance(conditions_payload, dict) or set(conditions_payload) != set(
			CONDITIONS
		):
			raise ValueError(f"Dataset conditions are incomplete for {task}.")
		identity_episodes[task] = {}
		for condition in CONDITIONS:
			records = conditions_payload[condition]
			if not isinstance(records, list) or len(records) != episodes:
				raise ValueError(f"Wrong episode count for {task}/{condition}.")
			identity_records = []
			for expected_index, record in enumerate(records):
				if not isinstance(record, dict) or record.get("episode_index") != expected_index:
					raise ValueError(f"Episode ordering changed for {task}/{condition}.")
				if record.get("frames") != frames or record.get("actions") != actions:
					raise ValueError(f"Episode counts changed for {task}/{condition}.")
				identity_records.append({
					"episode_index": expected_index,
					"frames": frames,
					"actions": actions,
					"arrays_sha256": require_sha256(
						record.get("arrays_sha256"),
						f"episodes.{task}.{condition}[{expected_index}].arrays_sha256",
					),
					"rgb_arrays_sha256": require_sha256(
						record.get("rgb_arrays_sha256"),
						f"episodes.{task}.{condition}[{expected_index}].rgb_arrays_sha256",
					),
					"rgb_trace_sha256": require_sha256(
						record.get("rgb_trace_sha256"), "episode rgb trace"
					),
					"gt_trace_sha256": require_sha256(
						record.get("gt_trace_sha256"), "episode gt trace"
					),
					"action_trace_sha256": require_sha256(
						record.get("action_trace_sha256"), "episode action trace"
					),
					"physics_trace_sha256": require_sha256(
						record.get("physics_trace_sha256"), "episode physics trace"
					),
					"background_trace_sha256": (
						require_sha256(
							record.get("background_trace_sha256"),
							"episode hard-background trace",
						)
						if condition == "hard"
						else None
					),
					"random_policy_reward": float(record.get("random_policy_reward")),
					"role_gt_visible_frames": record.get("role_gt_visible_frames"),
				})
				if not math.isfinite(identity_records[-1]["random_policy_reward"]):
					raise ValueError("Episode random-policy reward must be finite.")
				if condition == "clean" and record.get("background_trace_sha256") is not None:
					raise ValueError("Clean episodes cannot carry a background trace.")
				visible = identity_records[-1]["role_gt_visible_frames"]
				if (
					not isinstance(visible, dict)
					or set(visible) != set(expected_roles)
					or any(
						type(value) is not int or value < 0 or value > frames
						for value in visible.values()
					)
				):
					raise ValueError(f"GT visibility accounting changed for {task}/{condition}.")
			identity_episodes[task][condition] = identity_records

		# Clean and hard pixels are two renderings of each exact same trajectory.
		for clean, hard in zip(
			identity_episodes[task]["clean"], identity_episodes[task]["hard"]
		):
			for key in (
				"episode_index", "frames", "actions", "gt_trace_sha256",
				"action_trace_sha256", "physics_trace_sha256",
				"random_policy_reward", "role_gt_visible_frames",
			):
				if clean[key] != hard[key]:
					raise ValueError(
						f"Clean/hard same-trajectory contract failed for {task}/{key}."
					)

	return {
		"format": DATASET_FORMAT,
		"tasks": list(TASK_ROLES),
		"conditions": list(CONDITIONS),
		"roles": {task: list(roles) for task, roles in TASK_ROLES.items()},
		"resolution": resolution,
		"counts": {
			"episodes": episodes,
			"actions_per_episode": actions,
			"frames_per_episode": frames,
		},
		"protocol": expected_protocol,
		"seeds": dict(seeds),
		"support": identity_support,
		"episodes": identity_episodes,
	}


def compute_dataset_id(payload: dict[str, Any]) -> str:
	return sha256_json(dataset_identity_payload(payload))


def validate_dataset_files(manifest_path: Path, *, strict_counts: bool) -> dict[str, Any]:
	manifest_path = manifest_path.expanduser().resolve()
	payload = load_json(manifest_path)
	identity = dataset_identity_payload(payload)
	if payload.get("dataset_id") != sha256_json(identity):
		raise ValueError("Frozen dataset identity digest mismatch.")
	if strict_counts and (
		identity["resolution"] != STRICT_RESOLUTION
		or identity["counts"] != {
			"episodes": STRICT_EPISODES,
			"actions_per_episode": STRICT_ACTIONS_PER_EPISODE,
			"frames_per_episode": STRICT_FRAMES_PER_EPISODE,
		}
	):
		raise ValueError("Scientific selection requires the fixed 128/20x500 protocol.")
	root = manifest_path.parent
	for task, entry in payload["support"].items():
		path = resolve_member(root, entry.get("arrays"), f"support.{task}.arrays")
		if file_sha256(path) != entry["arrays_sha256"]:
			raise ValueError(f"Support arrays changed for {task}.")
	for task, condition_map in payload["episodes"].items():
		for condition, records in condition_map.items():
			for record in records:
				path = resolve_member(
					root,
					record.get("arrays"),
					f"episodes.{task}.{condition}[{record.get('episode_index')}].arrays",
				)
				if file_sha256(path) != record["arrays_sha256"]:
					raise ValueError(
						f"Episode arrays changed for {task}/{condition}/"
						f"{record.get('episode_index')}."
					)
				rgb_path = resolve_member(
					root,
					record.get("rgb_arrays"),
					f"episodes.{task}.{condition}[{record.get('episode_index')}].rgb_arrays",
				)
				if file_sha256(rgb_path) != record["rgb_arrays_sha256"]:
					raise ValueError(
						f"Episode RGB arrays changed for {task}/{condition}/"
						f"{record.get('episode_index')}."
					)
	return payload


def validate_backend_inputs(
	manifest_path: Path,
	*,
	strict_counts: bool,
) -> tuple[dict[str, Any], dict[str, Path], dict[tuple[str, str, int], Path]]:
	"""Validate the GT-free worker view and return contained input paths."""
	manifest_path = manifest_path.expanduser().resolve()
	payload = load_json(manifest_path)
	if set(payload) != {
		"format", "status", "dataset_id", "tasks", "conditions", "roles",
		"resolution", "counts", "protocol", "support", "episodes",
	}:
		raise ValueError("Backend input manifest has unexpected or missing fields.")
	if payload.get("format") != BACKEND_INPUT_FORMAT or payload.get("status") != "complete":
		raise ValueError("Backend input manifest is incomplete or has the wrong format.")
	require_sha256(payload.get("dataset_id"), "backend_inputs.dataset_id")
	if payload.get("tasks") != list(TASK_ROLES):
		raise ValueError("Backend input task order changed.")
	if payload.get("conditions") != list(CONDITIONS):
		raise ValueError("Backend input condition order changed.")
	roles = payload.get("roles")
	if not isinstance(roles, dict) or {
		key: tuple(value) for key, value in roles.items()
	} != TASK_ROLES:
		raise ValueError("Backend input ordered roles changed.")
	resolution = strict_int(payload.get("resolution"), "backend_inputs.resolution", minimum=1)
	counts = payload.get("counts")
	if not isinstance(counts, dict) or set(counts) != {
		"episodes", "actions_per_episode", "frames_per_episode"
	}:
		raise ValueError("Backend input counts are malformed.")
	episodes = strict_int(counts.get("episodes"), "counts.episodes", minimum=1)
	actions = strict_int(
		counts.get("actions_per_episode"), "counts.actions_per_episode", minimum=1
	)
	frames = strict_int(counts.get("frames_per_episode"), "counts.frames_per_episode", minimum=2)
	if frames != actions + 1:
		raise ValueError("Backend input frame/action contract changed.")
	if strict_counts and (
		resolution != STRICT_RESOLUTION
		or episodes != STRICT_EPISODES
		or actions != STRICT_ACTIONS_PER_EPISODE
		or frames != STRICT_FRAMES_PER_EPISODE
	):
		raise ValueError("Backend scientific inputs must use fixed 128/20x500 counts.")
	if payload.get("protocol") != {
		"support_only_prompts": True,
		"episode_ground_truth_present": False,
		"episode_first_frame_gt_prompt": False,
		"mid_episode_reprompt": False,
		"causal_frame_order": True,
	}:
		raise ValueError("Backend prompt/GT protocol changed.")
	root = manifest_path.parent
	support_paths: dict[str, Path] = {}
	support = payload.get("support")
	if not isinstance(support, dict) or set(support) != set(TASK_ROLES):
		raise ValueError("Backend support map is incomplete.")
	for task, expected_roles in TASK_ROLES.items():
		entry = support[task]
		if not isinstance(entry, dict) or set(entry) != {
			"roles", "records", "resolution", "arrays", "arrays_sha256",
			"asset_trace_sha256",
		}:
			raise ValueError(f"Backend support entry is malformed for {task}.")
		if (
			tuple(entry.get("roles", ())) != expected_roles
			or entry.get("records") != SUPPORT_RECORDS
			or entry.get("resolution") != resolution
		):
			raise ValueError(f"Backend support contract changed for {task}.")
		path = resolve_member(root, entry.get("arrays"), f"support.{task}.arrays")
		if "scoring" in path.parts:
			raise ValueError("Backend support path aliases the scoring-only tree.")
		if file_sha256(path) != require_sha256(
			entry.get("arrays_sha256"), f"support.{task}.arrays_sha256"
		):
			raise ValueError(f"Backend support arrays changed for {task}.")
		require_sha256(entry.get("asset_trace_sha256"), f"support.{task}.asset_trace")
		support_paths[task] = path

	episode_paths: dict[tuple[str, str, int], Path] = {}
	episode_map = payload.get("episodes")
	if not isinstance(episode_map, dict) or set(episode_map) != set(TASK_ROLES):
		raise ValueError("Backend episode map is incomplete.")
	for task in TASK_ROLES:
		condition_map = episode_map[task]
		if not isinstance(condition_map, dict) or set(condition_map) != set(CONDITIONS):
			raise ValueError(f"Backend condition map is incomplete for {task}.")
		for condition in CONDITIONS:
			records = condition_map[condition]
			if not isinstance(records, list) or len(records) != episodes:
				raise ValueError(f"Backend episode count changed for {task}/{condition}.")
			for episode_index, record in enumerate(records):
				if not isinstance(record, dict) or set(record) != {
					"episode_index", "frames", "rgb_arrays", "rgb_arrays_sha256",
					"rgb_trace_sha256",
				}:
					raise ValueError(
						f"Backend episode schema changed for {task}/{condition}/{episode_index}."
					)
				if record.get("episode_index") != episode_index or record.get("frames") != frames:
					raise ValueError(f"Backend episode order/count changed for {task}/{condition}.")
				path = resolve_member(
					root,
					record.get("rgb_arrays"),
					f"episodes.{task}.{condition}[{episode_index}].rgb_arrays",
				)
				if "scoring" in path.parts:
					raise ValueError("Backend RGB path aliases the scoring-only tree.")
				if file_sha256(path) != require_sha256(
					record.get("rgb_arrays_sha256"), "backend episode RGB SHA"
				):
					raise ValueError(f"Backend RGB arrays changed for {task}/{condition}.")
				require_sha256(record.get("rgb_trace_sha256"), "backend episode RGB trace")
				episode_paths[(task, condition, episode_index)] = path
	return payload, support_paths, episode_paths
