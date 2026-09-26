"""Collect leakage-safe Visual-Small RGB/action/reward rollouts.

The collector consumes only the newest native 64x64 RGB frame in the normal
wrapped observation.  It never asks the environment for an alternate image or
state.  Each sequence stores N+1 observations and one NPZ containing the N
transitions that connect them::

    obs[t] -- action[t], reward[t] --> obs[t + 1]

Only the train and validation video-background splits are accepted.  Test and
support sources are fail-closed at the CLI, configuration, and runtime layers.
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
from types import SimpleNamespace
import uuid

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

FORMAT = "visual_small_object_rollout_v1"
TASK = "reacher-visual-small"
ALLOWED_SPLITS = ("train", "validation")
ALL_BACKGROUND_SPLITS = ("train", "validation", "test", "support")
ALL_EXPECTED_SOURCES = {
	"train": tuple(f"video{index}.mp4" for index in range(0, 70)),
	"validation": tuple(f"video{index}.mp4" for index in range(70, 80)),
	"test": tuple(f"video{index}.mp4" for index in range(80, 85)),
	"support": tuple(f"video{index}.mp4" for index in range(85, 90)),
}
EXPECTED_SOURCES = {split: ALL_EXPECTED_SOURCES[split] for split in ALLOWED_SPLITS}
ACTION_REPEAT = 2
DEFAULT_EPISODES_PER_SOURCE = 1
DEFAULT_STEPS_PER_EPISODE = 250
DEFAULT_SEED = 271828
ENV_BASE_SEED_OFFSET = 1_000_003
BACKGROUND_BASE_SEED_OFFSET = 2_000_003
ACTION_BASE_SEED_OFFSET = 3_000_017
SEED_DOMAIN_SIZE = 8
SEED_PAYLOAD_MODULUS = 2**29
SEED_ORDINAL_STRIDE = 1_000_003
SEED_DOMAIN_TAGS = {
	("train", "env"): 0,
	("train", "background"): 1,
	("train", "action"): 2,
	("validation", "env"): 3,
	("validation", "background"): 4,
	("validation", "action"): 5,
}
UINT32_MODULUS = 2**32
TRANSITION_KEYS = ("actions", "rewards", "terminated", "truncated")


class Config(SimpleNamespace):
	"""Small dict-like config accepted by the environment factory."""

	def get(self, key, default=None):
		return getattr(self, key, default)


@dataclass(frozen=True)
class ValidatedRolloutManifest:
	"""Validated manifest value plus its resolved provenance."""

	payload: dict
	root: Path
	path: Path
	file_sha256: str


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Collect native Visual-Small RGB rollouts with aligned actions and "
			"rewards from train or validation only."
		)
	)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path, default=None)
	parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
	episode_group = parser.add_mutually_exclusive_group()
	episode_group.add_argument(
		"--episodes-per-source",
		type=int,
		default=None,
		help="Collect exactly this many episodes from every allowed source.",
	)
	episode_group.add_argument(
		"--episodes",
		type=int,
		default=None,
		help=(
			"Total episodes; must be an exact multiple of the active allowlist "
			"size so every source receives the same count."
		),
	)
	parser.add_argument(
		"--steps-per-episode",
		"--num-transitions",
		dest="steps_per_episode",
		type=int,
		default=DEFAULT_STEPS_PER_EPISODE,
		help="Number of transitions N; each sequence stores N+1 RGB frames.",
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument(
		"--env-seed",
		type=int,
		default=None,
		help=(
			"Base seed used only for DMC task state; defaults to a "
			"split/type-separated derivation from --seed."
		),
	)
	parser.add_argument(
		"--action-seed",
		type=int,
		default=None,
		help=(
			"Base seed for independent per-episode action generators; defaults "
			"to a split/type-separated derivation from --seed."
		),
	)
	parser.add_argument(
		"--background-seed",
		type=int,
		default=None,
		help=(
			"Base seed used only for video source/start rejection sampling; "
			"defaults to a split/type-separated derivation from --seed."
		),
	)
	parser.add_argument(
		"--max-source-reset-attempts",
		type=int,
		default=5000,
		help="Maximum reset attempts per episode while balancing source use.",
	)
	return parser.parse_args()


def _uint32_seed(value, label):
	value = int(value)
	if not 0 <= value < UINT32_MODULUS:
		raise ValueError(f"{label} must be in [0, {UINT32_MODULUS - 1}].")
	return value


def _resolved_seeds(args):
	seed = _uint32_seed(args.seed, "seed")
	env_seed = _uint32_seed(
		(seed + ENV_BASE_SEED_OFFSET) % UINT32_MODULUS
		if args.env_seed is None
		else args.env_seed,
		"env_seed",
	)
	background_seed = _uint32_seed(
		(seed + BACKGROUND_BASE_SEED_OFFSET) % UINT32_MODULUS
		if args.background_seed is None
		else args.background_seed,
		"background_seed",
	)
	action_seed = _uint32_seed(
		(seed + ACTION_BASE_SEED_OFFSET) % UINT32_MODULUS
		if args.action_seed is None
		else args.action_seed,
		"action_seed",
	)
	return seed, env_seed, background_seed, action_seed


def make_collector_config(args, split, env_seed, background_seed):
	if split not in ALLOWED_SPLITS:
		raise ValueError(f"Object rollout split is forbidden: {split!r}.")
	return Config(
		task=TASK,
		obs="rgb",
		seed=int(env_seed),
		video_background_enabled=True,
		video_background_root=str(args.video_root.expanduser().resolve()),
		video_background_manifest_dir=(
			None
			if args.manifest_dir is None
			else str(args.manifest_dir.expanduser().resolve())
		),
		video_background_split=split,
		video_background_seed=int(background_seed),
		video_background_strength=1.0,
		video_background_total_frames=1000,
		video_background_source_cache_size=8,
		flat_anchor=False,
	)


def find_background_wrapper(env):
	# Keep the environment stack optional for --help, manifest validation, and
	# transition contract tests on hosts without dm_control.
	from envs.wrappers.video_background import ColorMultiVideoBackgroundWrapper

	current = env
	visited = set()
	while current is not None and id(current) not in visited:
		visited.add(id(current))
		if isinstance(current, ColorMultiVideoBackgroundWrapper):
			return current
		current = getattr(current, "env", None)
	raise RuntimeError(
		"ColorMultiVideoBackgroundWrapper is missing from the RGB path."
	)


def latest_rgb(observation):
	"""Return a writable HWC copy of the newest wrapped RGB frame."""
	if hasattr(observation, "detach"):
		value = observation.detach().cpu().numpy()
	else:
		value = np.asarray(observation)
	if value.ndim != 3 or value.shape[0] < 3:
		raise ValueError(
			"Expected a stacked channel-first RGB observation; "
			f"got shape={value.shape}."
		)
	frame = np.transpose(value[-3:], (1, 2, 0))
	frame = np.array(frame, dtype=np.uint8, order="C", copy=True)
	if frame.shape != (64, 64, 3) or value.dtype != np.uint8:
		raise ValueError(
			"Expected the newest native uint8 64x64 RGB observation; "
			f"got source_shape={value.shape}, source_dtype={value.dtype}, "
			f"frame_shape={frame.shape}."
		)
	return frame


def decoded_rgb_sha256(image):
	array = np.asarray(image)
	if array.shape != (64, 64, 3) or array.dtype != np.uint8:
		raise ValueError(
			f"RGB hash requires uint8 HWC (64, 64, 3), got {array.shape}/{array.dtype}."
		)
	return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path):
	digest = hashlib.sha256()
	with Path(path).open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _relative_asset(root, path):
	root = Path(root).resolve()
	path = Path(path).resolve()
	try:
		relative = path.relative_to(root)
	except ValueError as exc:
		raise RuntimeError(f"Asset escapes output root {root}: {path}") from exc
	if not relative.parts or ".." in relative.parts or relative.is_absolute():
		raise RuntimeError(f"Invalid relative asset path: {relative}")
	return relative.as_posix()


def _source_name(background, expected_split, allowed_sources):
	if background.active_split != expected_split:
		raise RuntimeError(
			f"Background split changed to {background.active_split!r}; "
			f"expected {expected_split!r}."
		)
	active_source = background.active_source
	if active_source is None:
		raise RuntimeError("Background has no active source after reset.")
	source = Path(active_source).name
	if source != str(source) or Path(source).name != source:
		raise RuntimeError(f"Background source is not a basename: {source!r}.")
	if source not in allowed_sources:
		raise RuntimeError(
			f"Out-of-split source selected for {expected_split!r}: {source!r}."
		)
	return source


def _frame_index(background):
	value = background.frame_index
	if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
		raise RuntimeError(f"Invalid source frame index: {value!r}.")
	value = int(value)
	if value < 0:
		raise RuntimeError(f"Negative source frame index: {value}.")
	return value


def _check_next_frame_index(previous, current, sequence_id):
	if previous is None:
		return
	if current != previous + 1 and current != 0:
		raise RuntimeError(
			f"{sequence_id}: source frame index jumped from {previous} to {current}."
		)


def _persist_frame(
	root,
	sequence_id,
	t,
	observation,
	background,
	split,
	source,
	allowed_sources,
	previous_frame_index,
):
	current_source = _source_name(background, split, allowed_sources)
	if current_source != source:
		raise RuntimeError(
			f"{sequence_id}: source changed within one episode from "
			f"{source!r} to {current_source!r}."
		)
	frame_index = _frame_index(background)
	_check_next_frame_index(previous_frame_index, frame_index, sequence_id)
	image = latest_rgb(observation)
	relative = (
		Path("sequences") / sequence_id / "frames" / f"t_{int(t):06d}.png"
	)
	path = Path(root) / relative
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(image, mode="RGB").save(path, format="PNG")
	with Image.open(path) as stream:
		decoded = np.asarray(stream.convert("RGB"), dtype=np.uint8)
	if not np.array_equal(decoded, image):
		raise RuntimeError(f"PNG round-trip changed RGB bytes at {path}.")
	return {
		"t": int(t),
		"image": _relative_asset(root, path),
		"image_sha256": decoded_rgb_sha256(decoded),
		"source_frame_index": frame_index,
	}


def _action_bounds(env):
	low = np.asarray(env.action_space.low, dtype=np.float64)
	high = np.asarray(env.action_space.high, dtype=np.float64)
	if low.shape != high.shape or low.ndim != 1 or low.size < 1:
		raise ValueError(
			f"Expected one-dimensional action bounds, got {low.shape}/{high.shape}."
		)
	if not np.isfinite(low).all() or not np.isfinite(high).all():
		raise ValueError("Object rollout collection requires finite action bounds.")
	if not np.all(low < high):
		raise ValueError("Every action lower bound must be strictly below its upper bound.")
	return low, high


def _sample_action(rng, low, high):
	# Apply exactly the float32 action that is persisted in the transition asset.
	action = np.asarray(rng.uniform(low, high), dtype=np.float32)
	if action.shape != low.shape or not np.isfinite(action).all():
		raise RuntimeError(f"Invalid sampled action: shape={action.shape}.")
	return action


def _bool_scalar(value, label):
	if hasattr(value, "detach"):
		value = value.detach().cpu()
	if hasattr(value, "item"):
		value = value.item()
	if isinstance(value, np.ndarray):
		if value.size != 1:
			raise ValueError(f"{label} must be scalar, got shape {value.shape}.")
		value = value.reshape(()).item()
	return bool(value)


def _atomic_npz(path, *, actions, rewards, terminated, truncated):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
	np.savez(
		temporary,
		actions=actions,
		rewards=rewards,
		terminated=terminated,
		truncated=truncated,
	)
	os.replace(temporary, path)


def validate_transition_arrays(arrays, num_transitions, action_dim):
	"""Validate the frozen transition-array contract without environment imports."""
	keys = tuple(arrays.keys()) if hasattr(arrays, "keys") else tuple(arrays.files)
	if keys != TRANSITION_KEYS:
		raise ValueError(
			f"Transition keys must be ordered exactly as {TRANSITION_KEYS}, got {keys}."
		)
	expected = {
		"actions": ((int(num_transitions), int(action_dim)), np.dtype(np.float32)),
		"rewards": ((int(num_transitions),), np.dtype(np.float32)),
		"terminated": ((int(num_transitions),), np.dtype(np.bool_)),
		"truncated": ((int(num_transitions),), np.dtype(np.bool_)),
	}
	for key, (shape, dtype) in expected.items():
		array = np.asarray(arrays[key])
		if array.shape != shape or array.dtype != dtype:
			raise ValueError(
				f"Transition {key} must be {shape}/{dtype}, "
				f"got {array.shape}/{array.dtype}."
			)
	if not np.isfinite(np.asarray(arrays["actions"])).all():
		raise ValueError("Transition actions contain non-finite values.")
	if not np.isfinite(np.asarray(arrays["rewards"])).all():
		raise ValueError("Transition rewards contain non-finite values.")
	if np.any(
		np.asarray(arrays["terminated"]) & np.asarray(arrays["truncated"])
	):
		raise ValueError("A transition cannot be both terminated and truncated.")
	return True


def _validate_transition_asset(path, num_transitions, action_dim):
	with np.load(path, allow_pickle=False) as payload:
		validate_transition_arrays(payload, num_transitions, action_dim)


def _persist_transitions(
	root,
	sequence_id,
	actions,
	rewards,
	terminated,
	truncated,
	action_dim,
):
	actions = np.asarray(actions, dtype=np.float32).reshape(-1, action_dim)
	rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
	terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1)
	truncated = np.asarray(truncated, dtype=np.bool_).reshape(-1)
	num_transitions = len(actions)
	if not (
		len(rewards) == len(terminated) == len(truncated) == num_transitions
	):
		raise RuntimeError("Transition arrays are not aligned.")
	path = Path(root) / "sequences" / sequence_id / "transitions.npz"
	_atomic_npz(
		path,
		actions=actions,
		rewards=rewards,
		terminated=terminated,
		truncated=truncated,
	)
	_validate_transition_asset(path, num_transitions, action_dim)
	return {
		"path": _relative_asset(root, path),
		"file_sha256": file_sha256(path),
	}


def _close_env(env):
	close = getattr(env, "close", None)
	if callable(close):
		close()


def _preselect_background_source(allowed_sources, background_seed):
	"""Predict the compositor's source without decoding or constructing an env.

	``VideoBackgroundCompositor`` creates a fresh legacy ``RandomState`` from
	the background seed and uses its first ``randint`` to choose a source.  Use
	the identical operation here for rejection sampling; the real compositor
	still owns its independent RNG and therefore consumes its second draw for
	the start frame exactly as before.
	"""
	sources = tuple(map(str, allowed_sources))
	if not sources:
		raise ValueError("Background preselection requires a non-empty allowlist.")
	random = np.random.RandomState(int(background_seed))
	return sources[int(random.randint(0, len(sources)))]


def _select_balanced_environment(
	make_env,
	args,
	split,
	allowed_sources,
	manifest_sha256,
	combined_manifest_sha256,
	source_counts,
	max_attempts,
	env_seed,
	base_background_seed,
	candidate_ordinal,
	used_background_seeds,
):
	minimum_count = min(source_counts.values())
	targets = {
		source for source, count in source_counts.items() if count == minimum_count
	}
	for local_attempt in range(1, max_attempts + 1):
		background_seed = _domain_seed(
			base_background_seed, split, "background", candidate_ordinal
		)
		candidate_ordinal += 1
		if background_seed in used_background_seeds:
			raise RuntimeError(f"Background seed was reused: {background_seed}.")
		predicted_source = _preselect_background_source(
			allowed_sources, background_seed
		)
		if predicted_source not in targets:
			continue
		env = make_env(
			make_collector_config(args, split, env_seed, background_seed)
		)
		background = find_background_wrapper(env)
		if tuple(map(str, background.source_names)) != tuple(allowed_sources):
			_close_env(env)
			raise RuntimeError("Environment source allowlist changed during collection.")
		if (
			background.manifest_sha256 != manifest_sha256
			or background.combined_manifest_sha256 != combined_manifest_sha256
		):
			_close_env(env)
			raise RuntimeError("Environment manifest fingerprint changed during collection.")
		observation = env.reset()
		source = _source_name(background, split, allowed_sources)
		if source != predicted_source:
			_close_env(env)
			raise RuntimeError(
				"Background preselection disagrees with the real compositor: "
				f"predicted {predicted_source!r}, observed {source!r}."
			)
		if source not in targets:
			_close_env(env)
			raise RuntimeError(
				f"Preselected source {source!r} is not currently least-used."
			)
		used_background_seeds.add(background_seed)
		return (
			env,
			background,
			observation,
			source,
			background_seed,
			local_attempt,
			candidate_ordinal,
		)
	raise RuntimeError(
		f"Could not select a least-used {split} source after {max_attempts} resets; "
		f"eligible sources were {sorted(targets)}."
	)


def _domain_seed(base_seed, split, seed_type, ordinal):
	"""Return a unique uint32 seed from a split/type-disjoint domain."""
	try:
		tag = SEED_DOMAIN_TAGS[(split, seed_type)]
	except KeyError as exc:
		raise ValueError(f"Unknown seed domain: {(split, seed_type)!r}.") from exc
	ordinal = int(ordinal)
	if not 0 <= ordinal < SEED_PAYLOAD_MODULUS:
		raise ValueError("Seed ordinal exhausts the disjoint uint32 domain.")
	payload = (
		int(base_seed) % SEED_PAYLOAD_MODULUS
		+ SEED_ORDINAL_STRIDE * ordinal
	) % SEED_PAYLOAD_MODULUS
	return int(payload * SEED_DOMAIN_SIZE + tag)


def collect(args, root):
	if args.split not in ALLOWED_SPLITS:
		raise ValueError(f"Object rollout split is forbidden: {args.split!r}.")
	if args.episodes_per_source is not None and args.episodes_per_source < 1:
		raise ValueError("--episodes-per-source must be positive.")
	if args.episodes is not None and args.episodes < 1:
		raise ValueError("--episodes must be positive.")
	if args.steps_per_episode < 1:
		raise ValueError("--steps-per-episode must be positive.")
	if args.max_source_reset_attempts < 1:
		raise ValueError("--max-source-reset-attempts must be positive.")
	seed, base_env_seed, base_background_seed, base_action_seed = _resolved_seeds(args)
	# Deliberately lazy: importing this module, --help, and contract validation
	# must work on machines that do not have dm_control installed.
	from envs.dmcontrol import make_env
	from envs.wrappers.video_background import ColorMultiSplitSelector

	selector = ColorMultiSplitSelector(
		args.video_root.expanduser().resolve(),
		None
		if args.manifest_dir is None
		else args.manifest_dir.expanduser().resolve(),
	)
	allowed_sources = tuple(map(str, selector.source_names(args.split)))
	manifest_sha256 = selector.manifest_sha256(args.split)
	combined_manifest_sha256 = selector.combined_manifest_sha256
	if allowed_sources != EXPECTED_SOURCES[args.split]:
		raise RuntimeError(
			f"{args.split} allowlist differs from the frozen source partition."
		)
	if not allowed_sources or len(allowed_sources) != len(set(allowed_sources)):
		raise RuntimeError("The active split source allowlist is empty or duplicated.")
	if any(Path(source).name != source for source in allowed_sources):
		raise RuntimeError("The active split allowlist contains a non-basename source.")
	if args.episodes is not None:
		if args.episodes % len(allowed_sources):
			raise ValueError(
				"--episodes must be an exact multiple of the active source count "
				f"{len(allowed_sources)}."
			)
		episodes_per_source = args.episodes // len(allowed_sources)
	else:
		episodes_per_source = (
			DEFAULT_EPISODES_PER_SOURCE
			if args.episodes_per_source is None
			else int(args.episodes_per_source)
		)
	source_counts = {source: 0 for source in allowed_sources}
	sequences = []
	used_env_seeds = set()
	used_background_seeds = set()
	used_action_seeds = set()
	candidate_ordinal = 0
	num_episodes = len(allowed_sources) * episodes_per_source
	action_dim = None
	expected_action_low = None
	expected_action_high = None

	for episode in range(num_episodes):
		# Physics and action seeds are frozen by collection ordinal before source
		# selection. Rejection sampling varies only the background seed.
		env_seed = _domain_seed(base_env_seed, args.split, "env", episode)
		action_seed = _domain_seed(base_action_seed, args.split, "action", episode)
		if env_seed in used_env_seeds:
			raise RuntimeError(f"Environment seed was reused: {env_seed}.")
		used_env_seeds.add(env_seed)
		if action_seed in used_action_seeds:
			raise RuntimeError(f"Action seed was reused: {action_seed}.")
		used_action_seeds.add(action_seed)
		(
			env,
			background,
			observation,
			source,
			background_seed,
			selection_attempt,
			candidate_ordinal,
		) = _select_balanced_environment(
			make_env,
			args,
			args.split,
			allowed_sources,
			manifest_sha256,
			combined_manifest_sha256,
			source_counts,
			args.max_source_reset_attempts,
			env_seed,
			base_background_seed,
			candidate_ordinal,
			used_background_seeds,
		)
		max_episode_steps = int(env.max_episode_steps)
		if args.steps_per_episode > max_episode_steps:
			_close_env(env)
			raise ValueError(
				f"--steps-per-episode={args.steps_per_episode} exceeds the wrapped "
				f"episode limit {max_episode_steps}."
			)
		action_low, action_high = _action_bounds(env)
		if action_dim is None:
			action_dim = int(action_low.size)
			expected_action_low = action_low.copy()
			expected_action_high = action_high.copy()
		elif not (
			np.array_equal(action_low, expected_action_low)
			and np.array_equal(action_high, expected_action_high)
		):
			_close_env(env)
			raise RuntimeError("Action bounds changed between rollout environments.")
		sequence_id = f"{args.split}_episode_{episode:06d}"
		action_rng = np.random.default_rng(action_seed)
		frames = [
			_persist_frame(
				root,
				sequence_id,
				0,
				observation,
				background,
				args.split,
				source,
				allowed_sources,
				None,
			)
		]
		actions = []
		rewards = []
		terminated_values = []
		truncated_values = []
		last_done = False

		for t in range(args.steps_per_episode):
			action = _sample_action(action_rng, action_low, action_high)
			next_observation, reward, done, info = env.step(action)
			reward = float(reward)
			if not math.isfinite(reward):
				raise RuntimeError(
					f"{sequence_id}: transition {t} returned non-finite reward {reward}."
				)
			done = _bool_scalar(done, "done")
			terminated = _bool_scalar(
				info.get("terminated", False), "terminated"
			)
			truncated = bool(done and not terminated)
			if terminated and truncated:
				raise RuntimeError(
					f"{sequence_id}: transition {t} is both terminated and truncated."
				)
			actions.append(action.copy())
			rewards.append(np.float32(reward))
			terminated_values.append(terminated)
			truncated_values.append(truncated)
			frames.append(
				_persist_frame(
					root,
					sequence_id,
					t + 1,
					next_observation,
					background,
					args.split,
					source,
					allowed_sources,
					frames[-1]["source_frame_index"],
				)
			)
			observation = next_observation
			last_done = done
			if done and t + 1 < args.steps_per_episode:
				raise RuntimeError(
					f"{sequence_id}: episode ended after {t + 1} transitions, before "
					f"the requested {args.steps_per_episode}."
				)

		if len(frames) != args.steps_per_episode + 1:
			raise RuntimeError(f"{sequence_id}: observation/transition count mismatch.")
		transitions_asset = _persist_transitions(
			root,
			sequence_id,
			actions,
			rewards,
			terminated_values,
			truncated_values,
			action_dim,
		)
		sequences.append(
			{
				"sequence_id": sequence_id,
				"split": args.split,
				"episode": int(episode),
				"source": source,
				"env_seed": int(env_seed),
				"background_seed": int(background_seed),
				"action_seed": int(action_seed),
				"num_transitions": int(args.steps_per_episode),
				"frames": frames,
				"transitions_asset": transitions_asset,
				"source_selection_reset_attempt": int(selection_attempt),
				"ended_by_environment": bool(last_done),
			}
		)
		source_counts[source] += 1
		_close_env(env)

	expected_source_count = int(episodes_per_source)
	if any(count != expected_source_count for count in source_counts.values()):
		raise RuntimeError(
			"Every allowed source must have exactly "
			f"{expected_source_count} episodes; got {source_counts}."
		)
	selected_sources = [source for source, count in source_counts.items() if count]
	if tuple(selected_sources) != allowed_sources:
		raise RuntimeError("Collection did not cover the complete source allowlist.")
	if len(used_env_seeds | used_background_seeds | used_action_seeds) != 3 * num_episodes:
		raise RuntimeError("Environment/background/action seed domains overlap or reuse a seed.")

	return {
		"format": FORMAT,
		"collection": {
			"task": TASK,
			"split": args.split,
			"observation": "newest_wrapped_rgb",
			"native64": True,
			"resolution_status": "native64_agent_observation",
			"true_high_resolution": False,
			"upscaled_copy_saved": False,
			"no_test": True,
			"no_support": True,
			"action_repeat": ACTION_REPEAT,
			"action_policy": "independent_per_episode_uniform_action_space_v1",
			"seed": int(seed),
			"manifest_sha256": manifest_sha256,
			"combined_manifest_sha256": combined_manifest_sha256,
			"num_episodes": int(num_episodes),
			"episodes_per_source": int(episodes_per_source),
			"num_transitions_per_episode": int(args.steps_per_episode),
			"action_dim": action_dim,
			"action_dtype": "float32",
			"reward_dtype": "float32",
			"image_sha256_contract": "sha256(contiguous_decoded_uint8_hwc_rgb_bytes)",
			"transition_alignment": "obs[t] -- action[t], reward[t] --> obs[t+1]",
			"allowed_sources": list(allowed_sources),
			"selected_sources": selected_sources,
			"source_counts": source_counts,
			"source_selection": "least_used_rejection_v1",
			"env_seed_assignment": "fixed_episode_ordinal_disjoint_domain_v1",
			"background_seed_assignment": "source_rejection_disjoint_domain_v1",
			"action_seed_assignment": "fixed_episode_ordinal_disjoint_domain_v1",
		},
		"sequences": sequences,
	}


def _validate_relative_manifest_assets(root, payload):
	root = Path(root).resolve()
	seen_paths = set()
	sequences = payload.get("sequences")
	if not isinstance(sequences, list) or not sequences:
		raise RuntimeError("Manifest must contain at least one sequence.")
	for expected_episode, sequence in enumerate(sequences):
		if sequence["episode"] != expected_episode:
			raise RuntimeError("Sequence episode indices must be contiguous.")
		if sequence["split"] != payload["collection"]["split"]:
			raise RuntimeError("Sequence split differs from collection split.")
		if Path(sequence["source"]).name != sequence["source"]:
			raise RuntimeError("Sequence source must be a basename.")
		frames = sequence["frames"]
		if len(frames) != sequence["num_transitions"] + 1:
			raise RuntimeError("Sequence must contain N+1 frames for N transitions.")
		previous_index = None
		for expected_t, frame in enumerate(frames):
			if set(frame) != {
				"t", "image", "image_sha256", "source_frame_index"
			}:
				raise RuntimeError(
					"Frame records must contain exactly t, image, image_sha256, "
					"and source_frame_index."
				)
			if isinstance(frame["t"], bool) or frame["t"] != expected_t:
				raise RuntimeError("Frame t values must be contiguous from zero.")
			frame_index = frame["source_frame_index"]
			if (
				isinstance(frame_index, bool)
				or not isinstance(frame_index, int)
				or frame_index < 0
			):
				raise RuntimeError("source_frame_index must be a non-negative integer.")
			if not _is_sha256(frame["image_sha256"]):
				raise RuntimeError("image_sha256 must be a lowercase SHA-256 value.")
			asset = frame["image"]
			asset_path = Path(asset)
			if asset_path.is_absolute() or ".." in asset_path.parts:
				raise RuntimeError(f"Frame asset must be a safe relative path: {asset}.")
			if asset in seen_paths:
				raise RuntimeError(f"Duplicate manifest asset path: {asset}.")
			seen_paths.add(asset)
			path = (root / asset).resolve()
			if _relative_asset(root, path) != asset or not path.is_file():
				raise RuntimeError(f"Invalid frame asset: {asset}.")
			with Image.open(path) as stream:
				decoded = np.asarray(stream.convert("RGB"), dtype=np.uint8)
			if frame["image_sha256"] != decoded_rgb_sha256(decoded):
				raise RuntimeError(f"Decoded RGB hash mismatch: {asset}.")
			_check_next_frame_index(
				previous_index, frame_index, sequence["sequence_id"]
			)
			previous_index = frame_index
		asset_record = sequence["transitions_asset"]
		if set(asset_record) != {"path", "file_sha256"}:
			raise RuntimeError(
				"transitions_asset must contain exactly path and file_sha256."
			)
		asset = asset_record["path"]
		if not _is_sha256(asset_record["file_sha256"]):
			raise RuntimeError("transitions_asset.file_sha256 must be lowercase SHA-256.")
		asset_path = Path(asset)
		if asset_path.is_absolute() or ".." in asset_path.parts:
			raise RuntimeError(
				f"Transition asset must be a safe relative path: {asset}."
			)
		if asset in seen_paths:
			raise RuntimeError(f"Duplicate manifest asset path: {asset}.")
		seen_paths.add(asset)
		path = (root / asset).resolve()
		if _relative_asset(root, path) != asset or not path.is_file():
			raise RuntimeError(f"Invalid transition asset: {asset}.")
		if asset_record["file_sha256"] != file_sha256(path):
			raise RuntimeError(f"Transition asset hash mismatch: {asset}.")
		_validate_transition_asset(
			path,
			sequence["num_transitions"],
			payload["collection"]["action_dim"],
		)


def _is_sha256(value):
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in "0123456789abcdef" for character in value)
	)


def _background_manifest_fingerprints(background_manifest_dir=None):
	directory = (
		PROJECT_ROOT / "envs" / "background_manifests"
		if background_manifest_dir is None
		else Path(background_manifest_dir).expanduser().resolve()
	)
	if not directory.is_dir():
		raise FileNotFoundError(f"Background manifest directory not found: {directory}")
	fingerprints = {}
	owners = set()
	for split in ALL_BACKGROUND_SPLITS:
		path = directory / f"color_multi_{split}.json"
		if not path.is_file():
			raise FileNotFoundError(f"Background manifest not found: {path}")
		raw = path.read_bytes()
		try:
			value = json.loads(raw.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as exc:
			raise ValueError(f"Invalid background manifest JSON: {path}") from exc
		if not isinstance(value, dict) or set(value) != {
			"schema_version", "name", "sources"
		}:
			raise ValueError(
				f"Background manifest must contain exactly schema_version/name/sources: {path}"
			)
		if (
			isinstance(value.get("schema_version"), bool)
			or value.get("schema_version") != 1
		):
			raise ValueError(f"Unsupported background manifest schema: {path}")
		if (
			value.get("name") != split
			or not isinstance(value.get("sources"), list)
			or tuple(value.get("sources", ())) != ALL_EXPECTED_SOURCES[split]
		):
			raise ValueError(f"Background manifest violates frozen {split} sources: {path}")
		if owners.intersection(value["sources"]):
			raise ValueError(f"Background source overlap detected in {path}.")
		owners.update(value["sources"])
		fingerprints[split] = hashlib.sha256(raw).hexdigest()
	combined = "".join(
		f"{split}:{fingerprints[split]}\n" for split in ALL_BACKGROUND_SPLITS
	).encode("ascii")
	return fingerprints, hashlib.sha256(combined).hexdigest()


def load_and_validate_rollout_manifest(path, background_manifest_dir=None):
	"""Load and validate a frozen rollout manifest and all referenced assets.

	This entry point intentionally has no environment dependency, so downstream
	loaders and contract tests can reject malformed or cross-split data before
	initializing a simulator or learner.
	"""
	path = Path(path).expanduser().resolve()
	if path.is_dir():
		path = path / "manifest.json"
	if not path.is_file():
		raise FileNotFoundError(f"Rollout manifest not found: {path}")
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except (UnicodeDecodeError, json.JSONDecodeError) as exc:
		raise ValueError(f"Invalid rollout manifest JSON: {path}") from exc
	if not isinstance(payload, dict) or set(payload) != {
		"format", "collection", "sequences"
	}:
		raise ValueError("Rollout manifest must contain exactly format/collection/sequences.")
	if payload.get("format") != FORMAT:
		raise ValueError(f"Expected format={FORMAT!r}.")
	collection = payload.get("collection")
	if not isinstance(collection, dict):
		raise ValueError("Manifest collection metadata is required.")
	expected_collection_keys = {
		"task",
		"split",
		"observation",
		"native64",
		"resolution_status",
		"true_high_resolution",
		"upscaled_copy_saved",
		"no_test",
		"no_support",
		"action_repeat",
		"action_policy",
		"seed",
		"manifest_sha256",
		"combined_manifest_sha256",
		"num_episodes",
		"episodes_per_source",
		"num_transitions_per_episode",
		"action_dim",
		"allowed_sources",
		"selected_sources",
		"source_counts",
		"action_dtype",
		"reward_dtype",
		"image_sha256_contract",
		"transition_alignment",
		"source_selection",
		"env_seed_assignment",
		"background_seed_assignment",
		"action_seed_assignment",
	}
	if set(collection) != expected_collection_keys:
		raise ValueError(
			"Manifest collection keys differ from the frozen contract: "
			f"missing={sorted(expected_collection_keys - set(collection))}, "
			f"extra={sorted(set(collection) - expected_collection_keys)}."
		)
	split = collection.get("split")
	if split not in ALLOWED_SPLITS:
		raise ValueError(
			f"Rollout split {split!r} is forbidden; expected train or validation."
		)
	if collection.get("task") != TASK:
		raise ValueError(f"collection.task must be {TASK!r}.")
	if collection.get("observation") != "newest_wrapped_rgb":
		raise ValueError("collection.observation must be newest_wrapped_rgb.")
	if (
		collection.get("native64") is not True
		or collection.get("resolution_status") != "native64_agent_observation"
		or collection.get("true_high_resolution") is not False
		or collection.get("upscaled_copy_saved") is not False
	):
		raise ValueError("Manifest does not satisfy the native64 observation contract.")
	if collection.get("no_test") is not True or collection.get("no_support") is not True:
		raise ValueError("collection.no_test and collection.no_support must both be true.")
	if collection.get("action_repeat") != ACTION_REPEAT:
		raise ValueError(f"collection.action_repeat must be {ACTION_REPEAT}.")
	if collection.get("action_policy") != "independent_per_episode_uniform_action_space_v1":
		raise ValueError("Unexpected action_policy.")
	collection_seed = collection.get("seed")
	if (
		isinstance(collection_seed, bool)
		or not isinstance(collection_seed, int)
		or not 0 <= collection_seed < UINT32_MODULUS
	):
		raise ValueError("collection.seed must be a uint32 integer.")
	if (
		collection.get("action_dtype") != "float32"
		or collection.get("reward_dtype") != "float32"
		or collection.get("image_sha256_contract")
		!= "sha256(contiguous_decoded_uint8_hwc_rgb_bytes)"
		or collection.get("transition_alignment")
		!= "obs[t] -- action[t], reward[t] --> obs[t+1]"
		or collection.get("source_selection") != "least_used_rejection_v1"
		or collection.get("env_seed_assignment")
		!= "fixed_episode_ordinal_disjoint_domain_v1"
		or collection.get("background_seed_assignment")
		!= "source_rejection_disjoint_domain_v1"
		or collection.get("action_seed_assignment")
		!= "fixed_episode_ordinal_disjoint_domain_v1"
	):
		raise ValueError("Collection dtype/hash/alignment/seed policy metadata is invalid.")
	if not _is_sha256(collection.get("manifest_sha256")) or not _is_sha256(
		collection.get("combined_manifest_sha256")
	):
		raise ValueError("Manifest fingerprints must be lowercase SHA-256 values.")
	manifest_fingerprints, combined_fingerprint = _background_manifest_fingerprints(
		background_manifest_dir
	)
	if collection["manifest_sha256"] != manifest_fingerprints[split]:
		raise ValueError(f"collection.manifest_sha256 does not match {split}.")
	if collection["combined_manifest_sha256"] != combined_fingerprint:
		raise ValueError("collection.combined_manifest_sha256 does not match manifests.")
	allowed_sources = tuple(collection.get("allowed_sources", ()))
	if allowed_sources != EXPECTED_SOURCES[split]:
		raise ValueError(f"collection.allowed_sources is not the frozen {split} allowlist.")
	if tuple(collection.get("selected_sources", ())) != allowed_sources:
		raise ValueError("Every allowed source must be selected.")
	episodes_per_source = collection.get("episodes_per_source")
	if isinstance(episodes_per_source, bool) or not isinstance(episodes_per_source, int):
		raise ValueError("episodes_per_source must be a positive integer.")
	if episodes_per_source < 1:
		raise ValueError("episodes_per_source must be positive.")
	expected_counts = {source: episodes_per_source for source in allowed_sources}
	actual_declared_counts = collection.get("source_counts")
	if (
		actual_declared_counts != expected_counts
		or not isinstance(actual_declared_counts, dict)
		or any(
			isinstance(value, bool) or not isinstance(value, int)
			for value in actual_declared_counts.values()
		)
	):
		raise ValueError("source_counts must assign the same exact count to every source.")
	expected_episode_count = len(allowed_sources) * episodes_per_source
	if collection.get("num_episodes") != expected_episode_count:
		raise ValueError("num_episodes does not match allowlist coverage.")
	num_transitions = collection.get("num_transitions_per_episode")
	action_dim = collection.get("action_dim")
	for value, label in ((num_transitions, "num_transitions_per_episode"), (action_dim, "action_dim")):
		if isinstance(value, bool) or not isinstance(value, int) or value < 1:
			raise ValueError(f"collection.{label} must be a positive integer.")

	sequences = payload.get("sequences")
	if not isinstance(sequences, list) or len(sequences) != expected_episode_count:
		raise ValueError("sequences length does not match num_episodes.")
	seen_ids = set()
	env_seeds = set()
	background_seeds = set()
	action_seeds = set()
	actual_counts = {source: 0 for source in allowed_sources}
	for expected_episode, sequence in enumerate(sequences):
		if not isinstance(sequence, dict):
			raise ValueError(f"sequences[{expected_episode}] must be an object.")
		expected_sequence_keys = {
			"sequence_id", "split", "episode", "source", "env_seed",
			"background_seed", "action_seed", "num_transitions", "frames",
			"transitions_asset",
			"source_selection_reset_attempt", "ended_by_environment",
		}
		if set(sequence) != expected_sequence_keys:
			raise ValueError(
				f"sequences[{expected_episode}] keys differ from the frozen contract: "
				f"missing={sorted(expected_sequence_keys - set(sequence))}, "
				f"extra={sorted(set(sequence) - expected_sequence_keys)}."
			)
		sequence_id = sequence["sequence_id"]
		if sequence_id != f"{split}_episode_{expected_episode:06d}":
			raise ValueError("sequence_id/order differs from the frozen convention.")
		if sequence_id in seen_ids:
			raise ValueError(f"Duplicate sequence_id: {sequence_id}.")
		seen_ids.add(sequence_id)
		if (
			isinstance(sequence.get("episode"), bool)
			or sequence.get("episode") != expected_episode
			or sequence.get("split") != split
		):
			raise ValueError("Sequence episode/split does not match its manifest position.")
		source = sequence.get("source")
		if source not in actual_counts or Path(source).name != source:
			raise ValueError(f"Out-of-split or non-basename source: {source!r}.")
		actual_counts[source] += 1
		if sequence.get("num_transitions") != num_transitions:
			raise ValueError("All sequences must use num_transitions_per_episode.")
		selection_attempt = sequence.get("source_selection_reset_attempt")
		if (
			isinstance(selection_attempt, bool)
			or not isinstance(selection_attempt, int)
			or selection_attempt < 1
		):
			raise ValueError("source_selection_reset_attempt must be a positive integer.")
		if not isinstance(sequence.get("ended_by_environment"), bool):
			raise ValueError("ended_by_environment must be boolean.")
		for field, seed_type, seen in (
			("env_seed", "env", env_seeds),
			("background_seed", "background", background_seeds),
			("action_seed", "action", action_seeds),
		):
			value = sequence.get(field)
			if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < UINT32_MODULUS:
				raise ValueError(f"{sequence_id}.{field} is not a uint32 seed.")
			if value in seen:
				raise ValueError(f"{field} is reused: {value}.")
			if value % SEED_DOMAIN_SIZE != SEED_DOMAIN_TAGS[(split, seed_type)]:
				raise ValueError(f"{sequence_id}.{field} is outside its split/type domain.")
			seen.add(value)
	if actual_counts != expected_counts:
		raise ValueError("Sequence sources do not match source_counts.")
	if len(env_seeds | background_seeds | action_seeds) != 3 * expected_episode_count:
		raise ValueError("Environment/background/action seeds overlap or are reused.")
	_validate_relative_manifest_assets(path.parent, payload)
	return ValidatedRolloutManifest(
		payload=payload,
		root=path.parent,
		path=path,
		file_sha256=file_sha256(path),
	)


def _atomic_json(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
	with temporary.open("x", encoding="utf-8", newline="\n") as stream:
		json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
		stream.write("\n")
		stream.flush()
		os.fsync(stream.fileno())
	os.replace(temporary, path)


def _prepare_output(output, video_root):
	output = output.expanduser().resolve()
	video_root = video_root.expanduser().resolve()
	if output.exists():
		raise FileExistsError(f"Refusing to overwrite existing output: {output}")
	try:
		output.relative_to(video_root)
	except ValueError:
		pass
	else:
		raise ValueError("Rollout output must not be inside the external video root.")
	output.parent.mkdir(parents=True, exist_ok=True)
	staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.incomplete"
	if staging.exists():
		raise FileExistsError(f"Staging path already exists: {staging}")
	staging.mkdir()
	return output, staging


def main():
	args = parse_args()
	output, staging = _prepare_output(args.output, args.video_root)
	payload = collect(args, staging)
	_validate_relative_manifest_assets(staging, payload)
	_atomic_json(staging / "manifest.json", payload)
	load_and_validate_rollout_manifest(
		staging / "manifest.json", background_manifest_dir=args.manifest_dir
	)
	staging.replace(output)
	print(
		"VISUAL_SMALL_OBJECT_ROLLOUT_OK",
		json.dumps(
			{
				"output": str(output),
				"split": args.split,
				"sequences": len(payload["sequences"]),
				"frames": sum(len(item["frames"]) for item in payload["sequences"]),
				"transitions": sum(
					item["num_transitions"] for item in payload["sequences"]
				),
			},
			sort_keys=True,
		),
	)


if __name__ == "__main__":
	main()
