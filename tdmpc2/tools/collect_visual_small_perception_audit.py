"""Collect a leakage-safe Visual-Small Color-multi perception audit pack.

The collector saves only the newest 64x64 RGB observation returned by the
normal TD-MPC2 wrapper and public video-background provenance. Labels are left
empty for manual annotation. Train and validation are collected; held-out
evaluation sources are never accepted by this tool.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from envs.dmcontrol import make_env  # noqa: E402
from envs.wrappers.video_background import (  # noqa: E402
	ColorMultiVideoBackgroundWrapper,
)


FORMAT = "visual_small_perception_audit_v1"
TASK = "reacher-visual-small"
ALLOWED_SPLITS = ("train", "validation")
ROLES = ("base", "elbow", "control_tip", "goal")
CUTIE_OBJECTS = {
	"1": "proximal_link",
	"2": "distal_link",
	"3": "goal",
}
EPISODES_PER_SPLIT = 3
FRAMES_PER_EPISODE = 16
MANUAL_LABEL_TIMES = (0, 5, 10, 15)


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Collect three 16-frame episodes from each of train and validation "
			"for manual perception auditing."
		)
	)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path, default=None)
	parser.add_argument("--seed", type=int, default=271828)
	parser.add_argument("--force", action="store_true")
	return parser.parse_args()


def make_collector_config(args, split, split_seed):
	if split not in ALLOWED_SPLITS:
		raise ValueError(f"Perception audit split is forbidden: {split!r}.")
	return Config(
		task=TASK,
		obs="rgb",
		seed=int(split_seed),
		video_background_enabled=True,
		video_background_root=str(args.video_root.expanduser().resolve()),
		video_background_manifest_dir=(
			None
			if args.manifest_dir is None
			else str(args.manifest_dir.expanduser().resolve())
		),
		video_background_split=split,
		video_background_strength=1.0,
		video_background_total_frames=1000,
		video_background_source_cache_size=8,
		flat_anchor=False,
	)


def find_background_wrapper(env):
	current = env
	while current is not None:
		if isinstance(current, ColorMultiVideoBackgroundWrapper):
			return current
		current = getattr(current, "env", None)
	raise RuntimeError("ColorMultiVideoBackgroundWrapper is missing from the RGB path.")


def latest_rgb(observation):
	frame = observation[-3:].detach().cpu().permute(1, 2, 0).contiguous().numpy()
	if frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
		raise ValueError(
			"Expected the newest uint8 64x64 RGB observation; "
			f"got shape={frame.shape}, dtype={frame.dtype}."
		)
	return frame


def image_sha256(image):
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def annotation_placeholder(required):
	return {
		"annotation_required": bool(required),
		"points": {role: None for role in ROLES},
		"cutie_mask": {
			"status": "not_required",
			"image": None,
			"encoding": "indexed_png_uint8",
			"objects": dict(CUTIE_OBJECTS),
		},
	}


def collect_split(args, output, split, split_seed):
	env = make_env(make_collector_config(args, split, split_seed))
	background = find_background_wrapper(env)
	if background.active_split != split:
		raise RuntimeError(
			f"Background split mismatch: {background.active_split!r} != {split!r}."
		)
	allowed_sources = tuple(background.source_names)
	rng = np.random.default_rng(split_seed)
	sequences = []
	used_sources = set()
	reset_attempts = 0
	max_reset_attempts = 2000

	while len(sequences) < EPISODES_PER_SPLIT:
		if reset_attempts >= max_reset_attempts:
			raise RuntimeError(
				f"Could not select {EPISODES_PER_SPLIT} distinct {split} videos "
				f"after {max_reset_attempts} resets."
			)
		observation = env.reset()
		reset_attempts += 1
		source = Path(background.active_source).name
		if source not in allowed_sources:
			raise RuntimeError(f"Out-of-split source selected: {source!r}.")
		if source in used_sources:
			continue

		episode = len(sequences)
		warmup_steps = 4 + 3 * episode
		for _ in range(warmup_steps):
			action = rng.uniform(
				env.action_space.low, env.action_space.high
			).astype(env.action_space.dtype)
			observation = env.step(action)[0]

		sequence_id = f"{split}_episode_{episode:02d}"
		sequence_dir = output / "frames" / split / sequence_id
		sequence_dir.mkdir(parents=True, exist_ok=True)
		frames = []
		for t in range(FRAMES_PER_EPISODE):
			if t > 0:
				action = rng.uniform(
					env.action_space.low, env.action_space.high
				).astype(env.action_space.dtype)
				observation = env.step(action)[0]
			if Path(background.active_source).name != source:
				raise RuntimeError("Background source changed within one episode.")
			frame_index = background.frame_index
			if not isinstance(frame_index, int) or frame_index < 0:
				raise RuntimeError(
					f"Invalid source frame index: {frame_index!r}."
				)
			image = latest_rgb(observation)
			name = f"frame_{t:02d}.png"
			relative_image = Path("frames") / split / sequence_id / name
			Image.fromarray(image).save(output / relative_image)
			frame_record = {
				"index": t,
				"t": t,
				"image": relative_image.as_posix(),
				"image_sha256": image_sha256(image),
				"source_frame_index": frame_index,
				"environment_step": warmup_steps + t,
			}
			frame_record.update(annotation_placeholder(t in MANUAL_LABEL_TIMES))
			frames.append(frame_record)

		sequences.append(
			{
				"split": split,
				"episode": episode,
				"sequence_id": sequence_id,
				"source": source,
				"start_frame": frames[0]["source_frame_index"],
				"warmup_steps": warmup_steps,
				"selection_reset_attempt": reset_attempts,
				"manifest_sha256": background.manifest_sha256,
				"frames": frames,
			}
		)
		used_sources.add(source)

	return {
		"sequences": sequences,
		"allowed_sources": list(allowed_sources),
		"selected_sources": sorted(used_sources),
		"manifest_sha256": background.manifest_sha256,
		"combined_manifest_sha256": background.combined_manifest_sha256,
		"reset_attempts": reset_attempts,
	}


def main():
	args = parse_args()
	output = args.output.expanduser().resolve()
	if output.exists() and not output.is_dir():
		raise NotADirectoryError(f"Output path is not a directory: {output}")
	if output.exists() and any(output.iterdir()) and not args.force:
		raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
	output.mkdir(parents=True, exist_ok=True)

	all_sequences = []
	split_metadata = {}
	combined_hashes = set()
	for split_index, split in enumerate(ALLOWED_SPLITS):
		split_seed = int(args.seed) + 100003 * split_index
		result = collect_split(args, output, split, split_seed)
		all_sequences.extend(result.pop("sequences"))
		combined_hashes.add(result.pop("combined_manifest_sha256"))
		split_metadata[split] = result
	if len(combined_hashes) != 1:
		raise RuntimeError("Train and validation used different manifest collections.")

	payload = {
		"format": FORMAT,
		"roles": list(ROLES),
		"collection": {
			"task": TASK,
			"splits": list(ALLOWED_SPLITS),
			"no_test": True,
			"seed": int(args.seed),
			"episodes_per_split": EPISODES_PER_SPLIT,
			"frames_per_episode": FRAMES_PER_EPISODE,
			"manual_label_times": list(MANUAL_LABEL_TIMES),
			"action_repeat": 2,
			"label_policy": "manual_rgb_only",
			"combined_manifest_sha256": next(iter(combined_hashes)),
			"split_metadata": split_metadata,
		},
		"observation_geometry": {
			"stored_height": 64,
			"stored_width": 64,
			"coordinate_convention": "[x, y] in the stored 64x64 RGB image",
			"resolution_status": "native64_agent_observation",
			"true_high_resolution": False,
			"upscaled_copy_saved": False,
			"display_upscale_policy": (
				"Display-only interpolation is not a new observation and must not be "
				"reported as high-resolution evidence."
			),
		},
		"annotation_schema": {
			"required_times": list(MANUAL_LABEL_TIMES),
			"points": {role: "manual [x, y] or null" for role in ROLES},
			"cutie_mask": {
				"required": False,
				"encoding": "indexed_png_uint8",
				"background_id": 0,
				"objects": dict(CUTIE_OBJECTS),
				"rule": (
					"Audit masks are optional. Cutie prompts are generated separately "
					"from the existing four-point support pack: proximal link 1, "
					"distal link 2, goal 3."
				),
			},
		},
		"sequences": all_sequences,
	}
	with (output / "annotations.json").open("w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2, ensure_ascii=False)
		file.write("\n")

	print(
		f"Saved {len(all_sequences)} episodes and "
		f"{len(all_sequences) * FRAMES_PER_EPISODE} native 64x64 frames to {output}"
	)
	print("Splits: train, validation. Held-out evaluation sources were not used.")
	print(
		"Next: manually annotate the four points only at "
		f"t={','.join(str(value) for value in MANUAL_LABEL_TIMES)}."
	)


if __name__ == "__main__":
	main()
