"""Collect leakage-safe RGB-only support for Visual Small Color-multi.

Exactly one RGB frame is saved from each of six independently reset episodes.
The background is restricted to the immutable ``support`` split (video85-89).
Only rendered observations and public background provenance are recorded; the
collector never reads simulator state, physics coordinates, or rewards as
labels. All four role points must be annotated manually afterwards.
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


TASK = "reacher-visual-small"
SPLIT = "support"
SUPPORT_EPISODES = 6
SUPPORT_VIDEOS = tuple(f"video{index}.mp4" for index in range(85, 90))
ROLES = ("base", "elbow", "control_tip", "goal")


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def image_sha256(image):
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def parse_args():
	parser = argparse.ArgumentParser(
		description="Collect six manual RGB support frames from video85-89 only."
	)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--video-root", type=Path, required=True)
	parser.add_argument("--manifest-dir", type=Path, default=None)
	parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
	parser.add_argument("--seed", type=int, default=314159)
	parser.add_argument("--force", action="store_true")
	return parser.parse_args()


def make_collector_config(args):
	return Config(
		task=TASK,
		obs="rgb",
		seed=int(args.seed),
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
			f"Expected newest uint8 RGB frame with shape (64, 64, 3), "
			f"got shape={frame.shape}, dtype={frame.dtype}."
		)
	return frame


def main():
	args = parse_args()
	if args.split != SPLIT:
		raise ValueError(f"This collector only permits split={SPLIT!r}.")

	output = args.output.expanduser().resolve()
	if output.exists() and not output.is_dir():
		raise NotADirectoryError(f"Output path is not a directory: {output}")
	if output.exists() and any(output.iterdir()) and not args.force:
		raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
	frames_dir = output / "support_frames"
	frames_dir.mkdir(parents=True, exist_ok=True)

	env = make_env(make_collector_config(args))
	background = find_background_wrapper(env)
	if background.active_split != SPLIT:
		raise RuntimeError(
			f"Background split mismatch: {background.active_split!r} != {SPLIT!r}."
		)
	if tuple(background.source_names) != SUPPORT_VIDEOS:
		raise RuntimeError(
			f"Support manifest must contain exactly {SUPPORT_VIDEOS}, "
			f"got {tuple(background.source_names)}."
		)

	manifest_sha256 = background.manifest_sha256
	combined_manifest_sha256 = background.combined_manifest_sha256
	rng = np.random.default_rng(args.seed)
	records = []
	covered_videos = set()
	reset_attempts = 0
	max_reset_attempts = 1000
	while len(records) < SUPPORT_EPISODES:
		if reset_attempts >= max_reset_attempts:
			raise RuntimeError(
				f"Could not cover all support videos after {max_reset_attempts} resets; "
				f"covered {sorted(covered_videos)}."
			)
		observation = env.reset()
		reset_attempts += 1
		active_video = Path(background.active_source).name
		if active_video not in SUPPORT_VIDEOS:
			raise RuntimeError(f"Out-of-split support video selected: {active_video!r}.")
		# Until all five sources are represented, reject duplicate-source resets.
		# The sixth saved frame may use any support source.
		if len(covered_videos) < len(SUPPORT_VIDEOS) and active_video in covered_videos:
			continue

		index = len(records)
		prefix_steps = 4 + 3 * index
		for _ in range(prefix_steps):
			action = rng.uniform(env.action_space.low, env.action_space.high).astype(
				env.action_space.dtype
			)
			observation = env.step(action)[0]

		image = latest_rgb(observation)
		if Path(background.active_source).name != active_video:
			raise RuntimeError("Active support video changed within one episode.")
		frame_index = background.frame_index
		if not isinstance(frame_index, int) or frame_index < 0:
			raise RuntimeError(f"Invalid active video frame index: {frame_index!r}.")
		covered_videos.add(active_video)

		name = f"support_{index:02d}.png"
		Image.fromarray(image).save(frames_dir / name)
		records.append(
			{
				"index": index,
				"episode": index,
				"image": f"support_frames/{name}",
				"source": active_video,
				"image_sha256": image_sha256(image),
				"video_split": SPLIT,
				"active_video": active_video,
				"frame_index": frame_index,
				"manifest_sha256": manifest_sha256,
				"random_prefix_steps": prefix_steps,
				"selection_reset_attempt": reset_attempts,
				"points": {role: None for role in ROLES},
			}
		)
	if covered_videos != set(SUPPORT_VIDEOS):
		raise RuntimeError(
			f"Saved support set did not cover every allowed video: {covered_videos}."
		)

	template = {
		"format": "few_shot_task_anchor_annotations_v1",
		"coordinate_convention": "[x, y] in the original 64x64 RGB frame",
		"roles": list(ROLES),
		"collection": {
			"task": TASK,
			"observation": "rgb",
			"split": SPLIT,
			"seed": int(args.seed),
			"episodes": SUPPORT_EPISODES,
			"allowed_videos": list(SUPPORT_VIDEOS),
			"covered_videos": sorted(covered_videos),
			"selection_reset_attempts": reset_attempts,
			"manifest_sha256": manifest_sha256,
			"combined_manifest_sha256": combined_manifest_sha256,
			"label_policy": "manual_rgb_only",
		},
		"records": records,
	}
	with (output / "annotations.json").open("w", encoding="utf-8") as file:
		json.dump(template, file, indent=2, ensure_ascii=False)
		file.write("\n")

	print(f"Saved {SUPPORT_EPISODES} RGB-only support frames to {output}")
	print(f"Background split: {SPLIT} ({', '.join(SUPPORT_VIDEOS)})")
	print("Next: manually fill all four [x, y] points in annotations.json.")


if __name__ == "__main__":
	main()
