"""Collect a tiny, visually-only support set for TD-MPC2 FlatAnchor.

The script deliberately records RGB frames only. It never reads simulator state,
keypoints, contacts, rewards, or other privileged signals. The generated JSON is
an annotation template whose points must be filled manually.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dm_control import suite
from PIL import Image


ROLES = ("base", "elbow", "control_tip", "goal")


def image_sha256(image):
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--seed", type=int, default=314159)
	parser.add_argument("--frames", type=int, default=6)
	parser.add_argument("--force", action="store_true")
	return parser.parse_args()


def main():
	args = parse_args()
	output = args.output.expanduser().resolve()
	frames_dir = output / "support_frames"
	if output.exists() and any(output.iterdir()) and not args.force:
		raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
	frames_dir.mkdir(parents=True, exist_ok=True)

	env = suite.load(
		"reacher",
		"easy",
		task_kwargs={"random": args.seed},
		visualize_reward=False,
	)
	action_spec = env.action_spec()
	rng = np.random.default_rng(args.seed)
	records = []
	for index in range(args.frames):
		env.reset()
		# One frame per independent episode, after a different short random prefix.
		for _ in range(4 + 3 * index):
			action = rng.uniform(action_spec.minimum, action_spec.maximum).astype(
				action_spec.dtype
			)
			# Match TD-MPC2's DMControl action repeat of two.
			env.step(action)
			env.step(action)
		image = env.physics.render(height=64, width=64, camera_id=0)
		name = f"support_{index:02d}.png"
		Image.fromarray(image).save(frames_dir / name)
		records.append(
			{
				"index": index,
				"image": f"support_frames/{name}",
				"source": f"reacher-easy/manual-support-seed-{args.seed}",
				"image_sha256": image_sha256(image),
				"points": {role: None for role in ROLES},
			}
		)

	template = {
		"format": "few_shot_task_anchor_annotations_v1",
		"coordinate_convention": "[x, y] in the original 64x64 RGB frame",
		"roles": list(ROLES),
		"records": records,
	}
	with (output / "annotations.json").open("w", encoding="utf-8") as file:
		json.dump(template, file, indent=2, ensure_ascii=False)
	print(f"Saved {args.frames} RGB-only support frames to {output}")
	print("Next: manually fill all four [x, y] points in annotations.json.")


if __name__ == "__main__":
	main()
