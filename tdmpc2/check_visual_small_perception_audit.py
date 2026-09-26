"""Static and data-level gate for the Visual-Small perception audit pack."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT / "tools" / "collect_visual_small_perception_audit.py"
MANIFEST_DIR = ROOT / "envs" / "background_manifests"
FORMAT = "visual_small_perception_audit_v1"
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
FORBIDDEN_ATTRIBUTES = {
	"physics",
	"qpos",
	"qvel",
	"geom_xpos",
	"site_xpos",
	"body_xpos",
	"get_state",
}
FORBIDDEN_JSON_KEYS = FORBIDDEN_ATTRIBUTES | {
	"reward",
	"simulator_state",
	"ground_truth",
	"privileged_state",
}


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--audit",
		type=Path,
		help="annotations.json or the directory containing it",
	)
	parser.add_argument("--require-manual-labels", action="store_true")
	args = parser.parse_args()
	if args.require_manual_labels and args.audit is None:
		parser.error("--require-manual-labels requires --audit")
	return args


def check_collector_source():
	source = COLLECTOR.read_text(encoding="utf-8")
	tree = ast.parse(source)
	used_attributes = {
		node.attr.lower()
		for node in ast.walk(tree)
		if isinstance(node, ast.Attribute)
	}
	used_names = {
		node.id.lower()
		for node in ast.walk(tree)
		if isinstance(node, ast.Name)
	}
	forbidden = sorted((used_attributes | used_names).intersection(FORBIDDEN_ATTRIBUTES))
	assert not forbidden, f"Collector uses privileged simulator names: {forbidden}"
	assert "from dm_control" not in source
	assert ".render(" not in source
	assert ".close(" not in source
	assert "observation[-3:]" in source
	assert 'ALLOWED_SPLITS = ("train", "validation")' in source
	assert 'FORMAT = "visual_small_perception_audit_v1"' in source
	assert "env.step(action)[0]" in source
	return {
		"collector": COLLECTOR.name,
		"rgb_source": "newest wrapped observation",
		"allowed_splits": list(ALLOWED_SPLITS),
		"privileged_names": [],
	}


def load_manifests():
	result = {}
	all_sources = set()
	for split in ("train", "validation", "test", "support"):
		path = MANIFEST_DIR / f"color_multi_{split}.json"
		payload = json.loads(path.read_text(encoding="utf-8"))
		assert payload["name"] == split
		sources = tuple(payload["sources"])
		assert not all_sources.intersection(sources)
		all_sources.update(sources)
		result[split] = sources
	return result


def walk_json_keys(value):
	if isinstance(value, dict):
		for key, child in value.items():
			yield str(key).lower()
			yield from walk_json_keys(child)
	elif isinstance(value, list):
		for child in value:
			yield from walk_json_keys(child)


def raw_image_sha256(path):
	image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(), image


def check_point(point, label):
	assert isinstance(point, list) and len(point) == 2, f"{label} must be [x, y]"
	assert all(isinstance(value, (int, float)) for value in point), label
	assert all(math.isfinite(float(value)) for value in point), label
	assert all(0.0 <= float(value) <= 63.5 for value in point), label


def check_mask(root, mask_record, label):
	assert mask_record["encoding"] == "indexed_png_uint8"
	assert mask_record["objects"] == CUTIE_OBJECTS
	mask_value = mask_record["image"]
	if mask_value is None:
		assert mask_record["status"] == "not_required", label
		return
	assert mask_record["status"] == "optional_manual", label
	assert isinstance(mask_value, str) and mask_value, label
	mask_rel = Path(mask_value)
	assert not mask_rel.is_absolute() and ".." not in mask_rel.parts
	mask_path = (root / mask_rel).resolve()
	assert root in mask_path.parents and mask_path.is_file(), mask_path
	mask = np.asarray(Image.open(mask_path))
	assert mask.shape == (64, 64), f"{label}: mask shape {mask.shape}"
	assert set(np.unique(mask)).issubset({0, 1, 2, 3}), label


def check_audit(path, require_labels):
	path = Path(path).expanduser().resolve()
	if path.is_dir():
		path = path / "annotations.json"
	assert path.is_file(), path
	root = path.parent
	payload = json.loads(path.read_text(encoding="utf-8"))
	assert payload["format"] == FORMAT
	assert tuple(payload["roles"]) == ROLES
	assert not set(walk_json_keys(payload)).intersection(FORBIDDEN_JSON_KEYS)

	collection = payload["collection"]
	assert collection["task"] == "reacher-visual-small"
	assert tuple(collection["splits"]) == ALLOWED_SPLITS
	assert collection["no_test"] is True
	assert collection["episodes_per_split"] == EPISODES_PER_SPLIT
	assert collection["frames_per_episode"] == FRAMES_PER_EPISODE
	assert tuple(collection["manual_label_times"]) == MANUAL_LABEL_TIMES
	assert collection["label_policy"] == "manual_rgb_only"

	geometry = payload["observation_geometry"]
	assert geometry["stored_height"] == geometry["stored_width"] == 64
	assert geometry["resolution_status"] == "native64_agent_observation"
	assert geometry["true_high_resolution"] is False
	assert geometry["upscaled_copy_saved"] is False
	assert payload["annotation_schema"]["cutie_mask"]["objects"] == CUTIE_OBJECTS
	assert payload["annotation_schema"]["cutie_mask"]["required"] is False

	manifests = load_manifests()
	for forbidden_split in ("test", "support"):
		assert set(collection["split_metadata"].keys()).isdisjoint({forbidden_split})
	selected_by_split = {split: set() for split in ALLOWED_SPLITS}
	sequences_by_split = {split: [] for split in ALLOWED_SPLITS}
	seen_images = set()
	seen_hashes = set()
	for sequence in payload["sequences"]:
		split = sequence["split"]
		assert split in ALLOWED_SPLITS
		assert sequence["source"] in manifests[split]
		assert sequence["source"] not in manifests["test"]
		assert sequence["source"] not in manifests["support"]
		selected_by_split[split].add(sequence["source"])
		sequences_by_split[split].append(sequence)
		frames = sequence["frames"]
		assert len(frames) == FRAMES_PER_EPISODE
		assert sequence["start_frame"] == frames[0]["source_frame_index"]
		previous_source_frame = None
		for index, frame in enumerate(frames):
			assert frame["index"] == frame["t"] == index
			assert frame["environment_step"] == sequence["warmup_steps"] + index
			required = index in MANUAL_LABEL_TIMES
			assert frame["annotation_required"] is required
			if previous_source_frame is not None:
				current = frame["source_frame_index"]
				assert current == previous_source_frame + 1 or current == 0, (
					f"{sequence['sequence_id']}: non-consecutive source frames"
				)
			previous_source_frame = frame["source_frame_index"]

			image_rel = Path(frame["image"])
			assert not image_rel.is_absolute() and ".." not in image_rel.parts
			image_path = (root / image_rel).resolve()
			assert root in image_path.parents and image_path.is_file(), image_path
			assert str(image_path) not in seen_images
			seen_images.add(str(image_path))
			actual_sha, image = raw_image_sha256(image_path)
			assert image.shape == (64, 64, 3) and image.dtype == np.uint8
			assert frame["image_sha256"] == actual_sha
			assert actual_sha not in seen_hashes
			seen_hashes.add(actual_sha)

			assert tuple(frame["points"].keys()) == ROLES
			for role, point in frame["points"].items():
				if required and require_labels:
					check_point(point, f"{frame['image']}:{role}")
				else:
					assert point is None or required, frame["image"]
					if point is not None:
						check_point(point, f"{frame['image']}:{role}")
			check_mask(
				root,
				frame["cutie_mask"],
				frame["image"],
			)

	for split in ALLOWED_SPLITS:
		assert len(sequences_by_split[split]) == EPISODES_PER_SPLIT
		assert [sequence["episode"] for sequence in sequences_by_split[split]] == [0, 1, 2]
		assert len(selected_by_split[split]) == EPISODES_PER_SPLIT
		metadata = collection["split_metadata"][split]
		assert set(metadata["selected_sources"]) == selected_by_split[split]
		assert set(metadata["allowed_sources"]) == set(manifests[split])

	return {
		"annotations": str(path),
		"sequences": len(payload["sequences"]),
		"frames": len(seen_images),
		"manual_frames": len(payload["sequences"]) * len(MANUAL_LABEL_TIMES),
		"manual_labels_required": bool(require_labels),
		"selected_sources": {
			split: sorted(sources) for split, sources in selected_by_split.items()
		},
	}


def main():
	args = parse_args()
	result = {"static": check_collector_source()}
	if args.audit is not None:
		result["audit"] = check_audit(args.audit, args.require_manual_labels)
	print("VISUAL_SMALL_PERCEPTION_AUDIT_OK", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
	main()
