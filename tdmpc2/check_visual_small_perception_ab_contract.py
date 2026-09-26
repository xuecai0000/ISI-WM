"""Synthetic contract test for the Visual-Small perception A/B evaluator.

This check deliberately uses prediction caches.  It therefore exercises the
audit/cache validation, metrics, hard gates, selection policy, and artifacts
without importing DINO, Cutie, an environment, or a simulator.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
EVALUATOR_PATH = ROOT / "tools" / "evaluate_visual_small_perception_ab.py"
ROLES = ("base", "elbow", "control_tip", "goal")
SPLITS = ("train", "validation")
LABEL_TIMES = (0, 5, 10, 15)
CUTIE_OBJECTS = {
	"1": "proximal_link",
	"2": "distal_link",
	"3": "goal",
}


def _load_evaluator():
	spec = importlib.util.spec_from_file_location(
		"visual_small_perception_ab_evaluator", EVALUATOR_PATH
	)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not import evaluator from {EVALUATOR_PATH}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


EVALUATOR = _load_evaluator()


def _dump_json(path: Path, value):
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="\n") as stream:
		json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
		stream.write("\n")


def _raw_rgb_sha256(image: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _reference_points(episode: int, t: int) -> dict[str, list[float]]:
	# The labelled tip-goal distances are all distinct within each split, so a
	# correct implementation has twelve paired samples and a defined Spearman.
	label_index = min(t // 5, len(LABEL_TIMES) - 1)
	goal_x = 36.0 + 4.0 * episode + label_index
	return {
		"base": [31.5, 31.5],
		"elbow": [42.0, 31.5],
		"control_tip": [52.0, 31.5],
		"goal": [goal_x, 12.0],
	}


def _make_audit(root: Path) -> tuple[Path, dict]:
	frames_root = root / "frames"
	manifest_root = ROOT / "envs" / "background_manifests"
	manifest_by_split = {}
	for split in SPLITS:
		manifest_path = manifest_root / f"color_multi_{split}.json"
		manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
		manifest_by_split[split] = {
			"sources": list(manifest_payload["sources"]),
			"sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
		}
	sequences = []
	global_frame = 0
	for split_index, split in enumerate(SPLITS):
		for episode in range(3):
			sequence_id = f"{split}-episode-{episode}"
			source = manifest_by_split[split]["sources"][episode]
			frames = []
			for t in range(16):
				# Every decoded RGB tensor has a distinct fingerprint.
				value = global_frame
				image = np.empty((64, 64, 3), dtype=np.uint8)
				image[..., 0] = value
				image[..., 1] = (value * 3 + 17) % 256
				image[..., 2] = (value * 7 + 29) % 256
				image[0, 0] = (split_index, episode, t)
				relative_image = Path("frames") / split / f"episode_{episode}" / f"t_{t:02d}.png"
				image_path = root / relative_image
				image_path.parent.mkdir(parents=True, exist_ok=True)
				Image.fromarray(image, mode="RGB").save(image_path)
				required = t in LABEL_TIMES
				points = (
					_reference_points(episode, t)
					if required else {role: None for role in ROLES}
				)
				frames.append({
					"index": t,
					"t": t,
					"image": relative_image.as_posix(),
					"image_sha256": _raw_rgb_sha256(image),
					"source_frame_index": 100 + t,
					"environment_step": 100 + t,
					"annotation_required": required,
					"points": points,
					"cutie_mask": {
						"status": "not_required",
						"image": None,
						"encoding": "indexed_png_uint8",
						"objects": CUTIE_OBJECTS,
					},
				})
				global_frame += 1
			sequences.append({
				"sequence_id": sequence_id,
				"split": split,
				"episode": episode,
				"source": source,
				"start_frame": 100,
				"warmup_steps": 100,
				"manifest_sha256": manifest_by_split[split]["sha256"],
				"frames": frames,
			})
	payload = {
		"format": "visual_small_perception_audit_v1",
		"roles": list(ROLES),
		"collection": {
			"task": "reacher-visual-small",
			"splits": list(SPLITS),
			"no_test": True,
			"episodes_per_split": 3,
			"frames_per_episode": 16,
			"manual_label_times": list(LABEL_TIMES),
			"label_policy": "manual_rgb_only",
			"split_metadata": {
				split: {
					"selected_sources": manifest_by_split[split]["sources"][:3],
					"allowed_sources": manifest_by_split[split]["sources"],
				}
				for split in SPLITS
			},
		},
		"observation_geometry": {
			"stored_height": 64,
			"stored_width": 64,
			"resolution_status": "native64_agent_observation",
			"true_high_resolution": False,
			"upscaled_copy_saved": False,
		},
		"annotation_schema": {
			"points": {role: "manual [x, y] or null" for role in ROLES},
			"cutie_mask": {
				"required": False,
				"encoding": "indexed_png_uint8",
				"objects": CUTIE_OBJECTS,
			},
		},
		"sequences": sequences,
	}
	path = root / "annotations.completed.json"
	_dump_json(path, payload)
	return path, payload


def _shift_points(points: dict[str, list[float]], offset: float):
	return {
		role: [float(point[0] + offset), float(point[1])]
		for role, point in points.items()
	}


def _save_cutie_masks(
	root: Path,
	*,
	name: str,
	sequence_id: str,
	episode: int,
	t: int,
	offset_px: float,
) -> dict[str, str]:
	"""Write the three fixed OC object masks used by a cache frame."""
	points = _reference_points(episode, t)
	# The base is a fixed evaluator constant and is never shifted.  A bad cache
	# bends both links and moves the goal vertically, keeping every mask in frame.
	vertical_offset = int(round(offset_px))
	base = tuple(map(round, points["base"]))
	elbow = (
		int(round(points["elbow"][0])),
		int(round(points["elbow"][1])) + vertical_offset,
	)
	tip = (
		int(round(points["control_tip"][0])),
		int(round(points["control_tip"][1])) + vertical_offset,
	)
	goal_center = (
		int(round(points["goal"][0])),
		int(round(points["goal"][1])) + vertical_offset,
	)
	draw_specs = {
		"proximal_link": (base, elbow),
		"distal_link": (elbow, tip),
	}
	paths = {}
	for role, endpoints in draw_specs.items():
		mask = Image.new("L", (64, 64), 0)
		ImageDraw.Draw(mask).line(endpoints, fill=255, width=5)
		relative = (
			Path("prediction_masks") / name / sequence_id
			/ f"t_{t:02d}_{role}.png"
		)
		path = root / relative
		path.parent.mkdir(parents=True, exist_ok=True)
		mask.save(path)
		paths[role] = relative.as_posix()
	goal = Image.new("L", (64, 64), 0)
	draw = ImageDraw.Draw(goal)
	radius = 3
	draw.ellipse((
		goal_center[0] - radius,
		goal_center[1] - radius,
		goal_center[0] + radius,
		goal_center[1] + radius,
	), fill=255)
	relative = (
		Path("prediction_masks") / name / sequence_id / f"t_{t:02d}_goal.png"
	)
	path = root / relative
	path.parent.mkdir(parents=True, exist_ok=True)
	goal.save(path)
	paths["goal"] = relative.as_posix()
	return paths


def _make_cache(
	root: Path,
	*,
	audit_path: Path,
	audit_payload: dict,
	backend: str,
	offset_px: float,
	name: str,
) -> Path:
	sequences = []
	for sequence in audit_payload["sequences"]:
		frames = []
		for t in range(16):
			frame = {
				"index": t,
				"t": t,
				"runtime_ms": 2.0 if backend == "dino" else 4.0,
			}
			if backend == "dino":
				frame.update({
					"points": _shift_points(
						_reference_points(int(sequence["episode"]), t), offset_px
					),
					"confidence": {role: 0.9 for role in ROLES},
					"lost": {role: False for role in ROLES},
				})
			else:
				frame.update({
					"points_derived_from_masks": True,
					"masks": _save_cutie_masks(
						root,
						name=name,
						sequence_id=sequence["sequence_id"],
						episode=int(sequence["episode"]),
						t=t,
						offset_px=offset_px,
					),
					"confidence": {
						role: 0.9 for role in ("proximal_link", "distal_link", "goal")
					},
					"lost": {
						role: False for role in ("proximal_link", "distal_link", "goal")
					},
				})
			frames.append(frame)
		sequences.append({
			"sequence_id": sequence["sequence_id"],
			"split": sequence["split"],
			"episode": sequence["episode"],
			"source": sequence["source"],
			"start_frame": sequence["start_frame"],
			"frames": frames,
		})
	metadata = {
		"synthetic_contract": True,
		"resolution": {
			"source": "native64",
			"native_size": [64, 64],
			"processed_size": [448, 448],
			"true_high_resolution": False,
			"resize_note": "Synthetic native64 contract input.",
		},
	}
	if backend == "cutie":
		metadata["resolution"].update({
			"requested_tracker_size": [448, 448],
			"actual_tracker_size": [448, 448],
		})
	payload = {
		"format": "visual_small_perception_predictions_v1",
		"backend": backend,
		"audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
		"roles": list(ROLES),
		"metadata": metadata,
		"sequences": sequences,
	}
	path = root / f"{name}_{backend}.json"
	_dump_json(path, payload)
	return path


class VisualSmallPerceptionABContractTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._temporary = tempfile.TemporaryDirectory(prefix="perception_ab_contract_")
		cls.root = Path(cls._temporary.name)
		cls.audit_path, cls.audit_payload = _make_audit(cls.root)
		cls.good_dino_cache = _make_cache(
			cls.root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend="dino",
			offset_px=1.0,
			name="good",
		)
		cls.good_cutie_cache = _make_cache(
			cls.root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend="cutie",
			offset_px=0.1,
			name="good",
		)
		cls.bad_dino_cache = _make_cache(
			cls.root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend="dino",
			offset_px=12.0,
			name="bad",
		)
		cls.bad_cutie_cache = _make_cache(
			cls.root,
			audit_path=cls.audit_path,
			audit_payload=cls.audit_payload,
			backend="cutie",
			offset_px=12.0,
			name="bad",
		)

	@classmethod
	def tearDownClass(cls):
		cls._temporary.cleanup()

	def test_good_cutie_is_selected_and_all_artifacts_exist(self):
		# Explicitly exercise both cache loaders before the end-to-end call.
		audit = EVALUATOR.load_audit(self.audit_path)
		dino = EVALUATOR.load_prediction_cache(
			self.good_dino_cache, backend="dino", audit=audit
		)
		cutie = EVALUATOR.load_prediction_cache(
			self.good_cutie_cache, backend="cutie", audit=audit
		)
		self.assertEqual(dino["source"], "cache")
		self.assertEqual(cutie["source"], "cache")

		output_dir = self.root / "good_output"
		report = EVALUATOR.evaluate(
			annotations=self.audit_path,
			output_dir=output_dir,
			dino_cache=self.good_dino_cache,
			dino_live=None,
			cutie_cache=self.good_cutie_cache,
			cutie_live=None,
		)
		for backend in ("dino", "cutie"):
			for split in SPLITS:
				metrics = report["methods"][backend]["splits"][split]
				self.assertEqual(metrics["labeled_frames"], 12)
				self.assertEqual(metrics["tip_goal"]["paired_distances"], 12)
		self.assertEqual(report["definitions"]["confident_wrong_error_threshold_px"], 6.0)
		self.assertEqual(report["selection"]["status"], "cutie_selected")
		self.assertEqual(report["selection"]["selected_backend"], "cutie")
		self.assertEqual(report["selection"]["recommended_backend"], "cutie")
		self.assertTrue(report["selection"]["cutie_rl_eligible"])
		self.assertTrue(report["hard_gate_pass"])

		report_path = Path(report["outputs"]["json"])
		csv_path = Path(report["outputs"]["csv"])
		self.assertTrue(report_path.is_file())
		self.assertTrue(csv_path.is_file())
		self.assertGreater(csv_path.stat().st_size, 0)
		overlays = [Path(path) for path in report["outputs"]["overlays"]]
		self.assertEqual(len(overlays), 24)
		self.assertTrue(all(path.is_file() for path in overlays))

	def test_dino_only_mode_is_runnable_before_cutie_is_installed(self):
		report = EVALUATOR.evaluate(
			annotations=self.audit_path,
			output_dir=self.root / "dino_only_output",
			dino_cache=self.good_dino_cache,
			dino_live=None,
		)
		self.assertEqual(set(report["methods"]), {"dino"})
		self.assertEqual(report["selection"]["status"], "dino_only_ready")
		self.assertEqual(report["selection"]["selected_backend"], "dino")
		self.assertFalse(report["selection"]["cutie_rl_eligible"])
		self.assertTrue(report["hard_gate_pass"])

	def test_both_bad_backends_produce_perception_not_ready(self):
		report = EVALUATOR.evaluate(
			annotations=self.audit_path,
			output_dir=self.root / "bad_output",
			dino_cache=self.bad_dino_cache,
			dino_live=None,
			cutie_cache=self.bad_cutie_cache,
			cutie_live=None,
		)
		self.assertFalse(report["methods"]["dino"]["absolute_gate_pass"])
		self.assertFalse(report["methods"]["cutie"]["absolute_gate_pass"])
		self.assertEqual(report["selection"]["status"], "perception_not_ready")
		self.assertIsNone(report["selection"]["selected_backend"])
		self.assertIsNone(report["selection"]["recommended_backend"])
		self.assertFalse(report["selection"]["pass"])
		self.assertFalse(report["hard_gate_pass"])

	def test_cutie_that_passes_is_selected_even_when_dino_fails(self):
		report = EVALUATOR.evaluate(
			annotations=self.audit_path,
			output_dir=self.root / "mixed_output",
			dino_cache=self.bad_dino_cache,
			dino_live=None,
			cutie_cache=self.good_cutie_cache,
			cutie_live=None,
		)
		self.assertFalse(report["methods"]["dino"]["absolute_gate_pass"])
		self.assertTrue(report["methods"]["cutie"]["absolute_gate_pass"])
		self.assertEqual(report["selection"]["status"], "cutie_selected")
		self.assertEqual(report["selection"]["selected_backend"], "cutie")
		self.assertTrue(report["selection"]["cutie_rl_eligible"])

	def test_three_cutie_object_masks_produce_four_accurate_points(self):
		proximal = np.zeros((64, 64), dtype=bool)
		distal = np.zeros((64, 64), dtype=bool)
		goal = np.zeros((64, 64), dtype=bool)
		proximal[30:34, 32:43] = True
		distal[30:34, 42:53] = True
		yy, xx = np.ogrid[:64, :64]
		goal[(xx - 48) ** 2 + (yy - 15) ** 2 <= 3 ** 2] = True
		points = EVALUATOR.points_from_cutie_masks({
			"proximal_link": proximal,
			"distal_link": distal,
			"goal": goal,
		})
		expected = {
			"base": [31.5, 31.5],
			"elbow": [42.0, 31.5],
			"control_tip": [52.0, 31.5],
			"goal": [48.0, 15.0],
		}
		for role in ROLES:
			with self.subTest(role=role):
				self.assertIsNotNone(points[role])
				error = float(np.linalg.norm(
					np.asarray(points[role]) - np.asarray(expected[role])
				))
				self.assertLessEqual(error, 0.6)

	def test_cutie_cache_cannot_replace_masks_with_explicit_points(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		payload = json.loads(self.good_cutie_cache.read_text(encoding="utf-8"))
		for sequence in payload["sequences"]:
			for frame in sequence["frames"]:
				frame.pop("masks")
				frame.pop("points_derived_from_masks", None)
				frame["points"] = _reference_points(int(sequence["episode"]), frame["t"])
		path = self.root / "invalid_cutie_explicit_points_without_masks.json"
		_dump_json(path, payload)
		with self.assertRaises(EVALUATOR.ContractError):
			EVALUATOR.load_prediction_cache(path, backend="cutie", audit=audit)

	def test_cutie_cache_resolution_must_be_exactly_matched_448(self):
		audit = EVALUATOR.load_audit(self.audit_path)
		original = json.loads(self.good_cutie_cache.read_text(encoding="utf-8"))
		for field in ("requested_tracker_size", "actual_tracker_size", "processed_size"):
			with self.subTest(field=field):
				payload = copy.deepcopy(original)
				payload["metadata"]["resolution"][field] = [320, 320]
				path = self.root / f"invalid_cutie_resolution_{field}.json"
				_dump_json(path, payload)
				with self.assertRaises(EVALUATOR.ContractError):
					EVALUATOR.load_prediction_cache(path, backend="cutie", audit=audit)

	def test_confident_wrong_uses_strict_six_pixel_error_threshold(self):
		self.assertEqual(EVALUATOR.CONFIDENT_WRONG_ERROR_PX, 6.0)
		audit = EVALUATOR.load_audit(self.audit_path)
		predictions = EVALUATOR.load_prediction_cache(
			self.good_dino_cache, backend="dino", audit=audit
		)
		sequence_id = "train-episode-0"
		ground_truth = audit["sequences"][0]["frames"][0]["points"]["base"]

		exactly_six = copy.deepcopy(predictions)
		exactly_six["sequences"][sequence_id]["frames"][0]["points"]["base"] = [
			ground_truth[0] + 6.0,
			ground_truth[1],
		]
		metrics = EVALUATOR.compute_split_metrics(
			audit=audit,
			predictions=exactly_six,
			split="train",
			confidence_threshold=0.5,
		)
		self.assertEqual(metrics["roles"]["base"]["confident_wrong_count"], 0)

		over_six = copy.deepcopy(predictions)
		over_six["sequences"][sequence_id]["frames"][0]["points"]["base"] = [
			ground_truth[0] + 6.01,
			ground_truth[1],
		]
		metrics = EVALUATOR.compute_split_metrics(
			audit=audit,
			predictions=over_six,
			split="train",
			confidence_threshold=0.5,
		)
		self.assertEqual(metrics["roles"]["base"]["confident_wrong_count"], 1)

	def test_test_split_physics_and_escaping_paths_are_rejected(self):
		cases = {}

		test_split = copy.deepcopy(self.audit_payload)
		test_split["sequences"][0]["split"] = "test"
		cases["test split"] = test_split

		physics = copy.deepcopy(self.audit_payload)
		physics["physics"] = {"qpos": [0.0]}
		cases["physics"] = physics

		absolute = copy.deepcopy(self.audit_payload)
		absolute["sequences"][0]["frames"][0]["image"] = str(
			(self.root / absolute["sequences"][0]["frames"][0]["image"]).resolve()
		)
		cases["absolute image path"] = absolute

		escape = copy.deepcopy(self.audit_payload)
		escape["sequences"][0]["frames"][0]["image"] = "../outside.png"
		cases["parent path escape"] = escape

		for index, (name, payload) in enumerate(cases.items()):
			with self.subTest(name=name):
				path = self.root / f"rejected_{index}.json"
				_dump_json(path, payload)
				with self.assertRaises(EVALUATOR.ContractError):
					EVALUATOR.load_audit(path)


if __name__ == "__main__":
	unittest.main(verbosity=2)
