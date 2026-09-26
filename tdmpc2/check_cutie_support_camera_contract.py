"""CPU-only camera contracts using production code and a deterministic fake renderer.

No MuJoCo, video files, torch, CUDA, or existing support packs are needed. The
production collector is executed against a fake environment in a temporary
directory, including its saved-image/hash checks. This is not a real-render
overlay check; corrected real support packs still need that check before use.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterable
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from tdmpc2.common.support_camera_contract import (  # noqa: E402
	build_same_camera_contract,
	validate_same_camera_contract,
)
from tdmpc2.tools.validate_cutie_support_camera import (  # noqa: E402
	validate_support_pack,
)


COLLECTOR = ROOT / "tools" / "collect_cutie_multitask_support.py"
DMC = ROOT / "envs" / "dmcontrol.py"


def _execute_definitions(path, namespace, *, names, constants=()):
	tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
	selected = []
	for node in tree.body:
		if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
			selected.append(node)
		elif isinstance(node, ast.Assign) and any(
			isinstance(target, ast.Name) and target.id in constants
			for target in node.targets
		):
			selected.append(node)
	exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class _Box:
	def __init__(self, *, low, high, dtype):
		self.low, self.high = low, high
		self.dtype = np.dtype(dtype)


def _runtime_namespace():
	namespace = {
		"np": np,
		"gym": SimpleNamespace(spaces=SimpleNamespace(Box=_Box)),
		"defaultdict": defaultdict,
	}
	_execute_definitions(DMC, namespace, names={"get_obs_shape", "DMControlWrapper"})
	return namespace


class _Physics:
	def __init__(self):
		self.calls = []
		self.model = SimpleNamespace(
			ngeom=2, nsite=0, geom_bodyid=np.array([0, 1]),
			id2name=lambda index, kind: f"{kind}{index}",
		)

	def render(self, height, width, camera_id, *, segmentation=False):
		self.calls.append((camera_id, segmentation))
		# The visible objects move in image coordinates with the camera. A
		# camera-0 label cannot align with the camera-2 RGB by coincidence.
		x = 2 + int(camera_id) * 8
		regions = ((slice(8, 20), slice(x, x + 6)),
			(slice(30, 44), slice(x + 10, x + 16)))
		if segmentation:
			frame = np.full((height, width, 2), -1, dtype=np.int32)
			for object_id, region in enumerate(regions):
				frame[region] = (object_id, 5)
		else:
			frame = np.zeros((height, width, 3), dtype=np.uint8)
			frame[..., 2] = 64
			for color, region in zip(((255, 0, 0), (0, 255, 0)), regions):
				frame[region] = color
		return frame


def _make_runtime(domain):
	physics = _Physics()
	underlying = SimpleNamespace(
		physics=physics,
		observation_spec=lambda: {"position": SimpleNamespace(shape=(2,))},
		action_spec=lambda: SimpleNamespace(
			shape=(1,), minimum=-1.0, maximum=1.0, dtype=np.float32,
		),
	)
	wrapper_class = _runtime_namespace()["DMControlWrapper"]
	return wrapper_class(underlying, domain), wrapper_class


class _Background:
	active_split = "support"
	source_names = tuple(f"video{index}.mp4" for index in range(85, 90))
	manifest_sha256 = "fake-manifest"
	combined_manifest_sha256 = "fake-combined-manifest"

	def __init__(self, renderer):
		self.env = renderer
		self.action_space = renderer.action_space
		self.episode = -1
		self.closed = False

	def _observation(self):
		# Same call path/default camera as Pixels._get_obs, without torch.
		frame = self.env.render(width=64, height=64).transpose(2, 0, 1)
		return np.concatenate([frame, frame, frame])

	def reset(self):
		self.episode += 1
		self.active_source = self.source_names[self.episode % len(self.source_names)]
		self.frame_index = 0
		return self._observation()

	def step(self, _action):
		self.frame_index += 1
		return self._observation(), 0.0, False, {}

	def close(self):
		self.closed = True


def _collector_namespace(environment, wrapper_class):
	namespace = {
		"__name__": __name__, "__file__": str(COLLECTOR),
		"np": np, "Image": Image, "Path": Path, "hashlib": hashlib,
		"json": json, "re": re, "SimpleNamespace": SimpleNamespace,
		"Iterable": Iterable, "dataclass": dataclass,
		"enums": SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_GEOM=5, mjOBJ_SITE=6)),
		"build_same_camera_contract": build_same_camera_contract,
		"validate_same_camera_contract": validate_same_camera_contract,
		"DMControlWrapper": wrapper_class,
		"ColorMultiVideoBackgroundWrapper": _Background,
		"make_env": lambda _config: environment,
	}
	tree = ast.parse(COLLECTOR.read_text(encoding="utf-8"))
	names = {
		node.name for node in tree.body
		if isinstance(node, (ast.FunctionDef, ast.ClassDef))
		and node.name not in {"main", "_parse_args"}
	}
	_execute_definitions(COLLECTOR, namespace, names=names, constants={
		"FORMAT", "SUPPORT_SCHEMA", "SPLIT", "SUPPORT_EPISODES",
		"SUPPORT_VIDEOS", "IMAGE_SIZE",
	})
	return namespace


class CutieSupportCameraContract(unittest.TestCase):
	def test_runtime_defaults_preserve_quadruped_and_other_views(self):
		for domain, expected in (("quadruped", 2), ("finger", 0), ("ball_in_cup", 0)):
			with self.subTest(domain=domain):
				runtime, _ = _make_runtime(domain)
				runtime.render()
				self.assertEqual(runtime.camera_id, expected)
				self.assertEqual(runtime.env.physics.calls, [(expected, False)])

	def test_explicit_zero_overrides_quadruped_default(self):
		runtime, _ = _make_runtime("quadruped")
		runtime.render(camera_id=0)
		runtime.render(camera_id=None)
		self.assertEqual(runtime.env.physics.calls, [(0, False), (2, False)])

	def _assert_collected_camera(self, domain, camera_override=None):
		runtime, wrapper_class = _make_runtime(domain)
		if camera_override is not None:
			runtime.camera_id = camera_override
		expected_camera = runtime.camera_id
		background = _Background(runtime)
		# Exercise discovery through an additional wrapper, as in make_env.
		environment = SimpleNamespace(
			env=background, action_space=background.action_space,
			reset=background.reset, step=background.step, close=background.close,
		)
		namespace = _collector_namespace(environment, wrapper_class)
		spec = namespace["TaskSpec"](
			task=f"{domain}-camera-contract", roles=("first", "second"),
			selectors=(namespace["RoleSelector"](geom=("geom0",)),
				namespace["RoleSelector"](geom=("geom1",))),
		)
		with tempfile.TemporaryDirectory(prefix="cutie_camera_contract_") as raw:
			root = Path(raw)
			args = SimpleNamespace(
				seed=314159, video_root=root, manifest_dir=None, max_reset_attempts=6,
			)
			namespace["_collect_task"](args, root, spec)
			annotations = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
			catalog = json.loads((root / "geom_catalog.json").read_text(encoding="utf-8"))
			self.assertEqual(annotations["collection"]["camera_id"], expected_camera)
			self.assertEqual(catalog["camera_id"], expected_camera)
			expected_contract = build_same_camera_contract(expected_camera)
			self.assertEqual(
				annotations["collection"]["camera_contract"], expected_contract
			)
			self.assertEqual(catalog["camera_contract"], expected_contract)
			self.assertEqual(len(annotations["records"]), 6)
			for record in annotations["records"]:
				with Image.open(root / record["image"]) as image_file:
					image = np.array(image_file)
				with Image.open(root / record["indexed_mask"]) as mask_file:
					mask = np.array(mask_file)
				self.assertTrue(np.all(image[mask == 1] == (255, 0, 0)))
				self.assertTrue(np.all(image[mask == 2] == (0, 255, 0)))
				self.assertEqual(record["random_prefix_steps"], 4 + 3 * record["index"])
			summary = validate_support_pack(
				root,
				expected_task=spec.task,
				expected_camera_id=expected_camera,
			)
			self.assertTrue(summary["explicit_same_camera_contract"])
			self.assertFalse(summary["runtime_segmentation_allowed"])

			# A declaration that reintroduces camera-0 masks fails independently
			# of otherwise valid image/mask hashes.
			annotations["collection"]["camera_contract"]["mask_camera_id"] = (
				expected_camera + 1
			)
			(root / "annotations.json").write_text(
				json.dumps(annotations), encoding="utf-8"
			)
			with self.assertRaisesRegex(ValueError, "same simulator state and camera"):
				validate_support_pack(
					root,
					expected_task=spec.task,
					expected_camera_id=expected_camera,
				)
		self.assertTrue(background.closed)
		self.assertEqual({camera for camera, _ in runtime.env.physics.calls}, {expected_camera})
		self.assertEqual(sum(seg for _, seg in runtime.env.physics.calls), 6)

	def test_quadruped_labels_and_metadata_share_rgb_camera_two(self):
		self._assert_collected_camera("quadruped")

	def test_nonquadruped_labels_and_metadata_keep_camera_zero(self):
		self._assert_collected_camera("finger")

	def test_collector_derives_runtime_view_instead_of_task_lookup(self):
		self._assert_collected_camera("quadruped", camera_override=1)


if __name__ == "__main__":
	unittest.main(verbosity=2)
