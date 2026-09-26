"""Dependency-light contract checks for Visual-Small object-feature validation.

The fixtures in this file are temporary, synthetic, and intentionally tiny.
They exercise the frozen rollout/extractor/evaluator contracts without loading
dm_control, Cutie, a checkpoint, a GPU, or any external video.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
	sys.path.insert(0, str(REPOSITORY_ROOT))

from tdmpc2.tools import collect_visual_small_object_rollouts as collector
from tdmpc2.tools import extract_visual_small_cutie_object_features as extractor
from tdmpc2.tools import evaluate_visual_small_object_feature_probes as evaluator


def _write_json(path: Path, payload: dict) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	serialized = (
		json.dumps(
			payload,
			indent=2,
			sort_keys=True,
			ensure_ascii=False,
			allow_nan=False,
		)
		+ "\n"
	)
	with path.open("w", encoding="utf-8", newline="\n") as stream:
		stream.write(serialized)
	return path


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _copy_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
	return {name: np.array(value, copy=True) for name, value in arrays.items()}


def _disk(mask: np.ndarray, x: int, y: int, radius: int) -> None:
	y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
	x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
	yy, xx = np.ogrid[y0:y1, x0:x1]
	mask[y0:y1, x0:x1] |= (
		(xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius
	)


def _arm_mask(x: int, y: int) -> np.ndarray:
	mask = np.zeros((64, 64), dtype=bool)
	distance = max(abs(x - 32), abs(y - 32), 1)
	for alpha in np.linspace(0.0, 1.0, distance + 1):
		px = int(round(32 + alpha * (x - 32)))
		py = int(round(32 + alpha * (y - 32)))
		_disk(mask, px, py, 1)
	_disk(mask, x, y, 2)
	return mask


def _goal_mask(x: int, y: int) -> np.ndarray:
	mask = np.zeros((64, 64), dtype=bool)
	_disk(mask, x, y, 3)
	return mask


def _centroid(mask: np.ndarray) -> np.ndarray:
	yx = np.argwhere(mask)
	return np.asarray([yx[:, 1].mean(), yx[:, 0].mean()], dtype=np.float32)


def _synthetic_feature_arrays(
	states: list[tuple[int, int]], goal: tuple[int, int]
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
	frames = len(states)
	features = np.zeros((frames, 2, 2048), dtype=np.float32)
	masks = np.zeros((frames, 2, 64, 64), dtype=np.uint8)
	centroid = np.empty((frames, 2, 2), dtype=np.float32)
	confidence = np.full((frames, 2), 0.95, dtype=np.float32)
	lost = np.zeros((frames, 2), dtype=np.bool_)
	valid = np.ones((frames, 2), dtype=np.bool_)
	mask_score = np.full((frames, 2), 0.90, dtype=np.float32)
	runtime = np.full((frames,), 0.125, dtype=np.float32)
	union_masks: list[np.ndarray] = []
	for t, (x, y) in enumerate(states):
		role_masks = (_arm_mask(x, y), _goal_mask(*goal))
		union_masks.append(role_masks[0] | role_masks[1])
		for role, (role_x, role_y) in enumerate(((x, y), goal)):
			masks[t, role] = role_masks[role].astype(np.uint8)
			centroid[t, role] = _centroid(role_masks[role])
			slots = features[t, role].reshape(8, 256)
			slot_coordinate = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
			slots[:, 0] = np.float32(role_x / 63.0)
			slots[:, 1] = np.float32(role_y / 63.0)
			slots[:, 2] = np.float32(goal[0] / 63.0)
			slots[:, 3] = np.float32(goal[1] / 63.0)
			slots[:, 4] = np.float32(role)
			slots[:, 5] = slot_coordinate
			slots[:, 6] = slot_coordinate * np.float32((role_x + 1) / 64.0)
			slots[:, 7] = slot_coordinate * np.float32((role_y + 1) / 64.0)
	return (
		{
			"features": features,
			"masks": masks,
			"centroid_xy": centroid,
			"confidence": confidence,
			"lost": lost,
			"valid": valid,
			"mask_score": mask_score,
			"runtime_ms": runtime,
		},
		union_masks,
	)


def _synthetic_rgb(uid: int, object_union: np.ndarray) -> np.ndarray:
	yy, xx = np.indices((64, 64), dtype=np.uint16)
	image = np.empty((64, 64, 3), dtype=np.uint8)
	image[..., 0] = ((3 * xx + yy) % 251).astype(np.uint8)
	image[..., 1] = ((xx + 5 * yy + 17) % 253).astype(np.uint8)
	image[..., 2] = ((7 * xx + 2 * yy + 31) % 255).astype(np.uint8)
	image[object_union] = np.asarray(
		[(37 * uid + 11) % 256, (71 * uid + 23) % 256, (13 * uid + 47) % 256],
		dtype=np.uint8,
	)
	# The anchor is always in the arm mask.  Encoding the fixture ID here makes
	# decoded-image hashes unique without leaking it to background-only features.
	image[32, 32] = np.asarray(
		[uid % 256, (uid // 256) % 256, (uid // 65536) % 256], dtype=np.uint8
	)
	return image


class SyntheticPacks:
	"""Build exact train/validation rollout and feature manifests in temp space."""

	def __init__(self, root: Path, transitions: int = 2):
		self.root = root
		self.transitions = int(transitions)
		self.rollout_paths: dict[str, Path] = {}
		self.feature_paths: dict[str, Path] = {}
		self.rollout_payloads: dict[str, dict] = {}
		self.feature_payloads: dict[str, dict] = {}
		self._fingerprints, self._combined_fingerprint = (
			collector._background_manifest_fingerprints()
		)
		self._perception_provenance = {
			"algorithm": "visual_small_cutie_object_features_v1",
			"object_schema": "whole_arm_goal_v1",
			"roles": ["whole_arm", "goal"],
			"native_input_size": [64, 64],
			"tracker_size": [448, 448],
			"foreground_queries": 8,
			"feature_dim": 2048,
			"query_order": "official_query_post_process_cache_first_8_flattened",
			"model_size": "small",
			"amp": True,
			"prompt_radius": 2.0,
			"checkpoint_file_sha256": "a" * 64,
			"support_annotations_file_sha256": "b" * 64,
			"support_manifest_sha256": "c" * 64,
			"combined_manifest_sha256": "d" * 64,
			"config_tree_sha256": "e" * 64,
			"cutie_code_tree_sha256": "f" * 64,
			"adapter_file_sha256": "1" * 64,
			"extractor_file_sha256": "2" * 64,
		}
		uid = 1
		for split in collector.ALLOWED_SPLITS:
			uid = self._build_split(split, uid)

	def _build_split(self, split: str, uid: int) -> int:
		rollout_root = self.root / f"{split}_rollout"
		feature_root = self.root / f"{split}_features"
		rollout_root.mkdir(parents=True)
		feature_root.mkdir(parents=True)
		allowed = tuple(collector.EXPECTED_SOURCES[split])
		sequences: list[dict] = []
		feature_inputs: list[tuple[dict, dict[str, np.ndarray]]] = []
		for episode, source in enumerate(allowed):
			sequence_id = f"{split}_episode_{episode:06d}"
			phase = episode * 0.619 + (0.31 if split == "validation" else 0.0)
			actions = np.asarray(
				[
					[
						np.sin(phase + 1.13 * t),
						np.cos(0.73 * phase + 1.41 * t),
					]
					for t in range(self.transitions)
				],
				dtype=np.float32,
			)
			x = 8 + (episode * 11) % 47
			y = 8 + (episode * 17) % 47
			states = [(x, y)]
			for action in actions:
				x = int(np.clip(round(x + 7.0 * float(action[0])), 4, 59))
				y = int(np.clip(round(y + 7.0 * float(action[1])), 4, 59))
				states.append((x, y))
			goal = (9 + (episode * 19) % 46, 9 + (episode * 23) % 46)
			feature_arrays, unions = _synthetic_feature_arrays(states, goal)
			rewards = np.asarray(
				[
					1.0
					+ 0.5 * states[t][0] / 63.0
					+ 0.4 * float(actions[t, 0])
					- 0.3 * float(actions[t, 1])
					for t in range(self.transitions)
				],
				dtype=np.float32,
			)
			terminated = np.zeros((self.transitions,), dtype=np.bool_)
			truncated = np.zeros((self.transitions,), dtype=np.bool_)
			sequence_root = rollout_root / "sequences" / sequence_id
			sequence_root.mkdir(parents=True)
			frames = []
			for t, union in enumerate(unions):
				image = _synthetic_rgb(uid, union)
				uid += 1
				image_path = sequence_root / f"frame_{t:04d}.png"
				Image.fromarray(image, mode="RGB").save(image_path)
				frames.append(
					{
						"t": t,
						"image": image_path.relative_to(rollout_root).as_posix(),
						"image_sha256": collector.decoded_rgb_sha256(image),
						"source_frame_index": t,
					}
				)
			transition_path = sequence_root / "transitions.npz"
			np.savez(
				transition_path,
				actions=actions,
				rewards=rewards,
				terminated=terminated,
				truncated=truncated,
			)
			sequence = {
				"sequence_id": sequence_id,
				"split": split,
				"episode": episode,
				"source": source,
				"env_seed": collector._domain_seed(101, split, "env", episode),
				"background_seed": collector._domain_seed(
					202, split, "background", episode
				),
				"action_seed": collector._domain_seed(
					303, split, "action", episode
				),
				"num_transitions": self.transitions,
				"frames": frames,
				"transitions_asset": {
					"path": transition_path.relative_to(rollout_root).as_posix(),
					"file_sha256": _sha256(transition_path),
				},
				"source_selection_reset_attempt": 1,
				"ended_by_environment": False,
			}
			sequences.append(sequence)
			feature_inputs.append((sequence, feature_arrays))
		collection = {
			"task": collector.TASK,
			"split": split,
			"observation": "newest_wrapped_rgb",
			"native64": True,
			"resolution_status": "native64_agent_observation",
			"true_high_resolution": False,
			"upscaled_copy_saved": False,
			"no_test": True,
			"no_support": True,
			"action_repeat": 2,
			"action_policy": "independent_per_episode_uniform_action_space_v1",
			"seed": 271828,
			"manifest_sha256": self._fingerprints[split],
			"combined_manifest_sha256": self._combined_fingerprint,
			"num_episodes": len(allowed),
			"episodes_per_source": 1,
			"num_transitions_per_episode": self.transitions,
			"action_dim": 2,
			"action_dtype": "float32",
			"reward_dtype": "float32",
			"image_sha256_contract": (
				"sha256(contiguous_decoded_uint8_hwc_rgb_bytes)"
			),
			"transition_alignment": "obs[t] -- action[t], reward[t] --> obs[t+1]",
			"allowed_sources": list(allowed),
			"selected_sources": list(allowed),
			"source_counts": {source: 1 for source in allowed},
			"source_selection": "least_used_rejection_v1",
			"env_seed_assignment": "fixed_episode_ordinal_disjoint_domain_v1",
			"background_seed_assignment": "source_rejection_disjoint_domain_v1",
			"action_seed_assignment": "fixed_episode_ordinal_disjoint_domain_v1",
		}
		rollout_payload = {
			"format": collector.FORMAT,
			"collection": collection,
			"sequences": sequences,
		}
		rollout_path = _write_json(rollout_root / "manifest.json", rollout_payload)
		feature_sequences = []
		for sequence, arrays in feature_inputs:
			metadata = extractor.validate_feature_arrays(
				arrays, num_frames=self.transitions + 1
			)
			asset_path = feature_root / "episodes" / f"{sequence['sequence_id']}.npz"
			asset_path.parent.mkdir(parents=True, exist_ok=True)
			np.savez(asset_path, **arrays)
			runtime = arrays["runtime_ms"].astype(np.float64)
			feature_sequences.append(
				{
					"sequence_id": sequence["sequence_id"],
					"split": split,
					"episode": sequence["episode"],
					"source": sequence["source"],
					"env_seed": sequence["env_seed"],
					"background_seed": sequence["background_seed"],
					"action_seed": sequence["action_seed"],
					"source_selection_reset_attempt": 1,
					"ended_by_environment": False,
					"num_transitions": self.transitions,
					"frames": [
						{
							"t": frame["t"],
							"image_sha256": frame["image_sha256"],
							"source_frame_index": frame["source_frame_index"],
						}
						for frame in sequence["frames"]
					],
					"rollout_transitions_asset": copy.deepcopy(
						sequence["transitions_asset"]
					),
					"npz_asset": {
						"path": asset_path.relative_to(feature_root).as_posix(),
						"file_sha256": _sha256(asset_path),
						"arrays": {
							name: value
							for name, value in metadata.items()
							if name != "coverage"
						},
					},
					"coverage": metadata["coverage"],
					"runtime_ms": {
						"total": float(runtime.sum()),
						"mean": float(runtime.mean()),
						"p95": float(np.percentile(runtime, 95)),
					},
				}
			)
		feature_payload = {
			"format": extractor.FEATURE_FORMAT,
			"rollout_manifest_sha256": _sha256(rollout_path),
			"perception_provenance": copy.deepcopy(self._perception_provenance),
			"roles": ["whole_arm", "goal"],
			"object_schema": "whole_arm_goal_v1",
			"collection": {
				"task": collector.TASK,
				"split": split,
				"num_sequences": len(sequences),
				"num_frames": len(sequences) * (self.transitions + 1),
				"native_input_size": [64, 64],
				"tracker_size": [448, 448],
				"no_test": True,
				"no_support_trajectories": True,
				"causal": True,
				"episode_memory_reset": True,
				"permanent_support_loaded_once": True,
				"transitions_passed_to_cutie": False,
				"rollout_labels_passed_to_cutie": False,
			},
			"feature_contract": {
				"feature_dim": 2048,
				"foreground_queries": 8,
				"query_dim": 256,
				"query_order": (
					"official_query_post_process_cache_first_8_flattened"
				),
				"valid_definition": (
					"(~official_lost) & mask_nonempty & feature_finite"
				),
				"missing_policy": (
					"No future fill and no success-only filtering; consumers must mask valid=false."
				),
				"mask_values": [0, 1],
				"centroid_convention": (
					"[x,y] in native 64x64 RGB; NaN is allowed when invalid"
				),
			},
			"provenance": {"fixture_kind": "temporary_synthetic_only"},
			"runtime": {"backend": "synthetic_cpu"},
			"sequences": feature_sequences,
		}
		feature_path = _write_json(feature_root / "manifest.json", feature_payload)
		self.rollout_paths[split] = rollout_path
		self.feature_paths[split] = feature_path
		self.rollout_payloads[split] = rollout_payload
		self.feature_payloads[split] = feature_payload
		return uid

	def mutated_rollout(self, split: str, name: str, mutate) -> Path:
		payload = copy.deepcopy(self.rollout_payloads[split])
		mutate(payload)
		return _write_json(self.rollout_paths[split].parent / name, payload)

	def mutated_features(self, split: str, name: str, mutate) -> Path:
		payload = copy.deepcopy(self.feature_payloads[split])
		mutate(payload)
		return _write_json(self.feature_paths[split].parent / name, payload)


def _load_video_background_module():
	"""Load only the wrapper file with tiny gym/torch stubs."""
	fake_gym = ModuleType("gymnasium")

	class Wrapper:
		def __init__(self, env):
			self.env = env

	fake_gym.Wrapper = Wrapper
	fake_torch = ModuleType("torch")
	module_path = (
		Path(__file__).resolve().parent / "envs" / "wrappers" / "video_background.py"
	)
	spec = importlib.util.spec_from_file_location(
		"_visual_small_contract_video_background", module_path
	)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not load wrapper module from {module_path}.")
	module = importlib.util.module_from_spec(spec)
	with mock.patch.dict(
		sys.modules, {"gymnasium": fake_gym, "torch": fake_torch}
	):
		spec.loader.exec_module(module)
	return module


class VisualSmallObjectFeatureValidationContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._temporary = tempfile.TemporaryDirectory(
			prefix="visual_small_object_feature_contract_"
		)
		cls.root = Path(cls._temporary.name)
		cls.packs = SyntheticPacks(cls.root)
		cls.train_rollout = evaluator._load_rollout_manifest(
			cls.packs.rollout_paths["train"], "train"
		)
		cls.validation_rollout = evaluator._load_rollout_manifest(
			cls.packs.rollout_paths["validation"], "validation"
		)
		cls.train_features = evaluator._load_feature_manifest(
			cls.packs.feature_paths["train"], "train", cls.train_rollout
		)
		cls.validation_features = evaluator._load_feature_manifest(
			cls.packs.feature_paths["validation"],
			"validation",
			cls.validation_rollout,
		)

	@classmethod
	def tearDownClass(cls):
		cls._temporary.cleanup()

	def assertContractFailure(self, callable_, *args):
		with self.assertRaises((ValueError, RuntimeError, OSError)):
			callable_(*args)

	def test_00_wrapper_background_seed_override_and_fallback(self):
		module = _load_video_background_module()
		captured: list[int] = []

		class FakeSelector:
			def __init__(self, *_args, **_kwargs):
				pass

		class FakeCompositor:
			def __init__(self, **kwargs):
				captured.append(kwargs["seed"])

		module.ColorMultiSplitSelector = FakeSelector
		module.VideoBackgroundCompositor = FakeCompositor
		env = SimpleNamespace(observation_space=SimpleNamespace(shape=(9, 64, 64)))
		base = {
			"video_background_root": "unused-by-fake",
			"video_background_split": "train",
			"seed": 31415,
		}
		module.ColorMultiVideoBackgroundWrapper(env, dict(base))
		module.ColorMultiVideoBackgroundWrapper(
			env, {**base, "video_background_seed": 92653}
		)
		self.assertEqual(captured, [31415, 92653])

	def test_01_preselection_matches_compositor_and_skips_rejected_envs(self):
		module = _load_video_background_module()
		allowed = ("alpha.mp4", "beta.mp4", "gamma.mp4")
		selector = SimpleNamespace(
			resolve=lambda _split: tuple(Path(name) for name in allowed),
			source_names=lambda _split: allowed,
			manifest_sha256=lambda _split: "b" * 64,
			combined_manifest_sha256="c" * 64,
		)
		seed = collector._domain_seed(73, "train", "background", 9)
		compositor = module.VideoBackgroundCompositor(
			selector=selector,
			split="train",
			size=(64, 64),
			seed=seed,
		)
		compositor._load_source_frames = lambda _source: [
			np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(7)
		]
		random = np.random.RandomState(seed)
		expected_source = allowed[int(random.randint(0, len(allowed)))]
		expected_start = int(random.randint(0, 7))
		self.assertEqual(
			collector._preselect_background_source(allowed, seed), expected_source
		)
		compositor.reset()
		self.assertEqual(Path(compositor.active_source).name, expected_source)
		self.assertEqual(compositor._start, expected_start)

		target = "beta.mp4"
		base_seed = 991
		accepted_ordinal = next(
			ordinal
			for ordinal in range(100)
			if collector._preselect_background_source(
				allowed,
				collector._domain_seed(
					base_seed, "train", "background", ordinal
				),
			)
			== target
		)
		make_env_seeds: list[int] = []

		class FakeEnvironment:
			def __init__(self, background_seed):
				make_env_seeds.append(background_seed)
				self.background = SimpleNamespace(
					source_names=allowed,
					manifest_sha256="d" * 64,
					combined_manifest_sha256="e" * 64,
					active_split="train",
					active_source=None,
				)

			def reset(self):
				chosen = collector._preselect_background_source(
					allowed, make_env_seeds[-1]
				)
				self.background.active_source = Path("synthetic") / chosen
				return np.zeros((9, 64, 64), dtype=np.uint8)

			def close(self):
				pass

		def make_env(cfg):
			return FakeEnvironment(cfg.video_background_seed)

		args = SimpleNamespace(
			video_root=self.root,
			manifest_dir=None,
		)
		used: set[int] = set()
		with mock.patch.object(
			collector,
			"find_background_wrapper",
			side_effect=lambda env: env.background,
		):
			result = collector._select_balanced_environment(
				make_env,
				args,
				"train",
				allowed,
				"d" * 64,
				"e" * 64,
				{"alpha.mp4": 1, "beta.mp4": 0, "gamma.mp4": 1},
				100,
				123456,
				base_seed,
				0,
				used,
			)
		expected_seed = collector._domain_seed(
			base_seed, "train", "background", accepted_ordinal
		)
		self.assertEqual(make_env_seeds, [expected_seed])
		self.assertEqual(result[3], target)
		self.assertEqual(result[4], expected_seed)
		self.assertEqual(result[5], accepted_ordinal + 1)
		self.assertEqual(result[6], accepted_ordinal + 1)
		self.assertEqual(used, {expected_seed})

	def test_02_collector_transition_alignment_dtypes_and_n_plus_one(self):
		validated = collector.load_and_validate_rollout_manifest(
			self.packs.rollout_paths["train"]
		)
		payload = validated.payload
		self.assertEqual(payload["collection"]["action_repeat"], 2)
		self.assertEqual(
			payload["collection"]["transition_alignment"],
			"obs[t] -- action[t], reward[t] --> obs[t+1]",
		)
		for sequence in payload["sequences"]:
			n = sequence["num_transitions"]
			self.assertEqual(len(sequence["frames"]), n + 1)
			self.assertEqual([frame["t"] for frame in sequence["frames"]], list(range(n + 1)))
			asset = validated.root / sequence["transitions_asset"]["path"]
			with np.load(asset, allow_pickle=False) as arrays:
				self.assertTrue(collector.validate_transition_arrays(arrays, n, 2))
				self.assertEqual(arrays["actions"].dtype, np.float32)
				self.assertEqual(arrays["rewards"].dtype, np.float32)
				self.assertEqual(arrays["terminated"].dtype, np.bool_)
				self.assertEqual(arrays["truncated"].dtype, np.bool_)

	def test_03_collector_rejects_exact_schema_path_hash_and_held_out_failures(self):
		cases = {
			"top_extra.json": lambda p: p.update({"physics": {}}),
			"collection_extra.json": lambda p: p["collection"].update({"extra": 1}),
			"sequence_extra.json": lambda p: p["sequences"][0].update({"extra": 1}),
			"frame_extra.json": lambda p: p["sequences"][0]["frames"][0].update({"extra": 1}),
			"asset_extra.json": lambda p: p["sequences"][0]["transitions_asset"].update({"extra": 1}),
			"path_escape.json": lambda p: p["sequences"][0]["frames"][0].update({"image": "../escape.png"}),
			"image_hash.json": lambda p: p["sequences"][0]["frames"][0].update({"image_sha256": "0" * 64}),
			"n_plus_one.json": lambda p: p["sequences"][0]["frames"].pop(),
			"alignment.json": lambda p: p["collection"].update({"transition_alignment": "misaligned"}),
			"no_test.json": lambda p: p["collection"].update({"no_test": False}),
		}
		for name, mutate in cases.items():
			with self.subTest(name=name):
				path = self.packs.mutated_rollout("train", name, mutate)
				self.assertContractFailure(
					collector.load_and_validate_rollout_manifest, path
				)
		for forbidden in ("test", "support"):
			with self.subTest(split=forbidden):
				path = self.packs.mutated_rollout(
					"train",
					f"forbidden_{forbidden}.json",
					lambda payload, split=forbidden: payload["collection"].update(
						{"split": split}
					),
				)
				self.assertContractFailure(
					collector.load_and_validate_rollout_manifest, path
				)
		args = SimpleNamespace(
			video_root=self.root,
			manifest_dir=None,
		)
		for forbidden in ("test", "support"):
			self.assertContractFailure(
				collector.make_collector_config, args, forbidden, 1, 2
			)

	def test_04_collector_rejects_transition_and_asset_tampering(self):
		valid = {
			"actions": np.zeros((2, 2), dtype=np.float32),
			"rewards": np.zeros((2,), dtype=np.float32),
			"terminated": np.zeros((2,), dtype=np.bool_),
			"truncated": np.zeros((2,), dtype=np.bool_),
		}
		self.assertTrue(collector.validate_transition_arrays(valid, 2, 2))
		bad = _copy_arrays(valid)
		bad["actions"] = bad["actions"].astype(np.float64)
		self.assertContractFailure(collector.validate_transition_arrays, bad, 2, 2)
		bad = _copy_arrays(valid)
		bad["terminated"][0] = True
		bad["truncated"][0] = True
		self.assertContractFailure(collector.validate_transition_arrays, bad, 2, 2)
		bad_extra = dict(valid)
		bad_extra["extra"] = np.zeros((2,), dtype=np.float32)
		self.assertContractFailure(
			collector.validate_transition_arrays, bad_extra, 2, 2
		)

		sequence = self.packs.rollout_payloads["train"]["sequences"][0]
		asset = self.packs.rollout_paths["train"].parent / sequence["transitions_asset"]["path"]
		raw = asset.read_bytes()
		try:
			asset.write_bytes(raw + b"tamper")
			self.assertContractFailure(
				collector.load_and_validate_rollout_manifest,
				self.packs.rollout_paths["train"],
			)
		finally:
			asset.write_bytes(raw)

	def test_05_collector_three_seed_domains_are_disjoint_and_enforced(self):
		all_sets: list[set[int]] = []
		for split in collector.ALLOWED_SPLITS:
			validated = collector.load_and_validate_rollout_manifest(
				self.packs.rollout_paths[split]
			)
			for field, kind in (
				("env_seed", "env"),
				("background_seed", "background"),
				("action_seed", "action"),
			):
				values = {item[field] for item in validated.payload["sequences"]}
				self.assertEqual(len(values), len(validated.payload["sequences"]))
				self.assertEqual(
					{value % collector.SEED_DOMAIN_SIZE for value in values},
					{collector.SEED_DOMAIN_TAGS[(split, kind)]},
				)
				all_sets.append(values)
		for index, left in enumerate(all_sets):
			for right in all_sets[index + 1 :]:
				self.assertFalse(left & right)

		wrong_domain = self.packs.mutated_rollout(
			"train",
			"wrong_seed_domain.json",
			lambda p: p["sequences"][0].update(
				{"background_seed": p["sequences"][0]["env_seed"]}
			),
		)
		self.assertContractFailure(
			collector.load_and_validate_rollout_manifest, wrong_domain
		)
		reused = self.packs.mutated_rollout(
			"train",
			"reused_seed.json",
			lambda p: p["sequences"][1].update(
				{"env_seed": p["sequences"][0]["env_seed"]}
			),
		)
		self.assertContractFailure(
			collector.load_and_validate_rollout_manifest, reused
		)

	def test_06_extractor_exact_dtypes_valid_definition_and_lost_zero(self):
		arrays, _ = _synthetic_feature_arrays([(20, 20), (22, 21)], (45, 40))
		arrays["lost"][1, 0] = True
		arrays["valid"][1, 0] = False
		arrays["features"][1, 0] = 0.0
		arrays["masks"][1, 0] = 0
		arrays["centroid_xy"][1, 0] = np.nan
		metadata = extractor.validate_feature_arrays(arrays, num_frames=2)
		self.assertEqual(metadata["features"]["dtype"], "float32")
		self.assertEqual(metadata["masks"]["dtype"], "uint8")
		self.assertEqual(metadata["lost"]["dtype"], "bool")
		self.assertEqual(metadata["coverage"]["official_lost_count"], [1, 0])

		bad_extra = _copy_arrays(arrays)
		bad_extra["extra"] = np.zeros((2,), dtype=np.float32)
		self.assertContractFailure(extractor.validate_feature_arrays, bad_extra)
		bad_dtype = _copy_arrays(arrays)
		bad_dtype["confidence"] = bad_dtype["confidence"].astype(np.float64)
		self.assertContractFailure(extractor.validate_feature_arrays, bad_dtype)
		bad_valid = _copy_arrays(arrays)
		bad_valid["valid"][1, 0] = True
		self.assertContractFailure(extractor.validate_feature_arrays, bad_valid)
		bad_lost = _copy_arrays(arrays)
		bad_lost["features"][1, 0, 0] = np.float32(1.0)
		self.assertContractFailure(extractor.validate_feature_arrays, bad_lost)

	def test_07_query_pool_is_invariant_to_eight_slot_permutation(self):
		rng = np.random.default_rng(20260825)
		features = rng.standard_normal((5, 2, 2048)).astype(np.float32)
		valid = np.ones((5, 2), dtype=np.bool_)
		valid[3, 1] = False
		permutation = np.asarray([7, 0, 5, 2, 6, 1, 4, 3])
		permuted = features.reshape(5, 2, 8, 256)[:, :, permutation].reshape(
			5, 2, 2048
		)
		pooled = evaluator._query_pool(features, valid)
		permuted_pooled = evaluator._query_pool(permuted, valid)
		np.testing.assert_allclose(pooled, permuted_pooled, rtol=1e-6, atol=1e-6)
		np.testing.assert_array_equal(pooled[3, 1], np.zeros((512,), np.float32))

		# Object/query/mask features must receive exactly the same three-observation
		# history convention as the RGB baseline.  This prevents a hidden one-frame
		# versus three-frame disadvantage in the non-inferiority probe.
		values = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
		stacked = evaluator._temporal_stack_features(values)
		np.testing.assert_array_equal(
			stacked,
			np.asarray(
				[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
				dtype=np.float32,
			),
		)
		valid = evaluator._temporal_stack_valid(
			np.asarray([True, True, False, True], dtype=np.bool_)
		)
		np.testing.assert_array_equal(
			valid, np.asarray([True, True, False], dtype=np.bool_)
		)

	def test_08_dynamics_target_excludes_static_goal_trap(self):
		def dataset(split: str, rows: int) -> evaluator.ProbeDataset:
			spatial = np.zeros((rows, 2 * evaluator.SPATIAL_DIM_PER_ROLE), np.float32)
			next_spatial = np.zeros_like(spatial)
			whole = np.arange(rows, dtype=np.float32)[:, None] + np.arange(
				evaluator.SPATIAL_DIM_PER_ROLE, dtype=np.float32
			)[None, :] / 100.0
			spatial[:, : evaluator.SPATIAL_DIM_PER_ROLE] = whole - 0.25
			next_spatial[:, : evaluator.SPATIAL_DIM_PER_ROLE] = whole
			spatial[:, evaluator.SPATIAL_DIM_PER_ROLE :] = 777.0
			next_spatial[:, evaluator.SPATIAL_DIM_PER_ROLE :] = 999.0
			return evaluator.ProbeDataset(
				split=split,
				actions=np.stack(
					[
						np.linspace(-1, 1, rows),
						np.linspace(1, -1, rows),
					],
					axis=1,
				).astype(np.float32),
				rewards=np.ones((rows,), np.float32),
				query=np.zeros((rows, 1), np.float32),
				mask=np.zeros((rows, 1), np.float32),
				object=np.zeros((rows, 1), np.float32),
				spatial=spatial,
				next_spatial=next_spatial,
				rgb=np.zeros((rows, 1), np.float32),
				background_rgb=np.zeros((rows, 1), np.float32),
				reward_valid=np.ones((rows,), np.bool_),
				dynamics_valid=np.ones((rows,), np.bool_),
				sources=np.asarray(
					[f"source{index // 2}" for index in range(rows)], dtype=object
				),
				episodes=np.asarray(
					[f"episode{index // 2}" for index in range(rows)], dtype=object
				),
				terminated=np.zeros((rows,), np.bool_),
				truncated=np.zeros((rows,), np.bool_),
				coverage={},
			)

		train = dataset("train", 6)
		validation = dataset("validation", 4)
		train_z = {
			"object": np.arange(12, dtype=np.float32).reshape(6, 2),
			"rgb": np.arange(12, dtype=np.float32).reshape(6, 2),
		}
		validation_z = {
			"object": np.arange(8, dtype=np.float32).reshape(4, 2),
			"rgb": np.arange(8, dtype=np.float32).reshape(4, 2),
		}
		captured_targets: list[np.ndarray] = []

		def fake_fit_predict(_train_x, train_y, validation_x):
			captured_targets.append(np.array(train_y, copy=True))
			prediction = np.repeat(train_y.mean(axis=0, keepdims=True), len(validation_x), axis=0)
			return prediction, "f" * 64

		comparison = {
			"relative_improvement": 0.0,
			"ci95": [-0.1, 0.1],
			"candidate_mean_loss": 1.0,
			"baseline_mean_loss": 1.0,
		}
		with mock.patch.object(
			evaluator, "_fit_predict", side_effect=fake_fit_predict
		), mock.patch.object(
			evaluator,
			"_hierarchical_bootstrap_comparison",
			return_value=comparison,
		):
			result = evaluator._probe_dynamics(
				train, validation, train_z, validation_z
			)
		expected = train.next_spatial[:, : evaluator.SPATIAL_DIM_PER_ROLE]
		self.assertEqual(result["target_dim"], evaluator.SPATIAL_DIM_PER_ROLE)
		self.assertIn("whole_arm", result["target"])
		self.assertIn("excluded", result["goal_policy"])
		self.assertEqual(len(captured_targets), 4)
		for target in captured_targets:
			np.testing.assert_array_equal(target, expected)
			self.assertFalse(np.any(target == 999.0))

	def test_09_train_fit_occurs_before_validation_materialization(self):
		events: list[str] = []
		train = SimpleNamespace(
			split="train",
			actions=np.zeros((2, 2), dtype=np.float32),
			coverage={},
			sources=np.asarray(["train-source"], dtype=object),
			episodes=np.asarray(["train-episode"], dtype=object),
		)
		validation = SimpleNamespace(
			split="validation",
			actions=np.zeros((2, 2), dtype=np.float32),
			coverage={},
			sources=np.asarray(["validation-source"], dtype=object),
			episodes=np.asarray(["validation-episode"], dtype=object),
		)

		class Projector:
			def metadata(self):
				return {"fit": "train-only"}

		def load_dataset(rollout, _features):
			events.append(f"load:{rollout.split}")
			return train if rollout.split == "train" else validation

		def fit_projectors(dataset):
			events.append(f"fit:{dataset.split}")
			return {"fixture": Projector()}

		def transform(dataset, _projectors):
			events.append(f"transform:{dataset.split}")
			return {}

		args = argparse.Namespace(
			train_rollout=self.packs.rollout_paths["train"],
			train_features=self.packs.feature_paths["train"],
			validation_rollout=self.packs.rollout_paths["validation"],
			validation_features=self.packs.feature_paths["validation"],
			output=self.root / "fit_order_report.json",
		)
		with mock.patch.object(
			evaluator,
			"_load_rollout_manifest",
			side_effect=[self.train_rollout, self.validation_rollout],
		), mock.patch.object(
			evaluator,
			"_load_feature_manifest",
			side_effect=[self.train_features, self.validation_features],
		), mock.patch.object(
			evaluator, "_load_dataset", side_effect=load_dataset
		), mock.patch.object(
			evaluator, "_fit_projectors", side_effect=fit_projectors
		), mock.patch.object(
			evaluator, "_transform_modalities", side_effect=transform
		), mock.patch.object(
			evaluator, "_probe_reward", return_value={}
		), mock.patch.object(
			evaluator, "_probe_dynamics", return_value={}
		), mock.patch.object(
			evaluator, "_coverage_status", return_value={}
		), mock.patch.object(
			evaluator, "_conclusion_status", return_value={"status": "fixture"}
		):
			evaluator.evaluate(args)
		self.assertEqual(
			events,
			[
				"load:train",
				"fit:train",
				"transform:train",
				"load:validation",
				"transform:validation",
			],
		)

	def test_10_cross_split_overlap_and_provenance_rejection(self):
		fields = (
			"sequence_ids",
			"sources",
			"env_seeds",
			"background_seeds",
			"action_seeds",
			"image_sha256s",
		)
		for field in fields:
			with self.subTest(field=field):
				value = frozenset({next(iter(getattr(self.train_rollout, field)))})
				bad_validation = replace(self.validation_rollout, **{field: value})
				self.assertContractFailure(
					evaluator._validate_cross_split,
					self.train_rollout,
					bad_validation,
					self.train_features,
					self.validation_features,
				)
		bad_features = replace(
			self.validation_features,
			provenance={"algorithm": "different"},
			provenance_sha256="0" * 64,
		)
		self.assertContractFailure(
			evaluator._validate_cross_split,
			self.train_rollout,
			self.validation_rollout,
			self.train_features,
			bad_features,
		)

	def test_11_feature_rollout_provenance_schema_and_asset_tampering_rejected(self):
		cases = {
			"wrong_rollout_sha.json": lambda p: p.update(
				{"rollout_manifest_sha256": "0" * 64}
			),
			"privileged_provenance.json": lambda p: p[
				"perception_provenance"
			].update({"physics": "forbidden"}),
			"bogus_equal_provenance.json": lambda p: p.update(
				{"perception_provenance": {"algorithm": "bogus"}}
			),
			"wrong_query_order.json": lambda p: p["feature_contract"].update(
				{"query_order": "arbitrary"}
			),
			"feature_top_extra.json": lambda p: p.update({"extra": 1}),
			"feature_sequence_extra.json": lambda p: p["sequences"][0].update(
				{"extra": 1}
			),
			"feature_path_escape.json": lambda p: p["sequences"][0][
				"npz_asset"
			].update({"path": "../escape.npz"}),
			"feature_asset_sha.json": lambda p: p["sequences"][0][
				"npz_asset"
			].update({"file_sha256": "0" * 64}),
		}
		for name, mutate in cases.items():
			with self.subTest(name=name):
				path = self.packs.mutated_features("train", name, mutate)
				self.assertContractFailure(
					evaluator._load_feature_manifest,
					path,
					"train",
					self.train_rollout,
				)

		sequence = self.packs.feature_payloads["train"]["sequences"][0]
		asset = self.packs.feature_paths["train"].parent / sequence["npz_asset"]["path"]
		raw = asset.read_bytes()
		try:
			asset.write_bytes(raw + b"tamper")
			self.assertContractFailure(
				evaluator._load_feature_manifest,
				self.packs.feature_paths["train"],
				"train",
				self.train_rollout,
			)
		finally:
			asset.write_bytes(raw)

	def test_12_sparse_reward_forces_reward_conclusions_inconclusive(self):
		strong = {"relative_improvement": 0.5, "ci95": [0.2, 0.8]}
		control = {"relative_improvement": 0.0, "ci95": [-0.1, 0.1]}
		reward = {
			"validation_sources": evaluator.MIN_VALIDATION_SOURCES,
			"validation_episodes": evaluator.MIN_VALIDATION_EPISODES,
			"positive_support": {"sufficient": False},
			"comparisons": {
				"object_vs_constant": strong,
				"object_increment_over_action": strong,
				"action_increment": strong,
				"object_vs_rgb": strong,
				"shuffled_action_increment": control,
				"background_increment_over_action": control,
				"shuffled_target_vs_constant": control,
				"shuffled_target_vs_action": control,
			},
		}
		dynamics = {
			"validation_sources": evaluator.MIN_VALIDATION_SOURCES,
			"validation_episodes": evaluator.MIN_VALIDATION_EPISODES,
			"comparisons": {
				"object_vs_persistence": strong,
				"action_increment": strong,
				"object_vs_rgb": strong,
				"shuffled_action_increment": control,
			}
		}
		validation = SimpleNamespace(
			sources=np.asarray(
				[f"source-{index}" for index in range(evaluator.MIN_VALIDATION_SOURCES)],
				dtype=object,
			),
			episodes=np.asarray(
				[
					f"episode-{index}"
					for index in range(evaluator.MIN_VALIDATION_EPISODES)
				],
				dtype=object,
			),
		)
		status = evaluator._conclusion_status(
			{"status": "pass"}, reward, dynamics, validation
		)
		self.assertEqual(status["object_information"]["reward_status"], "inconclusive")
		self.assertEqual(status["action_increment"]["reward_status"], "inconclusive")
		self.assertEqual(status["action_increment"]["dynamics_status"], "pass")
		self.assertEqual(status["action_increment"]["status"], "pass")
		self.assertEqual(status["action_increment"]["hard_gate_modality"], "dynamics")
		self.assertEqual(status["RGB_noninferiority"]["reward_status"], "inconclusive")

		state_defined_reward = copy.deepcopy(reward)
		state_defined_reward["positive_support"]["sufficient"] = True
		state_defined_reward["comparisons"]["action_increment"] = control
		state_reward_status = evaluator._conclusion_status(
			{"status": "pass"}, state_defined_reward, dynamics, validation
		)
		self.assertEqual(state_reward_status["action_increment"]["reward_status"], "fail")
		self.assertEqual(state_reward_status["action_increment"]["dynamics_status"], "pass")
		self.assertEqual(state_reward_status["action_increment"]["status"], "pass")

		# A strong action-only oracle may beat the constant baseline.  It must not
		# be misreported as object information when object+a adds nothing over the
		# action-only model.
		action_only_oracle = copy.deepcopy(reward)
		action_only_oracle["positive_support"]["sufficient"] = True
		action_only_oracle["comparisons"]["object_vs_constant"] = strong
		action_only_oracle["comparisons"]["object_increment_over_action"] = control
		false_positive_status = evaluator._conclusion_status(
			{"status": "pass"}, action_only_oracle, dynamics, validation
		)
		self.assertEqual(
			false_positive_status["object_information"]["reward_status"], "fail"
		)
		self.assertFalse(
			false_positive_status["object_information"]["reward_pass"]
		)

	def test_13_lightweight_synthetic_oracle_and_negative_controls_evaluate(self):
		output = self.root / "synthetic_evaluator_report.json"
		args = argparse.Namespace(
			train_rollout=self.packs.rollout_paths["train"],
			train_features=self.packs.feature_paths["train"],
			validation_rollout=self.packs.rollout_paths["validation"],
			validation_features=self.packs.feature_paths["validation"],
			output=output,
		)
		with mock.patch.multiple(
			evaluator,
			BOOTSTRAP_SAMPLES=12,
			PCA_COMPONENTS=4,
			MIN_TRAIN_POSITIVES=1,
			MIN_VALIDATION_POSITIVES=1,
			MIN_VALIDATION_POSITIVE_SOURCES=1,
			MIN_VALIDATION_SOURCES=1,
			MIN_VALIDATION_EPISODES=1,
		):
			report = evaluator.evaluate(args)
		self.assertTrue(output.is_file())
		self.assertEqual(report["format"], evaluator.REPORT_FORMAT)
		self.assertTrue(report["reward_probe"]["positive_support"]["sufficient"])
		self.assertEqual(
			report["dynamics_probe"]["target_dim"],
			evaluator.SPATIAL_DIM_PER_ROLE,
		)
		self.assertTrue(
			report["configuration"]["object_representation"][
				"query_slot_permutation_invariant"
			]
		)
		self.assertEqual(
			report["configuration"]["object_representation"]["stack"],
			report["configuration"]["rgb_representation"]["stack"],
		)
		for split in ("train", "validation"):
			coverage = report["coverage"][split]
			self.assertIn("reward_history_all_roles_valid_fraction", coverage)
			self.assertIn(
				"dynamics_history_and_next_all_roles_valid_fraction", coverage
			)
			self.assertNotIn("current_all_roles_valid_fraction", coverage)
		self.assertFalse(report["isolation"]["test_trajectory_pixels_read"])
		self.assertFalse(report["isolation"]["support_trajectory_pixels_read"])
		self.assertTrue(report["isolation"]["verified_support_prompt_used_by_cutie"])
		self.assertEqual(
			report["scope"]["decision_level"],
			"eligibility_for_small_matched_rl_pilot",
		)
		self.assertFalse(report["scope"]["causal_background_invariance_established"])
		self.assertFalse(report["scope"]["paired_background_counterfactual_included"])
		reward_models = report["reward_probe"]["models"]
		self.assertLess(
			reward_models["object_plus_action"]["rmse"],
			reward_models["constant"]["rmse"],
		)
		self.assertLess(
			reward_models["object_plus_action"]["rmse"],
			reward_models["action_only"]["rmse"],
		)
		self.assertLess(
			reward_models["object_plus_action"]["rmse"],
			reward_models["object_plus_action_shuffled_target"]["rmse"],
		)
		self.assertTrue(
			{
				"object_increment_over_action",
				"shuffled_action_increment",
				"background_increment_over_action",
				"shuffled_target_vs_constant",
				"shuffled_target_vs_action",
			}.issubset(report["reward_probe"]["comparisons"])
		)
		self.assertIn(
			"shuffled_action_increment", report["dynamics_probe"]["comparisons"]
		)


if __name__ == "__main__":
	unittest.main(verbosity=2)
