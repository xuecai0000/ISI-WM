"""Dependency-light contracts for the ROF causal diagnostic ladder.

The ladder is an *offline diagnosis*, not another controller training run.  The
collector records exactly the observations seen by the frozen policy and puts
simulator state / segmentation in a separate ``labels__`` namespace.  The
evaluator may use those labels as probe targets or for scoring, but they can
never become policy inputs.

This file intentionally tests the public dataset boundary rather than a live
environment or a checkpoint.  It therefore catches temporal leakage and
fail-open summaries without CUDA, dm-control, Hydra, or Torch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
for _local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while _local_path in sys.path:
		sys.path.remove(_local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))


DATASET_FORMAT = "rof_causal_probe_dataset_v1"
RESULT_FORMAT = "rof_causal_ladder_result_v1"
TEMPORAL_ALIGNMENT = "obs_t_action_t_reward_t_done_t_obs_t_plus_1"
POLICY_KEYS = (
	"policy_rgb",
	"policy_object",
	"policy_object_mask",
	"policy_role_exists",
)
LABEL_KEYS = (
	"labels__state",
	"labels__gt_role_mask",
	"labels__gt_visible",
)
TRANSITION_KEYS = ("action", "reward", "done")
INDEX_KEYS = ("episode_id", "step")
LADDER_KEYS = ("A_mask", "B_sufficiency", "C_dynamics", "D_control")
DIAGNOSTIC_STATUSES = frozenset(("passed", "failed", "inconclusive"))


def _load_subjects():
	collector = importlib.import_module(
		"tdmpc2.tools.collect_rof_causal_probe_dataset"
	)
	evaluator = importlib.import_module(
		"tdmpc2.tools.evaluate_rof_causal_ladder"
	)
	return collector, evaluator


def _arrays(*, episode_id: int = 17, steps: int = 5, roles: int = 2,
			state_dim: int = 5, action_dim: int = 1) -> dict[str, np.ndarray]:
	"""Small, shape-realistic synthetic episode using the frozen T+1/T rule."""
	observations = steps + 1
	return {
		"policy_rgb": np.zeros((observations, 9, 64, 64), dtype=np.uint8),
		"policy_object": np.zeros(
			(observations, roles, 1770), dtype=np.float32
		),
		"policy_object_mask": np.zeros(
			(observations, roles, 3, 64, 64), dtype=np.bool_
		),
		"policy_role_exists": np.ones(
			(observations, roles), dtype=np.float32
		),
		"action": np.zeros((steps, action_dim), dtype=np.float32),
		"reward": np.arange(steps, dtype=np.float32),
		"done": np.asarray([False] * (steps - 1) + [True], dtype=np.bool_),
		"labels__state": np.zeros(
			(observations, state_dim), dtype=np.float64
		),
		"labels__gt_role_mask": np.zeros(
			(observations, roles, 64, 64), dtype=np.bool_
		),
		"labels__gt_visible": np.zeros(
			(observations, roles), dtype=np.bool_
		),
		"episode_id": np.full(observations, episode_id, dtype=np.int64),
		"step": np.arange(observations, dtype=np.int64),
	}


def _valid_episode(collector, arrays: dict[str, np.ndarray], *, steps: int = 5):
	return collector.validate_episode_arrays(
		arrays, role_count=2, state_dim=5, action_dim=1, steps=steps
	)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(path, **arrays)


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
	raw = json.dumps(
		payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
		allow_nan=False,
	).encode("utf-8")
	return hashlib.sha256(raw).hexdigest()


def _manifest(
	collector, root: Path, episode_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
):
	episodes = []
	for index, episode_id in enumerate(episode_ids):
		arrays = _arrays(episode_id=episode_id)
		path = root / "episodes" / f"episode_{episode_id}.npz"
		_write_npz(path, arrays)
		split = "train" if index < 3 else "validation" if index == 3 else "test"
		episodes.append({
			"episode_id": episode_id,
			"episode_index": index,
			"split": split,
			"steps": 5,
			"frames": 6,
			"temporal_semantics": TEMPORAL_ALIGNMENT,
			"planner_seed": 8675400 + index,
			"background_source": "clean",
			"background_start_frame_index": 0,
			"path": path.relative_to(root).as_posix(),
			"sha256": _sha256(path),
			"policy_input_trace_sha256": (
				collector.policy_input_trace_sha256(arrays)
			),
			"label_trace_sha256": collector.label_trace_sha256(arrays),
		})
	checkpoint = root / "final.pt"
	payload = {
		"format": DATASET_FORMAT,
		"task": "reacher-easy",
		"condition": "clean",
		"temporal_semantics": TEMPORAL_ALIGNMENT,
		"policy_observation_keys": list(POLICY_KEYS),
		"label_keys": list(LABEL_KEYS),
		"policy_input_contract": {
			"schema": "robust_object_field_v0",
			"keys": list(POLICY_KEYS),
			"privileged_labels_excluded": True,
			"hash_scope": "exactly_policy_input_keys",
		},
		"role_names": ["whole_arm", "goal"],
		"role_count": 2,
		"action_dim": 1,
		"steps": 5,
		"frames_per_episode": 6,
		"source": {
			"runtime_config": str((root / "runtime_config.json").resolve()),
			"runtime_config_sha256": "b" * 64,
			"checkpoint": str(checkpoint.resolve()),
			"checkpoint_sha256": _sha256(checkpoint),
			"checkpoint_step": 100000,
			"backend": "robust_object_field",
			"policy_input_contract": "robust_object_field_v0",
		},
		"collection": {
			"episodes": len(episodes),
			"condition": "clean",
			"action_dim": 1,
			"max_steps": 5,
			"env_seed": 424243,
			"background_seed": 1618034,
			"planner_seed_base": 8675400,
			"eval_mode": True,
			"camera_id": 0,
			"gt_query_order": (
				"after_rof_observation_before_agent_action_same_state"
			),
		},
		"label_contract": {
			"namespace": "labels__",
			"keys": list(LABEL_KEYS),
			"role_names": ["whole_arm", "goal"],
			"labels_never_policy_input": True,
			"queried_after_policy_observation": True,
			"same_physics_state_guard": "exact_before_after",
			"state_source": "dm_control.task.get_observation(current_physics)",
			"state_schema": [{
				"name": "position", "shape": [5], "start": 0, "stop": 5,
			}],
			"state_names": [f"position[{index}]" for index in range(5)],
			"gt_role_mask_source": (
				"same_state_mujoco_segmentation_policy_camera"
			),
			"gt_role_mapping": (
				"task_static_collect_cutie_multitask_support_selectors"
			),
			"visible_false_means_true_occlusion_not_missing_label": True,
		},
		"splits": {"train": [0, 1, 2], "validation": [3], "test": [4]},
		"episodes": episodes,
		"perception_runtime": {},
		"cutie_ready": {},
	}
	manifest = root / "manifest.json"
	manifest.write_text(
		json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
	)
	return manifest, payload


def _diagnostic(status: str) -> dict:
	return {
		"status": status,
		"checks": {"complete_inputs": True, "finite_metrics": True},
		"metrics": {},
	}


def _result(statuses: dict[str, str] | None = None) -> dict:
	statuses = statuses or {name: "passed" for name in LADDER_KEYS}
	diagnostics = {
		name: _diagnostic(statuses[name]) for name in LADDER_KEYS
	}
	all_passed = all(value == "passed" for value in statuses.values())
	return {
		"format": RESULT_FORMAT,
		"status": (
			"rof_causal_ladder_complete" if all_passed
			else "rof_causal_ladder_inconclusive"
		),
		"engineering_pass": True,
		"scientific_complete": all_passed,
		"controller_training_authorized": False,
		"diagnostics": diagnostics,
		"recommendation": (
			"diagnosis_complete_review_layer_attribution" if all_passed
			else "do_not_select_solution_complete_missing_diagnostics"
		),
	}


class DatasetBoundaryContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.collector, cls.evaluator = _load_subjects()

	def test_formats_and_namespaces_are_exact(self):
		self.assertEqual(self.collector.FORMAT, DATASET_FORMAT)
		self.assertEqual(self.evaluator.DATASET_FORMAT, DATASET_FORMAT)
		self.assertEqual(self.evaluator.RESULT_FORMAT, RESULT_FORMAT)
		self.assertEqual(tuple(self.collector.POLICY_INPUT_KEYS), POLICY_KEYS)
		self.assertEqual(tuple(self.collector.LABEL_KEYS), LABEL_KEYS)
		self.assertTrue(set(POLICY_KEYS).isdisjoint(LABEL_KEYS))
		self.assertTrue(all(name.startswith("policy_") for name in POLICY_KEYS))
		self.assertTrue(all(name.startswith("labels__") for name in LABEL_KEYS))

	def test_t_plus_one_observations_and_t_transitions(self):
		arrays = _arrays()
		_valid_episode(self.collector, arrays)
		for key in POLICY_KEYS + LABEL_KEYS + INDEX_KEYS:
			self.assertEqual(arrays[key].shape[0], 6, key)
		for key in TRANSITION_KEYS:
			self.assertEqual(arrays[key].shape[0], 5, key)
		self.assertEqual(arrays["step"].tolist(), [0, 1, 2, 3, 4, 5])

		for key in POLICY_KEYS + LABEL_KEYS + INDEX_KEYS:
			bad = deepcopy(arrays)
			bad[key] = bad[key][:-1]
			with self.assertRaises((ValueError, TypeError), msg=key):
				_valid_episode(self.collector, bad)
		for key in TRANSITION_KEYS:
			bad = deepcopy(arrays)
			bad[key] = np.concatenate((bad[key], bad[key][-1:]), axis=0)
			with self.assertRaises((ValueError, TypeError), msg=key):
				_valid_episode(self.collector, bad)

	def test_row_zero_and_episode_identity_fail_closed(self):
		arrays = _arrays()
		for mutation in ("nonzero_row0", "skipped_step", "mixed_episode"):
			bad = deepcopy(arrays)
			if mutation == "nonzero_row0":
				bad["step"] = bad["step"] + 1
			elif mutation == "skipped_step":
				bad["step"][2] += 1
			else:
				bad["episode_id"][-1] += 1
			with self.assertRaises((ValueError, TypeError), msg=mutation):
				_valid_episode(self.collector, bad)

	def test_policy_and_label_hash_domains_are_causally_disjoint(self):
		arrays = _arrays()
		policy_before = self.collector.policy_input_trace_sha256(arrays)
		labels_before = self.collector.label_trace_sha256(arrays)

		label_changed = deepcopy(arrays)
		label_changed["labels__state"][0, 0] = 123.0
		self.assertEqual(
			self.collector.policy_input_trace_sha256(label_changed), policy_before
		)
		self.assertNotEqual(
			self.collector.label_trace_sha256(label_changed), labels_before
		)

		policy_changed = deepcopy(arrays)
		policy_changed["policy_object"][0, 0, 0] = 123.0
		self.assertNotEqual(
			self.collector.policy_input_trace_sha256(policy_changed), policy_before
		)
		self.assertEqual(
			self.collector.label_trace_sha256(policy_changed), labels_before
		)

		action_changed = deepcopy(arrays)
		action_changed["action"][0, 0] = 1.0
		self.assertEqual(
			self.collector.policy_input_trace_sha256(action_changed), policy_before
		)
		self.assertEqual(
			self.collector.label_trace_sha256(action_changed), labels_before
		)

	def test_extra_privileged_or_unscoped_arrays_are_rejected(self):
		arrays = _arrays()
		for name in (
			"state", "gt_role_mask", "simulator_state", "policy_gt_state",
		):
			bad = deepcopy(arrays)
			bad[name] = np.zeros((6, 1), dtype=np.float32)
			with self.assertRaises((ValueError, TypeError), msg=name):
				_valid_episode(self.collector, bad)

	def test_manifest_revalidates_npz_hashes_and_label_policy_boundary(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			(root / "final.pt").write_bytes(b"synthetic")
			manifest, payload = _manifest(self.collector, root)
			self.collector.validate_dataset(manifest)
			self.evaluator.validate_manifest(manifest)

			for mutation in (
				"labels_enter_policy", "wrong_temporal_alignment", "missing_gt",
			):
				bad = deepcopy(payload)
				if mutation == "labels_enter_policy":
					bad["policy_observation_keys"].append("labels__state")
				elif mutation == "wrong_temporal_alignment":
					bad["temporal_semantics"] = (
						"action_t_obs_t_plus_1_reward_t"
					)
				else:
					bad["label_keys"].remove(
						"labels__gt_role_mask"
					)
				path = root / f"bad_{mutation}.json"
				path.write_text(json.dumps(bad), encoding="utf-8")
				with self.assertRaises((ValueError, TypeError), msg=mutation):
					self.collector.validate_dataset(path)

	def test_episode_splits_are_disjoint_and_complete(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			(root / "final.pt").write_bytes(b"synthetic")
			manifest, payload = _manifest(self.collector, root)
			self.evaluator.validate_manifest(manifest)
			for mutation in ("overlap", "missing", "record_disagrees"):
				bad = deepcopy(payload)
				if mutation == "overlap":
					bad["splits"]["validation"] = [2, 3]
				elif mutation == "missing":
					bad["splits"]["train"] = [0, 1]
				else:
					bad["episodes"][3]["split"] = "train"
				path = root / f"bad_split_{mutation}.json"
				path.write_text(json.dumps(bad), encoding="utf-8")
				with self.assertRaises((ValueError, TypeError), msg=mutation):
					self.collector.validate_dataset(path)


class SummaryFailClosedContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		_, cls.evaluator = _load_subjects()

	def _validate(self, payload: dict):
		validator = getattr(self.evaluator, "validate_result", None)
		if validator is None:
			raise AssertionError(
				"evaluate_rof_causal_ladder must expose validate_result(payload)"
			)
		return validator(payload)

	def test_complete_A_B_C_D_can_be_scientifically_complete(self):
		payload = _result()
		self._validate(payload)

	def test_missing_or_empty_ladder_rung_is_rejected(self):
		for rung in LADDER_KEYS:
			missing = _result()
			missing["diagnostics"].pop(rung)
			with self.assertRaises((ValueError, TypeError), msg=rung):
				self._validate(missing)
			empty = _result()
			empty["diagnostics"][rung]["checks"] = {}
			with self.assertRaises((ValueError, TypeError), msg=rung):
				self._validate(empty)

	def test_failed_or_inconclusive_rung_cannot_fail_open(self):
		for rung_status in ("failed", "inconclusive"):
			for rung in LADDER_KEYS:
				payload = _result({
					name: rung_status if name == rung else "passed"
					for name in LADDER_KEYS
				})
				self._validate(payload)
				bad = deepcopy(payload)
				bad["status"] = "rof_causal_ladder_complete"
				bad["scientific_complete"] = True
				with self.assertRaises((ValueError, TypeError), msg=(rung, rung_status)):
					self._validate(bad)

	def test_missing_checkpoint_makes_C_and_D_inconclusive_not_passed(self):
		payload = _result({
			"A_mask": "passed",
			"B_sufficiency": "passed",
			"C_dynamics": "inconclusive",
			"D_control": "inconclusive",
		})
		payload["diagnostics"]["C_dynamics"]["reason"] = (
			"checkpoint_unavailable"
		)
		payload["diagnostics"]["D_control"]["reason"] = (
			"checkpoint_unavailable"
		)
		self._validate(payload)
		for rung in ("C_dynamics", "D_control"):
			bad = deepcopy(payload)
			bad["diagnostics"][rung]["status"] = "passed"
			with self.assertRaises((ValueError, TypeError), msg=rung):
				self._validate(bad)

	def test_non_boolean_or_failed_checks_cannot_pass(self):
		for value in (False, 1, "true", None):
			payload = _result()
			payload["diagnostics"]["A_mask"]["checks"][
				"complete_inputs"
			] = value
			with self.assertRaises((ValueError, TypeError), msg=repr(value)):
				self._validate(payload)

	def test_diagnostic_never_authorizes_controller_training(self):
		payload = _result()
		payload["controller_training_authorized"] = True
		with self.assertRaises((ValueError, TypeError)):
			self._validate(payload)


if __name__ == "__main__":
	unittest.main(verbosity=2)
