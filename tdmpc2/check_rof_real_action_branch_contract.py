"""Dependency-light contracts for strict ROF real-action branch datasets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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

from tdmpc2.tools import collect_rof_real_action_branches as subject


ROLES = 1
ACTION_DIM = 1
STATE_DIM = 4
PHYSICS_DIM = 6
BRANCHES = 3
ANCHOR = 80
MAGNITUDE = 0.8


def _arrays() -> dict[str, np.ndarray]:
	result = {}
	for condition_index, condition in enumerate(subject.CONDITIONS):
		for part, lead in (
			('history', (subject.HISTORY_FRAMES,)),
			('future', (BRANCHES, subject.HORIZON)),
		):
			result[f'{condition}__{part}__policy_rgb'] = np.full(
				lead + (9, 64, 64), condition_index, dtype=np.uint8,
			)
			result[f'{condition}__{part}__policy_object'] = np.zeros(
				lead + (ROLES, 1770), dtype=np.float32,
			)
			result[f'{condition}__{part}__policy_object_mask'] = np.zeros(
				lead + (ROLES, 3, 64, 64), dtype=np.bool_,
			)
			result[f'{condition}__{part}__policy_role_exists'] = np.ones(
				lead + (ROLES,), dtype=np.float32,
			)
	result['history__action'] = np.zeros(
		(subject.HISTORY_FRAMES - 1, ACTION_DIM), dtype=np.float32,
	)
	result['branch__action'] = np.zeros(
		(BRANCHES, subject.HORIZON, ACTION_DIM), dtype=np.float32,
	)
	result['branch__action'][1, 0, 0] = MAGNITUDE
	result['branch__action'][2, 0, 0] = -MAGNITUDE
	result['branch__action'][:, 1:, 0] = np.asarray(
		[0.1, -0.2, 0.3, -0.4], dtype=np.float32,
	)
	result['branch__code'] = np.asarray([0, 1, -1], dtype=np.int32)
	result['branch__is_zero'] = np.asarray([True, False, False], dtype=np.bool_)
	result['oracle__official_state'] = np.zeros(
		(BRANCHES, subject.HORIZON + 1, STATE_DIM), dtype=np.float64,
	)
	result['oracle__physics_state'] = np.zeros(
		(BRANCHES, subject.HORIZON + 1, PHYSICS_DIM), dtype=np.float64,
	)
	result['oracle__physics_state'][1, 1:, 0] = 1.0
	result['oracle__physics_state'][2, 1:, 0] = -1.0
	result['oracle__gt_role_mask'] = np.zeros(
		(BRANCHES, subject.HORIZON + 1, ROLES, 64, 64), dtype=np.bool_,
	)
	result['oracle__gt_visible'] = np.zeros(
		(BRANCHES, subject.HORIZON + 1, ROLES), dtype=np.bool_,
	)
	result['oracle__absolute_step'] = np.arange(
		ANCHOR, ANCHOR + subject.HORIZON + 1, dtype=np.int64,
	)
	result['oracle__hard_official_state'] = np.array(
		result['oracle__official_state'], copy=True,
	)
	result['oracle__hard_gt_role_mask'] = np.array(
		result['oracle__gt_role_mask'], copy=True,
	)
	prefix_action = np.zeros((BRANCHES, ANCHOR, ACTION_DIM), dtype=np.float32)
	prefix_action[:, -(subject.HISTORY_FRAMES - 1):] = result['history__action']
	result['oracle__clean_prefix_action'] = prefix_action
	result['oracle__hard_prefix_action'] = np.array(prefix_action, copy=True)
	full = np.zeros(
		(BRANCHES, ANCHOR + subject.HORIZON + 1, PHYSICS_DIM),
		dtype=np.float64,
	)
	full[:, ANCHOR:] = result['oracle__physics_state']
	result['oracle__clean_full_physics_state'] = full
	result['oracle__hard_full_physics_state'] = np.array(full, copy=True)
	return result


def _artifact(path: Path, arrays: dict[str, np.ndarray]) -> str:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **arrays)
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, arrays: dict[str, np.ndarray]) -> Path:
	runtime = root / 'source' / 'runtime_config.json'
	checkpoint = root / 'source' / 'models' / 'final.pt'
	runtime.parent.mkdir(parents=True)
	checkpoint.parent.mkdir(parents=True)
	runtime.write_text('{}\n', encoding='utf-8')
	checkpoint.write_bytes(b'checkpoint')
	groups = []
	for root_id in range(3):
		path = root / 'groups' / f'root_{root_id:04d}.npz'
		sha = _artifact(path, arrays)
		hashes = subject.validate_group_arrays(
			arrays, role_count=ROLES, action_dim=ACTION_DIM,
			state_dim=STATE_DIM, physics_dim=PHYSICS_DIM,
			anchor_step=ANCHOR, branch_magnitude=MAGNITUDE,
		)
		groups.append({
			'root_id': root_id,
			'split': ('train', 'validation', 'test')[root_id],
			'anchor_step': ANCHOR,
			'relative_path': path.relative_to(root).as_posix(),
			'sha256': sha,
			'prefix_action_trace_sha256': 'a' * 64,
			'prefix_physics_trace_sha256': 'b' * 64,
			'continuation_action_trace_sha256': 'c' * 64,
			'clean_full_physics_trace_sha256': ['d' * 64] * BRANCHES,
			'hard_full_physics_trace_sha256': ['d' * 64] * BRANCHES,
			**hashes,
			'guards': {
				'fresh_reset_replay_only': True,
				'no_state_snapshot_restore': True,
				'all_sibling_prefix_physics_exact': True,
				'all_sibling_history_inputs_exact_within_condition': True,
				'clean_hard_actions_exact_per_branch': True,
				'clean_hard_physics_exact_per_branch': True,
				'clean_hard_official_state_exact_per_branch': True,
				'clean_hard_gt_exact_per_branch': True,
				'balanced_full_rank_real_actions': True,
				'immediate_physical_divergence': True,
				'one_root_one_split': True,
			},
		})
	payload = {
		'format': subject.FORMAT,
		'status': 'rof_same_state_action_branch_dataset_complete',
		'controller_training_authorized': False,
		'task': 'acrobot-swingup',
		'conditions': list(subject.CONDITIONS),
		'history_frames': subject.HISTORY_FRAMES,
		'horizon': subject.HORIZON,
		'policy_input_fields': [f'policy_{field}' for field in subject.POLICY_FIELDS],
		'role_names': ['whole_acrobot'],
		'role_count': ROLES,
		'action_dim': ACTION_DIM,
		'state_dim': STATE_DIM,
		'physics_dim': PHYSICS_DIM,
		'branch_magnitude': MAGNITUDE,
		'source': {
			'runtime_config': str(runtime.resolve()),
			'runtime_config_sha256': hashlib.sha256(runtime.read_bytes()).hexdigest(),
			'checkpoint': str(checkpoint.resolve()),
			'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
			'backend': 'robust_object_field',
		},
		'model_input_contract': {
			'schema': 'robust_object_field_v0_real_action_branch_v1',
			'keys': list(subject.MODEL_INPUT_KEYS),
			'oracle_namespace_excluded': True,
			'condition_id_excluded': True,
			'only_executed_actions': True,
		},
		'oracle_contract': {
			'namespace': subject.ORACLE_NAMESPACE,
			'keys': list(subject.ORACLE_KEYS),
			'never_model_input': True,
			'same_state_query_guard': 'exact_before_after',
		},
		'branch_protocol': {
			'history_frames': subject.HISTORY_FRAMES,
			'horizon': subject.HORIZON,
			'branch_action_index': subject.BRANCH_ACTION_INDEX,
			'prefix': 'fresh_reset_exact_open_loop_replay',
			'interventions': 'zero_plus_minus_each_scaled_action_basis',
			'continuation': 'identical_open_loop_actions_across_siblings',
			'fake_shuffled_action_futures': False,
			'background_pairing': 'independent_fresh_reset_replay_exact_physics',
			'root_grouping': 'all_siblings_and_background_twins_one_split_unit',
		},
		'root_splits': {'train': [0], 'validation': [1], 'test': [2]},
		'groups': groups,
	}
	manifest = root / subject.MANIFEST_NAME
	manifest.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
	return manifest


class ArrayContract(unittest.TestCase):
	def test_valid_real_branch_group(self):
		hashes = subject.validate_group_arrays(
			_arrays(), role_count=ROLES, action_dim=ACTION_DIM,
			state_dim=STATE_DIM, physics_dim=PHYSICS_DIM,
			anchor_step=ANCHOR, branch_magnitude=MAGNITUDE,
		)
		self.assertEqual(set(hashes), {
			'model_input_trace_sha256', 'oracle_trace_sha256',
			'anchor_physics_sha256', 'future_physics_sha256',
			'prefix_physics_trace_sha256',
			'prefix_action_trace_sha256',
			'continuation_action_trace_sha256',
			'clean_full_physics_trace_sha256',
			'hard_full_physics_trace_sha256',
		})

	def test_fake_or_unexecuted_branches_fail_closed(self):
		for mutation in ('duplicate_future', 'wrong_action', 'unequal_tail'):
			arrays = _arrays()
			if mutation == 'duplicate_future':
				arrays['oracle__physics_state'][1, 1] = arrays['oracle__physics_state'][0, 1]
			elif mutation == 'wrong_action':
				arrays['branch__action'][1, 0, 0] = 0.7
			else:
				arrays['branch__action'][1, 2, 0] = 0.7
			with self.assertRaises(ValueError, msg=mutation):
				subject.validate_group_arrays(
					arrays, role_count=ROLES, action_dim=ACTION_DIM,
					state_dim=STATE_DIM, physics_dim=PHYSICS_DIM,
					anchor_step=ANCHOR, branch_magnitude=MAGNITUDE,
				)

	def test_oracle_changes_do_not_enter_model_input_hash(self):
		arrays = _arrays()
		model_hash = subject.model_input_trace_sha256(arrays)
		oracle_hash = subject.oracle_trace_sha256(arrays)
		changed = deepcopy(arrays)
		changed['oracle__official_state'][1, 2, 0] = 123.0
		self.assertEqual(subject.model_input_trace_sha256(changed), model_hash)
		self.assertNotEqual(subject.oracle_trace_sha256(changed), oracle_hash)

	def test_extra_state_like_model_array_is_rejected(self):
		arrays = _arrays()
		arrays['policy_state'] = np.zeros((3, 1), dtype=np.float32)
		with self.assertRaises(ValueError):
			subject.validate_group_arrays(
				arrays, role_count=ROLES, action_dim=ACTION_DIM,
				state_dim=STATE_DIM, physics_dim=PHYSICS_DIM,
				anchor_step=ANCHOR, branch_magnitude=MAGNITUDE,
			)

	def test_hard_twin_and_prefix_evidence_are_revalidated(self):
		for mutation in (
			'hard_physics', 'hard_state', 'hard_gt', 'hard_action',
			'sibling_action', 'sibling_prefix',
		):
			arrays = _arrays()
			if mutation == 'hard_physics':
				arrays['oracle__hard_full_physics_state'][0, -1, 0] += 1.0
			elif mutation == 'hard_state':
				arrays['oracle__hard_official_state'][0, 1, 0] += 1.0
			elif mutation == 'hard_gt':
				arrays['oracle__hard_gt_role_mask'][0, 1, 0, 0, 0] = True
			elif mutation == 'hard_action':
				arrays['oracle__hard_prefix_action'][0, 0, 0] += 0.1
			elif mutation == 'sibling_action':
				arrays['oracle__clean_prefix_action'][1, 0, 0] += 0.1
			else:
				arrays['oracle__clean_full_physics_state'][1, 1, 0] += 1.0
			with self.assertRaises(ValueError, msg=mutation):
				subject.validate_group_arrays(
					arrays, role_count=ROLES, action_dim=ACTION_DIM,
					state_dim=STATE_DIM, physics_dim=PHYSICS_DIM,
					anchor_step=ANCHOR, branch_magnitude=MAGNITUDE,
				)


class ManifestContract(unittest.TestCase):
	def test_root_grouping_and_identity_validate(self):
		with tempfile.TemporaryDirectory() as temporary:
			manifest = _manifest(Path(temporary), _arrays())
			payload = subject.validate_dataset(manifest)
			self.assertEqual(payload['root_splits']['test'], [2])

	def test_fail_open_or_shuffled_protocol_is_rejected(self):
		for mutation in ('authorize', 'shuffle', 'split_twin', 'guard'):
			with tempfile.TemporaryDirectory() as temporary:
				manifest = _manifest(Path(temporary), _arrays())
				payload = json.loads(manifest.read_text(encoding='utf-8'))
				if mutation == 'authorize':
					payload['controller_training_authorized'] = True
				elif mutation == 'shuffle':
					payload['branch_protocol']['fake_shuffled_action_futures'] = True
				elif mutation == 'split_twin':
					payload['root_splits']['train'].append(1)
				else:
					payload['groups'][0]['guards']['clean_hard_physics_exact_per_branch'] = False
				manifest.write_text(json.dumps(payload) + '\n', encoding='utf-8')
				with self.assertRaises((ValueError, FileNotFoundError), msg=mutation):
					subject.validate_dataset(manifest)


if __name__ == '__main__':
	unittest.main()
