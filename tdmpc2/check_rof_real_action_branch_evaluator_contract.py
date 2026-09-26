"""Dependency-light contracts for the real-action branch evaluator."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tdmpc2.tools import evaluate_rof_real_action_branches as target
from tdmpc2 import check_rof_real_action_branch_contract as collector_fixture


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(*, roles: int = 1, action_dim: int = 1) -> dict[str, np.ndarray]:
	branches = 1 + 2 * action_dim
	history_action = np.zeros((2, action_dim), dtype=np.float32)
	branch_action = np.zeros((branches, 5, action_dim), dtype=np.float32)
	code = np.zeros(branches, dtype=np.int32)
	is_zero = np.zeros(branches, dtype=np.bool_)
	is_zero[0] = True
	row = 1
	for dimension in range(action_dim):
		branch_action[row, 0, dimension] = 1.0
		code[row] = dimension + 1
		row += 1
		branch_action[row, 0, dimension] = -1.0
		code[row] = -(dimension + 1)
		row += 1
	# A shared post-intervention continuation is allowed and must be identical.
	branch_action[:, 1:] = 0.125
	result = {
		'history__action': history_action, 'branch__action': branch_action,
		'branch__code': code, 'branch__is_zero': is_zero,
	}
	for condition in target.CONDITIONS:
		result[f'{condition}__history__policy_rgb'] = np.zeros(
			(3, 9, 64, 64), dtype=np.uint8,
		)
		result[f'{condition}__history__policy_object'] = np.zeros(
			(3, roles, 1770), dtype=np.float32,
		)
		result[f'{condition}__history__policy_object_mask'] = np.ones(
			(3, roles, 3, 64, 64), dtype=np.bool_,
		)
		result[f'{condition}__history__policy_role_exists'] = np.ones(
			(3, roles), dtype=np.float32,
		)
		result[f'{condition}__future__policy_rgb'] = np.zeros(
			(branches, 5, 9, 64, 64), dtype=np.uint8,
		)
		result[f'{condition}__future__policy_object'] = np.zeros(
			(branches, 5, roles, 1770), dtype=np.float32,
		)
		result[f'{condition}__future__policy_object_mask'] = np.ones(
			(branches, 5, roles, 3, 64, 64), dtype=np.bool_,
		)
		result[f'{condition}__future__policy_role_exists'] = np.ones(
			(branches, 5, roles), dtype=np.float32,
		)
	official = np.zeros((branches, 6, 2), dtype=np.float64)
	physics = np.zeros((branches, 6, 3), dtype=np.float64)
	for branch in range(1, branches):
		effect = branch_action[branch, 0].sum()
		official[branch, 1:, 0] = effect
		physics[branch, 1:, 0] = effect
	result['oracle__official_state'] = official
	result['oracle__physics_state'] = physics
	gt = np.zeros((branches, 6, roles, 64, 64), dtype=np.bool_)
	for role in range(roles):
		gt[:, :, role, role, role] = True
	result['oracle__gt_role_mask'] = gt
	result['oracle__gt_visible'] = gt.reshape(branches, 6, roles, -1).any(-1)
	anchor = 80
	result['oracle__absolute_step'] = np.arange(anchor, anchor + 6, dtype=np.int64)
	result['oracle__hard_official_state'] = np.array(official, copy=True)
	result['oracle__hard_gt_role_mask'] = np.array(gt, copy=True)
	prefix_action = np.zeros((branches, anchor, action_dim), dtype=np.float32)
	prefix_action[:, -2:] = history_action
	result['oracle__clean_prefix_action'] = prefix_action
	result['oracle__hard_prefix_action'] = np.array(prefix_action, copy=True)
	full = np.zeros((branches, anchor + 6, 3), dtype=np.float64)
	full[:, anchor:] = physics
	result['oracle__clean_full_physics_state'] = full
	result['oracle__hard_full_physics_state'] = np.array(full, copy=True)
	return result


class RootArchiveContractTests(unittest.TestCase):
	def test_exact_archive_accepts_real_balanced_branches(self):
		audit = target.validate_root_arrays(
			_arrays(action_dim=2), role_count=1, action_dim=2, root_id=7,
		)
		self.assertEqual(audit['branches'], 5)
		self.assertEqual(audit['action_design_rank'], 2)
		self.assertGreater(audit['min_immediate_physics_effect_rms'], 0.0)

	def test_reward_or_extra_key_is_rejected(self):
		arrays = _arrays()
		arrays['reward'] = np.zeros(5, dtype=np.float32)
		with self.assertRaisesRegex(ValueError, 'exact schema'):
			target.validate_root_arrays(
				arrays, role_count=1, action_dim=1, root_id=0,
			)

	def test_unbalanced_or_fake_branch_is_rejected(self):
		arrays = _arrays()
		arrays['branch__action'][2, 0, 0] = -0.5
		with self.assertRaisesRegex(ValueError, 'balanced negative'):
			target.validate_root_arrays(
				arrays, role_count=1, action_dim=1, root_id=0,
			)

	def test_root_anchor_must_be_exactly_shared(self):
		arrays = _arrays()
		arrays['oracle__physics_state'][1, 0, 0] = 1e-12
		with self.assertRaisesRegex(ValueError, 'anchor slice|same root physics'):
			target.validate_root_arrays(
				arrays, role_count=1, action_dim=1, root_id=0,
			)

	def test_stored_hard_twin_and_prefix_are_not_self_asserted(self):
		for key, index in (
			('oracle__hard_full_physics_state', (0, -1, 0)),
			('oracle__hard_official_state', (0, 1, 0)),
			('oracle__hard_gt_role_mask', (0, 1, 0, 1, 1)),
			('oracle__hard_prefix_action', (0, 0, 0)),
			('oracle__clean_prefix_action', (1, 0, 0)),
			('oracle__clean_full_physics_state', (1, 1, 0)),
		):
			arrays = _arrays()
			arrays[key][index] = arrays[key][index] + 1
			with self.assertRaises(ValueError, msg=key):
				target.validate_root_arrays(
					arrays, role_count=1, action_dim=1, root_id=0,
				)


class DatasetContractTests(unittest.TestCase):
	def test_loader_consumes_collector_contract_manifest(self):
		with tempfile.TemporaryDirectory() as raw:
			manifest = collector_fixture._manifest(
				Path(raw), collector_fixture._arrays(),
			)
			payload = json.loads(manifest.read_text(encoding='utf-8'))
			payload['source']['checkpoint_step'] = 30_000
			manifest.write_text(json.dumps(payload), encoding='utf-8')
			dataset = target.load_branch_dataset(
				manifest, require_collector_contract=True,
			)
			self.assertEqual(dataset.task, 'acrobot-swingup')
			self.assertEqual(dataset.root_splits['test'], (2,))

	def test_manifest_splits_complete_roots_only(self):
		with tempfile.TemporaryDirectory() as raw:
			root = Path(raw)
			records = []
			split_ids = {'train': [0], 'validation': [1], 'test': [2, 3, 4, 5]}
			for root_id in range(6):
				path = root / f'root_{root_id}.npz'
				np.savez_compressed(path, **_arrays())
				split = next(name for name, values in split_ids.items() if root_id in values)
				records.append({
					'root_id': root_id, 'split': split,
					'relative_path': path.name, 'sha256': _sha256(path),
				})
			manifest = {
				'format': target.DATASET_FORMAT,
				'status': 'rof_same_state_action_branch_dataset_complete',
				'controller_training_authorized': False,
				'conditions': list(target.CONDITIONS),
				'task': 'synthetic-task',
				'role_names': ['object'], 'action_dim': 1,
				'history_frames': 3, 'horizon': 5,
				'policy_input_fields': [
					'policy_rgb', 'policy_object', 'policy_object_mask',
					'policy_role_exists',
				],
				'model_input_contract': {
					'keys': list(target.collector.MODEL_INPUT_KEYS),
					'oracle_namespace_excluded': True,
					'condition_id_excluded': True,
					'only_executed_actions': True,
				},
				'oracle_contract': {
					'namespace': 'oracle__', 'keys': list(target.ORACLE_FIELDS),
					'never_model_input': True,
					'same_state_query_guard': 'exact_before_after',
				},
				'branch_protocol': {
					'root_grouping': (
						'all_siblings_and_background_twins_one_split_unit'
					),
					'fake_shuffled_action_futures': False,
					'interventions': 'zero_plus_minus_each_scaled_action_basis',
					'continuation': 'identical_open_loop_actions_across_siblings',
				},
				'root_splits': split_ids, 'groups': records,
			}
			manifest_path = root / 'dataset_manifest.json'
			manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
			dataset = target.load_branch_dataset(manifest_path)
			self.assertEqual([group.root_id for group in dataset.split('test')], [2, 3, 4, 5])
			manifest['root_splits']['train'].append(2)
			manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
			with self.assertRaisesRegex(ValueError, 'overlap'):
				target.load_branch_dataset(manifest_path)


class MetricContractTests(unittest.TestCase):
	def test_perfect_real_branch_ranking_and_effect(self):
		error = np.asarray([
			[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0],
		])
		self.assertEqual(target._ranking_auc(error), 1.0)
		true = np.asarray([[1.0, -1.0], [2.0, -2.0]])
		self.assertEqual(target._zero_effect_r2([(true, true.copy())]), 1.0)

	def test_bootstrap_is_root_clustered_and_deterministic(self):
		first = target._root_bootstrap_mean(
			[0.0, 1.0, 2.0, 3.0], seed=17, resamples=1000,
		)
		second = target._root_bootstrap_mean(
			[0.0, 1.0, 2.0, 3.0], seed=17, resamples=1000,
		)
		self.assertEqual(first, second)
		self.assertEqual(first['unit'], 'whole_root_sibling_and_twin_group')

	def test_nonlinear_capacity_pair_is_same_architecture_and_actionless(self):
		try:
			import torch
		except ImportError:
			self.skipTest('Torch is not installed in the dependency-light test runtime.')
		normalization = target.normalized.DeltaNormalization(
			u_mean=np.zeros(8, dtype=np.float32),
			u_scale=np.ones(8, dtype=np.float32),
			action_mean=np.zeros(1, dtype=np.float32),
			action_scale=np.ones(1, dtype=np.float32),
			delta_mean=np.zeros(8, dtype=np.float32),
			delta_scale=np.ones(8, dtype=np.float32),
			horizon_scale={h: np.ones(8, dtype=np.float32) for h in target.HORIZONS},
			simnorm_dim=8, floor_audit={},
		)
		torch.manual_seed(7)
		aware = target._nonlinear_model_factory(
			'nonlinear_action_aware', normalization, hidden_dim=16,
		)
		torch.manual_seed(7)
		actionless = target._nonlinear_model_factory(
			'nonlinear_actionless', normalization, hidden_dim=16,
		)
		self.assertEqual(
			sum(value.numel() for value in aware.parameters()),
			sum(value.numel() for value in actionless.parameters()),
		)
		history = torch.zeros((2, 3, 8), dtype=torch.float32)
		positive = torch.ones((2, 3, 1), dtype=torch.float32)
		negative = -positive
		with torch.no_grad():
			actionless_positive = actionless.step(history, positive)[0]
			actionless_negative = actionless.step(history, negative)[0]
			aware_positive = aware.step(history, positive)[0]
			aware_negative = aware.step(history, negative)[0]
		self.assertTrue(torch.equal(actionless_positive, actionless_negative))
		self.assertFalse(torch.equal(aware_positive, aware_negative))


class IsolationContractTests(unittest.TestCase):
	def _result(self):
		train_ids = list(range(50))
		validation_ids = list(range(50, 60))
		test_ids = list(range(60, 80))
		fit_seed = target.FORMAL_FIT_SEEDS[0]
		bootstrap_seed = fit_seed + target.FORMAL_BOOTSTRAP_SEED_OFFSET
		fit = {
			'train_root_ids': train_ids, 'validation_root_ids': validation_ids,
			'test_roots_seen_during_fit_or_selection': False,
			'parameter_count': 10,
		}
		def bootstrap(seed):
			return {
				'unit': 'whole_root_sibling_and_twin_group',
				'seed': seed, 'resamples': target.FORMAL_BOOTSTRAP_RESAMPLES,
				'root_count': len(test_ids),
			}
		condition_metrics = {}
		for condition in target.CONDITIONS:
			condition_seed = target.identifiable._stable_seed(
				bootstrap_seed, condition,
			)
			horizons = {}
			for horizon in target.HORIZONS:
				row = {
					'root_rows': [{'root_id': root_id} for root_id in test_ids],
				}
				for key, seed_part in (
					('aware_error', 'aware'),
					('aware_relative_gain_vs_persistence', 'persistence'),
					('aware_relative_gain_vs_actionless', 'actionless'),
					('real_sibling_action_ranking_auc', 'ranking'),
					('action_effect_r2_vs_zero_sibling', 'effect_r2'),
				):
					row[key] = bootstrap(target.identifiable._stable_seed(
						condition_seed, horizon, seed_part,
					))
				capacity = {'diagnostic_only': True}
				for key, seed_part in (
					('aware_error', 'nonlinear_aware_error'),
					('aware_relative_gain_vs_parameter_matched_actionless',
					 'nonlinear_actionless'),
					('real_sibling_action_ranking_auc', 'nonlinear_ranking'),
					('action_effect_r2_vs_zero_sibling', 'nonlinear_effect_r2'),
				):
					capacity[key] = bootstrap(target.identifiable._stable_seed(
						condition_seed, horizon, seed_part,
					))
				row['nonlinear_capacity_control'] = capacity
				horizons[str(horizon)] = row
			condition_metrics[condition] = {
				'horizons': horizons,
				'packet_validity': {'valid_rate': bootstrap(
					target.identifiable._stable_seed(condition_seed, 'validity')
				)},
			}
		return {
			'format': target.RESULT_FORMAT, 'status': target.RESULT_STATUS,
			'engineering_pass': True, 'policy_training_performed': False,
			'controller_training_authorized': False,
			'privileged_fields_used_as_model_input': False,
			'reward_or_return_used': False,
			'background_id_used_as_model_input': False,
			'protocol': {
				'split_unit': 'whole_root_sibling_and_twin_group',
				'bootstrap_unit': 'whole_root_sibling_and_twin_group',
				'real_interventions_only': True,
				'shuffled_logged_futures_used': False,
				'fit_domain': (
					'pooled_clean_hard_background_twins_without_condition_identifier'
				),
				'condition_or_background_identifier_used_for_fit': False,
				'seed': fit_seed, 'bootstrap_seed': bootstrap_seed,
				'bootstrap_resamples': target.FORMAL_BOOTSTRAP_RESAMPLES,
				'hidden_dim': target.FORMAL_HIDDEN_DIM,
				'formal_split_counts': dict(target.FORMAL_SPLIT_COUNTS),
				'fit_models': list(target.FIT_MODES),
			},
			'oracle_data_validity': {
				'passed': True,
				'oracle_fields_used_for_model_fit_selection_or_scoring': False,
			},
			'training': {
				mode: copy.deepcopy(fit) for mode in target.FIT_MODES
			},
			'root_splits': {
				'train': train_ids, 'validation': validation_ids, 'test': test_ids,
			},
			'condition_metrics': condition_metrics,
			'background_twin_metrics': {'horizons': {
				str(horizon): {
					'background_over_real_action_distance': bootstrap(
						target.identifiable._stable_seed(
							target.identifiable._stable_seed(
								bootstrap_seed, 'twins',
							), horizon, 'twin_ratio',
						)
					),
				} for horizon in target.HORIZONS
			}},
			'action_identifiable_representation_candidate': False,
			'gates': {
				'controller_training_authorized': False,
				'action_identifiable_representation_candidate': False,
				'nonlinear_capacity_control': {'diagnostic_only': True},
			},
		}

	def test_result_never_authorizes_controller_training(self):
		payload = self._result()
		target.validate_result(payload)
		payload['controller_training_authorized'] = True
		with self.assertRaisesRegex(ValueError, 'never authorize'):
			target.validate_result(payload)

	def test_oracle_boundary_fails_closed(self):
		payload = self._result()
		payload['oracle_data_validity'][
			'oracle_fields_used_for_model_fit_selection_or_scoring'
		] = True
		with self.assertRaisesRegex(ValueError, 'escaped'):
			target.validate_result(payload)

	def test_encoder_firewall_is_explicit_allow_list(self):
		source = inspect.getsource(target._model_packet).lower()
		self.assertIn('policy_fields', source)
		for token in ('oracle__', 'reward', 'background'):
			self.assertNotIn(token, source)


if __name__ == '__main__':
	unittest.main(verbosity=2)
