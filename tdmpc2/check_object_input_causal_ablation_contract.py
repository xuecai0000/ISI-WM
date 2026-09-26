"""Dependency-light contract tests for the frozen object-input causal ablation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from tdmpc2.tools import evaluate_object_input_causal_ablation as subject


class NumpyTorch:
	float32 = np.float32

	@staticmethod
	def isfinite(value):
		return np.isfinite(value)

	@staticmethod
	def count_nonzero(value):
		return np.count_nonzero(value)

	@staticmethod
	def equal(left, right):
		return np.array_equal(left, right)

	def __init__(self, payload=None):
		self.payload = payload

	def load(self, path, map_location=None, weights_only=None):
		return self.payload


torch = NumpyTorch()


class CloneableObservation(dict):
	def clone(self):
		return CloneableObservation({key: value.copy() for key, value in self.items()})


def runtime(task: str) -> dict:
	spec = subject.SUPPORTED[task]
	return {
		'task': task,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_role_names': list(spec['roles']),
		'cutie_object_num_roles': spec['role_count'],
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_role_dim': 64,
		'cutie_object_only_latent_dim': spec['latent_dim'],
		'latent_dim': spec['latent_dim'],
		'cutie_object_regression_encoder': spec['regression_encoder'][0],
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_support_schema': 'generic_indexed_v1',
		'cutie_object_allow_simulator_support': True,
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
		'cutie_object_true_entity_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'object_state_supervision_enabled': False,
		'object_state_supervision_collect_labels': False,
		'object_state_bottleneck_enabled': False,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'obs_shape': {'object': [spec['role_count'], 1770]},
	}


def observation(task: str):
	roles = subject.SUPPORTED[task]['role_count']
	value = np.arange(roles * 1770, dtype=np.float32).reshape(roles, 1770)
	return CloneableObservation({'object': value + 1.0})


class ObjectInputCausalAblationContract(unittest.TestCase):

	def test_supported_contracts_are_exact_k2_and_k3(self):
		finger = subject._validate_source(runtime('finger-spin'), 'finger-spin')
		walker = subject._validate_source(runtime('walker-stand'), 'walker-stand')
		self.assertEqual(finger['role_count'], 2)
		self.assertEqual(finger['latent_dim'], 128)
		self.assertEqual(walker['role_count'], 3)
		self.assertEqual(walker['latent_dim'], 192)
		bad = runtime('walker-stand')
		bad['object_state_supervision_enabled'] = True
		with self.assertRaisesRegex(ValueError, 'non-causal source'):
			subject._validate_source(bad, 'walker-stand')

	def test_query_ablation_preserves_geometry_status_and_source(self):
		obs = observation('finger-spin')
		before = obs['object'].copy()
		diagnostics = subject._new_diagnostics('finger-spin', 'query_appearance_zero')
		result = subject._apply_intervention(
			obs, task='finger-spin', mode='query_appearance_zero',
			episode_index=0, step_index=0, torch=torch, diagnostics=diagnostics,
		)
		self.assertIsInstance(result, CloneableObservation)
		policy = result['object'].reshape(2, 3, 590)
		raw = before.reshape(2, 3, 590)
		self.assertEqual(int(np.count_nonzero(policy[..., :512])), 0)
		self.assertTrue(np.array_equal(policy[..., 512:], raw[..., 512:]))
		self.assertTrue(np.array_equal(obs['object'], before))
		self.assertNotEqual(
			result['object'].__array_interface__['data'][0],
			obs['object'].__array_interface__['data'][0],
		)

	def test_k3_role_drop_is_exact_and_non_target_preserving(self):
		obs = observation('walker-stand')
		before = obs['object'].copy()
		diagnostics = subject._new_diagnostics('walker-stand', 'target_role_drop_20')
		inside = subject._apply_intervention(
			obs, task='walker-stand', mode='target_role_drop_20',
			episode_index=0, step_index=75, torch=torch, diagnostics=diagnostics,
		)['object']
		self.assertEqual(int(np.count_nonzero(inside[0])), 0)
		self.assertTrue(np.array_equal(inside[1:], before[1:]))
		self.assertTrue(np.array_equal(obs['object'], before))
		outside = subject._apply_intervention(
			obs, task='walker-stand', mode='target_role_drop_20',
			episode_index=0, step_index=74, torch=torch, diagnostics=diagnostics,
		)['object']
		self.assertTrue(np.array_equal(outside, before))

	def test_frozen_burst_grid_has_exact_20_and_50_frames(self):
		for mode, length in (
			('target_role_drop_20', 20), ('target_role_drop_50', 50),
		):
			count = 0
			for episode, start in enumerate(subject.BURST_STARTS):
				for step in range(subject.DECISION_STEPS):
					count += int(start <= step < start + length)
			self.assertEqual(count, subject.EPISODES * length)
		self.assertEqual(len(subject.BURST_STARTS), subject.EPISODES)

	def test_pairing_and_paired_return_statistics(self):
		def rows(offset):
			return [{
				'episode_index': index,
				'planner_seed': 100 + index,
				'planner_rng_start_sha256': 'a' * 64,
				'initial_rgb_sha256': 'b' * 64,
				'initial_raw_object_sha256': 'c' * 64,
				'background_source': f'video{index}.mp4',
				'background_start_frame_index': index,
				'comparison_prefix_decisions': subject.BURST_STARTS[index],
				'pre_intervention_raw_object_trace_sha256': 'd' * 64,
				'pre_intervention_policy_input_trace_sha256': 'e' * 64,
				'pre_intervention_action_trace_sha256': 'f' * 64,
				'length': 500,
				'reward': float(index + offset),
			} for index in range(20)]
		full, ablated = rows(10), rows(3)
		self.assertTrue(all(subject._pairing_checks(
			full, ablated, mode='target_role_drop_20'
		).values()))
		stats = subject._paired_statistics(full, ablated, seed=7)
		self.assertEqual(stats['mean'], 7.0)
		self.assertEqual(stats['bootstrap_95_ci'], [7.0, 7.0])

	def test_checkpoint_contract_matches_runtime_cardinality(self):
		for task in subject.SUPPORTED:
			raw = runtime(task)
			spec = subject.SUPPORTED[task]
			contract = {
				'format': 'tdmpc2_checkpoint_contract_v1',
				'flat_anchor_mode': 'cutie_object_only',
				'latent_dim': spec['latent_dim'],
				'cutie_object_belief_enabled': False,
				'cutie_object_observation': {
					'variant': 'full',
					'privileged_runtime_segmentation': False,
					'num_roles': spec['role_count'],
					'frame_dim': 590,
					'stack_frames': 3,
					'input_dim': 1770,
					'spatial_token_enabled': False,
				},
				'cutie_object_auxiliary': {'target': 'full_descriptor'},
			}
			if raw['cutie_object_regression_encoder'] is not None:
				contract['cutie_object_regression_encoder'] = raw[
					'cutie_object_regression_encoder'
				]
			with tempfile.TemporaryDirectory() as temporary:
				path = Path(temporary) / 'final.pt'
				path.write_bytes(b'synthetic-checkpoint')
				fake_torch = NumpyTorch({
					'model': {'weight': np.ones(1, dtype=np.float32)},
					'checkpoint_contract': contract,
				})
				report = subject._load_checkpoint_contract(
					path, raw, task, fake_torch
				)
				self.assertTrue(all(report['checks'].values()))

	def test_offline_support_is_explicit_but_runtime_privilege_is_disabled(self):
		raw = runtime('finger-spin')
		payload = {
			'format': 'cutie_indexed_mask_support_v1',
			'roles': ['finger', 'spinner'],
			'collection': {
				'task': 'finger-spin',
				'split': 'support',
				'observation': 'rgb',
				'support_schema': 'generic_indexed_v1',
				'label_policy': 'simulator_segmentation_support_only',
				'diagnostic_support': True,
			},
			'records': [],
		}
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / 'annotations.json'
			path.write_text(json.dumps(payload), encoding='utf-8')
			report = subject._support_provenance(raw, path, 'finger-spin')
			self.assertEqual(
				report['label_policy'], 'simulator_segmentation_support_only'
			)
			self.assertFalse(report['evaluation_runtime_simulator_segmentation'])
			payload['collection']['task'] = 'walker-stand'
			path.write_text(json.dumps(payload), encoding='utf-8')
			with self.assertRaisesRegex(ValueError, 'support provenance mismatch'):
				subject._support_provenance(raw, path, 'finger-spin')

	def test_cli_freezes_both_conditions_and_all_modes(self):
		args = subject.parse_args([
			'--task', 'finger-spin', '--runtime-config', 'runtime.json',
			'--checkpoint', 'models/final.pt', '--validate-only',
			'--output', 'summary.json',
		])
		self.assertEqual(tuple(args.conditions), ('clean', 'hard'))
		self.assertTrue(args.validate_only)
		self.assertEqual(subject.MODES, (
			'full', 'query_appearance_zero',
			'target_role_drop_20', 'target_role_drop_50',
		))


if __name__ == '__main__':
	unittest.main(verbosity=2)
