"""CPU-only contracts for long-budget checkpoint selection and aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools import aggregate_acrobot_factorized_proprio_500k as aggregate
from tools import evaluate_cutie_multitask_checkpoint as evaluator


class LongBudgetContractTest(unittest.TestCase):
	def setUp(self):
		self.temporary = tempfile.TemporaryDirectory(prefix='acrobot_500k_contract_')
		self.root = Path(self.temporary.name)

	def tearDown(self):
		self.temporary.cleanup()

	def _training(self, mode):
		root = self.root / mode
		(root / 'models').mkdir(parents=True)
		config = {
			'task': 'acrobot-swingup', 'obs': 'rgb', 'seed': 10,
			'steps': 500000, 'eval_freq': 25000, 'eval_episodes': 10,
			'save_eval_episode_trace': True, 'save_eval_checkpoints': True,
			'video_background_enabled': True, 'video_background_split': 'train',
			'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
			'cutie_object_observation_variant': 'cutie_proprio',
			'cutie_object_frame_schema': 'cutie_query_mask_status_plus_proprio_v1',
			'cutie_object_role_names': ['upper_arm', 'lower_arm'],
			'cutie_object_support_schema': 'generic_indexed_v1',
			'cutie_object_num_roles': 2, 'cutie_object_frame_dim': 1774,
			'cutie_object_stack_frames': 1, 'cutie_object_input_dim': 1774,
			'cutie_object_only_latent_dim': 128,
			'cutie_object_allow_simulator_runtime': False,
			'cutie_object_allow_simulator_kinematics_runtime': True,
			'cutie_proprio_mode': mode, 'visual_pose_checkpoint': None,
			'obs_shape': {'object': [2, 1774]}, 'latent_dim': 128,
		}
		(root / 'runtime_config.json').write_text(
			json.dumps(config), encoding='utf-8'
		)
		curve = []
		for step in aggregate.EXPECTED_STEPS:
			# Tie at 50k/75k verifies deterministic earliest-step selection.
			reward = 10.0 if step in {50000, 75000} else step / 500000
			curve.append((step, reward))
			if step:
				(root / 'models' / f'eval_{step}.pt').write_bytes(
					f'{mode}:{step}'.encode('ascii')
				)
		(root / 'models' / 'final.pt').write_bytes(f'{mode}:final'.encode('ascii'))
		with (root / 'eval.csv').open('w', newline='', encoding='utf-8') as stream:
			writer = csv.DictWriter(stream, fieldnames=['step', 'episode_reward'])
			writer.writeheader()
			for step, reward in curve:
				writer.writerow({'step': step, 'episode_reward': reward})
		with (root / 'eval_episodes.jsonl').open('w', encoding='utf-8') as stream:
			for index, (step, reward) in enumerate(curve):
				stream.write(json.dumps({
					'evaluation_index': index, 'step': step, 'episodes': 10,
					'episode_rewards': [reward] * 10,
					'reward_mean': reward, 'reward_std': 0.0,
				}) + '\n')
		(root / 'trainer_runtime.json').write_text(
			json.dumps({'steps': 500001}), encoding='utf-8'
		)
		return root

	def test_training_contract_selects_earliest_best_checkpoint(self):
		root = self._training('factorized')
		result = aggregate._validate_training(root, 'factorized')
		self.assertEqual(result['best_step'], 50000)
		self.assertEqual(len(result['periodic_checkpoints']), 20)
		self.assertLess(result['final_minus_peak'], 0)

	def test_evaluator_accepts_only_exact_periodic_checkpoint(self):
		root = self._training('factorized')
		output = self.root / 'evaluation.json'
		args = SimpleNamespace(
			episodes=20, erosion_pixels=0, condition='clean',
			env_seed=1, background_seed=2, planner_seed_base=3,
			output=output, runtime_config=root / 'runtime_config.json',
			checkpoint=root / 'models' / 'eval_50000.pt', checkpoint_step=50000,
			expected_training_steps=500000,
			expected_training_eval_freq=25000,
		)
		evaluator._validate_args(args)
		args.checkpoint_step = 50001
		with self.assertRaisesRegex(ValueError, 'periodic evaluation'):
			evaluator._validate_args(args)

	def test_held_out_contract_records_selection_without_test_leakage(self):
		root = self._training('factorized_proprio_only')
		path = self.root / 'held_out.json'
		path.write_text(json.dumps({
			'task': 'acrobot-swingup',
			'backend': 'cutie_proprio_factorized_proprio_only',
			'condition': 'hard',
			'episodes': [{'length': 500} for _ in range(20)],
			'provenance': {
				'checkpoint_kind': 'periodic_eval', 'checkpoint_step': 50000,
				'runtime_config': str((root / 'runtime_config.json').resolve()),
			},
			'summary': {
				'reward_mean': 100.0, 'reward_median': 90.0,
				'reward_std': 20.0, 'reward_min': 50.0, 'reward_max': 140.0,
			},
			'perception_runtime': {'frames': 10020},
		}), encoding='utf-8')
		result = aggregate._validate_evaluation(
			path, 'factorized_proprio_only', 'hard', 'best', 50000, root
		)
		self.assertEqual(result['checkpoint_step'], 50000)
		self.assertEqual(result['reward_mean'], 100.0)

	def _evaluation(self, root, mode, selection, condition):
		path = self.root / f'{mode}_{selection}_{condition}.json'
		step = 50000 if selection == 'best' else 500000
		path.write_text(json.dumps({
			'task': 'acrobot-swingup', 'backend': aggregate.BACKEND[mode],
			'condition': condition,
			'episodes': [{'length': 500} for _ in range(20)],
			'provenance': {
				'checkpoint_kind': (
					'periodic_eval' if selection == 'best' else 'final'
				),
				'checkpoint_step': step,
				'runtime_config': str((root / 'runtime_config.json').resolve()),
			},
			'summary': {
				'reward_mean': 100.0, 'reward_median': 90.0,
				'reward_std': 20.0, 'reward_min': 50.0, 'reward_max': 140.0,
			},
			'perception_runtime': {'frames': 10020},
		}), encoding='utf-8')
		return path

	def test_full_aggregation_cli_contract(self):
		argv = ['aggregate_acrobot_factorized_proprio_500k.py']
		for mode in aggregate.MODES:
			root = self._training(mode)
			prefix = mode.replace('_', '-')
			argv.extend([f'--{prefix}-root', str(root)])
			for selection in ('best', 'final'):
				for condition in ('clean', 'hard'):
					path = self._evaluation(root, mode, selection, condition)
					argv.extend([
						f'--{prefix}-{selection}-{condition}', str(path)
					])
		output = self.root / 'summary.json'
		argv.extend(['--output', str(output)])
		with mock.patch('sys.argv', argv):
			aggregate.main()
		payload = json.loads(output.read_text(encoding='utf-8'))
		self.assertEqual(payload['status'], 'acrobot_factorized_proprio_500k_complete')
		self.assertFalse(payload['paper_claim_authorized'])
		self.assertEqual(set(payload['runs']), set(aggregate.MODES))


if __name__ == '__main__':
	unittest.main(verbosity=2)
