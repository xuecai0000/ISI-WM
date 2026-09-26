"""Dependency-light contracts for the frozen-latent transition refit."""

from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


def _subject():
	return importlib.import_module('tdmpc2.tools.evaluate_rof_transition_refit')


def _latents_actions():
	latents = {
		0: np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32),
		1: np.asarray([[10.0], [14.0], [19.0]], dtype=np.float32),
		2: np.asarray([[1000.0], [2000.0]], dtype=np.float32),
	}
	actions = {
		0: np.asarray([[1.0], [2.0]], dtype=np.float32),
		1: np.asarray([[4.0], [5.0]], dtype=np.float32),
		2: np.asarray([[999.0]], dtype=np.float32),
	}
	return latents, actions


class _Predictor:
	def __init__(self, scale=1.0):
		self.scale = float(scale)

	def predict(self, z, action):
		return np.asarray(z) + self.scale * np.asarray(action)


class _Episode:
	def __init__(self, episode_id, action):
		self.episode_id = episode_id
		self.arrays = {'action': np.asarray(action, dtype=np.float32)}


class _Dataset:
	def __init__(self, condition, episodes):
		self.manifest = {'condition': condition}
		self._episodes = tuple(episodes)

	def split(self, name):
		if name != 'test':
			raise AssertionError(name)
		return self._episodes


def _minimal_method(episodes):
	return {'by_episode': [{'episode_id': value} for value in episodes]}


def _minimal_condition(subject, episodes=(8, 9)):
	methods = {name: _minimal_method(episodes) for name in subject.METHODS}
	return {
		'test_episode_ids': list(episodes),
		'horizons': {
			str(horizon): {'methods': deepcopy(methods)}
			for horizon in subject.HORIZONS
		},
	}


def _minimal_result(subject):
	condition = _minimal_condition(subject)
	return {
		'format': subject.FORMAT,
		'status': 'rof_transition_refit_complete',
		'engineering_pass': True,
		'scientific_complete': True,
		'controller_training_authorized': False,
		'policy_training_performed': False,
		'protocol': {
			'fit_conditions': ['clean', 'hard'],
			'selection_split': 'same_domain_validation_episodes',
			'privileged_labels_used': False,
		},
		'training': {
			fit: {name: {} for name in subject.ARCHITECTURES}
			for fit in ('clean', 'hard')
		},
		'evaluation': {
			fit: {'clean': deepcopy(condition), 'hard': deepcopy(condition)}
			for fit in ('clean', 'hard')
		},
	}


class PureBoundaryContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.subject = _subject()

	def test_00_import_is_dependency_light(self):
		# Torch must remain a lazy runtime import, not a module-level dependency.
		source = Path(self.subject.__file__).read_text(encoding='utf-8')
		prefix = source.split('def _model_factory', 1)[0]
		self.assertNotIn('\nimport torch', prefix)
		self.assertEqual(self.subject.HORIZONS, (1, 3, 5))
		self.assertEqual(len(self.subject.ARCHITECTURES), 2)

	def test_01_one_step_rows_never_cross_episodes(self):
		latents, actions = _latents_actions()
		current, action, next_z = self.subject.one_step_arrays(
			latents, actions, [0, 1]
		)
		np.testing.assert_array_equal(current[:, 0], [0.0, 1.0, 10.0, 14.0])
		np.testing.assert_array_equal(action[:, 0], [1.0, 2.0, 4.0, 5.0])
		np.testing.assert_array_equal(next_z[:, 0], [1.0, 3.0, 14.0, 19.0])
		self.assertNotIn(10.0, (next_z - current)[:, 0].tolist())

	def test_02_alignment_fails_closed(self):
		latents, actions = _latents_actions()
		actions[0] = np.zeros((1, 1), dtype=np.float32)
		with self.assertRaisesRegex(ValueError, r'T\+1'):
			self.subject.one_step_arrays(latents, actions, [0])

	def test_03_normalization_uses_selected_train_episodes_only(self):
		latents, actions = _latents_actions()
		first = self.subject.fit_normalization(latents, actions, [0, 1])
		latents[2][...] = -1e9
		actions[2][...] = 1e9
		second = self.subject.fit_normalization(latents, actions, [0, 1])
		for name in first.__dataclass_fields__:
			np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

	def test_04_rollout_has_real_and_shuffled_action_controls(self):
		z = np.asarray([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
		action = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
		row = self.subject.rollout_episode(
			z, action, _Predictor().predict, horizon=2,
			shuffled_action=action[::-1].copy(),
		)
		self.assertAlmostEqual(float(row['real_error'].max()), 0.0)
		self.assertGreater(float(row['shuffled_error'].mean()), 0.0)
		self.assertGreater(float(row['action_shuffle_input_difference'].mean()), 0.0)

	def test_05_rollout_rejects_cross_episode_shape(self):
		with self.assertRaisesRegex(ValueError, r'T\+1'):
			self.subject.rollout_episode(
				np.zeros((3, 1), dtype=np.float32),
				np.zeros((3, 1), dtype=np.float32), _Predictor().predict,
				horizon=1, shuffled_action=np.zeros((3, 1), dtype=np.float32),
			)

	def test_06_bootstrap_is_episode_level_deterministic(self):
		values = np.asarray([1.0, 2.0, 4.0, 8.0])
		first = self.subject.bootstrap_episode_mean(values, seed=7, resamples=1000)
		second = self.subject.bootstrap_episode_mean(values, seed=7, resamples=1000)
		self.assertEqual(first, second)
		self.assertEqual(first['episodes'], 4)

	def test_07_paired_bootstrap_sign_is_unambiguous(self):
		row = self.subject.paired_episode_bootstrap(
			np.asarray([1.0, 1.0, 1.0, 1.0]),
			np.asarray([2.0, 2.0, 2.0, 2.0]), seed=3, resamples=1000,
		)
		self.assertAlmostEqual(row['relative_improvement']['estimate'], 0.5)
		self.assertGreater(row['relative_improvement']['ci95'][0], 0.0)

	def test_08_action_mapping_does_not_read_labels_or_reward(self):
		class Poison(dict):
			def __getitem__(self, key):
				if key != 'action':
					raise AssertionError(f'forbidden read: {key}')
				return super().__getitem__(key)
		dataset = SimpleNamespace(episodes=(SimpleNamespace(
			episode_id=5,
			arrays=Poison(action=np.zeros((2, 1), dtype=np.float32)),
		),))
		result = self.subject.action_mapping(dataset)
		self.assertEqual(set(result), {5})

	def test_09_condition_evaluation_has_all_methods_horizons_and_episode_rows(self):
		action0 = np.asarray([[1.0], [2.0], [1.0], [2.0], [1.0], [2.0]], dtype=np.float32)
		action1 = np.asarray([[2.0], [1.0], [2.0], [1.0], [2.0], [1.0]], dtype=np.float32)
		def trajectory(action):
			return np.concatenate([
				np.zeros((1, 1), dtype=np.float32), np.cumsum(action, axis=0)
			], axis=0)
		dataset = _Dataset('clean', [_Episode(0, action0), _Episode(1, action1)])
		latents = {0: trajectory(action0), 1: trajectory(action1)}
		predictors = {
			'checkpoint_transition': _Predictor(0.8),
			self.subject.ARCHITECTURES[0]: _Predictor(1.0),
			self.subject.ARCHITECTURES[1]: _Predictor(1.0),
		}
		result = self.subject.evaluate_condition(
			dataset, latents, predictors, bootstrap_seed=11, bootstrap_resamples=1000
		)
		self.assertEqual(set(result['horizons']), {'1', '3', '5'})
		for horizon in ('1', '3', '5'):
			self.assertEqual(
				set(result['horizons'][horizon]['methods']), set(self.subject.METHODS)
			)
		matrix = {
			fit: {'clean': deepcopy(result), 'hard': deepcopy(result)}
			for fit in ('clean', 'hard')
		}
		attribution = self.subject.classify_evidence(matrix, {
			'clean': self.subject.ARCHITECTURES[0],
			'hard': self.subject.ARCHITECTURES[1],
		})
		self.assertEqual(
			set(attribution['one_step_domain_matrix_by_architecture']),
			set(self.subject.ARCHITECTURES),
		)
		for architecture in self.subject.ARCHITECTURES:
			for fit in ('clean', 'hard'):
				for test in ('clean', 'hard'):
					self.assertIn(
						'scientific_gate_pass',
						attribution['one_step_domain_matrix_by_architecture'][architecture][fit][test],
					)

	def test_10_lower_mse_without_action_effect_is_rejected(self):
		action = np.asarray([[1.0], [2.0], [1.0], [2.0], [1.0], [2.0]], dtype=np.float32)
		trajectory = np.concatenate([
			np.zeros((1, 1), dtype=np.float32), np.cumsum(action, axis=0)
		], axis=0)
		dataset = _Dataset('clean', [_Episode(0, action), _Episode(1, action)])
		result = self.subject.evaluate_condition(
			dataset, {0: trajectory, 1: trajectory}, {
				'checkpoint_transition': _Predictor(0.8),
				self.subject.ARCHITECTURES[0]: _Predictor(1.0),
				self.subject.ARCHITECTURES[1]: _Predictor(1.0),
			}, bootstrap_seed=13, bootstrap_resamples=1000,
		)
		matrix = {
			fit: {'clean': deepcopy(result), 'hard': deepcopy(result)}
			for fit in ('clean', 'hard')
		}
		for fit in ('clean', 'hard'):
			for test in ('clean', 'hard'):
				for horizon in self.subject.HORIZONS:
					cell = matrix[fit][test]['horizons'][str(horizon)]
					for architecture in self.subject.ARCHITECTURES:
						method = cell['methods'][architecture]
						pooled = method['pooled_frames']
						pooled['real_action_mse'] = 0.5 * pooled['persistence_mse']
						pooled['shuffled_action_mse'] = pooled['real_action_mse']
						for row in method['by_episode']:
							row['real_action_mse'] = 0.5 * row['persistence_mse']
							row['shuffled_action_mse'] = row['real_action_mse']
					if horizon == 1:
						for architecture in self.subject.ARCHITECTURES:
							comparison = cell['paired_refit_comparisons'][
								f'{architecture}_vs_checkpoint_transition'
							]
							comparison['relative_improvement']['ci95'] = [0.2, 0.4]
		attribution = self.subject.classify_evidence(matrix, {
			'clean': self.subject.ARCHITECTURES[0],
			'hard': self.subject.ARCHITECTURES[1],
		})
		self.assertEqual(
			attribution['attribution'],
			'refit_improves_checkpoint_but_not_action_conditioned_dynamics',
		)

	def test_11_result_validation_passes_exact_complete_matrix(self):
		payload = _minimal_result(self.subject)
		self.subject.validate_result(payload)
		json.dumps(payload, allow_nan=False)

	def test_12_result_validation_rejects_policy_authorization(self):
		payload = _minimal_result(self.subject)
		payload['controller_training_authorized'] = True
		with self.assertRaisesRegex(ValueError, 'never authorize'):
			self.subject.validate_result(payload)

	def test_13_result_validation_rejects_missing_cross_domain_cell(self):
		payload = _minimal_result(self.subject)
		del payload['evaluation']['hard']['clean']
		with self.assertRaisesRegex(ValueError, 'cross-domain'):
			self.subject.validate_result(payload)

	def test_14_atomic_output_refuses_overwrite(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / 'result.json'
			self.subject._atomic_json(path, {'ok': True})
			with self.assertRaises(FileExistsError):
				self.subject._atomic_json(path, {'ok': False})

	def test_15_source_contains_no_controller_optimizer(self):
		source = inspect.getsource(self.subject.evaluate)
		self.assertNotIn('agent.update(', source)
		self.assertNotIn('agent.train(', source)
		self.assertIn("parameter.requires_grad_(False)", source)


if __name__ == '__main__':
	unittest.main(verbosity=2)
