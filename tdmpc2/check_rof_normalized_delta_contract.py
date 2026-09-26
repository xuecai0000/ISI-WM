"""Dependency-light fail-closed contracts for the ROF normalized-delta probe."""

from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import re
import tempfile
import unittest

import numpy as np


def _subject():
	return importlib.import_module('tdmpc2.tools.evaluate_rof_normalized_delta')


def _toy_trace(subject, *, steps=10, action_dim=1):
	t = np.arange(steps + 1, dtype=np.float32)
	logits = np.stack([
		0.2 * np.sin(t / 3 + index) + 0.03 * t * (index - 3.5)
		for index in range(8)
	], axis=1)
	logits -= logits.mean(axis=1, keepdims=True)
	z = subject.clr_to_simnorm(logits, 8)
	u = subject.clr_transform(z, 8)
	action = np.stack([
		(np.arange(steps, dtype=np.float32) + 1) * (index + 1)
		for index in range(action_dim)
	], axis=1)
	return z, u, action


def _minimal_evaluation(subject):
	episodes = [16, 17, 18, 19]
	horizons = {}
	for horizon in subject.HORIZONS:
		methods = {}
		for method in subject.METHODS:
			if method == 'persistence':
				methods[method] = {
					'by_episode': [
						{'episode_id': episode, 'normalized_delta_mse': 1.0}
						for episode in episodes
					],
				}
			else:
				methods[method] = {
					'by_episode': [
						{'episode_id': episode, 'variants': {}}
						for episode in episodes
					],
				}
		horizons[str(horizon)] = {'methods': methods}
	return {'test_episode_ids': episodes, 'horizons': horizons}


def _minimal_result(subject):
	latent_dim, action_dim = 640, 1
	widths = subject.parameter_matched_widths(latent_dim, action_dim)
	counts = {}
	for architecture in subject.ARCHITECTURES:
		for mode in subject.ACTION_MODES:
			counts[(architecture, mode)] = subject.mlp_parameter_count(
				subject.architecture_input_dim(
					architecture, latent_dim, action_dim, mode,
				), widths[architecture][mode], latent_dim,
			)
	training = {}
	for condition in subject.FIT_CONDITIONS:
		training[condition] = {}
		for architecture in subject.ARCHITECTURES:
			training[condition][architecture] = {
				mode: {
					'train_episode_ids': list(range(12)),
					'validation_episode_ids': list(range(12, 16)),
					'test_episodes_seen_during_fit_or_selection': False,
					'converged': True,
					'parameter_count': counts[(architecture, mode)],
				}
				for mode in subject.ACTION_MODES
			}
	condition = _minimal_evaluation(subject)
	return {
		'format': subject.FORMAT, 'status': subject.STATUS,
		'engineering_pass': True, 'scientific_complete': True,
		'controller_training_authorized': False,
		'policy_training_performed': False,
		'privileged_targets_used': False,
		'simulator_state_used': False,
		'protocol': {
			'fit_conditions': list(subject.FIT_CONDITIONS),
			'episode_split_counts': dict(subject.SPLIT_COUNTS),
			'teacher_forcing_after_initial_history': False,
			'primary_metric': subject.PREREGISTERED_GATES['primary_metric'],
		},
		'training': training,
		'evaluation': {
			fit: {
				'clean': deepcopy(condition), 'hard': deepcopy(condition),
			}
			for fit in subject.FIT_CONDITIONS
		},
		'action_coverage_audit': {
			condition: {
				split: {
					'coverage_sufficient_for_incremental_action_attribution': True,
					'interpretation': (
						'Incremental predictive information, not causal controllability.'
					),
				}
				for split in ('train', 'test')
			}
			for condition in subject.FIT_CONDITIONS
		},
		'clr_epsilon_sensitivity': {
			condition: {} for condition in subject.FIT_CONDITIONS
		},
		'gates': {
			'preregistered_thresholds': dict(subject.PREREGISTERED_GATES),
		},
	}


class NormalizedDeltaContract(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.subject = _subject()

	def test_00_import_is_dependency_light_and_protocol_is_fixed(self):
		source = Path(self.subject.__file__).read_text(encoding='utf-8')
		prefix = source.split('def _model_factory', 1)[0]
		self.assertNotIn('\nimport torch', prefix)
		self.assertEqual(self.subject.HORIZONS, (1, 3, 5))
		self.assertEqual(self.subject.HISTORY_LENGTH, 3)
		self.assertEqual(self.subject.SPLIT_COUNTS, {
			'train': 12, 'validation': 4, 'test': 4,
		})
		self.assertEqual(set(self.subject.ARCHITECTURES), {
			'normalized_delta_markov', 'normalized_delta_history',
		})

	def test_01_clr_round_trip_and_zero_sum(self):
		z, _, _ = _toy_trace(self.subject)
		u, audit = self.subject.clr_transform(z, 8, return_audit=True)
		rebuilt = self.subject.clr_to_simnorm(u, 8)
		np.testing.assert_allclose(rebuilt, z, rtol=2e-5, atol=2e-6)
		self.assertLess(audit['output_max_abs_group_mean'], 1e-6)
		groups = u.reshape(len(u), -1, 8)
		np.testing.assert_allclose(groups.sum(axis=-1), 0.0, atol=2e-6)

	def test_02_clr_projection_is_zero_sum_and_reports_correction(self):
		delta = np.arange(32, dtype=np.float32).reshape(2, 16)
		projected, audit = self.subject.project_clr_innovation(delta, 8)
		np.testing.assert_allclose(
			projected.reshape(2, 2, 8).sum(axis=-1), 0.0, atol=1e-6,
		)
		self.assertGreater(audit['projection_correction_rms'], 0.0)
		self.assertLess(audit['post_projection_max_abs_group_sum'], 1e-6)

	def test_03_sequence_rows_are_causal_and_never_cross_episode(self):
		_, u0, a0 = _toy_trace(self.subject, steps=10)
		_, u1, a1 = _toy_trace(self.subject, steps=12)
		u1 = u1 + 100.0
		rows = self.subject.sequence_rows({4: u0, 9: u1}, {4: a0, 9: a1}, [4, 9])
		self.assertTrue(np.all(rows.start_indices[rows.episode_ids == 4] >= 2))
		self.assertEqual(rows.initial_u_history.shape[1], 3)
		self.assertEqual(rows.action_window.shape[1], 7)
		self.assertEqual(rows.target_future_u.shape[1], 5)
		first_other = np.flatnonzero(rows.episode_ids == 9)[0]
		self.assertGreater(float(rows.initial_u_history[first_other].mean()), 90.0)

	def test_04_future_targets_do_not_enter_initial_context(self):
		_, u, action = _toy_trace(self.subject, steps=10)
		before = self.subject.sequence_rows({0: u}, {0: action}, [0])
		modified = u.copy()
		modified[3] += np.linspace(-2.0, 2.0, modified.shape[1], dtype=np.float32)
		after = self.subject.sequence_rows({0: modified}, {0: action}, [0])
		np.testing.assert_array_equal(
			before.initial_u_history[0], after.initial_u_history[0],
		)
		self.assertFalse(np.array_equal(
			before.target_future_u[0, 0], after.target_future_u[0, 0],
		))

	def test_05_shuffle_preserves_past_and_changes_only_rollout_actions(self):
		_, u, action = _toy_trace(self.subject, steps=12)
		rows = self.subject.sequence_rows({0: u}, {0: action}, [0])
		shuffled_trace = action[::-1].copy()
		shuffled = self.subject.shuffled_future_rows(rows, shuffled_trace)
		np.testing.assert_array_equal(
			shuffled.action_window[:, :2], rows.action_window[:, :2],
		)
		for index, start in enumerate(rows.start_indices):
			np.testing.assert_array_equal(
				shuffled.action_window[index, 2:],
				shuffled_trace[start:start + 5],
			)
		self.assertGreater(float(np.mean(np.square(
			shuffled.action_window[:, 2:] - rows.action_window[:, 2:]
		))), 0.0)

	def test_06_actionless_features_are_truly_invariant_to_every_action(self):
		_, u, action = _toy_trace(self.subject, steps=12)
		rows = self.subject.sequence_rows({0: u}, {0: action}, [0])
		mean = np.asarray([5.0], dtype=np.float32)
		scale = np.asarray([2.0], dtype=np.float32)
		first_history = rows.action_window[:, :3]
		changed = first_history + 1e6
		for architecture in self.subject.ARCHITECTURES:
			first = self.subject.normalized_action_features(
				first_history, mean, scale, architecture=architecture,
				action_mode='actionless',
			)
			second = self.subject.normalized_action_features(
				changed, mean, scale, architecture=architecture,
				action_mode='actionless',
			)
			self.assertEqual(first.shape, (len(rows), 0))
			np.testing.assert_array_equal(first, second)
		self.assertEqual(self.subject.architecture_input_dim(
			'normalized_delta_markov', 640, 6, 'actionless',
		), 640)
		self.assertEqual(self.subject.architecture_input_dim(
			'normalized_delta_history', 640, 6, 'actionless',
		), 1920)

	def test_07_normalization_uses_only_selected_train_episodes(self):
		latents, actions = {}, {}
		for episode in range(13):
			_, u, action = _toy_trace(self.subject, steps=10)
			latents[episode] = u + episode * 0.01
			actions[episode] = action + episode * 0.02
		first = self.subject.fit_delta_normalization(
			latents, actions, list(range(12)), simnorm_dim=8,
		)
		latents[12][...] = 1e6
		actions[12][...] = -1e6
		second = self.subject.fit_delta_normalization(
			latents, actions, list(range(12)), simnorm_dim=8,
		)
		for name in ('u_mean', 'u_scale', 'action_mean', 'action_scale',
			'delta_mean', 'delta_scale'):
			np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
		for horizon in self.subject.HORIZONS:
			np.testing.assert_array_equal(
				first.horizon_scale[horizon], second.horizon_scale[horizon],
			)
		for scale in (first.u_scale, first.delta_scale, *first.horizon_scale.values()):
			groups = scale.reshape(-1, 8)
			np.testing.assert_array_equal(
				groups, np.repeat(groups[:, :1], 8, axis=1),
			)
		self.assertIn('floor_audit', self.subject.normalization_summary(first))

	def test_08_parameter_matching_is_within_five_percent(self):
		for latent_dim, action_dim in ((640, 1), (640, 6)):
			widths = self.subject.parameter_matched_widths(latent_dim, action_dim)
			counts = []
			for architecture in self.subject.ARCHITECTURES:
				for mode in self.subject.ACTION_MODES:
					counts.append(self.subject.mlp_parameter_count(
						self.subject.architecture_input_dim(
							architecture, latent_dim, action_dim, mode,
						), widths[architecture][mode], latent_dim,
					))
			reference = counts[0]
			for count in counts:
				self.assertLessEqual(abs(count / reference - 1.0), 0.05)

	def test_09_numpy_rollout_history_uses_predictions_not_future_truth(self):
		history = np.asarray([[[0.0], [1.0], [2.0]]], dtype=np.float32)
		next_u = np.asarray([[7.0]], dtype=np.float32)
		updated = self.subject.np_or_torch_cat_history(history, next_u)
		np.testing.assert_array_equal(updated[0, :, 0], [1.0, 2.0, 7.0])

	def test_10_convergence_measure_detects_plateau_and_descent(self):
		plateau = [
			{'validation_objective': 1.0 - index * 1e-5} for index in range(20)
		]
		descent = [
			{'validation_objective': 1.0 - index * 0.02} for index in range(20)
		]
		self.assertLess(
			self.subject._trailing_relative_improvement(plateau, 20), 0.01,
		)
		self.assertGreater(
			self.subject._trailing_relative_improvement(descent, 20), 0.01,
		)

	def test_10b_action_coverage_and_epsilon_sensitivity_are_audited(self):
		latents, actions, episodes = {}, {}, []
		raw = {}
		for episode_id in range(12):
			z, u, action = _toy_trace(self.subject, steps=20)
			action = np.sin(action / 3.0 + episode_id).astype(np.float32)
			latents[episode_id], actions[episode_id], raw[episode_id] = u, action, z
			episodes.append(SimpleNamespace(
				episode_id=episode_id, arrays={'action': action},
			))
		normalization = self.subject.fit_delta_normalization(
			latents, actions, list(range(12)), simnorm_dim=8,
		)
		dataset = SimpleNamespace(episodes=tuple(episodes))
		audit = self.subject.action_coverage_audit(
			dataset, latents, normalization, list(range(12)),
		)
		self.assertIn('covariance_effective_rank', audit)
		self.assertIn('lag1_autocorrelation', audit['per_dimension'])
		self.assertIn('nearest_neighbor_local_action_variance_ratio', audit['per_dimension'])
		self.assertIn('not causal controllability', audit['interpretation'])
		sensitivity = self.subject.clr_epsilon_sensitivity(raw, 8)
		self.assertEqual(set(sensitivity), {'1e-07', '1e-06', '1e-05'})
		self.assertEqual(
			sensitivity['1e-06']['clr_rms_difference_vs_1e_minus_6'], 0.0,
		)

	def test_11_result_validation_passes_complete_fail_closed_schema(self):
		payload = _minimal_result(self.subject)
		self.subject.validate_result(payload)
		json.dumps(payload, allow_nan=False)

	def test_12_result_validation_rejects_authorization_and_privileged_targets(self):
		payload = _minimal_result(self.subject)
		payload['controller_training_authorized'] = True
		with self.assertRaisesRegex(ValueError, 'never authorize'):
			self.subject.validate_result(payload)
		payload = _minimal_result(self.subject)
		payload['privileged_targets_used'] = True
		with self.assertRaisesRegex(ValueError, 'Privileged'):
			self.subject.validate_result(payload)

	def test_13_result_validation_rejects_nonconvergence_claim_and_missing_matrix(self):
		payload = _minimal_result(self.subject)
		payload['training']['hard']['normalized_delta_history']['action_aware'][
			'converged'
		] = False
		with self.assertRaisesRegex(ValueError, 'Scientific-complete'):
			self.subject.validate_result(payload)
		payload = _minimal_result(self.subject)
		del payload['evaluation']['clean']['hard']
		with self.assertRaisesRegex(ValueError, 'matrix'):
			self.subject.validate_result(payload)

	def test_14_atomic_output_refuses_overwrite(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / 'result.json'
			self.subject._atomic_json(path, {'ok': True})
			with self.assertRaises(FileExistsError):
				self.subject._atomic_json(path, {'ok': False})

	def test_15_source_has_no_policy_optimizer_or_privileged_array_read(self):
		source = Path(self.subject.__file__).read_text(encoding='utf-8')
		evaluate_source = inspect.getsource(self.subject.evaluate)
		self.assertNotIn('agent.update(', evaluate_source)
		self.assertNotIn('agent.train(', evaluate_source)
		self.assertIn('parameter.requires_grad_(False)', evaluate_source)
		array_reads = re.findall(r"episode\.arrays\[['\"]([^'\"]+)", source)
		self.assertEqual(set(array_reads), {'action'})
		for forbidden in (
			"episode.arrays['reward']", "episode.arrays['done']",
			"episode.arrays['labels__state']", 'optimizer = agent',
		):
			self.assertNotIn(forbidden, source)
		self.assertNotIn('causal_history_recovers', source)
		self.assertNotIn('missing_latent_information', source)
		self.assertIn('cannot by themselves prove causal controllability', source)
		self.assertIn('does not prove history', source)


if __name__ == '__main__':
	unittest.main(verbosity=2)
