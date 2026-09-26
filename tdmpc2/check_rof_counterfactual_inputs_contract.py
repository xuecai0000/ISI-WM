"""Dependency-light contracts for ROF counterfactual input diagnosis."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from tdmpc2.tools import evaluate_rof_counterfactual_inputs as subject


def arrays(*, length: int = 5, roles: int = 2) -> dict[str, np.ndarray]:
	rgb = np.arange(length * 9 * 64 * 64, dtype=np.int64)
	rgb = (rgb % 251).astype(np.uint8).reshape(length, 9, 64, 64)
	objects = np.empty((length, roles, 1770), dtype=np.float32)
	for time in range(length):
		for role in range(roles):
			for slot in range(3):
				start = slot * 590
				objects[time, role, start:start + 590] = (
					1000 * time + 100 * role + 10 * slot
					+ np.arange(590, dtype=np.float32) / 1000
				)
	masks = np.zeros((length, roles, 3, 64, 64), dtype=np.bool_)
	masks[:, 0, :, 4:12, 6:15] = True
	if roles > 1:
		masks[:, 1, :, 24:35, 30:42] = True
	gt = np.zeros((length, roles, 64, 64), dtype=np.bool_)
	for time in range(length):
		gt[time, 0, time:time + 2, 2:7] = True
		if roles > 1:
			gt[time, 1, 20:25, time:time + 3] = True
	return {
		'policy_rgb': rgb,
		'policy_object': objects,
		'policy_object_mask': masks,
		'policy_role_exists': np.ones((length, roles), dtype=np.float32),
		'labels__gt_role_mask': gt,
	}


def query_mean(roles: int = 2) -> np.ndarray:
	value = np.empty((roles, 3, 512), dtype=np.float32)
	for role in range(roles):
		for slot in range(3):
			value[role, slot] = 10 * role + slot + 0.25
	return value


class InterventionContract(unittest.TestCase):
	def setUp(self):
		self.arrays = arrays()
		self.mean = query_mean()

	def test_original_is_bit_identical_and_has_policy_keys_only(self):
		result = subject.intervene_episode(self.arrays, 'O', self.mean)
		self.assertEqual(set(result), set(subject.ladder.POLICY_KEYS))
		for name in subject.ladder.POLICY_KEYS:
			np.testing.assert_array_equal(result[name], self.arrays[name])
			self.assertIsNot(result[name], self.arrays[name])

	def test_r_replaces_only_each_frame_online_union_exterior(self):
		result = subject.intervene_episode(self.arrays, 'R', self.mean)
		before = self.arrays['policy_rgb'].reshape(-1, 3, 3, 64, 64)
		after = result['policy_rgb'].reshape(before.shape)
		union = self.arrays['policy_object_mask'].any(axis=1)
		np.testing.assert_array_equal(after[union[:, :, None].repeat(3, axis=2)],
			before[union[:, :, None].repeat(3, axis=2)])
		self.assertTrue(np.all(after[~union[:, :, None].repeat(3, axis=2)] == 128))
		for name in ('policy_object', 'policy_object_mask', 'policy_role_exists'):
			np.testing.assert_array_equal(result[name], self.arrays[name])

	def test_m_uses_episode_local_t_minus_two_to_t_alignment_only(self):
		result = subject.intervene_episode(self.arrays, 'M', self.mean)
		gt = self.arrays['labels__gt_role_mask']
		np.testing.assert_array_equal(result['policy_object_mask'][0, :, 0], gt[0])
		np.testing.assert_array_equal(result['policy_object_mask'][0, :, 1], gt[0])
		np.testing.assert_array_equal(result['policy_object_mask'][0, :, 2], gt[0])
		np.testing.assert_array_equal(result['policy_object_mask'][1, :, 0], gt[0])
		np.testing.assert_array_equal(result['policy_object_mask'][1, :, 1], gt[0])
		np.testing.assert_array_equal(result['policy_object_mask'][1, :, 2], gt[1])
		np.testing.assert_array_equal(result['policy_object_mask'][4, :, 0], gt[2])
		np.testing.assert_array_equal(result['policy_object_mask'][4, :, 1], gt[3])
		np.testing.assert_array_equal(result['policy_object_mask'][4, :, 2], gt[4])
		for name in ('policy_rgb', 'policy_object', 'policy_role_exists'):
			np.testing.assert_array_equal(result[name], self.arrays[name])

	def test_q_replaces_query_only_and_preserves_geometry_status(self):
		result = subject.intervene_episode(self.arrays, 'Q', self.mean)
		before = self.arrays['policy_object'].reshape(5, 2, 3, 590)
		after = result['policy_object'].reshape(before.shape)
		np.testing.assert_array_equal(
			after[..., :512], np.broadcast_to(self.mean, after[..., :512].shape)
		)
		np.testing.assert_array_equal(after[..., 512:], before[..., 512:])
		for name in ('policy_rgb', 'policy_object_mask', 'policy_role_exists'):
			np.testing.assert_array_equal(result[name], self.arrays[name])

	def test_rq_is_exact_composition_without_mask_or_status_change(self):
		rq = subject.intervene_episode(self.arrays, 'RQ', self.mean)
		r = subject.intervene_episode(self.arrays, 'R', self.mean)
		q = subject.intervene_episode(self.arrays, 'Q', self.mean)
		np.testing.assert_array_equal(rq['policy_rgb'], r['policy_rgb'])
		np.testing.assert_array_equal(rq['policy_object'], q['policy_object'])
		np.testing.assert_array_equal(
			rq['policy_object_mask'], self.arrays['policy_object_mask']
		)
		np.testing.assert_array_equal(
			rq['policy_role_exists'], self.arrays['policy_role_exists']
		)

	def test_scope_assertion_rejects_an_extra_change(self):
		original = subject._policy_copy(self.arrays)
		changed = subject._policy_copy(self.arrays)
		changed['policy_object_mask'][0, 0, 0, 0, 0] ^= True
		with self.assertRaisesRegex(ValueError, 'changed online masks'):
			subject.assert_intervention_contract(
				original, changed, self.arrays, 'Q', self.mean
			)


class ReferenceAndMetricContract(unittest.TestCase):
	def test_query_mean_uses_only_clean_train_episodes_by_role_and_slot(self):
		roles = ('arm', 'goal')
		episodes = []
		for episode_id, base in ((0, 1.0), (1, 3.0), (2, 999.0)):
			value = arrays(length=2)
			frames = value['policy_object'].reshape(2, 2, 3, 590)
			for role in range(2):
				for slot in range(3):
					frames[:, role, slot, :512] = base + 10 * role + slot
			episodes.append(SimpleNamespace(episode_id=episode_id, arrays=value))
		dataset = SimpleNamespace(
			manifest={'condition': 'clean'}, role_names=roles,
			split=lambda name: tuple(episodes[:2]) if name == 'train' else (episodes[2],),
		)
		mean = subject.clean_train_query_mean(dataset)
		for role in range(2):
			for slot in range(3):
				self.assertTrue(np.all(mean[role, slot] == 2.0 + 10 * role + slot))

	def test_query_reference_rejects_hard_dataset(self):
		dataset = SimpleNamespace(manifest={'condition': 'hard'})
		with self.assertRaisesRegex(ValueError, 'must be clean'):
			subject.clean_train_query_mean(dataset)

	def test_token_perturbation_separates_local_and_global(self):
		original = np.zeros((4, 2, 5, 64), dtype=np.float64)
		changed = original.copy()
		changed[:, :, :4] = 2.0
		metrics = subject.token_perturbation(original, changed)
		self.assertEqual(metrics['global']['rms_per_frame']['max'], 0.0)
		self.assertEqual(metrics['local']['rms_per_frame']['mean'], 2.0)
		self.assertGreater(metrics['full']['rms_per_frame']['mean'], 0.0)

	def test_train_ood_radii_and_scoring_are_finite(self):
		rng = np.random.default_rng(7)
		train = rng.normal(size=(200, 2, 5, 64))
		reference = subject.fit_train_clean_ood_reference(train)
		test = train[:10] + 10.0
		scored = subject.score_ood(test, reference)
		for name in ('full', 'local', 'global'):
			self.assertGreaterEqual(reference[name]['radius_99'], reference[name]['radius_95'])
			self.assertEqual(scored[name]['fraction_outside_99_radius'], 1.0)
			self.assertFalse(scored[name]['heuristic_support_flag'])

	def test_latent_variability_detects_constant_and_dynamic_latents(self):
		constant = {index: np.ones((6, 8), dtype=np.float64) for index in range(4)}
		constant_result = subject.latent_variability(constant, (0, 1, 2, 3))
		self.assertTrue(constant_result['near_constant_heuristic'])
		self.assertEqual(constant_result['pca_participation_ratio'], 0.0)
		dynamic = {
			index: np.arange(48, dtype=np.float64).reshape(6, 8) + index
			for index in range(4)
		}
		dynamic_result = subject.latent_variability(dynamic, (0, 1, 2, 3))
		self.assertFalse(dynamic_result['near_constant_heuristic'])
		self.assertGreater(dynamic_result['rms_consecutive_change']['mean'], 0.0)
		self.assertGreater(dynamic_result['pca_participation_ratio'], 0.0)

	def test_bootstrap_is_episode_level_and_deterministic(self):
		values = np.asarray([1.0, 2.0, 3.0, 8.0])
		left = subject._bootstrap_summary(values, seed=91, resamples=1000)
		right = subject._bootstrap_summary(values, seed=91, resamples=1000)
		self.assertEqual(left, right)
		self.assertEqual(left['episodes'], 4)
		self.assertEqual(left['estimate_episode_mean'], 3.5)

	def test_effectiveness_requires_raw_and_intended_token_changes(self):
		value = arrays(length=3)
		episode = SimpleNamespace(episode_id=7, arrays=value)
		dataset = SimpleNamespace(
			role_names=('arm', 'goal'), splits={'test': (7,)},
			split=lambda name: (episode,),
		)
		base = np.zeros((3, 2, 5, 64), dtype=np.float64)
		rgb = base.copy(); rgb[:, :, :4] = 1.0
		mask = base.copy(); mask[:, :, :4] = 2.0
		query = base.copy(); query[:, :, 4:] = 3.0
		rq = rgb + query
		encoded = {
			'O': {'tokens': {7: base}}, 'R': {'tokens': {7: rgb}},
			'M': {'tokens': {7: mask}}, 'Q': {'tokens': {7: query}},
			'RQ': {'tokens': {7: rq}},
		}
		result = subject.intervention_effectiveness(dataset, encoded, query_mean())
		self.assertEqual(result['R']['status'], 'effective')
		self.assertEqual(result['Q']['status'], 'effective')
		self.assertEqual(result['O']['perturbation_max_abs']['full'], 0.0)
		encoded['Q']['tokens'][7] = base.copy()
		with self.assertRaisesRegex(ValueError, 'did not change global tokens'):
			subject.intervention_effectiveness(dataset, encoded, query_mean())

	def test_encoded_path_separation_accepts_only_intended_routes(self):
		base = np.zeros((3, 2, 5, 64), dtype=np.float64)
		rgb = base.copy()
		rgb[:, :, :4] = 1.0
		mask = base.copy()
		mask[:, :, :4] = 2.0
		query = base.copy()
		query[:, :, 4:] = 3.0
		rgb_query = base.copy()
		rgb_query[:, :, :4] = 1.0
		rgb_query[:, :, 4:] = 3.0
		encoded = {
			'O': {'tokens': {0: base}},
			'R': {'tokens': {0: rgb}},
			'M': {'tokens': {0: mask}},
			'Q': {'tokens': {0: query}},
			'RQ': {'tokens': {0: rgb_query}},
		}
		result = subject.encoded_path_separation(encoded)
		self.assertTrue(all(row['passed'] for row in result.values()))
		encoded['RQ']['tokens'][0][0, 0, 0, 0] += 0.1
		with self.assertRaisesRegex(ValueError, 'separated encoder route'):
			subject.encoded_path_separation(encoded)

	def test_result_validator_forbids_training_authorization(self):
		bootstrap = {
			'model_over_persistence': {}, 'model_over_action_shuffle': {},
			'model_minus_persistence_mse': {},
			'model_minus_action_shuffle_mse': {},
		}
		stage = {
			'status': 'passed',
			'horizons': {
				name: {
					'by_episode': {str(index): {} for index in range(4)},
					'paired_episode_bootstrap': bootstrap,
				} for name in ('1', '3', '5')
			},
		}
		perturbation_views = {'full': {}, 'local': {}, 'global': {}}
		ood_views = {
			name: {'heuristic_support_flag': True}
			for name in ('full', 'local', 'global')
		}
		payload = {
			'format': subject.FORMAT,
			'status': 'rof_counterfactual_input_diagnostic_complete',
			'engineering_pass': True,
			'scientific_complete': True,
			'scientific_complete_semantics': 'all requested diagnostic outputs are present',
			'controller_training_authorized': False,
			'checks': {'all': True},
			'intervention_contract': {'modes': list(subject.INTERVENTIONS)},
			'interventions': {
				mode: {
					'token_perturbation_vs_O': perturbation_views,
					'train_clean_latent_ood': ood_views,
					'test_latent_variability': {
						'episodes': 4, 'pca_participation_ratio': 2.0,
					},
					'C_dynamics': stage,
				} for mode in subject.INTERVENTIONS
			},
			'encoded_path_separation': {'routes': {'passed': True}},
			'intervention_effectiveness': {
				'O': {
					'status': 'effective', 'perturbation_max_abs': {
						'full': 0.0, 'local': 0.0, 'global': 0.0,
					},
				},
				'R': {
					'status': 'effective', 'actual_rgb_elements_changed': 1,
					'local_token_max_abs_vs_O': 1.0,
				},
				'M': {'status': 'not_applicable'},
				'Q': {
					'status': 'effective', 'actual_object_elements_changed': 1,
					'global_token_max_abs_vs_O': 1.0,
				},
				'RQ': {'status': 'effective'},
			},
			'interpretation_guard': {
				'improvement': 'one way', 'no_improvement': 'not exclusion',
				'gt_scope': 'offline only', 'ood_scope': 'descriptive',
				'r_scope': 'ring', 'q_scope': 'time varying',
			},
		}
		self.assertEqual(subject.validate_result(payload)['status'], 'verified')
		payload['controller_training_authorized'] = True
		with self.assertRaisesRegex(ValueError, 'never authorize'):
			subject.validate_result(payload)


if __name__ == '__main__':
	unittest.main()
