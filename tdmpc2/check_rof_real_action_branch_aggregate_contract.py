"""Dependency-light contracts for the three-seed real-branch aggregator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tdmpc2.tools import aggregate_rof_real_action_branches as target
from tdmpc2.tools import evaluate_rof_real_action_branches as evaluator


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(value: float, *, seed: int) -> dict:
	return {
		'estimate': float(value), 'ci95': [float(value), float(value)],
		'root_count': evaluator.FORMAL_SPLIT_COUNTS['test'],
		'resamples': evaluator.FORMAL_BOOTSTRAP_RESAMPLES, 'seed': int(seed),
		'unit': 'whole_root_sibling_and_twin_group',
	}


def _result(source: dict, *, seed: int, candidate: bool) -> dict:
	train_ids = list(range(50))
	validation_ids = list(range(50, 60))
	test_ids = list(range(60, 80))
	bootstrap_seed = seed + evaluator.FORMAL_BOOTSTRAP_SEED_OFFSET
	fit = {
		'train_root_ids': train_ids, 'validation_root_ids': validation_ids,
		'test_roots_seen_during_fit_or_selection': False,
		'parameter_count': 10, 'converged': True,
	}
	condition_metrics = {}
	for condition in evaluator.CONDITIONS:
		condition_seed = evaluator.identifiable._stable_seed(
			bootstrap_seed, condition,
		)
		horizons = {}
		for horizon in evaluator.HORIZONS:
			def record(value, part):
				return _bootstrap(value, seed=evaluator.identifiable._stable_seed(
					condition_seed, horizon, part,
				))
			gain_persistence = record(0.2, 'persistence')
			gain_persistence['relative_improvement'] = {'estimate': 0.2}
			gain_actionless = record(0.1, 'actionless')
			gain_actionless['relative_improvement'] = {'estimate': 0.1}
			nonlinear_gain = record(0.1, 'nonlinear_actionless')
			nonlinear_gain['relative_improvement'] = {'estimate': 0.1}
			horizons[str(horizon)] = {
				'root_rows': [{'root_id': root_id} for root_id in test_ids],
				'aware_error': record(0.1, 'aware'),
				'aware_relative_gain_vs_persistence': gain_persistence,
				'aware_relative_gain_vs_actionless': gain_actionless,
				'real_sibling_action_ranking_auc': record(0.7, 'ranking'),
				'action_effect_r2_vs_zero_sibling': record(0.3, 'effect_r2'),
				'nonlinear_capacity_control': {
					'diagnostic_only': True,
					'aware_error': record(0.1, 'nonlinear_aware_error'),
					'aware_relative_gain_vs_parameter_matched_actionless': (
						nonlinear_gain
					),
					'real_sibling_action_ranking_auc': record(
						0.7, 'nonlinear_ranking',
					),
					'action_effect_r2_vs_zero_sibling': record(
						0.3, 'nonlinear_effect_r2',
					),
				},
			}
		condition_metrics[condition] = {
			'horizons': horizons,
			'packet_validity': {
				'valid_rate': _bootstrap(
					1.0, seed=evaluator.identifiable._stable_seed(
						condition_seed, 'validity',
					),
				), 'max_invalid_burst': 0,
			},
		}
	return {
		'format': evaluator.RESULT_FORMAT, 'status': evaluator.RESULT_STATUS,
		'engineering_pass': True,
		'action_identifiable_representation_candidate': candidate,
		'policy_training_performed': False,
		'controller_training_authorized': False,
		'privileged_fields_used_as_model_input': False,
		'reward_or_return_used': False,
		'background_id_used_as_model_input': False,
		'task': 'acrobot-swingup', 'source': source,
		'protocol': {
			'split_unit': 'whole_root_sibling_and_twin_group',
			'bootstrap_unit': 'whole_root_sibling_and_twin_group',
			'real_interventions_only': True,
			'shuffled_logged_futures_used': False,
			'fit_domain': (
				'pooled_clean_hard_background_twins_without_condition_identifier'
			),
			'condition_or_background_identifier_used_for_fit': False,
			'seed': seed,
			'bootstrap_seed': bootstrap_seed,
			'bootstrap_resamples': evaluator.FORMAL_BOOTSTRAP_RESAMPLES,
			'hidden_dim': evaluator.FORMAL_HIDDEN_DIM,
			'formal_split_counts': dict(evaluator.FORMAL_SPLIT_COUNTS),
			'fit_models': list(evaluator.FIT_MODES),
		},
		'oracle_data_validity': {
			'passed': True,
			'oracle_fields_used_for_model_fit_selection_or_scoring': False,
		},
		'training': {
			mode: copy.deepcopy(fit) for mode in evaluator.FIT_MODES
		},
		'root_splits': {
			'train': train_ids, 'validation': validation_ids, 'test': test_ids,
		},
		'condition_metrics': condition_metrics,
		'background_twin_metrics': {'horizons': {
			str(horizon): {
				'background_over_real_action_distance': _bootstrap(
					0.25, seed=evaluator.identifiable._stable_seed(
						evaluator.identifiable._stable_seed(
							bootstrap_seed, 'twins',
						), horizon, 'twin_ratio',
					),
				),
			} for horizon in evaluator.HORIZONS
		}},
		'gates': {
			'all_fits_converged': True,
			'controller_training_authorized': False,
			'action_identifiable_representation_candidate': candidate,
			'nonlinear_capacity_control': {
				'diagnostic_only': True, 'all_pass': False,
			},
		},
	}


class AggregateContractTests(unittest.TestCase):
	def _fixture(self, root: Path, candidates=(True, False, True)):
		assets = {}
		for name in ('dataset', 'runtime_config', 'checkpoint'):
			path = root / name
			path.write_bytes(name.encode('ascii'))
			assets[name] = path
		source = {
			name: str(path.resolve())
			for name, path in assets.items()
		} | {
			f'{name}_sha256': _sha256(path)
			for name, path in assets.items()
		}
		results = root / 'runs'
		results.mkdir()
		for seed, candidate in zip(target.EXPECTED_SEEDS, candidates):
			payload = _result(source, seed=seed, candidate=candidate)
			(results / f'seed_{seed}.json').write_text(
				json.dumps(payload), encoding='utf-8',
			)
		return results

	def test_two_of_three_and_no_controller_authorization(self):
		with tempfile.TemporaryDirectory() as raw:
			results = self._fixture(Path(raw))
			payload = target.aggregate(SimpleNamespace(
				results_root=results, task='acrobot-swingup',
			))
			self.assertTrue(payload['task_candidate'])
			self.assertEqual(payload['candidate_seed_count'], 2)
			self.assertFalse(payload['controller_training_authorized'])
			self.assertFalse(payload['policy_training_performed'])
			target.validate_summary(payload)

	def test_bound_input_mutation_fails_closed(self):
		with tempfile.TemporaryDirectory() as raw:
			root = Path(raw)
			results = self._fixture(root)
			(root / 'checkpoint').write_bytes(b'changed')
			with self.assertRaisesRegex(ValueError, 'changed after evaluation'):
				target.aggregate(SimpleNamespace(
					results_root=results, task='acrobot-swingup',
				))


if __name__ == '__main__':
	unittest.main(verbosity=2)
