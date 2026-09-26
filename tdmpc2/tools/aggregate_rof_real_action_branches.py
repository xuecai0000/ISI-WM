"""Aggregate three fixed-seed ROF real-action branch diagnostics.

The input evaluator results already contain root-clustered confidence intervals.
This aggregator applies only the preregistered two-of-three replication rule; it
does not pool roots across fit seeds or create a more optimistic confidence
interval.  It never authorizes controller training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

from tdmpc2.tools import evaluate_rof_real_action_branches as evaluator


FORMAT = 'rof_real_action_branch_three_seed_summary_v1'
STATUS = 'rof_real_action_branch_three_seed_complete'
EXPECTED_SEEDS = (20260913, 20260917, 20260923)
REQUIRED_PASSES = 2


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
	_require(path.is_file() and not path.is_symlink(), f'Invalid result path: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(payload, dict), f'Result is not a JSON object: {path}')
	evaluator.validate_result(payload)
	return payload


def _mean(values) -> float:
	rows = [float(value) for value in values]
	_require(rows and all(value == value for value in rows), 'Mean input is invalid.')
	return float(sum(rows) / len(rows))


def _metric_means(results: list[Mapping]) -> dict[str, Any]:
	conditions: dict[str, Any] = {}
	for condition in evaluator.CONDITIONS:
		horizons: dict[str, Any] = {}
		for horizon in evaluator.HORIZONS:
			key = str(horizon)
			rows = [result['condition_metrics'][condition]['horizons'][key]
				for result in results]
			horizons[key] = {
				'aware_relative_gain_vs_persistence': _mean(
					row['aware_relative_gain_vs_persistence'][
						'relative_improvement']['estimate'] for row in rows
				),
				'aware_relative_gain_vs_actionless': _mean(
					row['aware_relative_gain_vs_actionless'][
						'relative_improvement']['estimate'] for row in rows
				),
				'real_sibling_action_ranking_auc': _mean(
					row['real_sibling_action_ranking_auc']['estimate'] for row in rows
				),
				'action_effect_r2_vs_zero_sibling': _mean(
					row['action_effect_r2_vs_zero_sibling']['estimate'] for row in rows
				),
				'nonlinear_capacity_control': {
					'aware_relative_gain_vs_parameter_matched_actionless': _mean(
						row['nonlinear_capacity_control'][
							'aware_relative_gain_vs_parameter_matched_actionless'
						]['relative_improvement']['estimate'] for row in rows
					),
					'real_sibling_action_ranking_auc': _mean(
						row['nonlinear_capacity_control'][
							'real_sibling_action_ranking_auc'
						]['estimate'] for row in rows
					),
					'action_effect_r2_vs_zero_sibling': _mean(
						row['nonlinear_capacity_control'][
							'action_effect_r2_vs_zero_sibling'
						]['estimate'] for row in rows
					),
				},
			}
		validity = [result['condition_metrics'][condition]['packet_validity']
			for result in results]
		conditions[condition] = {
			'horizons': horizons,
			'packet_valid_rate': _mean(
				row['valid_rate']['estimate'] for row in validity
			),
			'max_packet_invalid_burst': max(
				int(row['max_invalid_burst']) for row in validity
			),
		}
	twins = {}
	for horizon in evaluator.HORIZONS:
		key = str(horizon)
		twins[key] = _mean(
			result['background_twin_metrics']['horizons'][key]
			['background_over_real_action_distance']['estimate']
			for result in results
		)
	return {
		'conditions': conditions,
		'background_over_real_action_distance': twins,
	}


def aggregate(args) -> dict[str, Any]:
	results_root = args.results_root.resolve()
	_require(results_root.is_dir() and not results_root.is_symlink(),
		f'Invalid results root: {results_root}')
	paths = sorted(results_root.glob('seed_*.json'))
	_require(len(paths) == len(EXPECTED_SEEDS),
		f'Expected exactly three result files, found {len(paths)}.')
	results = [_load(path) for path in paths]
	seeds = tuple(sorted(int(row['protocol']['seed']) for row in results))
	_require(seeds == EXPECTED_SEEDS, f'Unexpected fit seeds: {seeds!r}.')
	for row in results:
		protocol = row['protocol']
		fit_seed = int(protocol['seed'])
		_require(protocol.get('formal_split_counts') == evaluator.FORMAL_SPLIT_COUNTS,
			'Formal 50/10/20 split lock changed across results.')
		_require(protocol.get('hidden_dim') == evaluator.FORMAL_HIDDEN_DIM,
			'Formal hidden dimension changed across results.')
		_require(protocol.get('bootstrap_resamples')
			== evaluator.FORMAL_BOOTSTRAP_RESAMPLES,
			'Formal bootstrap resample count changed across results.')
		_require(protocol.get('bootstrap_seed') == (
			fit_seed + evaluator.FORMAL_BOOTSTRAP_SEED_OFFSET
		), 'Formal bootstrap seed mapping changed across results.')
		_require(protocol.get('fit_models') == list(evaluator.FIT_MODES),
			'Formal fit-model family changed across results.')
	reference_splits = results[0]['root_splits']
	_require(all(row['root_splits'] == reference_splits for row in results),
		'Fit seeds do not share the exact same root splits.')
	_require(all(row.get('task') == args.task for row in results),
		'Result task mismatch.')

	source_fields = ('dataset', 'dataset_sha256', 'runtime_config',
		'runtime_config_sha256', 'checkpoint', 'checkpoint_sha256')
	reference_source = results[0]['source']
	for row in results[1:]:
		_require(all(row['source'].get(key) == reference_source.get(key)
			for key in source_fields), 'Fit seeds do not share immutable inputs.')
	for path_key, sha_key in (
		('dataset', 'dataset_sha256'),
		('runtime_config', 'runtime_config_sha256'),
		('checkpoint', 'checkpoint_sha256'),
	):
		path = Path(str(reference_source[path_key])).resolve()
		_require(path.is_file() and not path.is_symlink(),
			f'Bound input is missing or symlinked: {path}')
		_require(_sha256(path) == reference_source[sha_key],
			f'Bound input changed after evaluation: {path}')

	passes = sum(bool(row['action_identifiable_representation_candidate'])
		for row in results)
	task_candidate = passes >= REQUIRED_PASSES
	payload = {
		'format': FORMAT,
		'status': STATUS,
		'engineering_pass': all(row.get('engineering_pass') is True for row in results),
		'task': args.task,
		'fit_seeds': list(seeds),
		'formal_protocol_lock': {
			'split_counts': dict(evaluator.FORMAL_SPLIT_COUNTS),
			'hidden_dim': evaluator.FORMAL_HIDDEN_DIM,
			'bootstrap_resamples': evaluator.FORMAL_BOOTSTRAP_RESAMPLES,
			'bootstrap_seed_by_fit_seed': {
				str(seed): seed + evaluator.FORMAL_BOOTSTRAP_SEED_OFFSET
				for seed in EXPECTED_SEEDS
			},
			'fit_models': list(evaluator.FIT_MODES),
		},
		'required_seed_passes': REQUIRED_PASSES,
		'candidate_seed_count': int(passes),
		'task_candidate': bool(task_candidate),
		'policy_training_performed': False,
		'controller_training_authorized': False,
		'causal_scope': (
			'real_same_state_interventional_representation_and_dynamics_diagnostic'
		),
		'recommendation': (
			'independent_review_then_separate_small_controller_pilot'
			if task_candidate else
			'do_not_scale_controller_fix_or_limit_visual_object_state'
		),
		'source': {key: reference_source[key] for key in source_fields},
		'result_records': [{
			'seed': int(row['protocol']['seed']),
			'relative_path': path.relative_to(results_root.parent).as_posix(),
			'sha256': _sha256(path),
			'candidate': bool(row['action_identifiable_representation_candidate']),
			'all_fits_converged': bool(row['gates']['all_fits_converged']),
			'nonlinear_capacity_control_pass': bool(
				row['gates']['nonlinear_capacity_control']['all_pass']
			),
		} for path, row in zip(paths, results)],
		'mean_point_estimates_across_fit_seeds': _metric_means(results),
		'decision_rule': 'at_least_2_of_3_fixed_fit_seeds_pass_every_gate',
		'interpretation_guard': (
			'Per-seed confidence intervals remain root-clustered; this aggregation '
			'applies a replication rule and does not pool roots across fit seeds.'
		),
	}
	validate_summary(payload)
	return payload


def validate_summary(payload: Mapping) -> None:
	_require(payload.get('format') == FORMAT and payload.get('status') == STATUS,
		'Summary format/status mismatch.')
	_require(payload.get('policy_training_performed') is False,
		'Policy training is forbidden.')
	_require(payload.get('controller_training_authorized') is False,
		'Aggregator must never authorize controller training.')
	_require(payload.get('fit_seeds') == list(EXPECTED_SEEDS),
		'Fixed seed set changed.')
	_require(payload.get('formal_protocol_lock') == {
		'split_counts': dict(evaluator.FORMAL_SPLIT_COUNTS),
		'hidden_dim': evaluator.FORMAL_HIDDEN_DIM,
		'bootstrap_resamples': evaluator.FORMAL_BOOTSTRAP_RESAMPLES,
		'bootstrap_seed_by_fit_seed': {
			str(seed): seed + evaluator.FORMAL_BOOTSTRAP_SEED_OFFSET
			for seed in EXPECTED_SEEDS
		},
		'fit_models': list(evaluator.FIT_MODES),
	}, 'Formal protocol lock changed.')
	_require(payload.get('task_candidate') is (
		int(payload.get('candidate_seed_count', -1)) >= REQUIRED_PASSES
	), 'Two-of-three decision is inconsistent.')
	_require(len(payload.get('result_records', ())) == len(EXPECTED_SEEDS),
		'Result record count mismatch.')
	json.dumps(payload, allow_nan=False)


def _atomic_json(path: Path, payload: Mapping) -> None:
	path = path.resolve()
	if path.exists():
		raise FileExistsError(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
	try:
		temporary.write_text(
			json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		os.replace(temporary, path)
	except Exception:
		temporary.unlink(missing_ok=True)
		raise


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--results-root', type=Path, required=True)
	parser.add_argument('--task', choices=('acrobot-swingup', 'cartpole-balance-sparse'),
		required=True)
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None) -> int:
	args = parse_args(argv)
	payload = aggregate(args)
	_atomic_json(args.output, payload)
	print(f'ROF_REAL_ACTION_BRANCH_AGGREGATE_COMPLETE task={payload["task"]}')
	print(f'TASK_CANDIDATE={payload["task_candidate"]}')
	print(f'OUTPUT={args.output.resolve()}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
