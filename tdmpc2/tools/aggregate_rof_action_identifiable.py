"""Strict aggregation for the ROF action-identifiability preflight.

The campaign is intentionally small and preregistered: four tasks, three fit
seeds, and the clean/hard causal-ladder datasets for every task.  This program
does not discover or prefer results.  It reads exactly the twelve paths implied
by ``TASKS`` and ``SEEDS``, revalidates every evaluator result, binds it to the
supplied immutable dataset manifests, and applies a per-task two-of-three seed
rule.

A successful aggregate can only recommend the next same-state interventional
preflight.  It can never authorize controller or policy training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tdmpc2.tools import evaluate_rof_action_identifiable as evaluator


FORMAT = 'rof_action_identifiable_campaign_summary_v1'
STATUS = 'rof_action_identifiable_campaign_complete'
DATASET_FORMAT = 'rof_causal_probe_dataset_v1'
TASKS = (
	'cartpole-balance-sparse',
	'finger-turn-easy',
	'reacher-easy',
	'walker-run',
)
SEEDS = (20260913, 20260917, 20260923)
CONDITIONS = ('clean', 'hard')
REQUIRED_SEED_PASSES = 2


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f'Cannot read JSON object {path}: {error}') from error
	_require(isinstance(value, dict), f'Expected a JSON object: {path}')
	return value


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	try:
		with path.open('rb') as stream:
			for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
				digest.update(block)
	except OSError as error:
		raise ValueError(f'Cannot hash {path}: {error}') from error
	return digest.hexdigest()


def dataset_manifest_path(dataset_root: Path, condition: str, task: str) -> Path:
	_require(condition in CONDITIONS, f'Unknown condition {condition!r}.')
	_require(task in TASKS, f'Unknown task {task!r}.')
	return (
		dataset_root.resolve() / f'full_{condition}' / 'datasets' / task
		/ 'dataset_manifest.json'
	)


def result_path(results_root: Path, task: str, seed: int) -> Path:
	_require(task in TASKS, f'Unknown task {task!r}.')
	_require(seed in SEEDS, f'Unknown seed {seed!r}.')
	return results_root.resolve() / task / f'seed_{seed}.json'


def _manifest_identity(path: Path, *, task: str, condition: str) -> dict[str, Any]:
	_require(path.is_file(), f'Dataset manifest is missing: {path}')
	payload = _read_json(path)
	_require(payload.get('format') == DATASET_FORMAT,
		f'Dataset format changed: {path}')
	_require(payload.get('task') == task, f'Dataset task changed: {path}')
	_require(payload.get('condition') == condition,
		f'Dataset condition changed: {path}')
	source = payload.get('source')
	_require(isinstance(source, dict), f'Dataset source is missing: {path}')
	for key in (
		'runtime_config', 'runtime_config_sha256', 'checkpoint',
		'checkpoint_sha256', 'checkpoint_step',
	):
		_require(key in source, f'Dataset source lacks {key}: {path}')
	return {
		'path': str(path.resolve()),
		'sha256': _sha256(path),
		'source': source,
	}


def load_dataset_bindings(dataset_root: Path) -> dict[str, dict[str, Any]]:
	bindings: dict[str, dict[str, Any]] = {}
	for task in TASKS:
		rows = {
			condition: _manifest_identity(
				dataset_manifest_path(dataset_root, condition, task),
				task=task, condition=condition,
			)
			for condition in CONDITIONS
		}
		clean_source = rows['clean']['source']
		hard_source = rows['hard']['source']
		for key in (
			'runtime_config', 'runtime_config_sha256', 'checkpoint',
			'checkpoint_sha256', 'checkpoint_step',
		):
			_require(clean_source[key] == hard_source[key],
				f'{task}: clean/hard source disagree on {key}.')
		for key in ('runtime_config', 'checkpoint'):
			path = Path(clean_source[key])
			_require(path.is_absolute() and path.is_file(),
				f'{task}: source {key} is not an existing absolute file: {path}')
			expected = clean_source[f'{key}_sha256']
			_require(_sha256(path) == expected,
				f'{task}: source {key} hash differs from the manifest.')
		bindings[task] = {
			'clean_dataset': rows['clean'],
			'hard_dataset': rows['hard'],
			'runtime_config': str(Path(clean_source['runtime_config']).resolve()),
			'runtime_config_sha256': clean_source['runtime_config_sha256'],
			'checkpoint': str(Path(clean_source['checkpoint']).resolve()),
			'checkpoint_sha256': clean_source['checkpoint_sha256'],
			'checkpoint_step': clean_source['checkpoint_step'],
		}
	return bindings


def _validate_result_binding(
	payload: Mapping[str, Any], *, path: Path, task: str, seed: int,
	binding: Mapping[str, Any],
) -> None:
	evaluator.validate_result(payload)
	_require(payload.get('task') == task,
		f'{path}: task differs from selected task {task}.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('seed') == seed,
		f'{path}: fit seed differs from selected seed {seed}.')
	_require(payload.get('controller_training_authorized') is False,
		f'{path}: diagnostic attempted to authorize controller training.')
	_require(payload.get('policy_training_performed') is False,
		f'{path}: policy training was performed.')
	source = payload.get('source', {})
	expected = {
		'clean_dataset': binding['clean_dataset']['path'],
		'clean_dataset_sha256': binding['clean_dataset']['sha256'],
		'hard_dataset': binding['hard_dataset']['path'],
		'hard_dataset_sha256': binding['hard_dataset']['sha256'],
		'runtime_config': binding['runtime_config'],
		'runtime_config_sha256': binding['runtime_config_sha256'],
		'checkpoint': binding['checkpoint'],
		'checkpoint_sha256': binding['checkpoint_sha256'],
		'checkpoint_step': binding['checkpoint_step'],
	}
	for key, value in expected.items():
		actual = source.get(key)
		if key.endswith('dataset') or key in ('runtime_config', 'checkpoint'):
			_require(isinstance(actual, str) and Path(actual).resolve() == Path(value),
				f'{path}: source binding differs for {key}.')
		else:
			_require(actual == value, f'{path}: source binding differs for {key}.')


def _condition_gate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
	# Keep the complete preregistered gate tree (including every per-horizon
	# Boolean), while rejecting non-JSON or non-finite content.
	encoded = json.dumps(row, sort_keys=True, allow_nan=False)
	value = json.loads(encoded)
	_require(isinstance(value, dict), 'Condition gate row is not an object.')
	return value


def aggregate(
	results_root: Path, dataset_root: Path, *,
	published_results_root: Path | None = None,
) -> dict[str, Any]:
	results_root = results_root.resolve()
	dataset_root = dataset_root.resolve()
	published_results_root = (
		results_root if published_results_root is None
		else published_results_root.resolve()
	)
	_require(results_root.is_dir(), f'Results root is missing: {results_root}')
	_require(dataset_root.is_dir(), f'Dataset root is missing: {dataset_root}')
	bindings = load_dataset_bindings(dataset_root)
	per_task: dict[str, Any] = {}
	result_records: list[dict[str, Any]] = []
	for task in TASKS:
		seed_rows: dict[str, Any] = {}
		passes = 0
		for seed in SEEDS:
			path = result_path(results_root, task, seed)
			_require(path.is_file(), f'Expected result is missing: {path}')
			payload = _read_json(path)
			_validate_result_binding(
				payload, path=path, task=task, seed=seed, binding=bindings[task],
			)
			gates = payload['gates']
			both = gates.get('both_conditions_candidate') is True
			passes += int(both)
			published_path = result_path(published_results_root, task, seed)
			seed_rows[str(seed)] = {
				'path': str(published_path), 'sha256': _sha256(path),
				'engineering_pass': payload.get('engineering_pass') is True,
				'observational_fit_complete': payload.get(
					'observational_fit_complete'
				) is True,
				'scientific_complete': False,
				'causal_claim_authorized': False,
				'both_conditions_candidate': both,
				'by_condition': {
					condition: _condition_gate_summary(gates['by_condition'][condition])
					for condition in CONDITIONS
				},
				'recommendation': payload.get('recommendation'),
			}
			result_records.append({
				'task': task, 'seed': seed, 'path': str(published_path),
				'sha256': _sha256(path),
			})
		task_pass = passes >= REQUIRED_SEED_PASSES
		per_task[task] = {
			'seeds': seed_rows,
			'candidate_seed_count': passes,
			'required_seed_count': REQUIRED_SEED_PASSES,
			'two_of_three_pass': task_pass,
		}
	passed_tasks = [task for task in TASKS if per_task[task]['two_of_three_pass']]
	all_tasks_pass = len(passed_tasks) == len(TASKS)
	all_observational_fits_complete = all(
		row['observational_fit_complete']
		for task in per_task.values() for row in task['seeds'].values()
	)
	payload = {
		'format': FORMAT,
		'status': STATUS,
		'engineering_pass': True,
		'observational_fit_complete': all_observational_fits_complete,
		'scientific_complete': False,
		'causal_claim_authorized': False,
		'development_candidate': all_tasks_pass,
		'controller_training_authorized': False,
		'policy_training_performed': False,
		'true_same_state_intervention_required_before_controller_training': True,
		'recommendation': (
			'run_true_same_state_action_branch_preflight_only'
			if all_tasks_pass else
			'do_not_train_controller_action_identifiability_not_replicated'
		),
		'protocol': {
			'tasks': list(TASKS), 'fit_seeds': list(SEEDS),
			'conditions': list(CONDITIONS),
			'per_task_seed_rule': 'at_least_2_of_3_both_conditions_candidate',
			'required_seed_passes': REQUIRED_SEED_PASSES,
			'campaign_rule': 'all_four_tasks_must_pass_the_per_task_seed_rule',
			'controller_training_authorization': 'forbidden',
		},
		'source': {
			'dataset_root': str(dataset_root),
			'published_results_root': str(published_results_root),
			'dataset_bindings': bindings,
			'result_records': result_records,
		},
		'per_task': per_task,
		'aggregate': {
			'task_count': len(TASKS), 'seed_count_per_task': len(SEEDS),
			'result_count': len(result_records),
			'passed_task_count': len(passed_tasks),
			'passed_tasks': passed_tasks,
			'failed_tasks': [task for task in TASKS if task not in passed_tasks],
			'all_tasks_two_of_three_pass': all_tasks_pass,
		},
		'interpretation_guard': (
			'This observational diagnostic may justify only a true same-state real-action '
			'branching experiment. It never authorizes controller training.'
		),
	}
	validate_summary(payload)
	return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
	_require(payload.get('format') == FORMAT and payload.get('status') == STATUS,
		'Campaign summary format/status is incomplete.')
	_require(payload.get('engineering_pass') is True,
		'Campaign engineering pass is missing.')
	_require(payload.get('scientific_complete') is False,
		'Observational campaign must remain scientifically incomplete.')
	_require(payload.get('causal_claim_authorized') is False,
		'Observational campaign must not authorize a causal claim.')
	_require(payload.get('controller_training_authorized') is False,
		'Campaign must never authorize controller training.')
	_require(payload.get('policy_training_performed') is False,
		'Campaign must never perform policy training.')
	_require(payload.get(
		'true_same_state_intervention_required_before_controller_training'
	) is True, 'Same-state intervention requirement is missing.')
	protocol = payload.get('protocol', {})
	_require(protocol.get('tasks') == list(TASKS), 'Task protocol changed.')
	_require(protocol.get('fit_seeds') == list(SEEDS), 'Seed protocol changed.')
	_require(protocol.get('conditions') == list(CONDITIONS),
		'Condition protocol changed.')
	_require(protocol.get('required_seed_passes') == REQUIRED_SEED_PASSES,
		'Seed replication rule changed.')
	per_task = payload.get('per_task', {})
	_require(set(per_task) == set(TASKS), 'Per-task result set is incomplete.')
	for task in TASKS:
		row = per_task[task]
		_require(set(row.get('seeds', {})) == {str(seed) for seed in SEEDS},
			f'{task}: seed set is incomplete.')
		passes = sum(
			seed_row.get('both_conditions_candidate') is True
			for seed_row in row['seeds'].values()
		)
		_require(row.get('candidate_seed_count') == passes,
			f'{task}: candidate count is inconsistent.')
		_require(row.get('two_of_three_pass') == (passes >= REQUIRED_SEED_PASSES),
			f'{task}: two-of-three result is inconsistent.')
	aggregate_row = payload.get('aggregate', {})
	_require(aggregate_row.get('result_count') == len(TASKS) * len(SEEDS),
		'Campaign result count is incomplete.')
	expected_passed = [
		task for task in TASKS if per_task[task]['two_of_three_pass']
	]
	_require(aggregate_row.get('passed_tasks') == expected_passed,
		'Aggregate passed-task list is inconsistent.')
	_require(aggregate_row.get('passed_task_count') == len(expected_passed),
		'Aggregate passed-task count is inconsistent.')
	all_pass = len(expected_passed) == len(TASKS)
	_require(aggregate_row.get('all_tasks_two_of_three_pass') == all_pass,
		'Aggregate campaign decision is inconsistent.')
	_require(payload.get('development_candidate') == all_pass,
		'Development candidate decision is inconsistent.')
	json.dumps(payload, allow_nan=False)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
	path = path.resolve()
	if path.exists():
		raise FileExistsError(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
	try:
		with temporary.open('x', encoding='utf-8', newline='\n') as stream:
			json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
			stream.write('\n')
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--results-root', type=Path, required=True)
	parser.add_argument('--dataset-root', type=Path, required=True)
	parser.add_argument('--published-results-root', type=Path)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args(argv)
	payload = aggregate(
		args.results_root, args.dataset_root,
		published_results_root=args.published_results_root,
	)
	_atomic_json(args.output, payload)
	print('ROF_ACTION_IDENTIFIABLE_CAMPAIGN_COMPLETE')
	print(
		f'TASKS_PASSING_2_OF_3={payload["aggregate"]["passed_task_count"]}/'
		f'{payload["aggregate"]["task_count"]}'
	)
	print(f'DEVELOPMENT_CANDIDATE={payload["development_candidate"]}')
	print('CONTROLLER_TRAINING_AUTHORIZED=false')
	print(f'SUMMARY={args.output.resolve()}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
