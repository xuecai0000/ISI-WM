"""Safely reaggregate the completed seed-7 shadow-belief pilot.

The original V5 aggregation compared immutable memory-probe burst artifacts
against the *current* burst-evaluator hash.  Historical artifacts must instead
be bound to the evaluator hash recorded by their own successful source run.
This tool verifies that this is the only failed check, leaves the failed
archive untouched, and writes a separate post-hoc engineering summary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path


TASKS = ('reacher-visual-small', 'cartpole-swingup')
TASK_ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cartpole-swingup': ('cart', 'pole'),
}
FAILED_CHECK = 'anchor_burst_20_identity_protocol_provenance'
SOURCE_EVALUATOR = 'tdmpc2/tools/evaluate_cutie_multitask_policy_burst.py'
RUN_FORMAT = 'cutie_learned_belief_seed7_pilot_v5_shadow'
HISTORICAL_BURST_EVALUATOR_SHA256 = (
	'597b459e69207eb05bfeadc66981e33cbf70c1f7d774bf44dff62120f232f30b'
)
INCORRECT_CURRENT_BURST_EVALUATOR_SHA256 = (
	'0b6aec36caeda8b1ac51f03ff9549f0040b050641b0068e4b7fffe3dcd776747'
)
JOB_KEYS = {
	'train',
	'normal_measurement_only', 'normal_learned_prior',
	'burst_20_measurement_only', 'burst_20_learned_prior',
}
TRAINING_ARTIFACT_KEYS = {
	'runtime', 'checkpoint', 'curve', 'trainer', 'replay', 'perception',
}
EVALUATION_ARTIFACT_KEYS = JOB_KEYS - {'train'}
ARTIFACT_KEYS = TRAINING_ARTIFACT_KEYS | EVALUATION_ARTIFACT_KEYS
SHADOW_GATE_KEYS = {
	'measurement_normal_retention_at_least_95pct',
	'reacher_controlled_burst_attribution',
}
SOURCE_ENGINEERING_GATE_KEYS = {
	'both_gpu_compile_preflights', 'all_jobs_completed',
	'artifact_and_protocol_structure', 'runtime_health',
	'strict_normal_and_burst_pairing',
	'diagnostic_oracle_support_provenance',
}
BASE_TASK_CHECK_KEYS = {
	'runtime_shadow_control_false', 'runtime_belief_enabled',
	'runtime_memory_off', 'runtime_burst_off', 'runtime_protocol',
	'runtime_object_only', 'checkpoint_shadow_control_false',
	'checkpoint_shadow_collection_mode',
	'checkpoint_training_online_prior_zero', 'checkpoint_aux_updates',
	'checkpoint_age20_coverage', 'checkpoint_reacquisition_coverage',
	'checkpoint_skip_rate', 'replay_isolated', 'trainer_complete',
	'curve_exact', 'perception_process_healthy',
	'perception_role_diagnostics_exact',
}
EVALUATION_TASK_CHECK_KEYS = {
	f'{condition}_{arm}_{suffix}'
	for condition in ('normal', 'burst_20')
	for arm in ('measurement_only', 'learned_prior')
	for suffix in (
		'identity', 'same_checkpoint', 'strict', 'gpu', 'implementation',
		'online_prior_uses_zero' if arm == 'measurement_only'
		else 'online_prior_active',
	)
}
ANCHOR_TASK_CHECK_KEYS = {
	'anchor_runtime_config_immutable', 'anchor_checkpoint_immutable',
	'anchor_normal_immutable', 'anchor_burst_20_immutable',
	'anchor_plan_burst_20_immutable', 'anchor_support_immutable',
	'anchor_normal_identity_protocol_provenance',
	'anchor_burst_20_identity_protocol_provenance',
	'normal_hard_zero_pairing', 'burst_20_hard_zero_pairing',
	'normal_same_checkpoint_pairing', 'burst_20_same_checkpoint_pairing',
	'reacher_burst_attribution',
}
TASK_CHECK_KEYS = (
	BASE_TASK_CHECK_KEYS | EVALUATION_TASK_CHECK_KEYS | ANCHOR_TASK_CHECK_KEYS
)


def _require(condition, message):
	if not condition:
		raise RuntimeError(message)


def _load(path):
	value = json.loads(path.read_text(encoding='utf-8'))
	_require(isinstance(value, dict), f'JSON root is not an object: {path}')
	return value


def _sha256(path):
	hash_value = hashlib.sha256()
	with path.open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			hash_value.update(block)
	return hash_value.hexdigest()


def _nonnegative_int(value, upper=None):
	return (
		isinstance(value, int) and not isinstance(value, bool) and value >= 0
		and (upper is None or value <= upper)
	)


def _role_diagnostics_exact(value, roles):
	frames = 10020
	if not isinstance(value, dict) or value.get('frames') != frames:
		return False
	if value.get('role_diagnostics_schema') != 'cutie_role_runtime_diagnostics_v1':
		return False
	metrics = value.get('role_metrics')
	episodes = value.get('episode_metrics')
	if not isinstance(metrics, dict) or set(metrics) != set(roles):
		return False
	for role in roles:
		row = metrics[role]
		if not isinstance(row, dict):
			return False
		if not all(_nonnegative_int(row.get(key), frames) for key in (
			'valid_frames', 'invalid_frames', 'lost_frames',
			'empty_mask_frames', 'nonfinite_feature_frames',
		)):
			return False
		if row['valid_frames'] + row['invalid_frames'] != frames:
			return False
		for key in (
			'valid_frame_rate', 'lost_frame_rate', 'empty_mask_frame_rate',
			'nonfinite_feature_frame_rate', 'mask_touches_border_rate',
		):
			metric = row.get(key)
			if (
				type(metric) not in (int, float)
				or not math.isfinite(float(metric))
				or not 0.0 <= float(metric) <= 1.0
			):
				return False
		for key in ('mean_mask_area_pixels', 'mean_confidence', 'mean_mask_score'):
			metric = row.get(key)
			if type(metric) not in (int, float) or not math.isfinite(float(metric)):
				return False
		if not _nonnegative_int(row.get('max_invalid_burst'), frames):
			return False
	if not isinstance(episodes, list) or len(episodes) != 20:
		return False
	for index, row in enumerate(episodes):
		if not isinstance(row, dict):
			return False
		if row.get('episode_index') != index or row.get('frames') != 501:
			return False
		invalid = row.get('per_role_invalid_frames')
		bursts = row.get('per_role_max_invalid_burst')
		if not isinstance(invalid, dict) or set(invalid) != set(roles):
			return False
		if not isinstance(bursts, dict) or set(bursts) != set(roles):
			return False
		if not all(
			_nonnegative_int(invalid[role], 501)
			and _nonnegative_int(bursts[role], 501)
			for role in roles
		):
			return False
	return sum(row['frames'] for row in episodes) == frames


def _burst_anchor_exact(
	anchor, task, items, inputs, historical_evaluator, historical_base_evaluator
):
	provenance = anchor.get('provenance', {})
	evaluation = anchor.get('evaluation', {})
	policy = anchor.get('policy_burst', {})
	cutie = provenance.get('cutie_inputs', {})
	strict = anchor.get('strict_checks')
	episodes = anchor.get('episodes')
	return (
		anchor.get('format') == 'cutie_multitask_policy_burst_evaluation_v1'
		and anchor.get('task') == task
		and anchor.get('arm') == 'hard_zero'
		and anchor.get('backend') == 'cutie_object_only'
		and anchor.get('training_seed') == 7
		and evaluation.get('split') == 'validation'
		and evaluation.get('episodes') == 20
		and evaluation.get('env_seed') == 424243
		and evaluation.get('background_seed') == 1618034
		and evaluation.get('planner_seed_base') == 8675400
		and evaluation.get('foreground_erosion_pixels') == 0
		and isinstance(episodes, list) and len(episodes) == 20
		and [row.get('episode_index') for row in episodes] == list(range(20))
		and isinstance(strict, dict) and bool(strict) and all(strict.values())
		and policy.get('format') == 'cutie_policy_burst_plan_v1'
		and policy.get('length') == 20 and policy.get('episodes') == 20
		and policy.get('plan_sha256_before') == items['plan_burst_20']['sha256']
		and policy.get('plan_sha256_after') == items['plan_burst_20']['sha256']
		and policy.get('wrapper_plan_sha256') == items['plan_burst_20']['sha256']
		and provenance.get('runtime_config_sha256')
			== items['runtime_config']['sha256']
		and provenance.get('runtime_config_sha256_after')
			== items['runtime_config']['sha256']
		and provenance.get('checkpoint_sha256') == items['checkpoint']['sha256']
		and provenance.get('checkpoint_sha256_after')
			== items['checkpoint']['sha256']
		and provenance.get('policy_burst_plan_sha256')
			== items['plan_burst_20']['sha256']
		and provenance.get('evaluator_sha256') == historical_evaluator
		and provenance.get('base_evaluator_sha256') == historical_base_evaluator
		and cutie.get('checkpoint_sha256') == inputs['cutie_checkpoint']['sha256']
		and cutie.get('support_sha256') == items['support']['sha256']
		and cutie.get('roles') == list(TASK_ROLES[task])
		and cutie.get('support_schema') == 'generic_indexed_v1'
		and _role_diagnostics_exact(
			anchor.get('perception_runtime'), TASK_ROLES[task]
		)
	)


def _contained(path, root):
	try:
		path.resolve().relative_to(root.resolve())
	except ValueError:
		return False
	return True


def _resolve_artifact(failed, item, task, name, relocation):
	if name in EVALUATION_ARTIFACT_KEYS:
		relative = item.get('relative_to_summary_root')
		_require(isinstance(relative, str), f'{task}/{name} lacks stage-relative path.')
		path = (failed / relative).resolve()
		_require(_contained(path, failed), f'{task}/{name} escapes failed archive.')
		return path
	_require(name in TRAINING_ARTIFACT_KEYS, f'Unknown artifact key: {name}')
	original_path = Path(item.get('path', '')).resolve()
	original_root = Path(relocation['original_training_root']).resolve()
	try:
		relative = original_path.relative_to(original_root)
	except ValueError as exc:
		raise RuntimeError(
			f'{task}/{name} is outside its original training root.'
		) from exc
	path = (
		failed / relocation['archived_relative_to_summary_root'] / relative
	).resolve()
	_require(_contained(path, failed), f'{task}/{name} relocation escapes archive.')
	return path


def _parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--failed-archive', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path)
	return parser.parse_args(argv)


def main(argv=None):
	args = _parse_args(argv)
	failed = args.failed_archive.resolve()
	_require(failed.is_dir(), f'Failed archive does not exist: {failed}')
	_require(re.fullmatch(
		r'cutie_learned_belief_seed7_pilot_v5_shadow\.failed\.[^.]+\.\d+(?:\.\d+)?',
		failed.name,
	) is not None, f'Unexpected failed-archive basename: {failed.name}')
	output = (
		args.output_dir.resolve() if args.output_dir is not None
		else failed.parent / 'cutie_learned_belief_seed7_pilot_v5_shadow_reaggregation_v1'
	)
	stage = output.with_name(output.name + '.incomplete')
	_require(output != failed and failed not in output.parents, (
		'Output directory must be outside the immutable failed archive.'
	))
	_require(stage != failed and failed not in stage.parents, (
		'Staging directory must be outside the immutable failed archive.'
	))
	_require(not output.exists() and not stage.exists(), (
		f'Refusing existing output/stage: {output} / {stage}'
	))

	summary_path = failed / 'shadow_belief_seed7_summary.json'
	inputs_path = failed / 'provenance' / 'inputs.json'
	relocations_path = (
		failed / 'provenance' / 'failed_training_root_relocations.json'
	)
	summary = _load(summary_path)
	inputs = _load(inputs_path)
	relocations_payload = _load(relocations_path)
	relocations = relocations_payload.get('records')
	_require(isinstance(relocations, list), 'Relocation records are unavailable.')
	relocation_by_task = {
		row.get('task'): row for row in relocations if isinstance(row, dict)
	}
	source_path = Path(inputs['source_memory_summary']['path'])
	source = _load(source_path)

	source_gates = source.get('engineering_gates')
	checks = {
		'failed_summary_format': summary.get('format') == RUN_FORMAT,
		'failed_summary_status': summary.get('status') == 'shadow_engineering_fail',
		'failed_engineering_flag': summary.get('engineering_pass') is False,
		'inputs_format': inputs.get('format') == 'cutie_shadow_belief_inputs_v1',
		'inputs_sha_matches_failed_summary': (
			summary.get('inputs', {}).get('sha256') == _sha256(inputs_path)
		),
		'source_summary_immutable': (
			source_path.is_file()
			and _sha256(source_path) == inputs['source_memory_summary']['sha256']
		),
		'source_summary_engineering_pass': (
			source.get('format') == 'cutie_object_memory_probe_v1'
			and source.get('status') == 'object_memory_probe_engineering_pass'
			and isinstance(source_gates, dict)
			and set(source_gates) == SOURCE_ENGINEERING_GATE_KEYS
			and all(value is True for value in source_gates.values())
		),
		'relocation_manifest_bound': (
			relocations_payload.get('format')
				== 'cutie_shadow_failed_training_root_relocations_v1'
			and summary.get('failure_archive', {}).get(
				'training_root_relocations_relative_to_summary_root'
			) == 'provenance/failed_training_root_relocations.json'
			and relocations_payload.get('failed_archive_root') == str(failed)
			and len(relocations) == len(TASKS)
			and set(relocation_by_task) == set(TASKS)
		),
		'summary_tasks_exact': (
			isinstance(summary.get('tasks'), dict)
			and set(summary['tasks']) == set(TASKS)
		),
		'source_tasks_exact': (
			isinstance(source.get('tasks'), dict)
			and set(source['tasks']) == set(TASKS)
		),
	}
	expected_failures = {
		f'{task} checks failed: {[FAILED_CHECK]}' for task in TASKS
	}
	checks['only_expected_engineering_failures'] = (
		set(summary.get('engineering_failures', [])) == expected_failures
		and len(summary.get('engineering_failures', [])) == len(TASKS)
	)

	anchor_evidence = {}
	resolved_artifacts = {}
	for task in TASKS:
		report = summary.get('tasks', {}).get(task)
		_require(isinstance(report, dict), f'Missing task report: {task}')
		row = relocation_by_task.get(task)
		checks[f'{task}:training_root_relocated'] = (
			isinstance(row, dict) and row.get('moved') is True
			and row.get('source_state') == 'true'
			and row.get('archived_relative_to_summary_root')
				== f'training_roots/{task}/run'
			and Path(row.get('original_training_root', '')).resolve()
				== Path(report.get('new_root', '')).resolve()
			and (failed / row['archived_relative_to_summary_root']).is_dir()
			and _contained(
				failed / row['archived_relative_to_summary_root'], failed
			)
		)
		checks[f'{task}:all_jobs_zero'] = (
			isinstance(report.get('job_return_codes'), dict)
			and set(report['job_return_codes']) == JOB_KEYS
			and all(value == 0 for value in report['job_return_codes'].values())
			and report.get('missing') == []
			and report.get('exception') is None
		)
		task_checks = report.get('checks')
		checks[f'{task}:only_expected_failed_check'] = (
			isinstance(task_checks, dict)
			and set(task_checks) == TASK_CHECK_KEYS
			and report.get('failed_checks') == [FAILED_CHECK]
			and task_checks.get(FAILED_CHECK) is False
			and all(
				value is True for key, value in task_checks.items()
				if key != FAILED_CHECK
			)
		)
		shadow_gate = report.get('shadow_gate')
		checks[f'{task}:shadow_gate_schema_exact'] = (
			isinstance(shadow_gate, dict)
			and set(shadow_gate) == SHADOW_GATE_KEYS
			and all(isinstance(value, bool) for value in shadow_gate.values())
		)
		artifacts = report.get('artifacts')
		checks[f'{task}:artifact_schema_exact'] = (
			isinstance(artifacts, dict) and set(artifacts) == ARTIFACT_KEYS
		)
		items = inputs['hard_zero_anchors'][task]
		anchor_path = Path(items['burst_20']['path'])
		anchor = _load(anchor_path)
		historical_evaluator = source['input_provenance']['implementation'][
			SOURCE_EVALUATOR
		]['sha256']
		historical_base_evaluator = source['input_provenance']['implementation'][
			'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py'
		]['sha256']
		current_evaluator = inputs['implementation'][SOURCE_EVALUATOR]['sha256']
		source_report_sha = source['tasks'][task]['burst_conditions'][
			'burst_20'
		]['arms']['hard_zero']['sha256']
		checks[f'{task}:anchor_file_immutable'] = (
			_sha256(anchor_path) == items['burst_20']['sha256']
			== source_report_sha
		)
		checks[f'{task}:historical_evaluator_binding'] = (
			historical_evaluator == HISTORICAL_BURST_EVALUATOR_SHA256
			and current_evaluator == INCORRECT_CURRENT_BURST_EVALUATOR_SHA256
			and anchor.get('provenance', {}).get('evaluator_sha256')
			== historical_evaluator
			and anchor.get('provenance', {}).get('base_evaluator_sha256')
			== historical_base_evaluator
		)
		checks[f'{task}:known_bug_was_current_vs_historical_hash'] = (
			historical_evaluator != current_evaluator
			and anchor.get('provenance', {}).get('evaluator_sha256')
			!= current_evaluator
		)
		checks[f'{task}:corrected_anchor_identity'] = _burst_anchor_exact(
			anchor, task, items, inputs, historical_evaluator,
			historical_base_evaluator,
		)
		anchor_evidence[task] = {
			'path': str(anchor_path.resolve()),
			'sha256': _sha256(anchor_path),
			'historical_evaluator_sha256': historical_evaluator,
			'incorrect_current_evaluator_sha256': current_evaluator,
		}
		resolved_artifacts[task] = {}
		for name, item in artifacts.items():
			_require(isinstance(item, dict), f'Malformed artifact: {task}/{name}')
			path = _resolve_artifact(failed, item, task, name, row)
			checks[f'{task}:artifact:{name}'] = (
				path.is_file() and _sha256(path) == item.get('sha256')
			)
			try:
				relative = path.resolve().relative_to(failed).as_posix()
			except ValueError:
				relative = None
			resolved_artifacts[task][name] = {
				'sha256': item.get('sha256'),
				'source_failed_archive_path': str(path.resolve()),
				'relative_to_source_failed_archive': relative,
			}

	failed_checks = sorted(key for key, value in checks.items() if value is not True)
	_require(not failed_checks, 'Post-hoc validation failed: ' + ', '.join(failed_checks))

	result = copy.deepcopy(summary)
	result['status'] = 'shadow_engineering_pass_posthoc_reaggregation'
	result['engineering_pass'] = True
	result['engineering_failures'] = []
	for task in TASKS:
		result['tasks'][task]['checks'][FAILED_CHECK] = True
		result['tasks'][task]['failed_checks'] = []
		result['tasks'][task]['artifacts'] = resolved_artifacts[task]
		original_root = result['tasks'][task].pop('new_root', None)
		row = relocation_by_task[task]
		result['tasks'][task]['source_original_training_root'] = original_root
		result['tasks'][task]['source_training_root'] = str(
			(failed / row['archived_relative_to_summary_root']).resolve()
		)
	result['shadow_gate_pass'] = all(
		all(value is True for value in report['shadow_gate'].values())
		for report in result['tasks'].values()
	)
	result['recommendation'] = (
		'shadow_design_valid_compare_belief_effect'
		if result['shadow_gate_pass'] else 'shadow_controller_not_preserved'
	)
	result['inputs'] = {
		'source_failed_archive_path': str(inputs_path.resolve()),
		'relative_to_source_failed_archive': 'provenance/inputs.json',
		'sha256': _sha256(inputs_path),
	}
	result.pop('failure_archive', None)
	result['posthoc_reaggregation'] = {
		'format': 'cutie_shadow_belief_posthoc_reaggregation_v1',
		'scope': 'engineering provenance correction only; no training or evaluation rerun',
		'correction': (
			'Historical burst anchors are bound to the evaluator SHA recorded by '
			'their immutable successful memory-probe source summary, not to the '
			'new V5 evaluator file.'
		),
		'source_failed_archive': str(failed),
		'source_failed_summary_sha256': _sha256(summary_path),
		'source_inputs_sha256': _sha256(inputs_path),
		'source_memory_summary_sha256': _sha256(source_path),
		'source_relocation_manifest_sha256': _sha256(relocations_path),
		'reaggregation_tool': str(Path(__file__).resolve()),
		'reaggregation_tool_sha256': _sha256(Path(__file__).resolve()),
		'source_snapshots': {
			'failed_summary': {
				'relative_to_summary_root': 'source_snapshots/failed_summary.json',
				'sha256': _sha256(summary_path),
			},
			'inputs': {
				'relative_to_summary_root': 'source_snapshots/inputs.json',
				'sha256': _sha256(inputs_path),
			},
			'relocations': {
				'relative_to_summary_root': 'source_snapshots/relocations.json',
				'sha256': _sha256(relocations_path),
			},
			'source_memory_summary': {
				'relative_to_summary_root': (
					'source_snapshots/source_memory_summary.json'
				),
				'sha256': _sha256(source_path),
			},
		},
		'anchor_evidence': anchor_evidence,
		'validation_checks_relative_to_summary_root': 'provenance/reaggregation_checks.json',
	}
	result['summary_relative_paths_authoritative'] = False
	result['artifact_paths_authoritative'] = (
		'tasks.*.artifacts.*.source_failed_archive_path'
	)

	stage.mkdir(parents=False)
	(stage / 'provenance').mkdir()
	(stage / 'source_snapshots').mkdir()
	for source_file, relative in (
		(summary_path, 'source_snapshots/failed_summary.json'),
		(inputs_path, 'source_snapshots/inputs.json'),
		(relocations_path, 'source_snapshots/relocations.json'),
		(source_path, 'source_snapshots/source_memory_summary.json'),
	):
		with (stage / relative).open('xb') as file:
			file.write(source_file.read_bytes())
	checks_payload = {
		'format': 'cutie_shadow_belief_posthoc_reaggregation_checks_v1',
		'all_pass': True,
		'provenance': {
			'source_failed_archive': str(failed),
			'source_failed_summary_sha256': _sha256(summary_path),
			'source_inputs_sha256': _sha256(inputs_path),
			'source_memory_summary_sha256': _sha256(source_path),
			'source_relocation_manifest_sha256': _sha256(relocations_path),
			'reaggregation_tool_sha256': _sha256(Path(__file__).resolve()),
		},
		'checks': checks,
	}
	checks_bytes = (
		json.dumps(checks_payload, ensure_ascii=False, indent=2, allow_nan=False)
		+ '\n'
	).encode('utf-8')
	result['posthoc_reaggregation']['validation_checks_sha256'] = (
		hashlib.sha256(checks_bytes).hexdigest()
	)
	checks_path = stage / 'provenance' / 'reaggregation_checks.json'
	with checks_path.open('xb') as file:
		file.write(checks_bytes)
	result_path = stage / 'shadow_belief_seed7_summary.json'
	with result_path.open('x', encoding='utf-8', newline='\n') as file:
		json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
		file.write('\n')
	os.replace(stage, output)
	print('CUTIE_SHADOW_BELIEF_REAGGREGATION_COMPLETE')
	print(f'SUMMARY={output / "shadow_belief_seed7_summary.json"}')
	print(json.dumps({
		'status': result['status'],
		'engineering_pass': result['engineering_pass'],
		'shadow_gate_pass': result['shadow_gate_pass'],
		'recommendation': result['recommendation'],
	}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
	main()
