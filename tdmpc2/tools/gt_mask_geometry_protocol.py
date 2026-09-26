"""Bind and aggregate the seed-7 GT-mask geometry diagnostic protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


FORMAT_INPUTS = 'gt_mask_geometry_seed7_inputs_v1'
FORMAT_SUMMARY = 'gt_mask_geometry_seed7_pilot_v1'
TASKS = ('reacher-visual-small', 'cartpole-swingup')
ARMS = ('cutie_mask_geometry', 'gt_mask_geometry')
ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cartpole-swingup': ('cart', 'pole'),
}
SCHEMAS = {
	'cutie_mask_geometry': 'cutie_mask_geometry_v1',
	'gt_mask_geometry': 'simulator_gt_mask_geometry_v1',
}
SEED = 7
STEPS = 100000
EVAL_FREQ = 20000
EVAL_EPISODES = 3
HELDOUT_EPISODES = 20
ENV_SEED = 424243
BACKGROUND_SEED = 1618034
PLANNER_SEED_BASE = 8675400
TRAIN_FRAMES = 109218
EVAL_FRAMES = 10020
GT_MIN_ROLE_VALID_RATE = 0.95
GT_MAX_ROLE_INVALID_BURST = 5
PAIR_FIELDS = (
	'episode_index',
	'initial_rgb_sha256',
	'background_source',
	'background_start_frame_index',
	'planner_seed',
	'planner_rng_start_sha256',
	'planner_rng_end_sha256',
	'length',
)
ANCHOR_REGRESSION_FIELDS = PAIR_FIELDS + (
	'initial_object_sha256', 'reward', 'success',
)
IMPLEMENTATION = (
	'tdmpc2/config.yaml',
	'tdmpc2/train.py',
	'tdmpc2/tdmpc2.py',
	'tdmpc2/common/buffer.py',
	'tdmpc2/common/layers.py',
	'tdmpc2/common/parser.py',
	'tdmpc2/common/seed.py',
	'tdmpc2/common/world_model.py',
	'tdmpc2/envs/__init__.py',
	'tdmpc2/envs/dmcontrol.py',
	'tdmpc2/envs/wrappers/cutie_object.py',
	'tdmpc2/envs/wrappers/gt_mask_oracle.py',
	'tdmpc2/envs/wrappers/tensor.py',
	'tdmpc2/envs/wrappers/timeout.py',
	'tdmpc2/envs/wrappers/video_background.py',
	'tdmpc2/perception/cutie_oc_adapter.py',
	'tdmpc2/trainer/online_trainer.py',
	'tdmpc2/tools/collect_cutie_multitask_support.py',
	'tdmpc2/check_gt_mask_geometry_contract.py',
	'tdmpc2/check_cutie_object_wrapper_contract.py',
	'tdmpc2/check_cutie_object_only_integration_contract.py',
	'tdmpc2/check_cutie_object_only_update.py',
	'tdmpc2/check_cutie_multitask_support_contract.py',
	'tdmpc2/tools/check_gt_mask_geometry_env.py',
	'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
	'tdmpc2/tools/evaluate_gt_mask_geometry_oracle.py',
	'tdmpc2/tools/gt_mask_geometry_protocol.py',
	'tdmpc2/tools/run_gt_mask_geometry_seed7_pilot.sh',
)
EXTERNAL_CUTIE = (
	'feature_extractor/cutie/cutie/inference/inference_core.py',
	'feature_extractor/cutie/cutie/inference/memory_manager.py',
	'feature_extractor/cutie/cutie/inference/object_manager.py',
	'feature_extractor/cutie/cutie/inference/image_feature_store.py',
	'feature_extractor/cutie/cutie/inference/kv_memory_store.py',
	'feature_extractor/cutie/cutie/model/transformer/object_transformer.py',
	'feature_extractor/cutie/cutie/model/cutie.py',
)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _json(path: Path) -> dict:
	value = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(value, dict):
		raise ValueError(f'Expected JSON object: {path}')
	return value


def _write(path: Path, value) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + '.incomplete')
	if path.exists() or temporary.exists():
		raise FileExistsError(path)
	try:
		with temporary.open('x', encoding='utf-8', newline='\n') as file:
			json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
			file.write('\n')
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _file_entry(path: Path, root: Path | None = None) -> dict:
	resolved = path.resolve()
	return {
		'path': str(resolved),
		'relative': (
			resolved.relative_to(root.resolve()).as_posix() if root is not None else None
		),
		'size': resolved.stat().st_size,
		'sha256': _sha256(resolved),
	}


def _tree_files(root: Path) -> dict[str, dict]:
	root = root.resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	return {
		path.relative_to(root).as_posix(): {
			'size': path.stat().st_size,
			'sha256': _sha256(path),
		}
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def _python_tree_files(root: Path) -> dict[str, dict]:
	root = root.resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	return {
		path.relative_to(root).as_posix(): {
			'size': path.stat().st_size,
			'sha256': _sha256(path),
		}
		for path in sorted(root.rglob('*.py'))
		if path.is_file()
	}


def _same_path(left, right: Path) -> bool:
	try:
		return Path(left).resolve() == right.resolve()
	except (TypeError, OSError):
		return False


def _require(condition, message):
	if not condition:
		raise ValueError(message)


def _valid_sha(value) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _validate_episode_rows(rows, *, task: str) -> list[float]:
	_require(isinstance(rows, list) and len(rows) == HELDOUT_EPISODES,
		f'{task}: expected 20 evaluation episodes')
	rewards = []
	for index, row in enumerate(rows):
		_require(isinstance(row, dict), f'{task}: malformed episode row')
		_require(row.get('episode_index') == index, f'{task}: episode index mismatch')
		_require(row.get('length') == 500, f'{task}: episode length mismatch')
		_require(_valid_sha(row.get('initial_rgb_sha256')),
			f'{task}: initial RGB hash missing')
		_require(_valid_sha(row.get('initial_object_sha256')),
			f'{task}: initial object hash missing')
		for key in ('planner_rng_start_sha256', 'planner_rng_end_sha256'):
			_require(_valid_sha(row.get(key)), f'{task}: {key} missing')
		reward = row.get('reward')
		_require(isinstance(reward, (int, float)) and math.isfinite(float(reward)),
			f'{task}: non-finite reward')
		rewards.append(float(reward))
	return rewards


def _evaluation_protocol(payload, *, task: str, expected_format: str) -> None:
	protocol = payload.get('evaluation', {})
	_require(payload.get('format') == expected_format, f'{task}: evaluator format')
	_require(payload.get('task') == task, f'{task}: evaluator task')
	_require(payload.get('backend') == 'cutie_object_only', f'{task}: backend')
	_require(payload.get('training_seed') == SEED, f'{task}: training seed')
	_require(payload.get('erosion_pixels') == 0, f'{task}: erosion')
	_require(
		protocol.get('split') == 'validation'
		and protocol.get('episodes') == HELDOUT_EPISODES
		and protocol.get('env_seed') == ENV_SEED
		and protocol.get('background_seed') == BACKGROUND_SEED
		and protocol.get('planner_seed_base') == PLANNER_SEED_BASE
		and protocol.get('object_only_alignment_draws') == 10000,
		f'{task}: held-out protocol mismatch',
	)


def _source_anchor(
	repo: Path,
	source: Path,
	task: str,
	*,
	support_path: Path,
	oc_repo: Path,
	cutie_checkpoint: Path,
	video_root: Path,
	manifest_dir: Path,
) -> dict:
	key = task.replace('-', '_')
	root = repo / 'logs' / task / str(SEED) / (
		f'cutie_object_hard_zero100k_cutie_object_memory_probe_100k_v1_seed7_{key}'
	)
	paths = {
		'runtime_config': root / 'runtime_config.json',
		'checkpoint': root / 'models' / 'final.pt',
		'historical_evaluation': (
			source / 'tasks' / task / 'evaluations' / 'normal' / 'hard_zero.json'
		),
	}
	for path in paths.values():
		if not path.is_file():
			raise FileNotFoundError(path)
	runtime = _json(paths['runtime_config'])
	evaluation = _json(paths['historical_evaluation'])
	_require(runtime.get('task') == task and runtime.get('seed') == SEED,
		f'{task}: source runtime identity')
	_require(
		runtime.get('steps') == STEPS
		and runtime.get('eval_freq') == EVAL_FREQ
		and runtime.get('eval_episodes') == EVAL_EPISODES
		and runtime.get('model_size') == 5,
		f'{task}: source schedule/model mismatch',
	)
	_require(
		runtime.get('video_background_enabled') is True
		and runtime.get('video_background_split') == 'train'
		and runtime.get('video_background_seed') == SEED
		and float(runtime.get('video_background_strength', -1)) == 1.0
		and runtime.get('video_background_total_frames') == 1000
		and runtime.get('video_background_source_cache_size') == 8
		and _same_path(runtime.get('video_background_root'), video_root)
		and _same_path(runtime.get('video_background_manifest_dir'), manifest_dir),
		f'{task}: source training-background mismatch',
	)
	_require(runtime.get('flat_anchor_mode') == 'cutie_object_only',
		f'{task}: source is not ObjectOnly')
	_require(
		runtime.get('cutie_object_observation_variant', 'full') == 'full'
		and runtime.get(
			'cutie_object_frame_schema', 'cutie_query_mask_status_v1'
		) == 'cutie_query_mask_status_v1'
		and runtime.get('cutie_object_allow_simulator_runtime', False) is False,
		f'{task}: source is not the established Full-Cutie observation',
	)
	_require(
		runtime.get('cutie_object_last_valid_memory', False) is False
		and runtime.get('cutie_object_policy_burst_plan') is None
		and runtime.get('cutie_object_belief_enabled', False) is False,
		f'{task}: source contains memory/burst/belief',
	)
	_require(
		runtime.get('cutie_object_support_schema') == 'generic_indexed_v1'
		and tuple(runtime.get('cutie_object_role_names', ())) == ROLES[task]
		and runtime.get('cutie_object_allow_simulator_support') is True
		and _same_path(runtime.get('cutie_object_support_path'), support_path),
		f'{task}: source support mismatch',
	)
	_require(
		_same_path(runtime.get('cutie_object_repo'), oc_repo)
		and _same_path(runtime.get('cutie_object_checkpoint'), cutie_checkpoint),
		f'{task}: source Cutie implementation/checkpoint mismatch',
	)
	_evaluation_protocol(
		evaluation, task=task,
		expected_format='cutie_multitask_checkpoint_evaluation_v1',
	)
	_validate_episode_rows(evaluation.get('episodes'), task=task)
	return {
		'root': str(root.resolve()),
		'artifacts': {name: _file_entry(path) for name, path in paths.items()},
		'historical_evaluator_sha256': evaluation.get('provenance', {}).get(
			'evaluator_sha256'
		),
	}


def bind(args) -> None:
	repo = args.repo.resolve()
	source = args.source_memory_root.resolve()
	support_base = args.support_base.resolve()
	manifest_dir = args.manifest_dir.resolve()
	oc_repo = args.oc_repo.resolve()
	checkpoint = args.cutie_checkpoint.resolve()
	for path in (repo, source, support_base, args.video_root, manifest_dir, oc_repo):
		if not path.exists():
			raise FileNotFoundError(path)
	if not checkpoint.is_file():
		raise FileNotFoundError(checkpoint)
	source_summary_path = source / 'memory_probe_summary.json'
	source_summary = _json(source_summary_path)
	_require(source_summary.get('status') == 'object_memory_probe_engineering_pass',
		'Source memory probe did not pass engineering gates.')
	gates = source_summary.get('engineering_gates')
	_require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()),
		'Source memory probe engineering gates are incomplete.')
	implementation = {}
	for relative in IMPLEMENTATION:
		path = repo / relative
		if not path.is_file():
			raise FileNotFoundError(path)
		implementation[relative] = _file_entry(path, repo)
	external = {}
	for relative in EXTERNAL_CUTIE:
		path = oc_repo / relative
		if not path.is_file():
			raise FileNotFoundError(path)
		external[relative] = _file_entry(path, oc_repo)
	config_root = oc_repo / 'feature_extractor/cutie/cutie/config'
	external_python_root = oc_repo / 'feature_extractor/cutie/cutie'
	if not config_root.is_dir():
		raise FileNotFoundError(config_root)
	if not external_python_root.is_dir():
		raise FileNotFoundError(external_python_root)
	supports = {}
	anchors = {}
	for task in TASKS:
		directory = support_base / task
		annotations = _json(directory / 'annotations.json')
		collection = annotations.get('collection', {})
		_require(
			annotations.get('format') == 'cutie_indexed_mask_support_v1'
			and tuple(annotations.get('roles', ())) == ROLES[task]
			and collection.get('support_schema') == 'generic_indexed_v1'
			and collection.get('task') == task
			and collection.get('label_policy')
				== 'simulator_segmentation_support_only',
			f'{task}: support identity mismatch',
		)
		supports[task] = {
			'root': str(directory.resolve()),
			'files': _tree_files(directory),
		}
		anchors[task] = _source_anchor(
			repo,
			source,
			task,
			support_path=directory / 'annotations.json',
			oc_repo=oc_repo,
			cutie_checkpoint=checkpoint,
			video_root=args.video_root.resolve(),
			manifest_dir=manifest_dir,
		)
	payload = {
		'format': FORMAT_INPUTS,
		'protocol': {
			'training_seed': SEED,
			'steps': STEPS,
			'eval_freq': EVAL_FREQ,
			'training_eval_episodes': EVAL_EPISODES,
			'heldout_episodes': HELDOUT_EPISODES,
			'env_seed': ENV_SEED,
			'background_seed': BACKGROUND_SEED,
			'planner_seed_base': PLANNER_SEED_BASE,
			'training_background': {
				'split': 'train', 'strength': 1.0, 'seed': SEED,
				'total_frames': 1000, 'source_cache_size': 8,
			},
			'gt_visibility_gate': {
				'min_role_valid_rate': GT_MIN_ROLE_VALID_RATE,
				'max_role_invalid_burst': GT_MAX_ROLE_INVALID_BURST,
				'scope': ['preflight', 'training', 'heldout'],
			},
			'heldout_condition': {
				'split': 'validation', 'dynamic_video_background': True,
				'foreground_erosion_pixels': 0, 'synthetic_burst': False,
				'not_clean_static_background': True,
			},
			'tasks': list(TASKS),
			'new_arms': list(ARMS),
			'comparison_scope': {
				'primary': 'gt_mask_geometry_minus_cutie_mask_geometry',
				'full_cutie': 'frozen_checkpoint_anchor_only',
				'rgb': 'not_rerun_in_this_tracker_geometry_diagnostic',
			},
			'scientific_scope': (
				'single-seed privileged diagnostic; GT means visible-surface '
				'simulator mask geometry, not a deployable method or perfect query oracle'
			),
		},
		'paths': {
			'repo': str(repo),
			'source_memory_root': str(source),
			'support_base': str(support_base),
			'video_root': str(args.video_root.resolve()),
			'manifest_dir': str(manifest_dir),
			'oc_repo': str(oc_repo),
		},
		'gpu_by_task': {
			'reacher-visual-small': str(args.gpu_reacher),
			'cartpole-swingup': str(args.gpu_cartpole),
		},
		'implementation': implementation,
		'local_python_sources': {
			'root': str((repo / 'tdmpc2').resolve()),
			'files': _python_tree_files(repo / 'tdmpc2'),
		},
		'external_cutie_sources': external,
		'external_cutie_python_sources': {
			'root': str(external_python_root.resolve()),
			'files': _python_tree_files(external_python_root),
		},
		'external_cutie_config': {
			'root': str(config_root.resolve()),
			'files': _tree_files(config_root),
		},
		'cutie_checkpoint': _file_entry(checkpoint),
		'manifests': _tree_files(manifest_dir),
		'video_files': _tree_files(args.video_root.resolve()),
		'supports': supports,
		'source_summary': _file_entry(source_summary_path),
		'anchors': anchors,
	}
	_write(args.output, payload)
	print('GT_MASK_GEOMETRY_INPUTS_OK', json.dumps({
		'output': str(args.output.resolve()), 'tasks': list(TASKS),
	}, allow_nan=False), flush=True)


def _read_rc(path: Path):
	try:
		return int(path.read_text(encoding='utf-8').strip())
	except (FileNotFoundError, ValueError):
		return None


def _runtime_diagnostics(value, expected_frames: int, roles, arm: str) -> dict:
	expected_privileged = arm == 'gt_mask_geometry'
	checks = {
		'object': isinstance(value, dict),
		'frames': isinstance(value, dict) and value.get('frames') == expected_frames,
		'schema': isinstance(value, dict) and value.get(
			'role_diagnostics_schema'
		) == 'cutie_role_runtime_diagnostics_v1',
		'roles': isinstance(value, dict) and set(value.get('role_metrics', {})) == set(roles),
		'worker': isinstance(value, dict) and value.get('worker_restarts') == 0,
		'timeouts': isinstance(value, dict) and value.get('timeouts') == 0,
		'variant': isinstance(value, dict) and value.get('observation_variant') == arm,
		'frame_schema': isinstance(value, dict) and value.get('frame_schema') == SCHEMAS[arm],
		'privileged': isinstance(value, dict) and value.get(
			'privileged_runtime_segmentation'
		) is expected_privileged,
	}
	if not isinstance(value, dict):
		return checks
	for name in ('valid_frame_rate', 'lost_role_rate'):
		number = value.get(name)
		checks[name] = isinstance(number, (int, float)) and math.isfinite(float(number)) and 0 <= float(number) <= 1
	episodes = value.get('episode_metrics')
	checks['episode_count'] = isinstance(episodes, list) and len(episodes) == (
		TRAIN_FRAMES // 501 if expected_frames == TRAIN_FRAMES else HELDOUT_EPISODES
	)
	checks['episode_frames'] = isinstance(episodes, list) and all(
		isinstance(row, dict) and row.get('frames') == 501 for row in episodes
	)
	role_metrics = value.get('role_metrics', {})
	checks['role_accounting'] = set(role_metrics) == set(roles) and all(
		isinstance(item, dict)
		and isinstance(item.get('valid_frames'), int)
		and isinstance(item.get('invalid_frames'), int)
		and item['valid_frames'] > 0
		and item['valid_frames'] + item['invalid_frames'] == expected_frames
		and item.get('nonfinite_feature_frames') == 0
		for item in role_metrics.values()
	)
	oracle = value.get('gt_mask_oracle', {})
	checks['query_exact_zero'] = oracle.get('query_feature_policy') == 'exact_zero_512_v1'
	if expected_privileged:
		visibility = oracle.get('visibility_failures', {})
		checks.update(
			oracle_enabled=(
				oracle.get('enabled') is True
				and oracle.get('format') == 'gt_mask_geometry_runtime_v1'
				and oracle.get('frames') == expected_frames
				and oracle.get('privileged_runtime_segmentation') is True
			),
			oracle_catalog=(
				isinstance(oracle.get('geom_catalog_path'), str)
				and _valid_sha(oracle.get('geom_catalog_sha256'))
				and oracle.get('camera_id') == 0
				and list(oracle.get('image_size', ())) == [64, 64]
			),
			visibility_accounting=(
				set(visibility) == set(roles)
				and all(
					visibility[role] == role_metrics[role].get('invalid_frames')
					== role_metrics[role].get('lost_frames')
					== role_metrics[role].get('empty_mask_frames')
					for role in roles
				)
			),
			oracle_visibility_quality=all(
				isinstance(role_metrics[role].get('valid_frame_rate'), (int, float))
				and math.isfinite(float(role_metrics[role]['valid_frame_rate']))
				and float(role_metrics[role]['valid_frame_rate'])
					>= GT_MIN_ROLE_VALID_RATE
				and isinstance(role_metrics[role].get('max_invalid_burst'), int)
				and role_metrics[role]['max_invalid_burst']
					<= GT_MAX_ROLE_INVALID_BURST
				for role in roles
			),
			no_cutie_worker_runtime=(
				value.get('runtime_unit')
					== 'milliseconds_per_same_state_segmentation_frame'
				and value.get('episode_reset_strategy')
					== 'stateless_same_frame_mujoco_segmentation_v1'
				and value.get('device_name') is None
				and value.get('logical_cuda_device') is None
			),
		)
	else:
		checks['oracle_disabled'] = oracle.get('enabled') is False
		checks['live_cutie_runtime'] = (
			value.get('runtime_unit')
				== 'milliseconds_per_tracked_frame_excluding_support_prompts'
			and isinstance(value.get('device_name'), str)
			and bool(value.get('device_name'))
			and value.get('logical_cuda_device') is not None
		)
	return checks


def _paired(left, right) -> dict:
	# Return right-left so callers name both the reference and candidate explicitly.
	deltas = [float(b) - float(a) for a, b in zip(left, right)]
	mean = statistics.fmean(deltas)
	median = statistics.median(deltas)
	std = statistics.stdev(deltas)
	half = 2.093024054408263 * std / math.sqrt(len(deltas))
	return {
		'mean_delta': mean,
		'median_delta': median,
		'sample_std': std,
		'conditional_paired_episode_95pct_t_interval_df19': [mean - half, mean + half],
		'win_tie_loss': [
			sum(value > 0 for value in deltas),
			sum(value == 0 for value in deltas),
			sum(value < 0 for value in deltas),
		],
		'deltas': deltas,
		'interval_scope': (
			'conditional paired-episode interval for one frozen training seed; '
			'not uncertainty across training seeds'
		),
	}


def _run_root(repo: Path, run_tag: str, task: str, arm: str) -> Path:
	key = task.replace('-', '_')
	experiment = f'cutie_object_{arm}100k_{run_tag}_seed7_{key}'
	return repo / 'logs' / task / str(SEED) / experiment


def _artifact(path: Path, repo: Path, stage: Path) -> dict:
	resolved = path.resolve()
	result = _file_entry(resolved)
	try:
		result['relative_to_summary_root'] = resolved.relative_to(stage.resolve()).as_posix()
	except ValueError:
		result['relative_to_summary_root'] = None
	try:
		result['relative_to_repo'] = resolved.relative_to(repo.resolve()).as_posix()
	except ValueError:
		result['relative_to_repo'] = None
	return result


def _rehash_inputs(inputs, repo: Path) -> list[str]:
	failures = []
	for relative, item in inputs.get('implementation', {}).items():
		path = repo / relative
		if not path.is_file() or _sha256(path) != item.get('sha256'):
			failures.append(f'implementation changed: {relative}')
	local_python = inputs.get('local_python_sources', {})
	if _python_tree_files(Path(local_python.get('root', ''))) != local_python.get('files'):
		failures.append('local Python source tree changed')
	for relative, item in inputs.get('external_cutie_sources', {}).items():
		path = Path(inputs['paths']['oc_repo']) / relative
		if not path.is_file() or _sha256(path) != item.get('sha256'):
			failures.append(f'external Cutie source changed: {relative}')
	external_python = inputs.get('external_cutie_python_sources', {})
	if _python_tree_files(Path(external_python.get('root', ''))) != external_python.get('files'):
		failures.append('external Cutie Python source tree changed')
	config = inputs.get('external_cutie_config', {})
	if _tree_files(Path(config.get('root', ''))) != config.get('files'):
		failures.append('external Cutie config tree changed')
	checkpoint = Path(inputs['cutie_checkpoint']['path'])
	if not checkpoint.is_file() or _sha256(checkpoint) != inputs['cutie_checkpoint'].get('sha256'):
		failures.append('Cutie checkpoint changed')
	manifest_root = Path(inputs['paths']['manifest_dir'])
	if _tree_files(manifest_root) != inputs.get('manifests'):
		failures.append('background manifests changed')
	video_root = Path(inputs['paths']['video_root'])
	if _tree_files(video_root) != inputs.get('video_files'):
		failures.append('dynamic background video contents changed')
	for task in TASKS:
		if _tree_files(Path(inputs['supports'][task]['root'])) != inputs['supports'][task]['files']:
			failures.append(f'{task}: support pack changed')
		for item in inputs['anchors'][task]['artifacts'].values():
			path = Path(item['path'])
			if not path.is_file() or _sha256(path) != item.get('sha256'):
				failures.append(f'{task}: source anchor artifact changed')
	source_summary = Path(inputs['source_summary']['path'])
	if not source_summary.is_file() or _sha256(source_summary) != inputs['source_summary'].get('sha256'):
		failures.append('source summary changed')
	return failures


def aggregate(args) -> int:
	import torch

	repo, stage = args.repo.resolve(), args.stage.resolve()
	inputs = _json(args.inputs)
	engineering_failures = []
	_require(inputs.get('format') == FORMAT_INPUTS, 'inputs format mismatch')
	engineering_failures.extend(_rehash_inputs(inputs, repo))
	expected_curve_steps = list(range(0, STEPS + 1, EVAL_FREQ))
	report = {
		'format': FORMAT_SUMMARY,
		'status': None,
		'scientific_scope': inputs['protocol']['scientific_scope'],
		'inputs_relative_to_summary_root': 'provenance/inputs.json',
		'tasks': {},
	}
	base_evaluator = repo / 'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py'
	new_evaluator = repo / 'tdmpc2/tools/evaluate_gt_mask_geometry_oracle.py'
	for task in TASKS:
		roles = ROLES[task]
		assigned_gpu = inputs['gpu_by_task'][task]
		directory = stage / 'tasks' / task
		task_report = {'roles': list(roles), 'arms': {}, 'pairing': {}, 'comparisons': {}}
		payloads = {}
		for arm in ARMS:
			root = _run_root(repo, args.run_tag, task, arm)
			preflight_rc = _read_rc(directory / 'preflight' / f'{arm}.rc')
			train_rc = _read_rc(directory / f'{arm}.train.rc')
			eval_rc = _read_rc(directory / 'evaluations' / f'{arm}.rc')
			arm_report = {
				'root': str(root), 'preflight_rc': preflight_rc,
				'training_rc': train_rc, 'evaluation_rc': eval_rc,
			}
			task_report['arms'][arm] = arm_report
			if (preflight_rc, train_rc, eval_rc) != (0, 0, 0):
				engineering_failures.append(
					f'{task}/{arm}: rc preflight={preflight_rc} train={train_rc} eval={eval_rc}'
				)
				continue
			paths = {
				'preflight': directory / 'preflight' / f'{arm}.json',
				'runtime': root / 'runtime_config.json',
				'curve': root / 'eval.csv',
				'checkpoint': root / 'models' / 'final.pt',
				'trainer': root / 'trainer_runtime.json',
				'replay': root / 'replay_runtime.json',
				'perception': root / 'perception_runtime.json',
				'evaluation': directory / 'evaluations' / f'{arm}.json',
			}
			missing = [str(path) for path in paths.values() if not path.is_file()]
			if missing:
				engineering_failures.append(f'{task}/{arm}: missing {missing}')
				continue
			try:
				preflight = _json(paths['preflight'])
				cfg = _json(paths['runtime'])
				trainer = _json(paths['trainer'])
				replay = _json(paths['replay'])
				perception = _json(paths['perception'])
				evaluation = _json(paths['evaluation'])
				with paths['curve'].open(encoding='utf-8', newline='') as file:
					rows = list(csv.DictReader(file))
				curve_steps = [int(float(row['step'])) for row in rows]
				curve_rewards = [float(row['episode_reward']) for row in rows]
				expected_privileged = arm == 'gt_mask_geometry'
				config_checks = {
					'task_seed_schedule': (
						cfg.get('task') == task and cfg.get('seed') == SEED
						and cfg.get('steps') == STEPS and cfg.get('eval_freq') == EVAL_FREQ
						and cfg.get('eval_episodes') == EVAL_EPISODES
					),
					'train_background': (
						cfg.get('video_background_enabled') is True
						and cfg.get('video_background_split') == 'train'
						and cfg.get('video_background_seed') == SEED
						and float(cfg.get('video_background_strength', -1)) == 1.0
						and _same_path(
							cfg.get('video_background_root'),
							Path(inputs['paths']['video_root']),
						)
						and cfg.get('video_background_total_frames') == 1000
						and cfg.get('video_background_source_cache_size') == 8
					),
					'model_and_compile': (
						cfg.get('model_size') == 5
						and cfg.get('compile') is True
						and cfg.get('visual_foreground_erosion_pixels') == 0
						and _same_path(
							cfg.get('video_background_manifest_dir'),
							Path(inputs['paths']['manifest_dir']),
						)
					),
					'object_only': (
						cfg.get('flat_anchor') is True
						and cfg.get('flat_anchor_mode') == 'cutie_object_only'
						and cfg.get('latent_dim') == 128
						and cfg.get('obs_shape') == {'object': [2, 1770]}
					),
					'object_dimensions': (
						cfg.get('cutie_object_num_roles') == 2
						and cfg.get('cutie_object_frame_dim') == 590
						and cfg.get('cutie_object_stack_frames') == 3
						and cfg.get('cutie_object_input_dim') == 1770
						and cfg.get('cutie_object_role_dim') == 64
						and cfg.get('cutie_object_only_latent_dim') == 128
					),
					'variant': cfg.get('cutie_object_observation_variant') == arm,
					'frame_schema': cfg.get('cutie_object_frame_schema') == SCHEMAS[arm],
					'privilege': cfg.get('cutie_object_allow_simulator_runtime') is expected_privileged,
					'no_memory_burst_belief': (
						cfg.get('cutie_object_last_valid_memory') is False
						and cfg.get('cutie_object_policy_burst_plan') is None
						and cfg.get('cutie_object_belief_enabled') is False
					),
					'support': (
						cfg.get('cutie_object_support_schema') == 'generic_indexed_v1'
						and tuple(cfg.get('cutie_object_role_names', ())) == roles
						and cfg.get('cutie_object_allow_simulator_support') is True
						and _same_path(cfg.get('cutie_object_support_path'), Path(inputs['supports'][task]['root']) / 'annotations.json')
					),
					'cutie_compatibility_inputs': (
						_same_path(cfg.get('cutie_object_repo'), Path(inputs['paths']['oc_repo']))
						and _same_path(
							cfg.get('cutie_object_checkpoint'),
							Path(inputs['cutie_checkpoint']['path']),
						)
					),
					'replay': replay.get('observation_keys') == ['object'] and replay.get('storage_device') == 'cuda:0',
					'throughput': isinstance(trainer.get('training_non_eval_steps_per_second'), (int, float)) and trainer.get('training_non_eval_steps_per_second', 0) > 0,
					'curve': curve_steps == expected_curve_steps and all(math.isfinite(value) for value in curve_rewards),
				}
				preflight_checks = preflight.get('checks', {})
				if (
					preflight.get('format') != 'gt_mask_geometry_environment_smoke_v1'
					or preflight.get('task') != task or preflight.get('variant') != arm
					or not isinstance(preflight_checks, dict) or not preflight_checks
					or not all(value is True for value in preflight_checks.values())
				):
					config_checks['preflight'] = False
				else:
					config_checks['preflight'] = True
				train_runtime_checks = _runtime_diagnostics(
					perception, TRAIN_FRAMES, roles, arm
				)
				checkpoint_payload = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
				state = checkpoint_payload.get('model') if isinstance(checkpoint_payload, dict) else None
				contract = checkpoint_payload.get('checkpoint_contract', {}) if isinstance(checkpoint_payload, dict) else {}
				observation_contract = contract.get('cutie_object_observation', {})
				keys = set(state) if isinstance(state, dict) else set()
				checkpoint_checks = {
					'state': isinstance(state, dict),
					'finite': isinstance(state, dict) and all(not torch.is_tensor(value) or bool(torch.isfinite(value).all()) for value in state.values()),
					'object_encoder': any(key.startswith('_encoder.object.') for key in keys),
					'no_rgb_encoder': not any(key.startswith('_encoder.rgb.') for key in keys),
					'observation_contract': observation_contract == {
						'format': 'cutie_object_observation_contract_v1',
						'variant': arm, 'frame_schema': SCHEMAS[arm],
						'privileged_runtime_segmentation': expected_privileged,
						'num_roles': 2, 'frame_dim': 590, 'stack_frames': 3,
						'input_dim': 1770,
					},
				}
				_evaluation_protocol(evaluation, task=task, expected_format='gt_mask_geometry_checkpoint_evaluation_v1')
				_require(evaluation.get('arm') == arm, f'{task}/{arm}: evaluation arm')
				rewards = _validate_episode_rows(evaluation.get('episodes'), task=task)
				provenance = evaluation.get('provenance', {})
				actual_inputs = provenance.get('actual_perception_inputs', {})
				observation = evaluation.get('observation_contract', {})
				observation_checks = (
					observation.get('checks', {}) if isinstance(observation, dict) else {}
				)
				eval_runtime = evaluation.get('perception_runtime')
				eval_runtime_checks = _runtime_diagnostics(
					eval_runtime, EVAL_FRAMES, roles, arm
				)
				evaluation_checks = {
					'source': (
						_same_path(provenance.get('runtime_config'), paths['runtime'])
						and provenance.get('runtime_config_sha256') == _sha256(paths['runtime'])
						and _same_path(provenance.get('checkpoint'), paths['checkpoint'])
						and provenance.get('checkpoint_sha256') == _sha256(paths['checkpoint'])
					),
					'evaluators': (
						provenance.get('evaluator_sha256') == _sha256(new_evaluator)
						and provenance.get('base_evaluator_sha256') == _sha256(base_evaluator)
					),
					'gpu': provenance.get('cuda_visible_devices') == assigned_gpu,
					'observation': (
						isinstance(observation, dict)
						and observation.get('variant') == arm
						and observation.get('frame_schema') == SCHEMAS[arm]
						and observation.get('role_frame_layout')
							== 'zero_query_512+mask_geometry_74+status_4'
						and observation.get('stack_frames') == 3
						and observation.get('privileged_runtime_segmentation')
							is expected_privileged
						and isinstance(observation_checks, dict)
						and bool(observation_checks)
						and all(value is True for value in observation_checks.values())
					),
					'perception_inputs': (
						actual_inputs.get('support_annotations_sha256')
							== inputs['supports'][task]['files']['annotations.json']['sha256']
						and actual_inputs.get('geom_catalog_sha256')
							== inputs['supports'][task]['files']['geom_catalog.json']['sha256']
						and actual_inputs.get('privileged_live_mujoco_segmentation')
							is expected_privileged
						and provenance.get('cutie_checkpoint_used_at_runtime')
							is (not expected_privileged)
						and (
							(
								actual_inputs.get('cutie_checkpoint_sha256') is None
								and actual_inputs.get('cutie_checkpoint') is None
							) if expected_privileged else (
								actual_inputs.get('cutie_checkpoint_sha256')
									== inputs['cutie_checkpoint']['sha256']
								and _same_path(
									actual_inputs.get('cutie_checkpoint'),
									Path(inputs['cutie_checkpoint']['path']),
								)
							)
						)
					),
				}
				all_groups = {
					'config': config_checks, 'training_runtime': train_runtime_checks,
					'checkpoint': checkpoint_checks, 'evaluation': evaluation_checks,
					'evaluation_runtime': eval_runtime_checks,
				}
				failed = {
					group: [name for name, passed in checks.items() if not passed]
					for group, checks in all_groups.items()
					if not all(checks.values())
				}
				if failed:
					engineering_failures.append(f'{task}/{arm}: checks {failed}')
				arm_report.update({
					'artifacts': {name: _artifact(path, repo, stage) for name, path in paths.items()},
					'checks': all_groups,
					'training_eval_steps': curve_steps,
					'training_eval_rewards': curve_rewards,
					'trainer_runtime': trainer,
					'training_perception_runtime': perception,
					'evaluation_perception_runtime': eval_runtime,
					'reward_mean': statistics.fmean(rewards),
					'reward_median': statistics.median(rewards),
					'reward_sample_std': statistics.stdev(rewards),
					'rewards': rewards,
				})
				payloads[arm] = evaluation
			except Exception as exc:
				engineering_failures.append(f'{task}/{arm}: parse/check failed: {exc}')

		# Freshly reevaluate the frozen Full-Cutie checkpoint under current code.
		full_rc = _read_rc(directory / 'evaluations' / 'full_cutie.rc')
		full_path = directory / 'evaluations' / 'full_cutie.json'
		task_report['full_cutie_anchor'] = {'evaluation_rc': full_rc}
		if full_rc != 0 or not full_path.is_file():
			engineering_failures.append(f'{task}/full_cutie: rc={full_rc} or missing JSON')
		else:
			try:
				full = _json(full_path)
				historical_path = Path(inputs['anchors'][task]['artifacts']['historical_evaluation']['path'])
				historical = _json(historical_path)
				_evaluation_protocol(full, task=task, expected_format='cutie_multitask_checkpoint_evaluation_v1')
				full_rewards = _validate_episode_rows(full.get('episodes'), task=task)
				historical_rewards = _validate_episode_rows(historical.get('episodes'), task=task)
				full_provenance = full.get('provenance', {})
				anchor_artifacts = inputs['anchors'][task]['artifacts']
				anchor_checks = {
					'exact_reward_regression': full_rewards == historical_rewards,
					'exact_episode_regression': all(
						all(
							a.get(field) == b.get(field)
							for field in ANCHOR_REGRESSION_FIELDS
						)
						for a, b in zip(full['episodes'], historical['episodes'])
					),
					'source': (
						full_provenance.get('runtime_config_sha256') == anchor_artifacts['runtime_config']['sha256']
						and full_provenance.get('checkpoint_sha256') == anchor_artifacts['checkpoint']['sha256']
						and full_provenance.get('evaluator_sha256') == _sha256(base_evaluator)
					),
					'gpu': full_provenance.get('cuda_visible_devices') == assigned_gpu,
				}
				if not all(anchor_checks.values()):
					engineering_failures.append(f'{task}/full_cutie: anchor checks {anchor_checks}')
				task_report['full_cutie_anchor'].update({
					'artifact': _artifact(full_path, repo, stage),
					'checks': anchor_checks,
					'reward_mean': statistics.fmean(full_rewards),
					'reward_median': statistics.median(full_rewards),
					'reward_sample_std': statistics.stdev(full_rewards),
					'rewards': full_rewards,
				})
				payloads['full_cutie'] = full
			except Exception as exc:
				engineering_failures.append(f'{task}/full_cutie parse/check failed: {exc}')

		if set(payloads) == set(('full_cutie',) + ARMS):
			mismatches = {
				field: [
					index for index, rows in enumerate(zip(*[
						payloads[name]['episodes'] for name in ('full_cutie', *ARMS)
					])) if len({row.get(field) for row in rows}) != 1
				]
				for field in PAIR_FIELDS
			}
			pairing_exact = not any(mismatches.values())
			device_names = {
				payloads[name].get('provenance', {}).get('device_name')
				for name in ('full_cutie', *ARMS)
			}
			manifest_hashes = {
				payloads[name].get('provenance', {}).get('validation_manifest_sha256')
				for name in ('full_cutie', *ARMS)
			}
			combined_manifest_hashes = {
				payloads[name].get('provenance', {}).get('combined_manifest_sha256')
				for name in ('full_cutie', *ARMS)
			}
			provenance_exact = (
				len(device_names) == 1 and None not in device_names
				and len(manifest_hashes) == 1
				and all(_valid_sha(value) for value in manifest_hashes)
				and len(combined_manifest_hashes) == 1
				and all(_valid_sha(value) for value in combined_manifest_hashes)
			)
			task_report['pairing'] = {
				'exact': pairing_exact and provenance_exact,
				'episode_fields_exact': pairing_exact,
				'provenance_exact': provenance_exact,
				'fields': list(PAIR_FIELDS),
				'mismatch_episode_indices': mismatches,
				'device_names': sorted(device_names, key=str),
				'validation_manifest_sha256': sorted(manifest_hashes, key=str),
				'combined_manifest_sha256': sorted(combined_manifest_hashes, key=str),
				'initial_object_hash_intentionally_excluded': True,
			}
			if not (pairing_exact and provenance_exact):
				engineering_failures.append(
					f'{task}: strict pairing/provenance mismatch {mismatches}'
				)
			full_rewards = task_report['full_cutie_anchor']['rewards']
			cutie_rewards = task_report['arms']['cutie_mask_geometry']['rewards']
			gt_rewards = task_report['arms']['gt_mask_geometry']['rewards']
			task_report['comparisons'] = {
				'gt_geometry_minus_cutie_geometry': _paired(cutie_rewards, gt_rewards),
				'full_cutie_minus_cutie_geometry': _paired(cutie_rewards, full_rewards),
				'gt_geometry_minus_full_cutie': _paired(full_rewards, gt_rewards),
			}
		else:
			engineering_failures.append(f'{task}: incomplete payload set {sorted(payloads)}')
		report['tasks'][task] = task_report

	engineering_pass = not engineering_failures
	candidates = {
		'tracker_bottleneck_candidate': False,
		'geometry_sufficient_candidate': False,
		'cutie_query_not_required_candidate': False,
		'representation_bottleneck_candidate': False,
	}
	if engineering_pass:
		full_means = {
			task: report['tasks'][task]['full_cutie_anchor']['reward_mean'] for task in TASKS
		}
		margins = {task: max(50.0, 0.1 * full_means[task]) for task in TASKS}
		cart = report['tasks']['cartpole-swingup']
		reacher = report['tasks']['reacher-visual-small']
		cart_tracker = cart['comparisons']['gt_geometry_minus_cutie_geometry']
		candidates['tracker_bottleneck_candidate'] = (
			cart_tracker['mean_delta'] >= margins['cartpole-swingup']
			and cart_tracker['win_tie_loss'][0] >= 12
			and cart['arms']['gt_mask_geometry']['reward_mean']
				>= 0.9 * full_means['cartpole-swingup']
			and reacher['arms']['gt_mask_geometry']['reward_mean']
				>= 0.95 * full_means['reacher-visual-small']
		)
		candidates['geometry_sufficient_candidate'] = all(
			report['tasks'][task]['arms']['gt_mask_geometry']['reward_mean']
			>= 0.95 * full_means[task] for task in TASKS
		)
		candidates['cutie_query_not_required_candidate'] = all(
			report['tasks'][task]['arms']['cutie_mask_geometry']['reward_mean']
			>= 0.95 * full_means[task] for task in TASKS
		)
		candidates['representation_bottleneck_candidate'] = all(
			abs(report['tasks'][task]['comparisons']['gt_geometry_minus_cutie_geometry']['mean_delta'])
			< margins[task]
			and report['tasks'][task]['arms']['gt_mask_geometry']['reward_mean']
			< 0.9 * full_means[task]
			and report['tasks'][task]['arms']['cutie_mask_geometry']['reward_mean']
			< 0.9 * full_means[task]
			for task in TASKS
		)
		report['development_interpretation'] = {
			'practical_margin_by_task': margins,
			**candidates,
			'eligible_to_expand_tracker_hypothesis_to_three_seeds': candidates['tracker_bottleneck_candidate'],
			'not_a_paper_go': True,
		}
	else:
		report['development_interpretation'] = {**candidates, 'not_interpretable': True, 'not_a_paper_go': True}

	if not engineering_pass:
		recommendation = 'engineering_fail_do_not_interpret_rewards'
	elif candidates['tracker_bottleneck_candidate']:
		recommendation = 'tracker_hypothesis_worth_three_seed_confirmation'
	elif candidates['representation_bottleneck_candidate']:
		recommendation = 'stop_tracker_work_and_revisit_representation_or_controller'
	else:
		recommendation = 'single_seed_mixed_or_inconclusive_do_not_scale_yet'
	report.update({
		'status': 'gt_mask_geometry_engineering_pass' if engineering_pass else 'gt_mask_geometry_engineering_fail',
		'engineering_pass': engineering_pass,
		'engineering_failures': engineering_failures,
		'recommendation': recommendation,
		'protocol': inputs['protocol'],
	})
	_write(args.output, report)
	print(json.dumps({
		'status': report['status'], 'engineering_pass': engineering_pass,
		'recommendation': recommendation, 'summary': str(args.output.resolve()),
	}, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
	return 0 if engineering_pass else 4


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest='command', required=True)
	binder = subparsers.add_parser('bind')
	binder.add_argument('--repo', type=Path, required=True)
	binder.add_argument('--source-memory-root', type=Path, required=True)
	binder.add_argument('--support-base', type=Path, required=True)
	binder.add_argument('--video-root', type=Path, required=True)
	binder.add_argument('--manifest-dir', type=Path, required=True)
	binder.add_argument('--oc-repo', type=Path, required=True)
	binder.add_argument('--cutie-checkpoint', type=Path, required=True)
	binder.add_argument('--gpu-reacher', required=True)
	binder.add_argument('--gpu-cartpole', required=True)
	binder.add_argument('--output', type=Path, required=True)
	aggregator = subparsers.add_parser('aggregate')
	aggregator.add_argument('--repo', type=Path, required=True)
	aggregator.add_argument('--stage', type=Path, required=True)
	aggregator.add_argument('--inputs', type=Path, required=True)
	aggregator.add_argument('--run-tag', required=True)
	aggregator.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	if args.command == 'bind':
		bind(args)
	else:
		raise SystemExit(aggregate(args))


if __name__ == '__main__':
	main()
