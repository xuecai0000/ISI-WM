"""Bind and aggregate the seed-7 Full-input auxiliary-target kill-test.

The new arm keeps Full-Cutie inference input and model architecture unchanged.
Only the training auxiliary target for current reconstruction and future
prediction changes from ``full_descriptor`` to
``geometry_status_full_denominator``. The frozen historical Full-Cutie model is
freshly reevaluated; it is not a contemporaneously retrained control.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
for local_path in (str(REPO_DIR), str(PROJECT_DIR)):
	while local_path in sys.path:
		sys.path.remove(local_path)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(1, str(PROJECT_DIR))

from common import cutie_object_auxiliary


FORMAT_INPUTS = 'full_input_geometry_loss_seed7_inputs_v1'
FORMAT_SUMMARY = 'full_input_geometry_loss_seed7_pilot_v1'
TASKS = ('reacher-visual-small', 'cartpole-swingup')
ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cartpole-swingup': ('cart', 'pole'),
}
THRESHOLDS = {
	'reacher-visual-small': 635.0,
	'cartpole-swingup': 420.0,
}
ARM = 'full_input_geometry_loss'
TARGET = 'geometry_status_full_denominator'
FRAME_SCHEMA = 'cutie_query_mask_status_v1'
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
MIN_ROLE_VALID_RATE = 0.50
MAX_ROLE_INVALID_BURST = 250
MAX_MS_PER_FRAME = 800.0
PAIR_FIELDS = (
	'episode_index',
	'initial_rgb_sha256',
	'initial_object_sha256',
	'background_source',
	'background_start_frame_index',
	'planner_seed',
	'planner_rng_start_sha256',
	'planner_rng_end_sha256',
	'length',
)
ANCHOR_REGRESSION_FIELDS = PAIR_FIELDS + ('reward', 'success')
IMPLEMENTATION = (
	'tdmpc2/config.yaml',
	'tdmpc2/train.py',
	'tdmpc2/tdmpc2.py',
	'tdmpc2/common/buffer.py',
	'tdmpc2/common/cutie_object_auxiliary.py',
	'tdmpc2/common/layers.py',
	'tdmpc2/common/parser.py',
	'tdmpc2/common/seed.py',
	'tdmpc2/common/world_model.py',
	'tdmpc2/envs/__init__.py',
	'tdmpc2/envs/dmcontrol.py',
	'tdmpc2/envs/wrappers/cutie_object.py',
	'tdmpc2/envs/wrappers/tensor.py',
	'tdmpc2/envs/wrappers/timeout.py',
	'tdmpc2/envs/wrappers/video_background.py',
	'tdmpc2/perception/cutie_oc_adapter.py',
	'tdmpc2/trainer/online_trainer.py',
	'tdmpc2/check_cutie_object_wrapper_contract.py',
	'tdmpc2/check_cutie_object_only_integration_contract.py',
	'tdmpc2/check_cutie_object_only_update.py',
	'tdmpc2/check_cutie_object_auxiliary_target_contract.py',
	'tdmpc2/check_cutie_multitask_support_contract.py',
	'tdmpc2/tools/check_full_input_geometry_loss_env.py',
	'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py',
	'tdmpc2/tools/evaluate_full_input_geometry_loss.py',
	'tdmpc2/tools/full_input_geometry_loss_protocol.py',
	'tdmpc2/tools/run_full_input_geometry_loss_seed7_pilot.sh',
)
IDENTITY_CONFIG_FIELDS = frozenset({'exp_name', 'work_dir'})
TREATMENT_CONFIG_FIELD = 'cutie_object_auxiliary_target'
# Fields introduced after the immutable Full-Cutie run. They are accepted only
# when absent from the source and exactly equal to these semantically inert
# values. Every field that existed in the historical runtime remains exact.
INERT_COMPATIBILITY_DEFAULTS = {
	'cutie_object_observation_variant': 'full',
	'cutie_object_frame_schema': FRAME_SCHEMA,
	'cutie_object_allow_simulator_runtime': False,
	'cutie_object_auxiliary_target': TARGET,
	'cutie_object_belief_enabled': False,
	'cutie_object_belief_use_for_control': False,
	'cutie_object_belief_batch_size': 128,
	'cutie_object_belief_burn_in': 3,
	'cutie_object_belief_min_burst': 5,
	'cutie_object_belief_max_burst': 20,
	'cutie_object_belief_recovery_frames': 1,
	'cutie_object_belief_update_frequency': 4,
	'cutie_object_belief_lr': 0.0003,
	'cutie_object_belief_loss_coef': 1.0,
	'cutie_object_belief_reacquisition_coef': 1.0,
	'cutie_object_belief_mask_seed_offset': 104729,
	'cutie_object_belief_replay_seed_offset': 130363,
}


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


def _require(condition, message):
	if not condition:
		raise ValueError(message)


def _valid_sha(value) -> bool:
	return (
		isinstance(value, str) and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _same_path(left, right: Path) -> bool:
	try:
		return Path(left).resolve() == right.resolve()
	except (TypeError, OSError):
		return False


def _file_entry(path: Path, root: Path | None = None) -> dict:
	resolved = path.resolve()
	return {
		'path': str(resolved),
		'relative': (
			resolved.relative_to(root.resolve()).as_posix()
			if root is not None else None
		),
		'size': resolved.stat().st_size,
		'sha256': _sha256(resolved),
	}


def _tree_files(root: Path, *, source_only=False) -> dict[str, dict]:
	root = root.resolve()
	if not root.is_dir():
		raise FileNotFoundError(root)
	files = {}
	for path in sorted(root.rglob('*')):
		if not path.is_file():
			continue
		# Full source tree: exclude only interpreter-generated bytecode. Configs,
		# shell entrypoints, package metadata, and non-Python source assets remain.
		if source_only and (
			'__pycache__' in path.parts
			or path.suffix.lower() in {'.pyc', '.pyo'}
		):
			continue
		files[path.relative_to(root).as_posix()] = {
			'size': path.stat().st_size,
			'sha256': _sha256(path),
		}
	return files


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
	_require(
		runtime.get('task') == task and runtime.get('seed') == SEED
		and runtime.get('steps') == STEPS and runtime.get('eval_freq') == EVAL_FREQ
		and runtime.get('eval_episodes') == EVAL_EPISODES
		and runtime.get('model_size') == 5,
		f'{task}: source identity/schedule mismatch',
	)
	_require(
		runtime.get('flat_anchor') is True
		and runtime.get('flat_anchor_mode') == 'cutie_object_only'
		and runtime.get('latent_dim') == 128,
		f'{task}: source is not the frozen Full-Cutie ObjectOnly arm',
	)
	_require(
		runtime.get('cutie_object_observation_variant', 'full') == 'full'
		and runtime.get('cutie_object_frame_schema', FRAME_SCHEMA) == FRAME_SCHEMA
		and runtime.get('cutie_object_allow_simulator_runtime', False) is False,
		f'{task}: source observation is not Full-Cutie',
	)
	_require(
		runtime.get('cutie_object_last_valid_memory', False) is False
		and runtime.get('cutie_object_policy_burst_plan') is None
		and runtime.get('cutie_object_belief_enabled', False) is False,
		f'{task}: source has a memory, burst, or belief treatment',
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
		f'{task}: source background mismatch',
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
		'effective_auxiliary_target': 'full_descriptor',
		'training_provenance': 'historical_noncontemporaneous_seed7',
		'artifacts': {name: _file_entry(path) for name, path in paths.items()},
	}


def bind(args) -> None:
	repo = args.repo.resolve()
	source = args.source_memory_root.resolve()
	support_base = args.support_base.resolve()
	video_root = args.video_root.resolve()
	manifest_dir = args.manifest_dir.resolve()
	oc_repo = args.oc_repo.resolve()
	cutie_checkpoint = args.cutie_checkpoint.resolve()
	for path in (repo, source, support_base, video_root, manifest_dir, oc_repo):
		if not path.exists():
			raise FileNotFoundError(path)
	if not cutie_checkpoint.is_file():
		raise FileNotFoundError(cutie_checkpoint)
	source_summary_path = source / 'memory_probe_summary.json'
	source_summary = _json(source_summary_path)
	_require(
		source_summary.get('status') == 'object_memory_probe_engineering_pass',
		'Source memory-probe engineering status is not pass.',
	)
	gates = source_summary.get('engineering_gates')
	_require(
		isinstance(gates, dict) and gates
		and all(value is True for value in gates.values()),
		'Source memory-probe gates are incomplete.',
	)
	implementation = {}
	for relative in IMPLEMENTATION:
		path = repo / relative
		if not path.is_file():
			raise FileNotFoundError(path)
		implementation[relative] = _file_entry(path, repo)
	config_root = oc_repo / 'feature_extractor/cutie/cutie/config'
	cutie_source_root = oc_repo / 'feature_extractor/cutie/cutie'
	for path in (config_root, cutie_source_root):
		if not path.is_dir():
			raise FileNotFoundError(path)
	supports, anchors = {}, {}
	for task in TASKS:
		directory = support_base / task
		annotations_path = directory / 'annotations.json'
		annotations = _json(annotations_path)
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
			repo, source, task, support_path=annotations_path,
			oc_repo=oc_repo, cutie_checkpoint=cutie_checkpoint,
			video_root=video_root, manifest_dir=manifest_dir,
		)
	expected_auxiliary = cutie_object_auxiliary.contract('full', TARGET)
	_require(
		expected_auxiliary == {
			'format': 'cutie_object_auxiliary_contract_v1',
			'target': TARGET,
			'effective_target': TARGET,
			'normalization': 'full_descriptor',
			'applies_to': ['current_reconstruction', 'future_prediction'],
			'decoder_output_dim': 1770,
			'query_values_per_frame': 512,
			'geometry_status_values_per_frame': 78,
			'supervised_values_per_frame': 78,
			'loss_denominator_values_per_role': 1770,
		},
		'Core auxiliary-target helper contract is unexpected.',
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
			'tasks': list(TASKS),
			'new_arm': ARM,
			'input_contract': {
				'variant': 'full', 'frame_schema': FRAME_SCHEMA,
				'role_shape': [2, 1770], 'full_input_preserved': True,
				'query_values_per_frame': 512,
				'query_input_masked_or_zeroed': False,
			},
			'auxiliary_target_contract': expected_auxiliary,
			'treatment_semantics': (
				'query targets are masked for both current reconstruction and future '
				'prediction after elementwise loss; the mean denominator remains 1770'
			),
			'architecture_semantics': (
				'encoder input, decoder output shape, decoder parameterization, and '
				'policy input are unchanged from Full-Cutie'
			),
			'heldout_condition': {
				'split': 'validation', 'dynamic_video_background': True,
				'foreground_erosion_pixels': 0, 'synthetic_burst': False,
				'not_clean_static_background': True,
			},
			'comparison_design': {
				'new_arm_training': 'current_run_seed7_100k',
				'full_cutie_training': 'historical_frozen_seed7_100k',
				'full_cutie_evaluation': 'fresh_exact_regression_required',
				'contemporaneous_baseline_training': False,
				'use': 'development_kill_test_only',
			},
			'decision_thresholds': THRESHOLDS,
			'tracker_availability_gate': {
				'minimum_valid_frame_rate_per_role': MIN_ROLE_VALID_RATE,
				'maximum_invalid_burst_per_role': MAX_ROLE_INVALID_BURST,
				'scope': ['preflight', 'training', 'heldout_new', 'heldout_anchor'],
				'purpose': (
					'lenient availability floor that admits known Cartpole pole tracking '
					'but rejects effectively absent roles'
				),
			},
			'runtime_health_gate': {
				'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
				'minimum_exclusive_ms_per_frame': 0.0,
				'maximum_inclusive_ms_per_frame': MAX_MS_PER_FRAME,
				'worker_restarts': 0,
				'timeouts': 0,
			},
			'scientific_scope': (
				'single-training-seed two-task development kill-test; historical Full-Cutie '
				'is non-contemporaneously trained; not training-seed uncertainty or paper GO'
			),
		},
		'paths': {
			'repo': str(repo), 'source_memory_root': str(source),
			'support_base': str(support_base), 'video_root': str(video_root),
			'manifest_dir': str(manifest_dir), 'oc_repo': str(oc_repo),
		},
		'gpu_by_task': {
			'reacher-visual-small': str(args.gpu_reacher),
			'cartpole-swingup': str(args.gpu_cartpole),
		},
		'implementation': implementation,
		'local_source_tree': {
			'root': str((repo / 'tdmpc2').resolve()),
			'exclusions': ['__pycache__ directories', '*.pyc', '*.pyo'],
			'files': _tree_files(repo / 'tdmpc2', source_only=True),
		},
		'external_cutie_source_tree': {
			'root': str(cutie_source_root.resolve()),
			'exclusions': ['__pycache__ directories', '*.pyc', '*.pyo'],
			'files': _tree_files(cutie_source_root, source_only=True),
		},
		'external_cutie_config_tree': {
			'root': str(config_root.resolve()),
			'files': _tree_files(config_root),
		},
		'cutie_checkpoint': _file_entry(cutie_checkpoint),
		'manifest_tree': {
			'root': str(manifest_dir), 'files': _tree_files(manifest_dir),
		},
		'video_tree': {'root': str(video_root), 'files': _tree_files(video_root)},
		'supports': supports,
		'source_summary': _file_entry(source_summary_path),
		'anchors': anchors,
	}
	_write(args.output, payload)
	print('FULL_INPUT_GEOMETRY_LOSS_INPUTS_OK', json.dumps({
		'output': str(args.output.resolve()), 'tasks': list(TASKS),
		'auxiliary_target': TARGET,
	}, allow_nan=False), flush=True)


def _read_rc(path: Path):
	try:
		return int(path.read_text(encoding='utf-8').strip())
	except (FileNotFoundError, ValueError):
		return None


def _gpu_contract(path: Path, *, expected_compile: bool) -> dict:
	if not path.is_file():
		raise FileNotFoundError(path)
	prefix = 'CUTIE_OBJECT_ONLY_UPDATE_OK '
	rows = [
		line[len(prefix):] for line in path.read_text(
			encoding='utf-8', errors='replace'
		).splitlines() if line.startswith(prefix)
	]
	if len(rows) != 1:
		raise ValueError(f'{path}: expected exactly one update-contract marker')
	value = ast.literal_eval(rows[0])
	if not isinstance(value, dict):
		raise ValueError(f'{path}: malformed update-contract payload')
	isolation = value.get('auxiliary_isolation', {})
	checks = {
		'compile_mode': value.get('compile') is expected_compile,
		'target': value.get('auxiliary_target') == TARGET,
		'contract': value.get('auxiliary_contract')
			== cutie_object_auxiliary.contract('full', TARGET),
		'pre_update_model_state_exact': isolation.get(
			'pre_update_model_state_exact'
		) is True,
		'parameter_counts_positive': (
			isinstance(isolation.get('model_parameter_count'), int)
			and isolation.get('model_parameter_count') > 0
			and isinstance(isolation.get('decoder_parameter_count'), int)
			and isolation.get('decoder_parameter_count') > 0
		),
		'decoder_output_shape': tuple(
			isolation.get('decoder_output_shape', ())
		) == (2, 2, 1770),
		'query_target_gradient_exact_zero': isolation.get(
			'query_gradient_nonzero'
		) == 0,
		'geometry_gradient_scale_exact': isolation.get(
			'geometry_gradient_max_abs_difference'
		) == 0.0,
		'masked_loss_full_denominator_exact': isolation.get(
			'masked_loss_full_denominator_exact'
		) is True,
		'full_descriptor_denominator': isolation.get(
			'loss_denominator_values_per_role'
		) == 1770,
		'query_still_active_in_encoder': (
			isinstance(isolation.get('encoder_query_delta'), (int, float))
			and math.isfinite(float(isolation['encoder_query_delta']))
			and isolation['encoder_query_delta'] > 0
		),
	}
	return {'checks': checks, 'payload': value}


def _rehash_inputs(inputs) -> list[str]:
	failures = []
	for relative, item in inputs.get('implementation', {}).items():
		path = Path(item['path'])
		if not path.is_file() or _sha256(path) != item.get('sha256'):
			failures.append(f'implementation changed: {relative}')
	for name, source_only in (
		('local_source_tree', True), ('external_cutie_source_tree', True),
		('external_cutie_config_tree', False), ('manifest_tree', False),
		('video_tree', False),
	):
		record = inputs.get(name, {})
		try:
			current = _tree_files(Path(record['root']), source_only=source_only)
		except (FileNotFoundError, KeyError, OSError) as exc:
			failures.append(f'{name} unavailable: {exc}')
		else:
			if current != record.get('files'):
				failures.append(f'{name} changed')
	checkpoint = Path(inputs['cutie_checkpoint']['path'])
	if not checkpoint.is_file() or _sha256(checkpoint) != inputs[
		'cutie_checkpoint'
	].get('sha256'):
		failures.append('Cutie checkpoint changed')
	for task in TASKS:
		record = inputs['supports'][task]
		if _tree_files(Path(record['root'])) != record.get('files'):
			failures.append(f'{task}: support tree changed')
		for name, item in inputs['anchors'][task]['artifacts'].items():
			path = Path(item['path'])
			if not path.is_file() or _sha256(path) != item.get('sha256'):
				failures.append(f'{task}: anchor {name} changed')
	source_summary = Path(inputs['source_summary']['path'])
	if not source_summary.is_file() or _sha256(source_summary) != inputs[
		'source_summary'
	].get('sha256'):
		failures.append('source summary changed')
	return failures


def _validate_episode_rows(rows, *, task: str) -> list[float]:
	_require(
		isinstance(rows, list) and len(rows) == HELDOUT_EPISODES,
		f'{task}: expected 20 evaluation episodes',
	)
	rewards = []
	for index, row in enumerate(rows):
		_require(isinstance(row, dict), f'{task}: malformed episode row')
		_require(row.get('episode_index') == index, f'{task}: episode index mismatch')
		_require(row.get('length') == 500, f'{task}: episode length mismatch')
		for field in (
			'initial_rgb_sha256', 'initial_object_sha256',
			'planner_rng_start_sha256', 'planner_rng_end_sha256',
		):
			_require(_valid_sha(row.get(field)), f'{task}: {field} missing')
		reward = row.get('reward')
		_require(
			isinstance(reward, (int, float)) and math.isfinite(float(reward)),
			f'{task}: non-finite reward',
		)
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


def _runtime_diagnostics(
	value, expected_frames: int, roles, *, expected_gpu: str, expected_device: str,
) -> dict:
	checks = {
		'object': isinstance(value, dict),
		'frames': isinstance(value, dict) and value.get('frames') == expected_frames,
		'schema': isinstance(value, dict) and value.get(
			'role_diagnostics_schema'
		) == 'cutie_role_runtime_diagnostics_v1',
		'roles': isinstance(value, dict) and set(
			value.get('role_metrics', {})
		) == set(roles),
		'worker': isinstance(value, dict) and value.get('worker_restarts') == 0,
		'timeouts': isinstance(value, dict) and value.get('timeouts') == 0,
		'variant': isinstance(value, dict) and value.get('observation_variant') == 'full',
		'frame_schema': isinstance(value, dict) and value.get(
			'frame_schema'
		) == FRAME_SCHEMA,
		'nonprivileged': isinstance(value, dict) and value.get(
			'privileged_runtime_segmentation'
		) is False,
		'episode_reset': isinstance(value, dict) and value.get(
			'episode_reset_strategy'
		) == 'fresh_inference_core_support_replay_v1',
		'physical_gpu_binding': (
			isinstance(value, dict)
			and value.get('cuda_visible_devices') == expected_gpu
			and value.get('logical_cuda_device') == 0
			and value.get('device_name') == expected_device
		),
	}
	if not isinstance(value, dict):
		return checks
	role_metrics = value.get('role_metrics', {})
	checks['finite_rates'] = all(
		isinstance(value.get(name), (int, float))
		and math.isfinite(float(value[name]))
		and 0.0 <= float(value[name]) <= 1.0
		for name in ('valid_frame_rate', 'lost_role_rate')
	)
	checks['finite_positive_runtime'] = (
		isinstance(value.get('ms_per_frame'), (int, float))
		and math.isfinite(float(value['ms_per_frame']))
		and 0.0 < float(value['ms_per_frame']) <= MAX_MS_PER_FRAME
	)
	checks['role_accounting'] = set(role_metrics) == set(roles) and all(
		isinstance(item, dict)
		and isinstance(item.get('valid_frames'), int)
		and isinstance(item.get('invalid_frames'), int)
		and item['valid_frames'] > 0
		and item['valid_frames'] + item['invalid_frames'] == expected_frames
		and item.get('nonfinite_feature_frames') == 0
		and isinstance(item.get('valid_frame_rate'), (int, float))
		and math.isfinite(float(item['valid_frame_rate']))
		and MIN_ROLE_VALID_RATE <= float(item['valid_frame_rate']) <= 1.0
		and isinstance(item.get('max_invalid_burst'), int)
		and 0 <= item['max_invalid_burst'] <= MAX_ROLE_INVALID_BURST
		for item in role_metrics.values()
	)
	episodes = value.get('episode_metrics')
	checks['episode_count'] = isinstance(episodes, list) and len(episodes) == (
		TRAIN_FRAMES // 501 if expected_frames == TRAIN_FRAMES else HELDOUT_EPISODES
	)
	checks['episode_frames'] = isinstance(episodes, list) and all(
		isinstance(row, dict) and row.get('frames') == 501 for row in episodes
	)
	checks['episode_accounting'] = isinstance(episodes, list) and all(
		isinstance(row, dict)
		and isinstance(row.get('invalid_frames_any_role'), int)
		and 0 <= row['invalid_frames_any_role'] <= 501
		and isinstance(row.get('max_invalid_burst_any_role'), int)
		and 0 <= row['max_invalid_burst_any_role'] <= 501
		and set(row.get('per_role_invalid_frames', {})) == set(roles)
		and set(row.get('per_role_max_invalid_burst', {})) == set(roles)
		and all(
			isinstance(number, int) and 0 <= number <= 501
			for number in row['per_role_invalid_frames'].values()
		)
		and all(
			isinstance(number, int) and 0 <= number <= 501
			for number in row['per_role_max_invalid_burst'].values()
		)
		for row in episodes
	)
	oracle = value.get('gt_mask_oracle', {})
	checks['query_input_policy'] = oracle.get(
		'query_feature_policy'
	) == 'cutie_query_mean_std_v1'
	checks['oracle_disabled'] = oracle.get('enabled') is False
	checks['live_cutie_runtime'] = (
		value.get('runtime_unit')
			== 'milliseconds_per_tracked_frame_excluding_support_prompts'
		and isinstance(value.get('device_name'), str)
		and bool(value.get('device_name'))
		and value.get('logical_cuda_device') is not None
	)
	return checks


def _config_equivalence(source: dict, candidate: dict) -> dict:
	differences = {}
	for key, source_value in source.items():
		if key in IDENTITY_CONFIG_FIELDS or key == TREATMENT_CONFIG_FIELD:
			continue
		if key not in candidate:
			differences[key] = {'source': source_value, 'candidate': '<missing>'}
		elif candidate[key] != source_value:
			differences[key] = {'source': source_value, 'candidate': candidate[key]}
	extra = {}
	for key in sorted(set(candidate) - set(source)):
		if key in IDENTITY_CONFIG_FIELDS or key == TREATMENT_CONFIG_FIELD:
			continue
		value = candidate[key]
		if key not in INERT_COMPATIBILITY_DEFAULTS or value != INERT_COMPATIBILITY_DEFAULTS[key]:
			extra[key] = value
	return {
		'exact_historical_fields_except_identity_and_treatment': not differences,
		'new_fields_are_explicit_inert_compatibility_defaults': not extra,
		'differences': differences,
		'unapproved_new_fields': extra,
		'identity_fields_excluded': sorted(IDENTITY_CONFIG_FIELDS),
		'treatment_field': TREATMENT_CONFIG_FIELD,
		'source_effective_treatment': source.get(
			TREATMENT_CONFIG_FIELD, 'full_descriptor'
		),
		'candidate_treatment': candidate.get(TREATMENT_CONFIG_FIELD),
	}


def _state_schema(state) -> dict:
	import torch

	if not isinstance(state, dict):
		return {}
	result = {}
	for name, value in sorted(state.items()):
		if torch.is_tensor(value):
			result[name] = {
				'kind': 'tensor', 'shape': list(value.shape),
				'dtype': str(value.dtype), 'values': value.numel(),
			}
		else:
			result[name] = {
				'kind': type(value).__name__, 'repr': repr(value),
			}
	return result


def _checkpoint(path: Path, *, expected_target: str) -> dict:
	import torch

	payload = torch.load(path, map_location='cpu', weights_only=False)
	state = payload.get('model') if isinstance(payload, dict) else None
	contract = payload.get('checkpoint_contract', {}) if isinstance(payload, dict) else {}
	observation = contract.get('cutie_object_observation')
	if observation is None and expected_target == 'full_descriptor':
		observation_effective = {
			'format': 'cutie_object_observation_contract_v1',
			'variant': 'full', 'frame_schema': FRAME_SCHEMA,
			'privileged_runtime_segmentation': False,
			'num_roles': 2, 'frame_dim': 590, 'stack_frames': 3,
			'input_dim': 1770,
		}
	else:
		observation_effective = observation
	auxiliary = contract.get('cutie_object_auxiliary')
	if auxiliary is None and expected_target == 'full_descriptor':
		auxiliary_effective = cutie_object_auxiliary.legacy_contract(
			observation_effective
		)
	else:
		auxiliary_effective = auxiliary
	expected_auxiliary = cutie_object_auxiliary.contract('full', expected_target)
	expected_observation = {
		'format': 'cutie_object_observation_contract_v1',
		'variant': 'full', 'frame_schema': FRAME_SCHEMA,
		'privileged_runtime_segmentation': False,
		'num_roles': 2, 'frame_dim': 590, 'stack_frames': 3,
		'input_dim': 1770,
	}
	keys = set(state) if isinstance(state, dict) else set()
	checks = {
		'state': isinstance(state, dict),
		'finite': isinstance(state, dict) and all(
			not torch.is_tensor(value) or bool(torch.isfinite(value).all())
			for value in state.values()
		),
		'object_encoder': any(key.startswith('_encoder.object.') for key in keys),
		'no_rgb_encoder': not any(key.startswith('_encoder.rgb.') for key in keys),
		'observation_contract': observation_effective == expected_observation,
		'auxiliary_contract': auxiliary_effective == expected_auxiliary,
	}
	schema = _state_schema(state)
	decoder_schema = {
		name: item for name, item in schema.items()
		if name.startswith('_object_decoder.')
	}
	checks['object_decoder'] = bool(decoder_schema)
	return {
		'checks': checks,
		'observation_contract': observation_effective,
		'auxiliary_contract': auxiliary_effective,
		'state_schema': schema,
		'decoder_state_schema': decoder_schema,
		'decoder_state_values': sum(
			item.get('values', 0) for item in decoder_schema.values()
		),
	}


def _paired(reference, candidate) -> dict:
	deltas = [float(b) - float(a) for a, b in zip(reference, candidate)]
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


def _run_root(repo: Path, run_tag: str, task: str) -> Path:
	key = task.replace('-', '_')
	return repo / 'logs' / task / str(SEED) / (
		f'cutie_object_{ARM}100k_{run_tag}_{key}'
	)


def _artifact(path: Path) -> dict:
	return _file_entry(path)


def aggregate(args) -> int:
	repo, stage = args.repo.resolve(), args.stage.resolve()
	inputs = _json(args.inputs)
	_require(inputs.get('format') == FORMAT_INPUTS, 'inputs format mismatch')
	engineering_failures = _rehash_inputs(inputs)
	expected_curve_steps = list(range(0, STEPS + 1, EVAL_FREQ))
	report = {
		'format': FORMAT_SUMMARY,
		'status': None,
		'engineering_pass': False,
		'scientific_scope': inputs['protocol']['scientific_scope'],
		'comparison_design': inputs['protocol']['comparison_design'],
		'inputs_relative_to_summary_root': 'provenance/inputs.json',
		'tasks': {},
	}
	contract_report = {}
	dependency_rc = _read_rc(stage / 'contracts' / 'dependency_light.rc')
	contract_report['dependency_light'] = {
		'rc': dependency_rc,
		'artifact': (
			_artifact(stage / 'contracts' / 'dependency_light.log')
			if (stage / 'contracts' / 'dependency_light.log').is_file() else None
		),
	}
	if dependency_rc != 0:
		engineering_failures.append(
			f'dependency-light contracts rc={dependency_rc}'
		)
	contract_devices = []
	for label, expected_compile in (
		('eager_gpu_reacher', False), ('compile_gpu_cartpole', True),
	):
		log = stage / 'contracts' / f'{label}.log'
		rc = _read_rc(stage / 'contracts' / f'{label}.rc')
		item = {'rc': rc, 'artifact': _artifact(log) if log.is_file() else None}
		if rc != 0:
			engineering_failures.append(f'{label} auxiliary GPU contract rc={rc}')
		else:
			try:
				parsed = _gpu_contract(log, expected_compile=expected_compile)
				item.update(parsed)
				failed = [
					name for name, passed in parsed['checks'].items() if not passed
				]
				if failed:
					engineering_failures.append(f'{label} checks failed: {failed}')
				contract_devices.append(parsed['payload'].get('device'))
			except Exception as exc:
				engineering_failures.append(f'{label} parse failed: {exc}')
		contract_report[label] = item
	contract_report['matched_physical_gpu_model'] = (
		len(contract_devices) == 2
		and len(set(contract_devices)) == 1
		and isinstance(contract_devices[0], str)
		and bool(contract_devices[0])
	)
	if not contract_report['matched_physical_gpu_model']:
		engineering_failures.append(
			f'GPU contract model mismatch: {contract_devices!r}'
		)
	report['contracts'] = contract_report
	base_evaluator = repo / 'tdmpc2/tools/evaluate_cutie_multitask_checkpoint.py'
	new_evaluator = repo / 'tdmpc2/tools/evaluate_full_input_geometry_loss.py'
	for task in TASKS:
		roles = ROLES[task]
		assigned_gpu = inputs['gpu_by_task'][task]
		contract_label = (
			'eager_gpu_reacher'
			if task == 'reacher-visual-small' else 'compile_gpu_cartpole'
		)
		expected_device = contract_report.get(contract_label, {}).get(
			'payload', {}
		).get('device')
		directory = stage / 'tasks' / task
		root = _run_root(repo, args.run_tag, task)
		task_report = {
			'threshold': THRESHOLDS[task], 'new_arm': {},
			'full_cutie_anchor': {}, 'pairing': {}, 'comparison': {},
		}
		new_eval = None
		anchor_eval = None
		preflight_rc = _read_rc(directory / 'preflight.rc')
		train_rc = _read_rc(directory / 'train.rc')
		eval_rc = _read_rc(directory / 'evaluations' / f'{ARM}.rc')
		task_report['new_arm'].update({
			'root': str(root), 'preflight_rc': preflight_rc,
			'training_rc': train_rc, 'evaluation_rc': eval_rc,
		})
		if (preflight_rc, train_rc, eval_rc) != (0, 0, 0):
			engineering_failures.append(
				f'{task}/{ARM}: rc preflight={preflight_rc} train={train_rc} eval={eval_rc}'
			)
		else:
			paths = {
				'preflight': directory / 'preflight.json',
				'runtime': root / 'runtime_config.json',
				'curve': root / 'eval.csv',
				'checkpoint': root / 'models' / 'final.pt',
				'trainer': root / 'trainer_runtime.json',
				'replay': root / 'replay_runtime.json',
				'perception': root / 'perception_runtime.json',
				'evaluation': directory / 'evaluations' / f'{ARM}.json',
			}
			missing = [str(path) for path in paths.values() if not path.is_file()]
			if missing:
				engineering_failures.append(f'{task}/{ARM}: missing {missing}')
			else:
				try:
					preflight = _json(paths['preflight'])
					cfg = _json(paths['runtime'])
					source_cfg = _json(Path(inputs['anchors'][task]['artifacts']['runtime_config']['path']))
					trainer = _json(paths['trainer'])
					replay = _json(paths['replay'])
					perception = _json(paths['perception'])
					new_eval = _json(paths['evaluation'])
					with paths['curve'].open(encoding='utf-8', newline='') as file:
						curve = list(csv.DictReader(file))
					curve_steps = [int(float(row['step'])) for row in curve]
					curve_rewards = [float(row['episode_reward']) for row in curve]
					equivalence = _config_equivalence(source_cfg, cfg)
					config_checks = {
						'task_seed_schedule': (
							cfg.get('task') == task and cfg.get('seed') == SEED
							and cfg.get('steps') == STEPS
							and cfg.get('eval_freq') == EVAL_FREQ
							and cfg.get('eval_episodes') == EVAL_EPISODES
						),
						'full_input_object_only': (
							cfg.get('flat_anchor') is True
							and cfg.get('flat_anchor_mode') == 'cutie_object_only'
							and cfg.get('latent_dim') == 128
							and cfg.get('obs_shape') == {'object': [2, 1770]}
							and cfg.get('cutie_object_observation_variant') == 'full'
							and cfg.get('cutie_object_frame_schema') == FRAME_SCHEMA
						),
						'auxiliary_target': cfg.get(
							'cutie_object_auxiliary_target'
						) == TARGET,
						'no_other_treatment': (
							cfg.get('cutie_object_allow_simulator_runtime') is False
							and cfg.get('cutie_object_last_valid_memory') is False
							and cfg.get('cutie_object_policy_burst_plan') is None
							and cfg.get('cutie_object_belief_enabled') is False
							and cfg.get('cutie_object_belief_use_for_control') is False
						),
						'config_exact_except_treatment_and_identity': (
							equivalence['exact_historical_fields_except_identity_and_treatment']
							and equivalence['new_fields_are_explicit_inert_compatibility_defaults']
							and equivalence['source_effective_treatment'] == 'full_descriptor'
							and equivalence['candidate_treatment'] == TARGET
						),
						'replay': (
							replay.get('observation_keys') == ['object']
							and replay.get('storage_device') == 'cuda:0'
						),
						'throughput': (
							trainer.get('steps') == STEPS
							and isinstance(trainer.get('training_non_eval_steps_per_second'), (int, float))
							and trainer.get('training_non_eval_steps_per_second', 0) > 0
						),
						'curve': (
							curve_steps == expected_curve_steps
							and all(math.isfinite(value) for value in curve_rewards)
						),
					}
					preflight_checks = preflight.get('checks', {})
					config_checks['preflight'] = (
						preflight.get('format')
							== 'full_input_geometry_loss_environment_smoke_v1'
						and preflight.get('task') == task
						and isinstance(preflight_checks, dict) and bool(preflight_checks)
						and all(value is True for value in preflight_checks.values())
						and preflight.get('auxiliary_target_contract')
							== cutie_object_auxiliary.contract('full', TARGET)
						and set(preflight.get('perception_runtime', {}).get(
							'role_metrics', {}
						)) == set(roles)
						and all(
							value > 0 for value in preflight.get(
								'observation_contract', {}
							).get('query_nonzero_descriptor_counts_by_role', {}).values()
						)
						and preflight.get('provenance', {}).get(
							'cuda_visible_devices'
						) == assigned_gpu
						and preflight.get('provenance', {}).get('device_name')
							== expected_device
						and preflight.get('provenance', {}).get(
							'cutie_ready', {}
						).get('current_device') == 0
						and preflight.get('provenance', {}).get(
							'cutie_ready', {}
						).get('cuda_visible_devices') == assigned_gpu
						and all(
							isinstance(item, dict)
							and isinstance(item.get('valid_frame_rate'), (int, float))
							and math.isfinite(float(item['valid_frame_rate']))
							and float(item['valid_frame_rate']) >= MIN_ROLE_VALID_RATE
							and isinstance(item.get('max_invalid_burst'), int)
							and item['max_invalid_burst'] <= MAX_ROLE_INVALID_BURST
							for item in preflight.get('perception_runtime', {}).get(
								'role_metrics', {}
							).values()
						)
					)
					training_runtime_checks = _runtime_diagnostics(
						perception, TRAIN_FRAMES, roles,
						expected_gpu=assigned_gpu, expected_device=expected_device,
					)
					new_checkpoint = _checkpoint(
						paths['checkpoint'], expected_target=TARGET
					)
					anchor_checkpoint = _checkpoint(
						Path(inputs['anchors'][task]['artifacts']['checkpoint']['path']),
						expected_target='full_descriptor',
					)
					architecture_checks = {
						'full_model_state_schema_unchanged': (
							new_checkpoint['state_schema']
							== anchor_checkpoint['state_schema']
						),
						'decoder_parameter_shapes_unchanged': (
							new_checkpoint['decoder_state_schema']
							== anchor_checkpoint['decoder_state_schema']
						),
						'decoder_parameter_value_count_unchanged': (
							new_checkpoint['decoder_state_values']
							== anchor_checkpoint['decoder_state_values']
							and new_checkpoint['decoder_state_values'] > 0
						),
						'decoder_output_dim_1770': new_checkpoint[
							'auxiliary_contract'
						].get('decoder_output_dim') == 1770,
					}
					_evaluation_protocol(
						new_eval, task=task,
						expected_format='full_input_geometry_loss_checkpoint_evaluation_v1',
					)
					rewards = _validate_episode_rows(new_eval.get('episodes'), task=task)
					provenance = new_eval.get('provenance', {})
					input_contract = new_eval.get('input_contract', {})
					auxiliary_contract = new_eval.get('auxiliary_target_contract', {})
					eval_runtime_checks = _runtime_diagnostics(
						new_eval.get('perception_runtime'), EVAL_FRAMES, roles,
						expected_gpu=assigned_gpu, expected_device=expected_device,
					)
					evaluation_checks = {
						'source': (
							provenance.get('runtime_config_sha256') == _sha256(paths['runtime'])
							and provenance.get('checkpoint_sha256') == _sha256(paths['checkpoint'])
						),
						'evaluators': (
							provenance.get('evaluator_sha256') == _sha256(new_evaluator)
							and provenance.get('base_evaluator_sha256') == _sha256(base_evaluator)
						),
						'gpu': provenance.get('cuda_visible_devices') == assigned_gpu,
						'full_query_input': (
							input_contract.get('variant') == 'full'
							and input_contract.get('full_input_preserved') is True
							and input_contract.get('query_values_per_frame') == 512
							and input_contract.get('query_input_masked_or_zeroed') is False
						),
						'auxiliary_contract': all(
							auxiliary_contract.get(key) == value
							for key, value in cutie_object_auxiliary.contract(
								'full', TARGET
							).items()
						),
						'perception_inputs': (
							provenance.get('actual_perception_inputs', {}).get(
								'support_annotations_sha256'
							) == inputs['supports'][task]['files']['annotations.json']['sha256']
							and provenance.get('actual_perception_inputs', {}).get(
								'cutie_checkpoint_sha256'
							) == inputs['cutie_checkpoint']['sha256']
						),
					}
					groups = {
						'config': config_checks,
						'training_runtime': training_runtime_checks,
						'new_checkpoint': new_checkpoint['checks'],
						'anchor_checkpoint': anchor_checkpoint['checks'],
						'architecture': architecture_checks,
						'evaluation': evaluation_checks,
						'evaluation_runtime': eval_runtime_checks,
					}
					failed = {
						group: [name for name, passed in checks.items() if not passed]
						for group, checks in groups.items() if not all(checks.values())
					}
					if failed:
						engineering_failures.append(f'{task}/{ARM}: checks {failed}')
					task_report['new_arm'].update({
						'artifacts': {name: _artifact(path) for name, path in paths.items()},
						'checks': groups,
						'config_equivalence': equivalence,
						'training_eval_steps': curve_steps,
						'training_eval_rewards': curve_rewards,
						'trainer_runtime': trainer,
						'training_perception_runtime': perception,
						'evaluation_perception_runtime': new_eval.get('perception_runtime'),
						'decoder_architecture': {
							'shape_and_parameter_count_unchanged': all(
								architecture_checks.values()
							),
							'decoder_state_values': new_checkpoint['decoder_state_values'],
							'output_dim': 1770,
						},
						'reward_mean': statistics.fmean(rewards),
						'reward_median': statistics.median(rewards),
						'reward_sample_std': statistics.stdev(rewards),
						'rewards': rewards,
					})
				except Exception as exc:
					engineering_failures.append(f'{task}/{ARM}: parse/check failed: {exc}')

		anchor_rc = _read_rc(directory / 'evaluations' / 'full_cutie.rc')
		anchor_path = directory / 'evaluations' / 'full_cutie.json'
		task_report['full_cutie_anchor']['evaluation_rc'] = anchor_rc
		if anchor_rc != 0 or not anchor_path.is_file():
			engineering_failures.append(
				f'{task}/full_cutie: rc={anchor_rc} or evaluation JSON missing'
			)
		else:
			try:
				anchor_eval = _json(anchor_path)
				historical_path = Path(inputs['anchors'][task]['artifacts']['historical_evaluation']['path'])
				historical = _json(historical_path)
				_evaluation_protocol(
					anchor_eval, task=task,
					expected_format='cutie_multitask_checkpoint_evaluation_v1',
				)
				anchor_rewards = _validate_episode_rows(anchor_eval.get('episodes'), task=task)
				historical_rewards = _validate_episode_rows(historical.get('episodes'), task=task)
				provenance = anchor_eval.get('provenance', {})
				artifacts = inputs['anchors'][task]['artifacts']
				anchor_runtime_checks = _runtime_diagnostics(
					anchor_eval.get('perception_runtime'), EVAL_FRAMES, roles,
					expected_gpu=assigned_gpu, expected_device=expected_device,
				)
				anchor_checks = {
					'exact_reward_regression': anchor_rewards == historical_rewards,
					'exact_episode_regression': all(
						all(a.get(field) == b.get(field) for field in ANCHOR_REGRESSION_FIELDS)
						for a, b in zip(anchor_eval['episodes'], historical['episodes'])
					),
					'source': (
						provenance.get('runtime_config_sha256')
							== artifacts['runtime_config']['sha256']
						and provenance.get('checkpoint_sha256')
							== artifacts['checkpoint']['sha256']
						and provenance.get('evaluator_sha256') == _sha256(base_evaluator)
					),
					'gpu': provenance.get('cuda_visible_devices') == assigned_gpu,
					'runtime': all(anchor_runtime_checks.values()),
				}
				if not all(anchor_checks.values()):
					engineering_failures.append(
						f'{task}/full_cutie anchor checks {anchor_checks}'
					)
				task_report['full_cutie_anchor'].update({
					'training_provenance': 'historical_noncontemporaneous_seed7',
					'fresh_evaluation': True,
					'artifact': _artifact(anchor_path),
					'checks': anchor_checks,
					'runtime_checks': anchor_runtime_checks,
					'evaluation_perception_runtime': anchor_eval.get(
						'perception_runtime'
					),
					'reward_mean': statistics.fmean(anchor_rewards),
					'reward_median': statistics.median(anchor_rewards),
					'reward_sample_std': statistics.stdev(anchor_rewards),
					'rewards': anchor_rewards,
				})
			except Exception as exc:
				engineering_failures.append(f'{task}/full_cutie parse/check failed: {exc}')

		if new_eval is not None and anchor_eval is not None:
			mismatches = {
				field: [
					index for index, (anchor_row, new_row) in enumerate(zip(
						anchor_eval['episodes'], new_eval['episodes']
					)) if anchor_row.get(field) != new_row.get(field)
				]
				for field in PAIR_FIELDS
			}
			manifest_pairs = {
				field: {
					anchor_eval.get('provenance', {}).get(field),
					new_eval.get('provenance', {}).get(field),
				}
				for field in (
					'device_name', 'validation_manifest_sha256',
					'combined_manifest_sha256',
				)
			}
			provenance_exact = (
				len(manifest_pairs['device_name']) == 1
				and None not in manifest_pairs['device_name']
				and all(
					len(manifest_pairs[field]) == 1
					and all(_valid_sha(value) for value in manifest_pairs[field])
					for field in (
						'validation_manifest_sha256', 'combined_manifest_sha256'
					)
				)
			)
			pairing_exact = not any(mismatches.values()) and provenance_exact
			task_report['pairing'] = {
				'exact': pairing_exact,
				'fields': list(PAIR_FIELDS),
				'mismatch_episode_indices': mismatches,
				'provenance_exact': provenance_exact,
				'provenance_values': {
					key: sorted(value, key=str) for key, value in manifest_pairs.items()
				},
				'initial_object_hash_required_because_input_representation_is_identical': True,
			}
			if not pairing_exact:
				engineering_failures.append(f'{task}: strict pairing mismatch {mismatches}')
			new_rewards = task_report['new_arm'].get('rewards')
			anchor_rewards = task_report['full_cutie_anchor'].get('rewards')
			if new_rewards is not None and anchor_rewards is not None:
				task_report['comparison'] = {
					'name': 'full_input_geometry_loss_minus_historical_full_cutie',
					'paired': _paired(anchor_rewards, new_rewards),
					'baseline_training_is_contemporaneous': False,
					'interpretation': 'development kill-test only',
				}
		report['tasks'][task] = task_report

	engineering_pass = not engineering_failures
	thresholds = {}
	for task in TASKS:
		mean = report['tasks'][task].get('new_arm', {}).get('reward_mean')
		thresholds[task] = {
			'reward_mean': mean,
			'required_minimum': THRESHOLDS[task],
			'pass': (
				isinstance(mean, (int, float))
				and math.isfinite(float(mean))
				and float(mean) >= THRESHOLDS[task]
			),
		}
	investment_go = engineering_pass and all(
		value['pass'] for value in thresholds.values()
	)
	if not engineering_pass:
		recommendation = 'engineering_fail_do_not_interpret_rewards'
	elif investment_go:
		recommendation = 'worth_three_seed_confirmation_not_paper_go'
	else:
		recommendation = 'stop_full_input_geometry_target_after_seed7_kill_test'
	report.update({
		'status': (
			'full_input_geometry_loss_seed7_engineering_pass'
			if engineering_pass else 'full_input_geometry_loss_seed7_engineering_fail'
		),
		'engineering_pass': engineering_pass,
		'engineering_failures': engineering_failures,
		'development_decision': {
			'thresholds': thresholds,
			'investment_go': investment_go,
			'all_tasks_pass': all(value['pass'] for value in thresholds.values()),
			'not_a_paper_go': True,
			'not_training_seed_uncertainty': True,
		},
		'recommendation': recommendation,
		'protocol': inputs['protocol'],
	})
	_write(args.output, report)
	print(json.dumps({
		'status': report['status'], 'engineering_pass': engineering_pass,
		'investment_go': investment_go, 'recommendation': recommendation,
		'summary': str(args.output.resolve()),
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
		return 0
	return aggregate(args)


if __name__ == '__main__':
	raise SystemExit(main())
