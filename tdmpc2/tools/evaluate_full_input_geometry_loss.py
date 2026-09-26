"""Strict held-out evaluator for the Full-input geometry/status-target arm.

This is a dependency-thin guard around the established multitask evaluator. It
does not change inference: policy input remains the complete two-role, 1770-D
Full-Cutie descriptor. The only training treatment is the object auxiliary
target used by both current reconstruction and future prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as base


FORMAT = 'full_input_geometry_loss_checkpoint_evaluation_v1'
TASKS = ('reacher-visual-small', 'cartpole-swingup')
ROLES = {
	'reacher-visual-small': ('whole_arm', 'goal'),
	'cartpole-swingup': ('cart', 'pole'),
}
TARGET = 'geometry_status_full_denominator'
FRAME_SCHEMA = 'cutie_query_mask_status_v1'
EXPECTED_FRAMES = 20 * 501
MIN_ROLE_VALID_RATE = 0.50
MAX_ROLE_INVALID_BURST = 250
MAX_MS_PER_FRAME = 800.0


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


def _valid_sha(value) -> bool:
	if not isinstance(value, str) or len(value) != 64:
		return False
	try:
		int(value, 16)
	except ValueError:
		return False
	return True


def _checkpoint_contract(path: Path) -> dict:
	import torch

	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
		raise ValueError('Checkpoint must contain a model state dictionary.')
	contract = payload.get('checkpoint_contract')
	if not isinstance(contract, dict):
		raise ValueError('New arm checkpoint is missing checkpoint_contract.')
	expected_auxiliary = cutie_object_auxiliary.contract('full', TARGET)
	expected_observation = {
		'format': 'cutie_object_observation_contract_v1',
		'variant': 'full',
		'frame_schema': FRAME_SCHEMA,
		'privileged_runtime_segmentation': False,
		'num_roles': 2,
		'frame_dim': 590,
		'stack_frames': 3,
		'input_dim': 1770,
	}
	checks = {
		'format': contract.get('format') == 'tdmpc2_checkpoint_contract_v1',
		'mode': contract.get('flat_anchor_mode') == 'cutie_object_only',
		'latent_dim': contract.get('latent_dim') == 128,
		'no_belief': contract.get('cutie_object_belief_enabled') is False,
		'observation': contract.get('cutie_object_observation') == expected_observation,
		'auxiliary_target': contract.get('cutie_object_auxiliary') == expected_auxiliary,
	}
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise ValueError(f'Checkpoint contract failed: {failed!r}.')
	state = payload['model']
	decoder = {
		name: value for name, value in state.items()
		if name.startswith('_object_decoder.') and torch.is_tensor(value)
	}
	if not decoder:
		raise ValueError('Checkpoint has no object decoder tensors.')
	if any(not torch.isfinite(value).all() for value in decoder.values()):
		raise ValueError('Object decoder contains non-finite values.')
	return {
		'checks': checks,
		'observation': expected_observation,
		'auxiliary': expected_auxiliary,
		'decoder_state_schema': {
			name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
			for name, value in sorted(decoder.items())
		},
		'decoder_state_values': sum(value.numel() for value in decoder.values()),
	}


def evaluate(args):
	raw = _json(args.runtime_config)
	roles = ROLES[args.task]
	support_path = Path(raw.get('cutie_object_support_path', '')).resolve()
	if not support_path.is_file():
		raise FileNotFoundError(support_path)
	config_checks = {
		'task': raw.get('task') == args.task,
		'object_only': (
			raw.get('flat_anchor') is True
			and raw.get('flat_anchor_mode') == 'cutie_object_only'
			and raw.get('latent_dim') == 128
		),
		'full_input': (
			raw.get('cutie_object_observation_variant') == 'full'
			and raw.get('cutie_object_frame_schema') == FRAME_SCHEMA
			and raw.get('cutie_object_input_dim') == 1770
			and raw.get('cutie_object_frame_dim') == 590
			and raw.get('cutie_object_stack_frames') == 3
		),
		'auxiliary_target': raw.get('cutie_object_auxiliary_target') == TARGET,
		'nonprivileged': raw.get('cutie_object_allow_simulator_runtime') is False,
		'roles': tuple(raw.get('cutie_object_role_names', ())) == roles,
		'generic_support': (
			raw.get('cutie_object_support_schema') == 'generic_indexed_v1'
			and raw.get('cutie_object_allow_simulator_support') is True
		),
		'no_memory': raw.get('cutie_object_last_valid_memory') is False,
		'no_burst': raw.get('cutie_object_policy_burst_plan') is None,
		'no_belief': (
			raw.get('cutie_object_belief_enabled') is False
			and raw.get('cutie_object_belief_use_for_control') is False
		),
	}
	failed = [name for name, passed in config_checks.items() if not passed]
	if failed:
		raise ValueError(f'Full-input geometry-loss config failed: {failed!r}.')

	checkpoint = _checkpoint_contract(args.checkpoint)
	base_args = argparse.Namespace(
		task=args.task,
		backend='cutie_object_only',
		runtime_config=args.runtime_config,
		checkpoint=args.checkpoint,
		training_seed=args.training_seed,
		expected_training_steps=args.expected_training_steps,
		expected_training_eval_freq=args.expected_training_eval_freq,
		expected_training_eval_episodes=args.expected_training_eval_episodes,
		episodes=args.episodes,
		env_seed=args.env_seed,
		background_seed=args.background_seed,
		planner_seed_base=args.planner_seed_base,
		erosion_pixels=0,
		output=args.output,
	)
	payload = base.evaluate(base_args)
	perception = payload.get('perception_runtime')
	ready = payload.get('provenance', {}).get('cutie_ready')
	if not isinstance(perception, dict) or not isinstance(ready, dict):
		raise RuntimeError('Full-Cutie runtime provenance is unavailable.')
	role_metrics = perception.get('role_metrics', {})
	episode_metrics = perception.get('episode_metrics')
	oracle = perception.get('gt_mask_oracle', {})
	runtime_checks = {
		'frames': perception.get('frames') == EXPECTED_FRAMES,
		'variant': perception.get('observation_variant') == 'full',
		'frame_schema': perception.get('frame_schema') == FRAME_SCHEMA,
		'nonprivileged': perception.get('privileged_runtime_segmentation') is False,
		'fresh_episode_reset': (
			perception.get('episode_reset_strategy')
				== 'fresh_inference_core_support_replay_v1'
			and ready.get('episode_reset_strategy')
				== 'fresh_inference_core_support_replay_v1'
		),
		'ready_variant': ready.get('observation_variant') == 'full',
		'ready_schema': ready.get('frame_schema') == FRAME_SCHEMA,
		'ready_nonprivileged': ready.get('privileged_runtime_segmentation') is False,
		'zero_worker_failures': (
			perception.get('worker_restarts') == 0
			and perception.get('timeouts') == 0
		),
		'physical_gpu_binding': (
			perception.get('cuda_visible_devices')
				== os.environ.get('CUDA_VISIBLE_DEVICES')
			and perception.get('logical_cuda_device') == 0
			and perception.get('device_name')
				== payload.get('provenance', {}).get('device_name')
			and ready.get('current_device') == 0
		),
		'finite_rates': all(
			isinstance(perception.get(name), (int, float))
			and math.isfinite(float(perception[name]))
			and 0.0 <= float(perception[name]) <= 1.0
			for name in ('valid_frame_rate', 'lost_role_rate')
		),
		'finite_positive_runtime': (
			isinstance(perception.get('ms_per_frame'), (int, float))
			and math.isfinite(float(perception['ms_per_frame']))
			and 0.0 < float(perception['ms_per_frame']) <= MAX_MS_PER_FRAME
		),
		'query_policy': oracle.get('query_feature_policy') == 'cutie_query_mean_std_v1',
		'oracle_disabled': oracle.get('enabled') is False,
		'role_metrics': (
			set(role_metrics) == set(roles)
			and all(
				isinstance(item, dict)
				and isinstance(item.get('valid_frames'), int)
				and isinstance(item.get('invalid_frames'), int)
				and item['valid_frames'] > 0
				and item['valid_frames'] + item['invalid_frames'] == EXPECTED_FRAMES
				and item.get('nonfinite_feature_frames') == 0
				and isinstance(item.get('valid_frame_rate'), (int, float))
				and math.isfinite(float(item['valid_frame_rate']))
				and MIN_ROLE_VALID_RATE <= float(item['valid_frame_rate']) <= 1.0
				and isinstance(item.get('max_invalid_burst'), int)
				and 0 <= item['max_invalid_burst'] <= MAX_ROLE_INVALID_BURST
				for item in role_metrics.values()
			)
		),
		'episode_metrics': (
			isinstance(episode_metrics, list) and len(episode_metrics) == 20
			and all(
				isinstance(item, dict) and item.get('frames') == 501
				and isinstance(item.get('invalid_frames_any_role'), int)
				and 0 <= item['invalid_frames_any_role'] <= 501
				and isinstance(item.get('max_invalid_burst_any_role'), int)
				and 0 <= item['max_invalid_burst_any_role'] <= 501
				and set(item.get('per_role_invalid_frames', {})) == set(roles)
				and set(item.get('per_role_max_invalid_burst', {})) == set(roles)
				and all(
					isinstance(number, int) and 0 <= number <= 501
					for number in item['per_role_invalid_frames'].values()
				)
				and all(
					isinstance(number, int) and 0 <= number <= 501
					for number in item['per_role_max_invalid_burst'].values()
				)
				for item in episode_metrics
			)
		),
		'manifest_hashes': (
			_valid_sha(payload.get('provenance', {}).get('validation_manifest_sha256'))
			and _valid_sha(payload.get('provenance', {}).get('combined_manifest_sha256'))
		),
	}
	failed_runtime = [
		name for name, passed in runtime_checks.items() if not passed
	]
	if failed_runtime:
		raise RuntimeError(
			f'Full-input evaluation runtime contract failed: {failed_runtime!r}.'
		)

	payload['format'] = FORMAT
	payload['arm'] = 'full_input_geometry_loss'
	payload['backend'] = 'cutie_object_only'
	payload['scientific_scope'] = (
		'single-training-seed development kill-test; inference receives the complete '
		'Full-Cutie query+geometry+status descriptor; the training-only treatment '
		'changes the auxiliary target for current reconstruction and future prediction'
	)
	payload['input_contract'] = {
		'variant': 'full',
		'frame_schema': FRAME_SCHEMA,
		'role_shape': [2, 1770],
		'full_input_preserved': True,
		'query_values_per_frame': 512,
		'query_input_policy': 'cutie_query_mean_std_v1',
		'query_input_masked_or_zeroed': False,
	}
	payload['auxiliary_target_contract'] = {
		**checkpoint['auxiliary'],
		'treatment_scope': 'training_only',
		'decoder_architecture_changed': False,
		'decoder_output_shape_changed': False,
		'decoder_parameterization_changed': False,
		'checkpoint_decoder_state_schema': checkpoint['decoder_state_schema'],
		'checkpoint_decoder_state_values': checkpoint['decoder_state_values'],
		'checks': checkpoint['checks'],
	}
	payload['runtime_contract_checks'] = {**config_checks, **runtime_checks}
	payload['tracker_availability_gate'] = {
		'minimum_valid_frame_rate_per_role': MIN_ROLE_VALID_RATE,
		'maximum_invalid_burst_per_role': MAX_ROLE_INVALID_BURST,
		'pass': runtime_checks['role_metrics'],
	}
	payload['runtime_health_gate'] = {
		'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
		'maximum_ms_per_frame': MAX_MS_PER_FRAME,
		'pass': (
			runtime_checks['fresh_episode_reset']
			and runtime_checks['finite_positive_runtime']
			and runtime_checks['zero_worker_failures']
		),
	}
	payload['provenance']['base_evaluator_sha256'] = _sha256(Path(base.__file__))
	payload['provenance']['evaluator_sha256'] = _sha256(Path(__file__).resolve())
	payload['provenance']['actual_perception_inputs'] = {
		'support_annotations': str(support_path),
		'support_annotations_sha256': _sha256(support_path),
		'cutie_checkpoint': payload['provenance']['cutie_inputs']['checkpoint'],
		'cutie_checkpoint_sha256': payload['provenance']['cutie_inputs'][
			'checkpoint_sha256'
		],
		'privileged_live_mujoco_segmentation': False,
	}
	return payload


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--training-seed', type=int, default=7)
	parser.add_argument('--expected-training-steps', type=int, default=100000)
	parser.add_argument('--expected-training-eval-freq', type=int, default=20000)
	parser.add_argument('--expected-training-eval-episodes', type=int, default=3)
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument('--env-seed', type=int, default=424243)
	parser.add_argument('--background-seed', type=int, default=1618034)
	parser.add_argument('--planner-seed-base', type=int, default=8675400)
	parser.add_argument('--output', type=Path, required=True)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	payload = evaluate(args)
	base._write(args.output, payload)
	print('FULL_INPUT_GEOMETRY_LOSS_EVAL_OK', json.dumps({
		'task': args.task,
		'arm': 'full_input_geometry_loss',
		'reward_mean': payload['summary']['reward_mean'],
		'output': str(args.output.resolve()),
	}, allow_nan=False), flush=True)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
