"""Real-environment smoke check for the Full-input auxiliary-target ablation.

The ablation changes the object auxiliary target used by both current
reconstruction and future prediction. Live policy observations must remain the
established Full-Cutie 1770-D descriptor, including non-zero query features.
This check reconstructs an immutable seed-7 source environment, applies the new
target config, and inspects several real Cutie observations before any expensive
training is allowed to start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


FORMAT = 'full_input_geometry_loss_environment_smoke_v1'
FRAME_DIM = 590
STACK_FRAMES = 3
QUERY_DIM = 512
OBJECT_SHAPE = (2, STACK_FRAMES * FRAME_DIM)
FRAME_SCHEMA = 'cutie_query_mask_status_v1'
AUXILIARY_TARGET = 'geometry_status_full_denominator'
MIN_ROLE_VALID_RATE = 0.50
MAX_ROLE_INVALID_BURST = 250
MAX_MS_PER_FRAME = 800.0


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file:
		for block in iter(lambda: file.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _env_value(env, name):
	seen = set()
	while env is not None and id(env) not in seen:
		seen.add(id(env))
		if hasattr(env, name):
			return getattr(env, name)
		env = getattr(env, 'env', None)
	return None


def _write(path: Path, payload) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + '.incomplete')
	if path.exists() or temporary.exists():
		raise FileExistsError(path)
	try:
		with temporary.open('x', encoding='utf-8', newline='\n') as file:
			json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
			file.write('\n')
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def run(args):
	import numpy as np
	import torch
	from omegaconf import OmegaConf
	from common import cutie_object_auxiliary
	from common.seed import set_seed
	from envs import make_env

	raw = json.loads(args.runtime_config.read_text(encoding='utf-8'))
	if not isinstance(raw, dict) or raw.get('task') != args.task:
		raise ValueError('Source runtime config task mismatch.')
	if raw.get('flat_anchor_mode') != 'cutie_object_only':
		raise ValueError('Source runtime must be the frozen Full-Cutie ObjectOnly anchor.')
	raw.update({
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': FRAME_SCHEMA,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_auxiliary_target': AUXILIARY_TARGET,
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'visual_foreground_erosion_pixels': 0,
		'compile': False,
	})
	cfg = OmegaConf.create(raw)
	auxiliary_contract = cutie_object_auxiliary.contract(
		'full', AUXILIARY_TARGET
	)
	set_seed(args.seed)
	env = None
	observations = []
	try:
		env = make_env(cfg)
		obs = env.reset()
		observations.append(np.asarray(obs['object']))
		action = torch.zeros(env.action_space.shape, dtype=torch.float32)
		for _ in range(args.steps):
			obs, _, done, _ = env.step(action)
			if done:
				raise RuntimeError('Smoke environment terminated unexpectedly.')
			observations.append(np.asarray(obs['object']))
		ready = _env_value(env, 'cutie_ready')
		metrics_method = _env_value(env, 'metrics')
		metrics = metrics_method() if callable(metrics_method) else None
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	values = np.stack(observations)
	if values.shape != (args.steps + 1, *OBJECT_SHAPE):
		raise RuntimeError(f'Unexpected observation shape {values.shape}.')
	if values.dtype != np.float32 or not np.isfinite(values).all():
		raise RuntimeError('Object observations must be finite float32.')
	frames = values.reshape(args.steps + 1, 2, STACK_FRAMES, FRAME_DIM)
	query = frames[..., :QUERY_DIM]
	query_nonzero = np.any(query != 0.0, axis=-1)
	query_nonzero_descriptors = query_nonzero.sum(axis=(0, 2)).astype(np.int64)
	query_nonzero_values = np.count_nonzero(query, axis=(0, 2, 3)).astype(np.int64)
	if not np.all(query_nonzero_descriptors > 0):
		raise RuntimeError(
			'Full-input smoke observed no active query descriptor for at least one role.'
		)
	if not isinstance(ready, dict) or not isinstance(metrics, dict):
		raise RuntimeError('Cutie wrapper runtime provenance is unavailable.')
	roles = tuple(raw.get('cutie_object_role_names', ()))
	role_metrics = metrics.get('role_metrics', {})
	checks = {
		'core_auxiliary_contract': (
			auxiliary_contract.get('target') == AUXILIARY_TARGET
			and auxiliary_contract.get('effective_target') == AUXILIARY_TARGET
			and auxiliary_contract.get('normalization') == 'full_descriptor'
			and auxiliary_contract.get('applies_to')
				== ['current_reconstruction', 'future_prediction']
			and auxiliary_contract.get('decoder_output_dim') == 1770
			and auxiliary_contract.get('supervised_values_per_frame') == 78
			and auxiliary_contract.get('loss_denominator_values_per_role') == 1770
		),
		'full_input_variant': ready.get('observation_variant') == 'full',
		'full_frame_schema': ready.get('frame_schema') == FRAME_SCHEMA,
		'not_privileged': ready.get('privileged_runtime_segmentation') is False,
		'roles': tuple(ready.get('roles', ())) == roles and set(role_metrics) == set(roles),
		'metric_frames': metrics.get('frames') == args.steps + 1,
		'zero_worker_failures': (
			metrics.get('worker_restarts') == 0 and metrics.get('timeouts') == 0
		),
		'finite_runtime_metrics': (
			isinstance(metrics.get('ms_per_frame'), (int, float))
			and np.isfinite(float(metrics['ms_per_frame']))
			and 0.0 < float(metrics['ms_per_frame']) <= MAX_MS_PER_FRAME
			and all(
				isinstance(metrics.get(name), (int, float))
				and np.isfinite(float(metrics[name]))
				and 0.0 <= float(metrics[name]) <= 1.0
				for name in ('valid_frame_rate', 'lost_role_rate')
			)
		),
		'fresh_episode_reset': (
			metrics.get('episode_reset_strategy')
				== 'fresh_inference_core_support_replay_v1'
			and ready.get('episode_reset_strategy')
				== 'fresh_inference_core_support_replay_v1'
		),
		'query_policy': metrics.get('gt_mask_oracle', {}).get(
			'query_feature_policy'
		) == 'cutie_query_mean_std_v1',
		'query_numerically_active_per_role': bool(
			np.all(query_nonzero_descriptors > 0)
		),
		'role_runtime_health': (
			len(role_metrics) == 2
			and all(
				isinstance(item, dict)
				and isinstance(item.get('valid_frames'), int)
				and item['valid_frames'] > 0
				and item.get('nonfinite_feature_frames') == 0
				for item in role_metrics.values()
			)
		),
		'role_availability_floor': (
			len(role_metrics) == 2
			and all(
				isinstance(item, dict)
				and isinstance(item.get('valid_frame_rate'), (int, float))
				and np.isfinite(float(item['valid_frame_rate']))
				and MIN_ROLE_VALID_RATE <= float(item['valid_frame_rate']) <= 1.0
				and isinstance(item.get('max_invalid_burst'), int)
				and 0 <= item['max_invalid_burst'] <= MAX_ROLE_INVALID_BURST
				for item in role_metrics.values()
			)
		),
	}
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise RuntimeError(f'Full-input environment smoke checks failed: {failed!r}.')
	return {
		'format': FORMAT,
		'task': args.task,
		'steps': args.steps,
		'checks': checks,
		'observation_contract': {
			'variant': 'full',
			'frame_schema': FRAME_SCHEMA,
			'role_shape': list(OBJECT_SHAPE),
			'query_values_per_frame': QUERY_DIM,
			'query_nonzero_descriptor_counts_by_role': {
				role: int(query_nonzero_descriptors[index])
				for index, role in enumerate(roles)
			},
			'query_nonzero_value_counts_by_role': {
				role: int(query_nonzero_values[index])
				for index, role in enumerate(roles)
			},
		},
		'auxiliary_target_contract': auxiliary_contract,
		'auxiliary_contract_evidence_scope': (
			'shared core helper identity only; numeric masking, denominator, and '
			'gradient behavior are tested by the GPU update contract'
		),
		'tracker_availability_gate': {
			'minimum_valid_frame_rate_per_role': MIN_ROLE_VALID_RATE,
			'maximum_invalid_burst_per_role': MAX_ROLE_INVALID_BURST,
		},
		'runtime_health_gate': {
			'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
			'maximum_ms_per_frame': MAX_MS_PER_FRAME,
		},
		'observation_sha256': hashlib.sha256(
			values.astype('<f4', copy=False).tobytes(order='C')
		).hexdigest(),
		'perception_runtime': metrics,
		'provenance': {
			'source_runtime_config': str(args.runtime_config.resolve()),
			'source_runtime_config_sha256': _sha256(args.runtime_config),
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
			'device_name': ready.get('device_name'),
			'cutie_ready': ready,
		},
	}


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=271828)
	parser.add_argument('--steps', type=int, default=16)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args(argv)
	if args.steps < 1 or not args.runtime_config.is_file():
		parser.error('--steps must be positive and runtime config must exist.')
	return args


def main(argv=None):
	args = parse_args(argv)
	payload = run(args)
	_write(args.output, payload)
	print('FULL_INPUT_GEOMETRY_LOSS_ENV_SMOKE_OK', json.dumps({
		'task': args.task,
		'frames': args.steps + 1,
		'query_active': True,
	}, allow_nan=False), flush=True)


if __name__ == '__main__':
	main()
