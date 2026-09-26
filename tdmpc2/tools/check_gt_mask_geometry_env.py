"""Real-environment smoke contract for the two mask-geometry observations.

This is intentionally a pre-training check.  It rebuilds the frozen seed-7
environment from an existing ObjectOnly runtime config, overrides only the
diagnostic observation schema, and verifies several same-process reset/step
frames.  The GT arm must use live MuJoCo segmentation without spawning Cutie;
the Cutie geometry arm must retain the live tracker while exporting an exact
zero query block.
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


FORMAT = 'gt_mask_geometry_environment_smoke_v1'
VARIANTS = ('cutie_mask_geometry', 'gt_mask_geometry')
SCHEMAS = {
	'cutie_mask_geometry': 'cutie_mask_geometry_v1',
	'gt_mask_geometry': 'simulator_gt_mask_geometry_v1',
}
GT_MIN_ROLE_VALID_RATE = 0.95
GT_MAX_ROLE_INVALID_BURST = 5


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
	from common.seed import set_seed
	from envs import make_env

	raw = json.loads(args.runtime_config.read_text(encoding='utf-8'))
	if not isinstance(raw, dict) or raw.get('task') != args.task:
		raise ValueError('Source runtime config task mismatch.')
	if raw.get('flat_anchor_mode') != 'cutie_object_only':
		raise ValueError('Source runtime must be the frozen ObjectOnly anchor.')
	raw.update({
		'cutie_object_observation_variant': args.variant,
		'cutie_object_frame_schema': SCHEMAS[args.variant],
		'cutie_object_allow_simulator_runtime': args.variant == 'gt_mask_geometry',
		'cutie_object_last_valid_memory': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_belief_enabled': False,
		'visual_foreground_erosion_pixels': 0,
		'compile': False,
	})
	cfg = OmegaConf.create(raw)
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
		metrics = _env_value(env, 'metrics')
		metrics = metrics() if callable(metrics) else None
	finally:
		if env is not None and callable(getattr(env, 'close', None)):
			env.close()

	values = np.stack(observations)
	if values.shape != (args.steps + 1, 2, 1770):
		raise RuntimeError(f'Unexpected observation shape {values.shape}.')
	if values.dtype != np.float32 or not np.isfinite(values).all():
		raise RuntimeError('Object observations must be finite float32.')
	frames = values.reshape(args.steps + 1, 2, 3, 590)
	if np.count_nonzero(frames[..., :512]):
		raise RuntimeError('Mask-geometry query block is not exact zero.')
	if not isinstance(ready, dict) or not isinstance(metrics, dict):
		raise RuntimeError('Wrapper runtime provenance is unavailable.')
	expected_privileged = args.variant == 'gt_mask_geometry'
	role_metrics = metrics.get('role_metrics', {})
	checks = {
		'variant': ready.get('observation_variant') == args.variant,
		'frame_schema': ready.get('frame_schema') == SCHEMAS[args.variant],
		'privileged_runtime': ready.get(
			'privileged_runtime_segmentation'
		) is expected_privileged,
		'metric_frames': metrics.get('frames') == args.steps + 1,
		'zero_worker_failures': (
			metrics.get('worker_restarts') == 0 and metrics.get('timeouts') == 0
		),
		'query_policy': metrics.get('gt_mask_oracle', {}).get(
			'query_feature_policy'
		) == 'exact_zero_512_v1',
		'role_metrics': (
			isinstance(role_metrics, dict) and len(role_metrics) == 2
			and all(
				isinstance(item, dict)
				and isinstance(item.get('valid_frames'), int)
				and item['valid_frames'] > 0
				for item in role_metrics.values()
			)
		),
	}
	if expected_privileged:
		oracle = metrics.get('gt_mask_oracle', {})
		checks.update(
			oracle_enabled=oracle.get('enabled') is True,
			oracle_frames=oracle.get('frames') == args.steps + 1,
			catalog=(
				isinstance(oracle.get('geom_catalog_path'), str)
				and isinstance(oracle.get('geom_catalog_sha256'), str)
				and len(oracle['geom_catalog_sha256']) == 64
				and oracle.get('camera_id') == 0
				and list(oracle.get('image_size', ())) == [64, 64]
			),
			visibility_accounting=(
				isinstance(oracle.get('visibility_failures'), dict)
				and set(oracle['visibility_failures']) == set(role_metrics)
				and all(
					oracle['visibility_failures'][role]
					== role_metrics[role].get('invalid_frames')
					== role_metrics[role].get('lost_frames')
					== role_metrics[role].get('empty_mask_frames')
					for role in role_metrics
				)
			),
			oracle_visibility_quality=all(
				isinstance(role_metrics[role].get('valid_frame_rate'), (int, float))
				and float(role_metrics[role]['valid_frame_rate'])
					>= GT_MIN_ROLE_VALID_RATE
				and isinstance(role_metrics[role].get('max_invalid_burst'), int)
				and role_metrics[role]['max_invalid_burst']
					<= GT_MAX_ROLE_INVALID_BURST
				for role in role_metrics
			),
		)
	else:
		checks['oracle_disabled'] = metrics.get('gt_mask_oracle', {}).get(
			'enabled'
		) is False
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise RuntimeError(f'Environment smoke checks failed: {failed!r}.')
	return {
		'format': FORMAT,
		'task': args.task,
		'variant': args.variant,
		'steps': args.steps,
		'checks': checks,
		'observation_sha256': hashlib.sha256(
			values.astype('<f4', copy=False).tobytes(order='C')
		).hexdigest(),
		'role_valid_counts': {
			role: int(round(item.get('valid_frame_rate', 0.0) * metrics['frames']))
			for role, item in role_metrics.items()
		},
		'perception_runtime': metrics,
		'provenance': {
			'runtime_config': str(args.runtime_config.resolve()),
			'runtime_config_sha256': _sha256(args.runtime_config),
			'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
			'device_name': ready.get('device_name'),
			'ready': ready,
		},
	}


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', required=True)
	parser.add_argument('--variant', choices=VARIANTS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--seed', type=int, default=271828)
	parser.add_argument('--steps', type=int, default=8)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args(argv)
	if args.steps < 1 or not args.runtime_config.is_file():
		parser.error('--steps must be positive and runtime config must exist.')
	return args


def main(argv=None):
	args = parse_args(argv)
	payload = run(args)
	_write(args.output, payload)
	print('GT_MASK_GEOMETRY_ENV_SMOKE_OK', json.dumps({
		'task': args.task,
		'variant': args.variant,
		'frames': args.steps + 1,
	}, allow_nan=False), flush=True)


if __name__ == '__main__':
	main()
