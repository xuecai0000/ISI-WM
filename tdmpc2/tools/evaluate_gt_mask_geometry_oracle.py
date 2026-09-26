"""Evaluate frozen mask-geometry ObjectOnly checkpoints on held-out videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tdmpc2.tools import evaluate_cutie_multitask_checkpoint as base


FORMAT = 'gt_mask_geometry_checkpoint_evaluation_v1'
TASKS = ('reacher-visual-small', 'cartpole-swingup')
ARMS = ('cutie_mask_geometry', 'gt_mask_geometry')
GT_MIN_ROLE_VALID_RATE = 0.95
GT_MAX_ROLE_INVALID_BURST = 5
FRAME_SCHEMAS = {
	'cutie_mask_geometry': 'cutie_mask_geometry_v1',
	'gt_mask_geometry': 'simulator_gt_mask_geometry_v1',
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


def evaluate(args):
	raw = _json(args.runtime_config)
	expected_privileged = args.arm == 'gt_mask_geometry'
	roles = tuple(raw.get('cutie_object_role_names', ()))
	if len(roles) != 2 or len(set(roles)) != 2:
		raise ValueError('Mask-geometry evaluation requires exactly two roles.')
	support_path = Path(raw.get('cutie_object_support_path', '')).resolve()
	if not support_path.is_file():
		raise FileNotFoundError(support_path)
	support = _json(support_path)
	collection = support.get('collection', {})
	catalog_path = support_path.parent / str(collection.get('geom_catalog', ''))
	if (
		not catalog_path.is_file()
		or _sha256(catalog_path) != collection.get('geom_catalog_sha256')
	):
		raise ValueError('Frozen support geom_catalog is unavailable or changed.')
	checks = {
		'task': raw.get('task') == args.task,
		'variant': raw.get('cutie_object_observation_variant') == args.arm,
		'frame_schema': raw.get('cutie_object_frame_schema') == FRAME_SCHEMAS[args.arm],
		'runtime_privilege': raw.get(
			'cutie_object_allow_simulator_runtime'
		) is expected_privileged,
		'object_only': (
			raw.get('flat_anchor') is True
			and raw.get('flat_anchor_mode') == 'cutie_object_only'
		),
		'generic_support': (
			raw.get('cutie_object_support_schema') == 'generic_indexed_v1'
			and raw.get('cutie_object_allow_simulator_support') is True
		),
		'no_memory': raw.get('cutie_object_last_valid_memory') is False,
		'no_burst': raw.get('cutie_object_policy_burst_plan') is None,
		'no_belief': raw.get('cutie_object_belief_enabled') is False,
	}
	failed = [name for name, passed in checks.items() if not passed]
	if failed:
		raise ValueError(f'Oracle runtime contract failed: {failed!r}.')
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
		raise RuntimeError('Mask-geometry runtime provenance is unavailable.')
	role_metrics = perception.get('role_metrics', {})
	episode_metrics = perception.get('episode_metrics')
	runtime_checks = {
		'frames': perception.get('frames') == 20 * 501,
		'variant': perception.get('observation_variant') == args.arm,
		'frame_schema': perception.get('frame_schema') == FRAME_SCHEMAS[args.arm],
		'privileged': perception.get(
			'privileged_runtime_segmentation'
		) is expected_privileged,
		'ready_variant': ready.get('observation_variant') == args.arm,
		'ready_schema': ready.get('frame_schema') == FRAME_SCHEMAS[args.arm],
		'zero_worker_failures': (
			perception.get('worker_restarts') == 0
			and perception.get('timeouts') == 0
		),
		'query_zero_policy': perception.get('gt_mask_oracle', {}).get(
			'query_feature_policy'
		) == 'exact_zero_512_v1',
		'role_metrics': (
			set(role_metrics) == set(roles)
			and all(
				isinstance(item, dict)
				and item.get('valid_frames', -1) >= 0
				and item.get('invalid_frames', -1) >= 0
				and item.get('valid_frames') + item.get('invalid_frames')
					== 20 * 501
				and item.get('nonfinite_feature_frames') == 0
				for item in role_metrics.values()
			)
		),
		'episode_metrics': (
			isinstance(episode_metrics, list) and len(episode_metrics) == 20
			and all(
				isinstance(item, dict) and item.get('frames') == 501
				for item in episode_metrics
			)
		),
	}
	if expected_privileged:
		oracle = perception.get('gt_mask_oracle', {})
		runtime_checks.update(
			oracle_enabled=oracle.get('enabled') is True,
			oracle_frames=oracle.get('frames') == 20 * 501,
			oracle_format=oracle.get('format') == 'gt_mask_geometry_runtime_v1',
			oracle_catalog=(
				Path(oracle.get('geom_catalog_path', '')).resolve()
					== catalog_path.resolve()
				and oracle.get('geom_catalog_sha256') == _sha256(catalog_path)
				and oracle.get('camera_id') == 0
				and tuple(oracle.get('image_size', ())) == (64, 64)
			),
			oracle_visibility_accounting=(
				set(oracle.get('visibility_failures', {})) == set(roles)
				and all(
					oracle['visibility_failures'][role]
					== role_metrics[role].get('invalid_frames')
					== role_metrics[role].get('lost_frames')
					== role_metrics[role].get('empty_mask_frames')
					for role in roles
				)
			),
			oracle_visibility_quality=all(
				isinstance(role_metrics[role].get('valid_frame_rate'), (int, float))
				and float(role_metrics[role]['valid_frame_rate'])
					>= GT_MIN_ROLE_VALID_RATE
				and isinstance(role_metrics[role].get('max_invalid_burst'), int)
				and role_metrics[role]['max_invalid_burst']
					<= GT_MAX_ROLE_INVALID_BURST
				for role in roles
			),
		)
	else:
		runtime_checks['oracle_disabled'] = perception.get(
			'gt_mask_oracle', {}
		).get('enabled') is False
	failed_runtime = [
		name for name, passed in runtime_checks.items() if not passed
	]
	if failed_runtime:
		raise RuntimeError(
			f'Mask-geometry evaluation runtime contract failed: {failed_runtime!r}.'
		)
	payload['format'] = FORMAT
	payload['arm'] = args.arm
	payload['backend'] = 'cutie_object_only'
	payload['scientific_scope'] = (
		'single-training-seed diagnostic; GT runtime segmentation is privileged '
		'and non-deployable; geometry arms reserve the 512-D query block as exact zero'
	)
	payload['observation_contract'] = {
		'variant': args.arm,
		'frame_schema': FRAME_SCHEMAS[args.arm],
		'role_frame_layout': 'zero_query_512+mask_geometry_74+status_4',
		'stack_frames': 3,
		'privileged_runtime_segmentation': expected_privileged,
		'checks': {**checks, **runtime_checks},
	}
	payload['provenance']['base_evaluator_sha256'] = _sha256(Path(base.__file__))
	payload['provenance']['evaluator_sha256'] = _sha256(Path(__file__).resolve())
	payload['provenance']['cutie_checkpoint_used_at_runtime'] = not expected_privileged
	payload['provenance']['actual_perception_inputs'] = {
		'support_annotations': str(support_path),
		'support_annotations_sha256': _sha256(support_path),
		'geom_catalog': str(catalog_path.resolve()),
		'geom_catalog_sha256': _sha256(catalog_path),
		'cutie_checkpoint': (
			payload['provenance'].get('cutie_inputs', {}).get('checkpoint')
			if not expected_privileged else None
		),
		'cutie_checkpoint_sha256': (
			payload['provenance'].get('cutie_inputs', {}).get('checkpoint_sha256')
			if not expected_privileged else None
		),
		'privileged_live_mujoco_segmentation': expected_privileged,
	}
	if expected_privileged:
		payload['provenance']['base_evaluator_compatibility_record'] = {
			'cutie_inputs': payload['provenance'].pop('cutie_inputs', None),
			'semantics': (
				'the support path is used to bind role-to-geom identities; the recorded '
				'Cutie checkpoint path is not used and no Cutie worker is constructed'
			),
			'support_path_used': True,
			'cutie_checkpoint_used': False,
		}
	return payload


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--arm', choices=ARMS, required=True)
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
	print('GT_MASK_GEOMETRY_EVAL_OK', json.dumps({
		'task': args.task,
		'arm': args.arm,
		'reward_mean': payload['summary']['reward_mean'],
	}, allow_nan=False), flush=True)


if __name__ == '__main__':
	main()
