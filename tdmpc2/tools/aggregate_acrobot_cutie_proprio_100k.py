"""Aggregate the paired 100k proprio-only and safe-fusion Acrobot run."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


MODES = ('proprio_only', 'fusion')
EXPECTED_STEPS = tuple(range(0, 100001, 10000))


def _auc(curve):
	return sum(
		(right['step'] - left['step'])
		* (left['episode_reward'] + right['episode_reward']) / 2.0
		for left, right in zip(curve, curve[1:])
	) / EXPECTED_STEPS[-1]


def _read_curve(path):
	with path.open(newline='', encoding='utf-8') as stream:
		curve = [
			{'step': int(float(row['step'])), 'episode_reward': float(row['episode_reward'])}
			for row in csv.DictReader(stream)
		]
	if tuple(row['step'] for row in curve) != EXPECTED_STEPS:
		raise ValueError(f'{path}: unexpected evaluation steps {curve!r}.')
	if not all(math.isfinite(row['episode_reward']) for row in curve):
		raise ValueError(f'{path}: non-finite reward.')
	return curve


def _read_trace(path, curve):
	rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
	if len(rows) != len(EXPECTED_STEPS):
		raise ValueError(f'{path}: expected {len(EXPECTED_STEPS)} trace rows, got {len(rows)}.')
	for expected_index, (expected_step, row, point) in enumerate(zip(EXPECTED_STEPS, rows, curve)):
		if row.get('evaluation_index') != expected_index or row.get('step') != expected_step:
			raise ValueError(f'{path}: malformed trace position {expected_index}: {row!r}.')
		rewards = row.get('episode_rewards')
		if row.get('episodes') != 10 or not isinstance(rewards, list) or len(rewards) != 10:
			raise ValueError(f'{path}: step {expected_step} must contain 10 returns.')
		if not all(math.isfinite(float(value)) for value in rewards):
			raise ValueError(f'{path}: non-finite episode return at step {expected_step}.')
		if not math.isclose(float(row['reward_mean']), point['episode_reward'], rel_tol=1e-5, abs_tol=1e-5):
			raise ValueError(f'{path}: trace/CSV mean mismatch at step {expected_step}.')
	return rows


def _validate(root, mode):
	root = Path(root).expanduser().resolve()
	config_path = root / 'runtime_config.json'
	config = json.loads(config_path.read_text(encoding='utf-8'))
	expected = {
		'task': 'acrobot-swingup', 'obs': 'rgb', 'seed': 9,
		'steps': 100000, 'eval_freq': 10000, 'eval_episodes': 10,
		'save_eval_episode_trace': True,
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'cutie_proprio',
		'cutie_object_frame_schema': 'cutie_query_mask_status_plus_proprio_v1',
		'cutie_object_num_roles': 2, 'cutie_object_frame_dim': 1774,
		'cutie_object_stack_frames': 1, 'cutie_object_input_dim': 1774,
		'cutie_object_only_latent_dim': 128, 'cutie_proprio_mode': mode,
		'video_background_enabled': True, 'video_background_split': 'train',
		'visual_pose_checkpoint': None,
		'cutie_object_allow_simulator_kinematics_runtime': True,
	}
	errors = {
		key: {'actual': config.get(key), 'expected': value}
		for key, value in expected.items() if config.get(key) != value
	}
	if config.get('obs_shape') != {'object': [2, 1774]}:
		errors['obs_shape'] = {'actual': config.get('obs_shape'), 'expected': {'object': [2, 1774]}}
	if int(config.get('latent_dim', -1)) != 128:
		errors['latent_dim'] = {'actual': config.get('latent_dim'), 'expected': 128}
	if errors:
		raise ValueError(f'{mode}: runtime contract mismatch {errors}.')
	curve_path = root / 'eval.csv'
	trace_path = root / 'eval_episodes.jsonl'
	curve = _read_curve(curve_path)
	trace = _read_trace(trace_path, curve)
	final = trace[-1]
	return {
		'root': str(root), 'runtime_config': str(config_path),
		'eval_csv': str(curve_path), 'eval_episode_trace': str(trace_path),
		'curve': curve, 'trace': trace,
		'final_episode_reward': curve[-1]['episode_reward'],
		'final_reward_std': final['reward_std'],
		'final_reward_min': final['reward_min'],
		'final_reward_max': final['reward_max'],
		'peak_episode_reward': max(row['episode_reward'] for row in curve),
		'normalized_auc': _auc(curve),
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--proprio-only-root', required=True)
	parser.add_argument('--fusion-root', required=True)
	parser.add_argument('--source-20k-summary', required=True)
	parser.add_argument('--output', required=True)
	args = parser.parse_args()
	source = Path(args.source_20k_summary).expanduser().resolve()
	prior = json.loads(source.read_text(encoding='utf-8'))
	if prior.get('status') != 'acrobot_cutie_proprio_20k_pilot_complete':
		raise ValueError(f'Invalid source 20k summary: {source}.')
	runs = {
		mode: _validate(getattr(args, f'{mode}_root'), mode)
		for mode in MODES
	}
	ranking = sorted(MODES, key=lambda name: runs[name]['normalized_auc'], reverse=True)
	payload = {
		'format': 'acrobot_cutie_proprio_100k_v1',
		'status': 'acrobot_cutie_proprio_100k_complete',
		'engineering_pass': True,
		'controller_training_authorized': False,
		'protocol': {
			'task': 'acrobot-swingup', 'seed': 9, 'steps': 100000,
			'eval_freq': 10000, 'eval_episodes': 10,
			'episode_length': 500, 'paired_environment_seed': 9,
			'observation_shape': [2, 1774], 'controller_latent_dim': 128,
			'keypoint_network_used': False,
			'note': 'single-training-seed long screening run; individual evaluation returns retained',
		},
		'source_20k_summary': str(source),
		'runs': runs,
		'ranking_by_normalized_auc': ranking,
		'deltas': {
			'fusion_minus_proprio_auc': runs['fusion']['normalized_auc'] - runs['proprio_only']['normalized_auc'],
			'fusion_minus_proprio_final': runs['fusion']['final_episode_reward'] - runs['proprio_only']['final_episode_reward'],
			'fusion_minus_proprio_final_std': runs['fusion']['final_reward_std'] - runs['proprio_only']['final_reward_std'],
		},
		'decision_rule': 'Require multi-seed confirmation before any scientific claim.',
	}
	output = Path(args.output).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_name(output.name + '.incomplete')
	temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
	os.replace(temporary, output)
	print('ACROBOT_CUTIE_PROPRIO_100K_AGGREGATE_OK')
	print(f'SUMMARY={output}')


if __name__ == '__main__':
	main()
