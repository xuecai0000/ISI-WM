"""Aggregate the fixed 20k Cutie/proprio/safe-fusion Acrobot pilot."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


MODES = ('cutie_only', 'proprio_only', 'fusion')
EXPECTED_STEPS = (0, 5000, 10000, 15000, 20000)


def read_curve(root):
	path = root / 'eval.csv'
	with path.open(newline='', encoding='utf-8') as stream:
		rows = list(csv.DictReader(stream))
	curve = [
		{'step': int(float(row['step'])), 'episode_reward': float(row['episode_reward'])}
		for row in rows
	]
	if tuple(row['step'] for row in curve) != EXPECTED_STEPS:
		raise ValueError(f'{path}: unexpected evaluation steps {curve!r}.')
	if not all(math.isfinite(row['episode_reward']) for row in curve):
		raise ValueError(f'{path}: non-finite reward.')
	return path, curve


def auc(curve):
	return sum(
		(right['step'] - left['step'])
		* (left['episode_reward'] + right['episode_reward']) / 2.0
		for left, right in zip(curve, curve[1:])
	) / EXPECTED_STEPS[-1]


def validate(root, mode):
	root = Path(root).expanduser().resolve()
	config_path = root / 'runtime_config.json'
	config = json.loads(config_path.read_text(encoding='utf-8'))
	expected = {
		'task': 'acrobot-swingup', 'obs': 'rgb', 'seed': 9,
		'steps': 20000, 'eval_freq': 5000, 'eval_episodes': 3,
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'cutie_proprio',
		'cutie_object_frame_schema': 'cutie_query_mask_status_plus_proprio_v1',
		'cutie_object_num_roles': 2, 'cutie_object_frame_dim': 1774,
		'cutie_object_stack_frames': 1, 'cutie_object_input_dim': 1774,
		'cutie_object_only_latent_dim': 128, 'cutie_proprio_mode': mode,
		'video_background_enabled': True, 'video_background_split': 'train',
		'visual_pose_checkpoint': None,
	}
	errors = {
		key: {'actual': config.get(key), 'expected': value}
		for key, value in expected.items() if config.get(key) != value
	}
	if config.get('obs_shape') != {'object': [2, 1774]}:
		errors['obs_shape'] = {
			'actual': config.get('obs_shape'), 'expected': {'object': [2, 1774]},
		}
	if int(config.get('latent_dim', -1)) != 128:
		errors['latent_dim'] = {'actual': config.get('latent_dim'), 'expected': 128}
	if config.get('cutie_object_allow_simulator_kinematics_runtime') is not (mode != 'cutie_only'):
		errors['kinematics_privilege'] = {
			'actual': config.get('cutie_object_allow_simulator_kinematics_runtime'),
			'expected': mode != 'cutie_only',
		}
	if errors:
		raise ValueError(f'{mode}: runtime contract mismatch {errors}.')
	curve_path, curve = read_curve(root)
	return {
		'root': str(root), 'runtime_config': str(config_path),
		'eval_csv': str(curve_path), 'curve': curve,
		'final_episode_reward': curve[-1]['episode_reward'],
		'peak_episode_reward': max(row['episode_reward'] for row in curve),
		'normalized_auc': auc(curve),
	}


def latency(path):
	rows = []
	for line in Path(path).read_text(encoding='utf-8').splitlines():
		if line.startswith('CUTIE_PROPRIO_LATENCY '):
			rows.append(json.loads(line.split(' ', 1)[1]))
	if len(rows) != 2 or {row['mode'] for row in rows} != {'cutie_only', 'fusion'}:
		raise ValueError(f'Latency log must contain cutie_only and fusion, got {rows!r}.')
	return {row['mode']: row for row in rows}


def main():
	parser = argparse.ArgumentParser()
	for mode in MODES:
		parser.add_argument(f'--{mode.replace("_", "-")}-root', required=True)
	parser.add_argument('--latency-log', required=True)
	parser.add_argument('--output', required=True)
	args = parser.parse_args()
	runs = {
		mode: validate(getattr(args, f'{mode}_root'), mode)
		for mode in MODES
	}
	ranking = sorted(MODES, key=lambda name: runs[name]['normalized_auc'], reverse=True)
	latencies = latency(args.latency_log)
	payload = {
		'format': 'acrobot_cutie_proprio_20k_pilot_v1',
		'status': 'acrobot_cutie_proprio_20k_pilot_complete',
		'engineering_pass': True,
		'controller_training_authorized': False,
		'protocol': {
			'task': 'acrobot-swingup', 'seed': 9, 'steps': 20000,
			'eval_freq': 5000, 'eval_episodes': 3,
			'observation_shape': [2, 1774], 'controller_latent_dim': 128,
			'keypoint_network_used': False,
			'note': 'single-seed screening pilot, not a scientific claim',
		},
		'runs': runs,
		'latency': latencies,
		'packing_latency_delta_ms': (
			latencies['fusion']['packing_mean_ms']
			- latencies['cutie_only']['packing_mean_ms']
		),
		'ranking_by_normalized_auc': ranking,
		'deltas': {
			'fusion_minus_cutie_auc': runs['fusion']['normalized_auc'] - runs['cutie_only']['normalized_auc'],
			'fusion_minus_proprio_auc': runs['fusion']['normalized_auc'] - runs['proprio_only']['normalized_auc'],
			'fusion_minus_cutie_final': runs['fusion']['final_episode_reward'] - runs['cutie_only']['final_episode_reward'],
			'fusion_minus_proprio_final': runs['fusion']['final_episode_reward'] - runs['proprio_only']['final_episode_reward'],
		},
		'decision_rule': (
			'Promote safe fusion only if it improves over Cutie-only without materially '
			'underperforming proprio-only; confirm with longer multi-seed training.'
		),
	}
	output = Path(args.output).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_name(output.name + '.incomplete')
	temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
	os.replace(temporary, output)
	print('ACROBOT_CUTIE_PROPRIO_AGGREGATE_OK')
	print(f'SUMMARY={output}')


if __name__ == '__main__':
	main()
