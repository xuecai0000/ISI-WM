"""Aggregate the fixed 20k Acrobot visual/proprio/fusion pilot."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


MODES = ('visual_only', 'proprio_only', 'fusion')
EXPECTED_STEPS = (0, 5000, 10000, 15000, 20000)


def _curve(root):
	path = root / 'eval.csv'
	with path.open(newline='', encoding='utf-8') as stream:
		rows = list(csv.DictReader(stream))
	curve = []
	for row in rows:
		step = int(float(row['step']))
		reward = float(row['episode_reward'])
		if not math.isfinite(reward):
			raise ValueError(f'Non-finite reward in {path}: {row!r}.')
		curve.append({'step': step, 'episode_reward': reward})
	if tuple(row['step'] for row in curve) != EXPECTED_STEPS:
		raise ValueError(
			f'{path} steps must be {EXPECTED_STEPS}, got '
			f'{tuple(row["step"] for row in curve)}.'
		)
	return path, curve


def _auc(curve):
	area = 0.0
	for left, right in zip(curve, curve[1:]):
		area += (
			(right['step'] - left['step'])
			* (left['episode_reward'] + right['episode_reward']) / 2.0
		)
	return area / EXPECTED_STEPS[-1]


def _validate(root, mode):
	root = Path(root).expanduser().resolve()
	config_path = root / 'runtime_config.json'
	config = json.loads(config_path.read_text(encoding='utf-8'))
	expected = {
		'task': 'acrobot-swingup',
		'obs': 'rgb',
		'seed': 8,
		'steps': 20000,
		'eval_freq': 5000,
		'eval_episodes': 3,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'multimodal_articulated_pose',
		'cutie_object_frame_schema': 'acrobot_multimodal_articulated_pose_v1',
		'cutie_object_num_roles': 2,
		'cutie_object_frame_dim': 25,
		'cutie_object_stack_frames': 1,
		'cutie_object_input_dim': 25,
		'cutie_object_only_latent_dim': 128,
		'articulated_pose_modalities': mode,
		'video_background_enabled': True,
		'video_background_split': 'train',
	}
	errors = {
		key: {'actual': config.get(key), 'expected': value}
		for key, value in expected.items() if config.get(key) != value
	}
	if config.get('obs_shape') != {'object': [2, 25]}:
		errors['obs_shape'] = {
			'actual': config.get('obs_shape'), 'expected': {'object': [2, 25]},
		}
	if int(config.get('latent_dim', -1)) != 128:
		errors['latent_dim'] = {'actual': config.get('latent_dim'), 'expected': 128}
	expected_privilege = mode != 'visual_only'
	if config.get('cutie_object_allow_simulator_kinematics_runtime') is not expected_privilege:
		errors['kinematics_privilege'] = {
			'actual': config.get('cutie_object_allow_simulator_kinematics_runtime'),
			'expected': expected_privilege,
		}
	if errors:
		raise ValueError(f'{mode} runtime contract mismatch: {errors}.')
	curve_path, curve = _curve(root)
	return {
		'root': str(root),
		'runtime_config': str(config_path),
		'eval_csv': str(curve_path),
		'curve': curve,
		'final_episode_reward': curve[-1]['episode_reward'],
		'peak_episode_reward': max(row['episode_reward'] for row in curve),
		'normalized_auc': _auc(curve),
	}


def main():
	parser = argparse.ArgumentParser()
	for mode in MODES:
		parser.add_argument(f'--{mode.replace("_", "-")}-root', required=True)
	parser.add_argument('--output', required=True)
	args = parser.parse_args()
	runs = {
		mode: _validate(getattr(args, f'{mode}_root'), mode)
		for mode in MODES
	}
	ranking = sorted(
		MODES, key=lambda name: runs[name]['normalized_auc'], reverse=True,
	)
	payload = {
		'format': 'acrobot_modalities_20k_pilot_v1',
		'status': 'acrobot_modalities_20k_pilot_complete',
		'engineering_pass': True,
		'controller_training_authorized': False,
		'protocol': {
			'task': 'acrobot-swingup', 'seed': 8, 'steps': 20000,
			'eval_freq': 5000, 'eval_episodes': 3,
			'observation_shape': [2, 25], 'controller_latent_dim': 128,
			'note': 'screening pilot; not a multi-seed scientific claim',
		},
		'runs': runs,
		'ranking_by_normalized_auc': ranking,
		'deltas': {
			'fusion_minus_visual_auc': (
				runs['fusion']['normalized_auc'] - runs['visual_only']['normalized_auc']
			),
			'fusion_minus_proprio_auc': (
				runs['fusion']['normalized_auc'] - runs['proprio_only']['normalized_auc']
			),
			'fusion_minus_visual_final': (
				runs['fusion']['final_episode_reward']
				- runs['visual_only']['final_episode_reward']
			),
			'fusion_minus_proprio_final': (
				runs['fusion']['final_episode_reward']
				- runs['proprio_only']['final_episode_reward']
			),
		},
		'interpretation_rule': (
			'If fusion clearly exceeds visual_only, proprioception resolves a material '
			'control bottleneck. If fusion also exceeds proprio_only, vision adds useful '
			'task information. Confirm any decision with longer multi-seed runs.'
		),
	}
	output = Path(args.output).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_name(output.name + '.incomplete')
	temporary.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
		encoding='utf-8', newline='\n',
	)
	os.replace(temporary, output)
	print('ACROBOT_MODALITIES_AGGREGATE_OK')
	print(f'SUMMARY={output}')


if __name__ == '__main__':
	main()
