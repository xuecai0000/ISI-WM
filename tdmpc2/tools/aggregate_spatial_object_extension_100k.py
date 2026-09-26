"""Aggregate the Cup Catch and Finger Spin spatial-object 100k screen."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


TASKS = ('cup-catch', 'finger-spin')
EXPECTED_STEPS = tuple(range(0, 100001, 10000))


def _load(path):
	return json.loads(Path(path).read_text(encoding='utf-8'))


def _curve(root):
	path = root / 'eval.csv'
	with path.open(newline='', encoding='utf-8') as stream:
		rows = [
			{'step': int(float(row['step'])), 'reward': float(row['episode_reward'])}
			for row in csv.DictReader(stream)
		]
	if tuple(row['step'] for row in rows) != EXPECTED_STEPS:
		raise ValueError(f'{path}: unexpected evaluation steps.')
	if not all(math.isfinite(row['reward']) for row in rows):
		raise ValueError(f'{path}: non-finite reward.')
	return rows


def _auc(curve):
	return sum(
		(b['step'] - a['step']) * (a['reward'] + b['reward']) / 2
		for a, b in zip(curve, curve[1:])
	) / EXPECTED_STEPS[-1]


def _validate_training(task, root):
	root = Path(root).expanduser().resolve()
	config = _load(root / 'runtime_config.json')
	expected_graph = str(
		(root.parents[3] / 'tdmpc2' / 'object_graphs' /
		 f'{task.replace("-", "_")}.json').resolve()
	)
	expected = {
		'task': task, 'obs': 'rgb', 'seed': 14, 'steps': 100000,
		'eval_freq': 10000, 'eval_episodes': 10,
		'save_eval_episode_trace': True,
		'video_background_enabled': True, 'video_background_split': 'train',
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': 'full',
		'cutie_object_spatial_token_enabled': True,
		'cutie_object_spatial_graph_path': expected_graph,
		'cutie_object_auxiliary_target': 'geometry_status_full_denominator',
		'cutie_object_frame_dim': 590, 'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770, 'cutie_object_only_latent_dim': 128,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
	}
	bad = {key: (config.get(key), value) for key, value in expected.items()
		   if config.get(key) != value}
	if bad:
		raise ValueError(f'{task}: runtime mismatch {bad}.')
	curve = _curve(root)
	trace = [json.loads(line) for line in (root / 'eval_episodes.jsonl').read_text(
		encoding='utf-8').splitlines() if line.strip()]
	if len(trace) != len(EXPECTED_STEPS) or any(
		row.get('episodes') != 10 or len(row.get('episode_rewards', [])) != 10
		for row in trace
	):
		raise ValueError(f'{task}: incomplete ten-episode training evaluations.')
	checkpoint = root / 'models' / 'final.pt'
	if not checkpoint.is_file():
		raise FileNotFoundError(checkpoint)
	return {
		'root': str(root), 'curve': curve, 'normalized_auc': _auc(curve),
		'final_reward': curve[-1]['reward'],
		'peak_reward': max(row['reward'] for row in curve),
		'final_reward_std': float(trace[-1]['reward_std']),
		'checkpoint': str(checkpoint.resolve()),
	}


def _validate_evaluation(task, condition, path):
	payload = _load(path)
	if payload.get('task') != task or payload.get('backend') != 'cutie_object_only':
		raise ValueError(f'{path}: evaluation identity mismatch.')
	if payload.get('condition') != condition:
		raise ValueError(f'{path}: condition mismatch.')
	episodes = payload.get('episodes', [])
	if len(episodes) != 20 or any(row.get('length') != 500 for row in episodes):
		raise ValueError(f'{path}: incomplete held-out evaluation.')
	ready = payload.get('provenance', {}).get('cutie_ready', {})
	if ready.get('spatial_token_enabled') is not True:
		raise ValueError(f'{path}: spatial-token runtime provenance is absent.')
	return {
		'path': str(Path(path).resolve()),
		**{key: float(value) for key, value in payload['summary'].items()
		   if key.startswith('reward_')},
		'perception_runtime': payload.get('perception_runtime'),
		'tracker_health': payload.get('tracker_health'),
	}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	for task in TASKS:
		key = task.replace('-', '_')
		parser.add_argument(f'--{key}-root', required=True)
		parser.add_argument(f'--{key}-clean', required=True)
		parser.add_argument(f'--{key}-hard', required=True)
	parser.add_argument('--output', required=True)
	args = parser.parse_args()
	runs = {}
	for task in TASKS:
		key = task.replace('-', '_')
		runs[task] = {
			'training': _validate_training(task, getattr(args, f'{key}_root')),
			'held_out': {condition: _validate_evaluation(
				task, condition, getattr(args, f'{key}_{condition}'))
				for condition in ('clean', 'hard')},
		}
	payload = {
		'format': 'spatial_object_extension_100k_v1',
		'status': 'spatial_object_extension_100k_complete',
		'engineering_pass': True,
		'paper_claim_authorized': False,
		'protocol': {
			'tasks': list(TASKS), 'training_seed': 14, 'steps': 100000,
			'training_background': 'hard/train',
			'training_eval': '10 fixed episodes every 10000 steps',
			'held_out_eval': '20 fixed episodes on clean and hard/validation',
			'policy_inputs': 'RGB -> frozen Cutie -> spatial object tokens only',
			'simulator_state_at_runtime': False,
		},
		'runs': runs,
		'note': 'One seed is a rental-screening result; paper claims require multiple seeds.',
	}
	output = Path(args.output).expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_name(output.name + '.incomplete')
	temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
		allow_nan=False) + '\n', encoding='utf-8', newline='\n')
	os.replace(temporary, output)
	print('SPATIAL_OBJECT_EXTENSION_100K_AGGREGATE_OK')
	print(f'SUMMARY={output}')


if __name__ == '__main__':
	main()
