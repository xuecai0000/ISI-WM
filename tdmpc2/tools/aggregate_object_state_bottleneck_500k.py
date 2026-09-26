"""Aggregate the two-seed mandatory predicted-state Acrobot experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


CONDITIONS = ('clean', 'hard')
SELECTIONS = ('final', 'best')


def load(path):
	with Path(path).open(encoding='utf-8') as stream:
		return json.load(stream)


def stats(values):
	values = [float(value) for value in values]
	if len(values) != 20 or not all(math.isfinite(value) for value in values):
		raise ValueError('Held-out evaluation must contain 20 finite rewards.')
	return {
		'mean': statistics.mean(values),
		'std': statistics.stdev(values),
		'median': statistics.median(values),
		'min': min(values),
		'max': max(values),
	}


def training(root, steps, eval_freq, eval_episodes):
	with (root / 'eval.csv').open(newline='', encoding='utf-8') as stream:
		curve = [
			{'step': int(float(row['step'])), 'reward': float(row['episode_reward'])}
			for row in csv.DictReader(stream)
		]
	expected = list(range(0, steps + 1, eval_freq))
	if [row['step'] for row in curve] != expected:
		raise ValueError(f'{root}: incomplete training curve.')
	if not all(math.isfinite(row['reward']) for row in curve):
		raise ValueError(f'{root}: non-finite training reward.')
	trace_path = root / 'eval_episodes.jsonl'
	with trace_path.open(encoding='utf-8') as stream:
		rows = [json.loads(line) for line in stream if line.strip()]
	if len(rows) != len(expected):
		raise ValueError(f'{root}: invalid evaluation trace.')
	for row, point in zip(rows, curve):
		rewards = row.get('episode_rewards')
		if int(row.get('step', -1)) != point['step'] or len(rewards or []) != eval_episodes:
			raise ValueError(f'{root}: misaligned evaluation trace.')
	best = max(curve[1:], key=lambda row: (row['reward'], -row['step']))
	auc = sum(
		0.5 * (left['reward'] + right['reward']) * (right['step'] - left['step'])
		for left, right in zip(curve, curve[1:])
	) / steps
	return {
		'curve': curve,
		'normalized_auc': auc,
		'peak_reward': best['reward'],
		'best_step': best['step'],
		'final_reward': curve[-1]['reward'],
	}


def held_out(stage, seed, selection, condition):
	suffix = '' if selection == 'final' else '_best'
	path = stage / 'evaluations' / f'seed{seed}{suffix}_{condition}.json'
	payload = load(path)
	protocol = payload.get('protocol', {})
	if protocol.get('state_supervision_collect_labels') is not False:
		raise ValueError(f'{path}: state labels were not disabled.')
	if protocol.get('state_supervision_label_reads') != 0:
		raise ValueError(f'{path}: evaluation read privileged labels.')
	contract = protocol.get('object_state_supervision', {})
	if contract.get('controller_visual_bypass') is not False:
		raise ValueError(f'{path}: mandatory bottleneck contract is absent.')
	return stats([row['reward'] for row in payload['episodes']])


def state_score(stage, seed, selection, condition):
	payload = load(stage / 'state_scores' / f'seed{seed}_{selection}_{condition}.json')
	score = payload['score']
	return {
		'reward_mean': float(payload['reward_mean']),
		'frame_rmse_mean': float(score['frame_rmse_mean']),
		'frame_rmse_p90': float(score['frame_rmse_p90']),
		'frame_rmse_p95': float(score['frame_rmse_p95']),
		'failure_bursts': score['failure_bursts'],
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', type=Path, required=True)
	parser.add_argument('--steps', type=int, default=500000)
	parser.add_argument('--eval-freq', type=int, default=25000)
	parser.add_argument('--eval-episodes', type=int, default=10)
	parser.add_argument('--seeds', default='12,13')
	args = parser.parse_args()
	seeds = [int(value) for value in args.seeds.split(',')]
	result = {
		'format': 'object_state_bottleneck_500k_summary_v1',
		'status': 'complete',
		'controller_training_authorized': False,
		'scientific_interpretation': 'screening_only_until_compared_with_frozen_auxiliary_baseline',
		'protocol': {
			'task': 'acrobot-swingup', 'steps': args.steps,
			'eval_freq': args.eval_freq, 'eval_episodes': args.eval_episodes,
			'held_out_episodes': 20, 'seeds': seeds,
			'test_input': 'RGB_to_Cutie_whole_entity_to_predicted_state_only',
			'privileged_state_at_test': False,
		},
		'seeds': {},
	}
	for seed in seeds:
		run_root = Path((args.stage / 'training' / f'seed{seed}.root').read_text().strip())
		entry = {'training': training(run_root, args.steps, args.eval_freq, args.eval_episodes)}
		entry['held_out'] = {
			selection: {
				condition: held_out(args.stage, seed, selection, condition)
				for condition in CONDITIONS
			}
			for selection in SELECTIONS
		}
		entry['state_scores'] = {
			selection: {
				condition: state_score(args.stage, seed, selection, condition)
				for condition in CONDITIONS
			}
			for selection in SELECTIONS
		}
		result['seeds'][str(seed)] = entry
	result['across_seeds'] = {
		'auc_mean': statistics.mean(entry['training']['normalized_auc'] for entry in result['seeds'].values()),
		'final_reward_mean': statistics.mean(entry['training']['final_reward'] for entry in result['seeds'].values()),
		'best_held_out_mean': {
			condition: statistics.mean(
				entry['held_out']['best'][condition]['mean'] for entry in result['seeds'].values()
			)
			for condition in CONDITIONS
		},
	}
	path = args.stage / 'object_state_bottleneck_500k_summary.json'
	path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
	print(path)


if __name__ == '__main__':
	main()
