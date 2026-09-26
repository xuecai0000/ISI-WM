"""Aggregate the short variable-object-graph regression without changing gates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(path):
	with Path(path).open(encoding='utf-8') as handle:
		return json.load(handle)


def _training(root):
	root = Path(root).resolve()
	with (root / 'eval.csv').open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))
	steps = np.asarray([float(row['step']) for row in rows], dtype=np.float64)
	rewards = np.asarray([float(row['episode_reward']) for row in rows], dtype=np.float64)
	if len(steps) < 2 or not np.all(np.diff(steps) > 0) or not np.isfinite(rewards).all():
		raise ValueError(f'Invalid eval.csv in {root}.')
	runtime = _load(root / 'runtime_config.json')
	if runtime.get('cutie_object_variable_graph_enabled') is not True:
		raise ValueError(f'{root} is not a variable object graph run.')
	return {
		'root': str(root),
		'steps': steps.tolist(),
		'rewards': rewards.tolist(),
		'auc': float(np.trapz(rewards, steps) / steps[-1]),
		'peak': float(rewards.max()),
		'final': float(rewards[-1]),
		'num_roles': int(runtime['cutie_object_num_roles']),
		'latent_dim': int(runtime['cutie_object_only_latent_dim']),
		'readout': runtime.get('cutie_object_variable_graph_readout', 'pool'),
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--entry', action='append', nargs=4, required=True,
		metavar=('TASK', 'RUN_ROOT', 'CLEAN_JSON', 'HARD_JSON'))
	parser.add_argument('--output', required=True)
	args = parser.parse_args()
	tasks = {}
	for task, root, clean_path, hard_path in args.entry:
		if task in tasks:
			raise ValueError(f'Duplicate task {task}.')
		conditions = {}
		for condition, path in (('clean', clean_path), ('hard', hard_path)):
			payload = _load(path)
			if payload.get('task') != task or payload.get('condition') != condition:
				raise ValueError(f'Evaluation identity mismatch: {path}')
			conditions[condition] = {
				**payload['summary'],
				'valid_frame_rate': payload['perception_runtime']['valid_frame_rate'],
				'max_invalid_burst': payload['perception_runtime']['max_invalid_burst'],
				'ms_per_frame': payload['perception_runtime']['ms_per_frame'],
			}
		tasks[task] = {'training': _training(root), 'held_out': conditions}
	payload = {
		'format': 'variable_object_graph_regression_summary_v1',
		'status': 'complete',
		'protocol': {
			'test_time_input': 'rgb_only',
			'empty_role_padding': False,
			'controller_latent_rule': 'K*64 for direct; 128 for pooling ablation',
		},
		'tasks': tasks,
	}
	path = Path(args.output).resolve()
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
	print(json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
	main()
