"""Aggregate pooled versus direct-role Finger readouts at 100k."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
	return json.loads(Path(path).read_text(encoding='utf-8'))


def training(root, expected_readout):
	root = Path(root).resolve()
	with (root / 'eval.csv').open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))
	steps = np.asarray([float(row['step']) for row in rows])
	rewards = np.asarray([float(row['episode_reward']) for row in rows])
	if steps[-1] != 100000 or len(rows) != 11:
		raise ValueError(f'Incomplete 100k curve: {root}')
	cfg = load(root / 'runtime_config.json')
	if cfg.get('cutie_object_variable_graph_enabled') is not True:
		raise ValueError(f'Not a variable graph checkpoint: {root}')
	if cfg.get('cutie_object_variable_graph_readout', 'pool') != expected_readout:
		raise ValueError(f'Readout identity mismatch: {root}')
	return {
		'root': str(root), 'steps': steps.tolist(), 'rewards': rewards.tolist(),
		'auc': float(np.trapz(rewards, steps) / steps[-1]),
		'peak': float(rewards.max()), 'final': float(rewards[-1]),
	}


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--arm', action='append', nargs=5, required=True,
		metavar=('NAME', 'RUN_ROOT', 'CLEAN', 'HARD', 'READOUT'))
	p.add_argument('--output', required=True)
	a = p.parse_args()
	arms = {}
	for name, root, clean_path, hard_path, readout in a.arm:
		if readout not in {'pool', 'direct'}:
			raise ValueError(f'Invalid readout {readout!r}.')
		conditions = {}
		for condition, path in (('clean', clean_path), ('hard', hard_path)):
			value = load(path)
			if value['task'] != 'finger-spin' or value['condition'] != condition:
				raise ValueError(path)
			conditions[condition] = value['summary']
		arms[name] = {'training': training(root, readout), 'held_out': conditions}
	result = {
		'format': 'variable_graph_finger_ablation_v1', 'status': 'complete',
		'test_time_input': 'rgb_only', 'seed': 24, 'steps': 100000,
		'arms': arms,
	}
	Path(a.output).write_text(
		json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
	main()
