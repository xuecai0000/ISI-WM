"""Fail-closed identity check for all 20 DMC direct-role experiment inputs."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from envs.wrappers.cutie_object import TASK_ROLE_NAMES
from tools.build_standard_variable_object_graphs import TASKS
from tools.collect_cutie_multitask_support import TASK_BY_NAME
from tools.validate_cutie_support_camera import validate_support_pack


EXPECTED = {
	'acrobot-swingup', 'cup-catch',
	'cartpole-balance', 'cartpole-balance-sparse',
	'cartpole-swingup', 'cartpole-swingup-sparse', 'cheetah-run',
	'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
	'hopper-hop', 'hopper-stand', 'pendulum-swingup',
	'quadruped-run', 'quadruped-walk', 'reacher-easy', 'reacher-hard',
	'walker-run', 'walker-stand', 'walker-walk',
}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--support-root', type=Path, required=True)
	parser.add_argument('--graph-root', type=Path, required=True)
	args = parser.parse_args()
	if set(TASKS) != EXPECTED or EXPECTED - set(TASK_BY_NAME):
		raise ValueError('Code task catalogs do not exactly cover the canonical 20.')
	for task in sorted(EXPECTED):
		roles = tuple(TASKS[task][0])
		if tuple(TASK_ROLE_NAMES[task]) != roles:
			raise ValueError(f'Wrapper role mismatch for {task}.')
		if tuple(TASK_BY_NAME[task].roles) != roles:
			raise ValueError(f'Collector role mismatch for {task}.')
		annotation_path = args.support_root / task / 'annotations.json'
		graph_path = args.graph_root / f'{task.replace("-", "_")}.json'
		annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
		graph = json.loads(graph_path.read_text(encoding='utf-8'))
		if task in {'quadruped-run', 'quadruped-walk'}:
			validate_support_pack(
				annotation_path,
				expected_task=task,
				expected_camera_id=2,
			)
		if tuple(annotation.get('roles', ())) != roles:
			raise ValueError(f'Support role mismatch for {task}.')
		if tuple(graph.get('source_roles', ())) != roles:
			raise ValueError(f'Graph role mismatch for {task}.')
		for record in annotation.get('records', ()):
			for field in ('image', 'indexed_mask'):
				if not (annotation_path.parent / record[field]).is_file():
					raise FileNotFoundError(task, field, record[field])
	print('FULL20_OBJECT_GRAPH_INPUTS_OK tasks=20 no_padding=true readout=direct')


if __name__ == '__main__':
	main()
