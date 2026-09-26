"""Build deterministic declarative object graphs for the extended DMC screen."""

import argparse
import json
from pathlib import Path


TASKS = {
	'acrobot-swingup': (('whole_acrobot',), ()),
	'cup-catch': (('cup', 'ball'), (('cup', 'ball', 'spatial_target_v1'),)),
	'cartpole-swingup': (('cart', 'pole'), (('cart', 'pole', 'revolute_joint_v1'),)),
	'finger-spin': (('finger', 'spinner'), (('finger', 'spinner', 'spatial_target_v1'),)),
	'reacher-easy': (('whole_arm', 'goal'), (('whole_arm', 'goal', 'spatial_target_v1'),)),
	'reacher-hard': (('whole_arm', 'goal'), (('whole_arm', 'goal', 'spatial_target_v1'),)),
	'cartpole-balance': (('cart', 'pole'), (('cart', 'pole', 'revolute_joint_v1'),)),
	'cartpole-balance-sparse': (('cart', 'pole'), (('cart', 'pole', 'revolute_joint_v1'),)),
	'cartpole-swingup-sparse': (('cart', 'pole'), (('cart', 'pole', 'revolute_joint_v1'),)),
	'finger-turn-easy': (
		('finger', 'spinner', 'target'),
		(
			('finger', 'spinner', 'spatial_target_v1'),
			('spinner', 'target', 'spatial_target_v1'),
		),
	),
	'finger-turn-hard': (
		('finger', 'spinner', 'target'),
		(
			('finger', 'spinner', 'spatial_target_v1'),
			('spinner', 'target', 'spatial_target_v1'),
		),
	),
	'pendulum-swingup': (('base', 'pendulum'), (('base', 'pendulum', 'revolute_joint_v1'),)),
	'hopper-stand': (
		('torso', 'leg', 'foot'),
		(('torso', 'leg', 'revolute_joint_v1'), ('leg', 'foot', 'revolute_joint_v1')),
	),
	'hopper-hop': (
		('torso', 'leg', 'foot'),
		(('torso', 'leg', 'revolute_joint_v1'), ('leg', 'foot', 'revolute_joint_v1')),
	),
	'walker-stand': (
		('torso', 'right_leg', 'left_leg'),
		(('torso', 'right_leg', 'revolute_joint_v1'), ('torso', 'left_leg', 'revolute_joint_v1')),
	),
	'walker-walk': (
		('torso', 'right_leg', 'left_leg'),
		(('torso', 'right_leg', 'revolute_joint_v1'), ('torso', 'left_leg', 'revolute_joint_v1')),
	),
	'walker-run': (
		('torso', 'right_leg', 'left_leg'),
		(('torso', 'right_leg', 'revolute_joint_v1'), ('torso', 'left_leg', 'revolute_joint_v1')),
	),
	'cheetah-run': (
		('torso', 'back_leg', 'front_leg'),
		(('torso', 'back_leg', 'revolute_joint_v1'), ('torso', 'front_leg', 'revolute_joint_v1')),
	),
	'quadruped-run': (
		('torso', 'front_legs', 'back_legs'),
		(('torso', 'front_legs', 'revolute_joint_v1'), ('torso', 'back_legs', 'revolute_joint_v1')),
	),
	'quadruped-walk': (
		('torso', 'front_legs', 'back_legs'),
		(('torso', 'front_legs', 'revolute_joint_v1'), ('torso', 'back_legs', 'revolute_joint_v1')),
	),
}


def file_name(task):
	return task.replace('-', '_') + '.json'


def payload(task, roles, edges):
	return {
		'format': 'support_conditioned_object_graph_v1',
		'graph_name': f'{task.replace("-", "_")}_direct_entities_v1',
		'task': task,
		'source_roles': list(roles),
		'semantic_roles': list(roles),
		'tracking_entities': [
			{
				'name': role, 'source_roles': [role],
				'projector': {'type': 'direct_role_v1', 'role': role},
			}
			for role in roles
		],
		'relations': [
			{'parent': parent, 'child': child, 'type': relation}
			for parent, child, relation in edges
		],
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--check', action='store_true')
	args = parser.parse_args()
	args.output.mkdir(parents=True, exist_ok=True)
	for task, (roles, edges) in TASKS.items():
		path = args.output / file_name(task)
		content = json.dumps(payload(task, roles, edges), indent=2, sort_keys=True) + '\n'
		if args.check:
			if not path.is_file() or path.read_text(encoding='utf-8') != content:
				raise ValueError(f'Graph is absent or stale: {path}')
		else:
			path.write_text(content, encoding='utf-8')
	print(f'STANDARD_VARIABLE_OBJECT_GRAPHS_OK count={len(TASKS)}')


if __name__ == '__main__':
	main()
