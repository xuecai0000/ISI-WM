"""Dependency-light contract for generic two-object Cutie support packs."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def _install_gym_stub_if_missing():
	try:
		import gymnasium  # noqa: F401
		return
	except ImportError:
		pass

	class Box:
		def __init__(self, low, high, shape, dtype):
			self.low = low
			self.high = high
			self.shape = tuple(shape)
			self.dtype = np.dtype(dtype)

	class Dict:
		def __init__(self, spaces):
			self.spaces = dict(spaces)

	class Wrapper:
		def __init__(self, env):
			self.env = env
			self.observation_space = env.observation_space

		def close(self):
			close = getattr(self.env, 'close', None)
			return close() if callable(close) else None

	gym = types.ModuleType('gymnasium')
	gym.Wrapper = Wrapper
	gym.spaces = SimpleNamespace(Box=Box, Dict=Dict)
	sys.modules['gymnasium'] = gym


_install_gym_stub_if_missing()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from tdmpc2.envs.wrappers import cutie_object as wrapper_module  # noqa: E402
from tdmpc2.common.support_camera_contract import (  # noqa: E402
	build_same_camera_contract,
)
from tdmpc2.perception.cutie_oc_adapter import (  # noqa: E402
	CutieOCConfig,
	load_indexed_support_prompts,
)
from tdmpc2.tools.build_standard_variable_object_graphs import (  # noqa: E402
	TASKS as GRAPH_TASKS,
)


def _sha256(array: np.ndarray) -> str:
	return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _feature(value: float = 1.0, *, num_roles: int = 2) -> np.ndarray:
	feature = np.full((num_roles, wrapper_module.FRAME_FEATURE_DIM), value, np.float32)
	status = wrapper_module.QUERY_POOL_DIM + wrapper_module.MASK_SPATIAL_DIM
	feature[:, status + 0] = 1.0
	feature[:, status + 1] = 0.0
	feature[:, status + 2] = 1.0
	feature[:, status + 3] = 1.0
	return feature


class _FakeEnv:
	def __init__(self):
		self.observation_space = wrapper_module.gym.spaces.Box(
			low=0, high=255, shape=(9, 64, 64), dtype=np.uint8
		)
		self.action_space = wrapper_module.gym.spaces.Box(
			low=-1, high=1, shape=(1,), dtype=np.float32
		)

	def reset(self):
		return torch.zeros((9, 64, 64), dtype=torch.uint8)

	def step(self, _action):
		return self.reset(), 0.0, False, {}

	def close(self):
		pass


class _FakeClient:
	def __init__(self, num_roles=2):
		self.num_roles = int(num_roles)
		self.last_diagnostics = {
			'valid': np.ones(self.num_roles, dtype=np.bool_),
			'lost': np.zeros(self.num_roles, dtype=np.bool_),
			'runtime_ms': 1.0,
		}

	def reset_track(self, _frame):
		return _feature(num_roles=self.num_roles)

	def track(self, _frame):
		return _feature(2.0, num_roles=self.num_roles)

	def close(self):
		pass


class CutieMultitaskSupportContract(unittest.TestCase):
	def _write_pack(self, root: Path, *, task='cup-catch', roles=('cup', 'ball')):
		(root / 'support_frames').mkdir()
		(root / 'indexed_masks').mkdir()
		records = []
		for index in range(6):
			image = np.full((64, 64, 3), index * 7, dtype=np.uint8)
			mask = np.zeros((64, 64), dtype=np.uint8)
			regions = (
				(slice(4, 16), slice(5, 17)),
				(slice(24, 36), slice(25, 37)),
				(slice(44, 56), slice(45, 57)),
			)
			if len(roles) > len(regions):
				raise ValueError('Fixture supports at most three roles.')
			for role_id, region in enumerate(regions[:len(roles)], start=1):
				mask[region] = role_id
			image_name = f'support_{index:02d}.png'
			mask_name = f'support_{index:02d}.png'
			Image.fromarray(image, mode='RGB').save(root / 'support_frames' / image_name)
			Image.fromarray(mask, mode='L').save(root / 'indexed_masks' / mask_name)
			records.append({
				'index': index,
				'episode': index,
				'image': f'support_frames/{image_name}',
				'image_sha256': _sha256(image),
				'indexed_mask': f'indexed_masks/{mask_name}',
				'indexed_mask_sha256': _sha256(mask),
				'active_video': f'video{85 + index % 5}.mp4',
				'frame_index': index,
				'random_prefix_steps': 4 + 3 * index,
				'selected_names': {},
				'role_pixel_counts': {
					role: int((mask == role_id).sum())
					for role_id, role in enumerate(roles, start=1)
				},
			})
		payload = {
			'format': 'cutie_indexed_mask_support_v1',
			'roles': list(roles),
			'collection': {
				'support_schema': 'generic_indexed_v1',
				'task': task,
				'observation': 'rgb',
				'split': 'support',
				'label_policy': 'simulator_segmentation_support_only',
				'diagnostic_support': True,
			},
			'records': records,
		}
		path = root / 'annotations.json'
		path.write_text(json.dumps(payload), encoding='utf-8')
		return path, payload

	def test_00_task_role_contract_supports_canonical_variable_cardinality(self):
		self.assertEqual(wrapper_module.ROLE_NAMES, ('whole_arm', 'goal'))
		expected_subset = {
			'reacher-visual-small': ('whole_arm', 'goal'),
			'cup-catch': ('cup', 'ball'),
			'cartpole-swingup': ('cart', 'pole'),
			'finger-spin': ('finger', 'spinner'),
			'finger-turn-easy': ('finger', 'spinner', 'target'),
			'finger-turn-hard': ('finger', 'spinner', 'target'),
			'acrobot-swingup': ('whole_acrobot',),
			'quadruped-run': ('torso', 'front_legs', 'back_legs'),
			'quadruped-walk': ('torso', 'front_legs', 'back_legs'),
		}
		for task, roles in expected_subset.items():
			self.assertEqual(wrapper_module.TASK_ROLE_NAMES[task], roles)
		self.assertTrue(all(
			1 <= len(roles) <= 3
			for roles in wrapper_module.TASK_ROLE_NAMES.values()
		))

	def test_00b_finger_turn_old_two_role_support_fails_closed(self):
		with tempfile.TemporaryDirectory(prefix='finger_turn_old_support_') as raw:
			path, _ = self._write_pack(
				Path(raw), task='finger-turn-easy', roles=('finger', 'spinner')
			)
			with self.assertRaisesRegex(ValueError, 'roles must be ordered exactly'):
				load_indexed_support_prompts(
					path,
					role_names=wrapper_module.TASK_ROLE_NAMES['finger-turn-easy'],
					expected_task='finger-turn-easy',
					allow_simulator_support=True,
				)

	def test_00c_finger_turn_graph_and_support_collector_include_target(self):
		expected_roles = ('finger', 'spinner', 'target')
		expected_edges = (
			('finger', 'spinner', 'spatial_target_v1'),
			('spinner', 'target', 'spatial_target_v1'),
		)
		for task in ('finger-turn-easy', 'finger-turn-hard'):
			self.assertEqual(GRAPH_TASKS[task], (expected_roles, expected_edges))
		self.assertEqual(GRAPH_TASKS['finger-spin'][0], ('finger', 'spinner'))
		collector = (
			ROOT / 'tdmpc2' / 'tools' / 'collect_cutie_multitask_support.py'
		).read_text(encoding='utf-8')
		self.assertIn('roles=("finger", "spinner", "target")', collector)
		self.assertIn('RoleSelector(site=(r"target",))', collector)

	def test_01_generic_support_requires_explicit_opt_in_and_exact_task(self):
		with tempfile.TemporaryDirectory(prefix='cutie_generic_support_') as raw:
			path, _ = self._write_pack(Path(raw))
			with self.assertRaisesRegex(ValueError, 'disabled'):
				load_indexed_support_prompts(
					path,
					role_names=('cup', 'ball'),
					expected_task='cup-catch',
				)
			support = load_indexed_support_prompts(
				path,
				role_names=('cup', 'ball'),
				expected_task='cup-catch',
				allow_simulator_support=True,
			)
			self.assertEqual(support.role_names, ('cup', 'ball'))
			self.assertEqual(len(support.frames), 6)
			self.assertEqual(len(support.masks), 6)
			self.assertEqual(support.metadata['support_schema'], 'generic_indexed_v1')
			with self.assertRaisesRegex(ValueError, 'task'):
				load_indexed_support_prompts(
					path,
					role_names=('cup', 'ball'),
					expected_task='finger-spin',
					allow_simulator_support=True,
				)

	def test_02_generic_support_hash_ids_and_roles_fail_closed(self):
		with tempfile.TemporaryDirectory(prefix='cutie_generic_corrupt_') as raw:
			root = Path(raw)
			path, payload = self._write_pack(root)
			payload['records'][0]['indexed_mask_sha256'] = '0' * 64
			path.write_text(json.dumps(payload), encoding='utf-8')
			with self.assertRaisesRegex(ValueError, 'hash mismatch'):
				load_indexed_support_prompts(
					path,
					role_names=('cup', 'ball'),
					expected_task='cup-catch',
					allow_simulator_support=True,
				)

	def test_02b_quadruped_requires_explicit_camera_two_contract(self):
		roles = ('torso', 'front_legs', 'back_legs')
		with tempfile.TemporaryDirectory(prefix='cutie_quadruped_camera_') as raw:
			path, payload = self._write_pack(
				Path(raw), task='quadruped-walk', roles=roles
			)
			# Exercise the missing structured contract specifically.  A real legacy
			# Quadruped pack already carries the scalar camera field, so retain that
			# field while omitting only ``camera_contract`` here.
			payload['collection']['camera_id'] = 2
			path.write_text(json.dumps(payload), encoding='utf-8')
			with self.assertRaisesRegex(ValueError, 'camera_contract'):
				load_indexed_support_prompts(
					path,
					role_names=roles,
					expected_task='quadruped-walk',
					allow_simulator_support=True,
				)
			payload['collection'].update({
				'camera_id': 2,
				'camera_contract': build_same_camera_contract(2),
			})
			path.write_text(json.dumps(payload), encoding='utf-8')
			support = load_indexed_support_prompts(
				path,
				role_names=roles,
				expected_task='quadruped-walk',
				allow_simulator_support=True,
			)
			self.assertEqual(support.metadata['camera_id'], 2)
			self.assertFalse(
				support.metadata['camera_contract']['runtime_segmentation_allowed']
			)
			payload['collection']['camera_contract']['mask_camera_id'] = 0
			path.write_text(json.dumps(payload), encoding='utf-8')
			with self.assertRaisesRegex(ValueError, 'same simulator state and camera'):
				load_indexed_support_prompts(
					path,
					role_names=roles,
					expected_task='quadruped-walk',
					allow_simulator_support=True,
				)

	def test_03_worker_config_binds_task_roles_schema_and_old_defaults(self):
		legacy = wrapper_module.CutieObjectWorkerConfig(
			repo_path='/oc', checkpoint_path='/w', support_path='/s'
		).validated()
		self.assertEqual(legacy.task, 'reacher-visual-small')
		self.assertEqual(legacy.role_names, ('whole_arm', 'goal'))
		self.assertEqual(legacy.support_schema, 'whole_arm_goal_v1')
		with self.assertRaisesRegex(ValueError, 'explicit'):
			wrapper_module.CutieObjectWorkerConfig(
				repo_path='/oc', checkpoint_path='/w', support_path='/s',
				task='cup-catch', role_names=('cup', 'ball'),
				support_schema='generic_indexed_v1',
			).validated()
		generic = wrapper_module.CutieObjectWorkerConfig(
			repo_path='/oc', checkpoint_path='/w', support_path='/s',
			task='cup-catch', role_names=('cup', 'ball'),
			support_schema='generic_indexed_v1', allow_simulator_support=True,
		).validated()
		self.assertEqual(generic.role_names, ('cup', 'ball'))
		acrobot = wrapper_module.CutieObjectWorkerConfig(
			repo_path='/oc', checkpoint_path='/w', support_path='/s',
			task='acrobot-swingup', role_names=('whole_acrobot',),
			support_schema='generic_indexed_v1', allow_simulator_support=True,
		).validated()
		self.assertEqual(acrobot.role_names, ('whole_acrobot',))
		finger_turn = wrapper_module.CutieObjectWorkerConfig(
			repo_path='/oc', checkpoint_path='/w', support_path='/s',
			task='finger-turn-easy', role_names=('finger', 'spinner', 'target'),
			support_schema='generic_indexed_v1', allow_simulator_support=True,
		).validated()
		self.assertEqual(finger_turn.role_names, ('finger', 'spinner', 'target'))
		with self.assertRaisesRegex(ValueError, 'requires role_names'):
			wrapper_module.CutieObjectWorkerConfig(
				repo_path='/oc', checkpoint_path='/w', support_path='/s',
				task='finger-turn-easy', role_names=('finger', 'spinner'),
				support_schema='generic_indexed_v1', allow_simulator_support=True,
			).validated()
		with self.assertRaisesRegex(ValueError, 'requires role_names'):
			wrapper_module.CutieObjectWorkerConfig(
				repo_path='/oc', checkpoint_path='/w', support_path='/s',
				task='acrobot-swingup', role_names=('arm', 'target'),
				support_schema='generic_indexed_v1', allow_simulator_support=True,
			).validated()
		with self.assertRaisesRegex(ValueError, 'requires role_names'):
			wrapper_module.CutieObjectWorkerConfig(
				repo_path='/oc', checkpoint_path='/w', support_path='/s',
				task='cup-catch', role_names=('finger', 'spinner'),
				support_schema='generic_indexed_v1', allow_simulator_support=True,
			).validated()

	def test_04_all_wrappers_keep_variable_k_by_1770(self):
		for task, roles in wrapper_module.TASK_ROLE_NAMES.items():
			with self.subTest(task=task):
				generic = (
					wrapper_module.TASK_SUPPORT_SCHEMAS[task]
					== 'generic_indexed_v1'
				)
				cfg = {
					'task': task,
					'multitask': False,
					'obs': 'rgb',
					'model_size': 5,
					'flat_anchor_mode': 'cutie_object_only',
					'cutie_object_role_names': list(roles),
					'cutie_object_support_schema': (
						'generic_indexed_v1' if generic else 'whole_arm_goal_v1'
					),
					'cutie_object_allow_simulator_support': generic,
					'cutie_object_num_roles': len(roles),
					'cutie_object_frame_dim': 590,
					'cutie_object_stack_frames': 3,
					'cutie_object_input_dim': 1770,
				}
				wrapper = wrapper_module.CutieObjectWrapper(
					_FakeEnv(), cfg, _client=_FakeClient(len(roles))
				)
				self.assertEqual(wrapper._role_names, roles)
				self.assertEqual(
					tuple(wrapper.observation_space.spaces['object'].shape),
					(len(roles), 1770),
				)
				self.assertEqual(
					tuple(wrapper.reset()['object'].shape), (len(roles), 1770)
				)
				wrapper.close()

	def test_05_reacher_generic_is_explicit_but_legacy_remains_default(self):
		base = {
			'task': 'reacher-visual-small', 'multitask': False, 'obs': 'rgb',
			'model_size': 5, 'flat_anchor_mode': 'cutie_object_only',
		}
		legacy = wrapper_module.CutieObjectWrapper(
			_FakeEnv(), base, _client=_FakeClient()
		)
		self.assertEqual(legacy._support_schema, 'whole_arm_goal_v1')
		legacy.close()
		generic_cfg = {
			**base,
			'cutie_object_role_names': ['whole_arm', 'goal'],
			'cutie_object_support_schema': 'generic_indexed_v1',
			'cutie_object_allow_simulator_support': True,
		}
		generic = wrapper_module.CutieObjectWrapper(
			_FakeEnv(), generic_cfg, _client=_FakeClient()
		)
		self.assertEqual(generic._support_schema, 'generic_indexed_v1')
		generic.close()

	def test_06_adapter_generic_schema_supports_variable_roles(self):
		config = CutieOCConfig(
			repo_path='unused', checkpoint_path='unused',
			role_names=('cup', 'ball'), object_schema='generic_indexed_v1',
		).validated()
		self.assertEqual(config.role_names, ('cup', 'ball'))
		one = CutieOCConfig(
			repo_path='unused', checkpoint_path='unused',
			role_names=('whole_acrobot',), object_schema='generic_indexed_v1',
		).validated()
		three = CutieOCConfig(
			repo_path='unused', checkpoint_path='unused',
			role_names=('torso', 'front_legs', 'back_legs'),
			object_schema='generic_indexed_v1',
		).validated()
		self.assertEqual(one.role_names, ('whole_acrobot',))
		self.assertEqual(len(three.role_names), 3)
		with self.assertRaisesRegex(ValueError, 'unique'):
			CutieOCConfig(
				repo_path='unused', checkpoint_path='unused',
				role_names=('leg', 'leg'), object_schema='generic_indexed_v1',
			).validated()


if __name__ == '__main__':
	unittest.main(verbosity=2)
