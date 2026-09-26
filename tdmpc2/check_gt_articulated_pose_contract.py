"""Dependency-light contract and optional real-env smoke for the pose oracle."""

import argparse
import math
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np

from common import gt_articulated_pose as contract
from envs.wrappers.gt_articulated_pose import GTArticulatedPoseWrapper


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(**overrides):
	values = {
		'task': contract.TASK,
		'obs': 'state',
		'seed': 7,
		'multitask': False,
		'model_size': 5,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': contract.VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': True,
		'cutie_object_frame_schema': contract.FRAME_SCHEMA,
		'cutie_object_role_names': list(contract.ROLE_NAMES),
		'cutie_object_num_roles': contract.NUM_ROLES,
		'cutie_object_frame_dim': contract.FRAME_DIM,
		'cutie_object_stack_frames': contract.STACK_FRAMES,
		'cutie_object_input_dim': contract.INPUT_DIM,
		'cutie_object_role_dim': contract.ROLE_DIM,
		'cutie_object_only_latent_dim': contract.LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_repo': None,
		'cutie_object_checkpoint': None,
		'cutie_object_support_path': None,
		'cutie_object_config_dir': None,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_native_highres_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'cutie_object_allow_simulator_support': False,
		'cutie_object_allow_simulator_runtime': False,
		'video_background_enabled': False,
		'visual_foreground_erosion_pixels': 0,
	}
	values.update(overrides)
	return Config(**values)


class _Axis:
	def __init__(self, names):
		self.names = list(names)


class _Axes:
	def __init__(self, names):
		self.row = _Axis(names)


class _NamedArray:
	def __init__(self, values):
		self._values = values
		self.axes = _Axes(values)

	def __getitem__(self, item):
		name, columns = item
		column_index = {'x': 0, 'y': 1, 'z': 2}
		return np.asarray([
			self._values[name][column_index[column]] for column in columns
		], dtype=np.float64)


class _FakeModel:
	"""Minimal public MuJoCo-model surface used by the wrapper contract."""

	nq = 2
	nv = 2
	nbody = 3
	nsite = 2

	@staticmethod
	def id2name(index, object_type):
		mapping = {
			'body': ('world', 'upper_arm', 'lower_arm'),
			'site': ('target', 'tip'),
		}
		return mapping[object_type][int(index)]


class _FakePhysics:
	def __init__(self):
		self.model = _FakeModel()
		self.data = SimpleNamespace(time=0.)
		self.named = SimpleNamespace(data=SimpleNamespace(
			xpos=_NamedArray({
				'world': np.zeros(3),
				'upper_arm': np.zeros(3),
				'lower_arm': np.zeros(3),
			}),
			site_xpos=_NamedArray({'target': np.zeros(3), 'tip': np.zeros(3)}),
		))

	def time(self):
		return self.data.time


class _FakeAcrobot(gym.Env):
	def __init__(self):
		self.physics = _FakePhysics()
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1., high=1., shape=(2,), dtype=np.float32
		)
		self._theta = np.asarray([0.2, -0.3], dtype=np.float64)
		self._sync()

	def _sync(self):
		base = np.asarray([0., 0., 2.], dtype=np.float64)
		upper = np.asarray([
			math.sin(self._theta[0]), 0., math.cos(self._theta[0])
		])
		lower = np.asarray([
			math.sin(self._theta[1]), 0., math.cos(self._theta[1])
		])
		elbow = base + upper
		tip = elbow + lower
		self.physics.named.data.xpos._values['upper_arm'] = base
		self.physics.named.data.xpos._values['lower_arm'] = elbow
		self.physics.named.data.site_xpos._values['tip'] = tip

	def reset(self, **kwargs):
		self.physics.data.time = 0.
		self._theta[:] = (0.2, -0.3)
		self._sync()
		return np.zeros(6, dtype=np.float32)

	def step(self, action):
		self._theta += np.asarray(action, dtype=np.float64)
		self.physics.data.time += 0.04
		self._sync()
		return np.zeros(6, dtype=np.float32), 1., False, {}


class GTArticulatedPoseContractTest(unittest.TestCase):
	def test_pure_contract_is_frozen_and_fail_closed(self):
		cfg = make_config()
		self.assertEqual(contract.validate_config(cfg), contract.observation_contract())
		ready = contract.observation_contract()
		self.assertEqual(ready['frame_schema'], 'acrobot_gt_articulated_pose_v1')
		self.assertEqual(ready['role_names'], ['upper_arm', 'lower_arm'])
		self.assertEqual(ready['frame_fields'][-1], 'absolute_frame_omega')
		self.assertEqual((ready['num_roles'], ready['input_dim']), (2, 21))
		self.assertTrue(ready['privileged_simulator_kinematics_runtime'])
		self.assertTrue(ready['diagnostic_only'])
		for override in (
			{'obs': 'rgb'},
			{'task': 'reacher-visual-small'},
			{'cutie_object_allow_simulator_kinematics_runtime': False},
			{'cutie_object_checkpoint': '/forbidden/cutie.pth'},
			{'cutie_object_belief_enabled': True},
			{'video_background_enabled': True},
		):
			with self.subTest(override=override), self.assertRaises(ValueError):
				contract.validate_config(make_config(**override))

	def test_reset_stack_shift_and_signed_global_omega(self):
		env = GTArticulatedPoseWrapper(_FakeAcrobot(), make_config())
		reset = env.reset()['object']
		self.assertEqual(reset.shape, (2, 21))
		self.assertEqual(reset.dtype, np.float32)
		for frame_index in range(1, 3):
			np.testing.assert_array_equal(
				reset[:, :7], reset[:, frame_index * 7:(frame_index + 1) * 7]
			)
		np.testing.assert_array_equal(
			reset[:, [6, 13, 20]], np.zeros((2, 3), dtype=np.float32)
		)
		stepped, reward, done, _ = env.step(np.asarray([0.1, -0.2]))
		stack = stepped['object']
		np.testing.assert_array_equal(stack[:, :7], reset[:, 7:14])
		np.testing.assert_array_equal(stack[:, 7:14], reset[:, 14:21])
		np.testing.assert_allclose(stack[:, -1], [2.5, -5.], rtol=1e-5, atol=1e-5)
		self.assertEqual(reward, 1.)
		self.assertFalse(done)
		self.assertTrue(np.isfinite(stack).all())
		self.assertEqual(env.observation_space['object'].shape, (2, 21))
		metrics = env.metrics()
		self.assertEqual((metrics['frames'], metrics['resets']), (2, 1))
		self.assertGreaterEqual(metrics['runtime_ms'], 0.)

	def test_wrong_anatomy_is_rejected(self):
		inner = _FakeAcrobot()
		inner.physics.model.nq = 3
		with self.assertRaisesRegex(ValueError, 'nq=nv=2'):
			GTArticulatedPoseWrapper(inner, make_config())
		inner = _FakeAcrobot()
		inner.physics.model.nsite = 1
		with self.assertRaisesRegex(ValueError, 'missing sites'):
			GTArticulatedPoseWrapper(inner, make_config())


def real_env_smoke():
	"""Exercise the actual dm-control construction when explicitly requested."""
	from envs.dmcontrol import make_env

	env = make_env(make_config())
	try:
		obs = env.reset()
		assert set(obs) == {'object'}
		assert np.asarray(obs['object']).shape == (2, 21)
		action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
		next_obs, _, _, _ = env.step(action)
		assert np.asarray(next_obs['object']).shape == (2, 21)
		assert np.isfinite(np.asarray(next_obs['object'])).all()
	finally:
		env.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--real-env-smoke', action='store_true')
	args, remaining = parser.parse_known_args()
	program = unittest.main(
		argv=[__file__, *remaining], exit=False, verbosity=2
	)
	if not program.result.wasSuccessful():
		raise SystemExit(1)
	if args.real_env_smoke:
		real_env_smoke()
		print('GT_ARTICULATED_POSE_REAL_ENV_SMOKE_OK')
