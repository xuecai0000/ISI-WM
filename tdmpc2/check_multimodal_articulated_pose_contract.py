"""Dependency-light tests for the equal-shape Acrobot modality ablation."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np

from common import multimodal_articulated_pose as contract
from envs.wrappers.multimodal_articulated_pose import (
	MultimodalArticulatedPoseWrapper,
)


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(checkpoint, mode, **overrides):
	values = {
		'task': contract.TASK, 'obs': 'rgb', 'seed': 8, 'multitask': False,
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': contract.VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': mode != 'visual_only',
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_support': False,
		'cutie_object_frame_schema': contract.FRAME_SCHEMA,
		'cutie_object_role_names': list(contract.ROLE_NAMES),
		'cutie_object_num_roles': contract.NUM_ROLES,
		'cutie_object_frame_dim': contract.INPUT_DIM,
		'cutie_object_stack_frames': 1,
		'cutie_object_input_dim': contract.INPUT_DIM,
		'cutie_object_role_dim': contract.ROLE_DIM,
		'cutie_object_only_latent_dim': contract.LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'articulated_pose_modalities': mode,
		'multimodal_proprio_velocity_scale': 10.0,
		'visual_pose_checkpoint': str(checkpoint),
		'visual_pose_device': 'cpu', 'visual_pose_history': 4,
		'visual_pose_image_size': 64, 'visual_pose_control_dt': 0.04,
		'visual_pose_confidence_threshold': 0.,
		'visual_pose_use_cutie_mask': False,
		'video_background_enabled': False,
		'visual_foreground_erosion_pixels': 0,
	}
	values.update(overrides)
	return Config(**values)


class _FakePhysics:
	def __init__(self):
		self.model = SimpleNamespace(nq=2, nv=2)
		self.data = SimpleNamespace(
			qpos=np.asarray([0.2, -0.5], dtype=np.float64),
			qvel=np.asarray([1.0, 2.0], dtype=np.float64),
		)


class _FakePixels(gym.Env):
	def __init__(self, with_physics=True):
		self.observation_space = gym.spaces.Box(
			0, 255, shape=(9, 64, 64), dtype=np.uint8,
		)
		self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
		if with_physics:
			self.physics = _FakePhysics()
		self.value = 0

	def observation(self):
		return np.full((9, 64, 64), self.value, dtype=np.uint8)

	def reset(self, **kwargs):
		self.value = 0
		return self.observation()

	def step(self, action):
		self.value += 1
		if hasattr(self, 'physics'):
			self.physics.data.qpos += np.asarray([0.1, -0.1])
		return self.observation(), 1., False, {}


class _FakePredictor:
	config = SimpleNamespace(
		history=4, image_size=64, action_dim=1, use_foreground_mask=False,
	)
	metadata = {'format': 'acrobot_keypoint_detector_v1'}

	def __init__(self):
		self.calls = 0

	def predict(self, rgb, actions, foreground_mask=None):
		self.calls += 1
		return {
			'world_xz': np.asarray([[0., .5], [0., 0.], [.5, 0.]]),
			'confidence': np.ones(3),
			'angular_velocity': np.asarray([0.5, -0.5]),
			'image_xy': np.zeros((3, 2)),
			'heatmaps': np.zeros((3, 16, 16)),
		}


class MultimodalArticulatedPoseTest(unittest.TestCase):
	def test_modes_have_identical_shape_and_exact_zero_ablation(self):
		with TemporaryDirectory() as temporary:
			checkpoint = Path(temporary) / 'detector.pt'
			checkpoint.write_bytes(b'multimodal-contract-test')
			for mode in contract.MODES:
				with self.subTest(mode=mode):
					predictor = _FakePredictor()
					env = MultimodalArticulatedPoseWrapper(
						_FakePixels(with_physics=mode != 'visual_only'),
						make_config(checkpoint, mode), predictor=predictor,
					)
					value = env.reset()['object']
					self.assertEqual(value.shape, (2, 25))
					self.assertTrue(np.isfinite(value).all())
					if mode == 'proprio_only':
						self.assertTrue((value[:, :21] == 0).all())
						self.assertEqual(predictor.calls, 0)
					else:
						self.assertFalse((value[:, :21] == 0).all())
						self.assertEqual(predictor.calls, 1)
					if mode == 'visual_only':
						self.assertTrue((value[:, 21:] == 0).all())
					else:
						self.assertTrue((value[:, 24] == 1).all())
					stepped, reward, done, _ = env.step(np.asarray([0.1], dtype=np.float32))
					self.assertEqual((reward, done), (1., False))
					self.assertEqual(stepped['object'].shape, (2, 25))

	def test_contract_rejects_hidden_privilege_and_shape_changes(self):
		with TemporaryDirectory() as temporary:
			checkpoint = Path(temporary) / 'detector.pt'
			checkpoint.write_bytes(b'multimodal-contract-test')
			with self.assertRaisesRegex(ValueError, 'config mismatch'):
				contract.validate_config(make_config(
					checkpoint, 'visual_only',
					cutie_object_allow_simulator_kinematics_runtime=True,
				))
			with self.assertRaisesRegex(ValueError, 'config mismatch'):
				contract.validate_config(make_config(
					checkpoint, 'fusion', cutie_object_input_dim=21,
				))


def real_env_smoke(checkpoint, mode):
	"""Exercise actual dm-control and the real detector when requested."""
	from envs.dmcontrol import make_env

	cfg = make_config(checkpoint, mode, visual_pose_device='cuda:0')
	env = make_env(cfg)
	try:
		observation = env.reset()
		value = np.asarray(observation['object'])
		assert value.shape == (2, 25) and np.isfinite(value).all()
		action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
		next_observation, _, _, _ = env.step(action)
		next_value = np.asarray(next_observation['object'])
		assert next_value.shape == (2, 25) and np.isfinite(next_value).all()
	finally:
		env.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--real-env-smoke', action='store_true')
	parser.add_argument('--checkpoint')
	parser.add_argument('--mode', choices=contract.MODES, default='fusion')
	args, remaining = parser.parse_known_args()
	program = unittest.main(argv=[__file__, *remaining], exit=False, verbosity=2)
	if not program.result.wasSuccessful():
		raise SystemExit(1)
	if args.real_env_smoke:
		if not args.checkpoint:
			raise SystemExit('--checkpoint is required for --real-env-smoke')
		real_env_smoke(args.checkpoint, args.mode)
		print(f'ACROBOT_MULTIMODAL_REAL_ENV_SMOKE_OK mode={args.mode}')
