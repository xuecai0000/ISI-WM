"""Contract tests for the deployable Acrobot visual-pose branch."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch

from common import visual_articulated_pose as contract
from envs.wrappers.visual_articulated_pose import (
	VisualArticulatedPoseWrapper, latest_rgb, points_to_role_frame,
)
from perception.acrobot_keypoint_detector import (
	AcrobotKeypointConfig, AcrobotKeypointNet, load_checkpoint, save_checkpoint,
)


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def make_config(checkpoint, **overrides):
	values = {
		'task': contract.TASK, 'obs': 'rgb', 'multitask': False,
		'flat_anchor': True, 'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': contract.VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_frame_schema': contract.FRAME_SCHEMA,
		'cutie_object_role_names': list(contract.ROLE_NAMES),
		'cutie_object_num_roles': contract.NUM_ROLES,
		'cutie_object_frame_dim': contract.FRAME_DIM,
		'cutie_object_stack_frames': contract.STACK_FRAMES,
		'cutie_object_input_dim': contract.INPUT_DIM,
		'cutie_object_role_dim': contract.ROLE_DIM,
		'cutie_object_only_latent_dim': contract.LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'visual_pose_checkpoint': str(checkpoint),
		'visual_pose_device': 'cpu', 'visual_pose_history': 4,
		'visual_pose_image_size': 64, 'visual_pose_control_dt': 0.04,
		'visual_pose_confidence_threshold': 0.,
		'visual_pose_use_cutie_mask': False,
	}
	values.update(overrides)
	return Config(**values)


class _FakePixels(gym.Env):
	def __init__(self):
		self.observation_space = gym.spaces.Box(0, 255, shape=(9, 64, 64), dtype=np.uint8)
		self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
		self.value = 0

	def observation(self):
		return np.full((9, 64, 64), self.value, dtype=np.uint8)

	def reset(self, **kwargs):
		self.value = 0
		return self.observation()

	def step(self, action):
		self.value += 1
		return self.observation(), 1., False, {}


class _FakePredictor:
	config = SimpleNamespace(history=4, image_size=64, action_dim=1, use_foreground_mask=False)
	metadata = {'format': 'acrobot_keypoint_detector_v1'}

	def __init__(self):
		self.calls = 0

	def predict(self, rgb, actions, foreground_mask=None):
		self.calls += 1
		angle = 0.1 * self.calls
		base = np.asarray([0., 1.], dtype=np.float32)
		elbow = base + 0.5 * np.asarray([np.sin(angle), np.cos(angle)])
		tip = elbow + 0.5 * np.asarray([np.sin(-angle), np.cos(-angle)])
		return {
			'world_xz': np.stack((base, elbow, tip)),
			'confidence': np.ones(3, dtype=np.float32),
			'angular_velocity': np.asarray([2.5, -2.5], dtype=np.float32),
			'image_xy': np.zeros((3, 2), dtype=np.float32),
			'heatmaps': np.zeros((3, 16, 16), dtype=np.float32),
		}


class VisualArticulatedPoseContractTest(unittest.TestCase):
	def test_model_shapes_and_checkpoint_round_trip(self):
		with TemporaryDirectory() as temporary:
			path = Path(temporary) / 'detector.pt'
			config = AcrobotKeypointConfig(history=4, image_size=64, action_dim=1)
			model = AcrobotKeypointNet(config)
			output = model(torch.zeros(2, 4, 3, 64, 64), torch.zeros(2, 3, 1))
			self.assertEqual(tuple(output['heatmaps'].shape), (2, 3, 16, 16))
			self.assertEqual(tuple(output['world_xz'].shape), (2, 3, 2))
			self.assertEqual(tuple(output['confidence'].shape), (2, 3))
			self.assertEqual(tuple(output['angular_velocity'].shape), (2, 2))
			save_checkpoint(path, model, training={'unit_test': True})
			loaded, metadata = load_checkpoint(path)
			self.assertEqual(loaded.config, config)
			self.assertEqual(metadata['point_names'], ['base', 'elbow', 'tip'])

	def test_wrapper_emits_oracle_compatible_causal_stack(self):
		with TemporaryDirectory() as temporary:
			checkpoint = Path(temporary) / 'identity.bin'
			checkpoint.write_bytes(b'unit-test-checkpoint')
			cfg = make_config(checkpoint)
			predictor = _FakePredictor()
			env = VisualArticulatedPoseWrapper(_FakePixels(), cfg, predictor=predictor)
			reset = env.reset()['object']
			self.assertEqual(reset.shape, (2, 21))
			for index in range(1, 3):
				np.testing.assert_array_equal(reset[:, :7], reset[:, index * 7:(index + 1) * 7])
			stepped, reward, done, _ = env.step(np.asarray([0.2], dtype=np.float32))
			self.assertEqual(stepped['object'].shape, (2, 21))
			np.testing.assert_allclose(stepped['object'][:, -1], [2.5, -2.5], atol=1e-4)
			self.assertEqual((reward, done), (1., False))
			self.assertFalse(hasattr(env.env, 'physics'))
			self.assertEqual(env.visual_pose_ready['detector_checkpoint_sha256'],
				'cc0e79d52fe71c13b15f13cfe73a1466590ede1ffe9315a100b65d5364a09aa5')

	def test_contract_rejects_privileged_runtime_and_missing_mask_provider(self):
		with TemporaryDirectory() as temporary:
			checkpoint = Path(temporary) / 'identity.bin'
			checkpoint.write_bytes(b'unit-test-checkpoint')
			with self.assertRaises(ValueError):
				contract.validate_config(make_config(
					checkpoint, cutie_object_allow_simulator_kinematics_runtime=True,
				))
			predictor = _FakePredictor()
			predictor.config = SimpleNamespace(
				history=4, image_size=64, action_dim=1, use_foreground_mask=True,
			)
			env = VisualArticulatedPoseWrapper(
				_FakePixels(), make_config(checkpoint, visual_pose_use_cutie_mask=True),
				predictor=predictor,
			)
			with self.assertRaisesRegex(RuntimeError, 'deployable'):
				env.reset()

	def test_descriptor_geometry_and_rgb_extraction(self):
		frame, theta = points_to_role_frame(
			np.asarray([[0., 1.], [0., 0.5], [0.5, 0.5]]), None, dt=0.04,
		)
		self.assertEqual(frame.shape, (2, 7))
		np.testing.assert_allclose(theta, [np.pi, np.pi / 2.])
		stack = np.zeros((9, 4, 5), dtype=np.uint8)
		stack[-3:] = 7
		self.assertEqual(latest_rgb(stack).shape, (4, 5, 3))
		self.assertTrue((latest_rgb(stack) == 7).all())


if __name__ == '__main__':
	unittest.main(verbosity=2)
