"""Dependency-light contract for Cutie's optional same-state high-res input.

This check deliberately separates the engineering contract from any reward
claim.  The treatment gives Cutie an extra 128x128 raster while the policy RGB
and the permanent support package remain 64x64.  It therefore validates a
runtime-resolution diagnostic, not a sensor-information-matched comparison.

Run from the repository root::

    python tdmpc2/check_cutie_native_highres_contract.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_DIR):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

# Reuse the established dependency-light Gym stub and fake feature helpers.
import check_cutie_object_wrapper_contract as base  # noqa: E402


module = base.module

VIDEO_PATH = PROJECT_DIR / 'envs' / 'wrappers' / 'video_background.py'
VIDEO_SPEC = importlib.util.spec_from_file_location(
	'tdmpc2_contract_video_background_highres', VIDEO_PATH
)
if VIDEO_SPEC is None or VIDEO_SPEC.loader is None:
	raise RuntimeError(f'Could not load {VIDEO_PATH}.')
video = importlib.util.module_from_spec(VIDEO_SPEC)
sys.modules[VIDEO_SPEC.name] = video
VIDEO_SPEC.loader.exec_module(video)


def _cfg(*, enabled: bool, size: int = 128) -> dict:
	return {
		'task': 'reacher-visual-small',
		'multitask': False,
		'obs': 'rgb',
		'model_size': 5,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_num_roles': 2,
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_native_highres_enabled': enabled,
		'cutie_object_native_highres_size': size,
	}


class _SameStateEnv(base._FakeEnv):
	"""Small deterministic environment with an auditable render protocol."""

	def __init__(self):
		super().__init__()
		self.reset_calls = 0
		self.step_calls = 0
		self.same_state_render_calls = 0
		self.rendered_state_fingerprints = []

	@property
	def state_fingerprint(self):
		return (int(self.latest), int(self.reset_calls), int(self.step_calls))

	def reset(self, **kwargs):
		self.reset_calls += 1
		return super().reset(**kwargs)

	def step(self, action):
		self.step_calls += 1
		return super().step(action)

	def cutie_same_state_rgb(self, *, height, width):
		self.same_state_render_calls += 1
		self.rendered_state_fingerprints.append(self.state_fingerprint)
		# A non-constant spatial signature proves that the wrapper forwards this
		# raster exactly, rather than resizing the policy's newest 64x64 frame.
		y, x = np.indices((int(height), int(width)), dtype=np.uint16)
		frame = np.empty((int(height), int(width), 3), dtype=np.uint8)
		frame[..., 0] = (x + self.latest) % 256
		frame[..., 1] = (y + 2 * self.latest) % 256
		frame[..., 2] = (x + y + 3 * self.latest) % 256
		return frame


class _ResolutionClient(base._FakeClient):
	def __init__(self, expected_size, schedule=None):
		super().__init__(schedule=schedule)
		self.expected_size = int(expected_size)
		self.ready = {
			'perception_input_source': (
				'same_state_native_dmc_render_with_visual_wrapper_replay_v1'
				if self.expected_size > 64
				else 'policy_observation_latest_rgb_v1'
			),
			'perception_input_size': (self.expected_size, self.expected_size),
			'policy_rgb_size': (64, 64),
			'support_input_size': (64, 64),
			'support_resolution_policy': 'frozen_native64_support_v1',
			'mask_output_size': (64, 64),
			'mask_geometry_resolution': (64, 64),
			'extra_sensor_information': self.expected_size > 64,
			'raw_sensor_information_parity': self.expected_size == 64,
			'fair_representation_comparison': self.expected_size == 64,
			'agent_observation_unchanged': True,
			'treatment': (
				'runtime_cutie_input_resolution_only'
				if self.expected_size > 64 else 'historical_policy_rgb64'
			),
			'episode_reset_strategy': 'fresh_inference_core_support_replay_v1',
			'device_name': 'contract-fake-device',
			'cuda_visible_devices': '0',
			'current_device': 0,
		}

	def _next(self, op, frame):
		self.ops.append(op)
		value = np.asarray(frame)
		assert value.shape == (self.expected_size, self.expected_size, 3)
		assert value.dtype == np.uint8
		self.frames.append(value.copy())
		feature = self.schedule.pop(0)
		status = module.QUERY_POOL_DIM + module.MASK_SPATIAL_DIM
		self.last_diagnostics = {
			'valid': feature[:, status + 2] > 0.5,
			'lost': feature[:, status + 1] > 0.5,
			'runtime_ms': 10.0,
		}
		return feature


def _numpy_state_equal(first, second):
	return (
		first[0] == second[0]
		and np.array_equal(first[1], second[1])
		and first[2:] == second[2:]
	)


class _Selector:
	combined_manifest_sha256 = 'combined-contract-hash'

	def __init__(self):
		self._sources = (Path('video70.mp4'),)

	def resolve(self, split):
		assert split == 'validation'
		return self._sources

	def source_names(self, split):
		assert split == 'validation'
		return ('video70.mp4',)

	def manifest_sha256(self, split):
		assert split == 'validation'
		return 'validation-contract-hash'


class CutieNativeHighresContract(unittest.TestCase):
	def test_00_default_is_off_and_size_is_128(self):
		config = (PROJECT_DIR / 'config.yaml').read_text(encoding='utf-8')
		self.assertIn('cutie_object_native_highres_enabled: false', config)
		self.assertIn('cutie_object_native_highres_size: 128', config)
		validated = module.CutieObjectWorkerConfig(
			repo_path='repo', checkpoint_path='checkpoint', support_path='support'
		).validated()
		self.assertFalse(validated.native_highres_enabled)
		self.assertEqual(validated.native_highres_size, 128)

	def test_01_disabled_path_never_requests_an_extra_render(self):
		env = _SameStateEnv()
		client = _ResolutionClient(64)
		wrapper = module.CutieObjectWrapper(env, _cfg(enabled=False), _client=client)
		first = wrapper.reset()
		second, reward, done, _ = wrapper.step(torch.zeros(2))
		self.assertEqual(set(first), {'object'})
		self.assertEqual(set(second), {'object'})
		self.assertEqual(tuple(first['object'].shape), (2, 1770))
		self.assertEqual(env.reset_calls, 1)
		self.assertEqual(env.step_calls, 1)
		self.assertEqual(env.same_state_render_calls, 0)
		self.assertEqual(client.ops, ['reset_track', 'track'])
		self.assertTrue(np.all(client.frames[0] == 22))
		self.assertTrue(np.all(client.frames[1] == 23))
		self.assertEqual(reward, 7.25)
		self.assertFalse(done)
		wrapper.close()

	def test_02_enabled_path_forwards_one_exact_same_state_128_frame(self):
		env = _SameStateEnv()
		client = _ResolutionClient(128)
		wrapper = module.CutieObjectWrapper(env, _cfg(enabled=True), _client=client)

		random.seed(101)
		np.random.seed(102)
		torch.manual_seed(103)
		python_before = random.getstate()
		numpy_before = np.random.get_state()
		torch_before = torch.random.get_rng_state().clone()

		first = wrapper.reset()
		reset_state = env.state_fingerprint
		second, reward, done, _ = wrapper.step(torch.zeros(2))
		step_state = env.state_fingerprint

		self.assertEqual(set(first), {'object'})
		self.assertEqual(set(second), {'object'})
		self.assertEqual(tuple(second['object'].shape), (2, 1770))
		self.assertEqual(env.reset_calls, 1)
		self.assertEqual(env.step_calls, 1)
		self.assertEqual(env.same_state_render_calls, 2)
		self.assertEqual(
			env.rendered_state_fingerprints,
			[(22, 1, 0), (23, 1, 1)],
		)
		self.assertEqual(reset_state, (22, 1, 0))
		self.assertEqual(step_state, (23, 1, 1))
		self.assertEqual(reward, 7.25)
		self.assertFalse(done)
		self.assertEqual(client.frames[0].shape, (128, 128, 3))
		self.assertEqual(client.frames[1].shape, (128, 128, 3))
		x = np.arange(128, dtype=np.uint16)
		np.testing.assert_array_equal(
			client.frames[0][0, :, 0], ((x + 22) % 256).astype(np.uint8)
		)
		self.assertEqual(
			wrapper.latest_source_rgb_sha256,
			hashlib.sha256(client.frames[1].tobytes(order='C')).hexdigest(),
		)
		self.assertEqual(random.getstate(), python_before)
		self.assertTrue(_numpy_state_equal(np.random.get_state(), numpy_before))
		torch.testing.assert_close(torch.random.get_rng_state(), torch_before)

		metrics = wrapper.metrics()
		self.assertEqual(metrics['perception_input_size'], [128, 128])
		self.assertEqual(metrics['policy_rgb_size'], [64, 64])
		self.assertEqual(metrics['support_input_size'], [64, 64])
		self.assertEqual(metrics['mask_output_size'], [64, 64])
		self.assertEqual(metrics['mask_geometry_resolution'], [64, 64])
		self.assertEqual(
			metrics['support_resolution_policy'], 'frozen_native64_support_v1'
		)
		self.assertEqual(
			metrics['episode_reset_strategy'],
			'fresh_inference_core_support_replay_v1',
		)
		self.assertTrue(metrics['extra_sensor_information'])
		self.assertFalse(metrics['raw_sensor_information_parity'])
		self.assertFalse(metrics['fair_representation_comparison'])
		self.assertTrue(metrics['agent_observation_unchanged'])
		self.assertEqual(
			metrics['treatment'], 'runtime_cutie_input_resolution_only'
		)
		self.assertEqual(metrics['same_state_render_frames'], 2)
		self.assertGreater(metrics['same_state_render_ms_per_frame'], 0.0)
		wrapper.close()

	def test_03_highres_missing_or_malformed_render_fails_closed(self):
		client = _ResolutionClient(128)
		wrapper = module.CutieObjectWrapper(
			base._FakeEnv(), _cfg(enabled=True), _client=client
		)
		with self.assertRaisesRegex(module.CutieObjectWorkerError, 'same-state RGB'):
			wrapper.reset()
		self.assertEqual(client.ops, [])
		wrapper.close()

		class BadRenderEnv(_SameStateEnv):
			def cutie_same_state_rgb(self, *, height, width):
				return np.zeros((64, 64, 3), dtype=np.uint8)

		client = _ResolutionClient(128)
		wrapper = module.CutieObjectWrapper(
			BadRenderEnv(), _cfg(enabled=True), _client=client
		)
		with self.assertRaisesRegex(module.CutieObjectWorkerError, 'uint8 HWC RGB'):
			wrapper.reset()
		self.assertEqual(client.ops, [])
		wrapper.close()

		for bad in (64, 129, True, '128'):
			with self.subTest(bad=bad):
				with self.assertRaisesRegex(ValueError, 'must be 128 or 256'):
					module.CutieObjectWrapper(
						_SameStateEnv(), _cfg(enabled=True, size=bad),
						_client=_ResolutionClient(128),
					)

	def test_04_mask_geometry_is_strictly_canonical_64(self):
		mask = np.zeros((64, 64), dtype=np.bool_)
		mask[63, 63] = True
		feature = module._mask_spatial_feature(mask)
		self.assertEqual(feature.shape, (74,))
		self.assertEqual(float(feature[63]), 1.0 / 64.0)
		self.assertEqual(float(feature[64]), 1.0)
		self.assertEqual(float(feature[65]), 1.0)
		self.assertEqual(float(feature[66]), 1.0 / 4096.0)
		np.testing.assert_array_equal(feature[67:71], np.ones(4, dtype=np.float32))
		np.testing.assert_array_equal(feature[71:74], np.zeros(3, dtype=np.float32))
		with self.assertRaisesRegex(ValueError, '64x64'):
			module._mask_spatial_feature(np.zeros((128, 128), dtype=np.bool_))

	def test_05_background_apply_current_reuses_frame_without_clock_or_rng_advance(self):
		compositor = video.VideoBackgroundCompositor(
			_Selector(), 'validation', size=(64, 64), seed=17
		)
		compositor._active_source = compositor._sources[0]
		compositor._frames = np.stack([
			np.full((64, 64, 3), (10, 20, 30), dtype=np.uint8),
			np.full((64, 64, 3), (40, 50, 60), dtype=np.uint8),
		])
		compositor._start = 0
		compositor._frame = 7
		compositor._last_frame_index = 1

		random.seed(201)
		np.random.seed(202)
		torch.manual_seed(203)
		python_before = random.getstate()
		numpy_before = np.random.get_state()
		torch_before = torch.random.get_rng_state().clone()
		local_before = compositor._random.get_state()
		active_before = compositor.active_source
		frame_before = compositor._frame
		index_before = compositor.frame_index

		clean = np.full((128, 128, 3), (0, 0, 255), dtype=np.uint8)
		composed = compositor.apply_current(clean)
		self.assertEqual(composed.shape, (128, 128, 3))
		self.assertTrue(np.all(composed == np.asarray((40, 50, 60), dtype=np.uint8)))
		self.assertEqual(compositor.active_source, active_before)
		self.assertEqual(compositor._frame, frame_before)
		self.assertEqual(compositor.frame_index, index_before)
		self.assertTrue(_numpy_state_equal(compositor._random.get_state(), local_before))
		self.assertEqual(random.getstate(), python_before)
		self.assertTrue(_numpy_state_equal(np.random.get_state(), numpy_before))
		torch.testing.assert_close(torch.random.get_rng_state(), torch_before)


if __name__ == '__main__':
	unittest.main(verbosity=2)
