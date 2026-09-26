"""Dependency-light contract for the live Cutie object wrapper."""

from __future__ import annotations

import hashlib
import inspect
import importlib.util
import random
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


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

MODULE_PATH = ROOT / 'tdmpc2' / 'envs' / 'wrappers' / 'cutie_object.py'
SPEC = importlib.util.spec_from_file_location(
	'tdmpc2_contract_cutie_object', MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
	raise RuntimeError(f'Could not load {MODULE_PATH}.')
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _stack(latest: int) -> torch.Tensor:
	frames = [
		torch.full((3, 64, 64), latest - 2, dtype=torch.uint8),
		torch.full((3, 64, 64), latest - 1, dtype=torch.uint8),
		torch.full((3, 64, 64), latest, dtype=torch.uint8),
	]
	return torch.cat(frames, dim=0)


class _FakeEnv:
	def __init__(self):
		self.observation_space = module.gym.spaces.Box(
			low=0, high=255, shape=(9, 64, 64), dtype=np.uint8
		)
		self.latest = 12
		self.actions = []
		self.close_calls = 0

	def reset(self, **_kwargs):
		self.latest += 10
		return _stack(self.latest)

	def step(self, action):
		self.actions.append(action)
		self.latest += 1
		return _stack(self.latest), 7.25, False, {'opaque': object()}

	def close(self):
		self.close_calls += 1


def _feature(value: float, *, valid=True, lost=False) -> np.ndarray:
	result = np.full(
		(2, module.FRAME_FEATURE_DIM), value, dtype=np.float32
	)
	status = module.QUERY_POOL_DIM + module.MASK_SPATIAL_DIM
	result[:, status + 0] = 0.9
	result[:, status + 1] = float(lost)
	result[:, status + 2] = float(valid)
	result[:, status + 3] = 0.8
	return result


class _FakeClient:
	def __init__(self, schedule=None):
		self.schedule = list(schedule or [
			_feature(1), _feature(2), _feature(3), _feature(4)
		])
		self.frames = []
		self.ops = []
		self.close_calls = 0
		self.last_diagnostics = None

	def _next(self, op, frame):
		self.ops.append(op)
		value = np.asarray(frame)
		assert value.shape == (64, 64, 3)
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

	def reset_track(self, frame):
		return self._next('reset_track', frame)

	def track(self, frame):
		return self._next('track', frame)

	def metrics(self):
		return {'worker_restarts': 0, 'timeouts': 0}

	def close(self):
		self.close_calls += 1


class CutieObjectWrapperContract(unittest.TestCase):
	def test_00_frozen_dimensions_and_role_order(self):
		self.assertEqual(module.ROLE_NAMES, ('whole_arm', 'goal'))
		self.assertEqual(module.QUERY_POOL_DIM, 512)
		self.assertEqual(module.MASK_SPATIAL_DIM, 74)
		self.assertEqual(module.FRAME_FEATURE_DIM, 590)
		self.assertEqual(module.STACKED_FEATURE_DIM, 1770)

	def test_01_generic_features_are_permutation_invariant_and_invalid_zero(self):
		features = np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048)
		masks = np.zeros((2, 64, 64), dtype=bool)
		masks[0, 8:24, 4:20] = True
		masks[1, 30:38, 40:48] = True
		base = SimpleNamespace(
			role_names=module.ROLE_NAMES,
			object_features=features,
			masks=masks,
			lost=np.asarray([False, False]),
			confidence=np.asarray([1.0, 1.0], np.float32),
			mask_score=np.asarray([0.9, 0.8], np.float32),
		)
		first = module.generic_object_frame(base)
		permuted = features.reshape(2, 8, 256)[:, ::-1].reshape(2, 2048)
		second = module.generic_object_frame(
			SimpleNamespace(**{**vars(base), 'object_features': permuted})
		)
		np.testing.assert_array_equal(first, second)
		self.assertEqual(first.shape, (2, 590))
		self.assertTrue(np.isfinite(first).all())

		bad = features.copy()
		bad[1, 0] = np.nan
		invalid = module.generic_object_frame(
			SimpleNamespace(**{**vars(base), 'object_features': bad})
		)
		np.testing.assert_array_equal(invalid[1, :512], np.zeros(512, np.float32))
		self.assertEqual(invalid[1, 512 + 74 + 2], 0.0)
		self.assertTrue(np.isfinite(invalid).all())
		diagnostics = module._result_diagnostics(
			SimpleNamespace(**{**vars(base), 'object_features': bad})
		)
		self.assertEqual(set(diagnostics), {
			'valid', 'lost', 'mask_nonempty', 'feature_finite',
			'mask_area_pixels', 'mask_touches_border', 'confidence', 'mask_score',
		})
		np.testing.assert_array_equal(diagnostics['valid'], [True, False])
		np.testing.assert_array_equal(diagnostics['feature_finite'], [True, False])
		np.testing.assert_array_equal(diagnostics['mask_area_pixels'], [256, 64])

	def test_02_wrapper_uses_only_latest_native_rgb_and_causal_stack(self):
		env, client = _FakeEnv(), _FakeClient()
		wrapper = module.CutieObjectWrapper(env, _client=client)
		first = wrapper.reset()
		self.assertEqual(set(first), {'rgb', 'object'})
		self.assertEqual(tuple(first['object'].shape), (2, 1770))
		self.assertEqual(tuple(wrapper.observation_space.spaces['object'].shape), (2, 1770))
		self.assertTrue(np.all(client.frames[0] == 22))
		first_parts = torch.tensor_split(first['object'], 3, dim=-1)
		for part in first_parts:
			torch.testing.assert_close(part, torch.from_numpy(_feature(1)))

		action = torch.tensor([0.25, -0.5])
		second, reward, done, info = wrapper.step(action)
		self.assertEqual(reward, 7.25)
		self.assertFalse(done)
		self.assertIn('opaque', info)
		self.assertIs(env.actions[0], action)
		self.assertTrue(np.all(client.frames[1] == 23))
		parts = torch.tensor_split(second['object'], 3, dim=-1)
		torch.testing.assert_close(parts[0], torch.from_numpy(_feature(1)))
		torch.testing.assert_close(parts[1], torch.from_numpy(_feature(1)))
		torch.testing.assert_close(parts[2], torch.from_numpy(_feature(2)))
		wrapper.step(action)
		self.assertEqual(client.ops, ['reset_track', 'track', 'track'])

	def test_03_reset_repeats_first_feature_and_resets_episode_operation(self):
		env, client = _FakeEnv(), _FakeClient()
		wrapper = module.CutieObjectWrapper(env, _client=client)
		wrapper.reset()
		wrapper.step(torch.zeros(2))
		observation = wrapper.reset()
		parts = torch.tensor_split(observation['object'], 3, dim=-1)
		for part in parts:
			torch.testing.assert_close(part, torch.from_numpy(_feature(3)))
		self.assertEqual(client.ops, ['reset_track', 'track', 'reset_track'])

	def test_04_metrics_track_validity_lost_bursts_and_runtime(self):
		schedule = [
			_feature(1, valid=True),
			_feature(2, valid=False, lost=True),
			_feature(3, valid=False, lost=True),
			_feature(4, valid=True),
			_feature(5, valid=False, lost=True),
		]
		env, client = _FakeEnv(), _FakeClient(schedule)
		wrapper = module.CutieObjectWrapper(env, _client=client)
		wrapper.reset()
		wrapper.step(None)
		wrapper.step(None)
		wrapper.step(None)
		# A new episode starts a new burst rather than joining two episodes.
		wrapper.reset()
		metrics = wrapper.metrics()
		self.assertEqual(metrics['frames'], 5)
		self.assertAlmostEqual(metrics['valid_frame_rate'], 2 / 5)
		self.assertAlmostEqual(metrics['lost_role_rate'], 6 / 10)
		self.assertEqual(metrics['max_invalid_burst'], 2)
		self.assertEqual(metrics['max_invalid_burst_event'], {
			'episode_index': 0,
			'start_step': 1,
			'end_step': 2,
			'length': 2,
			'failed_roles': ['goal', 'whole_arm'],
			'reasons': ['goal:lost', 'whole_arm:lost'],
		})
		for role in module.ROLE_NAMES:
			self.assertAlmostEqual(metrics['role_metrics'][role]['valid_frame_rate'], 2 / 5)
			self.assertAlmostEqual(metrics['role_metrics'][role]['lost_frame_rate'], 3 / 5)
			self.assertEqual(metrics['role_metrics'][role]['max_invalid_burst'], 2)
		self.assertEqual(len(metrics['episode_metrics']), 2)
		self.assertEqual(metrics['episode_metrics'][0]['frames'], 4)
		self.assertEqual(metrics['episode_metrics'][0]['first_invalid_step'], 1)
		self.assertEqual(metrics['episode_metrics'][1]['frames'], 1)
		self.assertEqual(metrics['ms_per_frame'], 10.0)
		self.assertEqual(metrics['worker_restarts'], 0)
		self.assertEqual(metrics['timeouts'], 0)

	def test_05_config_names_and_logical_cuda_mapping_are_frozen(self):
		cfg = {
			'cutie_object_repo': '/oc',
			'cutie_object_checkpoint': '/weights.pth',
			'cutie_object_support_path': '/support.json',
			'cutie_object_config_dir': '/config',
			'cutie_object_device': 'cuda:0',
			'cutie_object_tracker_height': 448,
			'cutie_object_tracker_width': 448,
			'cutie_object_model_size': 'small',
			'cutie_object_prompt_radius': 2.0,
			'cutie_object_amp': True,
			'cutie_object_worker_timeout_seconds': 123,
		}
		parsed = module._worker_config(cfg)
		self.assertEqual(parsed.repo_path, '/oc')
		self.assertEqual(parsed.checkpoint_path, '/weights.pth')
		self.assertEqual(parsed.config_dir, '/config')
		self.assertEqual(parsed.worker_timeout_seconds, 123)
		self.assertFalse(parsed.native_highres_enabled)
		self.assertEqual(parsed.native_highres_size, 128)
		highres = module._worker_config({
			**cfg,
			'cutie_object_native_highres_enabled': True,
			'cutie_object_native_highres_size': 256,
		})
		self.assertTrue(highres.native_highres_enabled)
		self.assertEqual(highres.native_highres_size, 256)
		with self.assertRaisesRegex(ValueError, '128 or 256'):
			module._worker_config({
				**cfg, 'cutie_object_native_highres_size': 192,
			})
		with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
			module.CutieObjectWorkerConfig(
				repo_path='/oc', checkpoint_path='/w', support_path='/s',
				device='cuda:1',
			).validated()
		with self.assertRaisesRegex(ValueError, 'non-empty path'):
			module.CutieObjectWorkerConfig(
				repo_path='None', checkpoint_path='/w', support_path='/s',
			).validated()

	def test_06_worker_is_mandatory_spawn_and_protocol_excludes_control_data(self):
		constructor = inspect.getsource(module._SpawnCutieClient.__init__)
		self.assertIn("get_context('spawn')", constructor)
		self.assertNotIn("get_context('fork')", constructor)
		request = inspect.getsource(module._SpawnCutieClient._request)
		self.assertIn("'frame':", request)
		self.assertNotIn("'action':", request)
		self.assertNotIn("'reward':", request)
		worker = inspect.getsource(module._cutie_worker_main)
		self.assertIn("set(request) != {'op', 'request_id', 'frame'}", worker)
		self.assertIn('adapter.reset_episode()', worker)

	def test_07_parent_rng_and_close_are_preserved_and_idempotent(self):
		random.seed(11)
		np.random.seed(12)
		torch.manual_seed(13)
		python_state = random.getstate()
		numpy_state = np.random.get_state()
		torch_state = torch.random.get_rng_state().clone()
		with module._preserve_parent_random_state():
			random.random()
			np.random.rand()
			torch.rand(3)
		self.assertEqual(random.getstate(), python_state)
		self.assertEqual(np.random.get_state()[0], numpy_state[0])
		np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
		torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

		env, client = _FakeEnv(), _FakeClient()
		wrapper = module.CutieObjectWrapper(env, _client=client)
		wrapper.close()
		wrapper.close()
		self.assertEqual(client.close_calls, 1)
		self.assertEqual(env.close_calls, 1)

	def test_08_step_before_reset_has_no_environment_or_tracker_side_effect(self):
		env, client = _FakeEnv(), _FakeClient()
		wrapper = module.CutieObjectWrapper(env, _client=client)
		with self.assertRaisesRegex(module.CutieObjectWorkerError, 'before reset'):
			wrapper.step(torch.zeros(2))
		self.assertEqual(env.actions, [])
		self.assertEqual(client.ops, [])
		wrapper.close()

	def test_09_object_only_hides_rgb_after_cutie_consumes_latest_frame(self):
		env, client = _FakeEnv(), _FakeClient()
		cfg = {
			'task': 'reacher-visual-small',
			'multitask': False,
			'obs': 'rgb',
			'model_size': 5,
			'flat_anchor_mode': 'cutie_object_only',
			'cutie_object_num_roles': 2,
			'cutie_object_frame_dim': 590,
			'cutie_object_stack_frames': 3,
			'cutie_object_input_dim': 1770,
		}
		wrapper = module.CutieObjectWrapper(env, cfg, _client=client)
		self.assertEqual(set(wrapper.observation_space.spaces), {'object'})
		observation = wrapper.reset()
		self.assertEqual(set(observation), {'object'})
		self.assertEqual(tuple(observation['object'].shape), (2, 1770))
		# The hidden RGB still reached Cutie at native resolution before being
		# removed from the agent-facing observation.
		self.assertEqual(client.frames[0].shape, (64, 64, 3))
		self.assertTrue(np.all(client.frames[0] == 22))
		self.assertEqual(
			wrapper.latest_source_rgb_sha256,
			hashlib.sha256(client.frames[0].tobytes(order='C')).hexdigest(),
		)
		second, reward, done, _ = wrapper.step(torch.zeros(2))
		self.assertEqual(set(second), {'object'})
		self.assertEqual(reward, 7.25)
		self.assertFalse(done)
		self.assertTrue(np.all(client.frames[1] == 23))
		self.assertEqual(
			wrapper.latest_source_rgb_sha256,
			hashlib.sha256(client.frames[1].tobytes(order='C')).hexdigest(),
		)
		wrapper.close()


if __name__ == '__main__':
	unittest.main(verbosity=2)
