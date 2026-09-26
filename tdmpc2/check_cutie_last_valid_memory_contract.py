"""Dependency-light contract for episode-local Cutie last-valid memory."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _install_torch_stub_if_missing():
	"""Provide only the CPU tensor surface this wrapper contract exercises."""
	try:
		import torch
		return torch
	except ImportError:
		pass

	class Tensor:
		def __init__(self, value):
			self._value = np.asarray(value)

		@property
		def ndim(self):
			return self._value.ndim

		@property
		def shape(self):
			return self._value.shape

		@property
		def dtype(self):
			return self._value.dtype

		def __getitem__(self, key):
			return Tensor(self._value[key])

		def detach(self):
			return self

		def cpu(self):
			return self

		def permute(self, *dims):
			return Tensor(np.transpose(self._value, dims))

		def contiguous(self):
			return Tensor(np.ascontiguousarray(self._value))

		def numpy(self):
			return self._value

		def __array__(self, dtype=None):
			return np.asarray(self._value, dtype=dtype)

	torch = types.ModuleType('torch')
	torch.Tensor = Tensor
	torch.uint8 = np.dtype(np.uint8)
	torch.is_tensor = lambda value: isinstance(value, Tensor)
	torch.as_tensor = lambda value: value if isinstance(value, Tensor) else Tensor(value)
	torch.from_numpy = lambda value: Tensor(value)
	sys.modules['torch'] = torch
	return torch


torch = _install_torch_stub_if_missing()


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
MODULE_PATH = ROOT / 'tdmpc2' / 'envs' / 'wrappers' / 'cutie_object.py'
SPEC = importlib.util.spec_from_file_location(
	'tdmpc2_contract_cutie_last_valid_memory', MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
	raise RuntimeError(f'Could not load {MODULE_PATH}.')
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _rgb_stack(value: int):
	return np.concatenate([
		np.full((3, 64, 64), value - 2, dtype=np.uint8),
		np.full((3, 64, 64), value - 1, dtype=np.uint8),
		np.full((3, 64, 64), value, dtype=np.uint8),
	], axis=0)


def _feature(contents, valid) -> np.ndarray:
	contents = tuple(float(value) for value in contents)
	valid = tuple(bool(value) for value in valid)
	result = np.empty((2, module.FRAME_FEATURE_DIM), dtype=np.float32)
	for role in range(2):
		result[role, :module.FRAME_CONTENT_DIM] = contents[role]
		result[role, module.FRAME_CONTENT_DIM:] = np.asarray([
			1.0 if valid[role] else 0.0,
			0.0 if valid[role] else 1.0,
			1.0 if valid[role] else 0.0,
			0.75 if valid[role] else 0.0,
		], dtype=np.float32)
	return result


class _FakeEnv:
	def __init__(self):
		self.observation_space = module.gym.spaces.Box(
			low=0, high=255, shape=(9, 64, 64), dtype=np.uint8
		)
		self.frame = 10

	def reset(self, **_kwargs):
		self.frame += 10
		return _rgb_stack(self.frame)

	def step(self, _action):
		self.frame += 1
		return _rgb_stack(self.frame), 0.0, False, {'terminated': False}

	def close(self):
		pass


class _FakeClient:
	def __init__(self, schedule):
		self.schedule = [np.array(value, copy=True) for value in schedule]
		self.last_diagnostics = None

	def _next(self, _frame):
		feature = self.schedule.pop(0)
		valid = feature[:, module.FRAME_CONTENT_DIM + 2] > 0.5
		lost = feature[:, module.FRAME_CONTENT_DIM + 1] > 0.5
		self.last_diagnostics = {
			'valid': valid.copy(),
			'lost': lost.copy(),
			'mask_nonempty': valid.copy(),
			'feature_finite': np.ones(2, dtype=np.bool_),
			'mask_area_pixels': np.where(valid, 100, 0).astype(np.int64),
			'mask_touches_border': np.zeros(2, dtype=np.bool_),
			'confidence': feature[:, module.FRAME_CONTENT_DIM + 0].copy(),
			'mask_score': feature[:, module.FRAME_CONTENT_DIM + 3].copy(),
			'runtime_ms': 1.0,
		}
		return feature

	def reset_track(self, frame):
		return self._next(frame)

	def track(self, frame):
		return self._next(frame)

	def metrics(self):
		return {'worker_restarts': 0, 'timeouts': 0}

	def close(self):
		pass


def _cfg(enabled=False):
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
		'cutie_object_last_valid_memory': enabled,
	}


def _parts(observation):
	value = observation['object']
	if hasattr(value, 'detach'):
		value = value.detach().cpu().numpy()
	return tuple(np.split(np.asarray(value), 3, axis=-1))


class CutieLastValidMemoryContract(unittest.TestCase):
	def test_00_default_is_false_and_layout_is_frozen(self):
		config = (ROOT / 'tdmpc2' / 'config.yaml').read_text(encoding='utf-8')
		self.assertIn('cutie_object_last_valid_memory: false', config)
		self.assertEqual(module.FRAME_CONTENT_DIM, 586)
		self.assertEqual(module.STATUS_DIM, 4)
		self.assertEqual(module.FRAME_FEATURE_DIM, 590)
		self.assertEqual(module.STACKED_FEATURE_DIM, 1770)

	def test_01_disabled_path_is_bitwise_unchanged(self):
		first = _feature((0.1, 0.2), (True, True))
		invalid = _feature((0.7, 0.8), (False, False))
		client = _FakeClient([first, invalid])
		wrapper = module.CutieObjectWrapper(
			_FakeEnv(), _cfg(False), _client=client
		)
		def unexpected_pipeline_call(*_args, **_kwargs):
			raise AssertionError('disabled last-valid path called its pipeline')
		wrapper._reset_last_valid_memory = unexpected_pipeline_call
		wrapper._apply_last_valid_memory = unexpected_pipeline_call
		reset = wrapper.reset()
		for part in _parts(reset):
			np.testing.assert_array_equal(part, first)
		observation, _, _, _ = wrapper.step(None)
		parts = _parts(observation)
		np.testing.assert_array_equal(parts[0], first)
		np.testing.assert_array_equal(parts[1], first)
		np.testing.assert_array_equal(parts[2], invalid)
		memory = wrapper.metrics()['last_valid_memory']
		self.assertFalse(memory['enabled'])
		self.assertEqual(memory['substitutions'], 0)
		self.assertEqual(memory['invalid_without_history'], 0)

	def test_02_invalid_roles_carry_content_but_keep_current_status(self):
		first = _feature((0.1, 0.2), (True, True))
		mixed = _feature((0.7, 0.3), (False, True))
		invalid = _feature((0.8, 0.9), (False, False))
		wrapper = module.CutieObjectWrapper(
			_FakeEnv(), _cfg(True), _client=_FakeClient([first, mixed, invalid])
		)
		wrapper.reset()
		second, _, _, _ = wrapper.step(None)
		latest = _parts(second)[-1]
		np.testing.assert_array_equal(
			latest[0, :module.FRAME_CONTENT_DIM],
			first[0, :module.FRAME_CONTENT_DIM],
		)
		np.testing.assert_array_equal(
			latest[0, module.FRAME_CONTENT_DIM:],
			mixed[0, module.FRAME_CONTENT_DIM:],
		)
		np.testing.assert_array_equal(latest[1], mixed[1])

		third, _, _, _ = wrapper.step(None)
		latest = _parts(third)[-1]
		for role, source in enumerate((first, mixed)):
			np.testing.assert_array_equal(
				latest[role, :module.FRAME_CONTENT_DIM],
				source[role, :module.FRAME_CONTENT_DIM],
			)
			np.testing.assert_array_equal(
				latest[role, module.FRAME_CONTENT_DIM:],
				invalid[role, module.FRAME_CONTENT_DIM:],
			)

		metrics = wrapper.metrics()
		memory = metrics['last_valid_memory']
		self.assertTrue(memory['enabled'])
		self.assertEqual(memory['content_dim'], 586)
		self.assertEqual(memory['status_dim'], 4)
		self.assertEqual(memory['substitutions'], 3)
		self.assertEqual(memory['invalid_without_history'], 0)
		whole_arm = memory['per_role']['whole_arm']
		goal = memory['per_role']['goal']
		self.assertEqual(
			(whole_arm['age'], whole_arm['max_age'], whole_arm['substitutions']),
			(2, 2, 2),
		)
		self.assertEqual(
			(goal['age'], goal['max_age'], goal['substitutions']),
			(1, 1, 1),
		)
		# Standard health metrics still describe raw tracker validity.
		self.assertEqual(metrics['role_metrics']['whole_arm']['valid_frames'], 1)
		self.assertEqual(metrics['role_metrics']['goal']['valid_frames'], 2)

	def test_03_reset_prevents_cross_episode_substitution(self):
		first = _feature((0.1, 0.2), (True, True))
		lost = _feature((0.7, 0.8), (False, False))
		new_episode_lost = _feature((0.4, 0.5), (False, False))
		wrapper = module.CutieObjectWrapper(
			_FakeEnv(), _cfg(True),
			_client=_FakeClient([first, lost, new_episode_lost]),
		)
		wrapper.reset()
		wrapper.step(None)
		observation = wrapper.reset()
		for part in _parts(observation):
			np.testing.assert_array_equal(part, new_episode_lost)
		memory = wrapper.metrics()['last_valid_memory']
		self.assertEqual(memory['substitutions'], 2)
		self.assertEqual(memory['invalid_without_history'], 2)
		for role in module.ROLE_NAMES:
			self.assertFalse(memory['per_role'][role]['has_memory'])
			self.assertIsNone(memory['per_role'][role]['age'])
			self.assertEqual(memory['per_role'][role]['max_age'], 1)

	def test_04_non_boolean_config_fails_closed(self):
		cfg = _cfg(False)
		cfg['cutie_object_last_valid_memory'] = 'false'
		with self.assertRaisesRegex(ValueError, 'must be a boolean'):
			module.CutieObjectWrapper(
				_FakeEnv(), cfg, _client=_FakeClient([_feature((0.1, 0.2), (True, True))])
			)

	def test_05_hybrid_mode_rejects_last_valid_memory(self):
		cfg = _cfg(True)
		cfg['flat_anchor_mode'] = 'cutie_hybrid'
		with self.assertRaisesRegex(ValueError, 'restricted.*cutie_object_only'):
			module.CutieObjectWrapper(
				_FakeEnv(), cfg,
				_client=_FakeClient([_feature((0.1, 0.2), (True, True))]),
			)


if __name__ == '__main__':
	suite = unittest.defaultTestLoader.loadTestsFromTestCase(
		CutieLastValidMemoryContract
	)
	result = unittest.TextTestRunner(verbosity=2).run(suite)
	if not result.wasSuccessful():
		raise SystemExit(1)
	print('CUTIE_LAST_VALID_MEMORY_CONTRACT_OK', {
		'content_dim': module.FRAME_CONTENT_DIM,
		'status_dim': module.STATUS_DIM,
		'stacked_shape': (2, module.STACKED_FEATURE_DIM),
		'default_enabled': False,
	})
