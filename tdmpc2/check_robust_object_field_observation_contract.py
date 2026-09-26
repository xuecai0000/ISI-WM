"""Dependency-light contracts for the ROF-WM V0 observation wrapper."""

from __future__ import annotations

import ast
from collections import deque
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TD_ROOT = ROOT / 'tdmpc2'
if str(TD_ROOT) not in sys.path:
	sys.path.insert(0, str(TD_ROOT))

from common import robust_object_field_observation as observation_schema  # noqa: E402


def masks_and_diagnostics(*, roles=2, offset=0):
	masks = np.zeros((roles, 64, 64), dtype=np.bool_)
	for index in range(roles):
		masks[index, 5 + offset + index:10 + offset + index, 8:14] = True
	areas = masks.reshape(roles, -1).sum(axis=-1, dtype=np.int64)
	border = np.concatenate(
		(masks[:, 0], masks[:, -1], masks[:, :, 0], masks[:, :, -1]), axis=-1
	).any(axis=-1)
	nonempty = areas > 0
	lost = np.zeros(roles, dtype=np.bool_)
	finite = np.ones(roles, dtype=np.bool_)
	diagnostics = {
		'valid': (~lost) & nonempty & finite,
		'lost': lost,
		'mask_nonempty': nonempty,
		'feature_finite': finite,
		'mask_area_pixels': areas,
		'mask_touches_border': border,
		'confidence': np.linspace(0.7, 0.9, roles, dtype=np.float32),
		'mask_score': np.linspace(0.6, 0.8, roles, dtype=np.float32),
		'runtime_ms': 5.5,
	}
	return masks, diagnostics


def feature(diagnostics, value=0.0):
	roles = len(diagnostics['valid'])
	result = np.full((roles, 590), value, dtype=np.float32)
	result[:, 586:590] = np.stack((
		diagnostics['confidence'],
		diagnostics['lost'].astype(np.float32),
		diagnostics['valid'].astype(np.float32),
		diagnostics['mask_score'],
	), axis=-1)
	return result


def packet_fixture():
	rgb = np.arange(9 * 64 * 64, dtype=np.uint8).reshape(9, 64, 64)
	mask, diagnostic = masks_and_diagnostics()
	masks = np.stack((mask, mask, mask), axis=1)
	objects = np.concatenate((
		feature(diagnostic, 1.0), feature(diagnostic, 2.0),
		feature(diagnostic, 3.0),
	), axis=-1)
	return rgb, objects, masks, diagnostic


class SchemaContracts(unittest.TestCase):
	def test_schema_and_shapes_match_model_contract_source(self):
		path = TD_ROOT / 'common' / 'robust_object_field.py'
		tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
		constants = {}
		for node in tree.body:
			if not isinstance(node, ast.Assign) or len(node.targets) != 1:
				continue
			target = node.targets[0]
			if isinstance(target, ast.Name) and target.id in {
				'SCHEMA', 'IMAGE_SIZE', 'STACK_FRAMES', 'RGB_CHANNELS',
				'FRAME_DIM', 'OBJECT_DIM', 'STATUS_START',
			}:
				constants[target.id] = eval(
					compile(ast.Expression(node.value), str(path), 'eval'),
					{'__builtins__': {}}, constants,
				)
		self.assertEqual(constants['SCHEMA'], observation_schema.SCHEMA)
		self.assertEqual(constants['IMAGE_SIZE'], observation_schema.IMAGE_SIZE)
		self.assertEqual(constants['STACK_FRAMES'], observation_schema.STACK_FRAMES)
		self.assertEqual(constants['RGB_CHANNELS'], observation_schema.RGB_CHANNELS)
		self.assertEqual(constants['FRAME_DIM'], observation_schema.FRAME_DIM)
		self.assertEqual(constants['OBJECT_DIM'], observation_schema.OBJECT_DIM)
		self.assertEqual(constants['STATUS_START'], observation_schema.STATUS_START)

	def test_exact_public_shapes_and_status_binding(self):
		rgb, objects, masks, diagnostic = packet_fixture()
		packet = observation_schema.build_packet(
			task='reacher-easy', role_names=('whole_arm', 'goal'),
			rgb_stack=rgb, object_stack=objects,
			object_mask_stack=masks, diagnostics=diagnostic, sequence_id=7,
		)
		self.assertEqual(packet.role_names, ('whole_arm', 'goal'))
		self.assertEqual(packet.rgb.shape, (9, 64, 64))
		self.assertEqual(packet.object.shape, (2, 1770))
		self.assertEqual(packet.object_mask.shape, (2, 3, 64, 64))
		np.testing.assert_array_equal(packet.role_exists, [1.0, 1.0])
		self.assertFalse(packet.object.flags.writeable)
		latest_hwc = np.ascontiguousarray(rgb[-3:].transpose(1, 2, 0))
		self.assertEqual(
			packet.source_rgb_sha256,
			hashlib.sha256(latest_hwc.tobytes(order='C')).hexdigest(),
		)
		public = packet.as_numpy_observation()
		observation_schema.validate_numpy_observation(public, role_count=2)
		self.assertEqual(
			set(public), {'rgb', 'object', 'object_mask', 'role_exists'}
		)

	def test_no_padding_dtype_or_same_response_mismatch_is_accepted(self):
		rgb, objects, masks, diagnostic = packet_fixture()
		cases = []
		cases.append((objects[:1], masks, diagnostic, 'object must'))
		cases.append((objects.astype(np.float64), masks, diagnostic, 'float32'))
		bad_status = dict(diagnostic)
		bad_status['confidence'] = diagnostic['confidence'].copy()
		bad_status['confidence'][0] = 0.1
		cases.append((objects, masks, bad_status, 'same Cutie response'))
		bad_area = dict(diagnostic)
		bad_area['mask_area_pixels'] = diagnostic['mask_area_pixels'].copy()
		bad_area['mask_area_pixels'][0] += 1
		cases.append((objects, masks, bad_area, 'mask areas'))
		for candidate_objects, candidate_masks, candidate_diagnostics, message in cases:
			with self.subTest(message=message), self.assertRaisesRegex(
				observation_schema.RobustObjectFieldObservationError, message
			):
				observation_schema.build_packet(
					task='reacher-easy', role_names=('whole_arm', 'goal'),
					rgb_stack=rgb, object_stack=candidate_objects,
					object_mask_stack=candidate_masks,
					diagnostics=candidate_diagnostics, sequence_id=0,
				)
		packet = observation_schema.build_packet(
			task='reacher-easy', role_names=('whole_arm', 'goal'),
			rgb_stack=rgb, object_stack=objects,
			object_mask_stack=masks, diagnostics=diagnostic, sequence_id=0,
		)
		public = packet.as_numpy_observation()
		public['role_exists'][1] = 0.0
		with self.assertRaisesRegex(
			observation_schema.RobustObjectFieldObservationError, 'padding roles'
		):
			observation_schema.validate_numpy_observation(public, role_count=2)

	def test_binding_changes_with_rgb_object_mask_or_sequence(self):
		rgb, objects, masks, diagnostic = packet_fixture()
		def make(rgb_value=rgb, object_value=objects, mask_value=masks,
				 diagnostics=diagnostic, sequence=0):
			return observation_schema.build_packet(
				task='reacher-easy', role_names=('whole_arm', 'goal'),
				rgb_stack=rgb_value, object_stack=object_value,
				object_mask_stack=mask_value, diagnostics=diagnostics,
				sequence_id=sequence,
			)
		base = make().binding_sha256
		changed_rgb = rgb.copy(); changed_rgb[-1, 0, 0] += 1
		changed_objects = objects.copy(); changed_objects[0, 0] += 1
		changed_mask, changed_diagnostic = masks_and_diagnostics(offset=1)
		changed_masks = masks.copy(); changed_masks[:, -1] = changed_mask
		for candidate in (
			make(rgb_value=changed_rgb),
			make(object_value=changed_objects),
			make(mask_value=changed_masks, diagnostics=changed_diagnostic),
			make(sequence=1),
		):
			self.assertNotEqual(base, candidate.binding_sha256)


def install_stubs():
	class Box:
		def __init__(self, low, high, shape, dtype):
			self.low, self.high = low, high
			self.shape, self.dtype = tuple(shape), np.dtype(dtype)

	class Dict:
		def __init__(self, spaces):
			self.spaces = dict(spaces)

	gym = ModuleType('gymnasium')
	gym.spaces = SimpleNamespace(Box=Box, Dict=Dict)
	sys.modules['gymnasium'] = gym
	torch = ModuleType('torch')
	torch.is_tensor = lambda _value: False
	torch.from_numpy = lambda value: value
	sys.modules['torch'] = torch

	model = ModuleType('common.robust_object_field')
	model.SCHEMA = observation_schema.SCHEMA
	model.calls = 0
	def validate_config(cfg, *, require_obs_shape=True):
		model.calls += 1
		if not cfg.get('robust_object_field_enabled', False):
			raise ValueError('ROF disabled')
		if require_obs_shape:
			raise AssertionError('wrapper must validate before obs_shape exists')
	model.validate_config = validate_config
	sys.modules['common.robust_object_field'] = model

	package = ModuleType('robust_object_field_contract_package')
	package.__path__ = []
	sys.modules[package.__name__] = package
	cutie = ModuleType(package.__name__ + '.cutie_object')
	cutie.TASK_ROLE_NAMES = {
		'reacher-easy': ('whole_arm', 'goal'),
		'finger-spin': ('finger', 'spinner'),
		'acrobot-swingup': ('whole_acrobot',),
	}
	cutie.task_role_names = lambda task, contract='canonical_v1': (
		cutie.TASK_ROLE_NAMES[task]
		if contract == 'canonical_v1' else ('upper_arm', 'lower_arm')
	)
	cutie.CutieObjectWorkerError = RuntimeError
	def native_hwc(observation):
		value = np.asarray(observation)
		return np.ascontiguousarray(value[-3:].transpose(1, 2, 0))
	cutie._native_hwc_rgb = native_hwc

	class Base:
		def __init__(self, env, cfg, *, _client=None):
			self.env, self._client = env, _client
			self._role_names = tuple(cfg['cutie_object_role_names'])
			self._object_frames = deque(maxlen=3)
			self._latest_source_rgb_sha256 = None
			self._metric_episode_step = -1

		@property
		def latest_source_rgb_sha256(self):
			return self._latest_source_rgb_sha256

		@property
		def cutie_ready(self):
			return {'status': 'ready'}

		def _track(self, rgb, *, reset):
			frame = native_hwc(rgb)
			self._latest_source_rgb_sha256 = hashlib.sha256(
				frame.tobytes(order='C')
			).hexdigest()
			value = (
				self._client.reset_track(frame) if reset
				else self._client.track(frame)
			)
			if reset:
				self._metric_episode_step = 0
				self._object_frames.clear()
				for _ in range(3):
					self._object_frames.append(value.copy())
			else:
				self._metric_episode_step += 1
				self._object_frames.append(value.copy())

		def reset(self):
			rgb = self.env.reset()
			self._track(rgb, reset=True)
			return self._observation(rgb)

		def step(self, action):
			rgb, reward, done, info = self.env.step(action)
			self._track(rgb, reset=False)
			return self._observation(rgb), reward, done, info

		def metrics(self):
			return {'base_frames': len(self._client.frames)}

		def close(self):
			pass
	cutie.CutieObjectWrapper = Base
	sys.modules[cutie.__name__] = cutie

	path = TD_ROOT / 'envs' / 'wrappers' / 'robust_object_field.py'
	spec = importlib.util.spec_from_file_location(
		package.__name__ + '.robust_object_field', path
	)
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module, model, Box


wrapper, model_contract, Box = install_stubs()


def config(**updates):
	value = {
		'task': 'reacher-easy',
		'robust_object_field_enabled': True,
		'robust_object_field_schema': observation_schema.SCHEMA,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'obs': 'rgb',
		'multitask': False,
		'model_size': 5,
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_num_roles': 2,
		'cutie_object_role_names': ['whole_arm', 'goal'],
		'cutie_object_task_role_contract': 'canonical_v1',
		'cutie_masked_rgb_enabled': False,
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_native_highres_enabled': False,
		'cutie_object_true_entity_enabled': False,
	}
	value.update(updates)
	return value


class FakeEnv:
	def __init__(self):
		self.value = 20
		self.observation_space = Box(0, 255, (9, 64, 64), np.uint8)

	def reset(self):
		self.value += 1
		return np.full((9, 64, 64), self.value, dtype=np.uint8)

	def step(self, _action):
		self.value += 1
		return np.full((9, 64, 64), self.value, dtype=np.uint8), 3.0, False, {}


class FakeClient:
	def __init__(self, *, corrupt=False):
		self.frames = []
		self.corrupt = corrupt

	def _next(self, frame):
		self.frames.append(frame.copy())
		masks, diagnostic = masks_and_diagnostics(offset=len(self.frames) - 1)
		value = feature(diagnostic, value=float(len(self.frames)))
		if self.corrupt:
			diagnostic['mask_area_pixels'][0] += 1
		self.last_masks, self.last_diagnostics = masks, diagnostic
		return value

	def reset_track(self, frame):
		return self._next(frame)

	def track(self, frame):
		return self._next(frame)

	def close(self):
		pass


class WrapperContracts(unittest.TestCase):
	def test_canonical_config_and_privilege_rejections(self):
		self.assertEqual(
			wrapper.validate_robust_object_field_observation_config(config()),
			('reacher-easy', ('whole_arm', 'goal')),
		)
		bad = (
			config(cutie_object_num_roles=3),
			config(cutie_object_role_names=['goal', 'whole_arm']),
			config(cutie_object_task_role_contract='legacy_acrobot_links_v1'),
			config(cutie_object_native_highres_enabled=True),
			config(cutie_object_true_entity_enabled=True),
			config(visual_foreground_erosion_pixels=1),
			config(visual_pose_checkpoint='/privileged.pt'),
		)
		for cfg in bad:
			with self.subTest(cfg=cfg), self.assertRaises(ValueError):
				wrapper.validate_robust_object_field_observation_config(cfg)
		self.assertGreaterEqual(model_contract.calls, 1 + len(bad))

	def test_wrapper_exports_exact_stacks_and_mask_history(self):
		wrapped = wrapper.RobustObjectFieldObservationWrapper(
			FakeEnv(), config(), _client=FakeClient()
		)
		self.assertEqual(
			set(wrapped.observation_space.spaces),
			{'rgb', 'object', 'object_mask', 'role_exists'},
		)
		first = wrapped.reset()
		self.assertEqual(first['rgb'].shape, (9, 64, 64))
		self.assertEqual(first['object'].shape, (2, 1770))
		self.assertEqual(first['object_mask'].shape, (2, 3, 64, 64))
		np.testing.assert_array_equal(first['role_exists'], [1.0, 1.0])
		np.testing.assert_array_equal(
			first['object_mask'][:, 0], first['object_mask'][:, 2]
		)
		first_mask = first['object_mask'][:, -1].copy()
		first_binding = wrapped.robust_object_field_last_binding_sha256
		second, reward, done, _ = wrapped.step(np.zeros(1))
		np.testing.assert_array_equal(second['object_mask'][:, 1], first_mask)
		self.assertFalse(np.array_equal(
			second['object_mask'][:, 1], second['object_mask'][:, 2]
		))
		self.assertEqual((reward, done), (3.0, False))
		self.assertNotEqual(
			first_binding, wrapped.robust_object_field_last_binding_sha256
		)
		self.assertEqual(
			wrapped.robust_object_field_role_names, ('whole_arm', 'goal')
		)
		self.assertEqual(wrapped.cutie_ready['role_exists'], 'all_one')
		self.assertFalse(wrapped.cutie_ready['privileged_runtime_segmentation'])
		self.assertEqual(wrapped.metrics()['robust_object_field_frames'], 2)

	def test_malformed_same_response_fails_before_policy(self):
		wrapped = wrapper.RobustObjectFieldObservationWrapper(
			FakeEnv(), config(), _client=FakeClient(corrupt=True)
		)
		with self.assertRaisesRegex(RuntimeError, 'atomic observation validation'):
			wrapped.reset()

	def test_worker_switch_and_runtime_method_are_privilege_free(self):
		cutie_source = (TD_ROOT / 'envs' / 'wrappers' / 'cutie_object.py').read_text(
			encoding='utf-8'
		)
		self.assertIn("cfg.get('robust_object_field_enabled', False)", cutie_source)
		source = (TD_ROOT / 'envs' / 'wrappers' / 'robust_object_field.py').read_text(
			encoding='utf-8'
		)
		start = source.index('\tdef _observation(self, rgb):')
		end = source.index('\n\t@property\n\tdef robust_object_field_role_names', start)
		method = source[start:end]
		for forbidden in ('self.env', '.physics', '.render(', 'segmentation('):
			self.assertNotIn(forbidden, method)


if __name__ == '__main__':
	unittest.main(verbosity=2)
