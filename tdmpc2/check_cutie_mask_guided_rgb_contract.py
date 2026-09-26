"""CPU contracts for the zero-initialized Cutie mask-guided RGB path.

This test deliberately needs no live simulator, Cutie repository, checkpoint,
or CUDA device.  It checks the public causal observation boundary with a fake
synchronous tracker, then checks the real TD-MPC2 encoder/model implementation.
"""

from __future__ import annotations

from collections import deque
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TD_ROOT = ROOT / 'tdmpc2'
if str(TD_ROOT) not in sys.path:
	sys.path.insert(0, str(TD_ROOT))

from common import MODEL_SIZE, init, layers, mask_guided_rgb  # noqa: E402
from common.buffer import Buffer  # noqa: E402
from common.world_model import WorldModel  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from tensordict import TensorDict  # noqa: E402


CONFIG_PATH = TD_ROOT / 'config.yaml'
ROLE_CASES = {
	'acrobot-swingup': ('whole_acrobot',),
	'reacher-easy': ('whole_arm', 'goal'),
	'hopper-stand': ('torso', 'leg', 'foot'),
}
TASK_BY_ROLE_COUNT = {
	len(role_names): task for task, role_names in ROLE_CASES.items()
}


class _Box:
	def __init__(self, low, high, shape, dtype):
		self.low = low
		self.high = high
		self.shape = tuple(shape)
		self.dtype = np.dtype(dtype)


class _Dict:
	def __init__(self, spaces):
		self.spaces = dict(spaces)


def _native_hwc(observation) -> np.ndarray:
	value = observation.detach().cpu().numpy() if torch.is_tensor(observation) else np.asarray(observation)
	if value.shape != (9, 64, 64) or value.dtype != np.uint8:
		raise ValueError('fake native RGB requires uint8 [9,64,64]')
	return np.ascontiguousarray(value[-3:].transpose(1, 2, 0))


def _fake_masks_and_diagnostics(roles: int, frame_index: int):
	masks = np.zeros((roles, 64, 64), dtype=np.bool_)
	for role in range(roles):
		y0 = 4 + 9 * role + frame_index
		x0 = 7 + 8 * role
		masks[role, y0:y0 + 5, x0:x0 + 6] = True
	areas = masks.reshape(roles, -1).sum(axis=-1, dtype=np.int64)
	lost = np.zeros(roles, dtype=np.bool_)
	valid = areas > 0
	diagnostics = {
		'confidence': np.linspace(0.55, 0.75, roles, dtype=np.float32)
		+ np.float32(0.01 * frame_index),
		'lost': lost,
		'valid': valid,
		'mask_score': np.linspace(0.45, 0.65, roles, dtype=np.float32)
		+ np.float32(0.01 * frame_index),
		'mask_area_pixels': areas,
	}
	return masks, diagnostics


def _fake_frame(diagnostics, marker: float) -> np.ndarray:
	roles = len(diagnostics['valid'])
	frame = np.full((roles, 590), marker, dtype=np.float32)
	frame[:, 586:590] = np.stack((
		diagnostics['confidence'],
		diagnostics['lost'].astype(np.float32),
		diagnostics['valid'].astype(np.float32),
		diagnostics['mask_score'],
	), axis=-1)
	return frame


def _expected_status(diagnostics) -> torch.Tensor:
	return torch.from_numpy(np.stack((
		diagnostics['confidence'],
		diagnostics['lost'].astype(np.float32),
		diagnostics['valid'].astype(np.float32),
		diagnostics['mask_score'],
	), axis=-1))


def _load_wrapper_subject():
	"""Load only the wrapper under test around a dependency-light fake parent."""
	package_name = 'cutie_mask_guided_rgb_contract_subject'
	package = ModuleType(package_name)
	package.__path__ = []
	sys.modules[package_name] = package

	cutie_name = package_name + '.cutie_object'
	cutie = ModuleType(cutie_name)
	cutie.FRAME_CONTENT_DIM = 586
	cutie.FRAME_FEATURE_DIM = 590
	cutie.TASK_ROLE_NAMES = dict(ROLE_CASES)

	def task_role_names(task, contract='canonical_v1'):
		if contract != 'canonical_v1':
			raise ValueError('fake subject accepts canonical_v1 only')
		return cutie.TASK_ROLE_NAMES[task]

	cutie.task_role_names = task_role_names
	cutie.CutieObjectWorkerError = RuntimeError
	cutie._native_hwc_rgb = _native_hwc

	class FakeCutieObjectWrapper:
		def __init__(self, env, cfg, *, _client=None):
			self.env = env
			self._client = _client
			self._role_names = tuple(cfg['cutie_object_role_names'])
			self._object_frames = deque(maxlen=3)
			self._latest_source_rgb_sha256 = None
			self._metric_episode_step = -1
			self._closed = False

		@property
		def latest_source_rgb_sha256(self):
			return self._latest_source_rgb_sha256

		@property
		def cutie_ready(self):
			return {'status': 'ready'}

		def _track(self, rgb, *, reset):
			frame = _native_hwc(rgb)
			self._latest_source_rgb_sha256 = hashlib.sha256(
				frame.tobytes(order='C')
			).hexdigest()
			value = (
				self._client.reset_track(frame)
				if reset else self._client.track(frame)
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
			return {'frames': len(self._client.frames)}

		def close(self):
			self._closed = True

	cutie.CutieObjectWrapper = FakeCutieObjectWrapper
	sys.modules[cutie_name] = cutie

	gym = ModuleType('gymnasium')
	gym.Wrapper = object
	gym.spaces = SimpleNamespace(Box=_Box, Dict=_Dict)
	previous_gym = sys.modules.get('gymnasium')
	sys.modules['gymnasium'] = gym
	try:
		path = TD_ROOT / 'envs' / 'wrappers' / 'cutie_mask_guided_rgb.py'
		spec = importlib.util.spec_from_file_location(
			package_name + '.cutie_mask_guided_rgb', path
		)
		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)
	finally:
		if previous_gym is None:
			del sys.modules['gymnasium']
		else:
			sys.modules['gymnasium'] = previous_gym
	return module


WRAPPER = _load_wrapper_subject()


def _base_contract_config(task='reacher-easy', roles=None, **updates):
	roles = ROLE_CASES[task] if roles is None else tuple(roles)
	value = {
		'task': task,
		'obs': 'rgb',
		'multitask': False,
		'model_size': 5,
		'latent_dim': 512,
		'flat_anchor': False,
		'cutie_masked_rgb_enabled': False,
		'robust_object_field_enabled': False,
		'cutie_mask_guided_rgb_enabled': True,
		'cutie_mask_guided_rgb_schema': mask_guided_rgb.SCHEMA,
		'cutie_mask_guided_rgb_hidden_channels': 32,
		'cutie_mask_guided_rgb_random_shift_pad': 3,
		'cutie_mask_guided_rgb_role_pool': 'max',
		'cutie_mask_guided_rgb_correction_limit': 0.25,
		'cutie_mask_guided_rgb_ablation_mode': 'none',
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
		'cutie_object_num_roles': len(roles),
		'cutie_object_role_names': list(roles),
		'cutie_object_task_role_contract': 'canonical_v1',
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'object_state_supervision_enabled': False,
		'object_state_supervision_collect_labels': False,
		'object_state_bottleneck_enabled': False,
		'cutie_object_native_highres_enabled': False,
		'cutie_object_spatial_token_enabled': False,
		'cutie_object_variable_graph_enabled': False,
		'cutie_object_true_entity_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_policy_burst_plan': None,
		'cutie_object_regression_encoder': None,
	}
	value.update(updates)
	return value


class _FakeEnv:
	def __init__(self):
		self.value = 20
		self.last_observation = None
		self.observation_space = _Box(0, 255, (9, 64, 64), np.uint8)
		self.forbidden_sensor_reads = 0

	def _rgb(self):
		self.value += 1
		self.last_observation = torch.full(
			(9, 64, 64), self.value, dtype=torch.uint8
		)
		return self.last_observation.clone()

	def reset(self):
		return self._rgb()

	def step(self, _action):
		return self._rgb(), 3.0, False, {}

	def render(self, *args, **kwargs):
		self.forbidden_sensor_reads += 1
		raise AssertionError('mask-guided wrapper must not render')


class _FakeClient:
	def __init__(self, roles):
		self.roles = int(roles)
		self.frames = []

	def _next(self, frame):
		self.frames.append(np.array(frame, copy=True))
		masks, diagnostics = _fake_masks_and_diagnostics(
			self.roles, len(self.frames) - 1
		)
		self.last_masks = masks
		self.last_diagnostics = diagnostics
		return _fake_frame(diagnostics, marker=float(len(self.frames)))

	def reset_track(self, frame):
		return self._next(frame)

	def track(self, frame):
		return self._next(frame)

	def close(self):
		pass


def _model_config(roles=2, *, guided=True, task=None):
	task = TASK_BY_ROLE_COUNT[roles] if task is None else task
	role_names = ROLE_CASES[task]
	if len(role_names) != roles:
		raise ValueError(f'{task!r} does not have K={roles} roles.')
	cfg = OmegaConf.load(CONFIG_PATH)
	cfg.task = task
	cfg.obs = 'rgb'
	cfg.seed = 123
	cfg.model_size = 5
	for key, value in MODEL_SIZE[5].items():
		cfg[key] = value
	cfg.multitask = False
	cfg.tasks = [task]
	cfg.task_dim = 0
	cfg.task_title = task.replace('-', ' ').title()
	cfg.action_dim = 2
	cfg.episode_length = 500
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	for key, value in _base_contract_config(
		task=task, roles=role_names
	).items():
		cfg[key] = value
	cfg.task = task
	cfg.cutie_mask_guided_rgb_enabled = bool(guided)
	cfg.obs_shape = (
		mask_guided_rgb.observation_shapes(cfg)
		if guided else {'rgb': (9, 64, 64)}
	)
	return cfg


def _observation(roles, batch=2):
	generator = torch.Generator(device='cpu')
	generator.manual_seed(271828 + roles)
	rgb = torch.randint(
		0, 256, (batch, 9, 64, 64), dtype=torch.uint8,
		generator=generator,
	)
	masks = torch.zeros(batch, roles, 3, 64, 64, dtype=torch.bool)
	for role in range(roles):
		for frame in range(3):
			y0 = 3 + 8 * role + frame
			x0 = 5 + 7 * role + 2 * frame
			masks[:, role, frame, y0:y0 + 9, x0:x0 + 11] = True
	status = torch.zeros(batch, roles, 3, 4, dtype=torch.float32)
	status[..., 0] = 0.85
	status[..., 2] = 1.0
	status[..., 3] = 0.75
	return {'rgb': rgb, 'object_mask': masks, 'tracker_status': status}


def _state_equal(left, right):
	left_state, right_state = left.state_dict(), right.state_dict()
	def equal_value(left_value, right_value):
		if torch.is_tensor(left_value) or torch.is_tensor(right_value):
			return (
				torch.is_tensor(left_value)
				and torch.is_tensor(right_value)
				and torch.equal(left_value, right_value)
			)
		return type(left_value) is type(right_value) and left_value == right_value
	return (
		set(left_state) == set(right_state)
		and all(equal_value(left_state[key], right_state[key]) for key in left_state)
	)


def _build_pair(roles, seed=314159):
	raw_cfg = _model_config(roles, guided=False)
	guided_cfg = _model_config(roles, guided=True)
	torch.manual_seed(seed)
	raw = WorldModel(raw_cfg)
	raw_rng = torch.get_rng_state().clone()
	torch.manual_seed(seed)
	guided = WorldModel(guided_cfg)
	guided_rng = torch.get_rng_state().clone()
	return raw, guided, raw_rng, guided_rng


def _agent_shell(cfg, seed):
	"""Construct only the CPU state needed by TDMPC2.save/load."""
	torch.manual_seed(seed)
	agent = TDMPC2.__new__(TDMPC2)
	nn.Module.__init__(agent)
	agent.cfg = cfg
	agent.model = WorldModel(cfg)
	agent._learned_object_belief = False
	agent._belief_aux_updates = 0
	agent._cutie_regression_encoder = None
	agent._cutie_object_mode = False
	agent._cutie_object_only = False
	agent._hybrid_graph = False
	agent._reward_graph = False
	agent._robust_object_field_auxiliary_contract = None
	agent._online_object_belief = None
	agent._online_object_belief_action = None
	return agent


class WrapperContractTests(unittest.TestCase):
	def test_fake_wrapper_reset_shift_and_exact_k(self):
		for task, role_names in ROLE_CASES.items():
			with self.subTest(task=task):
				cfg = _base_contract_config(task=task)
				env = _FakeEnv()
				client = _FakeClient(len(role_names))
				wrapped = WRAPPER.CutieMaskGuidedRGBWrapper(
					env, cfg, _client=client
				)
				self.assertEqual(
					set(wrapped.observation_space.spaces),
					{'rgb', 'object_mask', 'tracker_status'},
				)
				first = wrapped.reset()
				self.assertTrue(torch.equal(first['rgb'], env.last_observation))
				self.assertEqual(first['object_mask'].dtype, torch.bool)
				self.assertEqual(first['tracker_status'].dtype, torch.float32)
				self.assertEqual(
					tuple(first['object_mask'].shape),
					(len(role_names), 3, 64, 64),
				)
				self.assertEqual(
					tuple(first['tracker_status'].shape),
					(len(role_names), 3, 4),
				)
				self.assertTrue(torch.equal(
					first['object_mask'][:, 0], first['object_mask'][:, 2]
				))
				self.assertTrue(torch.equal(
					first['tracker_status'][:, 0], first['tracker_status'][:, 2]
				))
				self.assertTrue(torch.equal(
					first['tracker_status'][:, 2],
					_expected_status(client.last_diagnostics),
				))
				first_masks = first['object_mask'].clone()
				first_status = first['tracker_status'].clone()
				second, reward, done, _ = wrapped.step(torch.zeros(1))
				self.assertTrue(torch.equal(
					second['object_mask'][:, 1], first_masks[:, 2]
				))
				self.assertTrue(torch.equal(
					second['tracker_status'][:, 1], first_status[:, 2]
				))
				self.assertFalse(torch.equal(
					second['tracker_status'][:, 1], second['tracker_status'][:, 2]
				))
				self.assertTrue(torch.equal(
					second['tracker_status'][:, 2],
					_expected_status(client.last_diagnostics),
				))
				self.assertFalse(torch.equal(
					second['object_mask'][:, 1], second['object_mask'][:, 2]
				))
				self.assertEqual((reward, done), (3.0, False))
				self.assertEqual(set(first), {
					'rgb', 'object_mask', 'tracker_status',
				})
				self.assertNotIn('object', first)
				self.assertEqual(env.forbidden_sensor_reads, 0)
				self.assertEqual(wrapped.metrics()['mask_guided_rgb_frames'], 2)
				self.assertFalse(
					wrapped.cutie_ready['privileged_runtime_segmentation']
				)

	def test_privileged_modes_and_role_mismatch_fail_closed(self):
		bad = (
			_base_contract_config(cutie_object_allow_simulator_runtime=True),
			_base_contract_config(
				cutie_object_allow_simulator_kinematics_runtime=True
			),
			_base_contract_config(object_state_supervision_enabled=True),
			_base_contract_config(object_state_supervision_collect_labels=True),
			_base_contract_config(cutie_object_num_roles=3),
			_base_contract_config(cutie_object_role_names=['goal', 'whole_arm']),
		)
		for cfg in bad:
			with self.subTest(cfg=cfg), self.assertRaises(ValueError):
				WRAPPER.validate_mask_guided_rgb_observation_config(cfg)

		source = (
			TD_ROOT / 'envs' / 'wrappers' / 'cutie_mask_guided_rgb.py'
		).read_text(encoding='utf-8')
		start = source.index('\tdef _observation(self, rgb):')
		end = source.index('\n\t@property\n\tdef cutie_ready', start)
		method = source[start:end]
		for forbidden in ('self.env', '.physics', '.render(', 'segmentation(', 'kinematics'):
			self.assertNotIn(forbidden, method)

	def test_replay_schema_keeps_only_rgb_masks_and_status(self):
		for roles in (1, 2, 3):
			with self.subTest(roles=roles):
				cfg = _model_config(roles, guided=True)
				buffer = Buffer.__new__(Buffer)
				buffer.cfg = cfg
				observation = _observation(roles)
				replay = TensorDict(observation, batch_size=[2])
				buffer._validate_observation_schema(replay, 'replay insertion')
				leaking = replay.clone()
				leaking.set(
					'object', torch.zeros(2, roles, 1770, dtype=torch.float32)
				)
				with self.assertRaises(ValueError):
					buffer._validate_observation_schema(
						leaking, 'replay insertion'
					)


class ModelContractTests(unittest.TestCase):
	def test_zero_initialized_raw_cnn_parity_for_k1_k2_k3(self):
		for roles in (1, 2, 3):
			with self.subTest(roles=roles):
				raw, guided, raw_rng, guided_rng = _build_pair(roles)
				self.assertTrue(torch.equal(raw_rng, guided_rng))
				self.assertTrue(_state_equal(raw._dynamics, guided._dynamics))
				self.assertTrue(_state_equal(raw._reward, guided._reward))
				self.assertTrue(_state_equal(raw._pi, guided._pi))
				self.assertTrue(_state_equal(raw._Qs, guided._Qs))

				raw_convs = [raw._encoder['rgb'][index] for index in (2, 4, 6, 8)]
				guided_encoder = guided._encoder['rgb']
				guided_convs = [
					guided_encoder.rgb_spatial[index]
					for index in (0, 2, 4, 6)
				]
				for raw_conv, guided_conv in zip(raw_convs, guided_convs):
					self.assertTrue(_state_equal(raw_conv, guided_conv))
				self.assertEqual(
					float(guided_encoder.film_output.weight.abs().max()), 0.0
				)
				self.assertEqual(
					float(guided_encoder.film_output.bias.abs().max()), 0.0
				)

				obs = _observation(roles)
				fixed_shift = torch.tensor(
					[[[[1.0, 5.0]]], [[[6.0, 0.0]]]], dtype=torch.float32
				)
				shifted_rgb, _ = guided_encoder.augmentation(
					obs['rgb'], obs['object_mask'], shift_index=fixed_shift
				)
				with torch.no_grad():
					expected = raw._encoder['rgb'][1:](shifted_rgb)
					actual, correction = guided_encoder(
						obs, shift_index=fixed_shift, return_correction=True
					)
				self.assertTrue(torch.equal(expected, actual))
				self.assertEqual(float(correction.abs().max()), 0.0)
				self.assertEqual(tuple(actual.shape), (2, 512))
				groups = actual.reshape(2, -1, 8).sum(dim=-1)
				self.assertTrue(torch.allclose(groups, torch.ones_like(groups)))

				torch.manual_seed(161803)
				raw_random = raw.encode(obs['rgb'], task=None)
				raw_after = torch.get_rng_state().clone()
				torch.manual_seed(161803)
				guided_random = guided.encode(obs, task=None)
				guided_after = torch.get_rng_state().clone()
				self.assertTrue(torch.equal(raw_random, guided_random))
				self.assertTrue(torch.equal(raw_after, guided_after))

				sequence_rgb = torch.stack((
					obs['rgb'], torch.roll(obs['rgb'], 1, dims=-1),
					torch.roll(obs['rgb'], -1, dims=-2),
				))
				sequence = {
					'rgb': sequence_rgb,
					'object_mask': torch.stack((
						obs['object_mask'], obs['object_mask'], obs['object_mask'],
					)),
					'tracker_status': torch.stack((
						obs['tracker_status'], obs['tracker_status'],
						obs['tracker_status'],
					)),
				}
				torch.manual_seed(173205)
				raw_sequence = raw.encode(sequence_rgb, task=None)
				raw_sequence_rng = torch.get_rng_state().clone()
				torch.manual_seed(173205)
				guided_sequence = guided.encode(sequence, task=None)
				guided_sequence_rng = torch.get_rng_state().clone()
				self.assertTrue(torch.equal(raw_sequence, guided_sequence))
				self.assertTrue(torch.equal(
					raw_sequence_rng, guided_sequence_rng
				))

	def test_residual_receives_gradients_and_updates(self):
		_, guided, _, _ = _build_pair(2, seed=141421)
		encoder = guided._encoder['rgb']
		obs = _observation(2)
		fixed_shift = torch.tensor(
			[[[[2.0, 4.0]]], [[[5.0, 1.0]]]], dtype=torch.float32
		)
		weights = torch.linspace(-1.0, 1.0, 512).expand(2, -1)
		# TD-MPC2 trains this branch with Adam; use the same adaptive optimizer so
		# the zero-initialized two-stage gradient path is tested realistically.
		optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
		before_output = encoder.film_output.weight.detach().clone()
		before_stem = {
			name: value.detach().clone()
			for name, value in encoder.role_stem.named_parameters()
		}

		latent, correction = encoder(
			obs, shift_index=fixed_shift, return_correction=True
		)
		self.assertEqual(float(correction.abs().max()), 0.0)
		loss = (latent * weights).mean()
		loss.backward()
		self.assertGreater(
			float(encoder.film_output.weight.grad.abs().max()), 0.0
		)
		optimizer.step()
		self.assertFalse(torch.equal(before_output, encoder.film_output.weight))

		optimizer.zero_grad(set_to_none=True)
		latent, correction = encoder(
			obs, shift_index=fixed_shift, return_correction=True
		)
		self.assertGreater(float(correction.abs().max()), 0.0)
		loss = (latent * weights).mean()
		loss.backward()
		stem_gradient = max(
			float(parameter.grad.abs().max())
			for parameter in encoder.role_stem.parameters()
			if parameter.grad is not None
		)
		self.assertGreater(stem_gradient, 0.0)
		optimizer.step()
		self.assertTrue(any(
			not torch.equal(before_stem[name], value)
			for name, value in encoder.role_stem.named_parameters()
		))

	def test_registered_ablation_modes_are_matched_and_causal(self):
		none_cfg = _model_config(2)
		off_cfg = _model_config(2)
		off_cfg.cutie_mask_guided_rgb_ablation_mode = 'guidance_off'
		torch.manual_seed(200003)
		none_model = WorldModel(none_cfg)
		none_rng = torch.get_rng_state().clone()
		torch.manual_seed(200003)
		guided = WorldModel(off_cfg)
		off_rng = torch.get_rng_state().clone()
		self.assertTrue(_state_equal(none_model, guided))
		self.assertTrue(torch.equal(none_rng, off_rng))
		self.assertEqual(
			sum(parameter.numel() for parameter in none_model.parameters()),
			sum(parameter.numel() for parameter in guided.parameters()),
		)
		encoder = guided._encoder['rgb']
		observation = _observation(2)
		fixed_shift = torch.tensor(
			[[[[0.0, 0.0]]], [[[0.0, 0.0]]]], dtype=torch.float32
		)

		# The spatial negative control is a bijection: every bit and per-role
		# area is retained, but no non-periodic mask remains registered.
		encoder.ablation_mode = 'spatial_permute_v1'
		rolled = encoder._ablate_masks(observation['object_mask'].float())
		self.assertTrue(torch.equal(
			rolled.sum(dim=(-2, -1)),
			observation['object_mask'].sum(dim=(-2, -1)).float(),
		))
		self.assertFalse(torch.equal(
			rolled.bool(), observation['object_mask']
		))

		# Even with an active, nonzero mask head, guidance_off is exactly the
		# official RGB path and cannot leak a gradient through the correction.
		encoder.ablation_mode = 'guidance_off'
		nn.init.constant_(encoder.film_output.weight, 0.01)
		nn.init.constant_(encoder.film_output.bias, 0.01)
		with torch.no_grad():
			shifted_rgb, _ = encoder.augmentation(
				observation['rgb'], observation['object_mask'],
				shift_index=fixed_shift,
			)
			expected = encoder.rgb_norm(
				encoder.rgb_flatten(
					encoder.rgb_spatial(encoder.rgb_preprocess(shifted_rgb))
				)
			)
			actual, correction = encoder(
				observation, shift_index=fixed_shift, return_correction=True
			)
		self.assertEqual(float(correction.abs().max()), 0.0)
		self.assertTrue(torch.equal(expected, actual))
		encoder.zero_grad(set_to_none=True)
		actual, correction = encoder(
			observation, shift_index=fixed_shift, return_correction=True
		)
		weights = torch.linspace(-1.0, 1.0, 512).expand_as(actual)
		(actual * weights).mean().backward()
		branch_parameters = (
			list(encoder.role_stem.parameters())
			+ list(encoder.status_projection.parameters())
			+ list(encoder.film_output.parameters())
		)
		self.assertTrue(all(
			parameter.grad is None
			or float(parameter.grad.abs().max()) == 0.0
			for parameter in branch_parameters
		))
		self.assertTrue(any(
			parameter.grad is not None
			and float(parameter.grad.abs().max()) > 0.0
			for parameter in encoder.rgb_spatial.parameters()
		))

		for mode in mask_guided_rgb.ABLATION_MODES:
			cfg = _model_config(2)
			cfg.cutie_mask_guided_rgb_ablation_mode = mode
			mask_guided_rgb.validate_config(cfg)
			self.assertEqual(mask_guided_rgb.contract(cfg)['ablation_mode'], mode)
		bad = _model_config(2)
		bad.cutie_mask_guided_rgb_ablation_mode = 'unregistered'
		with self.assertRaises(ValueError):
			mask_guided_rgb.validate_config(bad)

	def test_every_tracker_status_field_changes_active_forward(self):
		_, guided, _, _ = _build_pair(1, seed=223607)
		encoder = guided._encoder['rgb']
		# Make the status-only route deterministic and active. This must fail if
		# any one of confidence/lost/valid/mask_score is dropped from the forward.
		for parameter in encoder.role_stem.parameters():
			nn.init.zeros_(parameter)
		nn.init.constant_(encoder.status_projection.weight, 0.01)
		nn.init.zeros_(encoder.status_projection.bias)
		nn.init.constant_(encoder.film_output.weight, 0.01)
		nn.init.zeros_(encoder.film_output.bias)
		observation = _observation(1)
		fixed_shift = torch.tensor(
			[[[[3.0, 3.0]]], [[[3.0, 3.0]]]], dtype=torch.float32
		)
		with torch.no_grad():
			_, baseline = encoder(
				observation, shift_index=fixed_shift, return_correction=True
			)
		for field in range(mask_guided_rgb.STATUS_DIM):
			with self.subTest(field=field):
				changed = {
					key: value.clone() for key, value in observation.items()
				}
				if field == 1:
					changed['tracker_status'][..., field] = 1.0
				elif field == 2:
					changed['tracker_status'][..., field] = 0.0
				else:
					changed['tracker_status'][..., field] = 0.25
				with torch.no_grad():
					_, correction = encoder(
						changed, shift_index=fixed_shift,
						return_correction=True,
					)
				self.assertFalse(torch.equal(baseline, correction))

	def test_checkpoint_roundtrip_and_role_contract_rejection(self):
		cfg = _model_config(2, guided=True)
		source = _agent_shell(cfg, seed=12345)
		stream = io.BytesIO()
		source.save(stream)
		stream.seek(0)
		payload = torch.load(stream, map_location='cpu', weights_only=False)
		self.assertEqual(
			payload['checkpoint_contract'].get('mask_guided_rgb'),
			dict(mask_guided_rgb.contract(cfg)),
		)

		target = _agent_shell(_model_config(2, guided=True), seed=54321)
		stream.seek(0)
		target.load(stream)
		self.assertTrue(_state_equal(source.model, target.model))

		corrupt = copy.deepcopy(payload)
		corrupt['checkpoint_contract']['mask_guided_rgb']['ordered_roles'] = [
			'goal', 'whole_arm'
		]
		with self.assertRaises(RuntimeError):
			target.load(corrupt)

		wrong_roles = _agent_shell(_model_config(3, guided=True), seed=54321)
		with self.assertRaises(RuntimeError):
			wrong_roles.load(copy.deepcopy(payload))


if __name__ == '__main__':
	unittest.main(verbosity=2)
