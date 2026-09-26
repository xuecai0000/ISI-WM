"""Live Cutie plus low-cost task-internal proprioception wrapper."""

import math
import time

import gymnasium as gym
import numpy as np
import torch

from common import cutie_proprio as contract
from envs.wrappers.cutie_object import CutieObjectWrapper


class _CutieConfigView:
	"""Expose the unchanged inner Cutie worker contract."""

	_OVERRIDES = {
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': 3,
		'cutie_object_input_dim': 1770,
	}

	def __init__(self, cfg):
		self._cfg = cfg

	def get(self, key, default=None):
		if key in self._OVERRIDES:
			return self._OVERRIDES[key]
		return self._cfg.get(key, default)

	def __getattr__(self, key):
		if key in self._OVERRIDES:
			return self._OVERRIDES[key]
		return getattr(self._cfg, key)


class CutieProprioWrapper(gym.Wrapper):
	"""Pack unchanged 2x1770 Cutie tokens and 2x4 proprioception."""

	def __init__(self, env, cfg, *, _client=None):
		contract.validate_config(cfg)
		self._cfg = cfg
		self._mode = contract.mode(cfg)
		self._uses_cutie = contract.uses_cutie(cfg)
		self._uses_proprio = contract.uses_proprio(cfg)
		self._velocity_scale = float(cfg.get('cutie_proprio_velocity_scale', 10.0))
		self._physics = self._find_physics(env) if self._uses_proprio else None
		if self._physics is not None:
			model = self._physics.model
			if int(model.nq) != 2 or int(model.nv) != 2:
				raise ValueError(
					'Cutie-proprio tasks require nq=nv=2, got '
					f'nq={int(model.nq)}, nv={int(model.nv)}.'
				)
		inner = CutieObjectWrapper(env, _CutieConfigView(cfg), _client=_client)
		super().__init__(inner)
		self._frames = 0
		self._packing_runtime_ms = 0.0
		self.observation_space = gym.spaces.Dict({
			'object': gym.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(contract.NUM_ROLES, contract.INPUT_DIM),
				dtype=np.float32,
			),
		})

	@staticmethod
	def _find_physics(env):
		seen = set()
		current = env
		while current is not None and id(current) not in seen:
			seen.add(id(current))
			physics = getattr(current, 'physics', None)
			if physics is not None:
				return physics
			next_env = getattr(current, 'env', None)
			if next_env is None:
				next_env = getattr(current, '_env', None)
			current = next_env
		raise ValueError('Cutie-proprio could not locate dm-control physics.')

	def _proprio(self):
		if not self._uses_proprio:
			return np.zeros((contract.NUM_ROLES, contract.PROPRIO_DIM), dtype=np.float32)
		qpos = np.asarray(self._physics.data.qpos, dtype=np.float64).reshape(-1)
		qvel = np.asarray(self._physics.data.qvel, dtype=np.float64).reshape(-1)
		if qpos.shape != (2,) or qvel.shape != (2,):
			raise RuntimeError(f'Invalid task qpos/qvel shapes: {qpos.shape}, {qvel.shape}.')
		if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
			raise RuntimeError('Task proprioception contains non-finite values.')
		task = str(self._cfg.task)
		if task in {'acrobot-swingup', 'reacher-visual-small'}:
			angles = (qpos[0], qpos[0] + qpos[1])
			velocities = (qvel[0], qvel[0] + qvel[1])
			return np.asarray([
				[
					math.sin(angle), math.cos(angle),
					math.tanh(velocity / self._velocity_scale), 1.0,
				]
				for angle, velocity in zip(angles, velocities)
			], dtype=np.float32)
		if task == 'cartpole-swingup':
			return np.asarray([
				[
					math.tanh(qpos[0]), 0.0,
					math.tanh(qvel[0] / self._velocity_scale), 1.0,
				],
				[
					math.sin(qpos[1]), math.cos(qpos[1]),
					math.tanh(qvel[1] / self._velocity_scale), 1.0,
				],
			], dtype=np.float32)
		raise RuntimeError(f'No proprioception adapter for task {task!r}.')

	def _pack(self, source):
		started = time.perf_counter()
		try:
			visual = source['object']
		except (KeyError, TypeError) as exc:
			raise RuntimeError('Inner Cutie wrapper did not emit object tokens.') from exc
		if hasattr(visual, 'detach'):
			visual = visual.detach().cpu().numpy()
		visual = np.asarray(visual, dtype=np.float32)
		if visual.shape != (contract.NUM_ROLES, contract.VISUAL_DIM):
			raise RuntimeError(f'Invalid Cutie descriptor shape {visual.shape}.')
		if not self._uses_cutie:
			visual = np.zeros_like(visual)
		value = np.concatenate((visual, self._proprio()), axis=-1)
		if value.shape != (contract.NUM_ROLES, contract.INPUT_DIM):
			raise AssertionError(f'Invalid Cutie-proprio shape {value.shape}.')
		if not np.isfinite(value).all():
			raise RuntimeError('Cutie-proprio emitted non-finite values.')
		self._frames += 1
		self._packing_runtime_ms += (time.perf_counter() - started) * 1000.0
		return {'object': torch.from_numpy(np.ascontiguousarray(value))}

	def reset(self, **kwargs):
		return self._pack(self.env.reset(**kwargs))

	def step(self, action):
		source, reward, done, info = self.env.step(action)
		return self._pack(source), reward, done, info

	@property
	def cutie_proprio_ready(self):
		return {
			**contract.observation_contract(self._cfg),
			'observation_shape': [contract.NUM_ROLES, contract.INPUT_DIM],
			'cutie': self.env.cutie_ready,
		}

	def metrics(self):
		return {
			**self.cutie_proprio_ready,
			'frames': int(self._frames),
			'mean_packing_runtime_ms': float(
				self._packing_runtime_ms / max(self._frames, 1)
			),
			'cutie_metrics': self.env.metrics(),
		}
