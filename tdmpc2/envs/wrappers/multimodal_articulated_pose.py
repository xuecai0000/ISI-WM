"""Equal-shape visual/proprioceptive Acrobot ablation wrapper."""

import math
import time

import gymnasium as gym
import numpy as np

from common import gt_articulated_pose
from common import multimodal_articulated_pose as contract
from common import visual_articulated_pose
from envs.wrappers.visual_articulated_pose import VisualArticulatedPoseWrapper


class _VisualConfigView:
	"""Present the inner RGB predictor with its original strict contract."""

	_OVERRIDES = {
		'cutie_object_observation_variant': visual_articulated_pose.VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_frame_schema': visual_articulated_pose.FRAME_SCHEMA,
		'cutie_object_frame_dim': visual_articulated_pose.FRAME_DIM,
		'cutie_object_stack_frames': visual_articulated_pose.STACK_FRAMES,
		'cutie_object_input_dim': visual_articulated_pose.INPUT_DIM,
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


class MultimodalArticulatedPoseWrapper(gym.Wrapper):
	"""Expose the same 2x25 object tensor in all three ablation modes.

	The first 21 values are the existing causal visual pose descriptor.  The
	last four values are current global joint angle/velocity features.  A mode
	removes a modality by writing exact zeros; it never changes observation or
	model dimensions.
	"""

	def __init__(self, env, cfg, *, predictor=None):
		contract.validate_config(cfg)
		self._cfg = cfg
		self._mode = contract.mode(cfg)
		self._uses_visual = contract.uses_visual(cfg)
		self._uses_proprio = contract.uses_proprio(cfg)
		self._velocity_scale = float(
			cfg.get('multimodal_proprio_velocity_scale', 10.0)
		)
		wrapped = env
		if self._uses_visual:
			wrapped = VisualArticulatedPoseWrapper(
				env, _VisualConfigView(cfg), predictor=predictor,
			)
		super().__init__(wrapped)
		self._physics = self._find_physics(env) if self._uses_proprio else None
		if self._physics is not None:
			model = self._physics.model
			if int(model.nq) != 2 or int(model.nv) != 2:
				raise ValueError(
					'multimodal Acrobot proprioception requires nq=nv=2, got '
					f'nq={int(model.nq)}, nv={int(model.nv)}.'
				)
		self._frames = 0
		self._runtime_ms = 0.0
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
			current = getattr(current, 'env', getattr(current, '_env', None))
		raise ValueError('multimodal articulated pose could not locate dm-control physics.')

	def _proprio(self):
		if not self._uses_proprio:
			return np.zeros((contract.NUM_ROLES, contract.PROPRIO_DIM), dtype=np.float32)
		qpos = np.asarray(self._physics.data.qpos, dtype=np.float64).reshape(-1)
		qvel = np.asarray(self._physics.data.qvel, dtype=np.float64).reshape(-1)
		if qpos.shape != (2,) or qvel.shape != (2,):
			raise RuntimeError(f'Invalid Acrobot qpos/qvel shapes: {qpos.shape}, {qvel.shape}.')
		if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
			raise RuntimeError('Acrobot proprioception contains non-finite values.')
		# qpos[1]/qvel[1] are relative to the first link.  Convert both roles to
		# absolute-frame quantities so their semantics match the visual descriptor.
		angles = np.asarray((qpos[0], qpos[0] + qpos[1]), dtype=np.float64)
		velocities = np.asarray((qvel[0], qvel[0] + qvel[1]), dtype=np.float64)
		return np.asarray([
			[
				math.sin(angle), math.cos(angle),
				math.tanh(velocity / self._velocity_scale), 1.0,
			]
			for angle, velocity in zip(angles, velocities)
		], dtype=np.float32)

	def _observation(self, source):
		started = time.perf_counter()
		if self._uses_visual:
			try:
				visual = np.asarray(source['object'], dtype=np.float32)
			except (KeyError, TypeError) as exc:
				raise RuntimeError('Inner visual-pose wrapper did not emit object.') from exc
			if visual.shape != (contract.NUM_ROLES, contract.VISUAL_DIM):
				raise RuntimeError(f'Invalid inner visual descriptor shape {visual.shape}.')
		else:
			visual = np.zeros(
				(contract.NUM_ROLES, contract.VISUAL_DIM), dtype=np.float32,
			)
		value = np.concatenate((visual, self._proprio()), axis=-1)
		if value.shape != (contract.NUM_ROLES, contract.INPUT_DIM):
			raise AssertionError(f'Invalid multimodal observation shape {value.shape}.')
		if not np.isfinite(value).all():
			raise RuntimeError('Multimodal articulated pose emitted non-finite values.')
		self._frames += 1
		self._runtime_ms += (time.perf_counter() - started) * 1000.0
		return {'object': np.array(value, dtype=np.float32, order='C', copy=True)}

	def reset(self, **kwargs):
		return self._observation(self.env.reset(**kwargs))

	def step(self, action):
		source, reward, done, info = self.env.step(action)
		return self._observation(source), reward, done, info

	@property
	def multimodal_pose_ready(self):
		return {
			**contract.observation_contract(self._cfg),
			'observation_shape': [contract.NUM_ROLES, contract.INPUT_DIM],
		}

	def metrics(self):
		payload = {
			**self.multimodal_pose_ready,
			'frames': int(self._frames),
			'mean_fusion_runtime_ms': float(self._runtime_ms / max(self._frames, 1)),
		}
		inner_metrics = getattr(self.env, 'metrics', None)
		if callable(inner_metrics):
			payload['visual_pose'] = inner_metrics()
		return payload
