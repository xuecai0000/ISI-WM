"""Privileged MuJoCo kinematics observation for Acrobot diagnostics only."""

from collections import deque
import math
import time

import gymnasium as gym
import numpy as np

from common import gt_articulated_pose as contract


class GTArticulatedPoseWrapper(gym.Wrapper):
	"""Export two causal global-segment pose histories and hide state/RGB."""

	def __init__(self, env, cfg):
		super().__init__(env)
		contract.validate_config(cfg)
		self._physics = self._find_physics(env)
		self._validate_anatomy_names()
		self._frames = deque(maxlen=contract.STACK_FRAMES)
		self._previous_theta = None
		self._previous_time = None
		self._metric_frames = 0
		self._metric_resets = 0
		self._metric_runtime_ms = 0.0
		self.observation_space = gym.spaces.Dict({
			'object': gym.spaces.Box(
				low=-np.inf,
				high=np.inf,
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
		raise ValueError('gt_articulated_pose could not locate dm-control physics.')

	def _validate_anatomy_names(self):
		model = self._physics.model
		if int(model.nq) != 2 or int(model.nv) != 2:
			raise ValueError(
				'gt_articulated_pose requires Acrobot nq=nv=2, got '
				f'nq={int(model.nq)}, nv={int(model.nv)}.'
			)
		# ``id2name`` is part of MuJoCo's public model API.  Do not depend on
		# NamedArray internals here: this check is meant to catch a server-side
		# dm-control/XML drift before any controller training begins.
		body_names = {
			str(name)
			for index in range(int(model.nbody))
			if (name := model.id2name(index, 'body')) is not None
		}
		site_names = {
			str(name)
			for index in range(int(model.nsite))
			if (name := model.id2name(index, 'site')) is not None
		}
		missing_bodies = {'upper_arm', 'lower_arm'} - body_names
		missing_sites = {'tip'} - site_names
		if missing_bodies or missing_sites:
			raise ValueError(
				'gt_articulated_pose Acrobot anatomy mismatch: '
				f'missing bodies={sorted(missing_bodies)}, '
				f'missing sites={sorted(missing_sites)}.'
			)

	@staticmethod
	def _xz(named_array, name):
		value = np.asarray(named_array[name, ['x', 'z']], dtype=np.float64)
		if value.shape != (2,) or not np.isfinite(value).all():
			raise RuntimeError(f'Non-finite or invalid kinematics for {name!r}: {value!r}.')
		return value

	def _read_frame(self, *, reset):
		start = time.perf_counter()
		physics = self._physics
		base = self._xz(physics.named.data.xpos, 'upper_arm')
		elbow = self._xz(physics.named.data.xpos, 'lower_arm')
		tip = self._xz(physics.named.data.site_xpos, 'tip')
		points = ((base, elbow), (elbow, tip))
		lengths = tuple(float(np.linalg.norm(end - begin)) for begin, end in points)
		if any(abs(length - 1.) > 1e-4 for length in lengths):
			raise RuntimeError(
				'gt_articulated_pose requires two unit Acrobot links, got '
				f'{lengths}.'
			)
		theta = np.asarray([
			math.atan2(float(end[0] - begin[0]), float(end[1] - begin[1]))
			for begin, end in points
		], dtype=np.float64)
		physics_time = getattr(physics, 'time', None)
		current_time = float(
			physics_time() if callable(physics_time) else np.asarray(physics.data.time)
		)
		if not math.isfinite(current_time):
			raise RuntimeError(f'Non-finite simulator time: {current_time}.')
		if reset:
			omega = np.zeros(contract.NUM_ROLES, dtype=np.float64)
		else:
			dt = current_time - self._previous_time
			if not math.isfinite(dt) or dt <= 0.:
				raise RuntimeError(f'Non-positive simulator time delta for pose oracle: {dt}.')
			delta = (theta - self._previous_theta + math.pi) % (2. * math.pi) - math.pi
			# Signed velocity of each segment's global (absolute-frame) angle.
			omega = delta / dt
		frame = np.asarray([
			[
				begin[0] / contract.TOTAL_LINK_LENGTH,
				begin[1] / contract.TOTAL_LINK_LENGTH,
				end[0] / contract.TOTAL_LINK_LENGTH,
				end[1] / contract.TOTAL_LINK_LENGTH,
				math.sin(angle), math.cos(angle), angular_velocity,
			]
			for (begin, end), angle, angular_velocity in zip(points, theta, omega)
		], dtype=np.float32)
		if frame.shape != (contract.NUM_ROLES, contract.FRAME_DIM):
			raise AssertionError(f'Pose oracle emitted invalid frame shape {frame.shape}.')
		if not np.isfinite(frame).all():
			raise RuntimeError('Pose oracle emitted non-finite values.')
		self._previous_theta = theta
		self._previous_time = current_time
		self._metric_frames += 1
		self._metric_runtime_ms += (time.perf_counter() - start) * 1000.
		return np.array(frame, dtype=np.float32, order='C', copy=True)

	def _stacked(self):
		value = np.concatenate(tuple(self._frames), axis=-1)
		if value.shape != (contract.NUM_ROLES, contract.INPUT_DIM):
			raise AssertionError(f'Pose oracle emitted invalid stack shape {value.shape}.')
		return {'object': np.array(value, dtype=np.float32, order='C', copy=True)}

	def reset(self, **kwargs):
		self.env.reset(**kwargs)
		self._previous_theta = None
		self._previous_time = None
		frame = self._read_frame(reset=True)
		self._frames.clear()
		for _ in range(contract.STACK_FRAMES):
			self._frames.append(frame.copy())
		self._metric_resets += 1
		return self._stacked()

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		self._frames.append(self._read_frame(reset=False))
		return self._stacked(), reward, done, info

	@property
	def gt_pose_ready(self):
		return {
			**contract.observation_contract(),
			'observation_shape': [contract.NUM_ROLES, contract.INPUT_DIM],
		}

	def metrics(self):
		return {
			**self.gt_pose_ready,
			'frames': int(self._metric_frames),
			'resets': int(self._metric_resets),
			'runtime_ms': float(self._metric_runtime_ms),
		}
