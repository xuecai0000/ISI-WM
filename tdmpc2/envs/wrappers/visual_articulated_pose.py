"""Causal RGB-to-articulated-pose wrapper for Acrobot."""

from collections import deque
import math
import time

import gymnasium as gym
import numpy as np

from common import visual_articulated_pose as contract
from perception.acrobot_keypoint_detector import AcrobotKeypointPredictor


def latest_rgb(observation):
	"""Extract the newest uint8 HWC frame from a stacked pixel observation."""
	if isinstance(observation, dict):
		observation = observation.get('rgb')
	if hasattr(observation, 'detach'):
		observation = observation.detach().cpu().numpy()
	value = np.asarray(observation)
	if value.ndim != 3:
		raise ValueError(f'Visual pose expected a CHW/HWC RGB stack, got {value.shape}.')
	if value.shape[-1] == 3:
		frame = value
	elif value.shape[0] >= 3:
		frame = np.moveaxis(value[-3:], 0, -1)
	else:
		raise ValueError(f'Visual pose cannot locate RGB channels in {value.shape}.')
	if frame.dtype != np.uint8:
		if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 255:
			raise ValueError('Visual pose RGB values must be finite and inside [0,255].')
		frame = np.rint(frame).astype(np.uint8)
	return np.array(frame, dtype=np.uint8, order='C', copy=True)


def points_to_role_frame(points_xz, previous_theta, *, dt, angular_velocity=None):
	"""Convert normalized base/elbow/tip XZ into the Oracle-compatible 2x7 frame."""
	points = np.asarray(points_xz, dtype=np.float64)
	if points.shape != (3, 2) or not np.isfinite(points).all():
		raise ValueError(f'Invalid predicted Acrobot points: {points!r}.')
	segments = ((points[0], points[1]), (points[1], points[2]))
	theta = np.asarray([
		math.atan2(float(end[0] - start[0]), float(end[1] - start[1]))
		for start, end in segments
	], dtype=np.float64)
	if angular_velocity is not None:
		omega = np.asarray(angular_velocity, dtype=np.float64)
		if omega.shape != (2,) or not np.isfinite(omega).all():
			raise ValueError(f'Invalid predicted angular velocity: {omega!r}.')
	elif previous_theta is None:
		omega = np.zeros(2, dtype=np.float64)
	else:
		if not math.isfinite(dt) or dt <= 0:
			raise ValueError(f'Visual pose dt must be positive, got {dt}.')
		delta = (theta - np.asarray(previous_theta) + math.pi) % (2 * math.pi) - math.pi
		omega = delta / dt
	frame = np.asarray([
		[start[0], start[1], end[0], end[1], math.sin(angle), math.cos(angle), rate]
		for (start, end), angle, rate in zip(segments, theta, omega)
	], dtype=np.float32)
	return frame, theta


class VisualArticulatedPoseWrapper(gym.Wrapper):
	"""Replace pixels with a predicted pose having the GT Oracle's exact shape."""

	def __init__(self, env, cfg, *, predictor=None):
		super().__init__(env)
		contract.validate_config(cfg)
		self._cfg = cfg
		self._history = int(cfg.get('visual_pose_history', 4))
		self._image_size = int(cfg.get('visual_pose_image_size', 64))
		self._control_dt = float(cfg.get('visual_pose_control_dt', 0.04))
		self._confidence_threshold = float(
			cfg.get('visual_pose_confidence_threshold', 0.0)
		)
		self._use_mask = bool(cfg.get('visual_pose_use_cutie_mask', False))
		self._predictor = predictor or AcrobotKeypointPredictor(
			cfg.get('visual_pose_checkpoint'),
			device=str(cfg.get('visual_pose_device', 'cuda:0')),
		)
		model_cfg = self._predictor.config
		mismatch = {}
		for name, expected in (
			('history', self._history), ('image_size', self._image_size),
			('use_foreground_mask', self._use_mask),
		):
			actual = getattr(model_cfg, name)
			if actual != expected:
				mismatch[name] = (actual, expected)
		if mismatch:
			raise ValueError(f'Visual pose checkpoint/config mismatch: {mismatch}.')
		self._action_dim = int(model_cfg.action_dim)
		if int(np.prod(env.action_space.shape)) != self._action_dim:
			raise ValueError(
				'Visual pose checkpoint action dimension does not match environment: '
				f'{self._action_dim} vs {env.action_space.shape}.'
			)
		self._rgb = deque(maxlen=self._history)
		self._actions = deque(maxlen=self._history - 1)
		self._masks = deque(maxlen=self._history)
		self._pose_frames = deque(maxlen=contract.STACK_FRAMES)
		self._previous_theta = None
		self._last_points = None
		self._frames = 0
		self._low_confidence_frames = 0
		self._runtime_ms = 0.
		self._confidence_sum = np.zeros(3, dtype=np.float64)
		self.observation_space = gym.spaces.Dict({
			'object': gym.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(contract.NUM_ROLES, contract.INPUT_DIM), dtype=np.float32,
			),
		})

	def _resize(self, frame):
		if frame.shape[:2] == (self._image_size, self._image_size):
			return frame
		# Dependency-light nearest resize. The trained resolution is normally 64.
		y = np.linspace(0, frame.shape[0] - 1, self._image_size).round().astype(int)
		x = np.linspace(0, frame.shape[1] - 1, self._image_size).round().astype(int)
		return np.ascontiguousarray(frame[y][:, x])

	def _foreground_mask(self):
		if not self._use_mask:
			return None
		provider = getattr(self.env, 'visual_pose_foreground_mask', None)
		if not callable(provider):
			raise RuntimeError(
				'visual_pose_use_cutie_mask requires a deployable '
				'visual_pose_foreground_mask() provider; simulator masks are forbidden.'
			)
		mask = np.asarray(provider(), dtype=np.bool_)
		if mask.shape != (self._image_size, self._image_size):
			raise ValueError(f'Visual pose foreground mask has invalid shape {mask.shape}.')
		return np.array(mask, dtype=np.float32, order='C', copy=True)

	def _append_source(self, observation, action=None):
		frame = self._resize(latest_rgb(observation))
		self._rgb.append(frame)
		if action is not None:
			value = np.asarray(action, dtype=np.float32).reshape(-1)
			if value.shape != (self._action_dim,) or not np.isfinite(value).all():
				raise ValueError(f'Invalid visual pose action {value!r}.')
			self._actions.append(value.copy())
		if self._use_mask:
			self._masks.append(self._foreground_mask())

	def _infer_frame(self, *, reset):
		started = time.perf_counter()
		output = self._predictor.predict(
			np.stack(self._rgb), np.stack(self._actions),
			np.stack(self._masks) if self._use_mask else None,
		)
		points = np.asarray(output['world_xz'], dtype=np.float64)
		confidence = np.asarray(output['confidence'], dtype=np.float64)
		angular_velocity = np.asarray(output['angular_velocity'], dtype=np.float64)
		if (
			points.shape != (3, 2) or confidence.shape != (3,)
			or angular_velocity.shape != (2,)
		):
			raise RuntimeError('Visual pose predictor returned incompatible shapes.')
		if (
			not np.isfinite(points).all() or not np.isfinite(confidence).all()
			or not np.isfinite(angular_velocity).all()
		):
			raise RuntimeError('Visual pose predictor returned non-finite values.')
		if np.any(confidence < self._confidence_threshold):
			self._low_confidence_frames += 1
		# Confidence is measured and gated by the long-sequence preflight. It is
		# deliberately not allowed to replace a prediction with simulator state.
		frame, theta = points_to_role_frame(
			points, None if reset else self._previous_theta, dt=self._control_dt,
			angular_velocity=(
				np.zeros(2, dtype=np.float64) if reset else angular_velocity
			),
		)
		self._previous_theta = theta
		self._last_points = points
		self._frames += 1
		self._confidence_sum += confidence
		self._runtime_ms += (time.perf_counter() - started) * 1000.
		return frame

	def _observation(self):
		value = np.concatenate(tuple(self._pose_frames), axis=-1)
		if value.shape != (contract.NUM_ROLES, contract.INPUT_DIM):
			raise AssertionError(f'Invalid visual pose observation shape {value.shape}.')
		return {'object': np.array(value, dtype=np.float32, order='C', copy=True)}

	def reset(self, **kwargs):
		observation = self.env.reset(**kwargs)
		self._rgb.clear()
		self._actions.clear()
		self._masks.clear()
		self._pose_frames.clear()
		self._previous_theta = None
		self._append_source(observation)
		for _ in range(self._history - 1):
			self._rgb.append(self._rgb[0].copy())
			if self._use_mask:
				self._masks.append(self._masks[0].copy())
		for _ in range(self._history - 1):
			self._actions.append(np.zeros(self._action_dim, dtype=np.float32))
		frame = self._infer_frame(reset=True)
		for _ in range(contract.STACK_FRAMES):
			self._pose_frames.append(frame.copy())
		return self._observation()

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		self._append_source(observation, action=action)
		self._pose_frames.append(self._infer_frame(reset=False))
		return self._observation(), reward, done, info

	@property
	def visual_pose_ready(self):
		return {
			**contract.observation_contract(self._cfg),
			'observation_shape': [contract.NUM_ROLES, contract.INPUT_DIM],
			'checkpoint_format': self._predictor.metadata.get('format'),
		}

	def metrics(self):
		denominator = max(self._frames, 1)
		return {
			**self.visual_pose_ready,
			'frames': int(self._frames),
			'low_confidence_frames': int(self._low_confidence_frames),
			'mean_confidence': (self._confidence_sum / denominator).tolist(),
			'mean_runtime_ms': float(self._runtime_ms / denominator),
		}
