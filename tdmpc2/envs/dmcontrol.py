from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch

from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.cutie_object import CutieObjectWrapper
from envs.wrappers.cutie_masked_rgb import CutieMaskedRGBWrapper
from envs.wrappers.cutie_mask_guided_rgb import CutieMaskGuidedRGBWrapper
from envs.wrappers.cutie_proprio import CutieProprioWrapper
from envs.wrappers.flat_anchor import FlatAnchorWrapper
from envs.wrappers.foreground_stress import ForegroundErosionWrapper
from envs.wrappers.gt_articulated_pose import GTArticulatedPoseWrapper
from envs.wrappers.multimodal_articulated_pose import (
	MultimodalArticulatedPoseWrapper,
)
from envs.wrappers.robust_object_field import (
	RobustObjectFieldObservationWrapper,
)
from envs.wrappers.visual_articulated_pose import VisualArticulatedPoseWrapper
from envs.wrappers.timeout import Timeout
from envs.wrappers.video_background import ColorMultiVideoBackgroundWrapper


def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class DMControlWrapper:
	def __init__(self, env, domain):
		self.env = env
		self.camera_id = 2 if domain == 'quadruped' else 0
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.action_spec_dtype = env.action_spec().dtype

	@property
	def unwrapped(self):
		return self.env
	
	def _obs_to_array(self, obs):
		return torch.from_numpy(
			np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32))
	
	def reset(self):
		return self._obs_to_array(self.env.reset().observation)

	def step(self, action):
		reward = 0
		action = action.astype(self.action_spec_dtype)
		for _ in range(2):
			step = self.env.step(action)
			reward += step.reward
		return self._obs_to_array(step.observation), reward, False, defaultdict(float)
	
	def render(self, width=384, height=384, camera_id=None):
		camera_id = self.camera_id if camera_id is None else camera_id
		return self.env.physics.render(height, width, camera_id)

	def close(self):
		"""Close the wrapped control environment when it exposes a close hook."""
		close = getattr(self.env, 'close', None)
		return close() if callable(close) else None


class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=3, size=64):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, size, size), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)
		self._size = size

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def cutie_same_state_rgb(self, *, height, width):
		"""Render perception-only RGB without changing the policy observation.

		This method is called only by the explicitly enabled Cutie high-resolution
		path, after ``reset``/``step`` has already rendered the ordinary 64x64
		observation.  ``physics.render`` is read-only, so both images correspond to
		the same simulator state.  No segmentation or state observation is queried.
		"""
		height, width = int(height), int(width)
		if height < 1 or width < 1:
			raise ValueError('Cutie RGB render dimensions must be positive.')
		frame = np.asarray(self.env.render(width=width, height=height))
		if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
			raise ValueError(
				'Cutie same-state render must be uint8 HWC RGB, got '
				f'{frame.shape} {frame.dtype}.'
			)
		return np.array(frame, dtype=np.uint8, order='C', copy=True)

	def reset(self):
		self.env.reset()
		return self._get_obs(is_reset=True)

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		return self._get_obs(), reward, done, info


def make_env(cfg):
	"""
	Make DMControl environment.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
	if (domain, task) not in suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	if bool(cfg.get('video_background_enabled', False)) and cfg.obs != 'rgb':
		raise ValueError('Dynamic video backgrounds require obs=rgb.')
	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 visualize_reward=False)
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	env = DMControlWrapper(env, domain)
	if cfg.get('cutie_object_observation_variant', 'full') == 'gt_articulated_pose':
		env = GTArticulatedPoseWrapper(env, cfg)
	elif cfg.obs == 'rgb':
		env = Pixels(env, cfg)
		erosion_pixels = int(cfg.get('visual_foreground_erosion_pixels', 0))
		if erosion_pixels:
			env = ForegroundErosionWrapper(env, erosion_pixels=erosion_pixels)
		if bool(cfg.get('video_background_enabled', False)):
			env = ColorMultiVideoBackgroundWrapper(env, cfg)
		observation_variant = cfg.get('cutie_object_observation_variant', 'full')
		if cfg.get('robust_object_field_enabled', False):
			try:
				env = RobustObjectFieldObservationWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif cfg.get('cutie_mask_guided_rgb_enabled', False):
			try:
				env = CutieMaskGuidedRGBWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif cfg.get('cutie_masked_rgb_enabled', False):
			try:
				env = CutieMaskedRGBWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif observation_variant == 'cutie_proprio':
			try:
				env = CutieProprioWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif observation_variant == 'multimodal_articulated_pose':
			try:
				env = MultimodalArticulatedPoseWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif observation_variant == 'visual_articulated_pose':
			try:
				env = VisualArticulatedPoseWrapper(env, cfg)
			except Exception:
				env.close()
				raise
		elif bool(cfg.get('flat_anchor', False)):
			if cfg.get('flat_anchor_mode', 'residual') in {
				'cutie_hybrid', 'cutie_object_only'
			}:
				try:
					env = CutieObjectWrapper(env, cfg)
				except Exception:
					# A wrapper constructor that fails cannot be returned to train.py's
					# lifecycle guard. Close the already-created simulator/background.
					env.close()
					raise
			else:
				env = FlatAnchorWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=500)
	return env
