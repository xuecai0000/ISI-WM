"""Unchanged RGB plus causally aligned native Cutie masks for spatial guidance.

The public observation deliberately excludes the 586-D Cutie appearance and
geometry descriptor.  The policy receives only the ordinary three-frame RGB
stack, the exact native per-role masks from those same frames, and four tracker
status scalars per role/frame.  Simulator segmentation and kinematics are never
queried by this wrapper.
"""

from __future__ import annotations

from collections import deque
import hashlib

import gymnasium as gym
import numpy as np
import torch

from common import mask_guided_rgb as contract
from .cutie_object import (
	CutieObjectWrapper,
	CutieObjectWorkerError,
	FRAME_CONTENT_DIM,
	FRAME_FEATURE_DIM,
	TASK_ROLE_NAMES,
	_native_hwc_rgb,
	task_role_names,
)


def _get(cfg, key, default=None):
	try:
		return cfg.get(key, default)
	except (AttributeError, KeyError):
		return getattr(cfg, key, default)


def validate_mask_guided_rgb_observation_config(cfg):
	"""Freeze the single-task exact-K role and observation contracts."""
	contract.validate_config(cfg, require_obs_shape=False)
	task = str(_get(cfg, 'task', ''))
	if task not in TASK_ROLE_NAMES:
		raise ValueError(
			f'Mask-guided RGB task must be one of {tuple(TASK_ROLE_NAMES)!r}, '
			f'got {task!r}.'
		)
	role_contract = str(_get(
		cfg, 'cutie_object_task_role_contract', 'canonical_v1'
	))
	expected_roles = tuple(task_role_names(task, role_contract))
	configured_roles = tuple(_get(cfg, 'cutie_object_role_names', ()))
	if configured_roles != expected_roles:
		raise ValueError(
			f'Mask-guided RGB {task!r} requires ordered roles '
			f'{expected_roles!r}, got {configured_roles!r}.'
		)
	if int(_get(cfg, 'cutie_object_num_roles', 0)) != len(expected_roles):
		raise ValueError(
			'Mask-guided RGB role count must equal the exact task role count.'
		)
	return task, expected_roles


class CutieMaskGuidedRGBWrapper(CutieObjectWrapper):
	"""Export ``rgb/object_mask/tracker_status`` from one synchronous sensor."""

	def __init__(self, env, cfg, *, _client=None):
		self._guided_task, self._guided_roles = (
			validate_mask_guided_rgb_observation_config(cfg)
		)
		# The parent needs its hybrid internal mode to keep RGB available.  This
		# private copy does not alter the runtime config or enable FlatAnchor.
		internal = dict(cfg) if isinstance(cfg, dict) else dict(vars(cfg))
		internal['flat_anchor_mode'] = 'cutie_hybrid'
		self._guided_mask_frames = deque(maxlen=contract.STACK_FRAMES)
		self._guided_frame_count = 0
		self._guided_last_binding_sha256 = None
		super().__init__(env, internal, _client=_client)
		if tuple(self._role_names) != self._guided_roles:
			self.close()
			raise CutieObjectWorkerError(
				'Mask-guided RGB parent changed the task-static role axis.'
			)
		roles = len(self._guided_roles)
		self.observation_space = gym.spaces.Dict({
			'rgb': env.observation_space,
			'object_mask': gym.spaces.Box(
				low=0, high=1,
				shape=(roles, contract.STACK_FRAMES,
					contract.IMAGE_SIZE, contract.IMAGE_SIZE),
				dtype=np.bool_,
			),
			'tracker_status': gym.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(roles, contract.STACK_FRAMES, contract.STATUS_DIM),
				dtype=np.float32,
			),
		})

	def _observation(self, rgb):
		if len(self._object_frames) != contract.STACK_FRAMES:
			raise CutieObjectWorkerError(
				'Mask-guided descriptor/status history is not initialized.'
			)
		current_hwc = _native_hwc_rgb(rgb)
		source_hash = hashlib.sha256(
			current_hwc.tobytes(order='C')
		).hexdigest()
		if self.latest_source_rgb_sha256 != source_hash:
			raise CutieObjectWorkerError(
				'Mask-guided RGB and Cutie response do not share a source frame.'
			)
		masks = np.asarray(getattr(self._client, 'last_masks', None))
		expected_mask_shape = (
			len(self._guided_roles), contract.IMAGE_SIZE, contract.IMAGE_SIZE
		)
		if masks.shape != expected_mask_shape or masks.dtype != np.bool_:
			raise CutieObjectWorkerError(
				'Mask-guided RGB requires same-response native boolean masks; '
				f'got {masks.shape} {masks.dtype}.'
			)
		if self._metric_episode_step == 0:
			self._guided_mask_frames.clear()
			for _ in range(contract.STACK_FRAMES):
				self._guided_mask_frames.append(masks.copy())
		else:
			if len(self._guided_mask_frames) != contract.STACK_FRAMES:
				raise CutieObjectWorkerError(
					'Mask-guided mask history is not initialized.'
				)
			self._guided_mask_frames.append(masks.copy())

		mask_stack = np.stack(tuple(self._guided_mask_frames), axis=1)
		status_stack = np.stack([
			frame[:, FRAME_CONTENT_DIM:FRAME_FEATURE_DIM]
			for frame in self._object_frames
		], axis=1).astype(np.float32, copy=False)
		if (
			mask_stack.shape != expected_mask_shape[:1] + (
				contract.STACK_FRAMES, contract.IMAGE_SIZE, contract.IMAGE_SIZE
			)
			or status_stack.shape != (
				len(self._guided_roles), contract.STACK_FRAMES, contract.STATUS_DIM
			)
			or not np.isfinite(status_stack).all()
		):
			raise CutieObjectWorkerError(
				'Mask-guided causal mask/status stack failed validation.'
			)
		# Bind the exact values exported to replay without retaining any extra RGB.
		digest = hashlib.sha256()
		digest.update(source_hash.encode('ascii'))
		digest.update(mask_stack.tobytes(order='C'))
		digest.update(status_stack.tobytes(order='C'))
		self._guided_last_binding_sha256 = digest.hexdigest()
		self._guided_frame_count += 1
		return {
			'rgb': rgb,
			'object_mask': torch.from_numpy(np.ascontiguousarray(mask_stack)),
			'tracker_status': torch.from_numpy(np.ascontiguousarray(status_stack)),
		}

	@property
	def cutie_ready(self):
		ready = super().cutie_ready
		if ready is None:
			return None
		return {
			**ready,
			'treatment': contract.SCHEMA,
			'policy_observation': contract.SCHEMA,
			'policy_original_rgb_background': True,
			'policy_native_role_masks': True,
			'policy_tracker_status': True,
			'policy_cutie_descriptors': False,
			'role_names': list(self._guided_roles),
			'role_count': len(self._guided_roles),
			'role_axis': 'task_static_exact_k_no_padding',
			'rgb_mask_status_binding': (
				'same_synchronous_cutie_response_native64_three_frame_causal_v1'
			),
			'privileged_runtime_segmentation': False,
			'privileged_runtime_kinematics': False,
		}

	@property
	def mask_guided_rgb_ready(self):
		return self.cutie_ready

	@property
	def mask_guided_rgb_last_binding_sha256(self):
		return self._guided_last_binding_sha256

	def metrics(self):
		return {
			**super().metrics(),
			'mask_guided_rgb_schema': contract.SCHEMA,
			'mask_guided_rgb_frames': self._guided_frame_count,
			'mask_guided_rgb_role_count': len(self._guided_roles),
			'mask_guided_rgb_last_binding_sha256': (
				self._guided_last_binding_sha256
			),
		}


__all__ = [
	'CutieMaskGuidedRGBWrapper',
	'validate_mask_guided_rgb_observation_config',
]
