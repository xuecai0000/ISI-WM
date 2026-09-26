"""Environment observation boundary for Robust Object Field V0.

This opt-in wrapper is a sibling of the established Cutie object and
masked-RGB baselines. It reuses the same isolated, synchronous Cutie worker and
does not alter either legacy observation path.
"""

from __future__ import annotations

from collections import deque
import hashlib
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from common import robust_object_field as model_contract
from common.robust_object_field_observation import (
	IMAGE_SIZE,
	OBJECT_DIM,
	RGB_CHANNELS,
	SCHEMA,
	STACK_FRAMES,
	build_packet,
	validate_numpy_observation,
)
from .cutie_object import (
	CutieObjectWrapper,
	CutieObjectWorkerError,
	TASK_ROLE_NAMES,
	_native_hwc_rgb,
	task_role_names,
)


def _get(cfg, key, default=None):
	try:
		return cfg.get(key, default)
	except (AttributeError, KeyError):
		return getattr(cfg, key, default)


def _as_numpy(value) -> np.ndarray:
	if torch.is_tensor(value):
		value = value.detach().cpu().numpy()
	return np.asarray(value)


def validate_robust_object_field_observation_config(
	cfg,
) -> tuple[str, tuple[str, ...]]:
	"""Freeze a canonical single-task exact-K sensor before worker startup."""
	model_contract.validate_config(cfg, require_obs_shape=False)
	task = _get(cfg, 'task')
	if task not in TASK_ROLE_NAMES:
		raise ValueError(
			f'ROF V0 task must be one of {tuple(TASK_ROLE_NAMES)!r}, got {task!r}.'
		)
	role_contract = _get(cfg, 'cutie_object_task_role_contract', 'canonical_v1')
	if role_contract != 'canonical_v1':
		raise ValueError('ROF V0 requires canonical task-static role semantics.')
	expected_roles = tuple(task_role_names(task, role_contract))
	configured_roles = tuple(_get(cfg, 'cutie_object_role_names', ()))
	if configured_roles != expected_roles:
		raise ValueError(
			f'ROF V0 {task!r} requires exact ordered roles {expected_roles!r}, '
			f'got {configured_roles!r}.'
		)
	if int(_get(cfg, 'cutie_object_num_roles', 0)) != len(expected_roles):
		raise ValueError(
			'ROF V0 cutie_object_num_roles must equal the task-static role count.'
		)
	bad = {}
	for key in (
		'cutie_object_native_highres_enabled',
		'cutie_object_true_entity_enabled',
	):
		if bool(_get(cfg, key, False)):
			bad[key] = (_get(cfg, key), False)
	if int(_get(cfg, 'visual_foreground_erosion_pixels', 0)) != 0:
		bad['visual_foreground_erosion_pixels'] = (
			_get(cfg, 'visual_foreground_erosion_pixels'), 0
		)
	if _get(cfg, 'visual_pose_checkpoint', None) is not None:
		bad['visual_pose_checkpoint'] = (
			_get(cfg, 'visual_pose_checkpoint'), None
		)
	if str(_get(cfg, 'robust_object_field_schema', SCHEMA)) != SCHEMA:
		bad['robust_object_field_schema'] = (
			_get(cfg, 'robust_object_field_schema'), SCHEMA
		)
	if bad:
		raise ValueError(f'ROF V0 observation contract mismatch: {bad}.')
	return str(task), expected_roles


class RobustObjectFieldObservationWrapper(CutieObjectWrapper):
	"""Export the exact public ROF V0 observation dictionary."""

	def __init__(self, env, cfg, *, _client=None):
		task, roles = validate_robust_object_field_observation_config(cfg)
		self._field_task = task
		self._field_roles = roles
		self._field_mask_frames = deque(maxlen=STACK_FRAMES)
		self._field_sequence_id = -1
		self._field_frame_count = 0
		self._field_last_packet = None
		super().__init__(env, cfg, _client=_client)
		if tuple(self._role_names) != roles:
			self.close()
			raise CutieObjectWorkerError(
				'ROF V0 base wrapper changed the task-static role axis.'
			)
		count = len(roles)
		self.observation_space = gym.spaces.Dict({
			'rgb': gym.spaces.Box(
				low=0, high=255,
				shape=(RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8,
			),
			'object': gym.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(count, OBJECT_DIM), dtype=np.float32,
			),
			'object_mask': gym.spaces.Box(
				# Gymnasium 0.29 accepts boolean Box dtypes but its scalar-bound
				# broadcaster rejects Python bool values. Integer 0/1 bounds retain
				# the exact boolean observation contract after dtype conversion.
				low=0, high=1,
				shape=(count, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
				dtype=np.bool_,
			),
			'role_exists': gym.spaces.Box(
				low=1.0, high=1.0, shape=(count,), dtype=np.float32,
			),
		})

	def _observation(self, rgb):
		"""Atomically bind each causal RGB/mask/descriptor stack."""
		if len(self._object_frames) != STACK_FRAMES:
			raise CutieObjectWorkerError(
				'ROF V0 descriptor history is not initialized.'
			)
		rgb_stack = _as_numpy(rgb)
		current_hwc = _native_hwc_rgb(rgb)
		expected_source_hash = hashlib.sha256(
			current_hwc.tobytes(order='C')
		).hexdigest()
		if self.latest_source_rgb_sha256 != expected_source_hash:
			raise CutieObjectWorkerError(
				'ROF V0 RGB is not the source of the current Cutie response.'
			)
		current_masks = getattr(self._client, 'last_masks', None)
		diagnostics = getattr(self._client, 'last_diagnostics', None)
		if current_masks is None or diagnostics is None:
			raise CutieObjectWorkerError(
				'ROF V0 requires same-response native masks and diagnostics.'
			)
		current_masks = np.asarray(current_masks)
		if self._metric_episode_step == 0:
			self._field_mask_frames.clear()
			for _ in range(STACK_FRAMES):
				self._field_mask_frames.append(current_masks.copy())
		else:
			if len(self._field_mask_frames) != STACK_FRAMES:
				raise CutieObjectWorkerError(
					'ROF V0 mask history is not initialized.'
				)
			self._field_mask_frames.append(current_masks.copy())

		object_stack = np.concatenate(tuple(self._object_frames), axis=-1)
		object_mask_stack = np.stack(
			tuple(self._field_mask_frames), axis=1
		)
		self._field_sequence_id += 1
		try:
			packet = build_packet(
				task=self._field_task,
				role_names=self._field_roles,
				rgb_stack=rgb_stack,
				object_stack=object_stack,
				object_mask_stack=object_mask_stack,
				diagnostics=diagnostics,
				sequence_id=self._field_sequence_id,
			)
			observation = packet.as_numpy_observation()
			validate_numpy_observation(
				observation, role_count=len(self._field_roles)
			)
		except (TypeError, ValueError, AssertionError) as exc:
			raise CutieObjectWorkerError(
				f'ROF V0 atomic observation validation failed: {exc}'
			) from exc
		if packet.source_rgb_sha256 != expected_source_hash:
			raise CutieObjectWorkerError(
				'ROF V0 packet source hash changed during assembly.'
			)
		self._field_last_packet = packet
		self._field_frame_count += 1
		return {
			key: torch.from_numpy(np.ascontiguousarray(value))
			for key, value in observation.items()
		}

	@property
	def robust_object_field_role_names(self) -> tuple[str, ...]:
		return self._field_roles

	@property
	def robust_object_field_last_binding_sha256(self) -> str | None:
		return (
			None if self._field_last_packet is None
			else self._field_last_packet.binding_sha256
		)

	@property
	def cutie_ready(self) -> dict[str, Any] | None:
		ready = super().cutie_ready
		if ready is None:
			return None
		return {
			**ready,
			'agent_observation_unchanged': False,
			'treatment': SCHEMA,
			'policy_observation': SCHEMA,
			'policy_rgb': True,
			'policy_cutie_descriptors': True,
			'policy_native_role_masks': True,
			'role_names': list(self._field_roles),
			'role_count': len(self._field_roles),
			'role_axis': 'task_static_exact_k_no_padding',
			'role_exists': 'all_one',
			'rgb_descriptor_mask_binding': (
				'same_synchronous_cutie_response_native64_three_frame_causal_v1'
			),
			'privileged_runtime_segmentation': False,
			'privileged_runtime_kinematics': False,
		}

	@property
	def robust_object_field_ready(self) -> dict[str, Any] | None:
		return self.cutie_ready

	def metrics(self) -> dict[str, Any]:
		packet = self._field_last_packet
		return {
			**super().metrics(),
			'robust_object_field_schema': SCHEMA,
			'robust_object_field_frames': self._field_frame_count,
			'robust_object_field_role_count': len(self._field_roles),
			'robust_object_field_last_sequence_id': (
				None if packet is None else packet.sequence_id
			),
			'robust_object_field_last_source_rgb_sha256': (
				None if packet is None else packet.source_rgb_sha256
			),
			'robust_object_field_last_object_sha256': (
				None if packet is None else packet.object_sha256
			),
			'robust_object_field_last_mask_sha256': (
				None if packet is None else packet.object_mask_sha256
			),
			'robust_object_field_last_binding_sha256': (
				None if packet is None else packet.binding_sha256
			),
		}


__all__ = [
	'RobustObjectFieldObservationWrapper',
	'SCHEMA',
	'validate_robust_object_field_observation_config',
]
