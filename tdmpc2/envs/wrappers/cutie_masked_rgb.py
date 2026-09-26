"""Same frozen Cutie sensor, ordinary RGB policy, black non-object pixels.

No object descriptors reach the policy. Each frame is masked exactly once by
its own current prediction before entering the causal three-frame pixel stack.
The existing Cutie worker still computes its descriptors to preserve the B/C
sensor contract; those values are used only by its existing runtime diagnostics.
"""

from collections import deque
from time import perf_counter

import numpy as np
import torch

from .cutie_object import (
	CutieObjectWrapper, CutieObjectWorkerError, IMAGE_SIZE, STACK_FRAMES,
	_native_hwc_rgb,
)


SCHEMA = 'cutie_union_black_rgb_v1'


def validate_masked_rgb_config(cfg):
	"""Fail closed: this is not a hybrid, GT-mask, or privileged-state path."""
	expected = {
		'cutie_masked_rgb_enabled': True,
		'cutie_masked_rgb_schema': SCHEMA,
		'flat_anchor': False,
		'obs': 'rgb',
		'model_size': 5,
		'multitask': False,
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
	}
	bad = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
	for key in (
		'cutie_object_allow_simulator_runtime',
		'cutie_object_allow_simulator_kinematics_runtime',
		'object_state_supervision_enabled', 'object_state_supervision_collect_labels',
		'object_state_bottleneck_enabled',
		'cutie_object_native_highres_enabled', 'cutie_object_spatial_token_enabled',
		'cutie_object_variable_graph_enabled', 'cutie_object_true_entity_enabled',
		'cutie_object_last_valid_memory', 'cutie_object_belief_enabled',
	):
		if cfg.get(key, False):
			bad[key] = (cfg.get(key), False)
	if cfg.get('cutie_object_policy_burst_plan') is not None:
		bad['cutie_object_policy_burst_plan'] = (cfg.get('cutie_object_policy_burst_plan'), None)
	if cfg.get('cutie_object_regression_encoder') is not None:
		bad['cutie_object_regression_encoder'] = (cfg.get('cutie_object_regression_encoder'), None)
	if cfg.get('latent_dim', 512) != 512:
		bad['latent_dim'] = (cfg.get('latent_dim'), 512)
	if bad:
		raise ValueError(f'Cutie masked RGB contract mismatch: {bad}.')


def mask_native_rgb(rgb, masks, num_roles):
	"""Union exact native binary masks, with no pooling, crop, or color changes."""
	frame = np.asarray(rgb)
	roles = np.asarray(masks)
	if frame.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or frame.dtype != np.uint8:
		raise CutieObjectWorkerError('Masked RGB requires native uint8 HWC RGB.')
	if roles.shape != (num_roles, IMAGE_SIZE, IMAGE_SIZE) or roles.dtype != np.bool_:
		raise CutieObjectWorkerError('Masked RGB requires same-response native binary role masks.')
	union = roles.any(axis=0)
	return np.ascontiguousarray(np.where(union[..., None], frame, 0).transpose(2, 0, 1))


class CutieMaskedRGBWrapper(CutieObjectWrapper):
	"""Cutie observation ablation exporting a plain uint8 [9,64,64] tensor."""

	def __init__(self, env, cfg, *, _client=None):
		validate_masked_rgb_config(cfg)
		# The parent wrapper's hybrid internals are private: the actual cfg retains
		# flat_anchor=False, and its RGB/object dictionary never leaves this class.
		internal = dict(cfg) if isinstance(cfg, dict) else dict(vars(cfg))
		internal['flat_anchor_mode'] = 'cutie_hybrid'
		self._masked_frames = deque(maxlen=STACK_FRAMES)
		self._masking_ms = 0.0
		self._masked_frame_count = 0
		super().__init__(env, internal, _client=_client)
		self.observation_space = env.observation_space

	def _observation(self, rgb):
		started = perf_counter()
		frame = mask_native_rgb(
			_native_hwc_rgb(rgb), getattr(self._client, 'last_masks', None), len(self._role_names)
		)
		if self._metric_episode_step == 0:
			self._masked_frames.clear()
			for _ in range(STACK_FRAMES):
				self._masked_frames.append(frame.copy())
		else:
			if len(self._masked_frames) != STACK_FRAMES:
				raise CutieObjectWorkerError('Masked RGB temporal history is not initialized.')
			self._masked_frames.append(frame)
		observation = torch.from_numpy(np.concatenate(tuple(self._masked_frames), axis=0))
		self._masking_ms += (perf_counter() - started) * 1000.0
		self._masked_frame_count += 1
		return observation

	@property
	def cutie_ready(self):
		ready = super().cutie_ready
		if ready is None:
			return None
		return {
			**ready, 'policy_observation': SCHEMA,
			'agent_observation_unchanged': False, 'treatment': SCHEMA,
			'policy_object_descriptors': False, 'policy_original_rgb_background': False,
			'mask_source': 'same_response_native64_role_masks',
			'mask_union': 'all_predicted_role_masks_no_extra_validity_filter',
			'mask_stack': 'each_frame_own_causal_mask_reset_repeat_three',
		}

	def metrics(self):
		return {
			**super().metrics(),
			'masked_rgb_schema': SCHEMA,
			'masked_rgb_frames': self._masked_frame_count,
			'masked_rgb_masking_mean_ms': self._masking_ms / max(1, self._masked_frame_count),
		}
