"""Contracts for the pure-visual Mask-Guided TD-MPC2 input path.

The policy receives the unchanged causal RGB stack together with native Cutie
role masks and tracker status from the same synchronous response.  Masks guide
the spatial RGB encoder through a bounded, zero-initialized residual; they do
not replace RGB and no simulator state or segmentation is available online.
"""

from __future__ import annotations

from typing import Mapping


SCHEMA = 'cutie_mask_guided_spatial_residual_v1'
IMAGE_SIZE = 64
STACK_FRAMES = 3
RGB_CHANNELS = 3 * STACK_FRAMES
STATUS_DIM = 4
MAX_ROLES = 8
ABLATION_MODES = ('none', 'guidance_off', 'spatial_permute_v1')
SPATIAL_PERMUTATION_SEED = 2718281


def _get(cfg, key, default=None):
	try:
		return cfg.get(key, default)
	except (AttributeError, KeyError):
		return getattr(cfg, key, default)


def enabled(cfg) -> bool:
	return bool(_get(cfg, 'cutie_mask_guided_rgb_enabled', False))


def role_count(cfg) -> int:
	return int(_get(cfg, 'cutie_object_num_roles', 0))


def observation_shapes(cfg) -> dict[str, tuple[int, ...]]:
	roles = role_count(cfg)
	return {
		'rgb': (RGB_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
		'object_mask': (roles, STACK_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
		'tracker_status': (roles, STACK_FRAMES, STATUS_DIM),
	}


def validate_config(cfg, *, require_obs_shape=True) -> None:
	"""Fail closed on the scientific and tensor-shape contract."""
	if not enabled(cfg):
		raise ValueError(
			'Mask-guided RGB validation requires '
			'cutie_mask_guided_rgb_enabled=true.'
		)
	expected = {
		'cutie_mask_guided_rgb_schema': SCHEMA,
		'flat_anchor': False,
		'obs': 'rgb',
		'multitask': False,
		'model_size': 5,
		'latent_dim': 512,
		'cutie_masked_rgb_enabled': False,
		'robust_object_field_enabled': False,
		'cutie_object_observation_variant': 'full',
		'cutie_object_frame_schema': 'cutie_query_mask_status_v1',
		'cutie_object_frame_dim': 590,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': 1770,
		'cutie_mask_guided_rgb_hidden_channels': 32,
		'cutie_mask_guided_rgb_random_shift_pad': 3,
		'cutie_mask_guided_rgb_role_pool': 'max',
		'cutie_mask_guided_rgb_correction_limit': 0.25,
	}
	bad = {
		key: (_get(cfg, key, None), value)
		for key, value in expected.items()
		if _get(cfg, key, None) != value
	}
	ablation_mode = _get(cfg, 'cutie_mask_guided_rgb_ablation_mode', 'none')
	if ablation_mode not in ABLATION_MODES:
		bad['cutie_mask_guided_rgb_ablation_mode'] = (
			ablation_mode, ABLATION_MODES,
		)
	for key in (
		'cutie_object_allow_simulator_runtime',
		'cutie_object_allow_simulator_kinematics_runtime',
		'object_state_supervision_enabled',
		'object_state_supervision_collect_labels',
		'object_state_bottleneck_enabled',
		'cutie_object_native_highres_enabled',
		'cutie_object_spatial_token_enabled',
		'cutie_object_variable_graph_enabled',
		'cutie_object_true_entity_enabled',
		'cutie_object_last_valid_memory',
		'cutie_object_belief_enabled',
	):
		if bool(_get(cfg, key, False)):
			bad[key] = (_get(cfg, key), False)
	if _get(cfg, 'cutie_object_policy_burst_plan', None) is not None:
		bad['cutie_object_policy_burst_plan'] = (
			_get(cfg, 'cutie_object_policy_burst_plan'), None,
		)
	if _get(cfg, 'cutie_object_regression_encoder', None) is not None:
		bad['cutie_object_regression_encoder'] = (
			_get(cfg, 'cutie_object_regression_encoder'), None,
		)
	roles = role_count(cfg)
	role_names = tuple(_get(cfg, 'cutie_object_role_names', ()))
	if not 1 <= roles <= MAX_ROLES:
		bad['cutie_object_num_roles'] = (roles, f'an integer in [1,{MAX_ROLES}]')
	if (
		len(role_names) != roles
		or len(set(role_names)) != roles
		or any(not isinstance(name, str) or not name.strip() for name in role_names)
	):
		bad['cutie_object_role_names'] = (
			role_names, f'{roles} unique non-empty task-static roles',
		)
	if require_obs_shape:
		actual = {
			key: tuple(value)
			for key, value in dict(_get(cfg, 'obs_shape', {})).items()
		}
		expected_shapes = observation_shapes(cfg)
		if actual != expected_shapes:
			bad['obs_shape'] = (actual, expected_shapes)
	if bad:
		raise ValueError(f'Mask-guided RGB contract mismatch: {bad}.')


def contract(cfg) -> Mapping[str, object]:
	validate_config(cfg)
	ablation_mode = str(_get(
		cfg, 'cutie_mask_guided_rgb_ablation_mode', 'none'
	))
	return {
		'format': SCHEMA,
		'task': str(_get(cfg, 'task', '')),
		'ordered_roles': list(_get(cfg, 'cutie_object_role_names', ())),
		'num_roles': role_count(cfg),
		'role_axis': 'task_static_exact_k_no_padding',
		'policy_inputs': ['unchanged_rgb', 'native_cutie_masks', 'tracker_status'],
		'tracker_status_order': [
			'confidence', 'lost', 'tracker_valid', 'mask_score',
		],
		'policy_cutie_descriptors': False,
		'simulator_runtime_inputs': False,
		'shared_per_mask_encoder': True,
		'all_tracker_status_fields_consumed': True,
		'role_pool': 'max',
		'fusion': 'bounded_spatial_film_residual_before_official_flatten',
		'zero_initialized_exact_rgb_start': True,
		'ablation_mode': ablation_mode,
		'guidance_active': ablation_mode != 'guidance_off',
		'mask_registration': (
			f'fixed_pixel_permutation_seed_{SPATIAL_PERMUTATION_SEED}'
			if ablation_mode == 'spatial_permute_v1'
			else 'native_jointly_shifted'
		),
		'correction_limit': float(_get(
			cfg, 'cutie_mask_guided_rgb_correction_limit', 0.25
		)),
	}
