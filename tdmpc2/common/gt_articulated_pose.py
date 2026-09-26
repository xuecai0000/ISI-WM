"""Dependency-light contract for the privileged Acrobot pose diagnostic."""


VARIANT = 'gt_articulated_pose'
TASK = 'acrobot-swingup'
ROLE_NAMES = ('upper_arm', 'lower_arm')
FRAME_SCHEMA = 'acrobot_gt_articulated_pose_v1'
FRAME_FIELDS = (
	'start_x', 'start_z', 'end_x', 'end_z',
	'sin_theta', 'cos_theta', 'absolute_frame_omega',
)
NUM_ROLES = 2
FRAME_DIM = 7
STACK_FRAMES = 3
INPUT_DIM = FRAME_DIM * STACK_FRAMES
ROLE_DIM = 64
LATENT_DIM = NUM_ROLES * ROLE_DIM
TOTAL_LINK_LENGTH = 2.0
SOURCE = (
	'dm_control.physics.named.data.xpos[upper_arm,lower_arm]+'
	'named.data.site_xpos[tip]'
)


def _value(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg) -> bool:
	return str(_value(cfg, 'cutie_object_observation_variant', 'full')) == VARIANT


def validate_config(cfg):
	"""Fail closed unless the complete privileged diagnostic is explicit."""
	expected = {
		'task': TASK,
		'obs': 'state',
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': True,
		'cutie_object_frame_schema': FRAME_SCHEMA,
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': FRAME_DIM,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': INPUT_DIM,
		'cutie_object_role_dim': ROLE_DIM,
		'cutie_object_only_latent_dim': LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_native_highres_enabled': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
		'video_background_enabled': False,
		'visual_foreground_erosion_pixels': 0,
		'cutie_object_allow_simulator_support': False,
		'cutie_object_allow_simulator_runtime': False,
	}
	bad = {
		key: (_value(cfg, key), value)
		for key, value in expected.items()
		if _value(cfg, key) != value
	}
	roles = _value(cfg, 'cutie_object_role_names')
	if isinstance(roles, str) or tuple(roles or ()) != ROLE_NAMES:
		bad['cutie_object_role_names'] = (roles, ROLE_NAMES)
	for key in (
		'cutie_object_repo', 'cutie_object_checkpoint',
		'cutie_object_support_path', 'cutie_object_config_dir',
		'cutie_object_policy_burst_plan',
	):
		if _value(cfg, key) is not None:
			bad[key] = (_value(cfg, key), None)
	if bool(_value(cfg, 'multitask', False)):
		bad['multitask'] = (_value(cfg, 'multitask'), False)
	if _value(cfg, 'model_size', 5) != 5:
		bad['model_size'] = (_value(cfg, 'model_size'), 5)
	if bad:
		raise ValueError(
			'gt_articulated_pose privileged diagnostic config mismatch: '
			f'{bad}.'
		)
	return observation_contract(cfg)


def observation_contract(cfg=None):
	"""Return immutable semantics recorded in every compatible checkpoint."""
	return {
		'format': 'gt_articulated_pose_observation_contract_v1',
		'variant': VARIANT,
		'privileged_simulator_kinematics': True,
		'privileged_simulator_kinematics_runtime': True,
		'task': TASK,
		'role_names': list(ROLE_NAMES),
		'frame_schema': FRAME_SCHEMA,
		'frame_fields': list(FRAME_FIELDS),
		'source': SOURCE,
		'diagnostic_only': True,
		'position_normalization_total_link_length': TOTAL_LINK_LENGTH,
		'angular_velocity': 'signed_wrapped_absolute_frame_delta_over_sim_time',
		'num_roles': NUM_ROLES,
		'frame_dim': FRAME_DIM,
		'stack_frames': STACK_FRAMES,
		'input_dim': INPUT_DIM,
	}


def auxiliary_contract(beta=1.0):
	return {
		'format': 'gt_articulated_pose_auxiliary_contract_v1',
		'target': 'articulated_pose',
		'loss': 'smooth_l1',
		'reduction': 'mean',
		'beta': float(beta),
		'finite_values_required': True,
		'applies_to': ['current_reconstruction', 'future_prediction'],
		'decoder_output_dim': INPUT_DIM,
	}
