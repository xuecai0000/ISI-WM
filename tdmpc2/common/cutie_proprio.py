"""Contract for Cutie object tokens with optional Acrobot proprioception."""


VARIANT = 'cutie_proprio'
TASK = 'acrobot-swingup'
ROLE_NAMES = ('upper_arm', 'lower_arm')
FRAME_SCHEMA = 'cutie_query_mask_status_plus_proprio_v1'
MODES = (
	'cutie_only', 'proprio_only', 'fusion', 'factorized',
	'factorized_proprio_only',
)
TASK_ROLE_NAMES = {
	'acrobot-swingup': ('upper_arm', 'lower_arm'),
	'cartpole-swingup': ('cart', 'pole'),
	'reacher-visual-small': ('whole_arm', 'goal'),
}
VISUAL_DIM = 1770
PROPRIO_DIM = 4
INPUT_DIM = VISUAL_DIM + PROPRIO_DIM
NUM_ROLES = 2
ROLE_DIM = 64
LATENT_DIM = 128
PROPRIO_FIELDS = (
	'sin_absolute_joint_angle', 'cos_absolute_joint_angle',
	'tanh_absolute_joint_velocity', 'proprio_available',
)


def _value(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg):
	return str(_value(cfg, 'cutie_object_observation_variant', 'full')) == VARIANT


def mode(cfg):
	return str(_value(cfg, 'cutie_proprio_mode', 'fusion'))


def uses_cutie(cfg):
	return mode(cfg) in {'cutie_only', 'fusion', 'factorized'}


def uses_proprio(cfg):
	return mode(cfg) in {
		'proprio_only', 'fusion', 'factorized', 'factorized_proprio_only',
	}


def uses_factorized_model(cfg):
	return mode(cfg) in {'factorized', 'factorized_proprio_only'}


def validate_config(cfg):
	selected_mode = mode(cfg)
	if selected_mode not in MODES:
		raise ValueError(f'cutie_proprio_mode must be one of {MODES}, got {selected_mode!r}.')
	task = str(_value(cfg, 'task', ''))
	if selected_mode in {'factorized', 'factorized_proprio_only'}:
		expected_roles = TASK_ROLE_NAMES.get(task)
		if expected_roles is None:
			raise ValueError(
				f'factorized Cutie-proprio has no adapter for task {task!r}.'
			)
	else:
		expected_roles = ROLE_NAMES
		if task != TASK:
			raise ValueError(
				f'{selected_mode} Cutie-proprio is restricted to {TASK!r}.'
			)
	expected = {
		'task': task,
		'obs': 'rgb',
		'multitask': False,
		'model_size': 5,
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': VARIANT,
		'cutie_object_frame_schema': FRAME_SCHEMA,
		'cutie_object_role_names': list(expected_roles),
		'cutie_object_support_schema': 'generic_indexed_v1',
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': INPUT_DIM,
		'cutie_object_stack_frames': 1,
		'cutie_object_input_dim': INPUT_DIM,
		'cutie_object_role_dim': ROLE_DIM,
		'cutie_object_only_latent_dim': LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_allow_simulator_kinematics_runtime': uses_proprio(cfg),
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_last_valid_memory': False,
		'cutie_object_belief_enabled': False,
		'cutie_object_belief_use_for_control': False,
	}
	bad = {
		key: (_value(cfg, key), value)
		for key, value in expected.items()
		if _value(cfg, key) != value
	}
	if float(_value(cfg, 'cutie_proprio_velocity_scale', 10.0)) <= 0:
		bad['cutie_proprio_velocity_scale'] = (
			_value(cfg, 'cutie_proprio_velocity_scale'), '>0',
		)
	for key in (
		'cutie_object_repo', 'cutie_object_checkpoint',
		'cutie_object_support_path',
	):
		value = _value(cfg, key)
		if not isinstance(value, str) or not value.strip():
			bad[key] = (value, 'non-empty path')
	if bad:
		raise ValueError(f'cutie_proprio config mismatch: {bad}.')
	return observation_contract(cfg)


def observation_contract(cfg=None):
	selected_mode = mode(cfg) if cfg is not None else 'fusion'
	task = str(_value(cfg, 'task', TASK)) if cfg is not None else TASK
	roles = TASK_ROLE_NAMES.get(task, ROLE_NAMES)
	return {
		'format': 'cutie_proprio_observation_contract_v1',
		'variant': VARIANT,
		'mode': selected_mode,
		'task': task,
		'role_names': list(roles),
		'frame_schema': FRAME_SCHEMA,
		'visual_source': 'live_causal_cutie_full_descriptor_v1',
		'proprio_source': 'dm_control.physics.data.qpos+qvel',
		'visual_fields_zeroed': selected_mode in {
			'proprio_only', 'factorized_proprio_only',
		},
		'proprio_fields_zeroed': selected_mode == 'cutie_only',
		'privileged_simulator_kinematics_runtime': (
			uses_proprio(cfg) if cfg is not None else True
		),
		'visual_dim': VISUAL_DIM,
		'proprio_dim': PROPRIO_DIM,
		'proprio_fields': list(PROPRIO_FIELDS),
		'input_dim': INPUT_DIM,
		'num_roles': NUM_ROLES,
		'latent_dim': LATENT_DIM,
		'safe_fusion': {
			'formula': (
				'concat(z_body_64,z_object_64); isolated body transition'
				if selected_mode in {'factorized', 'factorized_proprio_only'}
				else 'z_proprio+sigmoid(gate)*visual_delta'
			),
			'visual_delta_output_initialized_to_exact_zero': (
				selected_mode not in {'factorized', 'factorized_proprio_only'}
			),
			'gate_logit_initial_value': (
				None
				if selected_mode in {'factorized', 'factorized_proprio_only'}
				else -4.0
			),
		},
		'proprio_velocity_scale': float(
			_value(cfg, 'cutie_proprio_velocity_scale', 10.0)
		) if cfg is not None else 10.0,
	}


def auxiliary_contract(beta=0.1):
	return {
		'format': 'cutie_proprio_auxiliary_contract_v1',
		'loss': 'balanced_visual_and_proprio_smooth_l1',
		'beta': float(beta),
		'visual_loss': 'full_cutie_validity_weighted_role_mean',
		'proprio_loss': 'unweighted_mean',
		'mode_selective': True,
		'decoder_output_dim': INPUT_DIM,
		'applies_to': ['current_reconstruction', 'future_prediction'],
	}
