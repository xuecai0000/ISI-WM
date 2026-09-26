"""Contract for fair visual/proprioceptive Acrobot input ablations.

All three modes expose the same ``[2, 25]`` tensor and therefore use the same
128-D role encoder and controller.  Information is removed by zeroing fields,
not by changing the model architecture.
"""

import hashlib
from pathlib import Path

from common import gt_articulated_pose


VARIANT = 'multimodal_articulated_pose'
TASK = gt_articulated_pose.TASK
ROLE_NAMES = gt_articulated_pose.ROLE_NAMES
FRAME_SCHEMA = 'acrobot_multimodal_articulated_pose_v1'
VISUAL_FIELDS = tuple(
	f'visual_t{age}_{field}'
	for age in (2, 1, 0)
	for field in gt_articulated_pose.FRAME_FIELDS
)
PROPRIO_FIELDS = (
	'sin_absolute_joint_angle',
	'cos_absolute_joint_angle',
	'tanh_absolute_joint_velocity',
	'proprio_available',
)
FRAME_FIELDS = VISUAL_FIELDS + PROPRIO_FIELDS
NUM_ROLES = gt_articulated_pose.NUM_ROLES
VISUAL_DIM = gt_articulated_pose.INPUT_DIM
PROPRIO_DIM = len(PROPRIO_FIELDS)
INPUT_DIM = VISUAL_DIM + PROPRIO_DIM
ROLE_DIM = gt_articulated_pose.ROLE_DIM
LATENT_DIM = gt_articulated_pose.LATENT_DIM
MODES = ('visual_only', 'proprio_only', 'fusion')


def _value(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg) -> bool:
	return str(_value(cfg, 'cutie_object_observation_variant', 'full')) == VARIANT


def mode(cfg) -> str:
	return str(_value(cfg, 'articulated_pose_modalities', 'fusion'))


def uses_visual(cfg) -> bool:
	return mode(cfg) in {'visual_only', 'fusion'}


def uses_proprio(cfg) -> bool:
	return mode(cfg) in {'proprio_only', 'fusion'}


def _checkpoint_sha256(cfg):
	if not uses_visual(cfg):
		return None
	checkpoint = _value(cfg, 'visual_pose_checkpoint')
	if not isinstance(checkpoint, str) or not checkpoint.strip():
		raise ValueError('Visual/fusion mode requires visual_pose_checkpoint.')
	path = Path(checkpoint).expanduser()
	if not path.is_file():
		raise ValueError(f'Visual pose checkpoint does not exist: {path}.')
	digest = hashlib.sha256()
	with path.open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def validate_config(cfg):
	selected_mode = mode(cfg)
	if selected_mode not in MODES:
		raise ValueError(
			f'articulated_pose_modalities must be one of {MODES}, got {selected_mode!r}.'
		)
	expected = {
		'task': TASK,
		'obs': 'rgb',
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': uses_proprio(cfg),
		'cutie_object_frame_schema': FRAME_SCHEMA,
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': INPUT_DIM,
		'cutie_object_stack_frames': 1,
		'cutie_object_input_dim': INPUT_DIM,
		'cutie_object_role_dim': ROLE_DIM,
		'cutie_object_only_latent_dim': LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
		'cutie_object_allow_simulator_runtime': False,
		'cutie_object_allow_simulator_support': False,
	}
	bad = {
		key: (_value(cfg, key), value)
		for key, value in expected.items()
		if _value(cfg, key) != value
	}
	roles = _value(cfg, 'cutie_object_role_names')
	if isinstance(roles, str) or tuple(roles or ()) != ROLE_NAMES:
		bad['cutie_object_role_names'] = (roles, ROLE_NAMES)
	if bool(_value(cfg, 'multitask', False)):
		bad['multitask'] = (_value(cfg, 'multitask'), False)
	if int(_value(cfg, 'visual_pose_history', 4)) < gt_articulated_pose.STACK_FRAMES:
		bad['visual_pose_history'] = (
			_value(cfg, 'visual_pose_history'), f'>={gt_articulated_pose.STACK_FRAMES}',
		)
	if float(_value(cfg, 'multimodal_proprio_velocity_scale', 10.0)) <= 0:
		bad['multimodal_proprio_velocity_scale'] = (
			_value(cfg, 'multimodal_proprio_velocity_scale'), '>0',
		)
	if uses_visual(cfg):
		_checkpoint_sha256(cfg)
	if bad:
		raise ValueError(f'multimodal_articulated_pose config mismatch: {bad}.')
	return observation_contract(cfg)


def observation_contract(cfg=None):
	selected_mode = mode(cfg) if cfg is not None else 'fusion'
	return {
		'format': 'multimodal_articulated_pose_observation_contract_v1',
		'variant': VARIANT,
		'mode': selected_mode,
		'task': TASK,
		'role_names': list(ROLE_NAMES),
		'frame_schema': FRAME_SCHEMA,
		'frame_fields': list(FRAME_FIELDS),
		'visual_source': 'causal_rgb_history+past_actions',
		'proprio_source': 'dm_control.physics.data.qpos+qvel',
		'privileged_simulator_kinematics_runtime': (
			uses_proprio(cfg) if cfg is not None else True
		),
		'visual_fields_zeroed': selected_mode == 'proprio_only',
		'proprio_fields_zeroed': selected_mode == 'visual_only',
		'visual_dim': VISUAL_DIM,
		'proprio_dim': PROPRIO_DIM,
		'num_roles': NUM_ROLES,
		'input_dim': INPUT_DIM,
		'latent_dim': LATENT_DIM,
		'proprio_velocity_transform': 'tanh(absolute_joint_velocity/scale)',
		'proprio_velocity_scale': float(
			_value(cfg, 'multimodal_proprio_velocity_scale', 10.0)
		) if cfg is not None else 10.0,
		'detector_checkpoint_sha256': (
			_checkpoint_sha256(cfg) if cfg is not None else None
		),
	}


def auxiliary_contract(beta=1.0):
	return {
		'format': 'multimodal_articulated_pose_auxiliary_contract_v1',
		'target': 'selected_visual_proprioceptive_fields',
		'loss': 'smooth_l1',
		'reduction': 'mean',
		'beta': float(beta),
		'finite_values_required': True,
		'applies_to': ['current_reconstruction', 'future_prediction'],
		'decoder_output_dim': INPUT_DIM,
	}
