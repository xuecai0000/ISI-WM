"""Contracts for the deployable Acrobot visual-pose observation."""

import hashlib
from pathlib import Path

from common import gt_articulated_pose


VARIANT = 'visual_articulated_pose'
TASK = gt_articulated_pose.TASK
ROLE_NAMES = gt_articulated_pose.ROLE_NAMES
FRAME_SCHEMA = 'acrobot_visual_articulated_pose_v1'
FRAME_FIELDS = gt_articulated_pose.FRAME_FIELDS
NUM_ROLES = gt_articulated_pose.NUM_ROLES
FRAME_DIM = gt_articulated_pose.FRAME_DIM
STACK_FRAMES = gt_articulated_pose.STACK_FRAMES
INPUT_DIM = gt_articulated_pose.INPUT_DIM
ROLE_DIM = gt_articulated_pose.ROLE_DIM
LATENT_DIM = gt_articulated_pose.LATENT_DIM
TOTAL_LINK_LENGTH = gt_articulated_pose.TOTAL_LINK_LENGTH
POINT_NAMES = ('base', 'elbow', 'tip')


def _value(cfg, key, default=None):
	return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg) -> bool:
	return str(_value(cfg, 'cutie_object_observation_variant', 'full')) == VARIANT


def validate_config(cfg):
	"""Fail closed when a visual-pose run could silently become privileged."""
	expected = {
		'task': TASK,
		'obs': 'rgb',
		'flat_anchor': True,
		'flat_anchor_mode': 'cutie_object_only',
		'cutie_object_observation_variant': VARIANT,
		'cutie_object_allow_simulator_kinematics_runtime': False,
		'cutie_object_frame_schema': FRAME_SCHEMA,
		'cutie_object_num_roles': NUM_ROLES,
		'cutie_object_frame_dim': FRAME_DIM,
		'cutie_object_stack_frames': STACK_FRAMES,
		'cutie_object_input_dim': INPUT_DIM,
		'cutie_object_role_dim': ROLE_DIM,
		'cutie_object_only_latent_dim': LATENT_DIM,
		'cutie_object_auxiliary_target': 'full_descriptor',
	}
	bad = {
		key: (_value(cfg, key), value)
		for key, value in expected.items()
		if _value(cfg, key) != value
	}
	roles = _value(cfg, 'cutie_object_role_names')
	if isinstance(roles, str) or tuple(roles or ()) != ROLE_NAMES:
		bad['cutie_object_role_names'] = (roles, ROLE_NAMES)
	checkpoint = _value(cfg, 'visual_pose_checkpoint')
	if not isinstance(checkpoint, str) or not checkpoint.strip():
		bad['visual_pose_checkpoint'] = (checkpoint, 'non-empty path')
	if int(_value(cfg, 'visual_pose_history', 4)) < STACK_FRAMES:
		bad['visual_pose_history'] = (
			_value(cfg, 'visual_pose_history'), f'>={STACK_FRAMES}',
		)
	if float(_value(cfg, 'visual_pose_control_dt', 0.04)) <= 0:
		bad['visual_pose_control_dt'] = (
			_value(cfg, 'visual_pose_control_dt'), '>0',
		)
	if bool(_value(cfg, 'multitask', False)):
		bad['multitask'] = (_value(cfg, 'multitask'), False)
	if bad:
		raise ValueError(f'visual_articulated_pose config mismatch: {bad}.')
	return observation_contract(cfg)


def observation_contract(cfg=None):
	checkpoint = _value(cfg, 'visual_pose_checkpoint') if cfg is not None else None
	checkpoint_sha256 = None
	if checkpoint:
		path = Path(checkpoint).expanduser()
		if not path.is_file():
			raise ValueError(f'Visual pose checkpoint does not exist: {path}.')
		digest = hashlib.sha256()
		with path.open('rb') as stream:
			for chunk in iter(lambda: stream.read(1024 * 1024), b''):
				digest.update(chunk)
		checkpoint_sha256 = digest.hexdigest()
	return {
		'format': 'visual_articulated_pose_observation_contract_v1',
		'variant': VARIANT,
		'privileged_simulator_kinematics': False,
		'privileged_simulator_kinematics_runtime': False,
		'task': TASK,
		'role_names': list(ROLE_NAMES),
		'point_names': list(POINT_NAMES),
		'frame_schema': FRAME_SCHEMA,
		'frame_fields': list(FRAME_FIELDS),
		'source': 'causal_rgb_history+past_actions+optional_deployable_foreground_mask',
		'position_normalization_total_link_length': TOTAL_LINK_LENGTH,
		'angular_velocity': 'causal_network_prediction_of_signed_absolute_link_velocity',
		'num_roles': NUM_ROLES,
		'frame_dim': FRAME_DIM,
		'stack_frames': STACK_FRAMES,
		'input_dim': INPUT_DIM,
		'history': int(_value(cfg, 'visual_pose_history', 4)) if cfg is not None else 4,
		'uses_cutie_mask': bool(_value(cfg, 'visual_pose_use_cutie_mask', False))
			if cfg is not None else False,
		'detector_checkpoint_sha256': checkpoint_sha256,
	}


def auxiliary_contract(beta=1.0):
	contract = gt_articulated_pose.auxiliary_contract(beta)
	contract['format'] = 'visual_articulated_pose_auxiliary_contract_v1'
	contract['target'] = 'predicted_articulated_pose'
	return contract
