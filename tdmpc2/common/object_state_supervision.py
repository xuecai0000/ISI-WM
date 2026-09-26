"""Training-only official-state targets; never policy observation fields.

Schemas/scales are fixed before training, not fitted to held-out episodes.
All targets describe the same current simulator state as the visual observation.
"""

import math
import numpy as np


REPLAY_KEY = 'object_state_target'
FORMAT = 'object_state_supervision_v1'
VELOCITY_SCALE = 10.0
REACHER_DISTANCE_SCALE = 0.3
SCHEMAS = {
    'acrobot-swingup': {
        'fields': (('orientations', 4), ('velocity', 2)),
        'output': (
            'orientation_0', 'orientation_1', 'orientation_2', 'orientation_3',
            'joint_velocity_0_div10', 'joint_velocity_1_div10',
        ),
    },
    'cartpole-swingup': {
        'fields': (('position', 3), ('velocity', 2)),
        'output': (
            'cart_position', 'pole_cos_angle', 'pole_sin_angle',
            'cart_velocity_div10', 'pole_velocity_div10',
        ),
    },
    'reacher-visual-small': {
        'fields': (('position', 2), ('to_target', 2), ('velocity', 2)),
        'output': (
            'joint_0_sin', 'joint_1_sin', 'joint_0_cos', 'joint_1_cos',
            'finger_to_target_x_div0p3', 'finger_to_target_y_div0p3',
            'joint_velocity_0_div10', 'joint_velocity_1_div10',
        ),
    },
}


def _value(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, 'get') else getattr(cfg, key, default)


def enabled(cfg):
    return bool(_value(cfg, 'object_state_supervision_enabled', False))


def bottleneck_enabled(cfg):
    """Whether predicted state is the controller's mandatory visual bottleneck."""
    return bool(_value(cfg, 'object_state_bottleneck_enabled', False))


def _task(cfg):
    task = str(_value(cfg, 'task', ''))
    if task not in SCHEMAS:
        raise ValueError(f'No object-state supervision schema for {task!r}.')
    return task


def target_dim(cfg):
    return len(SCHEMAS[_task(cfg)]['output'])


def contract(cfg):
    """Checkpoint identity excludes collection mode, which is off at evaluation."""
    task = _task(cfg)
    return {
        'format': FORMAT,
        'enabled': enabled(cfg),
        'task': task,
        'coefficient': float(_value(cfg, 'object_state_supervision_coef', 0.0)),
        'target_source': 'dm_control.task.get_observation(current_physics)',
        'source_fields': [list(item) for item in SCHEMAS[task]['fields']],
        'target_fields': list(SCHEMAS[task]['output']),
        'target_dim': target_dim(cfg),
        'velocity_scale': VELOCITY_SCALE,
        'reacher_distance_scale': REACHER_DISTANCE_SCALE,
        'reacher_angle_transform': 'concat(sin(qpos),cos(qpos))',
        'target_clipping': False,
        'replay_key': REPLAY_KEY,
        'alignment': 'target[t] matches obs[t], including reset and terminal obs',
        'controller_input_contains_state': False,
        'controller_input_contains_predicted_state': bottleneck_enabled(cfg),
        'controller_visual_bypass': not bottleneck_enabled(cfg),
        'bottleneck_format': (
            'predicted_state_prefix_zero_pad_v1'
            if bottleneck_enabled(cfg) else None
        ),
        'bottleneck_state_dim': (
            target_dim(cfg) if bottleneck_enabled(cfg) else None
        ),
        'bottleneck_latent_dim': (
            int(_value(cfg, 'latent_dim', 0)) if bottleneck_enabled(cfg) else None
        ),
        'bottleneck_transition': (
            'state_and_action_to_next_predicted_state_v1'
            if bottleneck_enabled(cfg) else None
        ),
        'evaluation_label_collection': False,
    }


def validate_config(cfg):
    if not enabled(cfg):
        return None
    _task(cfg)
    expected = {
        'obs': 'rgb',
        'multitask': False,
        'flat_anchor': True,
        'flat_anchor_mode': 'cutie_object_only',
        'cutie_object_observation_variant': 'full',
        'cutie_object_allow_simulator_runtime': False,
        'cutie_object_allow_simulator_kinematics_runtime': False,
    }
    bad = {
        key: (_value(cfg, key, False if isinstance(value, bool) else None), value)
        for key, value in expected.items()
        if _value(cfg, key, False if isinstance(value, bool) else None) != value
    }
    coef = float(_value(cfg, 'object_state_supervision_coef', 0.0))
    if not math.isfinite(coef) or coef < 0:
        bad['object_state_supervision_coef'] = (coef, 'finite and >=0')
    if bottleneck_enabled(cfg):
        if coef <= 0:
            bad['object_state_supervision_coef'] = (coef, 'finite and >0 for bottleneck')
        latent_dim = int(_value(cfg, 'latent_dim', 0))
        if latent_dim < target_dim(cfg):
            bad['latent_dim'] = (latent_dim, f'>={target_dim(cfg)}')
    if bad:
        raise ValueError(f'Object-state supervision configuration mismatch: {bad}.')
    return contract(cfg)


def target_from_observation(cfg, observation):
    """Convert only fixed approved fields; reward, time and images are excluded."""
    task = _task(cfg)
    values = {}
    for name, count in SCHEMAS[task]['fields']:
        if name not in observation:
            raise ValueError(f'{task} official observation is missing {name!r}.')
        value = np.asarray(observation[name], dtype=np.float64)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f'{task} state field {name!r} is invalid: {value!r}.')
        values[name] = value
    if task == 'acrobot-swingup':
        chunks = (values['orientations'], values['velocity'] / VELOCITY_SCALE)
    elif task == 'cartpole-swingup':
        chunks = (values['position'], values['velocity'] / VELOCITY_SCALE)
    else:
        chunks = (
            np.sin(values['position']), np.cos(values['position']),
            values['to_target'] / REACHER_DISTANCE_SCALE,
            values['velocity'] / VELOCITY_SCALE,
        )
    target = np.concatenate(chunks).astype(np.float32)
    if target.shape != (target_dim(cfg),) or not np.isfinite(target).all():
        raise ValueError('Object-state target has an invalid shape or value.')
    return target
