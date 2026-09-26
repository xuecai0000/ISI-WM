"""CPU tests for privileged-target schemas, timing, isolation and replay shape."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common import object_state_supervision as supervision


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def config(task='acrobot-swingup', collect=True):
    return Config(
        task=task, obs='rgb', multitask=False, flat_anchor=True,
        flat_anchor_mode='cutie_object_only',
        cutie_object_observation_variant='full',
        cutie_object_input_dim=1770,
        object_state_supervision_enabled=True,
        object_state_supervision_coef=0.1,
        object_state_supervision_collect_labels=collect,
    )


def assert_raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}.')


def load_wrapper():
    # Test the wrapper without importing the optional simulator package graph.
    path = Path(__file__).resolve().parent / 'envs/wrappers/object_state_supervision.py'
    spec = importlib.util.spec_from_file_location('_state_target_wrapper_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_schema():
    cases = (
        ('acrobot-swingup', {'orientations': [1, 2, 3, 4], 'velocity': [10, 20]},
         [1, 2, 3, 4, 1, 2]),
        ('cartpole-swingup', {'position': [0.4, 0, 1], 'velocity': [-10, 30]},
         [0.4, 0, 1, -1, 3]),
        ('reacher-visual-small', {'position': [0, np.pi / 2], 'to_target': [0.3, -0.6],
                                  'velocity': [10, 20]},
         [0, 1, 1, 0, 1, -2, 1, 2]),
    )
    for task, observation, expected in cases:
        cfg = config(task)
        supervision.validate_config(cfg)
        original = {key: list(value) for key, value in observation.items()}
        got = supervision.target_from_observation(cfg, observation)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        assert got.dtype == np.float32
        assert got.shape == (supervision.target_dim(cfg),)
        assert observation == original
        # Extra values such as reward are not consumed by the target channel.
        observation['reward'] = object()
        np.testing.assert_array_equal(got, supervision.target_from_observation(cfg, observation))
        assert supervision.contract(cfg) == supervision.contract(config(task, collect=False))
        bad = dict(observation)
        field = supervision.SCHEMAS[task]['fields'][0][0]
        bad[field] = [np.nan]
        assert_raises(ValueError, lambda: supervision.target_from_observation(cfg, bad))
    cfg = config()
    cfg.cutie_object_allow_simulator_kinematics_runtime = True
    assert_raises(ValueError, lambda: supervision.validate_config(cfg))
    cfg = config()
    cfg.object_state_supervision_coef = float('nan')
    assert_raises(ValueError, lambda: supervision.validate_config(cfg))


class Source:
    def __init__(self):
        self.physics = self
        self.task = self
        self.time = 0
        self.calls = 0
        self.forbid = False

    def get_observation(self, physics):
        assert physics is self
        if self.forbid:
            raise AssertionError('Privileged source was accessed during evaluation.')
        self.calls += 1
        return {'orientations': np.ones(4) * self.time, 'velocity': np.ones(2) * 10 * self.time}


class VisualEnv:
    def __init__(self, source):
        self.env = source
        self.obs = {'object': np.zeros((2, 1770), dtype=np.float32)}
        self.info = {'success': 0.0}

    def reset(self):
        self.env.time = 0
        return self.obs

    def step(self, action):
        self.env.time += 1
        return self.obs, 0.0, self.env.time == 2, self.info

    def metrics(self):
        return {'cutie_marker': True}


def check_wrapper():
    module = load_wrapper()
    source = Source()
    inner = VisualEnv(source)
    wrapper = module.ObjectStateSupervisionWrapper(inner, config())
    assert source.calls == 0
    assert wrapper.reset() is inner.obs
    np.testing.assert_array_equal(wrapper.get_object_state_target(), np.zeros(6))
    for index in (1, 2):
        result = wrapper.step(0)
        assert result[0] is inner.obs and result[3] is inner.info
        np.testing.assert_array_equal(wrapper.get_object_state_target(), np.full(6, index))
    assert wrapper.state_supervision_label_reads == 3
    copied = wrapper.get_object_state_target()
    copied[:] = 99
    np.testing.assert_array_equal(wrapper.get_object_state_target(), np.full(6, 2))
    assert wrapper.metrics() == {'cutie_marker': True}
    source.forbid = True
    with module.without_state_labels(wrapper):
        assert wrapper.reset() is inner.obs
        wrapper.step(0)
        assert_raises(RuntimeError, wrapper.get_object_state_target)
        with module.without_state_labels(wrapper):
            wrapper.step(0)
    assert source.calls == 3
    assert_raises(RuntimeError, wrapper.get_object_state_target)
    source.forbid = False
    wrapper.reset()
    assert source.calls == 4
    # Exception paths also restore the mode but never stale cached labels.
    try:
        with module.without_state_labels(wrapper):
            raise LookupError('simulated evaluator failure')
    except LookupError:
        pass
    assert wrapper._collect
    assert_raises(RuntimeError, wrapper.get_object_state_target)
    eval_source = Source()
    eval_source.forbid = True
    evaluation = module.ObjectStateSupervisionWrapper(VisualEnv(eval_source), config(collect=False))
    evaluation.reset()
    evaluation.step(0)
    with module.without_state_labels(evaluation):
        evaluation.reset()
    assert not evaluation._collect
    assert evaluation._source is None
    assert evaluation.state_supervision_label_reads == 0
    assert eval_source.calls == 0


def check_replay():
    import torch
    from tensordict import TensorDict
    from common.buffer import Buffer

    cfg = config()
    buffer = Buffer.__new__(Buffer)
    buffer.cfg = cfg
    buffer._state_supervision_enabled = True
    buffer._device = torch.device('cpu')
    time, batch = 4, 2
    targets = torch.arange(time * batch * 6, dtype=torch.float32).reshape(time, batch, 6)
    objects = TensorDict({'object': torch.zeros(time, batch, 2, 1770)}, batch_size=(time, batch))
    td = TensorDict({
        'obs': objects, 'action': torch.arange(time * batch, dtype=torch.float32).reshape(time, batch, 1),
        'reward': torch.ones(time, batch), 'terminated': torch.zeros(time, batch),
        supervision.REPLAY_KEY: targets,
    }, batch_size=(time, batch))
    result = buffer._prepare_batch(td)
    assert len(result) == 6
    torch.testing.assert_close(result[5], targets)
    torch.testing.assert_close(result[1], td['action'][1:])
    assert result[0]['object'].shape[:2] == result[5].shape[:2]
    assert set(result[0].keys()) == {'object'}
    missing = td.exclude(supervision.REPLAY_KEY)
    assert_raises(ValueError, lambda: buffer._prepare_batch(missing))
    bad = td.clone()
    bad[supervision.REPLAY_KEY][1, 0, 0] = float('nan')
    assert_raises(ValueError, lambda: buffer._prepare_batch(bad))
    buffer._state_supervision_enabled = False
    assert len(buffer._prepare_batch(missing)) == 5
    assert_raises(ValueError, lambda: buffer._prepare_batch(td))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-torch', action='store_true')
    args = parser.parse_args()
    check_schema()
    check_wrapper()
    if args.require_torch:
        check_replay()
    print('OBJECT_STATE_SUPERVISION_PIPELINE_OK', {'replay_tested': args.require_torch})


if __name__ == '__main__':
    main()
