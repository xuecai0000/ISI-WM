"""Side-channel training labels, without modifying visual observations or info."""

from contextlib import contextmanager

from common import object_state_supervision as supervision


class ObjectStateSupervisionWrapper:
    """Outermost wrapper: collect after reset/step has produced its observation.

    Label access is lazy and disabled for held-out environments. The source is
    never located/read in a collect_labels=False environment. No target is
    attached to observations, rewards, actions, or the public step info dict.
    """

    def __init__(self, env, cfg):
        supervision.validate_config(cfg)
        self.env = env
        self.cfg = cfg
        self._collection_permitted = bool(cfg.get(
            'object_state_supervision_collect_labels', False
        ))
        self._collect = self._collection_permitted
        self._source = None
        self._latest = None
        self._label_reads = 0
        self._reset_reads = 0
        self._step_reads = 0
        self._evaluation_suspensions = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def state_supervision_label_reads(self):
        return self._label_reads

    def _find_source(self):
        current, visited = self.env, set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            # Gym wrappers expose forwarded physics/task attributes. Follow
            # their concrete links first to avoid deprecated forwarding APIs.
            namespace = vars(current)
            child = namespace.get('env', namespace.get('_env'))
            if child is not None:
                current = child
                continue
            task = getattr(current, 'task', None)
            physics = getattr(current, 'physics', None)
            if task is not None and physics is not None and callable(
                getattr(task, 'get_observation', None)
            ):
                return current
            break
        raise RuntimeError('Cannot find official dm-control task/physics for labels.')

    def _capture(self, phase):
        self._latest = None
        if not self._collect:
            return
        if self._source is None:
            self._source = self._find_source()
        target = supervision.target_from_observation(
            self.cfg, self._source.task.get_observation(self._source.physics)
        )
        self._latest = target
        self._label_reads += 1
        self._reset_reads += int(phase == 'reset')
        self._step_reads += int(phase == 'step')

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self._capture('reset')
        return observation

    def step(self, action):
        result = self.env.step(action)
        self._capture('step')
        return result

    def get_object_state_target(self):
        if not self._collect or self._latest is None:
            raise RuntimeError('Training state target unavailable; evaluation labels are forbidden.')
        return self._latest.copy()

    def state_supervision_metrics(self):
        return {
            'contract': supervision.contract(self.cfg),
            'collection_configured': self._collection_permitted,
            'label_reads': self._label_reads,
            'reset_label_reads': self._reset_reads,
            'step_label_reads': self._step_reads,
            'evaluation_suspensions': self._evaluation_suspensions,
            'labels_in_observation': False,
        }


def find_state_supervision_wrapper(env):
    current, visited = env, set()
    while current is not None and id(current) not in visited:
        if isinstance(current, ObjectStateSupervisionWrapper):
            return current
        visited.add(id(current))
        current = vars(current).get('env')
    return None


@contextmanager
def without_state_labels(env):
    """Disable auxiliary-state access throughout evaluation, including reset."""
    wrapper = find_state_supervision_wrapper(env)
    if wrapper is None:
        yield
        return
    previous = wrapper._collect
    before = wrapper.state_supervision_label_reads
    wrapper._collect = False
    wrapper._latest = None
    wrapper._evaluation_suspensions += 1
    try:
        yield
    finally:
        wrapper._collect = previous
        wrapper._latest = None
        if wrapper.state_supervision_label_reads != before:
            raise RuntimeError('Evaluation illegally collected privileged state labels.')
