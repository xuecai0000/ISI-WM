import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import reacher
from dm_control.utils import io as resources
from dm_control.utils import rewards
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015
_VISUAL_SMALL_TARGET = .03
_VISUAL_SMALL_REWARD_TARGET = .05


class VisualSmallReacher(reacher.Reacher):
  """Reacher with independent visual and reward target radii."""

  def __init__(self, random=None):
    self._reward_target_radius = _VISUAL_SMALL_REWARD_TARGET
    super().__init__(target_size=_VISUAL_SMALL_TARGET, random=random)

  def get_reward(self, physics):
    finger_radius = physics.named.model.geom_size['finger', 0]
    reward_radius = self._reward_target_radius + finger_radius
    return rewards.tolerance(
        physics.finger_to_target_dist(), bounds=(0, reward_radius))


@reacher.SUITE.add('custom')
def visual_small(time_limit=reacher._DEFAULT_TIME_LIMIT, random=None,
                 environment_kwargs=None):
  """Returns Reacher with a 0.03 target and the easy-task reward radius."""
  physics = reacher.Physics.from_xml_string(*reacher.get_model_and_assets())
  task = VisualSmallReacher(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


def get_model_and_assets(links):
    """Returns a tuple containing the model XML string and a dict of assets."""
    assert links in {3, 4}, 'Only 3 or 4 links are supported.'
    fn = 'reacher_three_links.xml' if links == 3 else 'reacher_four_links.xml'
    return resources.GetResource(os.path.join(_TASKS_DIR, fn)), common.ASSETS


@reacher.SUITE.add('custom')
def three_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = CustomThreeLinkReacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = CustomThreeLinkReacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = CustomThreeLinkReacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = CustomThreeLinkReacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinates."""
    return (self.named.data.geom_xpos['target', :2] -
            self.named.data.geom_xpos['finger', :2])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class CustomThreeLinkReacher(reacher.Reacher):
  """Custom Reacher tasks."""

  def __init__(self, target_size, random=None):
    super().__init__(target_size, random)

  def get_observation(self, physics):
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs
