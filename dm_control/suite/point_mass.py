# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 2
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('point_mass.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the hard point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['pointmass'])

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target())


class PointMass(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    super(PointMass, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

       If _randomize_gains is True, the relationship between the controls and
       the joints is randomized, so that each control actuates a random linear
       combination of joints.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    if self._randomize_gains:
      dir1 = self.random.randn(2)
      dir1 /= np.linalg.norm(dir1)
      # Find another actuation direction that is not 'too parallel' to dir1.
      parallel = True
      while parallel:
        dir2 = self.random.randn(2)
        dir2 /= np.linalg.norm(dir2)
        parallel = abs(np.dot(dir1, dir2)) > 0.9
      physics.model.wrap_prm[[0, 1]] = dir1
      physics.model.wrap_prm[[2, 3]] = dir2
    super(PointMass, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()

    # original
    # obs['position'] = physics.position()
    # obs['velocity'] = physics.velocity()
    # obs['observation'] = np.concatenate([physics.position(),physics.velocity()])

    # pixels
    # obs['observation'] = physics.render(64,64,0).copy()
    #flatten pixels
    obs['observation'] = physics.render(64,64,0).flatten().copy()
    # obs['observation'] = np.transpose(physics.render(64,64,0), (2,0,1)).copy()
    obs['desired_goal'] = physics.named.data.geom_xpos['target'][:2].copy()
    obs['achieved_goal'] = physics.named.data.geom_xpos['pointmass'][:2].copy()

    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    # target_size = 0.3
    # near_target = rewards.tolerance(physics.mass_to_target_dist(), sigmoid="quadratic",
    #                                 bounds=(0, 0), margin=target_size)
    # control_reward = rewards.tolerance(physics.control(), margin=1,
    #                                    value_at_margin=0,
    #                                    sigmoid='quadratic').mean()
    # small_control = (control_reward + 4) / 5
    # return near_target * small_control

    # Sparse reward

    # eps = 0.1
    # return 1 if physics.mass_to_target_dist() < eps else 0

    # Dense reward - distance only
    target_size = 0.3
    return rewards.tolerance(physics.mass_to_target_dist(), sigmoid="quadratic",
                                     bounds=(0, 0), margin=target_size)



  def before_step(self, action, physics):
    """Sets the control signal for the actuators to values in `action`."""
    # Support legacy internal code.
    
    #expecting action to be a one hot prediction.
    # possible_actions = np.array([[0,0], [1,1], [1, -1], [-1, 1], [-1,-1]])
    # action = possible_actions[np.argmax(action)]

    action = getattr(action, "continuous_actions", action)
    physics.data.qvel[:2] = action
    physics.set_control(action*0.)

  def compute_reward(self, achieved_goal, desired_goal, info):
    """Compute the step reward. This externalizes the reward function and makes
    it dependent on an a desired goal and the one that was achieved. If you wish to include
    additional rewards that are independent of the goal, you can include the necessary values
    to derive it in info and compute it accordingly.
    Args:
        achieved_goal (object): the goal that was achieved during execution
        desired_goal (object): the desired goal that we asked the agent to attempt to achieve
        info (dict): an info dictionary with additional information
    Returns:
        float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
        goal. Note that the following should always hold true:
            ob, reward, done, info = env.step()
            assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
    """
    dist = np.linalg.norm(achieved_goal - desired_goal)
    eps = 0.1
    return 1 if dist < eps else 0 