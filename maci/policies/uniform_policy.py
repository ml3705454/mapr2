from maci.core.serializable import Serializable
import tensorflow as tf
from maci.misc.overrides import overrides
from maci.policies.base import Policy

import numpy as np


class UniformPolicy(Policy, Serializable):
    """
    Fixed policy that randomly samples actions uniformly at random.

    Used for an initial exploration period instead of an undertrained policy.
    """
    def __init__(self, env_spec, agent_id, opponent=False, urange=[-1, 1.], if_softmax=False):
        Serializable.quick_init(self, locals())
        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._urange = urange
        self._if_softmax = if_softmax
        if opponent:
            self._action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
        self._name = 'uniform_policy_{}'.format(agent_id)
        super(UniformPolicy, self).__init__(env_spec)

    # Assumes action spaces are normalized to be the interval [-1, 1]
    @overrides
    def get_action(self, observation):
        actions = np.random.uniform(self._urange[0], self._urange[1], self._action_dim), np.zeros(0, 1)
        if not self._if_softmax:
            return actions
        else:
            return tf.nn.softmax(actions)

    @overrides
    def get_actions(self, observations):
        observations = np.array(observations)
        actions = np.random.uniform(self._urange[0], self._urange[1], (observations.shape[0], self._action_dim)), np.zeros(0, (observations.shape[0],))
        if not self._if_softmax:
            return actions
        else:
            return tf.nn.softmax(actions)

    def actions_for(self, observations, reuse=True):
        n_state_samples = tf.shape(observations)[0]
        action_shape = (n_state_samples, self._action_dim)
        actions = tf.random_uniform(action_shape, self._urange[0], self._urange[1])
        print('uniform', self._urange, self._if_softmax)
        if not self._if_softmax:
            return actions
        else:
            return tf.nn.softmax(actions)

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        pass 

