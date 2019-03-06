import tensorflow as tf
from maci.policies.nn_policy import NNPolicy
from maci.core.serializable import Serializable
from scipy.stats import poisson
import numpy as np
from maci.misc.utils import concat_obs_z
# tf.enable_eager_execution()


class MultiLevelPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, base_policy, conditional_policy, opponent_conditional_policy, agent_id, k, name='level_k'):
        Serializable.quick_init(self, locals())
        self._base_policy = base_policy
        self._conditional_policy = conditional_policy
        self._opponent_conditional_policy = opponent_conditional_policy
        self._k = k
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._action_dim = env_spec.action_space[agent_id].flat_dim

        self._name = name + '_agent_{}'.format(agent_id)
        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='{}_observation_agent_{}'.format(name, agent_id))

        # self._observation_ph = None
        # self._actions = None

        self.agent_id = agent_id
        self._actions, self.all_actions = self.actions_for(self._observation_ph, reuse=True, all_action=True)

        super(MultiLevelPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def get_all_actions(self, observations):
        feeds = {self._observation_ph: observations}
        all_actions = tf.get_default_session().run(self.all_actions, feeds)
        return all_actions

    def actions_for(self, observations, reuse=True, all_action=False):
        all_actions = []
        action = None
        for i in range(self._k + 1):
            if i == 0:
                policy = self._base_policy
                if self._k % 2 == 1:
                    action = tf.stop_gradient(policy.actions_for(observations, reuse=reuse))
                else:
                    with tf.variable_scope(self._name, reuse=reuse):
                        action = policy.actions_for(observations, reuse=reuse)
            else:
                if self._k % 2 == 0:
                    if i % 2 == 1:
                        policy = self._opponent_conditional_policy
                        action = tf.stop_gradient(policy.actions_for(observations, action, reuse=reuse))
                    else:
                        policy = self._conditional_policy
                        with tf.variable_scope(self._name, reuse=reuse):
                            action = policy.actions_for(observations, action, reuse=reuse)
                else:
                    if i % 2 == 1:
                        policy = self._conditional_policy
                        with tf.variable_scope(self._name, reuse=reuse):
                            action = policy.actions_for(observations, action, reuse=reuse)
                    else:
                        policy = self._opponent_conditional_policy
                        action = tf.stop_gradient(policy.actions_for(observations, action, reuse=reuse))
            all_actions.append(action)

        if all_action:
            return action, all_actions
        else:
            return action



class GeneralizedMultiLevelPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, policies, agent_id, k, mu=1., name='g_level_k', correct_tanh=True):
        Serializable.quick_init(self, locals())
        self._policies = policies
        assert k > 1
        self._k = k
        self._mu = mu
        self._dists = self.level_distribution(self._k, self._mu)
        if correct_tanh:
            self._correction_factor = 1.
        else:
            self._correction_factor = 0.

        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._action_dim = env_spec.action_space[agent_id].flat_dim

        self._name = name + '_agent_{}'.format(agent_id)
        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='{}_observation_agent_{}'.format(name, agent_id))

        self.agent_id = agent_id
        self._actions, self.all_actions = self.actions_for(self._observation_ph, reuse=True, all_action=True)

        super(GeneralizedMultiLevelPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def get_all_actions(self, observations):
        feeds = {self._observation_ph: observations}
        all_actions = tf.get_default_session().run(self.all_actions, feeds)
        return all_actions

    def level_distribution(self, k, mu):
        _dists = np.array([poisson.pmf(kk, mu) for kk in range(1, k+1)])
        return _dists / np.sum(_dists)

    def actions_for(self, observations, reuse=True, all_action=False):
        action = None
        all_actions = None
        with tf.variable_scope(self._name, reuse=reuse):
            for i, (dist, policy) in enumerate(zip(self._dists, self._policies)):
                if i == 0:
                    action = policy.actions_for(observations, reuse=reuse)
                    action = dist * (self._correction_factor + action)
                else:
                    tmp_action, all_actions = policy.actions_for(observations, reuse=reuse, all_action=True)
                    action = action + dist * (self._correction_factor + tmp_action)
            action = action - self._correction_factor
        if all_action:
            return action, all_actions
        else:
            return action

