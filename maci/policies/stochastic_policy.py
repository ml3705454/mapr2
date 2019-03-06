
from maci.environments.env_spec import MAEnvSpec

import tensorflow as tf

from maci.core.serializable import Serializable

from maci.misc.nn import feedforward_net

from .nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='stochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None


        self._actions = self.actions_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def actions_for(self, observations, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)


class StochasticNNConditionalPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 opponent_action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,

                 squash_func=tf.tanh,
                 name='conditional_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        self.agent_id = agent_id
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
            self._opponent_action_dim = opponent_action_space.flat_dim
        else:
            assert isinstance(env_spec, MAEnvSpec)
            assert agent_id is not None
            self._action_dim = env_spec.action_space[agent_id].flat_dim
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
        print('opp dim', self._opponent_action_dim)
        self._layer_sizes = list(hidden_layer_sizes) + [self._opponent_action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self.sampling = sampling
        self._name = name + '_agent_{}'.format(agent_id)

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._action_dim],
            name='actions_{}_agent_{}'.format(name, agent_id))
        self._opponent_actions = self.actions_for(self._observation_ph, self._actions_ph)

        super(StochasticNNConditionalPolicy, self).__init__(
            env_spec, self._observation_ph, self._opponent_actions, self._name)

    def get_action(self, observation, action):
        return self.get_actions(observation[None], action[None])[0], None

    def get_actions(self, observations, self_actions):
        feeds = {self._observation_ph: observations, self._actions_ph: self_actions}
        actions = tf.get_default_session().run(self._opponent_actions, feeds)
        return actions

    def actions_for(self, observations, actions, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]
        # n_action_samples = tf.shape(actions)[0]
        # assert n_state_samples == n_action_samples

        if n_action_samples > 1:
            observations = observations[:, None, :]
            actions = actions[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._opponent_action_dim)
        else:
            latent_shape = (n_state_samples, self._opponent_action_dim)

        latents = tf.random_normal(latent_shape)
        # print('latents', latents)
        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, actions, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            # print('raw_actions', raw_actions)
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('cond stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions,
                                                                                                        -self._u_range,
                                                                                                        self._u_range)

