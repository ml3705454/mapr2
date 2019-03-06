import tensorflow as tf
import numpy as np

from maci.core.serializable import Serializable

from maci.misc.nn import MLPFunction
from maci.misc import tf_utils
from maci.environments.env_spec import MAEnvSpec


class NNVFunction(MLPFunction):
    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 name='value_function',
                 joint=False,
                 agent_id=None):
        Serializable.quick_init(self, locals())
        self._name = name + '_agent_{}'.format(agent_id)

        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
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

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._observation_dim], name='observations_agent_{}'.format(agent_id))
        super(NNVFunction, self).__init__(
            inputs=(self._observations_ph, ),
            name=self._name,
            hidden_layer_sizes=hidden_layer_sizes)

    def eval(self, observations):
        return super(NNVFunction, self)._eval((observations, ))

    def output_for(self, observations, reuse=False):
        return super(NNVFunction, self)._output_for(
            (observations, ), reuse=reuse)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 name='q_function',
                 joint=False, agent_id=None, maddpg=False):
        Serializable.quick_init(self, locals())
        self._name = name + '_agent_{}'.format(agent_id)
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if maddpg:
                self._observation_dim = env_spec.observation_space.flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim
            if maddpg:
                self._observation_dim = env_spec.observation_space.flat_dim

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._observation_dim], name='observations_agent_{}'.format(agent_id))
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions_agent_{}'.format(agent_id))

        super(NNQFunction, self).__init__(
            inputs=(self._observations_ph, self._actions_ph),
            name=self._name,
            hidden_layer_sizes=hidden_layer_sizes)

    def output_for(self, observations, actions, reuse=False):
        return super(NNQFunction, self)._output_for(
            (observations, actions), reuse=reuse)

    def eval(self, observations, actions):
        return super(NNQFunction, self)._eval((observations, actions))


class NNJointQFunction(MLPFunction):
    def __init__(self, env_spec=None,
                 observation_space=None,
                 action_space=None,
                 opponent_action_space=None,
                 hidden_layer_sizes=(100, 100),
                 name='joint_q_function',
                 joint=False, agent_id=None):
        Serializable.quick_init(self, locals())
        self._name = name + '_agent_{}'.format(agent_id)
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

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._observation_dim], name='observations_agent_{}'.format(agent_id))
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions_agent_{}'.format(agent_id))
        self._opponent_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._opponent_action_dim], name='opponent_actions_agent_{}'.format(agent_id))

        super(NNJointQFunction, self).__init__(
            inputs=(self._observations_ph, self._actions_ph, self._opponent_actions_ph),
            name=self._name,
            hidden_layer_sizes=hidden_layer_sizes)

    def output_for(self, observations, actions, opponent_actions, reuse=False):
        return super(NNJointQFunction, self)._output_for(
            (observations, actions, opponent_actions), reuse=reuse)

    def eval(self, observations, actions, opponent_actions):
        return super(NNJointQFunction, self)._eval((observations, actions, opponent_actions))


class SumQFunction(Serializable):
    def __init__(self, env_spec, q_functions):
        Serializable.quick_init(self, locals())

        self.q_functions = q_functions
        agent_id = 0
        joint = True
        if isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim

        self._observations_ph = tf.placeholder(
            tf.float32, shape=[None, self._observation_dim], name='observations')
        self._actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions')

        self._output = self.output_for(
            self._observations_ph, self._actions_ph, reuse=True)

    def output_for(self, observations, actions, reuse=False):
        outputs = [
            qf.output_for(observations, actions, reuse=reuse)
            for qf in self.q_functions
        ]
        output = tf.add_n(outputs)
        return output

    def _eval(self, observations, actions):
        feeds = {
            self._observations_ph: observations,
            self._actions_ph: actions
        }

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_param_values(self):
        all_values_list = [qf.get_param_values() for qf in self.q_functions]

        return np.concatenate(all_values_list)

    def set_param_values(self, all_values):
        param_sizes = [qf.get_param_values().size for qf in self.q_functions]
        split_points = np.cumsum(param_sizes)[:-1]

        all_values_list = np.split(all_values, split_points)

        for values, qf in zip(all_values_list, self.q_functions):
            qf.set_param_values(values)