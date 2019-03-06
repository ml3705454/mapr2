import tensorflow as tf

from maci.core.serializable import Serializable

from maci.misc.mlp import MLPFunction
from maci.misc import tf_utils
from maci.environments.env_spec import MAEnvSpec
import numpy as np


class NNVFunction(MLPFunction):

    def __init__(self, env_spec, agent_id=None, hidden_layer_sizes=(100, 100), name='vf'):
        Serializable.quick_init(self, locals())
        self._observation_dim = env_spec.observation_space.flat_dim
        if agent_id is not None and agent_id != 'all':
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation',
        )
        super(NNVFunction, self).__init__(
            name, (self._obs_pl,), hidden_layer_sizes)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='qf', joint=False, agent_id=None, maddpg=False):
        Serializable.quick_init(self, locals())
        if isinstance(env_spec, MAEnvSpec):
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

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._action_dim],
            name='actions',
        )

        super(NNQFunction, self).__init__(
            name, (self._obs_pl, self._action_pl), hidden_layer_sizes)


# class NNQFunction(MLPFunction):
#     def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='qf', joint=False, agent_id=None):
#         Serializable.quick_init(self, locals())
#         if isinstance(env_spec, MAEnvSpec):
#             assert agent_id is not None
#             self._observation_dim = env_spec.observation_space[agent_id].flat_dim
#             if joint:
#                 self._action_dim = env_spec.action_space.flat_dim
#             else:
#                 self._action_dim = env_spec.action_space[agent_id].flat_dim
#         else:
#             self._action_dim = env_spec.action_space.flat_dim
#             self._observation_dim = env_spec.observation_space.flat_dim
#
#         self._obs_pl = tf.placeholder(
#             tf.float32,
#             shape=[None, self._observation_dim],
#             name='observation',
#         )
#
#         self._action_pl = tf.placeholder(
#             tf.float32,
#             shape=[None, self._action_dim],
#             name='actions',
#         )
#
#         super(NNQFunction, self).__init__(
#             name, (self._obs_pl, self._action_pl), hidden_layer_sizes)



# class NNDiscriminatorFunction(MLPFunction):
#     def __init__(self, env_spec, hidden_layer_sizes=(100, 100), num_skills=None):
#         assert num_skills is not None
#         Serializable.quick_init(self, locals())
#         Parameterized.__init__(self)
#
#         self._action_dim = env_spec.action_space.flat_dim
#         self._observation_dim = env_spec.observation_space.flat_dim
#
#         self._obs_pl = tf.placeholder(
#             tf.float32,
#             shape=[None, self._observation_dim],
#             name='observation',
#         )
#         self._action_pl = tf.placeholder(
#             tf.float32,
#             shape=[None, self._action_dim],
#             name='actions',
#         )
#
#         self._name = 'discriminator'
#         self._input_pls = (self._obs_pl, self._action_pl)
#         self._layer_sizes = list(hidden_layer_sizes) + [num_skills]
#         self._output_t = self.get_output_for(*self._input_pls)


class SumQFunction(Serializable):
    def __init__(self, env_spec, q_functions):
        Serializable.quick_init(self, locals())

        self.q_functions = q_functions

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
