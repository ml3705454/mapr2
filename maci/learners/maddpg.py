import numpy as np
import tensorflow as tf

from maci.misc import logger
from maci.misc.overrides import overrides
import maci.misc.tf_utils as U

from maci.misc import tf_utils

from .base import MARLAlgorithm

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])

def tf_run(target, ph, feeds):
    return tf.get_default_session().run(target, {ph: feeds})

class MADDPG(MARLAlgorithm):
    def __init__(
            self,
            base_kwargs,
            agent_id,
            env,
            pool,
            qf,
            target_qf,
            policy,
            target_policy,
            name='MADDPG',
            opponent_policy=None,
            plotter=None,
            policy_lr=1e-2,
            qf_lr=1e-2,
            joint=False,
            opponent_modelling=False,
            td_target_update_interval=1,
            discount=0.95,
            tau=0.01,
            reward_scale=1,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_policy=True,
            joint_policy=False,
            SGA=False,
            grad_norm_clipping=0.5
    ):
        super(MADDPG, self).__init__(**base_kwargs)
        self.name = name
        self._env = env
        self._pool = pool
        self.qf = qf
        self.target_qf = target_qf
        # self.target_qf._name = 'target_' + self.traget_qf._name
        self.policy = policy
        self.target_policy = target_policy
        self.opponent_policy = opponent_policy
        # self._target_policy._name = 'target_' + self._target_policy._name
        self.plotter = plotter
        self.grad_norm_clipping = grad_norm_clipping

        self._agent_id = agent_id
        self.joint = joint
        self.opponent_modelling = opponent_modelling

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._tau = tau
        self._reward_scale = reward_scale
        self.SGA = SGA

        self._qf_target_update_interval = td_target_update_interval

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_policy = train_policy
        self.joint_policy = joint_policy

        self._observation_dim = self.env.observation_spaces[self._agent_id].flat_dim
        self._opponent_observation_dim = self.env.observation_spaces.opponent_flat_dim(self._agent_id)
        self._action_dim = self.env.action_spaces[self._agent_id].flat_dim
        self._opponent_action_dim = self.env.action_spaces.opponent_flat_dim(self._agent_id)
        self._all_observation_dim = self.env.observation_spaces.flat_dim
        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_q_update()
        self._create_p_update()
        if self.opponent_modelling:
            self._create_opponent_p_update()
        self._create_target_ops()


        if use_saved_qf:
            saved_qf_params = qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())
        self._init_training()

        if use_saved_qf:
            self.qf.set_param_values(saved_qf_params)
        if use_saved_policy:
            self.policy.set_param_values(saved_policy_params)

    def _create_placeholders(self):
        """Create all necessary placeholders."""

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations_agent_{}'.format(self._agent_id))

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations_agent_{}'.format(self._agent_id))
        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='actions_agent_{}'.format(self._agent_id))
        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='next_actions_agent_{}'.format(self._agent_id))

        self._all_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._all_observation_dim],
            name='all_observations_agent_{}'.format(self._agent_id))

        self._all_next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._all_observation_dim],
            name='all_next_observations_agent_{}'.format(self._agent_id))
        self._opponent_current_actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._opponent_action_dim],
            name='opponent_actions_agent_{}'.format(self._agent_id))
        if self.joint:
            self._opponent_actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_actions_agent_{}'.format(self._agent_id))

            self._opponent_next_actions_ph = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_next_actions_agent_{}'.format(self._agent_id))

        if self.opponent_modelling:
            self._recent_opponent_observations_ph = tf.placeholder(
                tf.float32,shape=[None, self._observation_dim],
                name='recent_opponent_observations_agent_{}'.format(self._agent_id))
            self._recent_opponent_actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='recent_opponent_actions_agent_{}'.format(self._agent_id))

        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='rewards_agent_{}'.format(self._agent_id))

        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='terminals_agent_{}'.format(self._agent_id))


    def _create_q_update(self):
        """Create a minimization operation for Q-function update."""
        o_n_ph = self._next_observations_ph
        if self.name == 'MADDPG':
            o_n_ph = self._all_next_observations_ph
        with tf.variable_scope('target_q_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            next_actions = self._next_actions_ph
            if self.joint:
                next_actions = tf.concat([self._next_actions_ph, self._opponent_next_actions_ph], 1)
            q_value_targets = self.target_qf.output_for(
                observations=o_n_ph,
                actions=next_actions, reuse=tf.AUTO_REUSE)
            assert_shape(q_value_targets, [None])

        actions = self._actions_pl
        if self.joint:
            actions = tf.concat([self._actions_pl, self._opponent_actions_pl], 1)
        o_ph = self._observations_ph
        if self.name == 'MADDPG':
            o_ph = self._all_observations_ph
            # print('q all')
        self._q_values = self.qf.output_for(o_ph, actions, reuse=True)
        assert_shape(self._q_values, [None])

        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
                1 - self._terminals_pl) * self._discount * q_value_targets)
        assert_shape(ys, [None])

        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values) ** 2)
        if not self.SGA:
            with tf.variable_scope('q_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
                if self._train_qf:
                    optimizer = tf.train.AdamOptimizer(self._qf_lr)
                    q_train_op = U.minimize_and_clip(optimizer, bellman_residual, self.qf.get_params_internal(), self.grad_norm_clipping)
                    self._training_ops.append(q_train_op)

        self._bellman_residual = bellman_residual


    def _create_opponent_p_update(self):
        opponent_actions = self.opponent_policy.actions_for(
            observations=self._recent_opponent_observations_ph,
            reuse=tf.AUTO_REUSE)
        print(opponent_actions, [None, self._opponent_action_dim])
        assert_shape(opponent_actions, [None, self._opponent_action_dim])
        om_loss = 0.5 * tf.reduce_mean((self._recent_opponent_actions_pl - opponent_actions) ** 2)
        with tf.variable_scope('opponent_policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                # optimizer = tf.train.AdamOptimizer(self._policy_lr)
                # om_training_op = optimizer.minimize(
                #     loss=om_loss,
                #     var_list=self.opponent_policy.get_params_internal())
                optimizer = tf.train.AdamOptimizer(self._qf_lr)
                om_training_op = U.minimize_and_clip(optimizer, om_loss, self.opponent_policy.get_params_internal(),
                                                 self.grad_norm_clipping)

                self._training_ops.append(om_training_op)


    def _create_p_update(self):
        """Create a minimization operation for policy update """
        with tf.variable_scope('target_p_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            self_target_actions = self.target_policy.actions_for(
                observations=self._observations_ph,
                reuse=tf.AUTO_REUSE)
        raw_actions = None
        # if self.name == 'MADDPG':
        #     self_actions, raw_actions = self.policy.actions_for(
        #         observations=self._observations_ph,
        #         reuse=tf.AUTO_REUSE, with_raw=True)
        # else:
        self_actions = self.policy.actions_for(
                    observations=self._observations_ph,
                    reuse=tf.AUTO_REUSE)
        assert_shape(self_actions, [None, self._action_dim])

        actions = self_actions
        if self.joint:
            actions = tf.concat([self_actions, self._opponent_actions_pl], 1)

            # self._opponent_current_actions_pl
        # if self.name == 'MADDPG':
        #     actions = tf.concat([self_actions, self._opponent_current_actions_pl], 1)
        o_ph = self._observations_ph
        if self.name == 'MADDPG':
            o_ph = self._all_observations_ph
            # print('q all')
        q_targets = self.qf.output_for(o_ph, actions, reuse=tf.AUTO_REUSE)  # N
        assert_shape(q_targets, [None])
        pg_loss = -tf.reduce_mean(q_targets)
        if raw_actions is not None:
            print('raw reg')
            pg_loss += tf.reduce_mean(tf.square(raw_actions)) * 1e-3
        if not self.SGA:
            with tf.variable_scope('policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
                if self._train_policy:
                    # optimizer = tf.train.AdamOptimizer(self._policy_lr)
                    # pg_training_op = optimizer.minimize(
                    #     loss=pg_loss,
                    #     var_list=self.policy.get_params_internal())
                    optimizer = tf.train.AdamOptimizer(self._policy_lr)
                    pg_training_op = U.minimize_and_clip(optimizer, pg_loss, self.policy.get_params_internal(),
                                                         self.grad_norm_clipping)
                    self._training_ops.append(pg_training_op)
        self._pg_loss = pg_loss

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target functions."""
        if not self._train_qf:
            return

        source_q_params = self.qf.get_params_internal()
        target_q_params = self.target_qf.get_params_internal()
        source_p_params = self.policy.get_params_internal()
        target_p_params = self.target_policy.get_params_internal()

        self._target_ops = [
                               tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_q_params, source_q_params)
                           ] + [
                               tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_p_params, source_p_params)
                           ]

    # TODO: do not pass, policy, and pool to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.pool)

    @overrides
    def _init_training(self):
        # source_q_params = self.qf.get_params_internal()
        # target_q_params = self.target_qf.get_params_internal()
        # source_p_params = self.policy.get_params_internal()
        # target_p_params = self.target_policy.get_params_internal()
        # target_ops = [
        #                  tf.assign(target,  source)
        #                  for target, source in zip(target_q_params, source_q_params)
        #              ] + [
        #                  tf.assign(target,  source)
        #                  for target, source in zip(target_p_params, source_p_params)
        #              ]
        # self._sess.run(target_ops)
        pass

    @overrides
    def _do_training(self, iteration, batch):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)
        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)
        # self.log_diagnostics(batch)

    def _get_feed_dict(self, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""
        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_pl: batch['actions'],
            self._next_actions_ph: batch['next_actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch['rewards'],
            self._terminals_pl: batch['terminals']
        }
        if self.joint:
            feeds[self._opponent_actions_pl] = batch['opponent_actions']
            feeds[self._opponent_next_actions_ph] = batch['opponent_next_actions']
        if self.opponent_modelling:
            feeds[self._recent_opponent_observations_ph] = batch['recent_opponent_observations']
            feeds[self._recent_opponent_actions_pl] = batch['recent_opponent_actions']
        if self.name == 'MADDPG':
            # print('feeded all')
            feeds.update({
                self._all_observations_ph: batch['all_observations'],
                self._all_next_observations_ph: batch['all_next_observations'],
                self._opponent_current_actions_pl: batch['opponent_current_actions']
            })

        return feeds

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information.
        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.
        Also call the `draw` method of the plotter, if plotter is defined.
        """

        feeds = self._get_feed_dict(batch)
        qf, bellman_residual = self._sess.run(
            [self._q_values, self._bellman_residual], feeds)

        logger.record_tabular('qf-avg-agent-{}'.format(self._agent_id), np.mean(qf))
        logger.record_tabular('qf-std-agent-{}'.format(self._agent_id), np.std(qf))
        logger.record_tabular('mean-sq-bellman-error-agent-{}'.format(self._agent_id), bellman_residual)

        # self.policy.log_diagnostics(batch)
        # if self.plotter:
        #     self.plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.
        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """

        state = {
            'epoch_agent_{}'.format(self._agent_id): epoch,
            'policy_agent_{}'.format(self._agent_id): self.policy,
            'qf_agent_{}'.format(self._agent_id): self.qf,
            'env_agent_{}'.format(self._agent_id): self.env,
        }

        if self._save_full_state:
            state.update({'replay_buffer_agent_{}'.format(self._agent_id): self.pool})

        return state
