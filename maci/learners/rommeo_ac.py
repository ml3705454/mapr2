import numpy as np
import tensorflow as tf

from maci.misc import logger
from maci.misc.overrides import overrides

from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.misc import tf_utils
import tensorflow_probability as tfp
from maci.distributions.normal import Normal
import maci.misc.tf_utils as U

from .base import MARLAlgorithm

EPS = 1e-6


def squash_correction(actions):
    return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class ROMMEO(MARLAlgorithm):
    def __init__(
            self,
            base_kwargs,
            agent_id,
            env,
            pool,
            joint_qf,
            target_joint_qf,
            policy,
            opponent_policy,
            target_policy=None,
            plotter=None,
            policy_lr=1E-3,
            qf_lr=1E-3,
            tau=0.01,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=1,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_policy=True,
            joint=False,
            joint_policy=False,
            opponent_action_range=None,
            opponent_action_range_normalize=True,
            opponent_modelling=True,
            M=100,
            alpha=0.5
    ):
        super(ROMMEO, self).__init__(**base_kwargs)

        self._env = env
        self._pool = pool
        self.joint_qf = joint_qf
        self.target_joint_qf = target_joint_qf
        self.policy = policy
        self.target_policy = target_policy
        self.opponent_policy = opponent_policy
        self.plotter = plotter
        self.joint = joint
        self.joint_policy = joint_policy
        self.opponent_action_range = opponent_action_range
        self.opponent_action_range_normalize = opponent_action_range_normalize
        self.opponent_modelling = opponent_modelling
        self._agent_id = agent_id
        self.M = M
        self.alpha = alpha
        self._hidden_layers = [M, M]

        self._tau = tau
        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._reward_scale = reward_scale

        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_policy = train_policy

        self._observation_dim = self.env.observation_spaces[self._agent_id].flat_dim
        self._all_observation_dim = self.env.observation_spaces.flat_dim
        self._action_dim = self.env.action_spaces[self._agent_id].flat_dim
        self._opponent_action_dim = self.env.action_spaces.opponent_flat_dim(self._agent_id)

        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_opponent_prior_update()
        self._create_opponent_p_update()
        self._create_q_update()
        self._create_p_update()
        self._create_target_ops()

        if use_saved_qf:
            saved_qf_params = self.joint_qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = self.policy.get_param_values()

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())

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

        self._all_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._all_observation_dim],
            name='all_observations_agent_{}'.format(self._agent_id))

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations_agent_{}'.format(self._agent_id))

        self._all_next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._all_observation_dim],
            name='all_next_observations_agent_{}'.format(self._agent_id))

        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='actions_agent_{}'.format(self._agent_id))
        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='next_actions_agent_{}'.format(self._agent_id))
        self._opponent_actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_actions_agent_{}'.format(self._agent_id))
        self._opponent_next_actions_ph = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_next_actions_agent_{}'.format(self._agent_id))
        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='rewards_agent_{}'.format(self._agent_id))
        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='terminals_agent_{}'.format(self._agent_id))
        self._annealing_pl = tf.placeholder(
            tf.float32, shape=[],
            name='annealing_agent_{}'.format(self._agent_id))

        if self.opponent_modelling:
            self._recent_opponent_observations_ph = tf.placeholder(
                tf.float32,shape=[None, self._observation_dim],
                name='recent_opponent_observations_agent_{}'.format(self._agent_id))
            self._recent_opponent_actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='recent_opponent_actions_agent_{}'.format(self._agent_id))
        # self._noise_pl = noise_pl

    def _get_opponent_prior(self, ph):
        self._opponent_prior_scope = 'opponent_prior_agent_{}'.format(self._agent_id)
        with tf.variable_scope(self._opponent_prior_scope, reuse=tf.AUTO_REUSE):
            prior = self._opponent_prior = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._opponent_action_dim,
                reparameterize=False,
                cond_t_lst=(ph,),
                reg=True
            )
        return prior

    def _create_opponent_prior_update(self):
        prior = self._get_opponent_prior(self._recent_opponent_observations_ph)
        raw_actions = tf.atanh(self._recent_opponent_actions_pl)
        log_pis = prior.dist.log_prob(raw_actions)
        log_pis = log_pis - squash_correction(raw_actions)
        loss = -tf.reduce_mean(log_pis) + prior.reg_loss_t
        vars = U.scope_vars(self._opponent_prior_scope)
        with tf.variable_scope('opponent_prior_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                prior_training_op = optimizer.minimize(
                        loss=loss,
                        var_list=vars)
                self._training_ops.append(prior_training_op)

    def _create_opponent_p_update(self):
        opponent_actions, opponent_actions_log_pis, reg_loss = self.opponent_policy.actions_for(
            observations=self._observations_ph,
            reuse=tf.AUTO_REUSE, with_log_pis=True, return_reg=True)
        assert_shape(opponent_actions, [None, self._opponent_action_dim])

        prior = self._get_opponent_prior(self._observations_ph)
        raw_actions = tf.atanh(opponent_actions)
        prior_log_pis = prior.dist.log_prob(raw_actions)
        prior_log_pis = prior_log_pis - squash_correction(raw_actions)

        actions, agent_log_pis = self.policy.actions_for(observations=self._observations_ph,
                                                         reuse=tf.AUTO_REUSE,
                                                         with_log_pis=True,
                                                         opponent_actions=opponent_actions)

        q_values = self.joint_qf.output_for(
            self._observations_ph, actions, opponent_actions, reuse=True)


        opponent_p_loss = tf.reduce_mean(opponent_actions_log_pis) - tf.reduce_mean(prior_log_pis) - tf.reduce_mean(q_values) + self._annealing_pl * agent_log_pis
        opponent_p_loss = opponent_p_loss + reg_loss
        with tf.variable_scope('opponent_policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                om_training_op = optimizer.minimize(
                    loss=opponent_p_loss,
                    var_list=self.opponent_policy.get_params_internal())
                self._training_ops.append(om_training_op)

    def _create_q_update(self):
        """Create a minimization operation for Q-function update."""
        opponent_actions, opponent_actions_log_pis = self.opponent_policy.actions_for(
            observations=self._next_observations_ph,
            reuse=tf.AUTO_REUSE, with_log_pis=True)
        assert_shape(opponent_actions, [None, self._opponent_action_dim])

        prior = self._get_opponent_prior(self._next_observations_ph)
        raw_actions = tf.atanh(opponent_actions)
        prior_log_pis = prior.dist.log_prob(raw_actions)
        prior_log_pis = prior_log_pis - squash_correction(raw_actions)

        actions, actions_log_pis = self.policy.actions_for(observations=self._next_observations_ph,
                                                           reuse=tf.AUTO_REUSE,
                                                           with_log_pis=True,
                                                           opponent_actions=opponent_actions)

        with tf.variable_scope('target_joint_q_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            q_value_targets = self.target_joint_qf.output_for(
                observations=self._next_observations_ph,
                actions=actions,
                opponent_actions=opponent_actions)
            q_value_targets = q_value_targets - self._annealing_pl * actions_log_pis - opponent_actions_log_pis + prior_log_pis
            assert_shape(q_value_targets, [None])

        self._q_values = self.joint_qf.output_for(
            self._observations_ph, self._actions_pl, self._opponent_actions_pl, reuse=True)
        assert_shape(self._q_values, [None])

        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * q_value_targets)
        assert_shape(ys, [None])

        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)
        with tf.variable_scope('target_joint_qf_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_qf:
                td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=bellman_residual, var_list=self.joint_qf.get_params_internal())
                self._training_ops.append(td_train_op)

        self._bellman_residual = bellman_residual


    def _create_p_update(self):
        """Create a minimization operation for policy update """
        opponent_actions, opponent_actions_log_pis = self.opponent_policy.actions_for(
            observations=self._observations_ph,
            reuse=tf.AUTO_REUSE, with_log_pis=True)

        assert_shape(opponent_actions, [None, self._opponent_action_dim])

        actions, actions_log_pis, reg_loss = self.policy.actions_for(observations=self._observations_ph,
                                                           reuse=tf.AUTO_REUSE,
                                                           with_log_pis=True,
                                                           opponent_actions=opponent_actions, return_reg=True)

        q_values = self.joint_qf.output_for(
            self._observations_ph, actions, opponent_actions, reuse=True)
        assert_shape(q_values, [None])

        pg_loss = self._annealing_pl * tf.reduce_mean(actions_log_pis) - tf.reduce_mean(q_values)
        pg_loss = pg_loss + reg_loss

        with tf.variable_scope('policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                pg_training_op = optimizer.minimize(
                    loss=pg_loss,
                    var_list=self.policy.get_params_internal())
                self._training_ops.append(pg_training_op)


    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        if not self._train_qf:
            return

        source_q_params = self.joint_qf.get_params_internal()
        target_q_params = self.target_joint_qf.get_params_internal()
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
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch, annealing=1.):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch, annealing)
        self._sess.run(self._training_ops, feed_dict)
        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch, annealing):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_pl: batch['actions'],
            self._opponent_actions_pl: batch['opponent_actions'],
            self._next_actions_ph: batch['next_actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch['rewards'],
            self._terminals_pl: batch['terminals'],
            self._annealing_pl: annealing
        }

        if self.opponent_modelling:
            feeds[self._recent_opponent_observations_ph] = batch['recent_opponent_observations']
            feeds[self._recent_opponent_actions_pl] = batch['recent_opponent_actions']


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

        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()

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