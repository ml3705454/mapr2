""" Gaussian mixture policy. """
from maci.environments.env_spec import MAEnvSpec
from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from maci.misc.overrides import overrides
from maci.misc import logger
from maci.core.serializable import Serializable

from maci.distributions import Normal
from maci.policies import NNPolicy
from maci.misc import tf_utils

EPS = 1e-6

class GaussianPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, reparameterize=True, name='gaussian_policy',
                 joint=False,
                 opponent_policy=False,
                 agent_id=None):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())

        if isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
                if opponent_policy:
                    print('opponent_policy',opponent_policy)
                    self._action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim

        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim

        self._hidden_layers = hidden_layer_sizes
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg

        self.name = name + '_agent_{}'.format(agent_id)
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + self.name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, mu_latents=False,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False, n_action_samples=1, with_dist=False, return_mu=False,return_reg=False):
        name = name or self.name

        n_state_samples = tf.shape(observations)[0]
        latent_shape = (n_state_samples, self._action_dim)
        # latents = tf.random_normal(latent_shape)

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._action_dim,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        raw_actions = distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if return_mu:
            return distribution.mu_t

        if with_dist:
            return distribution

        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            log_pis = distribution.log_p_t
            if self._squash:
                log_pis -= self._squash_correction(raw_actions)
            if not return_reg:
                return actions, log_pis
            return actions, log_pis, distribution.reg_loss_t

        return actions
    
    def log_pis_for(self, actions):
        raw_actions = actions
        if self._squash:
           raw_actions = tf.atanh(actions) 
           log_pis = self._distribution.log_prob(raw_actions)
           log_pis -= self._squash_correction(raw_actions)
           return log_pis
        return self._distribution.log_prob(raw_actions)

    def build(self):
        self._observation_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._observation_dim),
            name='observations',
        )

        n_state_samples = tf.shape(self._observation_ph)[0]
        latent_shape = (n_state_samples, self._action_dim)
        # latents = tf.random_normal(latent_shape)


        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._action_dim,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observation_ph,),
                reg=self._reg,
            )

        raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._action = tf.tanh(raw_actions) if self._squash else raw_actions

    def get_mu(self, observations):
        feed_dict = {self._observation_ph: observations}

        # TODO.code_consolidation: these shapes should be double checked
        # for case where `observations.shape[0] > 1`
        mu = tf.get_default_session().run(
            self.distribution.mu_t, feed_dict)  # 1 x Da
        if self._squash:
            mu = np.tanh(mu)

        return mu

    @overrides
    def get_actions(self, observations):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the 
        observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        if self._is_deterministic: # Handle the deterministic case separately.

            feed_dict = {self._observation_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict)  # 1 x Da
            if self._squash:
                mu = np.tanh(mu)

            return mu

        return super(GaussianPolicy, self).get_actions(observations)

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observation_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mu, log_sig, log_pi = sess.run(
            (
                self.distribution.mu_t,
                self.distribution.log_sig_t,
                self.distribution.log_p_t,
            ),
            feeds
        )

        logger.record_tabular('policy-mus-mean', np.mean(mu))
        logger.record_tabular('policy-mus-min', np.min(mu))
        logger.record_tabular('policy-mus-max', np.max(mu))
        logger.record_tabular('policy-mus-std', np.std(mu))
        logger.record_tabular('log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('log-sigs-min', np.min(log_sig))
        logger.record_tabular('log-sigs-max', np.max(log_sig))
        logger.record_tabular('log-sigs-std', np.std(log_sig))
        logger.record_tabular('log-pi-mean', np.mean(log_pi))
        logger.record_tabular('log-pi-max', np.max(log_pi))
        logger.record_tabular('log-pi-min', np.min(log_pi))
        logger.record_tabular('log-pi-std', np.std(log_pi))


class GaussianConditionalPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, cond_policy, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, reparameterize=True, name='gaussian_cond_policy',
                 joint=False,
                 opponent_policy=False,
                 agent_id=None):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())

        if isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
                if opponent_policy:
                    print('opponent_policy', opponent_policy)
                    self._action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim

        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim
        self.cond_policy = cond_policy
        self._hidden_layers = hidden_layer_sizes
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg
        self._observation_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._observation_dim),
            name='observations',
        )
        self.name = name + '_agent_{}'.format(agent_id)
        self.build()

        self._scope_name = (
                tf.get_variable_scope().name + "/" + self.name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False, opponent_actions=None, with_dist=False, return_mu=False, return_reg=False):
        name = name or self.name

        n_state_samples = tf.shape(observations)[0]
        latent_shape = (n_state_samples, self._action_dim)
        # latents = tf.random_normal(latent_shape)


        opponent_actions_ = tf.stop_gradient(self.cond_policy.actions_for(observations, return_mu))

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._action_dim,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations, opponent_actions_),
                reg=self._reg
            )
            if opponent_actions is not None:
                distribution = Normal(
                    hidden_layers_sizes=self._hidden_layers,
                    Dx=self._action_dim,
                    reparameterize=self._reparameterize,
                    cond_t_lst=(observations, opponent_actions),
                    reg=self._reg
                )

        raw_actions = distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions
        if return_mu:
            return distribution.mu_t
        if with_dist:
            return distribution
        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            log_pis = distribution.log_p_t
            if self._squash:
                log_pis -= self._squash_correction(raw_actions)
            if not return_reg:
                return actions, log_pis
            return actions, log_pis, distribution.reg_loss_t

        return actions

    def log_pis_for(self, actions):
        raw_actions = actions
        if self._squash:
            raw_actions = tf.atanh(actions)
            log_pis = self._distribution.log_prob(raw_actions)
            log_pis = log_pis - self._squash_correction(raw_actions)
            return log_pis
        return self._distribution.log_prob(raw_actions)

    def build(self):
        n_state_samples = tf.shape(self._observation_ph)[0]
        latent_shape = (n_state_samples, self._action_dim)
        # latents = tf.random_normal(latent_shape)
        opponent_actions_ = tf.stop_gradient(self.cond_policy.actions_for(self._observation_ph, True))
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._action_dim,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observation_ph, opponent_actions_),
                reg=self._reg,
            )

        raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._action = tf.tanh(raw_actions) if self._squash else raw_actions

    def get_mu(self, observations):
        feed_dict = {self._observation_ph: observations}

        mu = tf.get_default_session().run(
            self.distribution.mu_t, feed_dict)  # 1 x Da
        if self._squash:
            mu = np.tanh(mu)

        return mu

    @overrides
    def get_actions(self, observations):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the
        observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        if self._is_deterministic:  # Handle the deterministic case separately.

            feed_dict = {self._observation_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict)  # 1 x Da
            if self._squash:
                mu = np.tanh(mu)

            return mu

        return super(GaussianConditionalPolicy, self).get_actions(observations)

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observation_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mu, log_sig, log_pi = sess.run(
            (
                self.distribution.mu_t,
                self.distribution.log_sig_t,
                self.distribution.log_p_t,
            ),
            feeds
        )

        logger.record_tabular('policy-mus-mean', np.mean(mu))
        logger.record_tabular('policy-mus-min', np.min(mu))
        logger.record_tabular('policy-mus-max', np.max(mu))
        logger.record_tabular('policy-mus-std', np.std(mu))
        logger.record_tabular('log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('log-sigs-min', np.min(log_sig))
        logger.record_tabular('log-sigs-max', np.max(log_sig))
        logger.record_tabular('log-sigs-std', np.std(log_sig))
        logger.record_tabular('log-pi-mean', np.mean(log_pi))
        logger.record_tabular('log-pi-max', np.max(log_pi))
        logger.record_tabular('log-pi-min', np.min(log_pi))
        logger.record_tabular('log-pi-std', np.std(log_pi))
