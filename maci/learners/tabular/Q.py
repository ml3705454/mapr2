from builtins import *

import random
from abc import abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np
from .base_tabular_learner import Agent, StationaryAgent
import maci.utils as utils
from copy import deepcopy


class BaseQAgent(Agent):
    def __init__(self, name, id_, action_num, env, alpha_decay_steps=10000., alpha=0.1, gamma=0.95, episilon=0.1, verbose=True, **kwargs):
        super().__init__(name, id_, action_num, env, **kwargs)
        self.episilon = episilon
        self.alpha_decay_steps = alpha_decay_steps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.Q = None
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.verbose = verbose
        self.pi_history = [deepcopy(self.pi)]

    def done(self, env):
        if self.verbose:
            utils.pv('self.full_name(game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

    # learning rate decay
    def step_decay(self):
        return self.alpha_decay_steps / (self.alpha_decay_steps + self.epoch)

    def act(self, s, exploration, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.action_num)
        else:
            if self.verbose:
                print('Agent {}--------------'.format(self.id_))
                print('Q of agent {}: state {}: {}'.format(self.id_, s, str(self.Q[s])))
                print('pi of agent {}: state {}: {}'.format(self.id_, s, self.pi[s]))
                print('payoff of agent {}: state {}: {}'.format(self.id_, s, self.R[s]))
                print('Agent {}--------------'.format(self.id_))
            return StationaryAgent.sample(self.pi[s])

    @abstractmethod
    def update(self, s, a, o, r, s2, env, done=False):
        pass

    @abstractmethod
    def update_policy(self, s, a, env):
        pass


class QAgent(BaseQAgent):
    def __init__(self, id_, action_num, env, **kwargs):
        super().__init__('q', id_, action_num, env, **kwargs)
        self.Q = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, self.action_num))
        self.count_R = defaultdict(partial(np.zeros, self.action_num))

    def done(self, env):
        self.R.clear()
        self.count_R.clear()
        super().done(env)

    def update(self, s, a, o, r, s2, env, done=False):
        self.count_R[s][a] += 1.0
        self.R[s][a] += (r - self.R[s][a]) / self.count_R[s][a]
        Q = self.Q[s]
        V = self.val(s2)
        decay_alpha = self.step_decay()
        if done:
            Q[a] = (1 - decay_alpha) * Q[a] + decay_alpha * r
        else:
            Q[a] = (1 - decay_alpha) * Q[a] + decay_alpha * (r + self.gamma * V - Q[a])
        print(self.epoch)
        self.update_policy(s, a, env)
        self.epoch += 1

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, env):
        Q = self.Q[s]
        self.pi[s] = (Q == np.max(Q)).astype(np.double)

class RRQAgent(QAgent):
    def __init__(self, id_, action_num, env, phi_type='count', **kwargs):
        super().__init__(id_, action_num, env, **kwargs)
        self.name = 'pr2q'
        self.phi_type = phi_type
        self.count_AOS = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.opponent_best_pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.action_num))
        self.pi_history = [deepcopy(self.pi)]
        self.opponent_best_pi_history = [deepcopy(self.opponent_best_pi)]
        self.Q = defaultdict(partial(np.random.rand, *(self.action_num, self.action_num)))
        self.Q_A = defaultdict(partial(np.random.rand, self.action_num))
        self.R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))
        self.count_R = defaultdict(partial(np.zeros, (self.action_num, self.action_num)))


    def update(self, s, a, o, r, s2, env, done=True):
        self.count_AOS[s][a][o] += 1.0
        decay_alpha = self.step_decay()
        if self.phi_type == 'count':
            count_sum = np.reshape(np.repeat(np.sum(self.count_AOS[s], 1), self.action_num), (self.action_num, self.action_num))
            self.opponent_best_pi[s] = self.count_AOS[s] / (count_sum + 0.01)
        elif self.phi_type == 'norm-exp':
            annealing_alpha = 1.
            baseline_Q_A = np.reshape(np.repeat(self.Q_A[s], self.action_num), (self.action_num, self.action_num))
            advantage = annealing_alpha * self.Q[s] - annealing_alpha * baseline_Q_A
            self.opponent_best_pi[s] = utils.softmax(advantage)

        self.count_R[s][a][o] += 1.0
        self.R[s][a][o] += (r - self.R[s][a][o]) / self.count_R[s][a][o]
        Q = self.Q[s]
        V = self.val(s2)
        if done:
            Q[a][o] = (1 - decay_alpha) * Q[a][o] + decay_alpha * r
            self.Q_A[s][a] = (1 - decay_alpha) * self.Q_A[s][a] + decay_alpha * r
        else:
            Q[a][o] = (1 - decay_alpha) * Q[a][o] + decay_alpha * (r + self.gamma * V)
            self.Q_A[s][a] = (1 - decay_alpha) * self.Q_A[s][a] + decay_alpha * (r + self.gamma * V)
        print(self.epoch)
        self.update_policy(s, a, env)
        self.epoch += 1

    def val(self, s):
        return np.max(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]),1))

    def update_policy(self, s, a, game):
        self.pi[s] = utils.softmax(np.sum(np.multiply(self.Q[s], self.opponent_best_pi[s]), 1))
        self.pi_history.append(deepcopy(self.pi))
        self.opponent_best_pi_history.append(deepcopy(self.opponent_best_pi))
        print('opponent pi of {}: {}'.format(self.id_, self.opponent_best_pi))