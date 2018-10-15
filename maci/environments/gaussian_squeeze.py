import numpy as np
from maci.misc.space import MADiscrete, MABox
from maci.environments.env_spec import MAEnvSpec


from rllab.core.serializable import Serializable

class GMSD(Serializable):
    def __init__(self, agent_num, mus=[0., 400.], sigmas=[100., 200.], action_low=0, action_high=10):
        Serializable.quick_init(self, locals())
        self.game_name = 'gaussian_squeeze'
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.agent_num = agent_num
        self.action_range = [action_low, action_high]
        lows = np.array([np.array([action_low]) for _ in range(self.agent_num)])
        highs = np.array([np.array([action_high]) for _ in range(self.agent_num)])
        self.action_spaces = MABox(lows=lows, highs=highs)
        self.observation_spaces = MADiscrete([1] * self.agent_num)
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

    def gaussian(self, x_vec):
        x = np.array([np.sum(x_vec)] * len(self.mus))
        return x.dot(np.exp(-np.square(x - self.mus) / np.square(self.sigmas))) / len(self.mus)

    def step(self, actions):
        assert len(actions) == self.agent_num
        reward_n = np.array([self.gaussian(actions)] * self.agent_num)
        self.rewards = reward_n
        print(reward_n)
        state_n = np.array(list([[1. * i] for i in range(self.agent_num)]))
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array(list([[1. * i] for i in range(self.agent_num)]))

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_joint_reward(self):
        return self.rewards

    def terminate(self):
        pass

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game, self.agent_num, self.action_range)
        return content
