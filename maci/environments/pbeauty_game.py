import numpy as np
from maci.misc.space import MADiscrete, MABox
from maci.environments.env_spec import MAEnvSpec


from maci.core.serializable import Serializable


class PBeautyGame(Serializable):
    def __init__(self, agent_num, game_name='pbeauty', p=0.67, reward_type='abs', action_low=-1., action_high=1.):
        Serializable.quick_init(self, locals())
        self.agent_num = agent_num
        self.p = p
        self.game_name = game_name
        self.reward_type = reward_type
        self.action_range = [action_low, action_high]
        lows = np.array([np.array([action_low]) for _ in range(self.agent_num)])
        highs = np.array([np.array([action_high]) for _ in range(self.agent_num)])
        self.action_spaces = MABox(lows=lows, highs=highs)
        self.observation_spaces = MADiscrete([1] * self.agent_num)
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.t = 0
        self.rewards = np.zeros((self.agent_num,))

    def step(self, actions):
        assert len(actions) == self.agent_num
        actions = np.array(actions).reshape((self.agent_num,))
        reward_n = np.zeros((self.agent_num,))
        if self.game_name == 'pbeauty':
            actions = (actions + 1.) * 50.
            action_mean = np.mean(actions) * self.p
            deviation_abs = np.abs(actions - action_mean)
            print(actions, np.mean(actions))
            if self.reward_type == 'abs':
                reward_n = -deviation_abs
            elif self.reward_type == 'one':
                i = np.argmin(deviation_abs)
                reward_n[i] = 1.
            elif self.reward_type == 'sqrt':
                reward_n = -np.sqrt(deviation_abs)
            elif self.reward_type == 'square':
                reward_n = -np.square(deviation_abs)
        elif self.game_name == 'entry':
            actions = (actions + 1.) / 2.
            # think about it?

        print(reward_n, np.mean(reward_n))
        state_n = np.array(list([[0. * i] for i in range(self.agent_num)]))
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array(list([[0. * i] for i in range(self.agent_num)]))

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