import numpy as np
from maci.misc.space import MADiscrete, MABox
from maci.environments.env_spec import MAEnvSpec


from rllab.core.serializable import Serializable


class DiscreteGame(Serializable):
    def __init__(self, game_name, agent_num, action_num=12):
        Serializable.quick_init(self, locals())
        self.game = game_name
        self.agent_num = agent_num
        self.action_num = action_num
        self.action_spaces = MADiscrete([action_num] * self.agent_num)
        self.observation_spaces = MADiscrete([1] * self.agent_num)
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.t = 0
        self.numplots = 0
        self.payoff = {}

        if self.game == 'lemonade':
            assert self.agent_num == 3
            def get_distance(a_n, i):
                assert len(a_n) == 3
                a_n_i = np.copy(a_n)
                a_n_i[0], a_n_i[i] = a_n_i[i], a_n_i[0]
                return np.abs(a_n_i[0] - a_n_i[1]) + np.abs(a_n_i[0] - a_n_i[2])
            self.payoff = lambda a_n, i: get_distance(a_n, i)

    @staticmethod
    def get_game_list():
        return {
            'lemonade': {'agent_num': 3, 'action_num': 21}
        }

    def step(self, actions):
        assert len(actions) == self.agent_num
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            # print('actions', actions)
            reward_n[i] = self.payoff(actions, i)
        self.rewards = reward_n
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


if __name__ == '__main__':
    print(DiscreteGame.get_game_list())
    game = DiscreteGame('zero_sum', agent_num=2)
    print(game)