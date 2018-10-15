import numpy as np
# from .base_game import BaseEnv


class MatrixGame():
    def __init__(self, game, agent_num, action_num, payoff=None):
        self.game = game
        self.agent_num = agent_num
        self.action_num = action_num
        self.action_space = np.array([range(action_num)] * self.agent_num)
        self.state_space = np.array([range(1)] * self.agent_num)
        self.t = 0
        self.numplots = 0
        if payoff is not None:
            payoff = np.array(payoff)
            assert payoff.shape == tuple([agent_num] + [action_num] * agent_num)
            self.payoff = payoff
        if payoff is None:
            self.payoff = np.zeros(tuple([agent_num] + [action_num] * agent_num))

        if game == 'coordination_0_0':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[1,-1],
                           [-1,-1]]
            self.payoff[1]=[[1,-1],
                           [-1,-1]]

        if game == 'coordination_0_1':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[1, 0],
                           [0, 1]]
            self.payoff[1]=[[1, 0],
                           [0, 1]]

        if game == 'coordination_same_action_with_preference':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[2, 0],
                           [0, 1]]
            self.payoff[1]=[[1, 0],
                           [0, 2]]

#     '''payoff tabular of zero-sum game scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)'''
        elif game == 'zero_sum_nash_0_1':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[5,2],
                            [-1,6]]
            self.payoff[1]=[[-5,-2],
                            [1,-6]]

#     '''payoff tabular of zero-sumgame scenario. matching pennies'''
        elif game == 'matching_pennies':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[1,-1],
                           [-1,1]]
            self.payoff[1]=[[-1,1],
                           [1,-1]]

        elif game == 'matching_pennies_3':
            assert self.agent_num == 3
            assert self.action_num == 2
            self.payoff[0]=[
                            [ [1,-1],
                              [-1,1] ],
                            [ [1, -1],
                             [-1, 1]]
                            ]
            self.payoff[1]=[
                            [ [1,-1],
                              [1,-1] ],
                            [[-1, 1],
                             [-1, 1]]
                            ]
            self.payoff[2] = [
                            [[-1, -1],
                             [1, 1]],
                            [[1, 1],
                             [-1, -1]]
                            ]

        elif game =='prison':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0]=[[-1,-3],
                           [0,-2]]
            self.payoff[1]=[[-1,0],
                           [-3,-2]]

        elif game == 'wolf_05_05':
            assert self.agent_num == 2
            assert self.action_num == 2
            self.payoff[0] = [[0, 3],
                             [1, 2]]
            self.payoff[1] = [[3, 2],
                             [0, 1]]
            # \alpha, \beta = 0, 0.9, nash is 0.5 0.5
            # Q tables given, matian best response, learn a nash e.

        elif game == 'rock_paper_scissors':
            assert self.agent_num == 2
            assert self.action_num == 3
            self.payoff[0] = [[0, -1, 1],
                              [1, 0, -1],
                              [-1, 1, 0]
                              ]
            self.payoff[1] = [[0, 1, -1],
                              [-1, 0, 1],
                              [1, -1, 0]
                              ]

        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'rock_paper_scissors': {'agent_num': 2, 'action_num': 3},
            'wolf_05_05': {'agent_num': 2, 'action_num': 2},
            'prison': {'agent_num': 2, 'action_num': 2},
            'matching_pennies_3': {'agent_num': 3, 'action_num': 2},
            'matching_pennies': {'agent_num': 2, 'action_num': 2},
            'zero_sum_nash_0_1': {'agent_num': 2, 'action_num': 2},
            'coordination_same_action_with_preference': {'agent_num': 2, 'action_num': 2},
            'coordination_0_0': {'agent_num': 2, 'action_num': 2},
        }

    def step(self, actions):
        assert len(actions) == self.agent_num
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            assert actions[i] in range(self.action_num)
            reward_n[i] = self.payoff[i][tuple(actions)]
        self.rewards = reward_n
        # print(actions, reward_n)
        state_n = np.array([0] * self.agent_num)
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array([0] * self.agent_num)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_joint_reward(self):
        return self.rewards

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Number of Action \n'.format(self.game, self.agent_num, self.action_num)
        content += 'Payoff Matrixs:\n\n'
        for i in range(self.agent_num):
            content += 'Agent {}, Payoff:\n {} \n\n'.format(i+1, str(self.payoff[i]))
        return content


if __name__ == '__main__':
    print(MatrixGame.get_game_list())
    game = MatrixGame('matching_pennies_3', agent_num=3, action_num=2)
    print(game)