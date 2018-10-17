from maci.environments import MatrixGame
from maci.learners.tabular import *

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    agent_num = 2
    action_num = 2
    agents = []

    # seed = 0
    # np.random.seed(seed)

    # matching_pennies
    # wolf_05_05
    # coordination_0_1
    game_name = 'wolf_05_05'
    iteration_num = 1000
    # pis = [{0: np.array([0., 1.])}, {0: np.array([0.9, 0.1])}]

    env = MatrixGame(game=game_name, agent_num=agent_num, action_num=action_num)

    for i in range(agent_num):
        agent = RRQAgent(i, action_num, env)
        agents.append(agent)

    exploration = False
    rewards_his = []
    for i in range(iteration_num):
        state_n = env.reset()
        actions = np.array([agent.act(state, exploration, env) for state, agent in zip(state_n, agents)])
        state_prime_n, rewards, _, _ = env.step(actions)
        rewards_his.append(rewards)
        for j, (state, reward, state_prime, agent) in enumerate(zip(state_n, rewards, state_prime_n, agents)):
            agent.update(state, actions[j], actions[1-j], reward, state_prime, env, done=True)
    rewards_his = np.array(rewards_his)
    print(agents[0].pi, agents[1].pi)
    history_pi_0 = [p[0][0] for p in agents[0].pi_history]
    history_pi_1 = [p[0][0] for p in agents[1].pi_history]
    pis = []
    for p1, p2 in zip(agents[0].pi_history, agents[1].pi_history):
        pis.append([p1[0][0], p2[0][0]])

    cmap = plt.get_cmap('viridis')
    colors = range(len(history_pi_1))
    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(211)


    scatter = ax.scatter(history_pi_0, history_pi_1, c=colors, s=1)
    ax.scatter(0.5, 0.5, c='r', s=10., marker='*')
    colorbar = fig.colorbar(scatter, ax=ax)

    ax.set_ylabel("Policy of Player 2")
    ax.set_xlabel("Policy of Player 1")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    ax = fig.add_subplot(212)

    ax.plot(history_pi_0)
    ax.plot(history_pi_1)

    plt.tight_layout()
    plt.show()