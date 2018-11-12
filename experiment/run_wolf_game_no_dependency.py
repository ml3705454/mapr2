import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.tri import (
    Triangulation, UniformTriRefiner, CubicTriInterpolator)
from copy import deepcopy

FONTSIZE = 12

def plot_dynamics(history_pi_0, history_pi_1, pi_alpha_gradient_history, pi_beta_gradient_history, title=''):
    cmap = plt.get_cmap('viridis')
    colors = range(len(history_pi_1))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    scatter = ax.scatter(history_pi_0, history_pi_1, c=colors, s=1)
    ax.scatter(0.5, 0.5, c='r', s=10., marker='*')
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Iterations', rotation=270, fontsize=FONTSIZE)

    skip = slice(0, len(history_pi_0), 50)
    ax.quiver(history_pi_0[skip],
              history_pi_1[skip],
              pi_alpha_gradient_history[skip],
              pi_beta_gradient_history[skip],
              units='xy', scale=10., zorder=3, color='blue',
              width=0.007, headwidth=3., headlength=4.)

    ax.set_ylabel("Policy of Player 2", fontsize=FONTSIZE)
    ax.set_xlabel("Policy of Player 1", fontsize=FONTSIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=FONTSIZE+8)
    plt.tight_layout()
    plt.show()

def IGA(pi_alpha,
        pi_beta,
        payoff_0,
        payoff_1,
        u_alpha,
        u_beta,
        iteration=1000,
        lr=0.01):
    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.]
    pi_beta_gradient_history = [0.]
    for i in range(iteration):
        pi_alpha_gradient = (pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
        pi_beta_gradient = (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_next = pi_alpha + lr * pi_alpha_gradient
        pi_beta_next = pi_beta + lr * pi_beta_gradient
        pi_alpha = max(0., min(1., pi_alpha_next))
        pi_beta = max(0., min(1., pi_beta_next))
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
    return pi_alpha_history, \
           pi_beta_history, \
           pi_alpha_gradient_history, \
           pi_beta_gradient_history


def WoLF_IGA(pi_alpha,
             pi_beta, 
             payoff_0, 
             payoff_1,
             u_alpha,
             u_beta,
             iteration=1000,
             pi_alpha_nash=0.5, 
             pi_beta_nash=0.5,
             lr_min=0.01, 
             lr_max=0.04):
    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.]
    pi_beta_gradient_history = [0.]
    # V_nash_alpha = V(pi_alpha_nash, pi_beta_nash, payoff_0)
    # V_nash_beta = V(pi_alpha_nash, pi_beta_nash, payoff_1)
    # print(V_nash_alpha, V_nash_beta)
    for i in range(iteration):
        lr_alpha = lr_max
        lr_beta = lr_max
        if V(pi_alpha, pi_beta, payoff_0) > V(pi_alpha_nash, pi_beta, payoff_0):
            lr_alpha = lr_min
        if V(pi_alpha, pi_beta, payoff_1) > V(pi_alpha, pi_beta_nash, payoff_0):
            lr_beta = lr_min

        pi_alpha_gradient = (pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
        pi_beta_gradient = (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_next = pi_alpha + lr_alpha * pi_alpha_gradient
        pi_beta_next = pi_beta + lr_beta * pi_beta_gradient
        pi_alpha = max(0., min(1., pi_alpha_next))
        pi_beta = max(0., min(1., pi_beta_next))
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
    return pi_alpha_history, \
           pi_beta_history, \
           pi_alpha_gradient_history, \
           pi_beta_gradient_history


def IGA_PP(pi_alpha,
           pi_beta,
           payoff_0,
           payoff_1,
           u_alpha,
           u_beta,
           iteration=10000,
           lr=0.01,
           gamma=0.01,
           single=False):
    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.]
    pi_beta_gradient_history = [0.]
    for i in range(iteration):
        pi_beta_pp = pi_beta + gamma * (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
        pi_alpha_gradient = (pi_beta_pp * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
        pi_alpha_next = pi_alpha + lr * pi_alpha_gradient
        if not single:
            pi_alpha_pp = pi_alpha + gamma * (pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)])
            pi_beta_gradient = (pi_alpha_pp * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
            pi_beta_next = pi_beta + lr * pi_beta_gradient
        else:
            pi_beta_gradient = (pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)])
            pi_beta_next = pi_beta + lr * pi_beta_gradient
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)

        pi_alpha = max(0., min(1., pi_alpha_next))
        pi_beta = max(0., min(1., pi_beta_next))
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
    return pi_alpha_history, \
           pi_beta_history, \
           pi_alpha_gradient_history, \
           pi_beta_gradient_history


def V(alpha, beta, payoff):
    u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
    return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]


if __name__ == '__main__':

    game_name = 'wolf_05_05'

    pi_alpha = 0.
    pi_beta = 0.9
    pi_alpha_nash = 0.5
    pi_beta_nash = 0.5
    payoff_0 = np.array([[0, 3], [1, 2]])
    payoff_1 = np.array([[3, 2], [0, 1]])
    # \alpha, \beta = 0, 0.9, nash is 0.5 0.5

    u_alpha = payoff_0[(0, 0)] - payoff_0[(0, 1)] - payoff_0[(1, 0)] + payoff_0[(1, 1)]
    u_beta = payoff_1[(0, 0)] - payoff_1[(0, 1)] - payoff_1[(1, 0)] + payoff_1[(1, 1)]
    print(u_alpha, u_beta)

    agent = 'WPL'

    if agent == 'IGA':
        pi_alpha_history, \
        pi_beta_history, \
        pi_alpha_gradient_history, \
        pi_beta_gradient_history = IGA(pi_alpha,
                                       pi_beta,
                                       payoff_0,
                                       payoff_1,
                                       u_alpha,
                                       u_beta)
    elif agent == 'WoLF-IGA':
        pi_alpha_history, \
        pi_beta_history, \
        pi_alpha_gradient_history, \
        pi_beta_gradient_history = WoLF_IGA(pi_alpha,
                                            pi_beta,
                                            payoff_0,
                                            payoff_1,
                                            u_alpha,
                                            u_beta)


    elif agent == 'IGA-PP':
        pi_alpha_history, \
        pi_beta_history, \
        pi_alpha_gradient_history, \
        pi_beta_gradient_history = IGA_PP(pi_alpha,
                                          pi_beta,
                                          payoff_0,
                                          payoff_1,
                                          u_alpha,
                                          u_beta,
                                          single=False)


    plot_dynamics(pi_alpha_history,
                  pi_beta_history,
                  pi_alpha_gradient_history,
                  pi_beta_gradient_history,
                  agent)
    print('Done')

