from builtins import *

from abc import ABCMeta, abstractmethod
from maci.utils import timeit
import time
import numpy as np



import gym
# from gym import error, spaces, utils
# from gym.utils import seeding

class BaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'ascii']
    }




#
# class Game(object):
#     __metaclass__ = ABCMeta
#
#     def __init__(self, name, gamma, max_steps):
#         self.name = name
#         self.gamma = gamma
#         self.max_steps = max_steps
#         self.t = 0
#         self.players = {}
#         self.state = None
#         self.new_episode = True
#         self.verbose = False
#         self.animation = False
#         self.numplots = 0
#         self.wins = np.zeros(2, dtype=np.int)
#
#     def add_player(self, i, player):
#         self.players[i] = player
#
#     def configuration(self):
#         return '{}({}, {})'.format(
#             self.name, self.players[0].name, self.players[1].name)
#
#     def report(self):
#         episodes = np.sum(self.wins)
#         print('step: {}, episode: {}'.format(self.t, episodes))
#         for i in range(2):
#             print('{}: win {} ({}%)'.format(
#                 i, self.wins[i], self.wins[i] / episodes * 100))
#
#     @abstractmethod
#     def numactions(self, id_):
#         pass
#
#     def done(self):
#         self.report()
#         for j, player in self.players.items():
#             player.done(self)
#
#     @timeit
#     def run(self, modes):
#         assert len(self.players) == 2
#         assert self.state is not None
#
#         print('configuration: {}'.format(self.configuration()))
#
#         for t in range(self.max_steps):
#             self.t = t
#             self.new_episode = False
#
#             if self.verbose:
#                 print('step: {}'.format(t))
#
#             actions = np.array(
#                 [self.players[0].act(self.state, modes[0], self),
#                  self.players[1].act(self.state, modes[1], self)],
#                 dtype=np.int8)
#             state_prime, rewards = self.simulate(actions)
#
#             for j, player in self.players.items():
#                 if modes[j]:
#                     player.update(
#                         self.state,
#                         actions[j],
#                         actions[1 - j],
#                         rewards[j],
#                         state_prime,
#                         self)
#
#             self.state = state_prime
#             if self.animation:
#                 time.sleep(0.25)
#
#     @abstractmethod
#     def simulate(self, actions):  # state, actions -> state_prime, reward
#         pass