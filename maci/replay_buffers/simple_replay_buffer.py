import numpy as np

from maci.core.serializable import Serializable

from maci.replay_buffers.replay_buffer import ReplayBuffer
from maci.environments.env_spec import MAEnvSpec


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size, joint=False, agent_id=None):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)
        self.joint = joint
        self._env_spec = env_spec
        self.agent_id = agent_id
        if isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            self._action_dim = env_spec.action_space[agent_id].flat_dim
            if joint:
                self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
                print(agent_id, self._opponent_action_dim )
                self._opponent_actions = np.zeros((max_replay_buffer_size, self._opponent_action_dim ))
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim

        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        if 'opponent_action' in kwargs:
            # print('added')
            # todo: fix adding opponent action
            self._opponent_actions[self._top] = kwargs['opponent_action']
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        self.indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[self.indices],
            actions=self._actions[self.indices],
            rewards=self._rewards[self.indices],
            terminals=self._terminals[self.indices],
            next_observations=self._next_obs[self.indices],
        )
        if self.joint:
            batch['opponent_actions'] = self._opponent_actions[self.indices]
        return batch

    def random_batch_by_indices(self, indices):
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        if self.joint:
            batch['opponent_actions'] = self._opponent_actions[indices]
        return batch

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(SimpleReplayBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            top=self._top,
            size=self._size,
        ))
        if self.joint:
            d.update((dict(o_a=self._opponent_actions.tobytes())))
        return d

    def __setstate__(self, d):
        super(SimpleReplayBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._top = d['top']
        self._size = d['size']
        if self.joint:
            self._opponent_actions = np.fromstring(d['o_a']).reshape(self._max_buffer_size, -1)
