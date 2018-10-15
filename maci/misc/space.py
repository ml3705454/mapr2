# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)
from rllab.spaces import Discrete, Box, Product
from functools import reduce

import numpy as np


class Space(object):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def flatten(self, x):
        raise NotImplementedError

    def unflatten(self, x):
        raise NotImplementedError

    def flatten_n(self, xs):
        raise NotImplementedError

    def unflatten_n(self, xs):
        raise NotImplementedError

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        raise NotImplementedError


class MASpace(Space):
    """
    Provides a classification multi-agent state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        return np.array([agent_space.sample() for agent_space in self.agent_spaces])

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        if len(x) != self.agent_num:
            return False
        for x_i, space in zip(x, self.agent_spaces):
            if not space.contains(x_i):
                return False
        return True

    def flatten(self, x):
        assert len(x) == self.agent_num
        return np.array([space.flatten(x_i) for x_i, space in zip(x, self.agent_spaces)])

    def unflatten(self, x):
        assert len(x) == self.agent_num
        return np.array([space.unflatten(x_i) for x_i, space in zip(x, self.agent_spaces)])

    def flatten_n(self, xs):
        assert len(xs) == self.agent_num
        return np.array([space.unflatten(xs_i) for xs_i, space in zip(xs, self.agent_spaces)])

    def unflatten_n(self, xs):
        assert len(xs) == self.agent_num
        return np.array([space.unflatten_n(xs_i) for xs_i, space in zip(xs, self.agent_spaces)])

    def __getitem__(self, i):
        assert (i >= 0) and (i < self.agent_num)
        return self.agent_spaces[i]

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        flat_dims = 0
        for space in self.agent_spaces:
            flat_dims += space.flat_dim
        return flat_dims

    def opponent_flat_dim(self, i):
        return self.flat_dim - self[i].flat_dim

    def __eq__(self, other):
        if self.agent_num != other.agent_num:
            return False
        for agent_space, other_space in zip(self.agent_spaces, other.agent_spaces):
            if not agent_space == other_space:
                return False
        return True

    def __repr__(self):
        return '\n'.join(["Agent {}, {}".format(i, space.__repr__()) for i, space in zip(range(self.agent_num), self.agent_spaces)])


class MADiscrete(MASpace):
    """
    Provides a classification multi-agent state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def __init__(self, ns):
        self.agent_num = len(ns)
        self.agent_spaces = np.array([Discrete(n) for n in ns])


class MABox(MASpace):
    """
    Provides a classification multi-agent state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def __init__(self, lows, highs, shapes=None):
        if shapes is None:
            assert len(lows) == len(highs)
            self.agent_num = len(lows)
            self.agent_spaces = np.array([Box(low, high) for low, high in zip(lows, highs)])
        else:
            assert len(lows) == len(highs) == len(shapes)
            self.agent_num = len(lows)
            self.agent_spaces = np.array([Box(low, high, shape) for low, high, shape in zip(lows, highs, shapes)])


    @property
    def shape(self):
        return tuple((space.shape for space in self.agent_spaces))

    @property
    def bounds(self):
        return tuple((space.bounds for space in self.agent_spaces))


class MultiDiscrete(Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def flatten(self, x):
        pass

    def unflatten(self, x):
        pass

    def flatten_n(self, xs):
        pass

    def unflatten_n(self, xs):
        pass

    @property
    def flat_dim(self):
        """
        The dimension of the flattened vector of the tensor representation
        """
        self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)
