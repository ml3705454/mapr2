from builtins import *
import numpy as np
EPS = 1e-6


def projection(PI, threshold=0):
    #print(PI)
    # 1- the unit vector is perpendicular to the 1 surface. Using this along with the
    # 	passed point P0 we get a parametric line equation:
    #   P = P0 + t * I, where t is the parameter and I is the unit vector.
    # 	the unit of projection has \sum P = 1 = \sum P0 + nt, where n is the dimension of P0
    # 	hence the point of projection, P' = P0 + ( (1 - \sum P0) / n ) I
    # * compute sum
    t = sum(PI)
    #print(t)
    # * compute t
    t = (1.0 - t) / len(PI)

    # * compute P'
    for i in range(len(PI)):
        PI[i] += t

    # 2- if forall p in P', p >=0 (and consequently <=1), we found the point.
    #	other wise, pick a negative dimension d, make it equal zero while decrementing
    #	other non zero dimensions. repeat until no negatives remain.
    done = False
    while not done:
        # comulate negative dimensions
        # and count positive ones. note that there must be at least
        # one positive dimension
        n = 0
        excess = 0
        for i in range(len(PI)):
            if PI[i] < threshold:
                excess += threshold-PI[i]
                PI[i] = threshold
            elif PI[i] > threshold:
                n += 1

        # none negative? then done
        if excess == 0:
            done = True
        else:
            # otherwise decrement by equal steps
            for i in range(len(PI)):
                if PI[i] > threshold:
                    PI[i] -= excess / n
    #print(PI)
    return PI


def makehash():
    import collections
    return collections.defaultdict(makehash)


def sigmoid(x, derivative=False):
  return x * (1 - x) if derivative else np.clip(1./(1.+np.exp(-x)),EPS, 1-EPS)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def chain_files(file_names):
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                yield line


def drange(start=0.0, stop=1.0, step=0.1):
    eps = 1.0e-6
    r = start
    while r < stop + eps if stop > start else r > stop - eps:
        yield min(max(min(start, stop), r), max(start, stop))
        r += step


def pv(*args, **kwargs):
    import sys
    import inspect
    import pprint

    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stdout if 'stdout' in kwargs and kwargs['stdout'] \
            else sys.stderr

        print('%s%s: %s' % (prefix, name, pprint.pformat(val)), file=iostream)


def weighted_mean(samples, weights):
    return sum(x * w for x, w in zip(samples, weights)) / sum(weights) \
        if sum(weights) > 0.0 else 0.0


def mean(samples):
    return sum(samples) / len(samples) if len(samples) else 0.0


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def forward(*args):
    print('\t'.join(str(i) for i in args))


def random_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def minmax(low, x, high):
    return min(max(low, x), high)

def timeit(func):
    import functools

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        import time

        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} s'.format(
            func.__name__, elapsedTime))
    return newfunc