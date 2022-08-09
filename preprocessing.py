import numpy as np


def make_one_hot(x):
    unique = np.unique(x)
    n, m = len(unique), len(x)

    def to_one_hot(x_i):
        oh = np.zeros(n)
        oh[np.argmax(unique == x_i)] = 1
        return oh

    return np.apply_along_axis(to_one_hot, 0, x.reshape(1, -1))
