import numpy as np


def make_one_hot(x, n_classes):
    def to_one_hot(n):
        oh = np.zeros(n_classes)
        oh[n] = 1
        return oh

    return np.array(list(map(to_one_hot, list(np.squeeze(x))))).T