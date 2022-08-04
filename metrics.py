import numpy as np


def accuracy(pred, y):
    return np.sum(pred == y) / y.shape[-1]