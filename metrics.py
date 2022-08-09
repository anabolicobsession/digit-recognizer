import numpy as np


def accuracy(y, pred):
    return np.sum(pred == y) / len(y)