import numpy as np


def accuracy(Y, Y_pred):
    return np.sum(Y_pred == Y) / Y.size
