import numpy as np


def cross_entropy(Y, A):
    return - np.sum(Y * np.log(A)) / A.shape[1]
