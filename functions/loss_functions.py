import numpy as np


def cross_entropy(Y, A):
    return - np.sum(Y * np.log(A)) / A.shape[1]


def d_cross_entropy(Y, A):
    return - Y / (A * Y.shape[1])
