import numpy as np

from utils.constants import EPS


def cross_entropy(Y, A):
    return - np.sum(Y * np.log(np.maximum(A, EPS))) / A.shape[1]


def d_cross_entropy(Y, A):
    return - Y / (A * Y.shape[1])
