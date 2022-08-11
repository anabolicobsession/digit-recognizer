import numpy as np


def add_bias_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])
