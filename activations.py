import numpy as np


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A
