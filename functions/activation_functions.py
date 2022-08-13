import numpy as np


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A


def d_softmax(dA, A):
    return A * (dA - np.sum(dA * A, axis=0, keepdims=True))


def relu(Z):
    return np.maximum(Z, 0)


def d_relu(dA, A):
    return (A > 0).astype(float) if dA is None else dA * (A > 0).astype(float)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def d_sigmoid(dA, A):
    return dA * A * (1 - A)


def tanh(Z):
    E, E_m = np.exp(Z), np.exp(-Z)
    return (E - E_m) / (E + E_m)


def d_tanh(dA, A):
    return dA * (1 - np.square(A))
