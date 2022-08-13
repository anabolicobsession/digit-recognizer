import numpy as np


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A


def d_softmax(A, dA=None):
    if dA is None:
        dZ = - np.expand_dims(A, 1) * np.expand_dims(A, 0)
        diagonal = A * (1 - A)

        for i in range(A.shape[1]):
            np.fill_diagonal(dZ[:, :, i], diagonal[:, i])
    else:
        dZ = A * (dA - np.sum(dA * A, axis=0, keepdims=True))

    return dZ


def relu(Z):
    return np.maximum(Z, 0)


def d_relu(A, dA=None):
    return (A > 0).astype(float) if dA is None else dA * (A > 0).astype(float)
