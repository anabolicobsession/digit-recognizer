import numpy as np


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A


def d_softmax(A):
    ds = - np.expand_dims(A, 1) * np.expand_dims(A, 0)
    diagonal = A * (1 - A)

    for i in range(A.shape[1]):
        np.fill_diagonal(ds[:, :, i], diagonal[:, i])

    return ds


def relu(Z):
    return Z * (Z > 0)


def d_relu(A):
    dZ = np.copy(A)
    dZ[A > 0] = 1
    return dZ

