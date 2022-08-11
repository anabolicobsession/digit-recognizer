import math

import numpy as np

from activations import softmax
from constants import EPS
from loss_functions import cross_entropy
from preprocessing import make_one_hot, add_bias_ones


class LogisticRegression:
    def __init__(
            self,
            learning_rate,
            n_epochs,
            mini_batch_size=32,
            beta1=0.9,
            beta2=0.999,
            verbose=False,
            metric=None
    ):
        self.W = None
        self.C = None
        self.rng = np.random.default_rng(0)

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose
        self.metric = metric

    def fit(self, X, y):
        X_b = add_bias_ones(X)
        n, m = X_b.shape
        self.C = len(np.unique(y))
        self.W = np.zeros((self.C, n))
        Y = make_one_hot(y)

        n_mini_batches = math.ceil(m / self.mini_batch_size)
        v_dW = 0
        s_dW = 0

        for i in range(self.n_epochs):
            perm = self.rng.permutation(m)
            X_shuffled = X_b[:, perm]
            Y_shuffled = Y[:, perm]

            for j in range(n_mini_batches):
                start = j * self.mini_batch_size
                end = min(start + self.mini_batch_size, m)
                X_mb = X_shuffled[:, start:end]
                Y_mb = Y_shuffled[:, start:end]

                A = self.__forward_propagation(X_mb)
                dW, db = self.__backward_propagation(X_mb, Y_mb, A)

                v_dW = self.beta1 * v_dW + (1 - self.beta1) * dW
                s_dW = self.beta2 * s_dW + (1 - self.beta2) * np.square(dW)

                self.W -= self.learning_rate * v_dW / (np.sqrt(s_dW) + EPS)

            if self.verbose:
                print(
                    f'epoch {i:2d} - '
                    f'loss: {cross_entropy(Y, self.__forward_propagation(X_b)):.2f}, '
                    f'metric: {self.metric(y, self.predict(X)):.2f}' if type(self.metric) is not None else ''
                )

    def predict(self, X):
        return np.argmax(self.W @ add_bias_ones(X), axis=0)

    def __forward_propagation(self, X):
        return softmax(self.W @ X)

    def __backward_propagation(self, X, Y, A):
        dA = - Y / (A * self.mini_batch_size)
        dA_dZ = np.zeros((self.C, self.C, self.mini_batch_size))

        for k in range(self.mini_batch_size):
            for i in range(self.C):
                for j in range(self.C):
                    dA_dZ[i, j, k] = A[i, k] * (1 - A[i, k]) if i == j else - A[i, k] * A[j, k]
        dZ = np.einsum('ij,ikj->kj', dA, dA_dZ)

        dW = dZ @ X.T
        db = np.sum(dZ, axis=1, keepdims=True)

        return dW, db
