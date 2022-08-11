import math

import numpy as np
from constants import EPS
from preprocessing import make_one_hot


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A


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
        self.b = None
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
        n, m = X.shape
        self.W = np.zeros((self.C, n))
        self.b = np.zeros((self.C, 1))
        Y = make_one_hot(y)
        self.C = len(np.unique(Y))

        n_mini_batches = math.ceil(m / self.mini_batch_size)
        v_dW, v_db = 0, 0
        s_dW, s_db = 0, 0

        for epoch in range(self.n_epochs):
            perm = self.rng.permutation(m)
            X_shuffled, Y_shuffled = X[:, perm], Y[:, perm]

            for mini_batch in range(n_mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min(start + self.mini_batch_size, m)
                X_mb = X_shuffled[:, start:end]
                Y_mb = Y_shuffled[:, start:end]

                Z = self.W @ X_mb + self.b
                A = softmax(Z)

                dA = - Y_mb / (A * self.mini_batch_size)
                dA_dZ = np.zeros((self.C, self.C, self.mini_batch_size))

                for k in range(self.mini_batch_size):
                    for i in range(self.C):
                        for j in range(self.C):
                            dA_dZ[i, j, k] = A[i, k] * (1 - A[i, k]) if i == j else - A[i, k] * A[j, k]
                dZ = np.einsum('ij,ikj->kj', dA, dA_dZ)

                dW = dZ @ X_mb.T
                db = np.sum(dZ, axis=1, keepdims=True)

                v_dW = self.beta1 * v_dW + (1 - self.beta1) * dW
                v_db = self.beta1 * v_db + (1 - self.beta1) * db

                s_dW = self.beta2 * s_dW + (1 - self.beta2) * np.power(dW, 2)
                s_db = self.beta2 * s_db + (1 - self.beta2) * np.power(db, 2)

                self.W -= self.learning_rate * v_dW / (np.sqrt(s_dW) + EPS)
                self.b -= self.learning_rate * v_db / (np.sqrt(s_db) + EPS)

            if self.verbose:
                print(
                    f'epoch {epoch:2d} - '
                    f'loss: {self.cross_entropy(X, Y):.2f}, '
                    f'metric: {self.metric(y, self.predict(X)):.2f}' if type(self.metric) is not None else ''
                )

    def predict(self, X):
        Z = self.W @ X + self.b
        return np.argmax(Z, axis=0)

    def cross_entropy(self, X, Y):
        Z = self.W @ X + self.b
        A = np.maximum(softmax(Z), EPS)
        return - np.sum(Y * np.log(A)) / X.shape[-1]
