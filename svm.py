import math

import numpy as np


def add_bias_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])


class SVM:
    def __init__(self, C, learning_rate, n_epochs, mini_batch_size=64, beta=0.9, verbose=False):
        self.W = None
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.verbose = verbose
        self.rng = np.random.default_rng(0)

    def fit(self, X, y):
        X_b = add_bias_ones(X)
        n_features = X_b.shape[0]
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_classes, n_classes - 1, n_features))
        to_ones = np.vectorize(lambda x, y: 1 if x == y else -1)

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                if j >= i:
                    opp_class = j + 1
                    indexes = np.logical_or(y == i, y == opp_class)
                    self.W[i, j] = self.__construct_hyperplane(X_b[:, indexes], to_ones(y[indexes], i))
                else:
                    self.W[i, j] = - self.W[j, i - 1]

                if self.verbose:
                    print(f'{(i * self.W.shape[1] + j + 1):2d}/{self.W.shape[0] * self.W.shape[1]} '
                          f'hyperplanes have been constructed')

    def predict(self, X):
        return np.argmax(np.sum(np.tensordot(self.W, add_bias_ones(X), 1), axis=1), axis=0)

    def __construct_hyperplane(self, X, y):
        n, m = X.shape
        w = np.zeros(n)
        v_dw = 0
        n_mini_batches = math.ceil(m / self.mini_batch_size)

        for i in range(self.n_epochs):
            perm = self.rng.permutation(m)
            X_shuffled, y_shuffled = X[:, perm], y[perm]

            for mini_batch in range(n_mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min(start + self.mini_batch_size, m)
                X_mb = X_shuffled[:, start:end]
                y_mb = y_shuffled[start:end]

                dw = w - self.C * (y_mb * X_mb) @ (1 - y_mb * (w @ X_mb) > 0)
                v_dw = (self.beta * v_dw + (1 - self.beta) * dw)
                w -= self.learning_rate * v_dw

        return w
