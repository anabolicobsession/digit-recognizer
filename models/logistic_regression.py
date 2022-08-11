import math

import numpy as np

from utils.preprocessing import add_bias_ones, make_one_hot
from functions.activation_functions import softmax, d_softmax
from functions.loss_functions import cross_entropy, d_cross_entropy
from utils.constants import EPS


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
                dW = self.__backward_propagation(X_mb, Y_mb, A)

                v_dW = self.beta1 * v_dW + (1 - self.beta1) * dW
                s_dW = self.beta2 * s_dW + (1 - self.beta2) * np.square(dW)

                self.W -= self.learning_rate * v_dW / (np.maximum(np.sqrt(s_dW), EPS))

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
        return np.einsum('ij,ikj->kj', d_cross_entropy(Y, A), d_softmax(A)) @ X.T
