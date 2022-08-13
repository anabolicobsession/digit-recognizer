import math

import numpy as np

from utils.preprocessing import make_one_hot, add_bias_ones
from utils.constants import EPS


class NeuralNetwork:
    def __init__(
            self,
            hidden_layers,
            activations,
            derivatives,
            loss,
            d_loss,
            learning_rate,
            n_epochs,
            mini_batch_size=32,
            beta1=0.9,
            beta2=0.999,
            verbose=False,
            metric=None
    ):
        self.W = [np.array([])]
        self.n_layers = len(hidden_layers) + 2
        self.rng = np.random.default_rng(0)

        self.layers = hidden_layers
        self.activations = [lambda x: x] + activations
        self.derivatives = [lambda x: x] + derivatives
        self.loss = loss
        self.d_loss = d_loss
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose
        self.metric = metric

    def fit(self, X, y):
        m = X.shape[1]
        Y = make_one_hot(y)
        self.layers = np.concatenate(([X.shape[0]], np.array(self.layers, dtype=int), [Y.shape[0]]))

        for i in range(1, self.n_layers):
            dev = np.sqrt(2 / self.layers[i - 1])
            shape = self.layers[i], self.layers[i - 1] + 1  # + 1 for a bias
            self.W.append(self.rng.normal(0., dev, shape))

        n_mini_batches = math.ceil(m / self.mini_batch_size)
        v_dW, s_dW = self.n_layers * [0], self.n_layers * [0]

        for epoch in range(self.n_epochs):
            indexes = self.rng.permutation(m)
            X_shuffled, Y_shuffled = X[:, indexes], Y[:, indexes]

            for mb in range(n_mini_batches):
                start = mb * self.mini_batch_size
                end = min(start + self.mini_batch_size, m)
                X_mb, Y_mb = X_shuffled[:, start:end], Y_shuffled[:, start:end]

                A = self.__forward_propagation(X_mb)
                dW = self.__backward_propagation(A, Y_mb)

                for i in range(1, self.n_layers):
                    v_dW[i] = self.beta1 * v_dW[i] + (1 - self.beta1) * dW[i]
                    s_dW[i] = self.beta2 * s_dW[i] + (1 - self.beta2) * np.square(dW[i])
                    self.W[i] -= self.learning_rate * v_dW[i] / np.maximum(np.sqrt(s_dW[i]), EPS)

            if self.verbose:
                print(
                    f'epoch {epoch:2d} - '
                    f'loss: {self.loss(Y, self.__forward_propagation(X)[-1]):.2f}, '
                    f'metric: {self.metric(y, self.predict(X)):.2f}' if type(self.metric) is not None else ''
                )

    def predict(self, X):
        A = X

        for i in range(1, self.n_layers - 1):
            A = self.activations[i](self.W[i] @ add_bias_ones(A))
            
        return np.argmax(self.W[-1] @ add_bias_ones(A), axis=0)

    def __forward_propagation(self, X):
        A = [X]

        for i in range(1, self.n_layers):
            A.append(self.activations[i](self.W[i] @ add_bias_ones(A[i - 1])))

        return A

    def __backward_propagation(self, A, Y):
        dW = self.n_layers * [np.array([])]
        dA = self.d_loss(Y, A[-1])

        for i in reversed(range(1, self.n_layers)):
            dZ = self.derivatives[i](dA, A[i])
            dW[i] = dZ @ add_bias_ones(A[i - 1]).T
            if i > 1: dA = self.W[i].T[:-1] @ dZ

        return dW
