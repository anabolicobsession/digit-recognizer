import math

import numpy as np
from constants import EPS


def make_one_hot(x, n_classes):
    def to_one_hot(n):
        oh = np.zeros(n_classes)
        oh[n] = 1
        return oh

    return np.array(list(map(to_one_hot, list(np.squeeze(x))))).T


def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=0))
    A = E / np.sum(E, axis=0)
    return A


class LogisticRegression:
    def __init__(self, n_classes, metric):
        self.W = None
        self.b = None
        self.n_classes = n_classes
        self.metric = metric
        self.rng = np.random.default_rng(0)

    def fit(self, X, y, learning_rate, n_epochs, mini_batch_size=64, beta1=0.9, beta2=0.99, print_epochs=True, print_every=10):
        """
        Fit the model with given data.

        :param X: a ndarray with the shape (n, m), where n is the number of features and m is the number of samples
        :param y: a ndarray with the shape (1, m)
        :param n_epochs: a number of epochs
        :param learning_rate: a parameter which determines a training speed
        :param mini_batch_size: a size of one mini batch of samples
        :param beta1: a parameter which controls momentum of weights
        :param beta2: a parameter which controls normalization of weights
        :param print_epochs: defines whether to print statistics between epochs
        :param print_every: how often to print statistics
        """
        n, m = X.shape
        self.W = np.zeros((self.n_classes, n))
        self.b = np.zeros((self.n_classes, 1))
        Y = make_one_hot(y, self.n_classes)

        n_mini_batches = math.ceil(m / mini_batch_size)
        v_dW, v_db = 0, 0
        s_dW, s_db = 0, 0

        for epoch in range(n_epochs):
            perm = self.rng.permutation(m)
            X_shuffled, Y_shuffled = X[:, perm], Y[:, perm]

            for mini_batch in range(n_mini_batches):
                start = mini_batch * mini_batch_size
                end = min(start + mini_batch, m)
                X_mb = X_shuffled[:, start:end]
                Y_mb = Y_shuffled[:, start:end]

                Z = self.W @ X_mb + self.b
                A = softmax(Z)

                dZ = - Y_mb * (1 - A) / mini_batch_size
                dW = dZ @ X_mb.T
                db = np.sum(dZ, axis=-1, keepdims=True)

                v_dW = beta1 * v_dW + (1 - beta1) * dW
                v_db = beta1 * v_db + (1 - beta1) * db

                s_dW = beta2 * s_dW + (1 - beta2) * np.power(dW, 2)
                s_db = beta2 * s_db + (1 - beta2) * np.power(db, 2)

                self.W -= learning_rate * v_dW / (np.sqrt(s_dW) + EPS)
                self.b -= learning_rate * v_db / (np.sqrt(s_db) + EPS)

            if print_epochs and epoch % print_every == 0:
                print(f'epoch: {epoch:2d}, '
                      f'loss: {self.cross_entropy(X, Y):.2f}, '
                      f'acc: {self.metric(self.predict(X), y):.2f}')

    def predict(self, X):
        """
        Predicts a class using previously trained parameters.
        
        :param X: a ndarray with the shape (n, m), where n is the number of features and m is the number of samples
        :return: a ndarray with the shape (1, m)
        """
        Z = self.W @ X + self.b
        pred = np.argmax(Z, axis=0, keepdims=True)
        return pred

    def cross_entropy(self, X, Y):
        Z = self.W @ X + self.b
        A = np.maximum(softmax(Z), EPS)
        ce = - np.sum(Y * np.log(A)) / X.shape[-1]
        return ce
