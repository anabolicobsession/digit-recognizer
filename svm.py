import numpy as np

from preprocessing import make_one_hot


def add_bias_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])


class SVM:
    def __init__(self, C, verbose=False):
        self.W = None
        self.n = None

        self.C = C
        self.verbose = verbose

        self.learning_rate = 0.01
        self.n_epochs = 10

    def fit(self, X, y):
        X_b = add_bias_ones(X)
        Y = make_one_hot(y)
        self.n = X_b.shape[0]
        n_classes = Y.shape[0]

        self.W = np.zeros((n_classes, n_classes - 1, self.n))
        to_ones = np.vectorize(lambda x, y: 1 if x == y else -1)

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                opp_class = j if i < j else j + 1
                indexes = np.logical_or(y == i, y == opp_class)
                self.W[i, j] = self.__construct_hyperplane(X_b[:, indexes], to_ones(y[indexes], i))

                if self.verbose:
                    print(f'{(i * self.W.shape[1] + j + 1):2d}/{self.W.shape[0] * self.W.shape[1]} '
                          f'hyperplanes have been constructed')

    def predict(self, X):
        return np.argmax(np.sum(np.tensordot(self.W, add_bias_ones(X), 1), axis=1), axis=0)

    def __construct_hyperplane(self, X, y):
        w = np.zeros(self.n)

        for i in range(self.n_epochs):
            dw = w - self.C * (y * X) @ (1 - y * (w @ X) > 0)
            w -= self.learning_rate * dw

        return w