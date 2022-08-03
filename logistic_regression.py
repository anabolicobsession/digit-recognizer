import numpy as np


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
    def __init__(self, n_classes):
        self.W = None
        self.b = None
        self.n_classes = n_classes

    def fit(self, X, y, epochs, learning_rate, print_epochs=False, print_every=10):
        """
        Fit the model with given data.

        :param X: a ndarray with the shape (n, m), where n is the number of features and m is the number of samples
        :param y: a ndarray with the shape (1, m)
        :param epochs: a number of epochs
        :param learning_rate: a parameter which determines a training speed
        :param print_epochs: defines whether to print statistics between epochs
        :param print_every: how often to print statistics
        """
        n, m = X.shape
        self.W = np.zeros((self.n_classes, n))
        self.b = np.zeros((self.n_classes, 1))
        Y = make_one_hot(y, self.n_classes)

        for i in range(epochs):
            Z = self.W @ X + self.b
            A = softmax(Z)

            dZ = - Y * (1 - A) / m
            dW = (dZ @ X.T) / m
            db = np.average(dZ, axis=-1).reshape(-1, 1)

            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if print_epochs and i % print_every == 0:
                acc = np.sum(y == np.argmax(Z, axis=0, keepdims=True)) / m
                loss = - np.sum(Y * np.log(A)) / m
                print(f'epoch: {i:4d}, acc: {acc:.3f}, loss: {loss:.3f}')

    def predict(self, X):
        """
        Predicts a class using previously trained parameters.
        
        :param X: a ndarray with the shape (n, m), where n is the number of features and m is the number of samples
        :return: a ndarray with the shape (1, m)
        """
        Z = self.W @ X + self.b
        pred = np.argmax(Z, axis=0, keepdims=True)
        return pred
