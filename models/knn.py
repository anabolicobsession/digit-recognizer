import numpy as np


class KNN:
    def __init__(self, K, verbose=False):
        self.points = None
        self.label = None
        self.K = K
        self.verbose = verbose

    def fit(self, X, y):
        self.points = np.copy(X)
        self.label = np.copy(y)

    def predict(self, X):
        return np.apply_along_axis(self.__predict, 0, X)

    def __predict(self, x):
        difference = np.expand_dims(x, axis=1) - self.points  # (n, p)
        norms = np.linalg.norm(difference, axis=0)  # (p,)
        neighbor_indexes = np.argpartition(norms, kth=self.K-1)[:self.K]  # (k,)
        values, counts = np.unique(self.label[neighbor_indexes], return_counts=True)
        return values[np.argmax(counts)]
