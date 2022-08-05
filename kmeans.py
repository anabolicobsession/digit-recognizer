import numpy as np


class KMeans:
    def __init__(self, K, metric=None):
        self.centroids = None
        self.K = K
        self.metric = metric
        self.rng = np.random.default_rng()

    def fit(self, X, n_epochs, y=None, supervised_learning=False, print_info=True):
        n, m = X.shape
        x_c = np.zeros(m) if y is None else np.squeeze(y)

        if y is None or not supervised_learning:
            self.centroids = self.rng.choice(X, size=self.K, axis=1, replace=False)

            for epoch in range(n_epochs):
                if print_info:
                    print(f'epoch {epoch:2d} - '
                          f'loss: {self.__distance_loss(X, x_c):.2f}, '
                          f'metric: {self.metric(self.predict(X), y):5.2f}, '
                          f'distribution: {self.__centroid_distribution(X)}')

                x_c = self.__find_nearest_centroids(X)

                for k in range(self.K):
                    self.centroids[:, k] = np.average(X[:, x_c == k], axis=1)
        else:
            self.centroids = np.zeros((n, self.K))

            for k in range(self.K):
                self.centroids[:, k] = np.average(X[:, x_c == k], axis=1)

            if print_info:
                print(f'after fitting centroids: loss {self.__distance_loss(X, x_c):.2f}'
                      f', metric {self.metric(self.predict(X), y):.2f}')

    def predict(self, X):
        return self.__find_nearest_centroids(X).reshape(1, -1)

    def __find_nearest_centroids(self, X):
        difference_tensor = np.expand_dims(X, 2) - np.expand_dims(self.centroids, 1)
        norms = np.linalg.norm(difference_tensor, axis=0)
        return np.argmin(norms, axis=1)

    def __distance_loss(self, X, x_c):
        summa = 0.

        for k in range(self.K):
            difference = self.centroids[:, k].reshape(-1, 1) - X[:, x_c == k]
            norms = np.linalg.norm(difference, axis=0)
            summa += np.sum(norms)

        return summa / X.shape[1]

    def __centroid_distribution(self, X):
        return np.unique(self.__find_nearest_centroids(X), return_counts=True)[1]
