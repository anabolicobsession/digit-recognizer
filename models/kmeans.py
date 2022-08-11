import numpy as np


class KMeans:
    def __init__(self, K, n_epochs=None, unsupervised_learning=True, verbose=False, metric=None):
        self.centroids = None
        self.K = K
        self.n_epochs = n_epochs
        self.unsupervised_learning = unsupervised_learning
        self.verbose = verbose
        self.metric = metric
        self.rng = np.random.default_rng()

    def fit(self, X, y=None):
        n, m = X.shape
        x_c = np.zeros(m) if y is None else np.squeeze(y)

        if self.unsupervised_learning:
            self.centroids = self.rng.choice(X, size=self.K, axis=1, replace=False)

            for epoch in range(self.n_epochs):
                if self.verbose:
                    print(f'epoch {epoch:2d} - '
                          f'loss: {self.__distance_loss(X, x_c):.3f}, '
                          f'metric: {self.metric(y, self.predict(X)):5.2f}, '
                          f'distribution: {self.__centroid_distribution(X)}')

                x_c = self.predict(X)

                for k in range(self.K):
                    self.centroids[:, k] = np.average(X[:, x_c == k], axis=1)
        else:
            assert y is not None, "Labels were not given, supervised learning isn't possible"
            self.centroids = np.zeros((n, self.K))

            for k in range(self.K):
                self.centroids[:, k] = np.average(X[:, x_c == k], axis=1)

            if self.verbose:
                print(f'loss: {self.__distance_loss(X, x_c):.2f}, '
                      f'metric: {self.metric(y, self.predict(X)):.2f}')

    def predict(self, X):
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
        return np.unique(self.predict(X), return_counts=True)[1]
