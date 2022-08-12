import numpy as np

from utils.preprocessing import make_one_hot


class NeuralNetwork:
    def __init__(
            self,
            hidden_layers,
            activations,
            derivatives,
            loss_function,
            learning_rate,
            n_epochs,
            eps=1e-9,
            verbose=False
    ):
        self.W = []
        self.b = []
        self.rng = np.random.default_rng(0)
        self.n_layers = len(hidden_layers) + 2

        assert len(activations) == len(derivatives), \
            'Mismatch between the number of functions and the number of derivatives'
        assert len(activations) == self.n_layers - 1, \
            'Mismatch between the number of functions and the number of layers'

        self.layers = hidden_layers
        self.activations = activations
        self.derivatives = derivatives
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, y):
        m = X.shape[1]
        Y = make_one_hot(y)
        self.layers = np.concatenate(([X.shape[0]], self.layers, [Y.shape[0]]))

        for i in range(1, self.n_layers):
            self.W.append(self.eps * self.rng.random((self.layers[i], self.layers[i - 1] + 1)))  # 1 for a bias

        for epoch in range(self.n_epochs):
            for i in reversed(range(self.n_layers - 1)):
                pass

    def predict(self, X):
        pass

    def __forward_propagation(self, X):
        A = X

        for i in range(self.n_layers - 1):
            A = self.activations[i](self.W[i] @ A + self.b[i])

        return A
