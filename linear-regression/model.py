import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # y_pred = x.dot(self.weight) + self.bias
            y_pred = np.dot(x, self.weight) + self.bias

            self.weight += self.lr * (1 / n_samples) * x.T.dot(y - y_pred)
            self.bias += self.lr * (1 / n_samples) * np.sum(y - y_pred)

    def predict(self, x):
        return np.dot(x, self.weight) + self.bias
