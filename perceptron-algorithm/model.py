import numpy as np


class Perceptron:
    def activation(self, x) -> np.array:
        return np.where(x > 0, 1, 0)

    def __init__(self, max_iter=500, lr=0.2) -> None:
        self.max_iter = max_iter
        self.lr = lr
        self.weight = None
        self.bias = None

    def fit(self, X, y) -> None:
        # Init weights and bias
        n_sample, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0

        for _ in range(self.max_iter):
            # predict y hat
            for idx, x in enumerate(X):
                linear_model = np.dot(x, self.weight) + self.bias
                y_pred = self.activation(linear_model)
                update = self.lr * (y[idx] - y_pred)
                # Update weights and bias
                self.weight += update * x
                self.bias += update

    def predict(self, X) -> np.array:
        linear_model = X.dot(self.weight) + self.bias
        y_pred = self.activation(linear_model)
        return y_pred
