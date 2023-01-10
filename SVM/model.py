import numpy as np


class SVM:
    def __init__(self, n_iter=1000, lr=0.001, lambda_param=0.01) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.weight = None
        self.bias = None
        self.lambda_param = lambda_param

    def fit(self, X, y):
        # Init weight and bias
        _, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0
        _y = np.where(y <= 0, -1, 1)

        # Update Rule
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = _y[idx] * np.dot(x_i, self.weight) >= 1
                if condition:
                    self.weight -= 2 * self.lr * self.lambda_param * self.weight
                else:
                    self.weight -= self.lr * (
                        2 * self.lambda_param * self.weight - _y[idx] * x_i
                    )
                    self.bias -= self.lr * _y[idx]

    def predict(self, X):
        linear_model = X.dot(self.weight) - self.bias
        return np.sign(linear_model)
