import numpy as np


class LogisticRegression:
    def __init__(self, max_iter=1000, lr=0.1, eps=1e-5) -> None:
        self.theta = None
        self.max_iter = max_iter
        self.lr = lr
        self.eps = eps

    def linear_part(self, theta, x) -> np.array:
        return x.dot(theta)

    def h(self, z) -> np.array:
        return 1 / 1 + np.exp(-z)

    def fit(self, x, y) -> None:
        def _gradient(theta, x, y) -> np.array:
            sample_size, _ = x.shape
            z = self.linear_part(theta, x)
            return (1 / sample_size) * x.T.dot(y - self.h(z))

        def _next_theta(theta, x, y) -> np.array:
            sample_size, _ = x.shape
            return theta - (1 / sample_size) * self.lr * _gradient(theta, x, y)

        _, feature_size = x.shape
        self.theta = np.zeros(feature_size)

        old_theta = self.theta
        new_theta = _next_theta(self.theta, x, y)

        while np.linalg.norm(old_theta - new_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = _next_theta(old_theta, x, y)

        self.theta = new_theta
        print(self.theta)

    def predict(self, x) -> np.array:
        z = self.linear_part(self.theta, x)
        y_pred = self.h(z)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred
