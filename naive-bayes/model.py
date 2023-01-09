import numpy as np


class NaiveBayes:
    def __init__(self) -> None:
        self._mean = None
        self._var = None
        self._priors = None
        self._classes = None

    def fit(self, X, y) -> None:
        sample_size, feature_size = X.shape
        self._classes = np.unique(y)
        class_size = len(self._classes)

        # Calculate means, variances of each features for each classes. and priors for each classes.
        self._mean = np.zeros((class_size, feature_size), dtype=np.float64)
        self._var = np.zeros((class_size, feature_size), dtype=np.float64)
        self._priors = np.zeros(class_size, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            # X_c = []
            # for x in X:
            #     for y_c in y:
            #         X_c.append(x if y_c == c)
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / np.float64(sample_size)

    def predict(self, X) -> np.array:
        # Get likelihood of each features for each classes
        def _likelihood(class_idx, x) -> np.array:
            mean = self._mean[class_idx]
            var = self._var[class_idx]
            numerator = np.exp(-((x - mean) ** 2) / 2 * var)
            denominator = np.sqrt(2 * np.pi * var)
            return numerator / denominator

        # Get posteriors for each classes
        def _predict(x) -> int:
            log_posteriors = []
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                posterior = np.sum(np.log(_likelihood(idx, x)))
                log_posteriors.append(posterior + prior)

            return self._classes[np.argmax(log_posteriors)]

        y_pred = [_predict(x) for x in X]
        return np.array(y_pred)
