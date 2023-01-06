import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import LinearRegression
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

model = LinearRegression()

model.fit(X_train, y_train)
