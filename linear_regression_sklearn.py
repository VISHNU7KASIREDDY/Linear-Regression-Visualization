import numpy as np
from sklearn.linear_model import LinearRegression

class SklearnLinearRegression:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        X = X.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X):
        X = X.reshape(-1, 1)
        return self.model.predict(X)

    def get_params(self):
        m = self.model.coef_[0]
        b = self.model.intercept_
        return m, b