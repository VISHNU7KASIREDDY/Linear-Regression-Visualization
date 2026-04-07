import numpy as np

class ManualLinearRegression:
    def __init__(self):
        self.m = 0
        self.b = 0
        self.loss_history = []

    def predict(self, X):
        return self.m * X + self.b

    def compute_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def train(self, X, y, alpha=0.01, epochs=100):
        n = len(X)
        self.loss_history = []

        for _ in range(epochs):
            y_pred = self.predict(X)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            self.m -= alpha * dm
            self.b -= alpha * db

        return self.m, self.b

    def get_params(self):
        return self.m, self.b

    def get_loss_history(self):
        return self.loss_history