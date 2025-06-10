import numpy as np
import pandas as pd

class MyLogisticRegressionMulticlass:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.classes_ = None
        self.models_ = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            theta = np.zeros(X.shape[1] + 1)
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            for _ in range(self.n_iter):
                linear_model = np.dot(X_b, theta)
                y_pred = self.sigmoid(linear_model)
                gradient = np.dot(X_b.T, (y_pred - y_binary)) / y.size
                theta -= self.lr * gradient
            self.models_[cls] = theta

    def predict_proba(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            theta = self.models_[cls]
            probs[:, idx] = self.sigmoid(np.dot(X_b, theta))
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)