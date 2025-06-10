import numpy as np
import pandas as pd

class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.theta = np.zeros(X.shape[1] + 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        for _ in range(self.n_iter):
            linear_model = np.dot(X_b, self.theta)
            y_pred = self.sigmoid(linear_model)
            gradient = np.dot(X_b.T, (y_pred - y)) / y.size
            self.theta -= self.lr * gradient

    def predict_proba(self, X):
        X = np.array(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(np.dot(X_b, self.theta))

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

# Usage:
df = pd.read_csv(r'data.csv')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)

my_model = MyLogisticRegression(lr=0.1, n_iter=1000)
my_model.fit(x_train, y_train)
preds = my_model.predict(x_test)
accuracy = np.mean(preds == y_test)
print("Custom model accuracy:", accuracy)

