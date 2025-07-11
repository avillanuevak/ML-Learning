import numpy as np
from DecisionTree.CustomDecisionTree import CustomDecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Only pass n_features if it is not None
            if self.n_features is not None:
                tree = CustomDecisionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    n_features=self.n_features,
                )
            else:
                tree = CustomDecisionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth
                )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to have predictions for each sample in columns
        tree_preds = np.swapaxes(predictions, 0, 1)
        # Get majority vote for each sample
        y_pred = np.array([self._most_common_label(pred) for pred in tree_preds])
        return y_pred

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def score(self, X, y):
        """Return the accuracy of the model on the given data."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)