import numpy as np

class CustomDecisionTreeNode:
    """
    Node class for the custom decision tree.

    Attributes:
        feature_index (int): Index of the feature used for splitting at this node.
        threshold (float): Threshold value for the split.
        left (CustomDecisionTreeNode): Left child node.
        right (CustomDecisionTreeNode): Right child node.
        value (any): Value to return if this node is a leaf (i.e., the predicted class).
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CustomDecisionTree:
    """
    A simple implementation of a binary decision tree classifier.

    Parameters:
        max_depth (int, optional): The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        min_samples_split (int): The minimum number of samples required to split an internal node.

    Attributes:
        root (CustomDecisionTreeNode): The root node of the decision tree.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree classifier from the training set (X, y).

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict class for X.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted classes for each sample.
        """
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.

        Args:
            X (np.ndarray): Feature matrix at the current node.
            y (np.ndarray): Target vector at the current node.
            depth (int): Current depth of the tree.

        Returns:
            CustomDecisionTreeNode: The constructed node (either internal or leaf).
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping conditions: max depth, pure node, or not enough samples to split
        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return CustomDecisionTreeNode(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, num_features)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return CustomDecisionTreeNode(value=leaf_value)

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return CustomDecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, num_features):
        """
        Finds the best feature and threshold to split the data to minimize Gini impurity.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            num_features (int): Number of features.

        Returns:
            tuple: (best feature index, best threshold value)
        """
        best_gini = 1.0
        best_feat, best_thresh = None, None
        for feat in range(num_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idxs = X[:, feat] <= thresh
                right_idxs = X[:, feat] > thresh
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue
                gini = self._gini_index(y[left_idxs], y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def _gini_index(self, left_y, right_y):
        """
        Calculates the Gini impurity for a split.

        Args:
            left_y (np.ndarray): Target values for the left split.
            right_y (np.ndarray): Target values for the right split.

        Returns:
            float: Weighted Gini impurity of the split.
        """
        m_left = len(left_y)
        m_right = len(right_y)
        m = m_left + m_right
        gini_left = 1.0 - sum((np.sum(left_y == c) / m_left) ** 2 for c in np.unique(left_y)) if m_left > 0 else 0
        gini_right = 1.0 - sum((np.sum(right_y == c) / m_right) ** 2 for c in np.unique(right_y)) if m_right > 0 else 0
        return (m_left / m) * gini_left + (m_right / m) * gini_right

    def _most_common_label(self, y):
        """
        Finds the most common label in y.

        Args:
            y (np.ndarray): Target vector.

        Returns:
            The most frequent label.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict(self, inputs, node):
        """
        Recursively traverses the tree to predict the class for a single sample.

        Args:
            inputs (np.ndarray): Feature values for a single sample.
            node (CustomDecisionTreeNode): Current node in the tree.

        Returns:
            Predicted class label.
        """
        if node.value is not None:
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
    
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Mean accuracy of self.predict(X) vs y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
