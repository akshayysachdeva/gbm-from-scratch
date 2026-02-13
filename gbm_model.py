import numpy as np


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = X.values if hasattr(X, "values") else X
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(value=np.mean(y))

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )

    def _best_split(self, X, y):
        best_error = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])

                error = (
                    np.sum((y[left_mask] - left_mean) ** 2) +
                    np.sum((y[right_mask] - right_mean) ** 2)
                )

                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def predict(self, X):
        X = X.values if hasattr(X, "values") else X
        return np.array([self._predict_row(row, self.root) for row in X])

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value

        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)


class GradientBoostingRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
        self.train_losses = []

    def fit(self, X, y):
        y = np.array(y)
        self.init_prediction = np.mean(y)
        pred = np.full(len(y), self.init_prediction)

        for i in range(self.n_estimators):

            residual = y - pred

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)

            update = tree.predict(X)
            pred = pred + self.learning_rate * update

            loss = np.mean((y - pred) ** 2)
            self.train_losses.append(loss)

            self.trees.append(tree)

            print(f"Tree {i+1}/{self.n_estimators}, Loss: {loss:.4f}")

    def predict(self, X):

        pred = np.full(len(X), self.init_prediction)

        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        return pred