import numpy as np
# =============================================================================
# SECTION 2.4: CLASSIFICATION
# =============================================================================


class NearestCentroidClassifier:

    def __init__(self):
        self.class_means = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_means = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            self.class_means[idx] = np.mean(X[y == c], axis=0)

        return self

    def predict(self, X):
        distances = np.zeros((X.shape[0], len(self.classes)))
        for idx in range(len(self.classes)):
            distances[:, idx] = np.sum(
                (X - self.class_means[idx]) ** 2, axis=1)

        return self.classes[np.argmin(distances, axis=1)]
