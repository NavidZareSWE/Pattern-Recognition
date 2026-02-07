import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None

    def fit(self, X):
        n_samples, n_features = X.shape

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # For high-dimensional data (n_features > n_samples),
        # use the trick: compute X*X^T instead of X^T*X
        # Let u be an eigenvector of the matrix X·Xᵀ, satisfying the equation:
        # (X·Xᵀ) · u = λ · u
        # Multiplying both sides by Xᵀ gives:
        # Xᵀ · (X·Xᵀ) · u = Xᵀ · λ · u
        # Rearranging  leads to:
        # (Xᵀ·X) · (Xᵀ·u) = λ · (Xᵀ·u)
        # Defining v as v = Xᵀ · u transforms the equation to:
        # (Xᵀ·X) · v = λ · v
        if n_features > n_samples:
            C = X_centered @ X_centered.T / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(C)

            # Sort -> descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Convert to principal components in original space
            # If u is eigenvector of X*X^T, then v = X^T*u is eigenvector of X^T*X
            # According to above proof
            components = X_centered.T @ eigenvectors

            norms = np.linalg.norm(components, axis=0)
            norms[norms == 0] = 1  # Avoid division by zero
            components = components / norms

            self.eigenvalues = eigenvalues
            self.components = components.T
        else:
            cov_matrix = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort -> descending order
            idx = np.argsort(eigenvalues)[::-1]
            self.eigenvalues = eigenvalues[idx]
            self.components = eigenvectors[:, idx].T

        total_variance = np.sum(self.eigenvalues[self.eigenvalues > 0])
        self.explained_variance_ratio = self.eigenvalues / total_variance
        self.cumulative_variance_ratio = np.cumsum(
            self.explained_variance_ratio)

        if self.n_components is not None:
            self.components = self.components[:self.n_components]
            self.eigenvalues = self.eigenvalues[:self.n_components]

        return self

    def transform(self, X, n_components=None):
        X_centered = X - self.mean
        if n_components is not None:
            return X_centered @ self.components[:n_components].T
        return X_centered @ self.components.T

    def inverse_transform(self, X_transformed, n_components=None):
        if n_components is not None:
            return X_transformed @ self.components[:n_components] + self.mean
        return X_transformed @ self.components + self.mean

    # Returns the number of components to capture at least [variance_threshold] of total variance.
    def get_n_components_for_variance(self, variance_threshold=0.9):
        return np.argmax(self.cumulative_variance_ratio >= variance_threshold) + 1


class FisherLDA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.class_means = None
        self.overall_mean = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.overall_mean = np.mean(X, axis=0)

        self.class_means = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            self.class_means[idx] = np.mean(X[y == c], axis=0)

        # Compute within-class scatter matrix S_W
        S_W = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            X_c = X[y == c]
            diff = X_c - self.class_means[idx]
            S_W += diff.T @ diff

        # Compute between-class scatter matrix S_B
        S_B = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes):
            n_c = np.sum(y == c)
            mean_diff = (self.class_means[idx] -
                         self.overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Store for analysis
        self.S_W = S_W
        self.S_B = S_B

        # Solving:  S_W^(-1) * S_B * v = λ * v
        S_W_reg = S_W + 1e-6 * np.eye(n_features)
        S_W_inv = np.linalg.pinv(S_W_reg)
        eigenvalues, eigenvectors = np.linalg.eigh(S_W_inv @ S_B)

        # Sort -> descending order
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[:, idx].T

        # Maximum meaningful components = n_classes - 1
        max_components = n_classes - 1
        if self.n_components is not None:
            n_comp = min(self.n_components, max_components)
        else:
            n_comp = max_components

        self.components = self.components[:n_comp]
        self.eigenvalues = self.eigenvalues[:n_comp]
        return self

    def transform(self, X, n_components=None):
        if n_components is not None:
            return X @ self.components[:n_components].T
        return X @ self.components.T
