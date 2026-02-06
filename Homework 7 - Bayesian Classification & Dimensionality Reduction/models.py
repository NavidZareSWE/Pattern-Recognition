import numpy as np


def calculate_unnormalized_covariance(data, mean):
    diff = data - mean
    return np.dot(diff.T, diff)


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            X_c = X[y == c]
            self.class_priors[idx] = len(X_c) / len(X)
            self.means[idx] = np.mean(X_c, axis=0)
            self.variances[idx] = np.var(
                X_c, axis=0) + 1e-9  # Additon to avoid 0

        return self

    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihoods = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            diff = X - self.means[idx]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances[idx]) +
                (diff ** 2) / self.variances[idx], axis=1
            )
            log_likelihoods[:, idx] = log_likelihood

        return log_likelihoods

    def predict(self, X):
        log_likelihoods = self._compute_log_likelihood(X)
        log_priors = np.log(self.class_priors)
        log_posteriors = log_likelihoods + log_priors

        return self.classes[np.argmax(log_posteriors, axis=1)]

    def get_params(self):
        return {
            'class_priors': self.class_priors,
            'means': self.means,
            'variances': self.variances
        }


class LDA:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.shared_cov = None
        self.shared_cov_inv = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.shared_cov = np.zeros((n_features, n_features))
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))

        # Compute class priors and means
        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            X_c = X[y == c]
            self.class_priors[idx] = len(X_c) / n_samples
            self.means[idx] = np.mean(X_c, axis=0)

        # Compute pooled (shared) covariance
        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            X_c = X[y == c]
            self.shared_cov += calculate_unnormalized_covariance(
                X_c, self.means[idx])
        self.shared_cov /= (n_samples - 1)
        self.shared_cov += 1e-6 * np.eye(n_features)  # Regularization

        try:
            self.shared_cov_inv = np.linalg.inv(self.shared_cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular (not inversable)
            self.shared_cov_inv = np.linalg.pinv(self.shared_cov)

        return self

    def _compute_discriminant(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        discriminants = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            diff = X - self.means[idx]
            # (x - μ)^T Σ⁻¹ (x - μ)
            # Step 1: Matrix multiply each row by Σ⁻¹
            #        Matrix multiplication with Σ⁻¹ applies this transformation to each row.

            # Step 2: Elementwise multiply by the original row
            #        Elementwise multiplication lets us compute this sum row-wise without loops.
            #        Finally, summing across the row gives the scalar squared Mahalanobis distance.

            mahal_dist = np.sum((diff @ self.shared_cov_inv) * diff, axis=1)
            # Discriminant = log prior - 0.5 * mahalanobis^2
            discriminants[:, idx] = np.log(
                self.class_priors[idx]) - 0.5 * mahal_dist

        return discriminants

    def predict(self, X):
        discriminants = self._compute_discriminant(X)
        return self.classes[np.argmax(discriminants, axis=1)]

    def get_params(self):
        return {
            'class_priors': self.class_priors,
            'means': self.means,
            'shared_covariance': self.shared_cov
        }


class QDA:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.covariances = None
        self.cov_inverse = None
        self.log_det_covs = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.covariances = []
        self.cov_inverse = []
        self.log_det_covs = []

        for idx, c in enumerate(self.classes):
            # Boolean indexing to filter data for the current class
            X_c = X[y == c]
            n_c = len(X_c)

            self.class_priors[idx] = n_c / n_samples
            self.means[idx] = np.mean(X_c, axis=0)

            # Class-specific covariance
            diff = X_c - self.means[idx]
            cov_c = (diff.T @ diff) / (n_c - 1)

            cov_c += 1e-4 * np.eye(n_features)  # Regularization
            self.covariances.append(cov_c)

            # Compute inverse and log determinant
            try:
                cov_inv = np.linalg.inv(cov_c)
                sign, log_det = np.linalg.slogdet(cov_c)
                log_det_cov = log_det if sign > 0 else -np.inf
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_c)
                log_det_cov = np.log(np.abs(np.linalg.det(cov_c)) + 1e-300)

            self.cov_inverse.append(cov_inv)
            self.log_det_covs.append(log_det_cov)

        return self

    def _compute_discriminant(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        discriminants = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            diff = X - self.means[idx]
            # (x - μ)^T Σ⁻¹ (x - μ)
            # Step 1: Matrix multiply each row by Σ⁻¹
            #        Matrix multiplication with Σ⁻¹ applies this transformation to each row.

            # Step 2: Elementwise multiply by the original row
            #        Elementwise multiplication lets us compute this sum row-wise without loops.
            #        Finally, summing across the row gives the scalar squared Mahalanobis distance.
            mahal_dist = np.sum(
                (diff @ self.cov_inverse[idx]) * diff, axis=1)
            # Discriminant = log prior - 0.5 * log|Σ| - 0.5 * mahalanobis^2
            discriminants[:, idx] = (np.log(self.class_priors[idx]) -
                                     0.5 * self.log_det_covs[idx] -
                                     0.5 * mahal_dist)

        return discriminants

    def predict(self, X):
        discriminants = self._compute_discriminant(X)
        return self.classes[np.argmax(discriminants, axis=1)]

    def get_params(self):
        return {
            'class_priors': self.class_priors,
            'means': self.means,
            'covariances': self.covariances
        }
