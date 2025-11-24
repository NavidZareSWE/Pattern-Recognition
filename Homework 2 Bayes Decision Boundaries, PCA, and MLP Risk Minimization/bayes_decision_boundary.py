import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    # Get student number for seeding randomness
    # Get student number for seeding randomness
    # student_number = int(input("Enter your student number: "))
    student_number = 40435071
    np.random.seed(student_number)

    # Generate samples for 2 classes (same as Part 1)
    num_samples = 100
    true_mean1 = np.array([2, 2])
    true_cov1 = np.array([[1, 0.5], [0.5, 1]])
    samples1 = np.random.multivariate_normal(
        true_mean1, true_cov1, num_samples)

    true_mean2 = np.array([5, 5])
    true_cov2 = np.array([[1, -0.5], [-0.5, 1]])
    samples2 = np.random.multivariate_normal(
        true_mean2, true_cov2, num_samples)

    def calculate_mean(data):
        sums = np.zeros(2)
        n = data.shape[0]

        for col_idx, col in enumerate(data.T):
            col_sum = 0
            for val in col:
                col_sum += val
            sums[col_idx] = col_sum

        return sums / n

    def calculate_covariance(data, mean):
        n = data.shape[0]           # Num rows  = (n_samples)
        diff = data - mean
        cov = np.dot(diff.T, diff) / n
        return cov

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)
    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    # Priors (assume equal)
    prior1 = 0.5
    prior2 = 0.5

    # ## DONE ##: Implement 2x2 matrix determinant from scratch
    def matrix_det(_2d_matrix):
        return (_2d_matrix[0][0] * _2d_matrix[1][1]) - (_2d_matrix[0][1] * _2d_matrix[1][0])

    # ## Done ##: Implement 2x2 matrix inverse from scratch
    def matrix_inv(cov):
        det = matrix_det(cov)
        # 1/det * [[d, -b], [-c, a]]
        return np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det

    # ## Done ##: Implement multivariate Gaussian log PDF from scratch
    # Use np.dot for multiplications

    def multivariate_gaussian_logpdf(x, mean, cov):
        dim = len(mean)
        diff = x - mean
        det_cov = matrix_det(cov)
        inv_cov = matrix_inv(cov)
        mahalanobis = np.dot(np.dot(diff, inv_cov), diff)
        return -0.5 * (dim * np.log(2 * np.pi) + np.log(det_cov) + mahalanobis)

    # ## Done ##: Implement Bayes classifier
    # Return 0 if >=0 else 1
    # Expected: Function that takes x (2D array) and returns class (0 or 1)
    def bayes_classifier(x):
        prior_ratio = prior1 / prior2
        log_priror_ratio = np.log(prior_ratio)
        log_pdf_1 = multivariate_gaussian_logpdf(
            x, calculated_mean1, cov_matrix1)
        log_pdf_2 = multivariate_gaussian_logpdf(
            x, calculated_mean2, cov_matrix2)
        score = log_priror_ratio + (log_pdf_1 - log_pdf_2)
        if score >= 0:
            return 0
        return 1

    def visualize_boundary_with_distributions(samples1, samples2, mean1, mean2, cov1, cov2, bayes_classifier):
        print("Visualizing the decision Boundary (this might take a while), please wait.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Create grid
        x_min, x_max = -1, 8
        y_min, y_max = -1, 8
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Calculate decision regions
        Z = np.array([bayes_classifier(pt) for pt in grid])
        Z = Z.reshape(xx.shape)

        # Plot 1: Decision boundary with samples
        cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
        ax1.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        ax1.scatter(samples1[:, 0], samples1[:, 1],
                    color='blue', label='Class 1')
        ax1.scatter(samples2[:, 0], samples2[:, 1],
                    color='red', label='Class 2')
        ax1.plot(mean1[0], mean1[1], 'b*', markersize=15, label='Mean 1')
        ax1.plot(mean2[0], mean2[1], 'r*', markersize=15, label='Mean 2')
        ax1.set_title('Bayes Decision Boundary')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Distribution contours
        from scipy.stats import multivariate_normal
        rv1 = multivariate_normal(mean1, cov1)
        rv2 = multivariate_normal(mean2, cov2)

        # Calculate PDFs
        pos = np.dstack((xx, yy))
        pdf1 = rv1.pdf(pos)
        pdf2 = rv2.pdf(pos)

        # Plot contours
        ax2.contour(xx, yy, pdf1, colors='blue', alpha=0.5, levels=5)
        ax2.contour(xx, yy, pdf2, colors='red', alpha=0.5, levels=5)
        ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        ax2.scatter(samples1[:, 0], samples1[:, 1],
                    color='blue', alpha=0.3, s=10)
        ax2.scatter(samples2[:, 0], samples2[:, 1],
                    color='red', alpha=0.3, s=10)
        ax2.set_title('Distribution Contours with Decision Boundary')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Print analysis
        print("\n=== Decision Boundary Analysis ===")
        print(f"Class 1 Mean: {mean1}")
        print(f"Class 2 Mean: {mean2}")
        print(f"Distance between means: {np.linalg.norm(mean2 - mean1):.2f}")

    print("Bayes classifier implemented. Run visualization to check.")

    # Visualization (provided): Plot samples and decision boundary
    # Create grid
    x_min, x_max = min(np.min(samples1[:, 0]), np.min(
        samples2[:, 0])) - 1, max(np.max(samples1[:, 0]), np.max(samples2[:, 0])) + 1
    y_min, y_max = min(np.min(samples1[:, 1]), np.min(
        samples2[:, 1])) - 1, max(np.max(samples1[:, 1]), np.max(samples2[:, 1])) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([bayes_classifier(pt) for pt in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1')
    plt.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2')
    plt.title('Bayes Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    visualize_boundary_with_distributions(samples1, samples2,
                                          calculated_mean1, calculated_mean2,
                                          cov_matrix1, cov_matrix2,
                                          bayes_classifier)


if __name__ == "__main__":
    main()
