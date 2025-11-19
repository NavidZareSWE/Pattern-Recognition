import numpy as np
import matplotlib.pyplot as plt


def main():
    # Get student number for seeding randomness
    # student_number = int(input("Enter your student number: "))
    student_number = 40435071
    np.random.seed(student_number)

    # Generate samples for 2 classes (you can extend to 3 if desired)
    # Class 1: Mean [2, 2], Covariance [[1, 0.5], [0.5, 1]]
    # Class 2: Mean [5, 5], Covariance [[1, -0.5], [-0.5, 1]]
    num_samples = 100
    mean1 = np.array([2, 2])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)

    mean2 = np.array([5, 5])
    cov2 = np.array([[1, -0.5], [-0.5, 1]])
    samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)

    # ## Done ##: Calculate the mean for each class from scratch
    # Expected output type: 1D numpy array of shape (2,)
    def calculate_mean(data):
        sums = np.zeros(2)
        n = data.shape[0]

        for col_idx, col in enumerate(data.T):
            col_sum = 0
            for val in col:
                col_sum += val
            sums[col_idx] = col_sum

        return sums / n

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)

    # ## Done ##: Calculate the covariance matrix for each class from scratch
    # Use np.dot for matrix multiplication, but no np.cov
    # Expected output type: 2x2 numpy array
    def calculate_covariance(data, mean):
        n = data.shape[0]           # Num rows  = (n_samples)
        diff = data - mean
        cov = np.dot(diff.T, diff) / n
        return cov

    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    # ## Done ##: Calculate eigenvalues for each covariance matrix from scratch
    # Expected output: 1D array of 2 values, sorted descending
    def calculate_eigenvalues(cov):
        def find_2D_determinant(_2d_matrix):
            return (_2d_matrix[0][0] * _2d_matrix[1][1]) - (_2d_matrix[0][1] * _2d_matrix[1][0])
        # Characteristic equation: det(cov - λI) = 0
        # λ^ 2 – diagonal_sum·λ + det = 0
        det = find_2D_determinant(cov)
        diagonal_sum = cov[0, 0] + cov[1, 1]
        delta = np.sqrt((diagonal_sum ** 2) - (4 * det * 1))
        if delta >= 0:
            lambda_1 = (diagonal_sum + delta) / 2 * 1
            lambda_2 = (diagonal_sum - delta) / 2 * 1
            return np.array(sorted([lambda_1, lambda_2], reverse=True))
        return None

    eigenvalues1 = calculate_eigenvalues(cov_matrix1)
    eigenvalues2 = calculate_eigenvalues(cov_matrix2)

    # ## Done ##: Calculate eigenvectors for each covariance matrix from scratch
    # Expected output: 2x2 array, columns are eigenvectors corresponding to eigenvalues

    def calculate_eigenvectors(cov, eigenvalues):
        # [a    b]
        # [c    d]
        a, b = cov[0, 0], cov[0, 1]
        c, d = cov[1, 0], cov[1, 1]

        eigenvectors = np.zeros((2, 2))
        zero_threshhold = 1e-10
        for i, lam in enumerate(eigenvalues):
            # (A - λI)v = 0

            # b != 0
            # (a - λ) * v1 + b * v2 = 0
            # v2 = (λ - a ) * v1 / b
            # if v1 = b => v2 = (λ - a)
            if abs(b) > zero_threshhold:
                v = np.array([b, lam - a])
            # c != 0
            # (d - λ) * v1 + c * v2 = 0
            # v1 = (λ - d) * v2 / c
            # if v2 = c => v2 = (λ - d)
            elif abs(c) > zero_threshhold:
                v = np.array([lam - d, c])
            else:
                # Even if λ - a = 0, that’s fine.
                # Then v2 = 0 and the eigenvector is along [1,0].
                if i == 0:
                    v = np.array([1, 0])
                # Even if λ - d = 0, that’s fine.
                # Then v1 = 0 and the eigenvector is along [0,1].
                else:
                    v = np.array([0, 1])

            # Normalize
            v = v / np.linalg.norm(v)
            eigenvectors[:, i] = v

        return eigenvectors

    eigenvectors1 = calculate_eigenvectors(cov_matrix1, eigenvalues1)
    eigenvectors2 = calculate_eigenvectors(cov_matrix2, eigenvalues2)

    print("Class 1 Mean:\n", calculated_mean1)
    print("Class 1 Covariance Matrix:\n", cov_matrix1)
    print("Class 1 Eigenvalues:\n", eigenvalues1)
    print("Class 1 Eigenvectors:\n", eigenvectors1)

    print("Class 2 Mean:\n", calculated_mean2)
    print("Class 2 Covariance Matrix:\n", cov_matrix2)
    print("Class 2 Eigenvalues:\n", eigenvalues2)
    print("Class 2 Eigenvectors:\n", eigenvectors2)

    # Visualization (provided): Scatter plot with eigenvectors for each class
    if all(v is not None for v in [calculated_mean1, cov_matrix1, eigenvalues1, eigenvectors1, calculated_mean2, cov_matrix2, eigenvalues2, eigenvectors2]):
        plt.figure(figsize=(8, 6))
        plt.scatter(samples1[:, 0], samples1[:, 1],
                    color='blue', label='Class 1')
        plt.scatter(samples2[:, 0], samples2[:, 1],
                    color='red', label='Class 2')

        # Plot eigenvectors for Class 1 from its mean
        for i in range(2):
            plt.arrow(calculated_mean1[0], calculated_mean1[1],
                      eigenvectors1[0, i] * np.sqrt(eigenvalues1[i]),
                      eigenvectors1[1, i] * np.sqrt(eigenvalues1[i]),
                      color='cyan', width=0.1, head_width=0.3)

        # Plot eigenvectors for Class 2 from its mean
        for i in range(2):
            plt.arrow(calculated_mean2[0], calculated_mean2[1],
                      eigenvectors2[0, i] * np.sqrt(eigenvalues2[i]),
                      eigenvectors2[1, i] * np.sqrt(eigenvalues2[i]),
                      color='magenta', width=0.1, head_width=0.3)

        plt.title('Scatter Plot with Eigenvectors for Each Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Complete the TODOs to see the visualization.")


if __name__ == "__main__":
    main()
