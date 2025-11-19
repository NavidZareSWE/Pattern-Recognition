import numpy as np
import matplotlib.pyplot as plt

def main():
    # Get student number for seeding randomness
    student_number = int(input("Enter your student number: "))
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

    # ## TODO ##: Calculate the mean for each class from scratch
    # Expected output type: 1D numpy array of shape (2,)
    def calculate_mean(data):
        return None  # Replace with your code

    calculated_mean1 = calculate_mean(samples1)
    calculated_mean2 = calculate_mean(samples2)

    # ## TODO ##: Calculate the covariance matrix for each class from scratch
    # Use np.dot for matrix multiplication, but no np.cov
    # Expected output type: 2x2 numpy array
    def calculate_covariance(data, mean):
        return None  # Replace with your code

    cov_matrix1 = calculate_covariance(samples1, calculated_mean1)
    cov_matrix2 = calculate_covariance(samples2, calculated_mean2)

    # ## TODO ##: Calculate eigenvalues for each covariance matrix from scratch
    # Expected output: 1D array of 2 values, sorted descending
    def calculate_eigenvalues(cov):
        return None  # Replace with your code

    eigenvalues1 = calculate_eigenvalues(cov_matrix1)
    eigenvalues2 = calculate_eigenvalues(cov_matrix2)

    # ## TODO ##: Calculate eigenvectors for each covariance matrix from scratch
    # Expected output: 2x2 array, columns are eigenvectors corresponding to eigenvalues
    def calculate_eigenvectors(cov, eigenvalues):
        return None  # Replace with your code

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
        plt.scatter(samples1[:, 0], samples1[:, 1], color='blue', label='Class 1')
        plt.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Class 2')

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