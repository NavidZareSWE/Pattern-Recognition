import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def main():
    # Get student number for seeding randomness
    # student_number = int(input("Enter your student number: "))
    student_number = 40435071
    np.random.seed(student_number)

    # Generate samples for 2 classes (same as before)
    num_samples = 100
    true_mean1 = np.array([2, 2])
    true_cov1 = np.array([[1, 0.5], [0.5, 1]])
    samples1 = np.random.multivariate_normal(
        true_mean1, true_cov1, num_samples)

    true_mean2 = np.array([5, 5])
    true_cov2 = np.array([[1, -0.5], [-0.5, 1]])
    samples2 = np.random.multivariate_normal(
        true_mean2, true_cov2, num_samples)

    # Calculate means and covs using functions from Part 1 (assume implemented)
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

    # Copy helper functions from Part 2
    def matrix_det(_2d_matrix):
        return (_2d_matrix[0][0] * _2d_matrix[1][1]) - (_2d_matrix[0][1] * _2d_matrix[1][0])

    def matrix_inv(cov):
        det = matrix_det(cov)
        # 1/det * [[d, -b], [-c, a]]
        return np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det

    def multivariate_gaussian_logpdf(x, mean, cov):
        dim = len(mean)
        diff = x - mean
        det_cov = matrix_det(cov)
        inv_cov = matrix_inv(cov)
        mahalanobis = np.dot(np.dot(diff, inv_cov), diff)
        return -0.5 * (dim * np.log(2 * np.pi) + np.log(det_cov) + mahalanobis)

    # Test points (generate some for classification)
    test_points = np.random.uniform(0, 7, size=(20, 2))

    # ## Done ##: Implement Maximum Likelihood (ML) classifier
    # Expected: Function that takes x (2D) and returns class (0 or 1)
    def ml_classifier(x):
        log_pdf_1 = multivariate_gaussian_logpdf(
            x, calculated_mean1, cov_matrix1)
        log_pdf_2 = multivariate_gaussian_logpdf(
            x, calculated_mean2, cov_matrix2)
        score = log_pdf_1 - log_pdf_2
        if score >= 0:
            return 0
        return 1

    # ## Done ##: Implement Maximum A Posteriori (MAP) classifier
    prior1 = 0.7
    prior2 = 0.3

    def map_classifier(x):
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

    # ## Done ##: Implement Risk-based MAP (Minimum Risk) classifier
    loss = np.array([[0, 1], [10, 0]])

    def risk_map_classifier(x):
        c11, c12 = loss[0]
        c21, c22 = loss[1]
        risk_prior_ratio = ((c21 - c11) * prior1) / ((c12 - c22) * prior2)
        log_risk_prior_ratio = np.log(risk_prior_ratio)
        log_pdf_1 = multivariate_gaussian_logpdf(
            x, calculated_mean1, cov_matrix1)
        log_pdf_2 = multivariate_gaussian_logpdf(
            x, calculated_mean2, cov_matrix2)
        score = log_risk_prior_ratio + (log_pdf_1 - log_pdf_2)
        if score >= 0:
            return 0
        return 1
# TEST

    # Classify test points
    ml_preds = [ml_classifier(pt) for pt in test_points]
    map_preds = [map_classifier(pt) for pt in test_points]
    risk_preds = [risk_map_classifier(pt) for pt in test_points]

    print("ML Predictions:", ml_preds)  # Expected: list of 0s and 1s
    print("MAP Predictions:", map_preds)
    print("Risk MAP Predictions:", risk_preds)

    ml_preds = np.array(ml_preds)
    map_preds = np.array(map_preds)
    risk_preds = np.array(risk_preds)
    # Pairwise comparisons
    same_ml_map = np.sum(ml_preds == map_preds)
    same_ml_risk = np.sum(ml_preds == risk_preds)
    same_map_risk = np.sum(map_preds == risk_preds)

    diff_ml_map = np.sum(ml_preds != map_preds)
    diff_ml_risk = np.sum(ml_preds != risk_preds)
    diff_map_risk = np.sum(map_preds != risk_preds)

    all_three_same = np.sum(
        (ml_preds == map_preds) & (map_preds == risk_preds)
    )

    print("\n=== Decision Agreement Summary ===")
    print(f"ML vs MAP:        Same = {same_ml_map}, Different = {diff_ml_map}")
    print(
        f"ML vs Risk-MAP:   Same = {same_ml_risk}, Different = {diff_ml_risk}")
    print(
        f"MAP vs Risk-MAP:  Same = {same_map_risk}, Different = {diff_map_risk}")
    print(
        f"\nAll three agree on {all_three_same} out of {len(test_points)} test points.\n")

    # Visualization (provided): Plot for MAP as example
    x_min, x_max = 0, 7
    y_min, y_max = 0, 7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([map_classifier(pt) for pt in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(samples1[:, 0], samples1[:, 1],
                color='blue', label='Class 1', alpha=0.5)
    plt.scatter(samples2[:, 0], samples2[:, 1],
                color='red', label='Class 2', alpha=0.5)
    plt.scatter(test_points[:, 0], test_points[:, 1],
                color='green', marker='x', label='Test Points')
    plt.title('MAP Decision Boundary with Samples and Test Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
############################################
    # Create comparison figure
    ############################################
    print("\n=== Generating Comparison Plot ===")

    # Recalculate for comparison with higher resolution
    xx_comp, yy_comp = np.meshgrid(np.arange(x_min-1, x_max+1, 0.05),
                                   np.arange(y_min-1, y_max+1, 0.05))
    grid_comp = np.c_[xx_comp.ravel(), yy_comp.ravel()]

    # Calculate decisions for all classifiers
    Z_ml = np.array([ml_classifier(pt)
                    for pt in grid_comp]).reshape(xx_comp.shape)
    Z_map = np.array([map_classifier(pt)
                     for pt in grid_comp]).reshape(xx_comp.shape)
    Z_risk = np.array([risk_map_classifier(pt)
                      for pt in grid_comp]).reshape(xx_comp.shape)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot individual classifiers
    classifiers_data = [
        (Z_ml, 'ML (Maximum Likelihood)', axes[0]),
        (Z_map, f'MAP (Priors: {prior1:.1f}, {prior2:.1f})', axes[1]),
        #     loss = np.array([[0, 1], [10, 0]])
        (Z_risk, f'Risk-based (Loss: [[0,1],[10,0]])', axes[2])
    ]

    for Z, title, ax in classifiers_data:
        ax.contourf(xx_comp, yy_comp, Z, cmap=cmap_light, alpha=0.8)
        ax.contour(xx_comp, yy_comp, Z, colors='black',
                   linewidths=2, levels=[0.5])
        ax.scatter(samples1[:, 0], samples1[:, 1],
                   color='blue', label='Class 1', alpha=0.5, s=30)
        ax.scatter(samples2[:, 0], samples2[:, 1],
                   color='red', label='Class 2', alpha=0.5, s=30)
        ax.scatter(test_points[:, 0], test_points[:, 1],
                   color='green', marker='x', label='Test', s=100, linewidths=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot agreement/disagreement regions
    ax4 = axes[3]

    # Create decision comparison map
    decision_map = np.zeros_like(Z_ml, dtype=int)
    for i in range(xx_comp.shape[0]):
        for j in range(xx_comp.shape[1]):
            ml_dec = Z_ml[i, j]
            map_dec = Z_map[i, j]
            risk_dec = Z_risk[i, j]

            if ml_dec == map_dec == risk_dec:
                decision_map[i, j] = ml_dec  # All agree (0 or 1)
            elif ml_dec == map_dec != risk_dec:
                decision_map[i, j] = 2  # Risk differs
            elif ml_dec == risk_dec != map_dec:
                decision_map[i, j] = 3  # MAP differs
            elif map_dec == risk_dec != ml_dec:
                decision_map[i, j] = 4  # ML differs

    # Custom colormap
    colors = ['#6495ED', '#FA8072', '#FFD700', '#98FB98', '#DDA0DD']
    cmap_custom = ListedColormap(colors)

    ax4.contourf(xx_comp, yy_comp, decision_map,
                 levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap=cmap_custom)
    ax4.scatter(samples1[:, 0], samples1[:, 1],
                color='blue', alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
    ax4.scatter(samples2[:, 0], samples2[:, 1],
                color='red', alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
    ax4.scatter(test_points[:, 0], test_points[:, 1],
                color='green', marker='x', s=100, linewidths=2)

    ax4.set_title('Agreement/Disagreement Regions', fontsize=11)
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.grid(True, alpha=0.3)

    # Legend for agreement map
    legend_elements = [
        mpatches.Patch(color='#6495ED', label='All agree → Class 0'),
        mpatches.Patch(color='#FA8072', label='All agree → Class 1'),
        mpatches.Patch(color='#FFD700', label='Only Risk differs'),
        mpatches.Patch(color='#98FB98', label='Only MAP differs'),
        mpatches.Patch(color='#DDA0DD', label='Only ML differs')
    ]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.suptitle(
        'Comparison of Decision Boundaries: ML vs MAP vs Risk-based', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
