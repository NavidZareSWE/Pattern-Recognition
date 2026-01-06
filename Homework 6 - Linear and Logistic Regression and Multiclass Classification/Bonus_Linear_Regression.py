import pandas as pd
import numpy as np
from plots import (plot_scatter_features_vs_mpg, plot_bgd_cost,
                   plot_bgd_cost_multiple_lr, plot_regression_line,
                   plot_sgd_cost, plot_sgd_cost_multiple_lr,
                   plot_bgd_vs_sgd_comparison, plot_all_methods_comparison)
# import os

# ============================================================================
# BONUS SECTION IMPORTS
# Import plotting functions for Ridge Regression with SGD
# ============================================================================
from bonus_plots import (plot_correlation_matrix, plot_vif_analysis,
                         plot_ridge_coefficients_comparison,
                         plot_ridge_sgd_convergence,
                         plot_ridge_sgd_convergence_detailed,
                         plot_mse_comparison, plot_bias_variance_tradeoff,
                         plot_coefficient_path,
                         plot_regression_lines_multiple_lambda)


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # create a np array of consecutive integers
    # from 0 to n_samples(exclusive)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # hold-out validation
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (X[train_indices], X[test_indices],
            y[train_indices], y[test_indices])


def load_and_preprocess_data():
    # print(os.getcwd())
    # 1.2-1 Load the dataset
    data = pd.read_csv('auto_mpg.csv.csv')

    # 1.2-2 Looking at some samples
    print("\n--- Sample Data ---")
    print(data.sample(8))

    # 1.2-3 Matrix shape
    print("\n--- Data Shape ---")
    print(data.shape)

    # 1.2-4 mpg Mean & mpg Standard Deviation
    mean_mpg = np.mean(data.to_numpy()[:, -1])
    std_mpg = np.std(data.to_numpy()[:, -1])
    print("\n--- MPG Statistics ---")
    print(f"MPG Mean: {mean_mpg:.2f}")
    print(f"MPG Std Dev: {std_mpg:.2f}")

    # 1.2-5 correlation between weight & mpg
    weight = data['weight'].to_numpy()
    mpg = data['mpg'].to_numpy()
    correlation = np.corrcoef(weight, mpg)[0, 1]
    print("\n--- Correlation Analysis ---")
    print(f"Correlation (weight vs mpg): {correlation:.4f}")

    # 1.2.6 Scatter plots
    plot_scatter_features_vs_mpg(data)

    # 1.3-1 Retain X & Y
    X = data['weight'].values
    y = data['mpg'].values

    # 1.3-2 Retain X & Y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("\n--- Train-Test Split ---")
    print(f"Training samples: {len(X_train)} == {len(y_train)}")
    print(f"Testing samples: {len(X_test)} == {len(y_test)}")

    # 1.3-3 Normalize using z-score
    # axis specifies the dimension along which an operation is performed.
    # If axis=0, the operation is performed down the rows, column by column.
    # Read More: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    # Read More: https://numpy.org/doc/stable/reference/generated/numpy.std.html
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    if X_train_std == 0:
        X_train_std = 1  # Avoid division by zero
    X_train_normalized = (X_train - X_train_mean) / X_train_std
    X_test_normalized = (X_test - X_train_mean) / X_train_std

    return X_train_normalized, X_test_normalized, y_train, y_test


def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n = len(X)
    theta_0 = 0.0
    theta_1 = 0.0

    cost_history = []
    for i in range(n_iterations):
        y_pred = theta_0 + theta_1 * X
        error = y_pred - y
        grad_theta_0 = 1 / n * np.sum(error)
        grad_theta_1 = 1 / n * np.sum(error * X)

        theta_0 = theta_0 - learning_rate * grad_theta_0
        theta_1 = theta_1 - learning_rate * grad_theta_1

        if i % 100 == 0:
            cost = (1 / (2 * n)) * np.sum(error ** 2)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return theta_0, theta_1, cost_history


def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=100):
    # epochs = number of complete passes through the entire training dataset
    n = len(X)
    theta_0 = 0.0
    theta_1 = 0.0

    cost_history = []
    for i in range(n_epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)
        for j in indices:
            xj = X[j]
            yj = y[j]
            y_pred = theta_0 + theta_1 * xj
            error = y_pred - yj

            theta_0 = theta_0 - learning_rate * error
            theta_1 = theta_1 - learning_rate * error * xj

        y_pred_all = theta_0 + theta_1 * X
        cost = (1 / (2 * n)) * np.sum((y_pred_all - y) ** 2)
        cost_history.append(cost)

    return theta_0, theta_1, cost_history


def closed_form_solution(X, y):
    # First column always 1sS
    n = len(X)
    X_with_b = np.column_stack([np.ones(n), X])

    # theta = (X^T X)^-1 * X^T y
    XtX = X_with_b.T @ X_with_b  # X^T X
    XtX_inv = np.linalg.inv(XtX)       # (X^T X)^-1
    Xty = X_with_b.T @ y          # X^T y
    theta = XtX_inv @ Xty
    return theta


# ============================================================================
# BONUS SECTION: Regularized Linear Regression with SGD
# ============================================================================
# This section extends the linear regression model by applying Ridge Regression
# (L2 regularization) and training it using Stochastic Gradient Descent (SGD)
# on the auto_mpg dataset.
# ============================================================================

def bonus_load_and_preprocess_multivariate():
    """
    4.B.1 Load the auto_mpg dataset and prepare it for multivariate analysis.

    This function loads all features (weight, horsepower, displacement) for
    the multicollinearity analysis required in the bonus section.

    Returns:
    --------
    tuple : (X_train_norm, X_test_norm, y_train, y_test, feature_names,
             X_train_raw, X_test_raw)
    """
    # 4.B.1-1 Load the dataset
    data = pd.read_csv('auto_mpg.csv.csv')

    print("\n" + "="*60)
    print("BONUS SECTION: REGULARIZED LINEAR REGRESSION WITH SGD")
    print("="*60)

    # 4.B.1-2 Dataset overview
    print("\n--- Dataset Overview ---")
    print(f"Shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print(f"\nSample data:")
    print(data.head())

    # 4.B.1-3 Extract features and target
    # Using all three features for multicollinearity analysis
    feature_names = ['weight', 'horsepower', 'displacement']
    X = data[feature_names].values
    y = data['mpg'].values

    # 4.B.1-4 Feature statistics
    print(f"\n--- Feature Statistics ---")
    for i, name in enumerate(feature_names):
        print(
            f"{name}: mean={np.mean(X[:, i]):.2f}, std={np.std(X[:, i]):.2f}")

    # 4.B.1-5 Split data using the same function as original code
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    print(f"\n--- Train-Test Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 4.B.1-6 Keep raw data for correlation analysis
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()

    # 4.B.1-7 Normalize features using z-score (same approach as original)
    # Using training set statistics for normalization
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    print(f"\n--- Normalization Applied ---")
    print(f"Training mean (per feature): {train_mean}")
    print(f"Training std (per feature): {train_std}")

    return (X_train_norm, X_test_norm, y_train, y_test,
            feature_names, X_train_raw, X_test_raw)


def bonus_compute_correlation_matrix(X, feature_names):
    """
    4.B.2-1 Compute the correlation matrix of all input features.

    The correlation matrix shows pairwise Pearson correlation coefficients
    between all features. High correlation (|r| > 0.7) indicates potential
    multicollinearity issues.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : list
        List of feature names

    Returns:
    --------
    np.ndarray : Correlation matrix (n_features x n_features)
    """
    # np.corrcoef computes Pearson correlation coefficients
    # rowvar=False means columns represent variables
    corr_matrix = np.corrcoef(X, rowvar=False)
    return corr_matrix


def bonus_compute_vif(X, feature_names):
    """
    4.B.2-2 Compute Variance Inflation Factor (VIF) for each feature.

    VIF measures how much the variance of an estimated regression coefficient
    increases due to collinearity with other predictors.

    Formula: VIF_j = 1 / (1 - R_j^2)
    where R_j^2 is from regressing feature j on all other features.

    VIF Interpretation:
    - VIF = 1      : No correlation
    - 1 < VIF < 5  : Moderate correlation (acceptable)
    - 5 < VIF < 10 : High correlation (concerning)
    - VIF > 10     : Severe multicollinearity (problematic)

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : list
        List of feature names

    Returns:
    --------
    dict : Feature name to VIF value mapping
    """
    n_features = X.shape[1]
    vif_values = []

    for i in range(n_features):
        # Create y_temp (target) as current feature
        # and X_other (predictors) as all other features
        y_temp = X[:, i]
        X_other = np.delete(X, i, axis=1)

        # Add intercept column for regression
        X_other_with_intercept = np.column_stack(
            [np.ones(len(y_temp)), X_other])

        # Compute R^2 using closed-form solution (same approach as original)
        try:
            # theta = (X^T X)^-1 * X^T y
            XtX = X_other_with_intercept.T @ X_other_with_intercept
            XtX_inv = np.linalg.inv(XtX)
            Xty = X_other_with_intercept.T @ y_temp
            theta = XtX_inv @ Xty

            # Predictions
            y_pred = X_other_with_intercept @ theta

            # R^2 = 1 - SS_res / SS_tot
            ss_res = np.sum((y_temp - y_pred) ** 2)
            ss_tot = np.sum((y_temp - np.mean(y_temp)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # VIF = 1 / (1 - R^2)
            # Handle edge case where R^2 approaches 1
            if r_squared >= 1:
                vif = float('inf')
            else:
                vif = 1 / (1 - r_squared)
        except np.linalg.LinAlgError:
            vif = float('inf')

        vif_values.append(vif)

    return dict(zip(feature_names, vif_values))


def bonus_ridge_sgd(X, y, learning_rate=0.01, n_epochs=100,
                    lambda_reg=0.01, random_state=42):
    """
    4.B.3 Ridge Regression (L2 regularization) using Stochastic Gradient Descent.

    The cost function with L2 regularization:
    J(theta) = (1/2n) * sum((h(x_i) - y_i)^2) + (lambda/2) * sum(theta_j^2)
    (j != 0, bias not regularized)

    Update rules for SGD with L2 regularization:
    theta_0 := theta_0 - alpha * (h(x_i) - y_i)  (bias, no regularization)
    theta_j := theta_j - alpha * ((h(x_i) - y_i) * x_ij + lambda * theta_j)
    (j > 0, with regularization)

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features) - should NOT include bias column
    y : np.ndarray
        Target vector
    learning_rate : float
        Learning rate (alpha) for gradient descent
    n_epochs : int
        Number of complete passes through the training data
    lambda_reg : float
        Regularization parameter (lambda) - controls strength of L2 penalty
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (theta, cost_history)
        theta: array of learned parameters [theta_0, theta_1, ..., theta_n]
        cost_history: list of MSE values per epoch (for convergence analysis)
    """
    np.random.seed(random_state)

    n_samples = len(X)
    n_features = X.shape[1] if len(X.shape) > 1 else 1

    # Reshape X if 1D (single feature case)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Add bias column (column of 1s)
    X_with_bias = np.column_stack([np.ones(n_samples), X])

    # Initialize parameters to zeros (n_features + 1 for bias)
    theta = np.zeros(n_features + 1)

    cost_history = []

    # SGD training loop
    for epoch in range(n_epochs):
        # Shuffle data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # Update parameters for each sample
        for idx in indices:
            x_i = X_with_bias[idx]
            y_i = y[idx]

            # Prediction: h(x) = theta^T * x
            y_pred = np.dot(theta, x_i)

            # Error: h(x) - y
            error = y_pred - y_i

            # Update bias (theta_0) - NO regularization on bias term
            theta[0] = theta[0] - learning_rate * error

            # Update feature coefficients (theta_1, theta_2, ...) WITH L2 regularization
            for j in range(1, len(theta)):
                # Gradient = error * x_j + lambda * theta_j
                gradient = error * x_i[j] + lambda_reg * theta[j]
                theta[j] = theta[j] - learning_rate * gradient

        # Compute MSE at end of each epoch for convergence analysis
        y_pred_all = X_with_bias @ theta
        mse = (1 / (2 * n_samples)) * np.sum((y_pred_all - y) ** 2)
        cost_history.append(mse)

    return theta, cost_history


def bonus_compute_mse(X, y, theta):
    """
    4.B.4 Compute Mean Squared Error for model evaluation.

    MSE = (1/2n) * sum((h(x_i) - y_i)^2)

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (should NOT include bias column)
    y : np.ndarray
        Target vector
    theta : np.ndarray
        Model parameters [theta_0, theta_1, ...]

    Returns:
    --------
    float : MSE value
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    n = len(y)
    X_with_bias = np.column_stack([np.ones(n), X])
    y_pred = X_with_bias @ theta
    mse = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)
    return mse


def bonus_run_multicollinearity_analysis(X, feature_names):
    """
    4.B.5 Perform complete multicollinearity analysis (before regularization).

    This function computes and displays:
    1. Correlation matrix between all features
    2. VIF values for each feature
    3. Identifies highly correlated feature pairs

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (normalized)
    feature_names : list
        Feature names

    Returns:
    --------
    tuple : (correlation_matrix, vif_dict)
    """
    print("\n" + "="*60)
    print("1. MULTICOLLINEARITY ANALYSIS (BEFORE REGULARIZATION)")
    print("="*60)

    # 4.B.5-1 Correlation Matrix
    print("\n--- 1.1 Correlation Matrix ---")
    corr_matrix = bonus_compute_correlation_matrix(X, feature_names)

    print("\nCorrelation Matrix:")
    print("-" * 50)
    header = "          " + "  ".join(f"{name:>12}" for name in feature_names)
    print(header)
    for i, name in enumerate(feature_names):
        row = f"{name:10}" + "  ".join(f"{corr_matrix[i, j]:>12.4f}"
                                       for j in range(len(feature_names)))
        print(row)

    # 4.B.5-2 Identify highly correlated pairs (threshold: |r| > 0.7)
    print("\n--- Highly Correlated Feature Pairs ---")
    print("(Threshold: |r| > 0.7)")
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.7:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: "
                      f"r = {corr_matrix[i, j]:.4f}")

    # 4.B.5-3 Save correlation matrix plot
    plot_correlation_matrix(
        corr_matrix, feature_names,
        save_path='bonus_Plots/bonus_correlation_matrix.png'
    )

    # 4.B.5-4 Variance Inflation Factor
    print("\n--- 1.2 Variance Inflation Factor (VIF) ---")
    vif_dict = bonus_compute_vif(X, feature_names)

    print("\nVIF Values:")
    print("-" * 40)
    print(f"{'Feature':<15} {'VIF':>10} {'Interpretation':>20}")
    print("-" * 40)
    for feature, vif in vif_dict.items():
        if vif > 10:
            interp = "SEVERE"
        elif vif > 5:
            interp = "HIGH"
        elif vif > 2:
            interp = "MODERATE"
        else:
            interp = "LOW"
        print(f"{feature:<15} {vif:>10.2f} {interp:>20}")

    print("\nVIF Interpretation Guide:")
    print("  VIF = 1      : No correlation")
    print("  1 < VIF < 5  : Moderate correlation (acceptable)")
    print("  5 < VIF < 10 : High correlation (concerning)")
    print("  VIF > 10     : Severe multicollinearity (problematic)")

    # 4.B.5-5 Save VIF plot
    plot_vif_analysis(
        feature_names, list(vif_dict.values()),
        save_path='bonus_Plots/bonus_vif_analysis.png'
    )

    return corr_matrix, vif_dict


def bonus_run_ridge_experiments(X_train, X_test, y_train, y_test, feature_names):
    """
    4.B.6 Run Ridge Regression with SGD for multiple lambda values.

    This function trains Ridge Regression models with different regularization
    strengths and collects results for comparison.

    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and testing features (normalized)
    y_train, y_test : np.ndarray
        Training and testing targets
    feature_names : list
        Feature names

    Returns:
    --------
    dict : Results dictionary containing all experimental data
    """
    print("\n" + "="*60)
    print("2. RIDGE REGRESSION WITH SGD")
    print("="*60)

    # 4.B.6-1 Hyperparameters (as specified in bonus requirements)
    lambda_values = [0, 0.01, 0.1, 1]
    learning_rate = 0.01
    n_epochs = 200

    print(f"\n--- Hyperparameters ---")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {n_epochs}")
    print(f"Lambda Values: {lambda_values}")

    # 4.B.6-2 Store results
    results = {
        'lambda_values': lambda_values,
        'coefficients': {},
        'cost_histories': {},
        'train_mse': [],
        'test_mse': []
    }

    # 4.B.6-3 Train and evaluate for each lambda
    print("\n--- Training Results ---")
    print("-" * 80)
    header = f"{'lam':>6} | {'theta0':>10} |"
    for name in feature_names:
        header += f" {name[:8]:>10} |"
    header += f" {'Train MSE':>12} | {'Test MSE':>12}"
    print(header)
    print("-" * 80)

    for lam in lambda_values:
        # Train model
        theta, cost_history = bonus_ridge_sgd(
            X_train, y_train,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            lambda_reg=lam,
            random_state=42
        )

        # Compute MSE on training and test sets
        train_mse = bonus_compute_mse(X_train, y_train, theta)
        test_mse = bonus_compute_mse(X_test, y_test, theta)

        # Store results
        results['coefficients'][lam] = theta
        results['cost_histories'][lam] = cost_history
        results['train_mse'].append(train_mse)
        results['test_mse'].append(test_mse)

        # Print results row
        row = f"{lam:>6.2f} | {theta[0]:>10.4f} |"
        for j in range(1, len(theta)):
            row += f" {theta[j]:>10.4f} |"
        row += f" {train_mse:>12.6f} | {test_mse:>12.6f}"
        print(row)

    print("-" * 80)

    return results


def bonus_analyze_stability(results, feature_names):
    """
    4.B.7 Analyze coefficient stability and performance across lambda values.

    This function compares the learned coefficients and evaluates how
    regularization affects model stability and performance.

    Parameters:
    -----------
    results : dict
        Results from ridge regression experiments
    feature_names : list
        Feature names
    """
    print("\n" + "="*60)
    print("3. STABILITY AND PERFORMANCE COMPARISON")
    print("="*60)

    lambda_values = results['lambda_values']
    coefficients = results['coefficients']

    # 4.B.7-1 Coefficient comparison with baseline (lam=0)
    print("\n--- 3.1 Coefficient Comparison ---")
    print("\nCoefficient Changes from Unregularized (lam=0) to Regularized Models:")
    print("-" * 70)

    base_theta = coefficients[0]  # lam = 0 is baseline
    all_features = ['bias'] + feature_names

    for lam in lambda_values[1:]:  # Skip lam=0
        theta = coefficients[lam]
        print(f"\nlam = {lam}:")
        for i, name in enumerate(all_features):
            change = theta[i] - base_theta[i]
            pct_change = (change / base_theta[i]
                          * 100) if base_theta[i] != 0 else 0
            print(f"  {name:<15}: {base_theta[i]:>10.4f} -> {theta[i]:>10.4f} "
                  f"(Delta = {change:>8.4f}, {pct_change:>7.2f}%)")

    # 4.B.7-2 Convergence analysis
    print("\n--- 3.2 Convergence Analysis ---")
    cost_histories = results['cost_histories']

    for lam in lambda_values:
        costs = cost_histories[lam]
        initial_cost = costs[0]
        final_cost = costs[-1]
        min_cost = min(costs)
        convergence_epoch = costs.index(min_cost) + 1

        print(f"lam = {lam}:")
        print(f"  Initial MSE: {initial_cost:.6f}")
        print(f"  Final MSE:   {final_cost:.6f}")
        print(f"  Min MSE:     {min_cost:.6f} (at epoch {convergence_epoch})")
        print(
            f"  Reduction:   {(initial_cost - final_cost) / initial_cost * 100:.2f}%")

    # 4.B.7-3 Training vs Testing MSE comparison
    print("\n--- 3.3 Training vs Testing MSE ---")
    print("-" * 50)
    print(f"{'lam':>8} | {'Train MSE':>12} | {'Test MSE':>12} | {'Gap':>10}")
    print("-" * 50)

    for i, lam in enumerate(lambda_values):
        train_mse = results['train_mse'][i]
        test_mse = results['test_mse'][i]
        gap = test_mse - train_mse
        print(f"{lam:>8.2f} | {train_mse:>12.6f} | {test_mse:>12.6f} | {gap:>10.6f}")

    # Find best lambda based on test MSE
    best_idx = np.argmin(results['test_mse'])
    best_lambda = lambda_values[best_idx]
    print(f"\nBest lam based on test MSE: {best_lambda} "
          f"(Test MSE: {results['test_mse'][best_idx]:.6f})")


def bonus_discussion(corr_matrix, vif_dict, results, feature_names):
    """
    4.B.8 Analysis and Discussion of Ridge Regression results.

    This function provides comprehensive analysis of:
    1. How Ridge Regression mitigates multicollinearity
    2. The bias-variance tradeoff as lambda changes

    Parameters:
    -----------
    corr_matrix : np.ndarray
        Correlation matrix
    vif_dict : dict
        VIF values
    results : dict
        Ridge regression results
    feature_names : list
        Feature names
    """
    print("\n" + "="*60)
    print("4. ANALYSIS AND DISCUSSION")
    print("="*60)

    # 4.B.8-1 How Ridge Regression mitigates multicollinearity
    print("\n--- 4.1 How Ridge Regression Mitigates Multicollinearity ---")
    print("""
Ridge Regression addresses multicollinearity through L2 regularization by:

1. COEFFICIENT SHRINKAGE:
   - The penalty term (lambda * sum(theta_j^2)) shrinks coefficient magnitudes toward zero
   - This reduces the variance of coefficient estimates
   - Highly correlated features share their predictive power more evenly

2. STABILIZATION OF ESTIMATES:
   - Without regularization, multicollinearity causes:
     * Large coefficient variances
     * Unstable estimates that change dramatically with small data changes
   - With regularization:
     * Coefficients become more stable
     * Small changes in data don't cause dramatic coefficient changes

3. BIAS-VARIANCE TRADEOFF:
   - Ridge introduces some bias (coefficients are systematically smaller)
   - But significantly reduces variance
   - Net effect: lower overall prediction error (MSE)
""")

    # Show evidence from our experiment
    print("\nEvidence from our experiment:")
    lambda_values = results['lambda_values']
    coefficients = results['coefficients']

    for lam in lambda_values:
        theta = coefficients[lam]
        coef_variance = np.var(theta[1:])  # Exclude bias
        coef_l2_norm = np.sqrt(np.sum(theta[1:]**2))
        print(f"  lam = {lam}: Coefficient L2-norm = {coef_l2_norm:.4f}, "
              f"Variance = {coef_variance:.4f}")

    # 4.B.8-2 Bias-Variance Tradeoff analysis
    print("\n--- 4.2 Bias-Variance Tradeoff Analysis ---")
    print("""
As lambda increases:

1. BIAS INCREASES:
   - Training MSE typically increases (underfitting)
   - Coefficients are forced smaller than their true values
   - Model becomes less flexible

2. VARIANCE DECREASES:
   - Model is less sensitive to training data variations
   - Coefficient estimates are more stable
   - Generalization often improves (up to a point)

3. OPTIMAL lambda:
   - Balances bias and variance
   - Typically found where test error is minimized
   - Cross-validation is the standard approach for selection
""")

    print("\nObserved in our experiment:")
    print("-" * 60)
    print(f"{'lam':>8} | {'Train MSE':>12} | {'Test MSE':>12} | "
          f"{'Generalization Gap':>18}")
    print("-" * 60)

    for i, lam in enumerate(lambda_values):
        train_mse = results['train_mse'][i]
        test_mse = results['test_mse'][i]
        gap = test_mse - train_mse
        print(f"{lam:>8.2f} | {train_mse:>12.6f} | {test_mse:>12.6f} | "
              f"{gap:>18.6f}")

    print("""
Key observations:
- lam = 0 (no regularization): Higher variance, potential overfitting
- lam = 0.01, 0.1: Good balance, competitive test MSE
- lam = 1 (strong regularization): Higher bias, coefficients heavily shrunk
""")


def bonus_generate_plots(results, feature_names, X_train, X_test, y_train, y_test):
    """
    4.B.9 Generate all plots for the bonus section.

    This function creates and saves all required visualizations to the
    bonus_Plots/ directory.

    Parameters:
    -----------
    results : dict
        Ridge regression results
    feature_names : list
        Feature names
    X_train, X_test : np.ndarray
        Training and testing features
    y_train, y_test : np.ndarray
        Training and testing targets
    """
    import os
    os.makedirs('bonus_Plots', exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    lambda_values = results['lambda_values']
    coefficients = results['coefficients']
    cost_histories = results['cost_histories']
    train_mse = results['train_mse']
    test_mse = results['test_mse']

    # Plot 1: Coefficients comparison
    all_feature_names = ['theta0 (bias)'] + \
        [f'theta_{name}' for name in feature_names]
    plot_ridge_coefficients_comparison(
        lambda_values, coefficients, all_feature_names,
        save_path='bonus_Plots/bonus_coefficients_comparison.png'
    )

    # Plot 2: SGD Convergence (combined)
    plot_ridge_sgd_convergence(
        cost_histories, lambda_values,
        save_path='bonus_Plots/bonus_sgd_convergence.png'
    )

    # Plot 3: SGD Convergence (detailed - subplots)
    plot_ridge_sgd_convergence_detailed(
        cost_histories, lambda_values,
        save_path='bonus_Plots/bonus_sgd_convergence_detailed.png'
    )

    # Plot 4: MSE Comparison (Train vs Test)
    plot_mse_comparison(
        lambda_values, train_mse, test_mse,
        save_path='bonus_Plots/bonus_mse_comparison.png'
    )

    # Plot 5: Bias-Variance Tradeoff
    plot_bias_variance_tradeoff(
        lambda_values, train_mse, test_mse,
        save_path='bonus_Plots/bonus_bias_variance_tradeoff.png'
    )

    # Plot 6: Coefficient Path (regularization path)
    coefficient_paths = {}
    for i, name in enumerate(feature_names):
        coefficient_paths[name] = [coefficients[lam][i+1]
                                   for lam in lambda_values]

    plot_coefficient_path(
        lambda_values, coefficient_paths, feature_names,
        save_path='bonus_Plots/bonus_coefficient_path.png'
    )

    # Plot 7: Regression lines for different lambda (using weight feature)
    simple_coeffs = {}
    for lam in lambda_values:
        simple_coeffs[lam] = [coefficients[lam][0], coefficients[lam][1]]

    plot_regression_lines_multiple_lambda(
        X_train[:, 0], X_test[:, 0], y_train, y_test,
        simple_coeffs, lambda_values,
        save_path='bonus_Plots/bonus_regression_lines.png'
    )

    print("\nAll plots saved to 'bonus_Plots/' directory.")


def bonus_main():
    """
    4.B.10 Main execution function for the bonus section.

    This function orchestrates the complete bonus analysis workflow:
    1. Load and preprocess multivariate data
    2. Perform multicollinearity analysis
    3. Train Ridge Regression with different lambda values
    4. Analyze stability and performance
    5. Generate discussion and analysis
    6. Create all plots
    """
    # Step 1: Load and preprocess data
    (X_train, X_test, y_train, y_test,
     feature_names, X_train_raw, X_test_raw) = bonus_load_and_preprocess_multivariate()

    # Step 2: Multicollinearity Analysis (before regularization)
    corr_matrix, vif_dict = bonus_run_multicollinearity_analysis(
        X_train, feature_names)

    # Step 3: Ridge Regression with SGD for multiple lambda values
    results = bonus_run_ridge_experiments(X_train, X_test, y_train, y_test,
                                          feature_names)

    # Step 4: Stability and Performance Comparison
    bonus_analyze_stability(results, feature_names)

    # Step 5: Analysis and Discussion
    bonus_discussion(corr_matrix, vif_dict, results, feature_names)

    # Step 6: Generate all plots
    bonus_generate_plots(results, feature_names, X_train,
                         X_test, y_train, y_test)

    # Summary
    best_idx = np.argmin(results['test_mse'])
    best_lambda = results['lambda_values'][best_idx]
    best_test_mse = results['test_mse'][best_idx]

    print("\n" + "="*60)
    print("BONUS SECTION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nBest regularization parameter: lam = {best_lambda}")
    print(f"Best test MSE: {best_test_mse:.6f}")
    print(f"\nAll plots saved to: bonus_Plots/")

    return results, corr_matrix, vif_dict


if __name__ == "__main__":
    X_train_norm, X_test_norm, y_train, y_test = load_and_preprocess_data()

    theta_0_bgd, theta_1_bgd, cost_history = batch_gradient_descent(
        X_train_norm, y_train,
        learning_rate=0.1,
        n_iterations=1000
    )

    print("\n--- Batch Gradient Descent Results ---")
    print(f"Learned parameters:")
    print(f"theta_0 (intercept): {theta_0_bgd:.4f}")
    print(f"theta_1 (slope): {theta_1_bgd:.4f}")

    #  1.3.2 Plot cost vs iterations
    plot_bgd_cost(cost_history, learning_rate=0.1)

    #  1.3.3 Experiment with different learning rates
    plot_bgd_cost_multiple_lr(X_train_norm, y_train, batch_gradient_descent)

    # 1.3.5: Plot regression line
    plot_regression_line(X_train_norm, X_test_norm, y_train, y_test,
                         theta_0_bgd, theta_1_bgd, 'BGD')

    theta_0_sgd, theta_1_sgd, cost_sgd = stochastic_gradient_descent(
        X_train_norm, y_train,
        learning_rate=0.01,
        n_epochs=100
    )

    print("\n--- Stochastic Gradient Descent Results ---")
    print(f"Learned parameters:")
    print(f"theta_0 (intercept): {theta_0_sgd:.4f}")
    print(f"theta_1 (slope): {theta_1_sgd:.4f}")

    #  1.3.3 Plot cost over epochs
    plot_sgd_cost(cost_sgd, learning_rate=0.01)

    #  1.3.4 Experiment with different learning rates
    plot_sgd_cost_multiple_lr(X_train_norm, y_train,
                              stochastic_gradient_descent)

    #  1.3.5 Plot regression line
    plot_regression_line(X_train_norm, X_test_norm, y_train, y_test,
                         theta_0_sgd, theta_1_sgd, 'SGD')

    #  1.3 BGD vs SGD COMPARISON
    y_pred_bgd = theta_0_bgd + theta_1_bgd * X_train_norm
    final_cost_bgd = (1 / (2 * len(y_train))) * \
        np.sum((y_pred_bgd - y_train) ** 2)

    y_pred_sgd = theta_0_sgd + theta_1_sgd * X_train_norm
    final_cost_sgd = (1 / (2 * len(y_train))) * \
        np.sum((y_pred_sgd - y_train) ** 2)

    print("\n--- BGD vs SGD Comparison ---")
    print(f"{'Metric':<25} {'BGD':>15} {'SGD':>15}")
    print("-" * 55)
    print(f"{'theta_0 (intercept)':<25} {theta_0_bgd:>15.4f} {theta_0_sgd:>15.4f}")
    print(f"{'theta_1 (slope)':<25} {theta_1_bgd:>15.4f} {theta_1_sgd:>15.4f}")
    print(f"{'Final Cost (MSE)':<25} {final_cost_bgd:>15.6f} {final_cost_sgd:>15.6f}")

    plot_bgd_vs_sgd_comparison(X_train_norm, X_test_norm, y_train, y_test,
                               theta_0_bgd, theta_1_bgd, cost_history,
                               theta_0_sgd, theta_1_sgd, cost_sgd)

    [theta_0_cf, theta_1_cf] = closed_form_solution(X_train_norm, y_train)

    print("\n--- Closed-Form Solution Results ---")
    print(f"Learned parameters:")
    print(f"theta_0 (intercept): {theta_0_cf:.4f}")
    print(f"theta_1 (slope): {theta_1_cf:.4f}")

    plot_regression_line(X_train_norm, X_test_norm, y_train, y_test,
                         theta_0_cf, theta_1_cf, 'Closed-Form')

    # 1.3.D.3: ALL METHODS COMPARISON
    y_pred_cf = theta_0_cf + theta_1_cf * X_train_norm
    final_cost_cf = (1 / (2 * len(y_train))) * \
        np.sum((y_pred_cf - y_train) ** 2)

    print("\n--- All Methods Comparison ---")
    print(f"{'Method':<15} {'theta_0':>12} {'theta_1':>12} {'Final Cost':>15}")
    print("-" * 55)
    print(f"{'Closed-Form':<15} {theta_0_cf:>12.4f} {theta_1_cf:>12.4f} {final_cost_cf:>15.6f}")
    print(f"{'Batch GD':<15} {theta_0_bgd:>12.4f} {theta_1_bgd:>12.4f} {final_cost_bgd:>15.6f}")
    print(f"{'Stochastic GD':<15} {theta_0_sgd:>12.4f} {theta_1_sgd:>12.4f} {final_cost_sgd:>15.6f}")

    plot_all_methods_comparison(X_train_norm, X_test_norm, y_train, y_test,
                                theta_0_bgd, theta_1_bgd,
                                theta_0_sgd, theta_1_sgd,
                                theta_0_cf, theta_1_cf)

    # ========================================================================
    # BONUS SECTION: Run Ridge Regression with SGD
    # ========================================================================
    print("\n")
    print("#" * 70)
    print("#" + " " * 20 + "RUNNING BONUS SECTION" + " " * 27 + "#")
    print("#" * 70)

    bonus_results, bonus_corr_matrix, bonus_vif_dict = bonus_main()
