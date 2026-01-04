import pandas as pd
import numpy as np
from plots import (plot_scatter_features_vs_mpg, plot_bgd_cost,
                   plot_bgd_cost_multiple_lr, plot_regression_line,
                   plot_sgd_cost, plot_sgd_cost_multiple_lr,
                   plot_bgd_vs_sgd_comparison, plot_all_methods_comparison)
# import os


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

    # θ = (XᵀX)⁻¹ · Xᵀy
    XtX = X_with_b.T @ X_with_b  # XᵀX
    XtX_inv = np.linalg.inv(XtX)       # (XᵀX)⁻¹
    Xty = X_with_b.T @ y          # Xᵀy
    theta = XtX_inv @ Xty
    return theta


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
