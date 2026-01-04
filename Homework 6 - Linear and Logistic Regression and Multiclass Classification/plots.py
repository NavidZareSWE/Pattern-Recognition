import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


# =============================================================================
# SECTION 1: LINEAR REGRESSION PLOTS
# =============================================================================

def plot_scatter_features_vs_mpg(data):
    """
    Task 1.2.6: Scatter plots of features vs mpg

    Args:
        data: pandas DataFrame with columns ['weight', 'horsepower', 'displacement', 'mpg']
    """
    features = ['weight', 'horsepower', 'displacement']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.scatter(data[feature], data['mpg'], alpha=0.6,
                   edgecolors='k', linewidth=0.5)

        correlation = np.corrcoef(data[feature], data['mpg'])[0, 1]

        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel('MPG')
        ax.set_title(
            f'{feature.capitalize()} vs MPG\n(Correlation: {correlation:.4f})')

        z = np.polyfit(data[feature], data['mpg'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[feature].min(), data[feature].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend line')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_2_6_scatter_plots.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_bgd_cost(cost_history, learning_rate):
    """
    Task 1.3.A.2: Plot cost function vs iterations for BGD

    Args:
        cost_history: list of cost values at each iteration
        learning_rate: learning rate used
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (MSE)')
    ax.set_title(f'BGD Cost Function (Learning Rate = {learning_rate})')
    ax.grid(True, alpha=0.3)

    final_cost = cost_history[-1]
    ax.annotate(f'Final Cost: {final_cost:.4f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_A_bgd_cost.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_bgd_cost_multiple_lr(X_train, y_train, bgd_func):
    """
    Task 1.3.A.3: Experiment with different learning rates for BGD

    Args:
        X_train: normalized training features
        y_train: training targets
        bgd_func: your batch_gradient_descent function
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        _, _, cost_history = bgd_func(
            X_train, y_train, learning_rate=lr, n_iterations=1000)

        ax = axes[idx]
        ax.plot(cost_history, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost (MSE)')
        ax.set_title(f'BGD Cost (lr = {lr})')
        ax.grid(True, alpha=0.3)

        ax.annotate(f'Final: {cost_history[-1]:.4f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_A_bgd_multiple_lr.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_regression_line(X_train, X_test, y_train, y_test, theta_0, theta_1, method_name='BGD'):
    """
    Task 1.3.A.5 / 1.3.D.2: Plot regression line with training and testing samples

    Args:
        X_train, X_test: normalized features
        y_train, y_test: targets
        theta_0, theta_1: learned parameters
        method_name: name of the method for title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(X_train, y_train, alpha=0.6, label='Training Data',
               edgecolors='k', linewidth=0.5, s=50)
    ax.scatter(X_test, y_test, alpha=0.6, label='Test Data',
               marker='s', edgecolors='k', linewidth=0.5, s=50)

    x_line = np.linspace(min(X_train.min(), X_test.min()) - 0.5,
                         max(X_train.max(), X_test.max()) + 0.5, 100)
    y_line = theta_0 + theta_1 * x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Regression Line')

    ax.set_xlabel('Weight (Normalized)')
    ax.set_ylabel('MPG')
    ax.set_title(
        f'{method_name} Regression\nθ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_{method_name.lower()}_regression_line.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_sgd_cost(cost_history, learning_rate):
    """
    Task 1.3.B.3: Plot cost function over epochs for SGD

    Args:
        cost_history: list of cost values at each epoch
        learning_rate: learning rate used
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost (MSE)')
    ax.set_title(f'SGD Cost Function (Learning Rate = {learning_rate})')
    ax.grid(True, alpha=0.3)

    final_cost = cost_history[-1]
    ax.annotate(f'Final Cost: {final_cost:.4f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_B_sgd_cost.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_sgd_cost_multiple_lr(X_train, y_train, sgd_func):
    """
    Task 1.3.B.4: Experiment with different learning rates for SGD

    Args:
        X_train: normalized training features
        y_train: training targets
        sgd_func: your stochastic_gradient_descent function
    """
    learning_rates = [0.001, 0.01, 0.05, 0.1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        np.random.seed(42)
        _, _, cost_history = sgd_func(
            X_train, y_train, learning_rate=lr, n_epochs=100)

        ax = axes[idx]
        ax.plot(cost_history, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost (MSE)')
        ax.set_title(f'SGD Cost (lr = {lr})')
        ax.grid(True, alpha=0.3)

        ax.annotate(f'Final: {cost_history[-1]:.4f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_B_sgd_multiple_lr.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_bgd_vs_sgd_comparison(X_train, X_test, y_train, y_test,
                               theta_0_bgd, theta_1_bgd, cost_bgd,
                               theta_0_sgd, theta_1_sgd, cost_sgd):
    """
    Task 1.3.C: Comparison of BGD and SGD

    Args:
        X_train, X_test, y_train, y_test: data
        theta_0_bgd, theta_1_bgd, cost_bgd: BGD results
        theta_0_sgd, theta_1_sgd, cost_sgd: SGD results
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cost comparison
    ax1 = axes[0]
    ax1.plot(cost_bgd, linewidth=2, label='BGD')
    ax1.plot(cost_sgd, linewidth=2, label='SGD')
    ax1.set_xlabel('Iteration / Epoch')
    ax1.set_ylabel('Cost (MSE)')
    ax1.set_title('Cost Function Comparison: BGD vs SGD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Regression lines comparison
    ax2 = axes[1]
    ax2.scatter(X_train, y_train, alpha=0.4, label='Training Data', s=30)
    ax2.scatter(X_test, y_test, alpha=0.4, label='Test Data', marker='s', s=30)

    x_line = np.linspace(min(X_train.min(), X_test.min()) - 0.5,
                         max(X_train.max(), X_test.max()) + 0.5, 100)
    y_line_bgd = theta_0_bgd + theta_1_bgd * x_line
    y_line_sgd = theta_0_sgd + theta_1_sgd * x_line

    ax2.plot(x_line, y_line_bgd, 'r-', linewidth=2.5, label='BGD')
    ax2.plot(x_line, y_line_sgd, 'g--', linewidth=2.5, label='SGD')

    ax2.set_xlabel('Weight (Normalized)')
    ax2.set_ylabel('MPG')
    ax2.set_title('Regression Lines: BGD vs SGD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_C_bgd_vs_sgd.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_methods_comparison(X_train, X_test, y_train, y_test,
                                theta_0_bgd, theta_1_bgd,
                                theta_0_sgd, theta_1_sgd,
                                theta_0_cf, theta_1_cf):
    """
    Task 1.3.D.3: Compare closed-form with BGD and SGD

    Args:
        X_train, X_test, y_train, y_test: data
        theta_*_bgd, theta_*_sgd, theta_*_cf: parameters from each method
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(X_train, y_train, alpha=0.5, label='Training Data',
               edgecolors='k', linewidth=0.3, s=50)
    ax.scatter(X_test, y_test, alpha=0.5, label='Test Data',
               marker='s', edgecolors='k', linewidth=0.3, s=50)

    x_line = np.linspace(min(X_train.min(), X_test.min()) - 0.5,
                         max(X_train.max(), X_test.max()) + 0.5, 100)

    ax.plot(x_line, theta_0_cf + theta_1_cf * x_line, 'r-', linewidth=3,
            label=f'Closed-Form (θ₀={theta_0_cf:.4f}, θ₁={theta_1_cf:.4f})')
    ax.plot(x_line, theta_0_bgd + theta_1_bgd * x_line, 'g--', linewidth=2,
            label=f'BGD (θ₀={theta_0_bgd:.4f}, θ₁={theta_1_bgd:.4f})')
    ax.plot(x_line, theta_0_sgd + theta_1_sgd * x_line, 'b:', linewidth=2,
            label=f'SGD (θ₀={theta_0_sgd:.4f}, θ₁={theta_1_sgd:.4f})')

    ax.set_xlabel('Weight (Normalized)')
    ax.set_ylabel('MPG')
    ax.set_title('Comparison: Closed-Form vs Gradient Descent Methods')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_3_D_all_methods_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# SECTION 2: LOGISTIC REGRESSION PLOTS
# =============================================================================

def plot_feature_histograms(X_standardized, numerical_cols):
    """
    Task 2.2.8: Plot histograms of numerical features after standardization

    Args:
        X_standardized: pandas DataFrame with standardized features
        numerical_cols: list of numerical column names to plot
    """
    n_cols = min(len(numerical_cols), 4)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        ax.hist(X_standardized[col], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col.capitalize()}\n(Standardized)')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Mean')
        ax.legend()

    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_8_feature_histograms.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_logistic_cost(cost_history, learning_rate):
    """
    Task 2.3.2: Plot cost function for logistic regression

    Args:
        cost_history: list of cost values
        learning_rate: learning rate used
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Cross-Entropy)')
    ax.set_title(f'Logistic Regression Cost (Learning Rate = {learning_rate})')
    ax.grid(True, alpha=0.3)

    final_cost = cost_history[-1]
    ax.annotate(f'Final Cost: {final_cost:.4f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_3_2_logistic_cost.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_logistic_cost_multiple_lr(X_train, y_train, fit_func):
    """
    Task 2.3.3: Experiment with different learning rates

    Args:
        X_train: training features (numpy array)
        y_train: training labels (numpy array)
        fit_func: your fit function from logistic_regression.py
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        _, _, cost_history = fit_func(
            X_train, y_train, learning_rate=lr, n_iterations=1000)

        ax = axes[idx]
        ax.plot(cost_history, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost (Cross-Entropy)')
        ax.set_title(f'Logistic Regression Cost (lr = {lr})')
        ax.grid(True, alpha=0.3)

        ax.annotate(f'Final: {cost_history[-1]:.4f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_3_3_logistic_multiple_lr.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Task 2.3.6: Confusion Matrix visualization

    Args:
        y_true: actual labels (numpy array)
        y_pred: predicted labels (numpy array)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    cm = np.array([[TN, FP], [FN, TP]])

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    fontsize=20, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

    ax.set_title('Confusion Matrix\n(Logistic Regression - Test Set)')

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_3_6_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# SECTION 3: MULTI-CLASS CLASSIFICATION PLOTS
# =============================================================================

def plot_ova_cost_functions(cost_histories, classes):
    """
    Task 3.3.3: Plot cross-entropy loss for OvA classifiers

    Args:
        cost_histories: dict {class_label: cost_history_list}
        classes: array of class labels
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for c in classes:
        ax.plot(cost_histories[c], label=f'Class {c} vs All', linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Cross-Entropy)')
    ax.set_title('One-vs-All (OvA) Cost Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_3_ova_cost_functions.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_ovo_cost_functions(cost_histories):
    """
    Task 3.3.3: Plot cross-entropy loss for OvO classifiers

    Args:
        cost_histories: dict {(class_i, class_j): cost_history_list}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for pair, cost_history in cost_histories.items():
        ax.plot(cost_history, label=f'{pair[0]} vs {pair[1]}', linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Cross-Entropy)')
    ax.set_title('One-vs-One (OvO) Cost Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_3_ovo_cost_functions.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_softmax_cost(cost_history):
    """
    Task 3.3.4: Plot softmax regression cost function

    Args:
        cost_history: list of cost values
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, linewidth=2, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Multi-class Cross-Entropy)')
    ax.set_title('Softmax Regression Cost Function')
    ax.grid(True, alpha=0.3)

    final_cost = cost_history[-1]
    ax.annotate(f'Final Cost: {final_cost:.4f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_4_softmax_cost.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_multiclass_accuracy_comparison(ova_acc, ovo_acc, softmax_acc):
    """
    Task 3.3.5: Compare accuracy of OvA, OvO, and Softmax

    Args:
        ova_acc: One-vs-All test accuracy
        ovo_acc: One-vs-One test accuracy
        softmax_acc: Softmax test accuracy
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['One-vs-All', 'One-vs-One', 'Softmax']
    accuracies = [ova_acc, ovo_acc, softmax_acc]
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    bars = ax.bar(methods, accuracies, color=colors,
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_5_accuracy_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_multiclass_cost_comparison(ova_cost_histories, ovo_cost_histories, softmax_cost_history, classes):
    """
    Task 3.3.5: Compare convergence of all three methods

    Args:
        ova_cost_histories: dict of OvA cost histories
        ovo_cost_histories: dict of OvO cost histories  
        softmax_cost_history: list of softmax costs
        classes: array of class labels
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ova_avg_cost = np.mean([ova_cost_histories[c] for c in classes], axis=0)
    ax.plot(ova_avg_cost, label='OvA (Average)', linewidth=2)

    ovo_avg_cost = np.mean(list(ovo_cost_histories.values()), axis=0)
    ax.plot(ovo_avg_cost, label='OvO (Average)', linewidth=2)

    ax.plot(softmax_cost_history, label='Softmax', linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_5_cost_convergence.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_outlier_impact(acc_with_outliers, acc_without_outliers):
    """
    Task 3.3.6: Impact of outlier removal on classification performance

    Args:
        acc_with_outliers: tuple (ova_acc, ovo_acc, softmax_acc) with outliers
        acc_without_outliers: tuple (ova_acc, ovo_acc, softmax_acc) without outliers
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['One-vs-All', 'One-vs-One', 'Softmax']
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, acc_with_outliers, width, label='With Outliers',
                   color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, acc_without_outliers, width, label='Without Outliers',
                   color='#27ae60', edgecolor='black')

    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Outlier Removal on Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_3_6_outlier_impact.png',
                dpi=150, bbox_inches='tight')
    plt.show()
