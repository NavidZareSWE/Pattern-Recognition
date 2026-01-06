"""
Bonus Plotting Functions for Ridge Regression with SGD
======================================================
This module contains all visualization functions for the bonus section
of the homework assignment on regularized linear regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure bonus_Plots directory exists
PLOT_DIR = 'bonus_Plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_correlation_matrix(corr_matrix, feature_names, save_path=None):
    """
    Plot the correlation matrix as a heatmap.
    
    Parameters:
    -----------
    corr_matrix : np.ndarray
        Correlation matrix of features
    feature_names : list
        Names of features
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.set_yticklabels(feature_names, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add correlation values as text
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color=text_color, fontsize=10)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_vif_analysis(feature_names, vif_values, save_path=None):
    """
    Plot VIF values as a horizontal bar chart.
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    vif_values : list
        VIF values for each feature
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by VIF value
    sorted_indices = np.argsort(vif_values)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_vif = [vif_values[i] for i in sorted_indices]
    
    # Create colors based on VIF thresholds
    colors = []
    for vif in sorted_vif:
        if vif > 10:
            colors.append('#e74c3c')  # Red - severe multicollinearity
        elif vif > 5:
            colors.append('#f39c12')  # Orange - moderate multicollinearity
        else:
            colors.append('#27ae60')  # Green - acceptable
    
    bars = ax.barh(sorted_features, sorted_vif, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (Moderate)')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (High)')
    
    # Add value labels on bars
    for i, (bar, vif) in enumerate(zip(bars, sorted_vif)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{vif:.2f}', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Multicollinearity Analysis: VIF Values', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(sorted_vif) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_ridge_coefficients_comparison(lambda_values, coefficients_dict, feature_names, save_path=None):
    """
    Plot model coefficients for different lambda values.
    
    Parameters:
    -----------
    lambda_values : list
        List of regularization parameter values
    coefficients_dict : dict
        Dictionary mapping lambda to (theta_0, theta_1, ..., theta_n)
    feature_names : list
        Names of features (including bias)
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_lambdas = len(lambda_values)
    n_features = len(feature_names)
    
    # Create bar positions
    x = np.arange(n_features)
    width = 0.8 / n_lambdas
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_lambdas))
    
    for i, (lam, color) in enumerate(zip(lambda_values, colors)):
        coeffs = coefficients_dict[lam]
        offset = (i - n_lambdas/2 + 0.5) * width
        bars = ax.bar(x + offset, coeffs, width, label=f'λ = {lam}', 
                     color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Coefficients', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Ridge Regression Coefficients for Different λ Values', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(title='Regularization')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_ridge_sgd_convergence(cost_histories, lambda_values, save_path=None):
    """
    Plot training loss (MSE) vs epochs for different lambda values.
    
    Parameters:
    -----------
    cost_histories : dict
        Dictionary mapping lambda to list of costs per epoch
    lambda_values : list
        List of regularization parameter values
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(lambda_values)))
    
    for lam, color in zip(lambda_values, colors):
        costs = cost_histories[lam]
        epochs = range(1, len(costs) + 1)
        ax.plot(epochs, costs, label=f'λ = {lam}', color=color, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('Ridge Regression SGD Convergence\nTraining Loss vs Epochs', fontsize=14, fontweight='bold')
    ax.legend(title='Regularization Parameter')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, None)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_ridge_sgd_convergence_detailed(cost_histories, lambda_values, save_path=None):
    """
    Plot detailed convergence for each lambda value in subplots.
    
    Parameters:
    -----------
    cost_histories : dict
        Dictionary mapping lambda to list of costs per epoch
    lambda_values : list
        List of regularization parameter values
    save_path : str, optional
        Path to save the figure
    """
    n_lambdas = len(lambda_values)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_lambdas))
    
    for i, (lam, color) in enumerate(zip(lambda_values, colors)):
        ax = axes[i]
        costs = cost_histories[lam]
        epochs = range(1, len(costs) + 1)
        
        ax.plot(epochs, costs, color=color, linewidth=2)
        ax.fill_between(epochs, costs, alpha=0.3, color=color)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('MSE', fontsize=10)
        ax.set_title(f'λ = {lam}\nFinal MSE: {costs[-1]:.6f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add convergence info
        min_cost = min(costs)
        min_epoch = costs.index(min_cost) + 1
        ax.axhline(y=min_cost, color='red', linestyle='--', alpha=0.7)
        ax.annotate(f'Min: {min_cost:.4f} @ epoch {min_epoch}', 
                   xy=(min_epoch, min_cost), xytext=(len(costs)*0.6, min_cost*1.1),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    fig.suptitle('Ridge Regression SGD Convergence Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_mse_comparison(lambda_values, train_mse, test_mse, save_path=None):
    """
    Plot training and testing MSE for different lambda values.
    
    Parameters:
    -----------
    lambda_values : list
        List of regularization parameter values
    train_mse : list
        Training MSE for each lambda
    test_mse : list
        Testing MSE for each lambda
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(lambda_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_mse, width, label='Training MSE', 
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_mse, width, label='Testing MSE', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_xlabel('Regularization Parameter (λ)', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Training vs Testing MSE for Different λ Values', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(lam) for lam in lambda_values])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_bias_variance_tradeoff(lambda_values, train_mse, test_mse, save_path=None):
    """
    Plot bias-variance tradeoff visualization.
    
    Parameters:
    -----------
    lambda_values : list
        List of regularization parameter values
    train_mse : list
        Training MSE for each lambda
    test_mse : list
        Testing MSE for each lambda
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use log scale for lambda on x-axis for better visualization
    lambda_plot = [max(lam, 0.001) for lam in lambda_values]  # Replace 0 with small value for log scale
    
    ax.plot(lambda_plot, train_mse, 'o-', label='Training MSE (Bias²)', 
            color='#3498db', linewidth=2, markersize=10)
    ax.plot(lambda_plot, test_mse, 's-', label='Testing MSE (Bias² + Variance)', 
            color='#e74c3c', linewidth=2, markersize=10)
    
    # Mark optimal lambda (lowest test MSE)
    best_idx = np.argmin(test_mse)
    ax.scatter([lambda_plot[best_idx]], [test_mse[best_idx]], 
               s=200, c='green', marker='*', zorder=5, label=f'Optimal λ = {lambda_values[best_idx]}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Parameter (λ) - Log Scale', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff in Ridge Regression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation for underfitting and overfitting regions
    ax.annotate('Higher Bias\nLower Variance', xy=(0.8, 0.7), xycoords='axes fraction',
               fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.annotate('Lower Bias\nHigher Variance', xy=(0.2, 0.3), xycoords='axes fraction',
               fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_regression_lines_multiple_lambda(X_train, X_test, y_train, y_test, 
                                          coefficients_dict, lambda_values, save_path=None):
    """
    Plot regression lines for different lambda values (for simple linear regression).
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and testing features (normalized)
    y_train, y_test : np.ndarray
        Training and testing targets
    coefficients_dict : dict
        Dictionary mapping lambda to (theta_0, theta_1)
    lambda_values : list
        List of regularization parameter values
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points
    ax.scatter(X_train, y_train, c='blue', alpha=0.5, label='Training Data', s=30)
    ax.scatter(X_test, y_test, c='red', alpha=0.5, label='Testing Data', s=30)
    
    # Plot regression lines
    x_line = np.linspace(X_train.min() - 0.5, X_train.max() + 0.5, 100)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(lambda_values)))
    
    for lam, color in zip(lambda_values, colors):
        theta_0, theta_1 = coefficients_dict[lam][:2]
        y_line = theta_0 + theta_1 * x_line
        ax.plot(x_line, y_line, color=color, linewidth=2, label=f'λ = {lam}')
    
    ax.set_xlabel('Normalized Weight', fontsize=12)
    ax.set_ylabel('MPG', fontsize=12)
    ax.set_title('Ridge Regression Lines for Different λ Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_coefficient_path(lambda_range, coefficient_paths, feature_names, save_path=None):
    """
    Plot regularization path showing how coefficients change with lambda.
    
    Parameters:
    -----------
    lambda_range : list
        Range of lambda values
    coefficient_paths : dict
        Dictionary with feature names as keys and lists of coefficients as values
    feature_names : list
        Names of features (excluding bias)
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
    
    for feature, color in zip(feature_names, colors):
        if feature in coefficient_paths:
            ax.plot(lambda_range, coefficient_paths[feature], 
                   label=feature, color=color, linewidth=2, marker='o', markersize=4)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Regularization Parameter (λ)', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Regularization Path: Coefficient Shrinkage', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig


def plot_summary_table_as_figure(results_df, save_path=None):
    """
    Create a summary table visualization.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing results summary
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#3498db']*len(results_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(results_df.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#2c3e50')
    
    ax.set_title('Ridge Regression Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return fig
