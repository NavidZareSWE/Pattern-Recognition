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
# SECTION 1: BAYESIAN CLASSIFICATION PLOTS
# =============================================================================

def plot_class_distribution(y, class_names=['Benign', 'Malignant']):
    """
    Task 1.1.2: Plot class distribution
    
    Args:
        y: array of class labels
        class_names: list of class names
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique, counts = np.unique(y, return_counts=True)
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution in Breast Cancer Dataset')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        percentage = count / len(y) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_1_2_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_histograms_by_class(X, y, feature_names, n_features=6):
    """
    Task 1.1.3: Plot class-wise histograms for features
    
    Args:
        X: feature matrix
        y: labels
        feature_names: list of feature names
        n_features: number of features to plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(n_features):
        ax = axes[idx]
        
        # Separate by class
        X_benign = X[y == 0, idx]
        X_malignant = X[y == 1, idx]
        
        ax.hist(X_benign, bins=25, alpha=0.6, label='Benign', color='#2ecc71', edgecolor='black')
        ax.hist(X_malignant, bins=25, alpha=0.6, label='Malignant', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel(feature_names[idx])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {feature_names[idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Class-wise Feature Histograms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_1_3_feature_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(corr_matrix, feature_names):
    """
    Task 1.1.3: Plot correlation matrix heatmap
    
    Args:
        corr_matrix: correlation matrix
        feature_names: list of feature names
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    
    ax.set_title('Feature Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_1_3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_eigenvalue_spectrum(eigenvalues, title='Eigenvalue Spectrum'):
    """
    Task 1.5.2: Plot eigenvalue spectrum
    
    Args:
        eigenvalues: array of eigenvalues
        title: plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.bar(range(len(eigenvalues)), eigenvalues, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title(f'{title} (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2 = axes[1]
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    ax2.bar(range(len(positive_eigenvalues)), positive_eigenvalues, 
            color='#e74c3c', edgecolor='black', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xlabel('Eigenvalue Index')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title(f'{title} (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_5_eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_multiclass(cm, class_labels, title='Confusion Matrix'):
    """
    Task 1.9.1: Plot confusion matrix
    
    Args:
        cm: confusion matrix
        class_labels: list of class labels
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels([f'Predicted {l}' for l in class_labels])
    ax.set_yticklabels([f'Actual {l}' for l in class_labels])
    
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{PLOTS_DIR}/1_9_{filename}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_variances(variances, feature_names, threshold_percentile=None):
    """
    Task 1.6.1: Plot feature variances
    
    Args:
        variances: array of feature variances
        feature_names: list of feature names
        threshold_percentile: percentile threshold for low variance
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sorted_indices = np.argsort(variances)
    sorted_variances = variances[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    colors = ['#e74c3c' if v < np.percentile(variances, 10) else '#3498db' 
              for v in sorted_variances]
    
    bars = ax.barh(range(len(sorted_variances)), sorted_variances, 
                   color=colors, edgecolor='black', alpha=0.7)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Variance')
    ax.set_title('Feature Variances (Sorted)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    if threshold_percentile is not None:
        threshold = np.percentile(variances, threshold_percentile)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'{threshold_percentile}th percentile threshold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_6_feature_variances.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_classifier_comparison(results_dict, metric='accuracy'):
    """
    Task 1.9: Compare classifier performance
    
    Args:
        results_dict: dict with classifier names as keys and metrics dict as values
        metric: metric to compare
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classifiers = list(results_dict.keys())
    train_scores = [results_dict[c]['train'][metric] for c in classifiers]
    test_scores = [results_dict[c]['test'][metric] for c in classifiers]
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Train', 
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test', 
                   color='#2ecc71', edgecolor='black')
    
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Classifier Comparison: {metric.capitalize()}')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_9_classifier_comparison_{metric}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# SECTION 2: DIMENSIONALITY REDUCTION PLOTS (PCA & Fisher LDA)
# =============================================================================

def plot_mean_face(mean_face, image_shape=(112, 92)):
    """
    Task 2.2.1.3: Plot mean face
    
    Args:
        mean_face: flattened mean face vector
        image_shape: tuple (height, width)
    """
    fig, ax = plt.subplots(figsize=(6, 7))
    
    mean_image = mean_face.reshape(image_shape)
    ax.imshow(mean_image, cmap='gray')
    ax.set_title('Mean Face', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_1_mean_face.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_original_vs_mean_subtracted(original_images, mean_subtracted_images, 
                                      labels, image_shape=(112, 92), n_samples=5):
    """
    Task 2.2.1.4: Compare original vs mean-subtracted images
    
    Args:
        original_images: original image vectors
        mean_subtracted_images: mean-subtracted image vectors
        labels: subject labels
        image_shape: tuple (height, width)
        n_samples: number of samples to show
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 7))
    
    # Select one random image per individual
    unique_labels = np.unique(labels)
    np.random.seed(42)
    selected_indices = []
    for label in unique_labels[:n_samples]:
        indices = np.where(labels == label)[0]
        selected_indices.append(np.random.choice(indices))
    
    for idx, sample_idx in enumerate(selected_indices):
        # Original
        axes[0, idx].imshow(original_images[sample_idx].reshape(image_shape), cmap='gray')
        axes[0, idx].set_title(f'Subject {labels[sample_idx]}')
        axes[0, idx].axis('off')
        
        # Mean-subtracted
        axes[1, idx].imshow(mean_subtracted_images[sample_idx].reshape(image_shape), cmap='gray')
        axes[1, idx].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Mean-Subtracted', fontsize=12)
    
    plt.suptitle('Original vs Mean-Subtracted Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_1_original_vs_mean_subtracted.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_eigenvalues(eigenvalues, cumulative_variance=None):
    """
    Task 2.2.2.3: Plot PCA eigenvalues and cumulative variance
    
    Args:
        eigenvalues: array of eigenvalues
        cumulative_variance: array of cumulative variance ratios
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Eigenvalues
    ax1 = axes[0]
    ax1.plot(eigenvalues[:100], 'b-', linewidth=2)
    ax1.set_xlabel('Principal Component Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('PCA Eigenvalues (Top 100)')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2 = axes[1]
    if cumulative_variance is not None:
        ax2.plot(cumulative_variance[:200], 'g-', linewidth=2)
        ax2.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
        ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% Variance')
        
        # Find number of components for 90% variance
        n_90 = np.argmax(cumulative_variance >= 0.9) + 1
        ax2.axvline(x=n_90, color='r', linestyle=':', alpha=0.7)
        ax2.annotate(f'n={n_90}', xy=(n_90, 0.9), xytext=(n_90+10, 0.85),
                     fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
        
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_2_pca_eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_eigenfaces(eigenfaces, n_components=50, image_shape=(112, 92)):
    """
    Task 2.2.2.4: Visualize top eigenfaces
    
    Args:
        eigenfaces: matrix of eigenvectors (principal components)
        n_components: number of eigenfaces to display
        image_shape: tuple (height, width)
    """
    n_rows = 5
    n_cols = 10
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    
    for idx in range(min(n_components, n_rows * n_cols)):
        row = idx // n_cols
        col = idx % n_cols
        
        eigenface = eigenfaces[idx].reshape(image_shape)
        axes[row, col].imshow(eigenface, cmap='gray')
        axes[row, col].set_title(f'PC {idx+1}', fontsize=8)
        axes[row, col].axis('off')
    
    plt.suptitle('Top 50 Eigenfaces (Principal Components)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_2_eigenfaces.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction_comparison(original, reconstructed, labels, 
                                    image_shape=(112, 92), n_samples=5):
    """
    Task 2.2.3.2: Compare original vs reconstructed images
    
    Args:
        original: original image vectors
        reconstructed: reconstructed image vectors
        labels: subject labels
        image_shape: tuple (height, width)
        n_samples: number of samples to show
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 7))
    
    np.random.seed(42)
    unique_labels = np.unique(labels)
    selected_indices = []
    for label in unique_labels[:n_samples]:
        indices = np.where(labels == label)[0]
        selected_indices.append(np.random.choice(indices))
    
    for idx, sample_idx in enumerate(selected_indices):
        # Original
        axes[0, idx].imshow(original[sample_idx].reshape(image_shape), cmap='gray')
        axes[0, idx].set_title(f'Subject {labels[sample_idx]}')
        axes[0, idx].axis('off')
        
        # Reconstructed
        axes[1, idx].imshow(reconstructed[sample_idx].reshape(image_shape), cmap='gray')
        axes[1, idx].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    
    plt.suptitle('Original vs PCA Reconstructed Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_3_reconstruction_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_accuracy_vs_components(n_components_list, accuracies_dict, title='Classification Accuracy vs Number of Components'):
    """
    Task 2.2.4.5: Plot accuracy vs number of principal components
    
    Args:
        n_components_list: list of number of components tested
        accuracies_dict: dict with classifier names as keys and accuracy lists as values
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (classifier, accuracies) in enumerate(accuracies_dict.items()):
        ax.plot(n_components_list, accuracies, 
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=2, markersize=6, label=classifier)
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = title.lower().replace(' ', '_').replace(':', '')
    plt.savefig(f'{PLOTS_DIR}/2_2_4_{filename}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_lda_eigenvalues(eigenvalues):
    """
    Task 2.3.3.2: Plot Fisher LDA eigenvalues
    
    Args:
        eigenvalues: array of eigenvalues
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(len(eigenvalues)), eigenvalues, 
           color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Discriminant Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Fisher LDA Eigenvalues (Sorted Descending)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_3_3_lda_eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_separability_metric(n_components_list, separability_scores):
    """
    Task 2.3.4.1: Plot separability metric vs number of components
    
    Args:
        n_components_list: list of number of components
        separability_scores: list of separability scores
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_components_list, separability_scores, 
            'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Number of LDA Components')
    ax.set_ylabel('Separability Score')
    ax.set_title('Class Separability vs Number of LDA Components')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_3_4_separability.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_multiclass_confusion_matrix(cm, n_classes=40, title='Confusion Matrix'):
    """
    Task 2.2.4.8 / 2.3.6: Plot confusion matrix for face recognition
    
    Args:
        cm: confusion matrix
        n_classes: number of classes
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.4f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{PLOTS_DIR}/2_{filename}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_vs_pca_lda_comparison(pca_accuracy, pca_lda_accuracy, n_components_list):
    """
    Task 2.2.4.7: Compare PCA-only vs PCA+LDA performance
    
    Args:
        pca_accuracy: list of PCA-only accuracies
        pca_lda_accuracy: list of PCA+LDA accuracies
        n_components_list: list of number of components
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_components_list, pca_accuracy, 'b-o', linewidth=2, 
            markersize=6, label='PCA Only')
    ax.plot(n_components_list, pca_lda_accuracy, 'r-s', linewidth=2, 
            markersize=6, label='PCA + LDA')
    
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Recognition Accuracy')
    ax.set_title('PCA vs PCA+LDA Face Recognition Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_2_4_pca_vs_pca_lda.png', dpi=150, bbox_inches='tight')
    plt.show()
