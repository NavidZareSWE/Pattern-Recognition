import pandas as pd
import numpy as np
from models import QDA, LDA, GaussianNaiveBayes
from plots_hw7 import (plot_feature_histograms_by_class,
                       plot_correlation_heatmap, plot_eigenvalue_spectrum,
                       plot_confusion_matrix_multiclass, plot_feature_variances,
                       plot_classifier_comparison)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):

    np.random.seed(random_state)
    n_samples = len(X)

    if stratify is None:
        n_test = int(n_samples * test_size)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        train_indices = []
        test_indices = []
        unique_classes = np.unique(stratify)

        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            n_cls_test = int(len(cls_indices) * test_size)
            if n_cls_test == 0 and len(cls_indices) > 1:
                n_cls_test = 1

            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    return (X[train_indices], X[test_indices],
            y[train_indices], y[test_indices])


def calculate_metrics(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }


def load_and_explore_data():
    # 1.1.1 Load the dataset
    # from wdbc.names
    column_names = ['id', 'diagnosis'] + [
        f'{stat}_{feat}' for stat in ['mean', 'se', 'worst']
        for feat in ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                     'compactness', 'concavity', 'concave_points', 'symmetry',
                     'fractal_dimension']
    ]

    data = pd.read_csv('Breast Cancer Wisconsin/wdbc.data',
                       header=None, names=column_names)

    print("\n" + "=" * 60)
    print("TASK 1.1: UNDERSTANDING THE DATA")
    print("=" * 60)

    # 1.1.1 Report number of samples, features, and classes
    print("\n--- 1.1.1 Dataset Overview ---")
    print(f"Number of samples: {data.shape[0]}")
    print(f"Number of features: {data.shape[1] - 2}")  # Exclude ID & diagnosis
    print(f"Number of classes: {data['diagnosis'].nunique()}")
    print(f"Classes: {data['diagnosis'].unique()}")

    # 1.1.2 Compute class proportions and discuss imbalance
    # Encode labels: M (Malignant) = 1, B (Benign) = 0
    y = (data['diagnosis'] == 'M').astype(int).values
    X = data.drop(['id', 'diagnosis'], axis=1).values
    feature_names = column_names[2:]
    print("\n--- 1.1.2 Class Distribution ---")
    benign_count = np.sum(y == 0)
    malignant_count = np.sum(y == 1)
    total = len(y)

    print(f"Benign (B=0): {benign_count} ({benign_count/total*100:.2f}%)")
    print(
        f"Malignant (M=1): {malignant_count} ({malignant_count/total*100:.2f}%)")

    imbalance_ratio = max(benign_count, malignant_count) / \
        min(benign_count, malignant_count)
    print(
        f"Imbalanced? {'Yes' if imbalance_ratio > 1.5 else 'No'}\n"
        f"Imbalance Ratio: {imbalance_ratio:.2f}:1"
    )

    # # 1.1.3 Exploratory visual analysis
    print("\n--- 1.1.3 Exploratory Visual Analysis ---")
    print("Plotting class-wise histograms for selected features...")
    plot_feature_histograms_by_class(X, y, feature_names, n_features=6)

    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)
    plot_correlation_heatmap(corr_matrix, feature_names)

    # # 1.1.4 Identify highly correlated feature pairs
    print("\n--- 1.1.4 Highly Correlated Feature Pairs ---")
    n_features = len(feature_names)
    correlations = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            correlations.append((feature_names[i], feature_names[j],
                                abs(corr_matrix[i, j])))

    correlations.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Rank':<6} {'Feature 1':<25} {'Feature 2':<25} {'|Correlation|':>15}")
    print("-" * 75)
    for rank, (f1, f2, corr) in enumerate(correlations[:3], 1):
        print(f"{rank:<6} {f1:<25} {f2:<25} {corr:>15.4f}")

    return X, y, feature_names


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("TASK 1.4: MODEL IMPLEMENTATION & CLASSIFICATION")
    print("=" * 60)

    results = {}

    # --- Gaussian Naive Bayes ---
    print("\n--- Training Gaussian Naive Bayes ---")
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    gnb_params = gnb.get_params()
    print(f"Class priors: {gnb_params['class_priors']}")
    print(f"Means shape: {gnb_params['means'].shape}")
    print(f"Variances shape: {gnb_params['variances'].shape}")

    gnb_train_pred = gnb.predict(X_train)
    gnb_test_pred = gnb.predict(X_test)

    results['GNB'] = {
        'model': gnb,
        'train': calculate_metrics(y_train, gnb_train_pred),
        'test': calculate_metrics(y_test, gnb_test_pred)
    }

    # --- LDA ---
    print("\n--- Training Linear Discriminant Analysis (LDA) ---")
    lda = LDA()
    lda.fit(X_train, y_train)

    lda_params = lda.get_params()
    print(f"Class priors: {lda_params['class_priors']}")
    print(f"Means shape: {lda_params['means'].shape}")
    print(f"Shared covariance shape: {lda_params['shared_covariance'].shape}")

    lda_train_pred = lda.predict(X_train)
    lda_test_pred = lda.predict(X_test)

    results['LDA'] = {
        'model': lda,
        'train': calculate_metrics(y_train, lda_train_pred),
        'test': calculate_metrics(y_test, lda_test_pred)
    }

    # --- QDA ---
    print("\n--- Training Quadratic Discriminant Analysis (QDA) ---")
    qda = QDA()
    qda.fit(X_train, y_train)

    qda_params = qda.get_params()
    print(f"Class priors: {qda_params['class_priors']}")
    print(f"Means shape: {qda_params['means'].shape}")
    print(f"Number of covariance matrices: {len(qda_params['covariances'])}")

    qda_train_pred = qda.predict(X_train)
    qda_test_pred = qda.predict(X_test)

    results['QDA'] = {
        'model': qda,
        'train': calculate_metrics(y_train, qda_train_pred),
        'test': calculate_metrics(y_test, qda_test_pred)
    }

    # Print results summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS (Task 1.4.5-6)")
    print("=" * 60)

    print(f"\n{'Classifier':<15} {'Set':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    for classifier_name in ['GNB', 'LDA', 'QDA']:
        for set_name in ['train', 'test']:
            metrics = results[classifier_name][set_name]
            print(f"{classifier_name:<15} {set_name:<10} {metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f}")

    return results


def analyze_covariance_matrix(X_train):

    print("\n" + "=" * 60)
    print("TASK 1.5: COVARIANCE MATRIX ANALYSIS")
    print("=" * 60)

    # 1.5.1 Compute empirical covariance matrix
    # Transposing X_train with .T rearranges the data so that features are in columns, enabling accurate covariance computation between them.
    cov_matrix = np.cov(X_train.T)
    n_features = cov_matrix.shape[0]

    print(f"\n--- 1.5.1 Covariance Matrix Properties ---")
    print(f"Shape: {cov_matrix.shape}")

    # Compute rank
    rank = np.linalg.matrix_rank(cov_matrix)
    print(f"Rank: {rank}")
    print(f"Full rank would be: {n_features}")

    if rank < n_features:
        print("Status: SINGULAR (rank-deficient)")
    else:
        print("Status: Full rank")

    # 1.5.2 Compute eigenvalues
    print(f"\n--- 1.5.2 Eigenvalue Analysis ---")
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    # Breakdown of [::-1]: First ':' indicates the start of the slice (beginning of the sequence);
    # second ':' indicates the end of the slice (end of the sequence); '-1' specifies the step,
    # meaning to take elements in reverse order.
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    print(f"Largest eigenvalue: {eigenvalues[0]:.6e}")
    print(f"Smallest eigenvalue: {eigenvalues[-1]:.6e}")
    print(f"Eigenvalue range: {eigenvalues[0] / eigenvalues[-1]:.2e}")

    # 1.5.3 Condition number
    print(f"\n--- 1.5.3 Condition Number ---")
    cond_number = np.linalg.cond(cov_matrix)
    print(f"Condition number: {cond_number:.2e}")

    plot_eigenvalue_spectrum(
        eigenvalues, title='Covariance Matrix Eigenvalue Spectrum')

    return {
        'covariance': cov_matrix,
        'eigenvalues': eigenvalues,
        'rank': rank,
        'condition_number': cond_number
    }


def analyze_feature_variances(X_train, feature_names):
    print("\n" + "=" * 60)
    print("TASK 1.6: VARIANCE ANALYSIS")
    print("=" * 60)

    # 1.6.1 Compute variance of each feature
    variances = np.var(X_train, axis=0)

    print(f"\n--- 1.6.1 Feature Variances ---")
    print(f"{'Rank':<6} {'Feature':<30} {'Variance':>15}")
    print("-" * 55)

    sorted_indices = np.argsort(variances)
    for rank, idx in enumerate(sorted_indices[:10], 1):
        print(f"{rank:<6} {feature_names[idx]:<30} {variances[idx]:>15.6e}")

    # Identify low-variance features (bottom 10th percentile)
    variance_threshold = np.percentile(variances, 10)
    low_var_features = np.where(variances < variance_threshold)[0]

    print(f"\n10th percentile variance threshold: {variance_threshold:.6e}")
    print(f"Number of low-variance features: {len(low_var_features)}")
    print("Low-variance features:")
    for idx in low_var_features:
        print(f"  - {feature_names[idx]}: {variances[idx]:.6e}")

    plot_feature_variances(variances, feature_names, threshold_percentile=10)

    return variances, low_var_features


def variance_based_feature_elimination(X_train, X_test, feature_names, percentile=10):
    print("\n" + "=" * 60)
    print("TASK 1.7: VARIANCE-BASED FEATURE ELIMINATION")
    print("=" * 60)

    variances = np.var(X_train, axis=0)
    # Finds the variance value at the specified percentile (default 10th). Features with variance below this get removed.
    threshold = np.percentile(variances, percentile)

    # 1.7.1-2 Identify and remove low-variance features
    keep_arr = variances >= threshold
    removed_features = [feature_names[i]
                        for i in range(len(feature_names)) if not keep_arr[i]]
    kept_features = [feature_names[i]
                     for i in range(len(feature_names)) if keep_arr[i]]

    print(f"\n--- 1.7.1-2 Feature Elimination ---")
    print(f"Variance threshold ({percentile}th percentile): {threshold:.6e}")
    print(f"Features removed: {len(removed_features)}")
    print(f"Features kept: {len(kept_features)}")
    print("\nRemoved features:")
    for f in removed_features:
        print(f"  - {f}")

    # 1.7.3-4 Analyze covariance before and after
    print(f"\n--- 1.7.3-4 Covariance Analysis Comparison ---")

    # Before
    cov_before = np.cov(X_train.T)
    rank_before = np.linalg.matrix_rank(cov_before)
    cond_before = np.linalg.cond(cov_before)
    eig_before = np.linalg.eigvalsh(cov_before)

    X_train_reduced = X_train[:, keep_arr]
    X_test_reduced = X_test[:, keep_arr]

    # After
    cov_after = np.cov(X_train_reduced.T)
    rank_after = np.linalg.matrix_rank(cov_after)
    cond_after = np.linalg.cond(cov_after)
    eig_after = np.linalg.eigvalsh(cov_after)

    print(f"{'Metric':<25} {'Before':>15} {'After':>15}")
    print("-" * 55)
    print(
        f"{'Dimensions':<25} {X_train.shape[1]:>15} {X_train_reduced.shape[1]:>15}")
    print(f"{'Rank':<25} {rank_before:>15} {rank_after:>15}")
    print(f"{'Condition Number':<25} {cond_before:>15.2e} {cond_after:>15.2e}")
    print(f"{'Min Eigenvalue':<25} {eig_before.min():>15.2e} {eig_after.min():>15.2e}")
    print(f"{'Max Eigenvalue':<25} {eig_before.max():>15.2e} {eig_after.max():>15.2e}")

    return X_train_reduced, X_test_reduced, kept_features,


def retrain_with_reduced_features(X_train_reduced, X_test_reduced, y_train, y_test, kept_features):
    print("\n" + "=" * 60)
    print("TASK 1.8: RE-ESTIMATION WITH REDUCED FEATURES")
    print("=" * 60)

    results = {}

    # --- LDA with reduced features ---
    lda_reduced = LDA()
    lda_reduced.fit(X_train_reduced, y_train)

    lda_train_pred = lda_reduced.predict(X_train_reduced)
    lda_test_pred = lda_reduced.predict(X_test_reduced)

    results['LDA_reduced'] = {
        'model': lda_reduced,
        'train': calculate_metrics(y_train, lda_train_pred),
        'test': calculate_metrics(y_test, lda_test_pred)
    }

    # --- QDA with reduced features ---
    qda_reduced = QDA()
    qda_reduced.fit(X_train_reduced, y_train)

    qda_train_pred = qda_reduced.predict(X_train_reduced)
    qda_test_pred = qda_reduced.predict(X_test_reduced)

    results['QDA_reduced'] = {
        'model': qda_reduced,
        'train': calculate_metrics(y_train, qda_train_pred),
        'test': calculate_metrics(y_test, qda_test_pred)
    }

    # Print results
    print("\n--- Classification Results (Reduced Features) ---")
    print(f"Number of features: {X_train_reduced.shape[1]}")
    print(f"\n{'Classifier':<15} {'Set':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)

    for classifier_name in ['LDA_reduced', 'QDA_reduced']:
        for set_name in ['train', 'test']:
            metrics = results[classifier_name][set_name]
            print(f"{classifier_name:<15} {set_name:<10} {metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f}")

    return results


# =============================================================================
# SECTION 1.9: PERFORMANCE EVALUATION AND ERROR ANALYSIS
# =============================================================================

def performance_evaluation(results_original, results_reduced):
    print("\n" + "=" * 60)
    print("TASK 1.9: PERFORMANCE EVALUATION & ERROR ANALYSIS")
    print("=" * 60)

    # Combine results for comparison
    all_results = {
        'GNB': results_original['GNB'],
        'LDA': results_original['LDA'],
        'QDA': results_original['QDA'],
        'LDA (reduced)': results_reduced['LDA_reduced'],
        'QDA (reduced)': results_reduced['QDA_reduced']
    }

    # 1.9.1 Report metrics
    print("\n--- 1.9.1 Performance Metrics ---")
    print(
        f"\n{'Classifier':<18} {'Set':<8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 65)

    for classifier_name, classifier_results in all_results.items():
        for set_name in ['train', 'test']:
            m = classifier_results[set_name]
            print(f"{classifier_name:<18} {set_name:<8} {m['accuracy']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f}")

    # 1.9.2 Error analysis
    print("\n--- 1.9.2 Error Analysis (Test Set) ---")
    print(f"{'Classifier':<18} {'Total Errors':>15} {'FP':>8} {'FN':>8}")
    print("-" * 50)

    for classifier_name in ['LDA', 'QDA', 'LDA (reduced)', 'QDA (reduced)']:
        m = all_results[classifier_name]['test']
        print(
            f"{classifier_name:<18} {m['FP'] + m['FN']:>15} {m['FP']:>8} {m['FN']:>8}")

    # Plot confusion matrices
    for classifier_name in ['LDA', 'QDA']:
        cm = all_results[classifier_name]['test']['confusion_matrix']
        plot_confusion_matrix_multiclass(cm, ['Benign', 'Malignant'],
                                         title=f'Confusion Matrix ({classifier_name} Test)')

    # 1.9.3 Generalization analysis
    print("\n--- 1.9.3 Generalization Analysis ---")
    for classifier_name in ['LDA', 'QDA']:
        train_acc = all_results[classifier_name]['train']['accuracy']
        test_acc = all_results[classifier_name]['test']['accuracy']
        gap = train_acc - test_acc
        print(
            f"{classifier_name}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}")

    # Plot comparison
    plot_classifier_comparison(all_results, metric='accuracy')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Section 1.1: Data Loading and Exploration
    X, y, feature_names = load_and_explore_data()

    # Section 1.4: Train and evaluate classifiers
    # Section 1.4.1: Split data
    print("\n" + "=" * 60)
    print("TASK 1.4.1: TRAIN-TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    results_original = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test)

    # Section 1.5: Covariance matrix analysis
    cov_analysis = analyze_covariance_matrix(X_train)

    # Section 1.6: Variance analysis
    variances, low_var_features = analyze_feature_variances(
        X_train, feature_names)

    # Section 1.7: Feature elimination
    X_train_reduced, X_test_reduced, kept_features = \
        variance_based_feature_elimination(
            X_train, X_test, feature_names, percentile=10)

    # Section 1.8: Retrain with reduced features
    results_reduced = retrain_with_reduced_features(
        X_train_reduced, X_test_reduced, y_train, y_test, kept_features)

    # Section 1.9: Performance evaluation
    performance_evaluation(results_original, results_reduced)
