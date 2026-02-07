import numpy as np
import os
from DimensionReduction import PCA, FisherLDA
from Classification import NearestCentroidClassifier
from models import QDA
from plots_hw7 import (plot_mean_face, plot_original_vs_mean_subtracted,
                       plot_pca_eigenvalues, plot_eigenfaces,
                       plot_reconstruction_comparison, plot_accuracy_vs_components,
                       plot_lda_eigenvalues, plot_separability_metric,
                       plot_multiclass_confusion_matrix, plot_pca_vs_pca_lda_comparison)


# =============================================================================
# SECTION 2.1: DATA LOADING (ORL Face Database)
# =============================================================================

def load_orl_database(data_path='The ORL face database'):

    images = []
    labels = []

    # ORL database has 40 subjects, 10 images each (Said in Problem Defenition)
    for subject_id in range(1, 41):
        subject_dir = os.path.join(data_path, f's{subject_id}')
        if not os.path.exists(subject_dir):
            continue

        for img_id in range(1, 11):
            img_path = os.path.join(subject_dir, f'{img_id}.pgm')
            if os.path.exists(img_path):
                img = read_pgm(img_path)
                # Converts the 2D array (112×92) into a 1D array of 10,304 elements (10304×1 vectors)
                images.append(img.flatten())
                labels.append(subject_id)

    X = np.array(images)
    y = np.array(labels)

    image_shape = (112, 92)

    return X, y, image_shape


def read_pgm(filename):
    # The read_pgm function:
    #   1. reads a PGM image from a specified file
    #   2. validates its format
    #   3. extracts image dimensions
    #   4. reads the pixel data
    #   5. returning the image as a 2D NumPy array of floating-point values.

    with open(filename, 'rb') as f:
        magic = f.readline().decode('utf-8').strip()
        # Check if the magic number is 'P5', indicating a binary PGM image file.
        if magic != 'P5':
            raise ValueError(f'Invalid PGM magic number: {magic}')

        # Skip comments
        # Continuously read lines from the file until a non-comment line (not starting with '#') is found.
        # Comments are ignored and skipped.
        line = f.readline().decode('utf-8')
        while line.startswith('#'):
            line = f.readline().decode('utf-8')

        width, height = map(int, line.strip().split())

        # Read image data - only read exactly width*height bytes
        expected_size = width * height
        image_data = f.read(expected_size)
        image = np.frombuffer(image_data, dtype=np.uint8, count=expected_size)
        image = image.reshape((height, width))

    return image.astype(np.float64)


def perform_pca_analysis(X, y, image_shape):
    print("\n" + "=" * 60)
    print("TASK 2.2.2: PCA IMPLEMENTATION")
    print("=" * 60)

    # 2.2.1.3 Compute mean face
    mean_face = np.mean(X, axis=0)
    print(f"\n--- 2.2.1.3 Mean Face ---")
    print(f"Mean face shape: {mean_face.shape}")
    plot_mean_face(mean_face, image_shape)

    # 2.2.1.4 Normalize (center) images
    X_centered = X - mean_face
    plot_original_vs_mean_subtracted(X, X_centered, y, image_shape)

    # 2.2.1.5 Additional normalization - mean subtraction
    print(f"\n--- 2.2.1.5 Normalization Options ---")
    X_norm1 = X_centered
    print("Mean subtraction only")
    print(
        f"  - Variance range: [{X_norm1.var(axis=0).min():.4f}, {X_norm1.var(axis=0).max():.4f}]")

    # 2.2.2.1-2 Fit PCA
    print(f"\n--- 2.2.2.1-2 PCA Fitting ---")
    pca = PCA()
    pca.fit(X)

    print(f"Number of principal components: {len(pca.eigenvalues)}")
    print(f"Top 5 eigenvalues: {pca.eigenvalues[:5]}")

    # 2.2.2.3 Plot eigenvalues
    plot_pca_eigenvalues(pca.eigenvalues, pca.cumulative_variance_ratio)

    # 2.2.2.4 Visualize top 50 eigenfaces
    plot_eigenfaces(pca.components, n_components=50, image_shape=image_shape)

    # 2.2.3 Projection and Reconstruction
    print("\n" + "=" * 60)
    print("TASK 2.2.3: PROJECTION & RECONSTRUCTION")
    print("=" * 60)

    # 2.2.3.1-2 Project and reconstruct
    n_components_test = 50
    X_projected = pca.transform(X, n_components=n_components_test)
    X_reconstructed = pca.inverse_transform(
        X_projected, n_components=n_components_test)

    print(
        f"\n--- 2.2.3.1-2 Reconstruction with {n_components_test} components ---")
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"Mean squared reconstruction error: {reconstruction_error:.4f}")

    plot_reconstruction_comparison(X, X_reconstructed, y, image_shape)

    # 2.2.3.3 Components for 90% variance
    n_90 = pca.get_n_components_for_variance(0.9)
    print(f"\n--- 2.2.3.3 Variance Retention ---")
    print(f"Components for 90% variance: {n_90}")
    print(
        f"Components for 95% variance: {pca.get_n_components_for_variance(0.95)}")
    print(
        f"Components for 99% variance: {pca.get_n_components_for_variance(0.99)}")

    return pca


def perform_fisher_lda_analysis(X, y, image_shape, pca=None, n_pca_components=100):

    print("\n" + "=" * 60)
    print("TASK 2.3: FISHER LINEAR DISCRIMINANT ANALYSIS")
    print("=" * 60)

    # 2.3.1 Preprocessing
    print(f"\n--- 2.3.1 Data Preprocessing ---")

    # Apply PCA first for dimensionality reduction (common practice)
    if pca is not None:
        X_pca = pca.transform(X, n_components=n_pca_components)
        print(f"Applied PCA: {X.shape[1]} -> {X_pca.shape[1]} dimensions")
    else:
        X_pca = X
        print(f"No PCA preprocessing, using original {X.shape[1]} dimensions")

    # 2.3.2 Compute scatter matrices
    print(f"\n--- 2.3.2 Scatter Matrices ---")

    n_samples, n_features = X_pca.shape
    classes = np.unique(y)
    n_classes = len(classes)
    overall_mean = np.mean(X_pca, axis=0)
    class_means = np.zeros((n_classes, n_features))
    for idx, c in enumerate(classes):
        class_means[idx] = np.mean(X_pca[y == c], axis=0)

    # Within-class scatter S_W
    S_W = np.zeros((n_features, n_features))
    for idx, c in enumerate(classes):
        X_c = X_pca[y == c]
        diff = X_c - class_means[idx]
        S_W += diff.T @ diff

    # Between-class scatter S_B
    S_B = np.zeros((n_features, n_features))
    for idx, c in enumerate(classes):
        n_c = np.sum(y == c)
        mean_diff = (class_means[idx] - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    print(f"S_W shape: {S_W.shape}")
    print(f"S_B shape: {S_B.shape}")
    print(f"S_W rank: {np.linalg.matrix_rank(S_W)}")
    print(f"S_B rank: {np.linalg.matrix_rank(S_B)}")

    # 2.3.3 Fit Fisher LDA
    print(f"\n--- 2.3.3 Eigenvalue Problem ---")
    lda = FisherLDA()
    lda.fit(X_pca, y)

    print(f"Number of discriminants: {len(lda.eigenvalues)}")
    print(f"Maximum possible discriminants: {n_classes - 1}")
    print(f"Top 5 eigenvalues: {lda.eigenvalues[:5]}")

    plot_lda_eigenvalues(np.abs(lda.eigenvalues))

    # 2.3.4 Separability analysis
    print(f"\n--- 2.3.4 Separability Analysis ---")

    # Define separability metric: sum of eigenvalues
    n_components_list = list(range(1, min(40, len(lda.eigenvalues) + 1)))
    separability_scores = [np.sum(np.abs(lda.eigenvalues[:n]))
                           for n in n_components_list]

    plot_separability_metric(n_components_list, separability_scores)

    print(f"Maximum discriminants constrained by: min(n_classes-1, n_features)")
    print(f"For ORL: min(40-1, {n_features}) = {min(39, n_features)}")

    # 2.3.5 Projection
    print(f"\n--- 2.3.5 Projection to LDA Space ---")

    # Choose optimal number of components
    optimal_n = min(39, len(lda.eigenvalues))  # Use all available
    X_lda = lda.transform(X_pca, n_components=optimal_n)

    print(f"Projected dimensions: {X_lda.shape[1]}")
    print(f"Trade-off: More components = more info but higher dimensionality")

    return lda, X_lda, X_pca


def evaluate_pca_classification(X, y, pca, image_shape):
    def train_test_split_leave_one_out(X, y):
        train_indices = []
        test_indices = []

        unique_labels = np.unique(y)

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)

            test_indices.append(label_indices[0])
            train_indices.extend(label_indices[1:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    print("\n" + "=" * 60)
    print("TASK 2.2.4: FACE RECOGNITION / CLASSIFICATION")
    print("=" * 60)

    # 2.2.4.1 Split data
    print(f"\n--- 2.2.4.1 Train-Test Split ---")
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split_leave_one_out(X, y)
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")

    pca_train = PCA()
    pca_train.fit(X_train)

    # 2.2.4.2-4 Evaluate different classifiers and component numbers
    print(f"\n--- 2.2.4.2-4 Classifier Evaluation ---")

    n_components_list = [10, 20, 30, 50, 75, 100, 150, 200]
    results = {
        'Nearest Centroid': [],
        'Bayesian': []
    }

    for n_comp in n_components_list:
        X_train_pca = pca_train.transform(X_train, n_components=n_comp)
        X_test_pca = pca_train.transform(X_test, n_components=n_comp)

        nc = NearestCentroidClassifier()
        nc.fit(X_train_pca, y_train)
        nc_pred = nc.predict(X_test_pca)
        nc_acc = np.mean(nc_pred == y_test)
        results['Nearest Centroid'].append(nc_acc)

        bc = QDA()
        bc.fit(X_train_pca, y_train)
        bc_pred = bc.predict(X_test_pca)
        bc_acc = np.mean(bc_pred == y_test)
        results['Bayesian'].append(bc_acc)

    print(f"\n{'n_components':<15} {'Nearest Centroid':>20} {'Bayesian':>15}")
    print("-" * 55)
    for i, n_comp in enumerate(n_components_list):
        print(f"{n_comp:<15} {results['Nearest Centroid'][i]:>20.4f} "
              f"{results['Bayesian'][i]:>15.4f}")

    # 2.2.4.5 Plot accuracy vs components
    plot_accuracy_vs_components(n_components_list, results,
                                title='PCA: Classification Accuracy vs Number of Components')

    # 2.2.4.6 Trade-off analysis
    print(f"\n--- 2.2.4.6 Trade-off Analysis ---")
    best_nc_idx = np.argmax(results['Nearest Centroid'])
    best_bc_idx = np.argmax(results['Bayesian'])

    print(f"Best Nearest Centroid: {n_components_list[best_nc_idx]} components "
          f"({results['Nearest Centroid'][best_nc_idx]:.4f})")
    print(f"Best Bayesian: {n_components_list[best_bc_idx]} components "
          f"({results['Bayesian'][best_bc_idx]:.4f})")

    return results, n_components_list, pca_train, X_train, X_test, y_train, y_test


def evaluate_pca_plus_lda(X_train, X_test, y_train, y_test, pca_train):
    print(f"\n--- 2.2.4.7 PCA vs PCA+LDA Comparison ---")

    n_pca_components_list = [50, 75, 100, 150]
    pca_only_acc = []
    pca_lda_acc = []

    for n_pca in n_pca_components_list:
        # PCA only
        X_train_pca = pca_train.transform(X_train, n_components=n_pca)
        X_test_pca = pca_train.transform(X_test, n_components=n_pca)

        nc = NearestCentroidClassifier()
        nc.fit(X_train_pca, y_train)
        pca_only_acc.append(np.mean(nc.predict(X_test_pca) == y_test))

        # PCA + LDA
        lda = FisherLDA()
        lda.fit(X_train_pca, y_train)

        X_train_lda = lda.transform(X_train_pca)
        X_test_lda = lda.transform(X_test_pca)

        nc_lda = NearestCentroidClassifier()
        nc_lda.fit(X_train_lda, y_train)
        pca_lda_acc.append(np.mean(nc_lda.predict(X_test_lda) == y_test))

    print(f"\n{'n_PCA':<15} {'PCA Only':>15} {'PCA + LDA':>15}")
    print("-" * 50)
    for i, n_pca in enumerate(n_pca_components_list):
        print(f"{n_pca:<15} {pca_only_acc[i]:>15.4f} {pca_lda_acc[i]:>15.4f}")

    plot_pca_vs_pca_lda_comparison(
        pca_only_acc, pca_lda_acc, n_pca_components_list)

    return pca_only_acc, pca_lda_acc


def evaluate_lda_classification(X_pca, y, lda):

    print("\n" + "=" * 60)
    print("TASK 2.3.6: LDA CLASSIFICATION EVALUATION")
    print("=" * 60)

    # Split data
    np.random.seed(42)
    train_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        indices = np.where(y == c)[0]
        train_mask[indices[1:]] = True  # Leave one out

    X_train = X_pca[train_mask]
    X_test = X_pca[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]

    lda_train = FisherLDA()
    lda_train.fit(X_train, y_train)

    # Evaluate with different numbers of LDA components
    n_components_list = list(
        range(5, min(40, len(lda_train.eigenvalues) + 1), 5))
    accuracies = []

    for n_comp in n_components_list:
        X_train_lda = lda_train.transform(X_train, n_components=n_comp)
        X_test_lda = lda_train.transform(X_test, n_components=n_comp)

        clf = NearestCentroidClassifier()
        clf.fit(X_train_lda, y_train)
        pred = clf.predict(X_test_lda)
        accuracies.append(np.mean(pred == y_test))

    print(f"\n--- Classification Results ---")
    print(f"{'n_LDA':<15} {'Accuracy':>15}")
    print("-" * 35)
    for i, n_comp in enumerate(n_components_list):
        print(f"{n_comp:<15} {accuracies[i]:>15.4f}")

    plot_accuracy_vs_components(n_components_list, {'LDA Classifier': accuracies},
                                title='LDA: Classification Accuracy vs Number of Discriminants')

    # Best confusion matrix
    best_idx = np.argmax(accuracies)
    best_n = n_components_list[best_idx]

    X_train_lda = lda_train.transform(X_train, n_components=best_n)
    X_test_lda = lda_train.transform(X_test, n_components=best_n)

    clf = NearestCentroidClassifier()
    clf.fit(X_train_lda, y_train)
    best_pred = clf.predict(X_test_lda)

    # Compute confusion matrix
    classes = np.unique(y)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_test, best_pred):
        true_idx = np.where(classes == true)[0][0]
        pred_idx = np.where(classes == pred)[0][0]
        cm[true_idx, pred_idx] += 1

    print(f"\n--- Best Model: {best_n} LDA components ---")
    print(f"Accuracy: {accuracies[best_idx]:.4f}")

    plot_multiclass_confusion_matrix(cm, n_classes=40,
                                     title=f'Confusion Matrix (LDA, n={best_n})')

    print(f"\n--- Analysis ---")
    print("Diminishing returns with more components:")
    for i in range(1, len(accuracies)):
        improvement = accuracies[i] - accuracies[i-1]
        print(f"  {n_components_list[i-1]} -> {n_components_list[i]}: "
              f"{improvement:+.4f}")

    return accuracies, n_components_list


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Section 2.1: Load ORL database
    X, y, image_shape = load_orl_database()

    # Section 2.2: PCA Analysis
    pca = perform_pca_analysis(X, y, image_shape)

    # Section 2.2.4: PCA Classification
    pca_results, n_components_list, pca_train, X_train, X_test, y_train, y_test = \
        evaluate_pca_classification(X, y, pca, image_shape)

    # Section 2.2.4.7: PCA vs PCA+LDA
    pca_only_acc, pca_lda_acc = evaluate_pca_plus_lda(
        X_train, X_test, y_train, y_test, pca_train)

    # Section 2.3: Fisher LDA Analysis
    lda, X_lda, X_pca = perform_fisher_lda_analysis(
        X, y, image_shape, pca, n_pca_components=100)

    # Section 2.3.6: LDA Classification
    lda_accuracies, lda_n_components = evaluate_lda_classification(
        X_pca, y, lda)

    # Final summary
    print("\n--- FINAL SUMMARY ---")
    print(f"\nPCA Results:")
    print(
        f"  Best accuracy: {max(max(pca_results['Nearest Centroid']), max(pca_results['Bayesian'])):.4f}")

    print(f"\nPCA + LDA Results:")
    print(f"  Best accuracy: {max(pca_lda_acc):.4f}")

    print(f"\nLDA-only Results:")
    print(f"  Best accuracy: {max(lda_accuracies):.4f}")
