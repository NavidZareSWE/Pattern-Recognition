import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import fit, predict_prob, predict


def softmax(theta, X):
    xThetaT = X @ theta.T
    xThetaT_stable = xThetaT - np.max(xThetaT, axis=1, keepdims=True)
    exp_xThetaT = np.exp(xThetaT_stable)
    probabilities = exp_xThetaT / np.sum(exp_xThetaT, axis=1, keepdims=True)
    return probabilities


def one_hot_encode(y, classes):
    n_samples = len(y)
    one_hot = np.zeros((n_samples, len(classes)))

    for idx, label in enumerate(y):
        class_idx = np.where(classes == label)[0][0]
        one_hot[idx, class_idx] = 1

    return one_hot


# #############################################################################
# One-vs-All (OvA) Classifier
# #############################################################################

def one_vs_all_fit(X, y, learning_rate=0.01, n_iterations=1000):
    classes = np.unique(y)
    classifiers = {}

    for c in classes:
        print(f"\nTraining classifier for class {c}...")
        y_binary = (y == c).astype(int)

        weights, bias, _ = fit(X, y_binary, learning_rate, n_iterations)
        classifiers[c] = {'weights': weights, 'bias': bias}

    return {'classifiers': classifiers, 'classes': classes}


def one_vs_all_predict_prob(X, model):
    classes = model['classes']
    # Shape: (n_samples, n_classes)
    probs = np.zeros((X.shape[0], len(classes)))

    for idx, c in enumerate(classes):
        # "class c vs all others"
        clf = model['classifiers'][c]
        probs[:, idx] = predict_prob(X, clf['weights'], clf['bias'])
    return probs


def one_vs_all_predict(X, model):
    probs = one_vs_all_predict_prob(X, model)
    return model['classes'][np.argmax(probs, axis=1)]


# #############################################################################
# One-vs-One (OvO) Classifier
# #############################################################################

def one_vs_one_fit(X, y, learning_rate=0.01, n_iterations=1000):
    classes = np.unique(y)
    classifiers = {}

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class_i, class_j = classes[i], classes[j]
            print(f"\nTraining classifier: {class_i} vs {class_j}")

            # Boolean Logic to select ONLY samples from these two classes
            bool_logic = (y == class_i) | (y == class_j)
            X_pair = X[bool_logic]
            y_binary = (y[bool_logic] == class_i).astype(int)

            weights, bias, _ = fit(
                X_pair, y_binary, learning_rate, n_iterations)
            classifiers[(class_i, class_j)] = {
                'weights': weights, 'bias': bias}

    return {'classifiers': classifiers, 'classes': classes}


def one_vs_one_predict(X, model):
    classes = model['classes']
    # Shape: (n_samples, n_classes)
    votes = np.zeros((X.shape[0], len(classes)))

    for (class_i, class_j), clf in model['classifiers'].items():
        preds = predict(X, clf['weights'], clf['bias'])

        idx_i = np.where(classes == class_i)[0][0]
        idx_j = np.where(classes == class_j)[0][0]

        # ex:  Classifier (0, 1) predicts: [1, 0, 1]
        votes[:, idx_i] += preds
        votes[:, idx_j] += (1 - preds)

    return classes[np.argmax(votes, axis=1)]


# #############################################################################
# Softmax Regression
# #############################################################################

def softmax_fit(X, y, learning_rate=0.1, n_iterations=1000):
    classes = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(classes)

    weights = np.zeros((n_classes, n_features))
    cost_history = []

    y_one_hot = one_hot_encode(y, classes)

    print("\n--- Training Softmax Regression ---")
    for i in range(n_iterations):
        y_pred = softmax(weights, X)

        # Gradient
        error = y_pred - y_one_hot
        dw = (1 / n_samples) * np.dot(error.T, X)
        weights -= learning_rate * dw

        # Cost
        epsilon = 1e-15
        cost = -(1 / n_samples) * np.sum(
            y_one_hot * np.log(np.clip(y_pred, epsilon, 1 - epsilon))
        )
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return {
        'weights': weights,
        'classes': classes,
        'cost_history': cost_history
    }


def softmax_predict_prob(X, model):
    return softmax(model['weights'], X)


def softmax_predict(X, model):
    probas = softmax_predict_prob(X, model)
    return model['classes'][np.argmax(probas, axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def handle_missing_and_duplicates(X_train, X_test, y_train, y_test):
    print("\n--- Data Cleaning ---")
    # Check & Remove Duplicates (Training Set)
    # Row By Row duplicate Check
    train_combined = np.column_stack([X_train, y_train])
    # Returns the sorted unique elements of an array
    # return_indexbool, optional
    # If True, also return the indices of ar(along the specified axis, if provided, or in the flattened array) that result in the unique array.
    # Read More: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    unique_train, unique_indices = np.unique(
        train_combined, axis=0, return_index=True)

    duplicates_train = len(X_train) - len(unique_indices)
    if duplicates_train > 0:
        X_train = X_train[unique_indices]
        y_train = y_train[unique_indices]

    # Check & Remove Duplicates (Test Set)
        # Row By Row duplicate Check
    test_combined = np.column_stack([X_test, y_test])
    unique_test, unique_indices_test = np.unique(
        test_combined, axis=0, return_index=True)

    duplicates_test = len(X_test) - len(unique_indices_test)
    if duplicates_test > 0:
        X_test = X_test[unique_indices_test]
        y_test = y_test[unique_indices_test]

    # Handle Missing Values
    missing_train = np.isnan(X_train).sum()
    missing_test = np.isnan(X_test).sum()

    if missing_train > 0 or missing_test > 0:
        train_means = np.nanmean(X_train, axis=0)

        # Fill train
        num_cols = X_train.shape[1]
        for i in range(num_cols):
            _n = np.isnan(X_train[:, i])
            X_train[_n, i] = train_means[i]

        # Fill test with train means
        num_cols = X_test.shape[1]
        for i in range(num_cols):
            _n = np.isnan(X_test[:, i])
            X_test[_n, i] = train_means[i]

    return X_train, X_test, y_train, y_test


def detect_and_remove_outliers(X_train, X_test, y_train, y_test, threshold=2.75):
    print("\n--- Outlier Detection (Z-score) ---")

    # Calculate Z-score using training stats
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_std = np.where(X_train_std == 0, 1, X_train_std)

    z_train = np.abs((X_train - X_train_mean) / X_train_std)
    z_test = np.abs((X_test - X_train_mean) / X_train_std)  # Use train stats!

    # Find rows WITHOUT outliers (all z-scores < threshold)
    train_mask = (z_train < threshold).all(axis=1)
    test_mask = (z_test < threshold).all(axis=1)

    # Count and report
    train_outliers = len(X_train) - train_mask.sum()
    test_outliers = len(X_test) - test_mask.sum()
    print(f"Threshold: {threshold}")
    print(f"Outliers removed (train): {train_outliers}")
    print(f"Outliers removed (test): {test_outliers}")

    # Remove outliers
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print(f"Remaining samples - Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data():
    data = pd.read_csv('wine_dataset.csv')

    # 3.2-1 Feature statistics
    print("\n--- Feature Statistics ---")
    print("+" + "-"*75 + "+")
    print(f"| {'Feature':<30} {'Mean':>20} {'Std':>20} |")
    print("+" + "-"*75 + "+")
    feature_cols = [col for col in data.columns if col != 'class_label']
    for col in feature_cols:
        print(
            f"| {col:<30} {data[col].mean():>20.2f} {data[col].std():>20.2f} |")
    print("+" + "-"*75 + "+")

    # 3.2-2 No Categorical Features

    # 3.2-3 Train-test split (75-25)
    X = data.drop('class_label', axis=1).values
    y = data['class_label'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("\n--- Train-Test Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3.2-4 Handle missing values and duplicates
    X_train, X_test, y_train, y_test = handle_missing_and_duplicates(
        X_train, X_test, y_train, y_test
    )
    # 3.2-5 Detect and remove outliers
    X_train, X_test, y_train, y_test = detect_and_remove_outliers(
        X_train, X_test, y_train, y_test, threshold=2.75
    )

    # 3.2-6 Normalize using min-max normalization
    print("\n--- Normalization (Min-Max) ---")
    X_train_min = X_train.min(axis=0)
    X_train_max = X_train.max(axis=0)
    X_train_range = X_train_max - X_train_min
    X_train_range = np.where(X_train_range == 0, 1, X_train_range)

    X_train_normalized = (X_train - X_train_min) / X_train_range
    X_test_normalized = (X_test - X_train_min) / X_train_range

    return X_train_normalized, X_test_normalized, y_train, y_test


if __name__ == "__main__":
    X_train_norm, X_test_norm, y_train, y_test = load_and_preprocess_data()

    # One-vs-All
    print("\n" + "=" * 50)
    print("METHOD 1: ONE-VS-ALL")
    print("=" * 50)
    ova_model = one_vs_all_fit(
        X_train_norm, y_train, learning_rate=0.1, n_iterations=1000)
    ova_pred = one_vs_all_predict(X_test_norm, ova_model)
    ova_acc = accuracy(y_test, ova_pred)
    print(f"\nOvA Test Accuracy: {ova_acc:.4f}")

    # One-vs-One
    print("\n" + "=" * 50)
    print("METHOD 2: ONE-VS-ONE")
    print("=" * 50)
    ovo_model = one_vs_one_fit(
        X_train_norm, y_train, learning_rate=0.1, n_iterations=1000)
    ovo_pred = one_vs_one_predict(X_test_norm, ovo_model)
    ovo_acc = accuracy(y_test, ovo_pred)
    print(f"\nOvO Test Accuracy: {ovo_acc:.4f}")

    # Softmax
    print("\n" + "=" * 50)
    print("METHOD 3: SOFTMAX REGRESSION")
    print("=" * 50)
    softmax_model = softmax_fit(
        X_train_norm, y_train, learning_rate=0.1, n_iterations=1000)
    softmax_pred = softmax_predict(X_test_norm, softmax_model)
    softmax_acc = accuracy(y_test, softmax_pred)
    print(f"\nSoftmax Test Accuracy: {softmax_acc:.4f}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"\n{'Method':<20} {'Accuracy':>10}")
    print("-" * 30)
    print(f"{'One-vs-All':<20} {ova_acc:>10.4f}")
    print(f"{'One-vs-One':<20} {ovo_acc:>10.4f}")
    print(f"{'Softmax':<20} {softmax_acc:>10.4f}")
