import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def encode_categorical(data):
    data_encoded = data.copy()

    # "Inherent Ordering" Means:
    # The order itself is predictive of the outcome
    # Ordinal encoding for ordered categories
    education_order = {'unknown': 0, 'primary': 1,
                       'secondary': 2, 'tertiary': 3}
    data_encoded['education'] = data_encoded['education'].map(education_order)

    # yes/no cols
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        data_encoded[col] = data_encoded[col].map({'yes': 1, 'no': 0})
    data_encoded['y'] = data_encoded['y'].map({'yes': 1, 'no': 0})

    # One-hot encoding for nominal categories
    nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
    for col in nominal_cols:
        # Convert categorical variable into dummy/indicator variables.
        # drop_first: bool, default False
        #
        # If set to True, this option drops the first dummy variable
        # for each categorical column, reducing the number of dummy
        # variables from k to k-1.
        # The presence of the dropped category can be inferred
        # from the others. Use this option to simplify the model
        # and enhance interpretability without losing information.
        # Read More: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
        dummies = pd.get_dummies(
            data_encoded[col], prefix=col, drop_first=True, dtype=int)
        data_encoded = pd.concat([data_encoded, dummies], axis=1)
        data_encoded = data_encoded.drop(col, axis=1)

    return data_encoded


def filter_features_by_covariance(data_encoded):
    X = data_encoded.drop('y', axis=1)
    # Reson for Transpose:
    # A 1-D or 2-D array containing multiple variables and observations.
    # Each row of m represents a variable, and each column a single
    # observation of all those variables. Also see rowvar below.
    # Read More: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    cov_matrix = np.cov(X.T)
    num_features = cov_matrix.shape[0]
    feature_covariances = {}

    for i in range(num_features):
        upper_triangle_row = cov_matrix[i, i+1:]
        el_2 = i + 1
        for j, cov_value in enumerate(upper_triangle_row):
            feature_pair = (i, el_2)
            feature_covariances[feature_pair] = abs(cov_value)
            el_2 += 1

    sorted_pairs = sorted(
        feature_covariances.items(),
        key=lambda x: x[1],
        reverse=True
    )
    unique_features = set()
    for rank, (pair, cov_val) in enumerate(sorted_pairs[:7]):
        feature_i, feature_j = pair
        feature_name_i = data_encoded.columns[feature_i]
        feature_name_j = data_encoded.columns[feature_j]
        unique_features.add(feature_name_i)
        unique_features.add(feature_name_j)

    top_features = list(unique_features)
    data_selected = data_encoded[top_features + ['y']]

    return data_selected


def standardize_features(data_selected):
    X = data_selected.drop('y', axis=1)
    y = data_selected['y']

    potential_numerical_cols = ['age', 'balance', 'day', 'duration',
                                'campaign', 'pdays', 'previous']
    numerical_cols = [
        col for col in potential_numerical_cols if col in X.columns]
    X_standardized = X.copy()
    if numerical_cols:
        for col in numerical_cols:
            col_mean = X[col].mean()
            col_std = X[col].std()
            if col_std == 0:
                col_std = 1             # Avoid division by zero

            X_standardized[col] = (X[col] - col_mean) / col_std
    else:
        print("No continuous numerical features found in selected features.")

    # Combine back with y
    data_standardized = pd.concat([X_standardized, y], axis=1)

    return data_standardized


def load_and_preprocess_data():
    # 2.2-1 Load the dataset
    data = pd.read_csv('bank-full.csv', sep=';')

    # 2.2-2 Number of Samples and Features
    print("\n--- Data Overview ---")
    print(f"Total samples: {data.shape[0]:,}")
    print(f"Total features: {data.shape[1]}\n")

    # 2.2-3 Number of numerical and categorical features
    print("--- Data Info ---")
    print(data.info())
    print("\n")

    # 2.2-4 Encode Categorical Features
    data_encoded = encode_categorical(data)

    # 2.2-5 Check if the dataset is balanced or imbalanced
    counts = data_encoded['y'].value_counts()
    ratio = counts.max() / counts.min()

    print("--- Is Dataset Balanced? ---")
    print(data_encoded['y'].value_counts(normalize=True) * 100)
    print(f"\nImbalance ratio: {ratio:.1f}:1")
    print(f"Result: {'Imbalanced' if ratio > 2 else 'Balanced'}")

    # 2.2-6 Feature Selection via Correlation Analysis
    data_selected = filter_features_by_covariance(data_encoded)

    # 2.2-7 Standardization of Numerical Features
    data_standardized = standardize_features(data_selected)

    # 2.2-9 Split into training and testing sets
    # Extract features and labels from standardized data
    from sklearn.model_selection import train_test_split
    X = data_standardized.drop('y', axis=1)
    y = data_standardized['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n" + "=" * 40)
    print("         TRAIN-TEST SPLIT")
    print("=" * 40)
    print(f"  {'Dataset':<12} {'Samples':>10} {'Features':>10}")
    print("-" * 40)
    print(f"  {'X_train':<12} {X_train.shape[0]:>10,} {X_train.shape[1]:>10}")
    print(f"  {'X_test':<12} {X_test.shape[0]:>10,} {X_test.shape[1]:>10}")
    print(f"  {'y_train':<12} {y_train.shape[0]:>10,} {'-':>10}")
    print(f"  {'y_test':<12} {y_test.shape[0]:>10,} {'-':>10}")
    print("=" * 40)

    return X_train, X_test, y_train, y_test


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(y, y_pred):
    n = len(y)
    epsilon = 1e-15  # Avoid 0
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    cost = -(1/n) * np.sum(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )
    return cost


def initialize_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias


def predict_prob(X, weights, bias):
    # z = X · w + b
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def predict(X, weights, bias, threshold=0.5):
    probabilities = predict_prob(X, weights, bias)
    return (probabilities >= threshold).astype(int)


def fit(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights, bias = initialize_parameters(n_features)
    cost_history = []

    print("\n--- Training Progress ---")
    for i in range(n_iterations):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        cost = compute_cost(y, y_pred)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return weights, bias, cost_history


def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate accuracy as the proportion of true results (TP + TN)
    # to the total number of cases (TP + TN + FP + FN).
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate precision as the ratio of true positives (TP) to
    # the sum of true positives and false positives (TP + FP);
    # return 0 if there are no predicted positives.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Calculate recall as the ratio of true positives (TP) to
    # the sum of true positives and false negatives (TP + FN);
    # return 0 if there are no actual positives.
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 score as the harmonic mean of precision and recall;
    # return 0 if both are zero to avoid division by zero.
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix
    }


if __name__ == "__main__":
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. Convert to numpy arrays (fit expects numpy arrays)
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values

    # 3. Train the model
    weights, bias, cost_history = fit(
        X_train_np,
        y_train_np,
        learning_rate=0.01,
        n_iterations=1000
    )

    # 4. Make predictions
    y_train_pred = predict(X_train_np, weights, bias)
    y_test_pred = predict(X_test_np, weights, bias)

    # 5. Evaluate performance
    print("\n--- Training Set Metrics ---")
    train_metrics = calculate_metrics(y_train_np, y_train_pred)
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1 Score:  {train_metrics['f1_score']:.4f}")

    print("\n--- Test Set Metrics ---")
    test_metrics = calculate_metrics(y_test_np, y_test_pred)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1_score']:.4f}")

    # 6. Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.grid(True)
    plt.show()
