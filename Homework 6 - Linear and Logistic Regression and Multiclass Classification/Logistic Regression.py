import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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
            data_encoded[col], prefix=col, drop_first=True)
        data_encoded = pd.concat([data_encoded, dummies], axis=1)
        data_encoded = data_encoded.drop(col, axis=1)

    return data_encoded


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


pass


if __name__ == "__main__":
    load_and_preprocess_data()
