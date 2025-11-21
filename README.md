# Pattern Recognition Algorithms

A comprehensive collection of pattern recognition and statistical learning algorithms implemented in Python. This repository provides clean, well-documented implementations of fundamental algorithms used in machine learning and data analysis.

## Features

- **Bayesian Decision Theory**
  - Bayes classifier with prior probabilities
  - Maximum likelihood estimation
  - Minimum error rate classification
  - Multi-class decision boundaries

- **Data Preprocessing**
  - Whitening transformation (ZCA whitening)
  - Principal Component Analysis (PCA)
  - Data normalization and standardization
  - Covariance matrix computation

- **Classification Algorithms**
  - Naive Bayes classifier
  - Gaussian discriminant analysis
  - K-Nearest Neighbors (KNN)
  - Linear discriminant analysis

## Usage

### Bayes Decision Theory
```python
from algorithms.bayes import BayesClassifier

classifier = BayesClassifier()
classifier.train(training_data, labels)
prediction = classifier.predict(test_sample)
```

### Whitening Transformation
```python
from algorithms.whitening import whitening_transform

whitened_data = whitening_transform(data)
```


## Examples

Check the `/examples` directory for detailed usage examples including:
- Image classification with Bayes classifier
- Data preprocessing pipelines
- Feature extraction and dimensionality reduction

## Technologies

- Python 3.8+
- NumPy
- Matplotlib (for visualizations)
