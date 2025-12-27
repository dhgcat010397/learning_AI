# K-Nearest Neighbors (KNN)

## Introduction

K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for both classification and regression tasks in machine learning. It works by finding the k closest data points (neighbors) in the training set to a new data point and making predictions based on the majority class (for classification) or average value (for regression) of those neighbors.

This section demonstrates how to implement KNN using scikit-learn with the Iris dataset, including data preparation, model training, and evaluation.

The notebook `demo.ipynb` contains a complete example with the Iris dataset for easy understanding.

## How KNN Works

### Algorithm Steps
1. **Choose k**: Select the number of nearest neighbors to consider.
2. **Calculate Distance**: Compute the distance between the new point and all points in the training set (commonly using Euclidean distance).
3. **Find Neighbors**: Identify the k points with the smallest distances.
4. **Make Prediction**:
   - For classification: Return the most common class among the k neighbors.
   - For regression: Return the average value of the k neighbors.

### Distance Metrics
- **Euclidean Distance**: Most common, straight-line distance.
- **Manhattan Distance**: Sum of absolute differences.
- **Minkowski Distance**: Generalized distance metric.

## Advantages

- Simple to understand and implement.
- No training phase (lazy learning).
- Works well with small datasets.
- Can be used for both classification and regression.

## Disadvantages

- Computationally expensive for large datasets (needs to compute distances to all training points).
- Sensitive to irrelevant features and the curse of dimensionality.
- Requires careful choice of k and distance metric.
- Not suitable for high-dimensional data.

## Parameters

- **n_neighbors (k)**: Number of neighbors to consider. Common values: 3, 5, 7.
- **weights**: How to weight the neighbors ('uniform' or 'distance').
- **metric**: Distance metric to use ('euclidean', 'manhattan', etc.).
- **algorithm**: Algorithm for computing neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').

## Example Usage

### Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("datasets/iris_dataset.csv")

# Prepare features and target
X = df.iloc[:, 0:4]  # Features
y = df.iloc[:, 4]    # Target (class_name)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training
```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
```

### Model Evaluation
```python
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, knn.predict_proba(X_test), multi_class='ovr'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
```

## Choosing the Right k

- **Small k**: Can be noisy and sensitive to outliers.
- **Large k**: Smoother decision boundaries but may include points from other classes.
- Use cross-validation to find optimal k.

## When to Use KNN

- Small to medium-sized datasets.
- When interpretability is important.
- When you have continuous features.
- For baseline comparisons with other algorithms.

## How to Use

1. Open the notebook `demo.ipynb` in VS Code or Jupyter.
2. Run the cells to see the complete KNN implementation with the Iris dataset.
3. Modify parameters like `k`, dataset, or evaluation metrics to experiment.
4. Try different distance metrics or weighting schemes.

## Notes

- KNN is sensitive to feature scaling; consider normalizing features.
- For multi-class problems, use appropriate averaging for ROC-AUC.
- Consider dimensionality reduction techniques for high-dimensional data.
- Refer to scikit-learn documentation: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

If you have questions, check the notebook or reach out!</content>
<parameter name="filePath">d:\Dev\Python\AI\project\learning_AI\Algorithms\K_Nearest_Neighbors\README.md