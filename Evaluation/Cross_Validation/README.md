# Cross-Validation in Model Evaluation

## Introduction

This section introduces cross-validation techniques for evaluating the performance of machine learning models. Cross-validation helps assess how well a model generalizes to unseen data by partitioning the dataset into subsets, training on some and testing on others. We use the scikit-learn library to implement common cross-validation methods such as `cross_val_score` and `cross_validate`.

The notebook `demo.ipynb` contains illustrative examples using a customer churn dataset for easy understanding.

## Cross-Validation Methods

### 1. cross_val_score

#### Explanation
`cross_val_score` performs cross-validation and returns an array of scores for each fold. It is a simple way to evaluate a model's performance using a specified scoring metric.

#### Advantages
- Easy to use for quick evaluation.
- Returns scores directly for averaging.

#### Disadvantages
- Does not provide detailed results like training times or fitted estimators.

#### Example
```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_leaf=10)
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Average score:", scores.mean())
```

### 2. cross_validate

#### Explanation
`cross_validate` is more comprehensive, allowing multiple scoring metrics and returning detailed results including fit times, score times, and optionally the fitted estimators.

#### Advantages
- Provides more information (e.g., fit times, multiple scores).
- Can return estimators for further use.

#### Disadvantages
- Slightly more complex to set up.

#### Example
```python
from sklearn.model_selection import cross_validate

clf = DecisionTreeClassifier(min_samples_leaf=10)
cv_results = cross_validate(clf, X_train, y_train, cv=10, return_estimator=True)
print('Mean score:', cv_results['test_score'].mean())
```

### 3. Custom Scoring

#### Explanation
You can specify multiple scoring metrics (e.g., ROC-AUC, accuracy, F1) to evaluate different aspects of model performance.

#### Example
```python
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

scoring = ['roc_auc', 'accuracy', 'f1']
scores = cross_validate(clf, X_train, y_train, cv=10, scoring=scoring)
print("Mean ROC-AUC:", scores['test_roc_auc'].mean())
```

## How to Use

1. Open the notebook `demo.ipynb` in VS Code or Jupyter.
2. Run the cells to see cross-validation with sample data from the customer churn dataset.
3. Modify the classifier, dataset, or parameters to experiment with your own models.
4. Compare scores across folds to assess model stability and performance.

## Notes

- Cross-validation helps prevent overfitting by evaluating on multiple data splits.
- Common choices for `cv` include 5 or 10 folds; adjust based on dataset size.
- For imbalanced datasets, consider stratified cross-validation.
- Refer to scikit-learn documentation: [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

If you have questions, check the notebook or reach out!