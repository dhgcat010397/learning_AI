# Grid Search for Hyperparameter Optimization

## Introduction

This section introduces Grid Search as a method for hyperparameter tuning in machine learning models. Grid Search exhaustively searches through a specified parameter grid to find the best combination of hyperparameters that maximizes model performance. We use the scikit-learn library to implement Grid Search with cross-validation.

The notebook `demo.ipynb` contains illustrative examples using a customer churn dataset for easy understanding.

## Grid Search Methods

### 1. GridSearchCV

#### Explanation
`GridSearchCV` performs an exhaustive search over specified parameter values for an estimator. It uses cross-validation to evaluate each combination and selects the best one based on the scoring metric.

#### Advantages
- Systematic and thorough search.
- Guarantees finding the best combination within the grid.

#### Disadvantages
- Computationally expensive for large grids.
- Time-consuming for complex models.

#### Example
```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

## How to Use

1. Open the notebook `demo.ipynb` in VS Code or Jupyter.
2. Run the cells to see Grid Search with sample data from the customer churn dataset.
3. Modify the estimator, parameter grid, or scoring metric to experiment with your own models.
4. Compare the best parameters and scores to optimize your model.

## Notes

- Grid Search is ideal for small to medium parameter spaces.
- For larger spaces, consider RandomizedSearchCV.
- Always use cross-validation to avoid overfitting.
- Refer to scikit-learn documentation: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

If you have questions, check the notebook or reach out!