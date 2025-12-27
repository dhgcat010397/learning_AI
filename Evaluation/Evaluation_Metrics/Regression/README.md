# Accuracy and Precision in Model Evaluation

## Introduction

This section introduces methods for evaluating the performance of regression models in machine learning. We use the scikit-learn library to calculate common metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R-squared (R²). These metrics help measure how accurately the model predicts compared to actual values.

The notebook `how_to_use.ipynb` contains illustrative examples with simple sample data for easy understanding.

## Evaluation Methods

### 1. Mean Absolute Error (MAE)

#### Explanation
MAE measures the average absolute difference between predicted and actual values. It indicates the average deviation without considering the direction of the error (positive or negative).

#### Formula
\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]
Where:
- \( y_i \): Actual value
- \( \hat{y}_i \): Predicted value
- \( n \): Number of samples

#### Advantages
- Easy to understand and calculate.
- Not heavily influenced by outliers like MSE.

#### Disadvantages
- Does not distinguish between large and small errors (all errors are equally weighted).

#### Example
```python
from sklearn.metrics import mean_absolute_error

y_test = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)  # Output: 0.5
```

### 2. Root Mean Squared Error (RMSE)

#### Explanation
RMSE is the square root of the Mean Squared Error (MSE). It measures the standard deviation of the prediction errors, emphasizing larger errors more.

#### Formula
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

#### Advantages
- Suitable when you want to penalize large errors (outliers) more.
- Units match the target variable, making it easy to interpret.

#### Disadvantages
- Heavily influenced by outliers.

#### Example
```python
from sklearn.metrics import root_mean_squared_error

y_test = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

rmse = root_mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)  # Output: 0.6123724356957945
```

### 3. Mean Squared Error (MSE)

#### Explanation
MSE measures the average of the squared differences between predicted and actual values. It heavily penalizes large errors due to squaring.

#### Formula
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

#### Advantages
- Suitable for optimization (e.g., gradient descent) due to continuous derivatives.
- Emphasizes large errors.

#### Disadvantages
- Units are squared, making direct interpretation difficult.
- Heavily influenced by outliers.

#### Example
```python
from sklearn.metrics import root_mean_squared_error

y_test = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = root_mean_squared_error(y_test, y_pred)  # squared=True by default
print("MSE:", mse)  # Output: 0.375
```

### 4. R-squared (R²)

#### Explanation
R² measures the proportion of variance in the dependent variable explained by the model. It indicates how much of the data's variability is accounted for by the model.

#### Formula
\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]
Where:
- \( SS_{res} \): Residual sum of squares
- \( SS_{tot} \): Total sum of squares

#### Advantages
- Ranges from 0 to 1 (or negative if worse than baseline).
- Easy to compare across models.

#### Disadvantages
- Does not indicate absolute error.
- Can artificially increase with irrelevant features in complex models.

#### Example
```python
from sklearn.metrics import r2_score

y_test = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

r2 = r2_score(y_test, y_pred)
print("R2:", r2)  # Output: 0.9565217391304348
```

## How to Use

1. Open the notebook `how_to_use.ipynb` in VS Code or Jupyter.
2. Run the cells to see calculations with sample data.
3. Modify `y_test` and `y_pred` to experiment with your own data.
4. Compare metrics to select the best model for your regression task.

## Notes

- These metrics are suitable for regression; not for classification.
- In practice, combine multiple metrics for comprehensive evaluation.
- Refer to scikit-learn documentation: [Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

If you have questions, check the notebook or reach out!</content>
<parameter name="filePath">d:\Dev\Python\AI\project\learning_AI\Accuracy_Precision\README.md