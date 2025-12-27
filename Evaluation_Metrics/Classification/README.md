# Evaluation Metrics for Classification

## Introduction

This section introduces methods for evaluating the performance of classification models in machine learning. We use the scikit-learn library to calculate common metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC. These metrics help measure how accurately the model classifies compared to actual labels.

The notebook `how_to_use.ipynb` contains illustrative examples with simple sample data for easy understanding.

## Evaluation Methods

### 1. Accuracy

#### Explanation
Accuracy measures the proportion of correct predictions out of all predictions made. It is the most straightforward metric for classification.

#### Formula
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]
Where:
- \( TP \): True Positives
- \( TN \): True Negatives
- \( FP \): False Positives
- \( FN \): False Negatives

#### Advantages
- Easy to understand and calculate.
- Provides an overall sense of model performance.

#### Disadvantages
- Can be misleading in imbalanced datasets (e.g., when one class dominates).

#### Example
```python
from sklearn.metrics import accuracy_score

Y_test = [0, 1, 1, 0]
Y_pred = [0, 1, 0, 0]

acc = accuracy_score(Y_test, Y_pred)
print("Accuracy:", acc)  # Output: 0.75
```

### 2. Precision

#### Explanation
Precision measures the proportion of true positive predictions out of all positive predictions. It indicates how many of the predicted positives are actually positive.

#### Formula
\[
Precision = \frac{TP}{TP + FP}
\]

#### Advantages
- Useful when the cost of false positives is high (e.g., spam detection).

#### Disadvantages
- Does not account for false negatives.

#### Example
```python
from sklearn.metrics import precision_score

Y_test = [0, 1, 1, 0]
Y_pred = [0, 1, 0, 0]

prec = precision_score(Y_test, Y_pred)
print("Precision:", prec)  # Output: 1.0
```

### 3. Recall

#### Explanation
Recall measures the proportion of true positive predictions out of all actual positives. It indicates how many of the actual positives were correctly identified.

#### Formula
\[
Recall = \frac{TP}{TP + FN}
\]

#### Advantages
- Useful when the cost of false negatives is high (e.g., disease detection).

#### Disadvantages
- Does not account for false positives.

#### Example
```python
from sklearn.metrics import recall_score

Y_test = [0, 1, 1, 0]
Y_pred = [0, 1, 0, 0]

rec = recall_score(Y_test, Y_pred)
print("Recall:", rec)  # Output: 0.5
```

### 4. F1 Score

#### Explanation
F1 Score is the harmonic mean of Precision and Recall. It provides a balance between the two metrics.

#### Formula
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

#### Advantages
- Balances precision and recall.
- Useful for imbalanced datasets.

#### Disadvantages
- Can be harder to interpret than accuracy.

#### Example
```python
from sklearn.metrics import f1_score

Y_test = [0, 1, 1, 0]
Y_pred = [0, 1, 0, 0]

f1 = f1_score(Y_test, Y_pred)
print("F1 Score:", f1)  # Output: 0.6666666666666666
```

### 5. ROC-AUC

#### Explanation
ROC-AUC measures the area under the Receiver Operating Characteristic curve. It evaluates the model's ability to distinguish between classes across different thresholds.

#### Formula
The ROC curve plots True Positive Rate (TPR) vs. False Positive Rate (FPR). AUC is the area under this curve.

\[
TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}
\]

#### Advantages
- Threshold-independent.
- Good for imbalanced datasets.

#### Disadvantages
- Requires probability predictions.
- Can be computationally intensive.

#### Example
```python
from sklearn.metrics import roc_auc_score

Y_test = [0, 1, 1, 0]
Y_pred_prob = [0.1, 0.9, 0.4, 0.2]  # probabilities for positive class

roc_auc = roc_auc_score(Y_test, Y_pred_prob)
print("ROC-AUC:", roc_auc)  # Output: 0.75
```

## How to Use

1. Open the notebook `how_to_use.ipynb` in VS Code or Jupyter.
2. Run the cells to see calculations with sample data.
3. Modify `Y_test`, `Y_pred`, and `Y_pred_proba` to experiment with your own data.
4. Compare metrics to select the best model for your classification task.

## Notes

- These metrics are suitable for classification; not for regression.
- For multi-class problems, consider averaging methods (macro, micro, weighted).
- In practice, combine multiple metrics for comprehensive evaluation.
- Refer to scikit-learn documentation: [Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

If you have questions, check the notebook or reach out!</content>
<parameter name="filePath">d:\Dev\Python\AI\project\learning_AI\Evaluation_Metrics\Classification\README.md