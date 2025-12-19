# Decision Tree Algorithm Demo

This demo showcases the implementation of a Decision Tree classifier using scikit-learn on a custom mouse dataset. The dataset simulates mouse characteristics (front and back leg lengths) to classify mice into two groups (normal or mutant).

## Introduction to Decision Trees

Decision Trees are supervised machine learning algorithms used for both classification and regression tasks. They work by recursively splitting the dataset into subsets based on the most significant feature at each node, creating a tree-like structure where each leaf node represents a class label or a continuous value.

### How Decision Trees Work
- **Root Node**: The entire dataset.
- **Splitting**: At each internal node, the algorithm selects the best feature and threshold to split the data, aiming to maximize information gain or minimize impurity (e.g., using Gini impurity or entropy).
- **Leaf Nodes**: Terminal nodes that provide the final prediction.

### Advantages
- Easy to interpret and visualize.
- Can handle both numerical and categorical data.
- No need for feature scaling or normalization.
- Can capture non-linear relationships.

### Disadvantages
- Prone to overfitting, especially with deep trees.
- Sensitive to small changes in the training data.
- May not perform well on imbalanced datasets without adjustments.

### Applications
- Classification tasks (e.g., spam detection, medical diagnosis).
- Regression tasks (e.g., predicting house prices).
- Feature selection and understanding data patterns.

In this demo, we use a Decision Tree to classify mice based on leg lengths, demonstrating the algorithm's interpretability through visualization.

## Dataset

The dataset consists of 22 samples with 3 features each:
- Front leg length (integer from 1 to 4)
- Back leg length (integer from 1 to 6)
- Label: 0 (normal mouse) or 1 (mutant mouse)

Example data points:
- `[1,1,0]`: Front leg=1, Back leg=1, Label=0
- `[1,3,1]`: Front leg=1, Back leg=3, Label=1

## Features

- **Data Visualization**: Scatter plot showing the two groups (Group A: blue, Group B: red).
- **Decision Tree Training**: Uses Gini impurity criterion to build the tree.
- **Tree Visualization**: Plots the decision tree structure with feature names and class labels.

## Dependencies

- numpy
- matplotlib
- scikit-learn

Install with:
```
pip install numpy matplotlib scikit-learn
```

## How to Run

1. Ensure dependencies are installed.
2. Open `demo.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells sequentially.
4. The output will display:
   - Scatter plot of the dataset.
   - Decision tree visualization.

## Code Explanation

- **Data Preparation**: Load dataset as NumPy array, split into features (X) and labels (Y).
- **Grouping**: Use boolean indexing to separate groups for visualization.
- **Model Training**: Fit DecisionTreeClassifier on the data.
- **Visualization**: Use matplotlib for scatter plot and sklearn's plot_tree for tree diagram.

## Expected Output

- Scatter plot with blue/red points representing the two classes.
- Decision tree diagram showing splits based on leg lengths.

This demo illustrates basic decision tree concepts and visualization in machine learning.