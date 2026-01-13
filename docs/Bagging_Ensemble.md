Các phương pháp Bagging dùng để giải quyết Overfitting

## Bagging Ensemble Learning Using Decision Tree

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

base_clf = DecisionTreeClassifier(radom_state=42)

bg_clg = BaggingClassifier(base_clf, n_estimators=100, random_state=42)

bg_clf.fit(X_train, y_train)

y_pred = bg_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {} ".format(accuracy))
```

Nếu bạn sử dụng giải thuật KNN cho bài toán Classification, bạn có thể sử dụng:

```python
from sklearn.neighbors import KNeighborsClassifier
```

`n_estimators`: số mẫu/mô hình bạn muốn tạo ra (thông thường là từ 100 đến 1000)

---

## Out of bag (OOB) Instances

- On average, for each model, 63% of the training instance are samples.
- The remaining 37% constitute the OOB instance.
  
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

base_clf = DecisionTreeClassifier(random_state=42)

bg_clf = BaggingClassifier(base_clf, n_estimators=100, oob_score=True, random_state=42)

bg_clf.fit(X_train, y_train)

oob_accuracy = bg_clf.oob_score_
print("Accuracy of OOB: {}".format(oob_accuracy))

y_pred = bg_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

rf_clf.fit(X_train, y_train)

oob_accuracy = rf_clf.oob_score_
print(f"Accuracy of OOB: {oob_accuracy:.2f}")

y_pred = rf_clf.predict(X_test)

accuracy = accuracy.(y_test, y_pred)
print(f"Accuracy on Full dataset: {accuracy:.2f}")
```

#### Feature Importance

```python
# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print("\nFeature Importances:")
print(importance_df)

# Plot feature importance
import matplotlib.pyplot as plt
importance_df.plot(kind="barh", x="Feature", y="Importance", color="lightgreen")
plt.show()
```

#### Hyperparameters

1. `n_estimators`: Number of decision trees in the Random Forest ensemble (100 - 1000) (default = 100).
2. `Bootstrap`: True (default = True).
3. `max_features`: (default = sqrt(n-features)).
   - Number of features to consider when looking for the best split.
   - A common choice is **sqrt** which uses the square root of the total number of features.
4. `oob_score`: If set to **True**, this hyperparameter enables the calculation of the Out-of-Bag (OOB) score (default = False).
5. `random_state`: Setting a specific seed ensures reproducibility in your results.

###### Hyperparameter of individual tree
1. `max_depth`: Maximum depth of each individual decision tree (10 - 20). Higher value more complexity (default = None).
   - You can set it to **None** to allow trees to expand until they contain fewer than **min_samples_split** samples.
2. `min_samples_split`: Min number of samples required to split an internal node (2 - 20). Higher value less complexity (default = 2).
3. `min_samples_leaf`: Minimum number of samples required to be in a leaf node (1 - 5) (default = 1).