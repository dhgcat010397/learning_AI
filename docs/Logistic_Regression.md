## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(scaled_X_train, y_train)

# Make prediction on test data
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Predicted Probabilities
pre_proba = model.predict_proba(X_test)

# Decision Function Scores
decision_scores = model.decision_function(X_test)

# Model Coefficients
coefficents = model.coef_
intercept = model.intercept_
```