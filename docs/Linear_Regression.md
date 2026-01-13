## Linear Regression

- **Least-Squares Linear Regression**
- **Ridge Regression and L2 Regularization**
- **Lasso Regression and L1 Regularization**

#### 1. Least-Squares Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test+split(X, y, test_size=0.2, random_state=42)

li_reg = LinearRegression()
li_reg.fit(X_train, y_train)

print(f"Linear model intercept b is: {li_reg.intercept_:.4f}")
print(f"Linear model coeff a is: {li_reg.coef_:.4f}")


# Đánh giá mô hình bằng r2_score
from sklearn.metrics import r2_score

y_pred = li_reg.predict(X_test)
print(f"R2 score of model: {r2_score(y_test, y_pred):.4f}")
```

#### 2. Ridge Regression and L2 Regularization

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaler_X_test = scaler.transform(X_test)

rid_reg = Ridge(alpha=1.0)
rid_reg.fit(scaled_X_train, y_train)

print(f"Ridge model intercept b is: {rid_reg.intercept_:.4f}")
print(f"Ridge model coeff a is: {rid_reg.coef_:.4f}")

from sklearn.metrics import r2_score
y_pred = rid_reg.predict(scaled_X_test)
print(f"R2 score of model: {r2_score(y_test, y_pred):.4f}")
```
