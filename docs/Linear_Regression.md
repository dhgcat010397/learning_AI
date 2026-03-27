## Linear Regression

- **Least-Squares Linear Regression**
- **Ridge Regression and L2 Regularization**
- **Lasso Regression and L1 Regularization**

#### Application
- Least-square linear regression: reasonable dataset, linear relation between features and target.
- Ridge regression: correlated features and overfitting.
- Lasso regression: when you believe that some features should be excluded. We want to simplify the model with large number features.

---

#### 1. Hàm hồi quy tuyến tính đơn giản

Trong mô hình đơn giản nhất, ta có phương trình:

$$y = \beta_0 + \beta_1x + \epsilon$$

Trong đó:
- $y$: Biến phụ thuộc (Dependent variable).
- $x$: Biến độc lập (Independent variable).
- $\beta_0$: Hệ số chặn (Intercept).
- $\beta_1$: Hệ số góc (Slope).
- $\epsilon$: Sai số (Error term).

---

#### 2. Hàm mất mát (Loss Function)

Mục tiêu của phương pháp này là tối thiểu hóa tổng bình phương các sai số dư ($SSE$ - Sum of Squared Errors):

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1x_i))^2$$

---

#### 3. Công thức tính các hệ số ($\beta$)

Các giá trị tối ưu của $\beta_0$ và $\beta_1$ được tính bằng:

Hệ số góc ($\beta_1$):

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Hệ số chặn ($\beta_0$):

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

(Với $\bar{x}$ và $\bar{y}$ là giá trị trung bình của $x$ và $y$)

---

#### 4. Dạng ma trận (Cho hồi quy đa biến)

Đối với mô hình có nhiều biến độc lập, công thức được viết dưới dạng ma trận:

$$\mathbf{y} = \mathbf{X}\beta + \epsilon$$

Nghiệm của bài toán bình phương tối thiểu (Ordinary Least Squares - OLS) là:

$$\hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

---

#### 5. Least-Squares Linear Regression

Công thức tổng quát để tính tổng bình phương các sai số dư là:

$$RSS(a, b) = \sum_{i=1}^{n} (y_i - (ax_i + b))^2$$

- $RSS(a, b)$: Tổng bình phương các sai số dư (Residual Sum of Squares), đây là hàm mục tiêu cần tối thiểu hóa trong bài toán hồi quy.
- $\sum_{i=1}^{n}$: Ký hiệu tổng sigma, thực hiện phép cộng từ quan sát thứ $1$ đến quan sát thứ $n$.
- $y_i$: Giá trị thực tế của biến phụ thuộc tại điểm dữ liệu thứ $i$.
- $ax_i + b$: Giá trị dự báo ($\hat{y}_i$) dựa trên mô hình tuyến tính với hệ số góc $a$ và hệ số chặn $b$.
- $(y_i - (ax_i + b))$: Sai số (residual) giữa giá trị thực tế và giá trị dự báo.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

li_reg = LinearRegression()
li_reg.fit(X_train, y_train)

print(f"Linear model intercept b is: {li_reg.intercept_:.4f}")
print(f"Linear model coeff a is: {li_reg.coef_:.4f}")


# Đánh giá mô hình bằng r2_score
from sklearn.metrics import r2_score
y_pred = li_reg.predict(X_test)
print(f"R2 score of model: {r2_score(y_test, y_pred):.4f}")
```

---

#### 6. Ridge Regression and L2 Regularization

Ridge Regression thêm một lượng phạt (regularization) tương đương với bình phương độ lớn của hệ số $a$.

$$J_{Ridge}(a, b) = \sum_{i=1}^{n} (Y_i - (aX_i + b))^2 + \lambda a^2$$

Hoặc viết gọn bằng RSS:

$$J_{Ridge}(a, b) = RSS(a, b) + \lambda a^2$$

- $\lambda$ (Lambda): Tham số điều tiết (Regularization parameter). Nếu $\lambda$ càng lớn, hệ số $a$ sẽ càng bị ép về gần 0 (nhưng không bao giờ bằng 0).
- $a^2$: L2 Regularization.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

rid_reg = Ridge(alpha=1.0)
rid_reg.fit(scaled_X_train, y_train)

print(f"Ridge model intercept b is: {rid_reg.intercept_:.4f}")
print(f"Ridge model coeff a is: {rid_reg.coef_:.4f}")

from sklearn.metrics import r2_score
y_pred = rid_reg.predict(scaled_X_test)
print(f"R2 score of model: {r2_score(y_test, y_pred):.4f}")
```

---

#### 7. Lasso Regression and L1 Regularization

Lasso Regression thêm một lượng phạt (regularization) tương đương với giá trị tuyệt đối của hệ số $a$.

$$J_{Lasso}(a, b) = \sum_{i=1}^{n} (Y_i - (aX_i + b))^2 + \lambda |a|$$

Hoặc viết gọn bằng RSS:

$$J_{Lasso}(a, b) = RSS(a, b) + \lambda |a|$$

- $|a|$: L1 Regularization.
- Đặc điểm: Lasso có khả năng ép hệ số $a$ về đúng bằng 0, do đó nó có thể được dùng để lựa chọn đặc trưng (Feature Selection).

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

las_reg = Lasso(alpha=1.0)
las_reg.fit(scaled_X_train, y_train)

print(f"Lasso model intercept b is: {las_reg.intercept_:.4f}")
print(f"Lasso model coeff a is: {las_reg.coef_:.4f}")

from sklearn.metrics import r2_score
y_pred = las_reg.predict(scaled_X_test)
print(f"R2 score of model: {r2_score(y_test, y_pred):.4f}")
```