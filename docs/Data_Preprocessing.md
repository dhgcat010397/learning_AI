# Data Preprocessing
- **Data Cleaning**
- **Data Transformation**
- **Feature Selection**
- **Data Visualization**

Đồng thời, tùy thuộc vào một số yêu cầu cụ thể, ta có thể thực hiện thêm một số quá trình:
- Data Balancing
- Handling Skewed Target Variable
- Normalization of Target Variable

---

## 1. Data Cleaning:

- Handling missing data

Loại bỏ dữ liệu bị khiếm khuyết ra khỏi dataset bằng lệnh:

```python
df.dropna(inplace=True)
```

Hoặc điền nó bằng 1 giá trị nào đó, thông thường người ta hay lấy giá trị trung bình của 1 cột dữ liệu nào đó để điền vào:

```python
df['columns_name'].fillna(df['column_name'].mean(), inplace=True)
```

- Handling Outliers

```python
import numpy as np
from scipy import stats

z_scores = np.abs(stats.zscore(df['column_name']))
df = df[(z_scores < 3>)]
```

---

## 2. Data Transformation

#### 2.1. Feature Scaling:

Normalize or standardize feature.

- Normalization: min-max scaling

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
scaler_X_train = scaler.transform(X_train)
scaler_X_test = scaler.transform(X_test)
```

- Standardization: Z-score normalization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
scaler_X_train = scaler.transform(X_train)
scaler_X_test = scaler.transform(X_test)
```

#### 2.2. Encoding Categorical Data:

Convert categorical data into numbers.

Ta có 2 phương pháp:
- One-hot encoding:

```python
import pandas as pd

df_encoded = pd.get_dummies(df, columns=['Geography'])
```

Ví dụ:
Bảng dữ liệu ban đầu:

| | Geography | Gender |
| :--- | :--- | :--- |
| **0** | France | 0 |
| **1** | Spain | 0 |
| **2** | France | 0 |
| **3** | France | 0 |
| **4** | Spain | 0 |
| **5** | Spain | 1 |

Bảng dữ liệu sau khi **One-hot encoding**:

| | Gender | Geography_France | Geography_Spain |
| :--- | :---: | :---: | :---: |
| **0** | 0 | 1 | 0 |
| **1** | 0 | 0 | 1 |
| **2** | 0 | 1 | 0 |
| **3** | 0 | 1 | 0 |
| **4** | 0 | 0 | 1 |
| **5** | 1 | 0 | 1 |

- Label encoding:
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
```

Ví dụ:
Bảng dữ liệu ban đầu:

| | Geography | Gender |
| :--- | :--- | :--- |
| **0** | France | Female |
| **1** | Spain | Female |
| **2** | France | Female |
| **3** | France | Female |
| **4** | Spain | Female |
| **5** | Spain | Male |

Bảng dữ liệu sau khi **Label encoding**:

| | Geography | Gender |
| :--- | :--- | :--- |
| **0** | France | 0 |
| **1** | Spain | 0 |
| **2** | France | 0 |
| **3** | France | 0 |
| **4** | Spain | 0 |
| **5** | Spain | 1 |

#### 2.3. Feature Engineering:

- Create new feature.
- Transform existing features.
  
#### 2.4. Handling Time Series Data

Xử lý các dữ liệu phụ thuộc vào thời gian

#### 2.5. Text Data Processing

Xử lý các dữ liệu dạng text
