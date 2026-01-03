# 🌳 Decision Tree: Từ Cơ bản đến Nâng cao

Decision Tree (Cây quyết định) là một trong những thuật toán **Supervised Learning** phổ biến nhất trong Machine Learning. Nó được sử dụng cho cả **Classification (Phân loại)** và **Regression (Hồi quy)**.

---

## 📌 1. Decision Tree là gì?

Decision Tree là một cấu trúc phân cấp tương tự như sơ đồ luồng (flowchart):

- **Root Node (Nút gốc):** Đại diện cho toàn bộ tập dữ liệu.
- **Internal Node (Nút điều kiện):** Đại diện cho một thuộc tính (feature) và một câu hỏi quyết định.
- **Leaf Node (Nút lá):** Đại diện cho kết quả cuối cùng (nhãn lớp hoặc giá trị số).

---

## ⚖️ 2. Phân biệt Classification và Regression Tree

| Đặc điểm | Classification Tree | Regression Tree |
|----------|---------------------|-----------------|
| **Mục tiêu** | Dự đoán nhãn lớp (ví dụ: Spam/Not Spam) | Dự đoán giá trị liên tục (ví dụ: Giá nhà) |
| **Giá trị nút lá** | Nhãn lớp xuất hiện nhiều nhất (Mode) | Giá trị trung bình của các mẫu (Mean) |
| **Tiêu chí chia nút** | Gini Impurity hoặc Entropy (Information Gain) | Variance Reduction hoặc Mean Squared Error (MSE) |

---

## 🛠️ 3. Các thuật toán chia nút phổ biến

### 🔹 Cho Classification
- **Gini Impurity**  
  

\[
  Gini = 1 - \sum_{i=1}^{n} (P_i)^2
  \]

  
  → Đo mức độ "vẩn đục" của dữ liệu. Càng gần 0 thì dữ liệu càng thuần khiết.

- **Entropy (Information Gain)**  
  

\[
  Entropy = - \sum_{i=1}^{n} P_i \cdot \log_2(P_i)
  \]

  
  → Đo độ hỗn loạn thông tin.

### 🔹 Cho Regression
- **Mean Squared Error (MSE)**  
  

\[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

  
  → Chia sao cho tổng bình phương sai lệch giữa giá trị thực và giá trị trung bình tại các nút con là nhỏ nhất.

---

## 💻 4. Code minh họa (Python & Scikit-learn)

```python
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split

# --- 1. CLASSIFICATION (Phân loại hoa Iris) ---
iris = load_iris()
X_clf, y_clf = iris.data, iris.target

clf_tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
clf_tree.fit(X_clf, y_clf)

print("--- Cấu trúc cây phân loại ---")
print(export_text(clf_tree, feature_names=iris.feature_names))

# --- 2. REGRESSION (Dự báo giá nhà California) ---
housing = fetch_california_housing()
X_reg, y_reg = housing.data[:500], housing.target[:500]  # Lấy mẫu nhỏ để demo

reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_reg, y_reg)

print("\n--- Cấu trúc cây hồi quy ---")
print(export_text(reg_tree, feature_names=housing.feature_names))
```

---

## 🎯 5. Khi nào chọn chỉ số nào? (Cheat Sheet)

### Cho Classification
| Tình huống | Chỉ số ưu tiên |
|------------|----------------|
| Dữ liệu cân bằng | Accuracy |
| Muốn tránh báo động giả | Precision |
| Muốn tránh bỏ sót bệnh nhân/tội phạm | Recall |
| Dữ liệu mất cân bằng | F1-Score / AUC-ROC |

### Cho Regression
| Tình huống | Chỉ số ưu tiên | Lý do |
|------------|----------------|-------|
| Dữ liệu có nhiều Outliers | MAE | Không bị lỗi lớn làm sai lệch kết quả |
| Muốn phạt nặng các lỗi lớn | RMSE / MSE | Lỗi càng lớn thì "hình phạt" càng nặng |
| Báo cáo kinh doanh | MAPE | Dễ hiểu dưới dạng phần trăm sai số |
| Đánh giá độ khớp tổng quát | R-squared | Biết mô hình tốt hơn đoán mò bao nhiêu phần trăm |

---

## ⚠️ 6. Ưu và Nhược điểm

**Ưu điểm:**
- Dễ hiểu, dễ trực quan hóa.
- Không cần chuẩn hóa dữ liệu (scaling).
- Xử lý được cả dữ liệu số và phân loại.

**Nhược điểm:**
- Dễ bị **Overfitting** (quá khớp).
- Nhạy cảm với dữ liệu nhiễu.
- Cần giới hạn `max_depth` hoặc sử dụng **Random Forest** để khắc phục.

---

## 📊 7. Visualization (Tuỳ chọn)

Bạn có thể vẽ cây quyết định bằng thư viện **Graphviz** hoặc **Matplotlib** để trực quan hóa mô hình. Ví dụ với Graphviz:

```python
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(clf_tree, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")
