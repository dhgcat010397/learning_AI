# 📊 Evaluation Metrics for Classification

Tài liệu này tổng hợp các chỉ số đánh giá quan trọng trong bài toán phân loại (Classification) của Machine Learning. Việc hiểu rõ các chỉ số này giúp bạn đánh giá mô hình một cách khách quan, đặc biệt là khi làm việc với dữ liệu mất cân bằng (Imbalanced Data).

---

## 📋 1. Confusion Matrix (Ma trận nhầm lẫn)

Ma trận nhầm lẫn là một bảng tóm tắt kết quả dự đoán của mô hình so với thực tế.



* **TP (True Positive):** Dự đoán là Dương tính (1) và thực tế là Dương tính (1).
* **TN (True Negative):** Dự đoán là Âm tính (0) và thực tế là Âm tính (0).
* **FP (False Positive):** Dự đoán là Dương tính (1) nhưng thực tế là Âm tính (0). (Sai lầm loại I)
* **FN (False Negative):** Dự đoán là Âm tính (0) nhưng thực tế là Dương tính (1). (Sai lầm loại II)

---

## 📏 2. Các chỉ số đo lường chi tiết

### 🔹 Accuracy (Độ chính xác tổng quát)
Tỉ lệ dự đoán đúng (cả Positive và Negative) trên tổng số mẫu.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
> **Lưu ý:** Chỉ số này thường không đáng tin cậy nếu dữ liệu bị lệch (ví dụ: 95% dữ liệu là lớp A, mô hình chỉ cần đoán bừa là A cũng đạt Accuracy 95%).

### 🔹 Precision (Độ chính xác - Dương tính)
Trong những mẫu mô hình **dự đoán là Positive**, có bao nhiêu mẫu thực sự là Positive?
$$\text{Precision} = \frac{TP}{TP + FP}$$
*Ưu tiên khi cần giảm thiểu số ca bị "oan sai" (ví dụ: bộ lọc thư rác).*

### 🔹 Recall / Sensitivity (Độ nhạy)
Trong những mẫu **thực tế là Positive**, mô hình đã "bắt" được bao nhiêu mẫu?
$$\text{Recall} = \frac{TP}{TP + FN}$$
*Ưu tiên khi cần giảm thiểu số ca bị bỏ sót (ví dụ: xét nghiệm bệnh hiểm nghèo).*

### 🔹 F1-Score
Giá trị trung bình điều hòa giữa Precision và Recall. Nó giúp cân bằng hai chỉ số này khi chúng có sự chênh lệch lớn.
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 📈 3. Chỉ số nâng cao: ROC & AUC



* **ROC Curve (Receiver Operating Characteristic):** Đường cong biểu diễn mối tương quan giữa **TPR** (Recall) và **FPR** (Tỉ lệ dương tính giả) tại các ngưỡng (threshold) khác nhau.
* **AUC (Area Under Curve):** Diện tích dưới đường cong ROC. Giá trị AUC nằm từ 0.5 (ngẫu nhiên) đến 1.0 (hoàn hảo). Mô hình có AUC càng cao thì khả năng phân biệt giữa các lớp càng tốt.

---

## 💻 4. Mã nguồn minh họa (Python)

Sử dụng thư viện `scikit-learn` để tính toán nhanh các chỉ số:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Dữ liệu mẫu: y_true là thực tế, y_pred là dự đoán từ mô hình
y_true = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]

# Tính toán các chỉ số
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")

# Hiển thị báo cáo chi tiết
print("\nBáo cáo phân loại:")
print(classification_report(y_true, y_pred))

# Vẽ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Confusion Matrix')
plt.show()
```

---

### 🎯 5. Khi nào chọn chỉ số nào? (Cheat Sheet)

| Tình huống | Chỉ số ưu tiên |
| :--- | :--- |
| Dữ liệu cân bằng, các lớp quan trọng như nhau | **Accuracy** |
| Muốn tránh báo động giả (không muốn làm phiền người dùng) | **Precision** |
| Muốn tránh bỏ sót (không muốn để lọt bệnh nhân/tội phạm) | **Recall** |
| Dữ liệu mất cân bằng nghiêm trọng | **F1-Score / AUC-ROC** |