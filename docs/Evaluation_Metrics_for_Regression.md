# 📈 Evaluation Metrics for Regression

Tài liệu này tổng hợp các chỉ số quan trọng để đánh giá hiệu suất của các mô hình hồi quy (Regression). Khác với Classification, trong Regression chúng ta đo lường khoảng cách giữa giá trị dự đoán và giá trị thực tế.

---

## 📋 1. Các chỉ số đo lường lỗi (Error Metrics)

Giả sử:
- $y_i$: Giá trị thực tế.
- $\hat{y}_i$: Giá trị dự đoán.
- $n$: Tổng số mẫu dữ liệu.

### 🔹 MAE (Mean Absolute Error)
Trung bình cộng giá trị tuyệt đối của các sai số. MAE cho biết trung bình mô hình dự đoán sai lệch bao nhiêu đơn vị.
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Ưu điểm:** Dễ hiểu, cùng đơn vị với biến mục tiêu, không quá nhạy cảm với Outliers (điểm dữ liệu ngoại lai).

### 🔹 MSE (Mean Squared Error)
Trung bình bình phương các sai số.
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Ưu điểm:** Phù hợp để tính toán đạo hàm trong tối ưu hóa.
- **Nhược điểm:** Do có bình phương, MSE cực kỳ nhạy cảm với Outliers. Nếu có một lỗi lớn, MSE sẽ tăng vọt.

### 🔹 RMSE (Root Mean Squared Error)
Căn bậc hai của MSE. Đây là chỉ số được sử dụng phổ biến nhất.
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Ưu điểm:** Đưa sai số về cùng đơn vị với biến mục tiêu (thay vì đơn vị bình phương như MSE), giúp việc diễn giải dễ dàng hơn.

### 🔹 MAPE (Mean Absolute Percentage Error)
Trung bình phần trăm sai số tuyệt đối.
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
- **Ưu điểm:** Rất hữu ích trong báo cáo kinh doanh vì kết quả trả về dạng phần trăm (ví dụ: mô hình sai lệch 5%).

---

## 📊 2. Các chỉ số đo lường độ phù hợp (Goodness of Fit)

### 🔹 R-squared ($R^2$ - Hệ số xác định)
Cho biết tỷ lệ phần trăm sự biến thiên của biến mục tiêu được giải thích bởi mô hình.
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
Trong đó:
- $SS_{res}$ (Residual Sum of Squares): Tổng bình phương sai số dự đoán.
- $SS_{tot}$ (Total Sum of Squares): Tổng bình phương sai số so với giá trị trung bình.

- **Ý nghĩa:** $R^2$ càng gần 1, mô hình càng khớp tốt với dữ liệu. $R^2 = 0$ nghĩa là mô hình chỉ tương đương với việc lấy giá trị trung bình để dự đoán.

---

## 💻 3. Mã nguồn minh họa (Python)

Sử dụng thư viện `scikit-learn` để tính toán:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dữ liệu mẫu (Giá nhà thực tế và giá nhà dự đoán - đơn vị: tỷ VNĐ)
y_true = [2.5, 3.0, 4.8, 1.2, 5.5]
y_pred = [2.4, 3.2, 4.5, 1.5, 5.8]

# 2. Tính toán các chỉ số
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Tính MAPE thủ công hoặc dùng sklearn 0.24+
mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

print(f"--- Kết quả đánh giá Regression ---")
print(f"MAE:   {mae:.3f}")
print(f"MSE:   {mse:.3f}")
print(f"RMSE:  {rmse:.3f}")
print(f"R2:    {r2:.3f}")
print(f"MAPE:  {mape:.2f}%")
```

---

## 🎯 4. Khi nào chọn chỉ số nào? (Cheat Sheet)

| Tình huống bài toán | Chỉ số ưu tiên | Lý do |
| :--- | :---: | :--- |
| **Dữ liệu có nhiều Outliers** | **MAE** | Không bị các lỗi quá lớn làm sai lệch đánh giá tổng thể. |
| **Muốn triệt tiêu các lỗi lớn** | **RMSE** | Lỗi càng lớn thì "hình phạt" (penalty) càng nặng do phép bình phương. |
| **Báo cáo kinh doanh/quản lý** | **MAPE** | Dễ hiểu dưới dạng phần trăm sai số. |
| **Đánh giá độ khớp tổng quát** | **R-squared** | Biết được mô hình tốt hơn việc đoán mò (trung bình) bao nhiêu phần trăm. |