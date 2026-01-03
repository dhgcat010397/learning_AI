# 🧪 Cross-validation (Kiểm định chéo) trong Machine Learning

## 1. Cross-validation là gì?
**Cross-validation (CV)** là một kỹ thuật thống kê được sử dụng để đánh giá hiệu suất của các mô hình học máy (Machine Learning). Thay vì chỉ chia dữ liệu thành hai phần (Huấn luyện và Kiểm thử) một cách đơn giản, CV chia dữ liệu thành nhiều phần nhỏ để đảm bảo mô hình được đánh giá trên toàn bộ dữ liệu hiện có.

Mục tiêu chính của Cross-validation là dự đoán khả năng **tổng quát hóa** (Generalization) của mô hình đối với các dữ liệu mới mà nó chưa từng thấy trước đây.

---

## 2. Tại sao chúng ta cần Cross-validation?

Để hiểu tại sao cần CV, chúng ta cần xem xét các thành phần của **Sai số tổng quát hóa (Generalization Error)**:

### 🔴 Độ chệch (Bias)
* Là thước đo mức độ sai lệch trung bình giữa các dự đoán của mô hình so với giá trị thực tế.
* Phát sinh khi mô hình quá đơn giản (insufficient complexity), không bắt bài được các quy luật ẩn của dữ liệu.
* **Hậu quả:** Dẫn đến tình trạng **Học chưa tới (Underfitting)**, dự đoán sai trên cả tập huấn luyện và tập kiểm thử.

### 🔵 Độ biến động (Variance)
* Đại diện cho sự nhạy cảm của mô hình đối với nhiễu trong dữ liệu huấn luyện.
* Đo lường sự thay đổi của dự đoán khi huấn luyện trên các tập dữ liệu con khác nhau.
* **Hậu quả:** Các mô hình có độ biến động cao thường quá phức tạp, học cả nhiễu, dẫn đến **Quá khớp (Overfitting)**. Mô hình hoạt động cực tốt trên tập huấn luyện nhưng rất kém trên dữ liệu kiểm định và kiểm thử.

### 🟢 Sai số không thể giảm thiểu (Irreducible Error)
* Là giới hạn cố hữu không thể tránh khỏi của mọi mô hình.



**Cross-validation giúp chúng ta tìm ra điểm cân bằng (Trade-off) giữa Bias và Variance để mô hình đạt hiệu suất tối ưu nhất.**

---

## 3. Kỹ thuật K-Fold Cross-validation (Phổ biến nhất)

Đây là phương pháp tiêu chuẩn trong công nghiệp:
1. Chia tập dữ liệu thành **K** phần bằng nhau (gọi là "folds").
2. Lặp lại quá trình huấn luyện **K** lần.
3. Trong mỗi lần lặp, chọn 1 phần làm tập kiểm định (Validation set) và $K-1$ phần còn lại làm tập huấn luyện (Training set).
4. Tính trung bình kết quả của $K$ lần thực hiện để có điểm số cuối cùng.



---

## 4. Ứng dụng: Tối ưu hóa Siêu tham số (Hyperparameter Tuning)

Cross-validation thường được kết hợp với các phương pháp tìm kiếm để chọn ra bộ cài đặt tốt nhất cho mô hình:

### ⏹️ Grid Search (Tìm kiếm theo lưới)
* **Ưu điểm:** Đơn giản, dễ triển khai và đảm bảo khám phá toàn bộ không gian tìm kiếm đã thiết lập.
* **Nhược điểm:** Rất tốn tài nguyên tính toán (computationally expensive), đặc biệt khi có nhiều siêu tham số và phạm vi tìm kiếm rộng.

### 🎲 Random Search (Tìm kiếm ngẫu nhiên)
* **Ưu điểm:** Hiệu quả hơn về mặt tính toán so với Grid Search. Có thể tìm thấy siêu tham số tốt với ít lượt đánh giá hơn.
* **Nhược điểm:** Không đảm bảo 100% sẽ tìm thấy bộ siêu tham số tối ưu toàn cục (globally optimal).

---

## 5. Ví dụ mã nguồn nhanh với Scikit-learn

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Khởi tạo mô hình
model = RandomForestRegressor()

# Thực hiện K-Fold CV với K=5
# Sử dụng 'neg_mean_absolute_error' để phù hợp với cơ chế maximize của sklearn
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

# Chuyển về MAE dương
mae_scores = -scores
print(f"MAE trung bình qua 5-folds: {mae_scores.mean()}")