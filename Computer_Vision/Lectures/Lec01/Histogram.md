## Histogram

Histogram (biểu đồ tần suất) là 1 biểu đồ biểu thị sự phân phối các mức độ sáng (intensity) của các điểm ảnh (pixels) trong một bức ảnh.

Nói 1 cách đơn giản: Histogram cho bạn biết bức ảnh đó có bao nhiêu điểm ảnh tối, bao nhiêu điểm ảnh sáng và bao nhiêu điểm ảnh có mức xám trung bình.

---

#### 1. Cấu trúc của một Histogram

Một Histogram chuẩn cho ảnh xám (grayscale) thường có:
- Trục hoành (Trục X): Đại diện cho các giá trị mức xám, chạy từ 0 đến 255 (với 0 là đen hoàn toàn và 255 là trắng hoàn toàn).
- Trục tung (Trục Y): Đại diện cho số lượng điểm ảnh ứng với mỗi giá trị mức xám đó.

---

#### 2. Các ứng dụng quan trọng của Histogram

Histogram là một công cụ cực kỳ mạnh mẽ vì nó giúp máy tính "hiểu" được đặc điểm ánh sáng của ảnh mà không cần nhìn vào từng pixel:
- **Phân tích độ phơi sáng (Exposure):**
  - Nếu Histogram lệch hẳn về bên trái: Ảnh quá tối (Under-exposed).
  - Nếu Histogram lệch hẳn về bên phải: Ảnh quá sáng (Over-exposed).
  - Nếu Histogram tập trung ở giữa: Ảnh có độ tương phản thấp.
- **Cân bằng Histogram (Histogram Equalization):** Đây là kỹ thuật trải đều các giá trị pixel trên toàn bộ dải từ 0-255 để làm tăng độ tương phản của ảnh, giúp các chi tiết ẩn trong vùng quá tối hoặc quá sáng hiện rõ hơn.
- **Tách ngưỡng (Thresholding):** Dựa vào Histogram, ta có thể chọn một điểm "ngắt" để tách vật thể ra khỏi nền (ví dụ: biến ảnh thành đen trắng hoàn toàn).
- **Nhận dạng vật thể (Object Recognition):** Mỗi vật thể hoặc cảnh vật thường có một "dấu vân tay" Histogram riêng biệt. Việc so sánh Histogram giữa hai ảnh có thể giúp máy tính xác định xem chúng có cùng nội dung hay không.

---

#### 3. Cách tính Histogram bằng OpenCV (Python)

Để vẽ Histogram trong Python, chúng ta thường dùng hàm `cv2.calcHist()`.

```python
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv2.imread('image.jpg', 0) # 0 là đọc ảnh xám

# Tính toán histogram
# [img]: ảnh nguồn, [0]: kênh màu, None: không dùng mask
# [256]: số lượng bin (mức giá trị), [0, 256]: dải giá trị
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Vẽ biểu đồ
plt.plot(hist)
plt.title('Image Histogram')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Number of Pixels')
plt.show()
```

---

#### 4. Histogram màu (Color Histogram)

Đối với ảnh màu (RGB), Histogram sẽ được tính riêng cho từng kênh: Đỏ (Red), Xanh lá (Green), và Xanh dương (Blue). Sự kết hợp của 3 biểu đồ này sẽ cho biết đặc điểm màu sắc tổng thể của bức ảnh.