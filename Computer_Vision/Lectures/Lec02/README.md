## Phép tích chập (Convolution)

Phép tích chập (Convolution) là phép toán nền tảng nhất trong Xử lý ảnh và Trí tuệ nhân tạo (đặc biệt là Mạng thần kinh tích chập - CNN). Hiểu một cách đơn giản, đây là quá trình dùng một "bộ lọc" nhỏ quét qua bức ảnh để trích xuất các đặc trưng quan trọng.

#### 1. Cơ chế hoạt động của Phép tích chập

Để thực hiện phép tích chập, ta cần hai thành phần chính:
- **Input (Ảnh gốc):** Ma trận các điểm ảnh ($I$).
- **Kernel/Filter (Bộ lọc):** Một ma trận nhỏ (thường có kích thước $3 \times 3$ hoặc $5 \times 5$) chứa các trọng số ($K$).

**Quy trình thực hiện:**
  1. Đặt Kernel đè lên một vùng trên ảnh gốc.
  2. Nhân từng phần tử tương ứng của Kernel với giá trị điểm ảnh tại vùng đó.
  3. Cộng tất cả các kết quả lại để được một con số duy nhất. Con số này là giá trị cho điểm ảnh tương ứng trên ảnh mới (Feature Map).
  4. Dịch chuyển Kernel sang vị trí tiếp theo và lặp lại.

#### 2. Các tham số quan trọng trong Convolution

Khi thực hiện tích chập, có 3 khái niệm bạn cần nắm vững:
- **Stride (Bước nhảy):** Khoảng cách mà Kernel dịch chuyển mỗi lần. Nếu Stride = 1, nó dịch chuyển từng pixel; Stride = 2, nó nhảy cách 1 pixel (làm giảm kích thước ảnh đầu ra).
- **Padding (Đệm):** Thêm các vòng giá trị (thường là số 0) bao quanh mép ảnh gốc. Việc này giúp giữ nguyên kích thước ảnh sau khi tích chập và không bỏ lỡ thông tin ở các cạnh ảnh.
- **Channel (Kênh):** Nếu ảnh màu (RGB), tích chập sẽ được thực hiện trên cả 3 kênh màu rồi cộng dồn lại.

#### 3. Ứng dụng: Tại sao chúng ta cần Tích chập?

Tùy vào các con số bên trong Kernel, phép tích chập sẽ mang lại những hiệu ứng khác nhau cho bức ảnh:
- **Làm mờ (Blurring):** Dùng bộ lọc trung bình để làm mịn ảnh, giảm nhiễu.
- **Phát hiện cạnh (Edge Detection):** Sử dụng các bộ lọc như Sobel hoặc Prewitt. Nó tìm ra những nơi có sự thay đổi đột ngột về màu sắc (đó chính là đường viền).
- **Làm sắc nét (Sharpening):** Tăng cường sự khác biệt giữa các điểm ảnh cạnh nhau.

#### 4. Ví dụ Code với OpenCV

Dưới đây là cách bạn tự tạo một Kernel để làm sắc nét ảnh bằng hàm `cv2.filter2D()`:

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')

# Định nghĩa một Kernel 3x3 để làm sắc nét (Sharpen)
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

# Áp dụng phép tích chập
output = cv2.filter2D(img, -1, kernel)

cv2.imshow('Original', img)
cv2.imshow('Sharpened', output)
cv2.waitKey(0)
```

#### 5. Vai trò trong Deep Learning (CNN)

Trong lập trình truyền thống, con người tự thiết kế Kernel (như ví dụ trên). Nhưng trong Deep Learning, máy tính sẽ tự học để tìm ra các con số tối ưu trong Kernel.
- Các lớp tích chập đầu tiên thường học cách tìm cạnh, đường thẳng.
- Các lớp sâu hơn học cách tìm hình dạng phức tạp như mắt, mũi, bánh xe...

---

Để tính toán kích thước của ảnh (hoặc Feature Map) sau khi thực hiện phép tích chập, chúng ta sử dụng một công thức toán học cố định. Việc hiểu rõ công thức này giúp bạn thiết kế mạng thần kinh (CNN) mà không làm mất dữ liệu quan trọng hoặc làm kích thước ảnh bị lỗi.

#### 1. Các tham số trong công thức

Giả sử chúng ta có một ảnh đầu vào hình vuông (nếu ảnh hình chữ nhật, ta tính riêng cho chiều rộng và chiều cao):
- **$W$ (Input size):** Kích thước chiều rộng/cao của ảnh đầu vào.
- **$K$ (Kernel size):** Kích thước của bộ lọc (Filter).
- **$P$ (Padding):** Số lượng pixel đệm thêm vào xung quanh rìa ảnh.
- **$S$ (Stride):** Bước nhảy của bộ lọc khi quét qua ảnh.

#### 2. Công thức tính kích thước đầu ra ($O$)

Kích thước đầu ra được tính bằng công thức:

$$O = \frac{W - K + 2P}{S} + 1$$

**Lưu ý quan trọng:** Kết quả của phép tính này phải là một số nguyên. Nếu phép chia có dư, tùy vào thư viện (như PyTorch hay TensorFlow) mà kết quả sẽ được làm tròn xuống hoặc ảnh sẽ bị bỏ sót phần rìa.

#### 3. Ví dụ minh họa cụ thể

**Trường hợp 1: Tích chập cơ bản (Không Padding)**
- Ảnh đầu vào: $W = 5$
- Kernel: $K = 3$
- Padding: $P = 0$
- Stride: $S = 1$

$$O = \frac{5 - 3 + 2(0)}{1} + 1 = \frac{2}{1} + 1 = 3$$

👉 Ảnh từ **5x5** sẽ giảm xuống còn **3x3**.

**Trường hợp 2: Sử dụng Padding để giữ nguyên kích thước (Same Padding)**
Đây là kỹ thuật cực kỳ phổ biến để ảnh không bị nhỏ lại sau mỗi lớp.
- Ảnh đầu vào: $W = 5$
- Kernel: $K = 3$
- Stride: $S = 1$
- Để $O = 5$, ta cần tìm $P$:

$$5 = \frac{5 - 3 + 2P}{1} + 1 \Rightarrow 4 = 2 + 2P \Rightarrow P = 1$$

👉 Khi dùng Kernel **3x3** và **Padding 1**, kích thước ảnh sẽ được bảo toàn.

**Trường hợp 3: Sử dụng Stride lớn để giảm kích thước (Downsampling)**
- Ảnh đầu vào: $W = 224$ (kích thước chuẩn ImageNet)
- Kernel: $K = 7$
- Padding: $P = 3$
- Stride: $S = 2$

$$O = \frac{224 - 7 + 2(3)}{2} + 1 = \frac{223}{2} + 1 = 111.5 + 1 = 112.5$$

👉 Trong thực tế, kết quả sẽ được lấy phần nguyên là **112**.

#### 4. Tại sao công thức này lại quan trọng?

- **Kiểm soát thông tin:** Nếu bạn không dùng Padding ($P=0$), ảnh sẽ bị thu nhỏ dần qua mỗi lớp. Đến một lúc nào đó, kích thước sẽ trở thành $1 \times 1$ và bạn không thể thực hiện thêm phép tích chập nào nữa.
- **Thiết kế Network:** Khi xây dựng các mạng như ResNet hay VGG, kỹ sư phải tính toán sao cho sau một số lớp, kích thước ảnh giảm đi một nửa nhưng số lượng kênh (Channels) tăng lên để giữ cân bằng khối lượng tính toán.
- **Hiểu về Receptive Field:** Công thức này giúp xác định một điểm ảnh ở lớp đầu ra "nhìn thấy" bao nhiêu diện tích ở lớp đầu vào ban đầu

Mẹo nhỏ cho bạn:
- Nếu muốn giữ nguyên kích thước ảnh với Stride = 1, hãy chọn: $P = \frac{K - 1}{2}$. (Ví dụ: Kernel 3 thì Pad 1, Kernel 5 thì Pad 2).
- Nếu muốn giảm kích thước ảnh đi một nửa, người ta thường chọn Stride = 2.