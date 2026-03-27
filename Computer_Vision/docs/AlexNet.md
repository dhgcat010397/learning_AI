## AlexNet

AlexNet là một ví dụ kinh điển vì nó là mạng thần kinh sâu đầu tiên tạo nên cuộc cách mạng trong Computer Vision. Lớp tích chập đầu tiên (Conv1) của AlexNet rất đặc biệt vì nó sử dụng các bộ lọc lớn và bước nhảy dài.

Tóm tắt toàn bộ "dòng chảy" của AlexNet:
1. **Input:** Ảnh $227 \times 227 \times 3$.
2. **Conv/Pool:** Nén và trích xuất đặc trưng thành khối $6 \times 6 \times 256$.
3. **Flatten:** Trải phẳng thành hàng dọc $9,216$ phần tử.
4. **FC6/FC7:** Suy luận logic qua các tầng trung gian (mỗi tầng $4,096$ neurons).
5. **FC8:** Đưa ra dự đoán cuối cùng cho $1,000$ loại đối tượng.

---

Dưới đây là các thông số thực tế của lớp đầu tiên trong mạng AlexNet:

#### 1. Các thông số đầu vào (Parameters)

- **$W$ (Input size):** $227 \times 227$ (Ảnh màu RGB).
- **$K$ (Kernel size):** $11 \times 11$.
- **$S$ (Stride):** $4$.
- **$P$ (Padding):** $0$ (AlexNet nguyên bản không sử dụng padding ở lớp đầu tiên).

#### 2. Áp dụng công thức tính toán

Chúng ta áp dụng công thức:

$$O = \frac{W - K + 2P}{S} + 1$$

Thay số vào:

$$O = \frac{227 - 11 + 2(0)}{4} + 1 = \frac{216}{4} + 1 = 54 + 1 = 55$$

**Kết quả:** Kích thước đầu ra của lớp đầu tiên là $55 \times 55$.

#### 3. Phân tích kết quả

- **Sự sụt giảm kích thước:** Bạn có thể thấy kích thước ảnh giảm cực mạnh từ $227 \times 227$ xuống còn $55 \times 55$. Điều này giúp giảm thiểu khối lượng tính toán khổng lồ ngay từ bước đầu, vì dữ liệu thô ban đầu rất lớn.
- **Số lượng tham số (Kênh đầu ra):** Trong AlexNet, lớp này sử dụng 96 bộ lọc. Vì vậy, đầu ra đầy đủ không chỉ là $55 \times 55$, mà là một khối dữ liệu có kích thước $55 \times 55 \times 96$.
- **Tại sao dùng Stride = 4?** Nếu dùng Stride = 1, kích thước đầu ra sẽ là $217 \times 217$. Việc xử lý 96 kênh ở kích thước lớn như vậy sẽ làm quá tải bộ nhớ GPU thời bấy giờ (năm 2012).

#### 4. Một lưu ý nhỏ về "Sự cố 224 hay 227"

Trong nhiều tài liệu, bạn sẽ thấy người ta ghi ảnh đầu vào AlexNet là $224 \times 224$. Tuy nhiên, nếu bạn thay $224$ vào công thức:

$$\frac{224 - 11}{4} + 1 = \frac{213}{4} + 1 = 53.25 + 1 = 54.25$$

Con số này không chia hết! Vì vậy, các tác giả AlexNet thực tế đã cắt ảnh (crop) về kích cỡ $227 \times 227$ để phép tính ra số nguyên đẹp là $55$.

---

Sau khi qua lớp Convolution để trích xuất đặc trưng, dữ liệu thường vẫn còn rất lớn và chứa nhiều chi tiết thừa. Để giải quyết vấn đề này, AlexNet (và hầu hết các mạng CNN) sử dụng một bước gọi là **Max Pooling** (Lớp gộp cực đại).

#### 1. Max Pooling là gì?

Max Pooling là một kỹ thuật **lấy mẫu xuống (downsampling)**. Nó quét một bộ lọc qua ảnh (tương tự Convolution) nhưng thay vì nhân các trọng số, nó chỉ đơn giản là **chọn ra giá trị lớn nhất** trong vùng mà bộ lọc bao phủ.

**Tại sao lại lấy giá trị lớn nhất (Max)?**
- **Giữ lại đặc trưng mạnh nhất:** Trong Computer Vision, giá trị pixel cao thường đại diện cho một đặc trưng quan trọng (như một cạnh sắc nét hoặc một góc).
- **Bất biến với sự dịch chuyển (Translation Invariance):** Nếu vật thể trong ảnh bị lệch đi một vài pixel, giá trị Max trong vùng đó vẫn không đổi, giúp mô hình nhận diện vật thể tốt hơn dù vị trí của chúng thay đổi nhẹ.

#### 2. Công thức tính kích thước đầu ra sau Pooling

Tin vui là công thức tính kích thước sau khi Pooling hoàn toàn giống với Convolution:

$$O = \frac{W - K + 2P}{S} + 1$$

Tuy nhiên, có hai điểm khác biệt thực tế:
1. **Padding ($P$) trong Pooling:** Thường được mặc định là 0.
2. **Kernel ($K$) và Stride ($S$):** Trong Pooling, người ta thường thiết kế sao cho các vùng không chồng lấp hoặc chồng lấp rất ít.

#### 3. Ví dụ thực tế từ AlexNet (Lớp Pool1)

Ngay sau lớp Convolution 1 ($55 \times 55 \times 96$) mà chúng ta vừa tính ở trên, AlexNet áp dụng một lớp Max Pooling với các thông số:
- **$W$ (Input):** $55$.
- **$K$ (Kernel size):** $3 \times 3$.
- **$S$ (Stride):** $2$.
- **$P$ (Padding):** $0$.

**Tính toán:**

$$O = \frac{55 - 3 + 2(0)}{2} + 1 = \frac{52}{2} + 1 = 26 + 1 = 27$$

**Kết quả:** Sau lớp Pooling này, kích thước dữ liệu giảm từ $55 \times 55$ xuống còn $27 \times 27$. Lưu ý rằng số lượng kênh (96) vẫn giữ nguyên.

#### 4. Lợi ích của Max Pooling trong AlexNet

- **Giảm số lượng tham số:** Giúp giảm bớt khối lượng tính toán cho các lớp phía sau (giảm từ 55 xuống 27 là giảm hơn 4 lần số lượng pixel).
- **Tránh Overfitting:** Bằng cách nén dữ liệu và loại bỏ các chi tiết nhiễu nhỏ, mô hình sẽ tập trung vào các đặc điểm tổng quát hơn thay vì học thuộc lòng từng pixel.
- **Overlapping Pooling:** Đặc biệt trong AlexNet, các tác giả sử dụng $K=3$ và $S=2$ (vùng pooling chồng lên nhau). Họ nhận thấy rằng việc này giúp giảm lỗi top-1 và top-5 nhiều hơn so với việc dùng vùng không chồng lấp ($K=2, S=2$).

#### Tóm tắt sự thay đổi kích thước trong AlexNet (Giai đoạn đầu):

1. **Ảnh gốc:** $227 \times 227 \times 3$
2. **Sau Conv1 ($K=11, S=4$):** $55 \times 55 \times 96$
3. **Sau Pool1 ($K=3, S=2$):** $27 \times 27 \times 96$

---

Lớp **Fully Connected (FC)**, hay còn gọi là lớp kết nối đầy đủ, là giai đoạn "ra quyết định" của mạng AlexNet. Sau khi các lớp Convolution và Pooling đã làm tốt việc trích xuất các đặc trưng (như mắt, mũi, bánh xe, đường kẻ), lớp FC sẽ tập hợp tất cả các đặc trưng đó lại để phân loại xem ảnh đó là gì.

#### 1. Quá trình Trải phẳng (Flattening)

Trước khi vào lớp FC, dữ liệu đang ở dạng một khối 3D (ví dụ: $6 \times 6 \times 256$). Các lớp FC không thể xử lý dữ liệu dạng khối, vì vậy chúng ta phải thực hiện bước Flatten: biến khối 3D thành một vector 1D (một hàng dọc duy nhất)

**Ví dụ trong AlexNet:**
Ở cuối các lớp tích chập, dữ liệu có kích thước là $6 \times 6 \times 256$.
- Tổng số nút (neurons) sau khi trải phẳng sẽ là: $6 \times 6 \times 256 = 9,216$ nút.

#### 2. Cấu trúc của các lớp FC trong AlexNet

AlexNet có 3 lớp Fully Connected ở cuối cùng:
1. **FC6:** Nhận $9,216$ đầu vào và kết nối với 4,096 neurons.
2. **C7:** Tiếp tục kết nối 4,096 neurons từ lớp trước sang 4,096 neurons mới.
3. **FC8 (Output Layer):** Kết nối từ $4,096$ neurons sang 1,000 neurons.
   - *Tại sao là 1,000?* Vì bộ dữ liệu ImageNet mà AlexNet tham gia có 1,000 lớp đối tượng khác nhau (chó, mèo, máy bay, tàu hỏa...).

#### 3. Cơ chế hoạt động: "Mọi nút kết nối với mọi nút"

Đúng như tên gọi "Fully Connected", mỗi neuron ở lớp sau sẽ nhận tín hiệu từ tất cả các neurons ở lớp trước đó. Mỗi kết nối này đi kèm với một **trọng số (weight)**.
- Nếu một đặc trưng (ví dụ: "tai mèo") xuất hiện, các trọng số liên quan đến lớp "Mèo" sẽ được kích hoạt mạnh hơn.
- Cuối cùng, lớp FC8 sẽ dùng hàm **Softmax** để biến các con số thành xác suất (ví dụ: 90% là Mèo, 5% là Chó, 5% là Cú).

#### 4. Vấn đề của lớp FC: Khổng lồ và Dễ Overfit

Mặc dù rất mạnh mẽ, các lớp FC trong AlexNet lại là nơi chiếm nhiều bộ nhớ nhất:
- Chỉ riêng lớp FC6 đã chiếm khoảng: $9,216 \times 4,096 \approx 37.7$ triệu tham số!
- Do có quá nhiều tham số, các lớp này rất dễ xảy ra hiện tượng Overfitting (mạng học thuộc lòng ảnh cũ mà không nhận diện được ảnh mới).

**Giải pháp của AlexNet:** Các tác giả đã giới thiệu kỹ thuật Dropout. Trong quá trình huấn luyện, họ sẽ ngẫu nhiên "tắt" 50% các neuron trong lớp FC. Việc này buộc các neuron còn lại phải tự học cách làm việc độc lập và hiệu quả hơn, không dựa dẫm vào nhau.

---

Trong các mạng thần kinh trước AlexNet, người ta chủ yếu sử dụng các hàm kích hoạt như Sigmoid hoặc Tanh. Tuy nhiên, ReLU (Rectified Linear Unit) đã xuất hiện và thay đổi hoàn toàn cuộc chơi.

Các tác giả của AlexNet đã chứng minh rằng một mạng sử dụng ReLU có thể đạt được tỉ lệ lỗi 25% nhanh hơn 6 lần so với một mạng sử dụng hàm Tanh.

#### 1. ReLU là gì?

Hàm ReLU có công thức toán học cực kỳ đơn giản:

$$f(x) = \max(0, x)$$

- Nếu đầu vào $x$ là số dương, nó giữ nguyên giá trị đó.
- Nếu đầu vào $x$ là số âm hoặc bằng 0, nó trả về 0.

#### 2. Tại sao ReLU lại là "Vũ khí bí mật"?

Có 3 lý do chính khiến ReLU giúp AlexNet vượt trội:

**① Giải quyết vấn đề Triệt tiêu đạo hàm (Vanishing Gradient)**
Với các hàm như **Sigmoid**, khi giá trị đầu vào rất lớn hoặc rất nhỏ, đạo hàm (độ dốc) của hàm này sẽ tiến dần về 0.
   - Trong quá trình huấn luyện (Backpropagation), chúng ta dùng đạo hàm để cập nhật trọng số.
   - Nếu đạo hàm bằng 0, mạng sẽ ngừng học (bị "đơ").
   - **Với ReLU:** Đạo hàm luôn bằng 1 đối với mọi giá trị dương. Điều này giúp dòng thông tin truyền ngược về các lớp phía trước không bị mất đi, cho phép mạng học sâu hơn.

**② Tính toán cực nhanh**
- **Sigmoid/Tanh:** Phải tính toán các phép toán mũ ($e^x$), chia, và trừ phức tạp.
- **ReLU:** Chỉ là một phép so sánh với số 0 (if x > 0 return x else return 0). Trong lập trình máy tính, phép toán này tốn ít tài nguyên hơn hàng chục lần so với hàm mũ.

**③ Tạo ra tính "Thưa thớt" (Sparsity)**
Trong một mạng thần kinh, không phải neuron nào cũng cần thiết cho mọi bức ảnh.
- Vì ReLU biến tất cả các giá trị âm thành 0, nó tạo ra một ma trận "thưa" nơi chỉ có một số neuron quan trọng được kích hoạt.
- Điều này giúp mô hình hoạt động hiệu quả hơn và giảm nhiễu.

#### 3. So sánh trực quan

| Đặc điểm | Sigmoid/Tanh | ReLU |
| :--- | :--- | :--- |
| Công thức | Phức tạp (ex) | "Cực đơn giản (max(0,x))" |
| Tốc độ hội tụ | Chậm | Rất nhanh (Gấp 6 lần) |
| Đạo hàm | Bị triệt tiêu ở 2 đầu | Luôn bằng 1 ở phần dương |
| Ứng dụng | Hiện nay ít dùng cho lớp ẩn | Tiêu chuẩn cho mọi mạng CNN |

#### 4. Nhược điểm duy nhất: "Dying ReLU"

Mặc dù rất mạnh, ReLU có một điểm yếu là nếu một neuron bị rơi vào vùng giá trị âm và trả về 0 liên tục, nó có thể "chết" luôn và không bao giờ tham gia vào quá trình học nữa.

Để khắc phục điều này, các biến thể sau này như Leaky ReLU đã ra đời (thay vì bằng 0, nó sẽ cho phép một giá trị âm rất nhỏ như $0.01x$ đi qua).

---

#### Tổng kết về AlexNet

Đến đây, bạn đã thấy "bức tranh lớn" về AlexNet:
1. **Convolution:** Trích xuất đặc trưng.
2. **Max Pooling:** Nén dữ liệu.
3. **ReLU:** Tăng tốc độ học thần tốc.
4. **Dropout/FC:** Ra quyết định và chống học vẹt.