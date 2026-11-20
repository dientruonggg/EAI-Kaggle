# THỂ LỆ & QUY CHUẨN KỸ THUẬT VÒNG BẢNG & CHUNG KẾT
## Cuộc thi Edge AI Challenge 2026 Lần 2 (CLB Pioneer - ĐH Bách Khoa ĐN)
*Nguồn trích xuất: Thể Lệ Cuộc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf (Ngày ban hành: 02/11/2026)*

---

## 📅 1. LỊCH TRÌNH THI ĐẤU DỰ KIẾN

| Giai đoạn | Mốc thời gian | Nội dung |
| :--- | :--- | :--- |
| **Đăng ký** | 24/10/2026 - 29/10/2026 | Mở đơn đăng ký tham gia |
| **Warm-up** | 01/11/2026 - 07/11/2026 | Mở vòng Warm-up, kiểm thử hệ thống thi đấu |
| **Kickoff** | 08/11/2026 | Kickoff & Buổi đào tạo kỹ thuật (Training 1) |
| **Vòng loại (Qualifiers)** | **09/11/2026 - 22/11/2026** | **Thi đấu vòng bảng chính thức trên Kaggle** |
| **Công bố Chung kết** | 23/11/2026 | Công bố các đội vào Chung kết + Bàn giao Kit phần cứng |
| **Chung kết (Finals)** | **05/12/2026** | **Thi đấu trực tiếp trên ESP32-S3 + Camera OV7670** |

---

## 🏷️ 2. BỘ NHÃN & BÀI TOÁN PHÁT HIỆN VẬT THỂ (OBJECT DETECTION)

Dữ liệu hình ảnh được chuẩn hóa theo bối cảnh đồ vật thực tế, quan sát trực tiếp bằng camera OV7670. Áp dụng thống nhất 7 lớp vật thể:

| Class ID | Tên Class (`classes.txt`) | Tên Tiếng Việt | Ghi chú & Đặc điểm hình ảnh |
| :---: | :--- | :--- | :--- |
| **0** | `bottle` | Chai | Gồm nhiều kiểu chai, kích thước và màu sắc |
| **1** | `cup` | Cốc / Ly | Bao gồm các mẫu cốc khác nhau |
| **2** | `phone` | Điện thoại | Có thể nằm ngang hoặc dựng đứng |
| **3** | `book` | Sách | Nhiều màu bìa, độ dày và kích thước |
| **4** | `mouse` | Chuột máy tính | Chuột có dây hoặc không dây |
| **5** | `box` | Hộp | Hộp có hình dạng gần hộp chữ nhật / lập phương |
| **6** | `toy_car` | Ô tô đồ chơi | Nhiều mẫu, màu sắc và hướng đặt |

### Quy cách dữ liệu hình ảnh:
- **Kích thước ảnh gốc:** `320 × 240 pixel` (chuẩn OV7670).
- **Kích thước đầu vào mô hình:** Khuyến nghị `96 × 96`, `128 × 128` hoặc `160 × 160 pixel` (không vượt quá `320 × 240`).
- **Định dạng nhãn:** YOLO format (`class_id x_center y_center width height` chuẩn hóa 0–1).
- **Giới hạn số box:** Mỗi ảnh test được phép có tối đa 100 bounding box dự đoán.

---

## 💬 3. BÀI TOÁN PHÂN LOẠI Ý ĐỊNH BẰNG TEXTCNN (INTENT CLASSIFICATION)

Mỗi câu hỏi đầu vào là chuỗi văn bản tiếng Việt. Mô hình TextCNN phân loại vào 1 trong 5 nhóm ý định:

| Nhãn Intent | Ý nghĩa bài toán | Ví dụ câu hỏi |
| :--- | :--- | :--- |
| `COUNT` | Đếm số lượng vật thể | *"Co bao nhieu chai?"*, *"Đếm số cốc trong hình"* |
| `EXIST` | Xác định sự tồn tại (Có / Không) | *"Co dien thoai hay khong?"*, *"Trong ảnh có sách không?"* |
| `LIST` | Liệt kê danh sách các lớp vật thể | *"Trong anh co gi?"*, *"Kể tên các đồ vật xuất hiện"* |
| `LOCATION` | Xác định vùng vị trí ngang | *"Cuon sach nam o dau?"*, *"Điện thoại ở bên nào?"* |
| `OUT_OF_SCOPE` | Câu hỏi ngoài phạm vi dữ liệu | *"Hom nay troi co mua khong?"*, *"Thời tiết thế nào?"* |

> ⚠️ **Quy định ngặt nghèo:** Mô hình intent BẮT BUỘC phải là mô hình học máy (TextCNN); **không được thay thế hoàn toàn bằng các luật từ khóa (keyword rules)**.

---

## 📈 4. CÔNG THỨC CHẤM ĐIỂM VÒNG BẢNG (KAGGLE)

$$S = 0.70 \times (\text{mAP50} \times 100) + 0.30 \times (\text{Accuracy} \times 100)$$

- **mAP50 (70%):** mAP@0.5 trên các dòng `DETECTION`. Bounding box quy đổi về pixel tuyệt đối `[x_min, y_min, x_max, y_max]` trên ảnh gốc `320 × 240`.
- **Accuracy (30%):** Tỷ lệ câu hỏi dự đoán đúng intent trên các dòng `INTENT`.
- **Giới hạn nộp:** Tối đa 4 lần nộp / 24 giờ.
- **Xếp hạng:** Căn cứ vào điểm Private Leaderboard. Tie-break: Đội có mAP50 cao hơn xếp trên; nếu bằng nhau, xét Accuracy.

---

## ⚙️ 5. RÀNG BUỘC KỸ THUẬT TRIỂN KHAI PHẦN CỨNG (ESP32-S3)

| Tiêu chí | Giới hạn bắt buộc |
| :--- | :--- |
| **Phần cứng đích** | **ESP32-S3 N16R8** (Xtensa dual-core LX7 @ 240MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash) + Camera OV7670 |
| **Cơ chế suy luận Detection** | **MỘT mô hình duy nhất; MỘT lần suy luận cho mỗi ảnh.**<br>🚫 **TUYỆT ĐỐI CẤM Ensemble và Test-Time Augmentation (TTA)!** |
| **Số tham số mô hình (Detection)** | $\le \mathbf{200.000}$ tham số (Strict Hard limit). |
| **Dung lượng file mô hình (.tflite)** | $\le \mathbf{1.5\text{ MB}}$ cho Detection và $\le \mathbf{1.5\text{ MB}}$ cho Intent TextCNN. |
| **Chuẩn lượng tử hóa** | **Full Integer INT8** (`inference_input_type = int8`, `inference_output_type = int8`). |
| **Độ tương thích toán tử** | 100% toán tử phải nằm trong danh mục hỗ trợ của **TensorFlow Lite Micro (TFLM)**. |
| **Môi trường suy luận** | 100% chạy on-device trên ESP32-S3; **nghiêm cấm** chuyển ảnh/câu hỏi ra máy tính, cloud hay server. |
| **Thời gian phản hồi** | $\le \mathbf{10\text{ giây}}$ cho mỗi lượt hỏi đáp (tính từ nhận request HTTP POST đến khi trả JSON). |

---

## 🏆 6. QUY CÁCH VÒNG CHUNG KẾT (HỎI ĐÁP Q&A TRỰC TIẾP)

### Luồng xử lý End-to-End:
1. ESP32-S3 nhận request qua HTTP POST `/ask` với payload JSON: `{"q": "Co bao nhieu chai trong hinh"}`.
2. Chuẩn hóa câu hỏi & suy luận TextCNN INT8 để xác định `intent`.
3. Trích xuất lớp vật thể và logic câu hỏi từ điển/quy tắc.
4. Chụp khung hình từ camera OV7670 ($320 \times 240$), tiền xử lý và chạy Object Detection INT8.
5. Bộ xử lý logic (VQA Engine) tổng hợp kết quả và trả JSON: `{"answer": "..."}`.

### Định dạng đáp án chuẩn hóa bắt buộc:
- `COUNT`: Trả về số nguyên dạng chuỗi, ví dụ: `"2"`. Nếu không có vật thể trả `"0"`.
- `EXIST`: Trả về `"Co"` hoặc `"Khong"`.
- `LIST`: Trả về danh sách dạng `"2 bottle, 1 cup"`. Nếu không có vật thể nào trong ảnh trả `"Khong co"`.
- `LOCATION`: Trả về `"ben trai"`, `"o giua"` hoặc `"ben phai"` (xác định dựa trên tọa độ $x$ của tâm bounding box chia 3 vùng theo chiều ngang $W=320$).
- `OUT_OF_SCOPE`: Trả về `"OOS"`.

### Cơ cấu điểm Chung kết:
- **COUNT & EXIST:** 35%
- **LIST:** 25%
- **LOCATION:** 30%
- **OUT_OF_SCOPE:** 10%
*(Mỗi đội thực hiện 20 câu hỏi trực tiếp trên cảnh vật thể thật do BTC bố trí).*
