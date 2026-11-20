# PIPELINE DỮ LIỆU, GÁN NHÃN INTENT & CHIẾN LƯỢC HÒA TRỘN DỮ LIỆU NGOÀI
## Cuộc thi Edge AI Challenge 2026 (Lần 2) — Qualifiers

---

## 🎯 1. BỘ DỮ LIỆU MẪU (SEED DATASET) 7 LỚP VẬT THỂ

Theo Thể lệ, cuộc thi chỉ tập trung vào 7 lớp vật thể quan sát bằng camera OV7670 ($320 \times 240\text{ px}$):
`0: bottle`, `1: cup`, `2: phone`, `3: book`, `4: mouse`, `5: box`, `6: toy_car`.

### Cơ chế sinh dữ liệu mẫu (`qualifiers/src/data_tools/build_seed_dataset.py`):
1. **Trích xuất ảnh thực tế:** Thu hoạch các vật thể `bottle`, `cup`, `phone`, `book`, `mouse`, `box`, `toy_car` từ tập dữ liệu COCO, resize chuẩn về $320 \times 240$.
2. **Tổng hợp cảnh đồ vật trên bàn (Tabletop Synthesis):** Tự động sắp đặt các đồ vật với nhiều số lượng (1 đến 4 vật thể/ảnh), đa dạng màu sắc, kích thước và phân bổ đều trên 3 vùng ngang (`ben trai`, `o giua`, `ben phai`).
3. **Phân chia 3 tập chuẩn:**
   - `train`: 112 ảnh, 296 bounding box
   - `val`: 24 ảnh, 43 bounding box
   - `test`: 24 ảnh, 41 bounding box

---

## 🏷️ 2. HỆ THỐNG GÁN NHÃN TỰ ĐỘNG THEO 5 INTENT (`auto_qa_annotator.py`)

Từ bounding box của từng ảnh, script [`auto_qa_annotator.py`](file:///home/bim/Projects/EAI-2026/qualifiers/src/data_tools/auto_qa_annotator.py) tự động sinh các cặp câu hỏi — đáp án tiếng Việt chuẩn hóa theo đúng Mục VI.3 của Thể lệ:

| Nhãn Intent | Câu hỏi mẫu sinh tự động | Quy tắc đáp án | Ví dụ kết quả thực tế |
| :--- | :--- | :--- | :--- |
| **`COUNT`** | *"Trong anh co may cuon sach?"*<br>*"Dem so xe do choi trong anh."* | Trả về số nguyên dạng chuỗi. Nếu không có vật thể trả `"0"`. | `4`<br>`0` |
| **`EXIST`** | *"Trong anh co sach khong?"*<br>*"Hinh anh co smartphone nao khong?"* | Trả về `"Co"` nếu $N \ge 1$; trả về `"Khong"` nếu $N = 0$. | `Co`<br>`Khong` |
| **`LIST`** | *"Hay liet ke toan bo cac vat the trong hinh."*<br>*"Trong anh co gi?"* | Danh sách `<count> <class_en>`, sắp xếp theo tên. Nếu rỗng trả `"Khong co"`. | `4 book, 1 mouse`<br>`Khong co` |
| **`LOCATION`** | *"Vi tri cua chuot la o dau?"*<br>*"Chai nuoc nam o ben trai, o giua hay ben phai?"* | Tính tâm $x$ của box trên ảnh $W=320$:<br>- $x < 106.7$: `"ben trai"`<br>- $106.7 \le x \le 213.3$: `"o giua"`<br>- $x > 213.3$: `"ben phai"` | `ben trai`<br>`ben phai` |
| **`OUT_OF_SCOPE`** | *"Hom nay troi co mua khong?"*<br>*"Ban co biet hat khong?"* | Các câu hỏi ngoài phạm vi 7 đồ vật tabletop. | `OOS` |

Kết quả lưu tại:
- [`qualifiers/data/qa_train.csv`](file:///home/bim/Projects/EAI-2026/qualifiers/data/qa_train.csv) (739 cặp QA)
- [`qualifiers/data/qa_val.csv`](file:///home/bim/Projects/EAI-2026/qualifiers/data/qa_val.csv) (158 cặp QA)
- [`qualifiers/data/qa_test.csv`](file:///home/bim/Projects/EAI-2026/qualifiers/data/qa_test.csv) (156 cặp QA)

---

## ⚡ 3. SO SÁNH ONE-HOT VÀ AUTO-ENCODER (AUTO-EN) CHO INTENT

### Câu hỏi: *Phân loại Intent thì dùng One-Hot hay Auto-Encoder được không?*

#### 1. Dùng One-Hot Encoding (Khuyên dùng chính thức):
- **Cơ chế:** Mã hóa 5 intent thành vector nhị phân 5 chiều:
  - `COUNT`: $[1, 0, 0, 0, 0]$
  - `EXIST`: $[0, 1, 0, 0, 0]$
  - `LIST`: $[0, 0, 1, 0, 0]$
  - `LOCATION`: $[0, 0, 0, 1, 0]$
  - `OUT_OF_SCOPE`: $[0, 0, 0, 0, 1]$
- **Ưu điểm vượt trội:**
  - Đúng 100% Thể lệ Mục III.3 (*"xây dựng mô hình TextCNN để phân loại intent"*).
  - Tầng cuối của TextCNN chỉ là `Dense(5, activation="softmax")`.
  - Mô hình chỉ có **8.677 parameters**, khi lượng tử hóa INT8 chỉ nặng **15.89 KB**, chạy suy luận cực nhanh ($< 2\text{ms}$) trên ESP32-S3.

#### 2. Dùng Auto-Encoder (AutoEn):
- **Có thể dùng được không?** **CÓ**, và cực kỳ hiệu quả trong vai trò **Bộ lọc phát hiện bất thường (Out-Of-Distribution / Out-Of-Scope Filter)**.
- **Cách hoạt động (`qualifiers/src/intent/autoencoder_ood.py`):**
  - Huấn luyện một Text Auto-Encoder chỉ trên các câu hỏi hợp lệ (COUNT, EXIST, LIST, LOCATION).
  - Khi có câu hỏi đưa vào:
    - Nếu câu hỏi bình thường $\rightarrow$ Sai số tái tạo (Reconstruction Error MSE) rất thấp ($\le 0.05$).
    - Nếu là câu hỏi ngoài phạm vi (*"Thời tiết hôm nay thế nào?"*, *"Giá xăng bao nhiêu?"*) $\rightarrow$ Mô hình không tái tạo được, MSE vọt cao ($> 0.08$) $\rightarrow$ Gán nhãn ngay là `OUT_OF_SCOPE`!
- **Kết luận:**
  - Dùng **TextCNN với One-Hot Softmax** làm mô hình chính thức nộp bài (đạt chuẩn thể lệ, nhẹ và chuẩn TFLite Micro).
  - Có thể dùng cơ chế tái tạo (Reconstruction Error) của **Auto-Encoder** làm lớp kiểm định phụ trợ chống đoán mò cho các câu hỏi lạ.

---

## 🔄 4. CHIẾN LƯỢC HÒA TRỘN DỮ LIỆU NGOÀI KHỚP PHÂN PHỐI BTC

Khi Ban Tổ Chức công bố bộ dữ liệu công khai trên Kaggle (dự kiến Kickoff 08/11 hoặc mở bảng 09/11):
> ⚠️ **CẢNH BÁO NGUY HIỂM:** Tuyệt đối không ném bừa hàng nghìn ảnh dữ liệu ngoài (như toàn bộ COCO hay OpenImages) vào tập train. Điều này sẽ gây ra **Covariate Shift** và **Label Skew**, làm mô hình bị lệch phân phối so với ảnh test thật của BTC!

### Quy trình 3 bước tự động bằng [`distribution_matcher.py`](file:///home/bim/Projects/EAI-2026/qualifiers/src/data_tools/distribution_matcher.py):

1. **Bước 1 — Đo đạc phân phối chuẩn của BTC (Reference Profiling):**
   ```bash
   .venv/bin/python qualifiers/src/data_tools/distribution_matcher.py --btc_dir qualifiers/data/train_btc
   ```
   Script sẽ tính toán ma trận phân phối mục tiêu:
   - Tỷ lệ xuất hiện của từng lớp: $P_{\text{BTC}}(\text{bottle}), P_{\text{BTC}}(\text{cup}), \dots$
   - Mật độ số box trung bình trên mỗi ảnh.
   - Diện tích tương đối và tỉ lệ khung hình của bounding box.

2. **Bước 2 — Tính toán trọng số lấy mẫu (Inverse Density Weighting):**
   Với mỗi ảnh ngoài, tính điểm ưu tiên dựa trên độ thiếu hụt của các lớp so với BTC:
   $$w_c = \frac{P_{\text{BTC}}(c)}{P_{\text{EXT}}(c)}$$
   Các lớp mà tập BTC có nhiều nhưng tập ngoài đang thiếu (ví dụ `box`, `cup`) sẽ được ưu tiên bốc vào.

3. **Bước 3 — Hòa trộn & Kiểm định sai số phân phối (Distribution Audit):**
   Script chọn lọc đúng số lượng ảnh ngoài cần thiết sao cho tập dữ liệu sau khi hòa trộn có phân phối $P_{\text{Merged}}(c)$ khớp với $P_{\text{BTC}}(c)$ với sai số lệch $< 0.5\%$.
   Báo cáo kiểm định được lưu tự động tại `distribution_audit.json`.
