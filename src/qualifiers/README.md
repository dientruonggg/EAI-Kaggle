# EDGE AI CHALLENGE 2026 (LẦN 2) — VÒNG LOẠI (QUALIFIERS)

> **Căn cứ tài liệu chính thức:** `The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf`  
> **Đơn vị tổ chức:** CLB Sinh viên Nghiên cứu Khoa học Pioneer — Khoa Điện tử & Trí tuệ Nhân tạo, Đại học Bách khoa, Đại học Đà Nẵng.

---

## 📌 1. TỔNG QUAN & LỊCH TRÌNH VÒNG BẢNG

- **Kaggle Competition:** `edge-ai-challenge-2026-lan-2`
- **Mở vòng bảng (Qualifiers):** `09/11/2026 — 22/11/2026`
- **Công bố vào Chung kết & Phát Kit ESP32-S3:** `23/11/2026`
- **Vòng Chung kết (Thi đấu trực tiếp):** `05/12/2026`
- **Công thức tính điểm vòng bảng:**
  $$S = 0.70 \times (\text{mAP50} \times 100) + 0.30 \times (\text{Accuracy} \times 100)$$

---

## 🎯 2. ĐẶC TẢ BÀI TOÁN & BỘ NHÃN MỚI

### A. Phát hiện vật thể (Object Detection) — 7 Lớp
Khác với vòng Warm-up (10 lớp giao thông), vòng bảng và chung kết áp dụng bộ nhãn đồ vật thực tế chuẩn bị cho camera OV7670:

| Class ID | Tên Lớp (`classes.txt`) | Tên Tiếng Việt | Ghi chú từ BTC |
| :---: | :--- | :--- | :--- |
| **0** | `bottle` | Chai | Nhiều kiểu chai, kích thước và màu sắc |
| **1** | `cup` | Cốc / Ly | Mẫu cốc khác nhau |
| **2** | `phone` | Điện thoại | Nằm ngang hoặc dựng đứng |
| **3** | `book` | Sách | Nhiều màu bìa và kích thước |
| **4** | `mouse` | Chuột máy tính | Chuột có dây hoặc không dây |
| **5** | `box` | Hộp | Hộp có hình dạng gần hộp chữ nhật |
| **6** | `toy_car` | Ô tô đồ chơi | Nhiều mẫu, màu sắc và hướng đặt |

- **Kích thước ảnh nguồn:** `320 × 240 pixel`.
- **Kích thước đầu vào mô hình khuyến nghị:** `96 × 96`, `128 × 128`, hoặc `160 × 160 pixel`.
- **Tọa độ Bounding Box trong file nộp:** Pixel tuyệt đối `[x_min, y_min, x_max, y_max]` trên ảnh `320 × 240`. Tối đa 100 boxes/ảnh.

### B. Phân loại ý định (Intent TextCNN) — 5 Nhóm
- `COUNT`: Đếm số lượng vật thể (*"Co bao nhieu chai?"*)
- `EXIST`: Xác định sự tồn tại (*"Co dien thoai hay khong?"*)
- `LIST`: Liệt kê các lớp vật thể (*"Trong anh co gi?"*)
- `LOCATION`: Xác định vùng vị trí ngang (*"Cuon sach nam o dau?"*)
- `OUT_OF_SCOPE`: Câu hỏi ngoài phạm vi (*"Hom nay troi co mua khong?"*)

---

## ⚡ 3. RÀNG BUỘC KỸ THUẬT TRIỂN KHAI PHẦN CỨNG (MỤC V)

Để bảo đảm tính hợp lệ khi bước vào Chung kết trên **ESP32-S3 N16R8 + Camera OV7670**:

1. **Detection Invariant:** 
   - **MỘT mô hình duy nhất; MỘT lần suy luận cho mỗi ảnh.**
   - 🚫 **TUYỆT ĐỐI CẤM Ensemble và Test-Time Augmentation (TTA)!**
2. **Giới hạn tham số:** $\le \mathbf{200.000}$ tham số (Model MobileNetV2-BBox hiện tại: **94.284 tham số** ~ 47% budget).
3. **Giới hạn dung lượng:** File `.tflite` $\le \mathbf{1.5\text{ MB}}$ cho Detection và $\le \mathbf{1.5\text{ MB}}$ cho Intent TextCNN.
4. **Lượng tử hóa:** Bắt buộc **Full Integer INT8** (`int8 input/output`).
5. **Độ tương thích toán tử:** 100% toán tử được hỗ trợ native bởi **TensorFlow Lite Micro (TFLM)**.

---

## 📁 4. CẤU TRÚC THƯ MỤC `qualifiers/`

```
qualifiers/
├── data/
│   ├── classes.txt               # 7 nhãn chính thức (bottle, cup, phone, book, mouse, box, toy_car)
│   ├── dataset.yaml              # Config dataset 7 classes, 320x240
│   ├── intent_train.csv          # Dataset câu hỏi huấn luyện TextCNN (tiếng Việt)
│   ├── intent_val.csv            # Dataset câu hỏi đánh giá
│   ├── train/ (images, labels)   # Sẵn sàng đón dataset từ BTC (09/11)
│   ├── val/ (images, labels)     # Sẵn sàng đón dataset từ BTC (09/11)
│   └── test/ (images)            # Ảnh test chính thức
├── src/
│   ├── detection/
│   │   ├── model.py              # MobileNetV2-BBox FPN (94k params < 200k)
│   │   ├── dataset.py            # Dataset loader chuẩn hóa 320x240 -> 160x160
│   │   ├── losses.py             # Focal Loss & CIoU Loss
│   │   ├── train.py              # Pipeline train Single Model
│   │   ├── infer.py              # Suy luận Single Model (No Ensemble/TTA)
│   │   └── quantize.py           # PTQ Full Integer INT8 export & audit
│   ├── intent/
│   │   ├── text_cnn.py           # Huấn luyện TextCNN PyTorch & Infer
│   │   └── export_tflite.py      # Lượng tử hóa TextCNN sang INT8 TFLite Micro
│   ├── vqa_engine/
│   │   ├── question_parser.py    # Xử lý tiếng Việt & trích xuất thực thể
│   │   ├── spatial_reasoner.py   # Phân tích vùng ngang: "ben trai", "o giua", "ben phai"
│   │   ├── answer_generator.py   # Sinh đáp án theo chuẩn Chung kết (Section VI.3)
│   │   └── mock_ask_server.py    # Mockup server HTTP POST /ask giả lập ESP32-S3
│   ├── evaluation.py             # Đánh giá mAP50 (7 classes) + Accuracy + Score S
│   └── eval_local.py             # Script đánh giá nhanh local
├── kaggle_runner/
│   ├── kaggle_train_pipeline.py  # Script train song song trên 2x T4 GPU
│   ├── run_remote_train.py       # Trigger push/pull Kaggle Kernel
│   └── kernel-metadata.json      # Metadata Kaggle
├── submissions/                  # Chứa các file submission.csv
├── weights/                      # Chứa checkpoints và file .tflite INT8
├── submit.py                     # CLI kiểm tra định dạng & nộp bài lên Kaggle
├── build_submission.py           # Ghép Detection + Intent thành submission.csv
├── run_pipeline.py               # Master pipeline chạy từ đầu đến cuối
├── RULES_SUMMARY.md              # Bảng tóm tắt thể lệ chi tiết trích từ PDF
└── README.md                     # Tài liệu hướng dẫn này
```

---

## 🚀 5. HƯỚNG DẪN THỰC THI (QUICK START)

### 1. Huấn luyện TextCNN & Phân loại Intent:
```bash
.venv/bin/python qualifiers/src/intent/text_cnn.py
```

### 2. Xuất TextCNN sang INT8 TFLite Micro:
```bash
.venv/bin/python qualifiers/src/intent/export_tflite.py
```

### 3. Huấn luyện mô hình Detection (Single Model, < 200k params):
```bash
.venv/bin/python qualifiers/src/detection/train.py
```

### 4. Lượng tử hóa Full Integer INT8 & Kiểm thử phần cứng:
```bash
.venv/bin/python qualifiers/src/detection/quantize.py --img_size 160
```

### 5. Chạy suy luận Single Model trên tập Test:
```bash
.venv/bin/python qualifiers/src/detection/infer.py \
    --model qualifiers/weights/mobilenet_bbox_160_int8.tflite \
    --images_dir qualifiers/data/test/images \
    --output qualifiers/submissions/detection_preds.csv
```

### 6. Ghép thành file `submission.csv` hoàn chỉnh & Kiểm định định dạng:
```bash
.venv/bin/python qualifiers/build_submission.py \
    --detection qualifiers/submissions/detection_preds.csv \
    --intent qualifiers/submissions/intent_preds.csv \
    --output qualifiers/submissions/submission_qualifiers.csv
```

### 7. Nộp bài trực tiếp lên Kaggle:
```bash
python3 qualifiers/submit.py qualifiers/submissions/submission_qualifiers.csv "Single Model MobileNetV2-BBox INT8" --submit
```

### 8. Chạy thử nghiệm Mock ESP32-S3 VQA Server:
```bash
.venv/bin/python qualifiers/src/vqa_engine/mock_ask_server.py
```
*(Trong terminal khác, test gửi câu hỏi:)*
```bash
curl -X POST http://localhost:8080/ask \
     -H "Content-Type: application/json" \
     -d '{"q": "Co bao nhieu chai trong hinh?"}'
# Trả về: {"answer": "2"}
```
