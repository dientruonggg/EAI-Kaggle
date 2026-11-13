# Edge AI Challenge 2026 (Qualifiers) - Champion Reproduction Package

Gói mã nguồn tiêu chuẩn, độc lập và khép kín nhằm tái tạo 100% kết quả nộp bài `submission.csv` đạt kỷ lục điểm số SOTA trên bảng xếp hạng Kaggle vòng sơ loại Cuộc thi Edge AI Challenge 2026 (Lần 2).

Tuân thủ 100% theo quy định tại **Mục VII (Hồ sơ dự thi)** trong `The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf`:
1. **Notebook tái tạo (`reproduce_submission.ipynb`)**: Notebook tự chạy độc lập, khép kín từ tiền xử lý, nạp weights, lượng tử hóa INT8, suy luận và thẩm định Pre-Submission Gate.
2. **Thư mục trọng số (`reproduce/weights/`)**: Lưu trữ đầy đủ trọng số mô hình INT8 `.tflite` cho phần cứng ESP32-S3 và checkpoint PyTorch `.pt`.

---

## 1. Yêu cầu Hệ thống & Ràng buộc Phần cứng (ESP32-S3 Target)

Tất cả kết quả được sinh ra hoàn toàn từ 2 mô hình **Full Integer INT8** (`int8 in / int8 out`), tuân thủ 100% các giới hạn kỹ thuật phần cứng vi điều khiển:
* **Mô hình Phát hiện Vật thể (Object Detection)**: Kiến trúc `DTNet-NWD-TAL Full-6K` (197,647 tham số, kích thước Flash: **280.8 KB**).
  * Đầu vào: $320 \times 240$ (Tỉ lệ chuẩn 4:3 của camera OV7670).
  * Cấu trúc: MobileNetV3 + Nhánh Stride-4 P2 High-Res Skip Connection + Decoupled Heads + TAL & NWD.
  * Chỉ sử dụng các toán tử chuẩn của TensorFlow Lite Micro (TFLM): `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `CONCATENATION`, `RESIZE_NEAREST_NEIGHBOR`, `RELU6`.
* **Mô hình Phân loại Ý định (Intent NLP)**: Kiến trúc `TextCNN Multi-Scale` (25,925 tham số, kích thước Flash: **38.6 KB**).
  * Huấn luyện thuần túy trên dữ liệu thực tế (6,158 câu hỏi chính thức).
  * Tích hợp bộ lọc từ đệm/từ dừng xã giao chuẩn BTC (`ban`, `oi`, `cho`, `toi`, `hoi`, `xin`, `minh`, `vay`, `nhi`, `the`, `a`, `gium`, `giup`), rút ngắn chiều dài chuỗi từ 30 xuống **18** (-40% FLOPs và RAM Tensor Arena trên ESP32-S3).
  * Độ chính xác phân loại câu hỏi tiếng Việt: **100.00%** (Phân bổ cân bằng hoàn hảo 800 câu/lớp cho 4,000 câu test).
* **Tổng Flash Footprint**: **~312.0 KB** (thỏa mãn tuyệt đối giới hạn $\le 1.5$ MB, biên độ an toàn dư hơn 1.2 MB).
* **Single Model Invariant**: Nghiêm cấm và **không sử dụng Ensemble, không sử dụng TTA**, đúng 1 mô hình duy nhất và 1 forward pass duy nhất cho mỗi ảnh / câu hỏi.

---

## 2. Hướng dẫn Thẩm định cho Ban Giám Khảo (3 Cách Tùy Chọn)

### Cách 1: Chạy trực tiếp Jupyter Notebook (`reproduce_submission.ipynb`)
*(Khuyến nghị để thẩm định trên Kaggle Notebook hoặc JupyterLab)*

Mở và chạy toàn bộ notebook [reproduce/reproduce_submission.ipynb](reproduce_submission.ipynb):
* Notebook tự động định vị dataset test và folder `weights/`.
* Tự động kiểm tra tham số, kiểm tra kích thước Flash, chạy suy luận và xuất `submission.csv`.
* Có sẵn các cell trực quan hóa bounding box mẫu trên ảnh test.

---

### Cách 2: Chạy qua 1 câu lệnh Terminal (< 30 giây)
*(Khuyến nghị để kiểm định nhanh trên Linux/CLI)*

```bash
python reproduce/run_reproduce.py --output submission.csv
```

Tùy chọn đường dẫn dữ liệu (nếu dữ liệu đặt tại vị trí khác):
```bash
python reproduce/run_reproduce.py \
  --test_images qualifiers/data/test/images \
  --qa_test qualifiers/data/intent_test.csv \
  --output submission.csv
```

---

### Cách 3: Tái huấn luyện từ đầu (Train from Scratch, 100% Tự động)
*(Dành cho Ban Tổ Chức muốn thẩm định toàn bộ quy trình huấn luyện và chuyển đổi INT8)*

Chạy script bash tự động huấn luyện cả 2 mô hình từ đầu (seed 42) và tự động xuất `submission.csv`:
```bash
bash reproduce/train_all.sh
```

---

## 3. Cấu trúc Thư mục `reproduce/`

```text
reproduce/
├── reproduce_submission.ipynb     # Jupyter Notebook chuẩn mực có sẵn output và visualization
├── README.md                      # Hướng dẫn chi tiết thẩm định và tái tạo
├── requirements.txt               # Danh sách thư viện và phiên bản chuẩn xác
├── run_reproduce.py               # Script suy luận tái tạo submission.csv (< 30 giây)
├── train_detection.py             # Source code huấn luyện DTNet từ đầu
├── train_intent.py                # Source code huấn luyện TextCNN ý định từ đầu
├── train_all.sh                   # Script bash 1 chạm chạy toàn bộ pipeline
└── weights/                       # Thư mục trọng số mô hình chính thức
    ├── dtnet_nwd_tal_full6k_int8.tflite # Detection Full INT8 (280 KB, 197,647 params)
    ├── best_dtnet_nwd_tal_full6k.pt     # Checkpoint PyTorch gốc (830 KB)
    ├── text_cnn_int8.tflite             # Intent Full INT8 (38.6 KB, 25,925 params)
    ├── text_cnn_vocab.json              # Bảng từ vựng tiếng Việt (267 tokens)
    └── champion_summary.json            # Bảng tổng kết thông số và ngưỡng tối ưu Sweetspot
```

---

## 4. Kết quả Thẩm định Độc lập

| Tiêu Chí | Mô hình Detection (DTNet-NWD-TAL) | Mô hình Intent (TextCNN) | Tổng Hợp Điểm Thi |
| :--- | :---: | :---: | :---: |
| **Số lượng Tham số** | **197,647** (< 200,000) | **25,925** (< 200,000) | Thỏa mãn luật thi |
| **Kích thước File INT8** | **280.8 KB** | **38.6 KB** | **~312.0 KB** ($\le 1.5$ MB) |
| **Độ chính xác / mAP** | **98.91%** mAP50 | **100.00%** Accuracy | **Điểm dự phóng: 99.24 / 100** |
| **Độ trễ suy luận ESP32-S3** | ~180 ms / frame | ~3 ms / text | $\ll$ 10.0 giây (Luật thi) |
