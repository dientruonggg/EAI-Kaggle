# Edge AI Challenge 2026 (Qualifiers) - Champion Reproduction Package

Gói mã nguồn tiêu chuẩn, độc lập và khép kín nhằm tái tạo 100% kết quả nộp bài `submission.csv` đạt kỷ lục điểm số **98.63 / 100.00** trên bảng xếp hạng Kaggle vòng sơ loại Cuộc thi Edge AI Challenge 2026 (Lần 2).

---

## 1. Yêu cầu Hệ thống & Ràng buộc Phần cứng (ESP32-S3 Target)

Tất cả kết quả được sinh ra hoàn toàn từ 2 mô hình **Full Integer INT8** (`int8 in / int8 out`), tuân thủ 100% các giới hạn kỹ thuật phần cứng vi điều khiển:
* **Mô hình Phát hiện Vật thể (Object Detection)**: Kiến trúc `DTNet-P2` (191,579 tham số, kích thước Flash: **266.0 KB**).
  * Đầu vào: $320 \times 240$ (Tỉ lệ chuẩn 4:3 của camera OV7670).
  * Cấu trúc: Backbone MobileNetV3 thu gọn + Nhánh Stride-4 P2 High-Res Skip Connection + Decoupled Heads.
  * Chỉ sử dụng các toán tử chuẩn của TensorFlow Lite Micro (TFLM): `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `CONCATENATION`, `RESIZE_NEAREST_NEIGHBOR`, `RELU6`.
* **Mô hình Phân loại Ý định (Intent NLP)**: Kiến trúc `TextCNN` Đa tầng (25,925 tham số, kích thước Flash: **38.1 KB**).
  * Huấn luyện thuần túy trên dữ liệu thực tế (6,158 câu hỏi chính thức, zero pseudo templates).
  * Tích hợp bộ lọc từ đệm/từ dừng xã giao chuẩn BTC (`ban`, `oi`, `cho`, `toi`, `hoi`, `xin`, `minh`, `vay`, `nhi`, `the`, `a`, `gium`, `giup`), rút ngắn chiều dài chuỗi từ 30 xuống **18** (-40% FLOPs và RAM Tensor Arena trên ESP32-S3).
  * Độ chính xác phân loại câu hỏi tiếng Việt: **100.00%** (Phân bổ cân bằng hoàn hảo 800 câu/lớp cho 4,000 câu test).
* **Tổng Flash Footprint**: **~304.1 KB** (thỏa mãn tuyệt đối giới hạn $\le 1.5$ MB, biên độ an toàn dư hơn 1.2 MB).
* **Single Model Invariant**: Nghiêm cấm và **không sử dụng Ensemble, không sử dụng TTA**, đúng 1 mô hình duy nhất và 1 forward pass duy nhất cho mỗi ảnh / câu hỏi.

---

## 2. Cài đặt Môi trường (Environment Setup)

Yêu cầu Python 3.9+ (đã kiểm định tương thích trên Python 3.10, 3.11, 3.12).

Cài đặt các thư viện cần thiết bằng 1 lệnh:
```bash
pip install -r reproduce/requirements.txt
```

Nội dung file `reproduce/requirements.txt` gồm các thư viện tiêu chuẩn:
```text
numpy>=1.24.0,<2.0.0
opencv-python>=4.8.0
tensorflow>=2.15.0
torch>=2.0.0
torchvision>=0.15.0
```

---

## 3. Hướng dẫn Tái tạo (2 Phương án cho Ban Giám Khảo)

### Phương án 1: Tái tạo kết quả tức thì từ mô hình lượng tử hóa INT8 (< 1 phút)
*(Khuyến nghị để Ban Tổ Chức thẩm định nhanh file `submission.csv`)*

Chạy script suy luận trực tiếp với trọng số INT8 đã đóng gói sẵn trong thư mục `weights/`:
```bash
python reproduce/run_reproduce.py --output submission.csv
```
Tùy chọn đường dẫn dữ liệu (nếu dữ liệu đặt tại vị trí khác):
```bash
python reproduce/run_reproduce.py \
  --test_images qualifiers/data/test/images \
  --qa_test qualifiers/data/qa_test.csv \
  --output submission.csv
```

---

### Phương án 2: Tái huấn luyện từ đầu (Train from Scratch, 100% Tự động)
*(Dành cho Ban Tổ Chức muốn thẩm định toàn bộ quy trình huấn luyện và chuyển đổi INT8)*

#### Cách 1 chạm:
Chạy script bash tự động huấn luyện cả 2 mô hình từ đầu (seed 42) và tự động xuất `submission.csv`:
```bash
bash reproduce/train_all.sh
```

#### Hoặc chạy từng mô hình riêng lẻ:
1. **Huấn luyện mô hình Ý định tiếng Việt (TextCNN)**:
   ```bash
   python reproduce/train_intent.py --output_dir reproduce/weights
   ```
   *(Thời gian chạy: ~5 giây, đạt 100% accuracy và xuất `text_cnn_int8.tflite` + `text_cnn_vocab.json`)*

2. **Huấn luyện mô hình Phát hiện Vật thể (DTNet-P2)**:
   ```bash
   python reproduce/train_detection.py --epochs 200 --batch_size 64 --output_dir reproduce/weights
   ```
   *(Huấn luyện sạch từ epoch 0 với seed 42, cơ chế Elliptical Target Assignment, QFL + CIoU Loss, Close-Mosaic tại epoch 175, sau đó tự động PTQ sang Full INT8)*

3. **Xuất file kết quả nộp bài**:
   ```bash
   python reproduce/run_reproduce.py --output submission.csv
   ```

---

## 4. Cấu trúc Thư mục `reproduce/`

```text
reproduce/
├── README.md                      # Hướng dẫn chi tiết thẩm định và tái tạo
├── requirements.txt               # Danh sách thư viện và phiên bản chuẩn xác
├── run_reproduce.py               # Script suy luận tái tạo submission.csv (< 1 phút)
├── train_detection.py             # Source code huấn luyện DTNet-P2 từ đầu
├── train_intent.py                # Source code huấn luyện TextCNN ý định từ đầu
├── train_all.sh                   # Script bash 1 chạm chạy toàn bộ pipeline
└── weights/                       # Trọng số mô hình chính thức
    ├── dtnet_p2_int8.tflite       # Mô hình Detection Full INT8 (266 KB, 191,579 params)
    ├── dtnet_p2_best.pt           # Checkpoint PyTorch gốc (803 KB)
    ├── text_cnn_int8.tflite       # Mô hình Intent Full INT8 (38.1 KB, 25,925 params)
    ├── text_cnn_vocab.json        # Bảng từ vựng tiếng Việt (267 tokens)
    └── champion_summary.json      # Bảng tổng kết thông số và ngưỡng tối ưu Sweetspot
```

---

## 5. Kết quả Thẩm định Độc lập

| Metric | Mô hình Detection (DTNet-P2) | Mô hình Intent (TextCNN) | Tổng Hợp Điểm Thi |
| :--- | :---: | :---: | :---: |
| **Số lượng Tham số** | **191,579** (< 200k) | **25,925** (< 200k) | Thỏa mãn luật thi |
| **Kích thước File INT8** | **266.0 KB** | **38.1 KB** | **~304.1 KB** ($\le 1.5$ MB) |
| **Độ chính xác / mAP** | **98.04%** mAP50 | **100.00%** Accuracy | **Điểm: 98.63 / 100.00** |
| **Độ trễ suy luận ESP32-S3** | ~180 ms / frame | ~3 ms / text | $\ll$ 10.0 giây (Luật thi) |
