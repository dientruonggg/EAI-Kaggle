# 🏆 CHIẾN LƯỢC & MA TRẬN BĂNG THỬ NGHIỆM (BENCHMARK MATRIX)
## Edge AI Challenge 2026 — Vòng loại Kaggle & Vòng chung kết ESP32-S3

---

## 🎯 Mục tiêu kép của Đội thi
1. **Vòng loại (Kaggle Leaderboard):** Tối đa hóa điểm số mAP (mAP@0.5 và mAP@0.5:0.95) để **đứng Top bảng xếp hạng**, đảm bảo vé vào Vòng chung kết.
2. **Vòng chung kết (ESP32-S3 + OV7670):** Có sẵn các phương án mô hình siêu nhẹ (INT8, $< 500$ KB, $< 200$ KB RAM) có thể chạy $100\%$ on-device, không bỡ ngỡ khi bước vào phần cứng.

---

## 📊 1. BẢNG MA TRẬN ỨNG CỬ VIÊN MÔ HÌNH (MODEL SELECTION MATRIX)

| Tier | Tên Mô Hình | Backbone / Params | GFLOPs (Input Size) | mAP kỳ vọng (COCO/VOC) | Tốc độ Kaggle T4 (ms) | Khả năng nhúng ESP32-S3 | Vai trò chiến lược trong cuộc thi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Max Accuracy** *(Cày Top Kaggle)* | **YOLO11x** | C3k2 + SPPF (56.9M) | 194.9 (640x640) | **Rất cao (54.7%)** | ~12.5 ms | ❌ Không thể | **Gánh điểm Leaderboard chính thức.** Kết hợp Mosaic, Mixup, TTA để lấy Top 1-3. |
| **Tier 1: Max Accuracy** *(Cày Top Kaggle)* | **RT-DETR-X / L** | HGNetv2 + Transformer (65M / 32M) | 215 / 110 (640x640) | **Cực cao (54.8%)** | ~14.0 ms | ❌ Không thể | **Đua Top không cần NMS.** Rất mạnh ở các góc chụp phức tạp. Ghép ensemble với YOLO11x qua WBF. |
| **Tier 2: The Sweet Spot** *(Cân bằng)* | **YOLO11s** | C3k2 + SPPF (9.4M) | 21.5 (640x640) | **Cao (47.0%)** | ~3.5 ms | ⚠️ Khó (Cần Prune 80% + INT8) | **Mô hình đối chứng đa năng.** Nhanh, chuẩn, dễ huấn luyện để thử nghiệm hyperparameter. |
| **Tier 2: The Sweet Spot** *(Cân bằng)* | **NanoDet-Plus-m** | Custom ShuffleNetV2 (1.2M) | 1.8 (416x416) | **Khá (30.4%)** | ~2.1 ms | ⚠️ Khả thi trên PSRAM | **Cầu nối giữa Kaggle và MCU.** Anchor-free, thiết kế thuần cho Edge computing. |
| **Tier 3: Ultra-Fast** *(Sẵn sàng ESP32)* | **YOLO11n** | C3k2 + SPPF (2.6M) | 6.5 (640x640) / 0.8 (160x160) | **Tốt (39.5% @ 640)** | ~1.5 ms | ✅ Khả thi (với input 128-160 INT8) | **Mô hình dự phòng chiến lược.** Có thể scale resolution xuống $160 \times 160$ để chạy thử nghiệm. |
| **Tier 3: Ultra-Fast** *(Sẵn sàng ESP32)* | **MobileNetV2-SSDLite (0.35x)** | Inverted Residuals (0.8M) | 0.25 (160x160) | **Vừa phải (~22-26%)** | ~1.0 ms | ✅ **Rất cao (TFLite Micro)** | **Chuẩn mực TinyML.** 100% toán tử được hỗ trợ native trong TensorFlow Lite for Microcontrollers. |
| **Tier 3: Ultra-Fast** *(Sẵn sàng ESP32)* | **FOMO (MobileNetV2 0.35x)** | Centroid Heatmap (0.3M) | 0.08 (128x128) | **Đạt chuẩn (Centroid mAP)** | ~0.5 ms | 🚀 **VUA TỐC ĐỘ (15-25 FPS)** | **Vũ khí bí mật cho Vòng chung kết.** Bỏ bounding box, chỉ bắt tâm vật thể. Chạy mượt nhất trên ESP32-S3 + OV7670. |
| **Tier 4: Baseline** *(Đối chứng)* | **Faster R-CNN** | ResNet50-FPN (41.5M) | ~180 (800x800) | **Chuẩn (37-40%)** | ~25 ms | ❌ Không thể | **Baseline học thuật** để viết báo cáo kỹ thuật vòng loại. |

---

## 🛠️ 2. KHUNG PIPELINE TRIỂN KHAI & THỰC NGHIỆM (EXECUTION PIPELINE)

```
EAI-2026/
├── configs/                  # Cấu hình hyperparameters, data.yaml
├── datasets/                 # Chứa dữ liệu thử nghiệm (format chuẩn YOLO)
│   ├── images/ (train, val, test)
│   └── labels/ (train, val, test)
├── src/
│   ├── data/                 # Scripts tải dataset, augmentations (Albumentations)
│   ├── models/               # Định nghĩa kiến trúc (FOMO, MobileNetV2-SSD, NanoDet)
│   ├── training/             # Scripts train PyTorch DDP / Ultralytics
│   ├── evaluation/           # Đánh giá mAP, confusion matrix, FPS test
│   └── export/               # Pipeline xuất ONNX, TFLite Full INT8, ESP-DL C-header
├── notebooks/                # Các notebook hoàn chỉnh sẵn sàng ném lên Kaggle
└── benchmarks/               # Log kết quả benchmark mAP, latency, size
```

### Quy trình 4 bước chuẩn bị trước khi BTC công bố đề:
1. **Bước 1: Chuẩn hóa Dataset Format:** Toàn bộ dữ liệu benchmark dùng chuẩn **YOLO Format** (`class_id x_center y_center width height` normalized $0-1$).
2. **Bước 2: Dựng Pipeline Training Baseline trên Kaggle:**
   - Viết sẵn template notebook với GPU T4 x 2 (sử dụng `torch.distributed` hoặc Ultralytics multi-GPU).
   - Tích hợp kỹ thuật Kaggle Grandmaster: **Mosaic, MixUp, Albumentations (SafeCrop, ColorJitter), CosineAnnealingLR**.
3. **Bước 3: Xây dựng Kỹ thuật Ensemble WBF (Weighted Boxes Fusion):**
   - Viết sẵn script gộp box dự đoán từ 2-3 model khác nhau (ví dụ: YOLO11x + RT-DETR) để tăng trực tiếp từ $+1.5$ đến $+3.0$ điểm mAP trên Leaderboard.
4. **Bước 4: Pipeline Xuất INT8 & Test Latency Giả Lập:**
   - Script tự động convert từ `.pt` $\rightarrow$ `.onnx` $\rightarrow$ `.tflite` (Full INT8 với Representative Dataset gồm 100 ảnh).
   - Kiểm tra xem mô hình có chứa toán tử lạ (unsupported ops) làm crash TFLite Micro hay không.

---

## 📋 3. BẢNG PHÂN BỔ NHIỆM VỤ ĐỘI THI (TEAM ROLES)

| Vai trò | Nhiệm vụ chính | Output cần bàn giao |
| :--- | :--- | :--- |
| **Kaggle Leaderboard Master** | Chuyên trách **Tier 1 (YOLO11x / RT-DETR)**, tuning hyperparameter, augmentation, WBF ensemble để đua top. | Notebook training Kaggle đạt mAP cao nhất. |
| **Edge & Quantization Engineer** | Chuyên trách **Tier 2 & Tier 3 (YOLO11n, MobileNetV2-SSD, FOMO)**, pipeline lượng tử hóa INT8, kiểm tra dung lượng RAM/Flash. | File `.tflite` INT8 $< 500$ KB và benchmark sai số lượng tử hóa. |
| **Embedded & Camera Specialist** | Cấu hình ESP32-S3 với OV7670 qua ngoại vi `LCD_CAM` DMA, đo đạc FPS thực tế, chuẩn bị khung TFLite Micro C++. | Firmware ESP-IDF/PlatformIO đọc frame từ OV7670 và nạp model test. |
| **Tiny NLP & Logic Developer** | Thiết kế bộ phân loại ý định (Intent Classifier) và trích xuất thực thể (Slot Parser) bằng Regex / Tiny MLP chạy offline trên C++. | Module C++ nhận chuỗi câu hỏi và sinh câu trả lời theo đúng định dạng. |
