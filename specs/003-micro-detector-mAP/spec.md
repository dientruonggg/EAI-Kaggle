# Feature Specification: Robust ESP32-S3 Micro-Scale Detector Calibration & Training

**Feature Branch**: `003-micro-detector-mAP`  
**Created**: 2026-11-08  
**Status**: In Progress (Spec-Driven Development)  
**Input**: Phân tích chẩn đoán hệ thống và nâng cấp toàn diện mô hình Object Detection (< 200k params, TFLite INT8, ESP32-S3) sau khi đạt 100% Intent

---

## 🎯 Background & Problem Statement

Hệ thống đã đạt mốc **100.00% Intent Accuracy** với TextCNN INT8 (19.36 KB, 11,493 tham số), đáp ứng trọn vẹn 30/30 điểm Intent theo Hiến chương.
Tuy nhiên, qua kiểm thử nghiệm thu thực tế trên tập validation (24 ảnh `qual_val_0001` .. `qual_val_0024`), nhánh **Object Detection** đang gặp lỗi nghiêm trọng:
1. **mAP@0.5 = 0.00%**: Mô hình dự đoán sai hoàn toàn tọa độ và nhãn.
2. **Bùng nổ False Positives**: Sinh trung bình 27 box/ảnh (tổng 647 box trên 24 ảnh val), làm Precision sụt giảm về gần 0.
3. **Nguyên nhân cốt lõi trong mã nguồn:**
   - **Lỗi Unmasked Box Loss:** `MSE` được tính trên toàn bộ $20 \times 20 = 400$ grid cells, trong khi 95% số ô là nền rỗng có target `[0,0,0,0]`. MSE ép toàn bộ đầu ra về 0, áp đảo hoàn toàn tín hiệu của các bounding box thật.
   - **Lỗi Focal Loss chưa cân bằng $\alpha_t$:** Chưa phân tách hệ số phạt giữa nền và vật thể.
   - **Lỗi Decoding trong `infer.py`:** Hàm giải mã bỏ qua xác suất `probs[0]` (background) mà chỉ lấy `np.argmax(probs[1:])`, khiến các ô nền bị nhận nhầm thành vật thể khi softmax bị nhiễu.
   - **Tập dữ liệu huấn luyện còn nhỏ & mất cân bằng:** Chỉ có 112 ảnh (296 boxes), các lớp `mouse` (25), `box` (27), `phone` (29) bị lép vế trước `toy_car` (68), và chưa tích hợp bộ lọc che khuất bàn ăn (`balance_and_occlude_dataset.py`).

---

## 🚀 User Scenarios & Acceptance Criteria

### User Story 1 - Masked Loss & Objectness-Gated Detector Training (Priority: P1)

Tái cấu trúc hàm mất mát (loss function) trong `qualifiers/src/detection/losses.py` và `train.py` sao cho:
- Box loss (Smooth L1 / Masked CIoU) **CHỈ ĐƯỢC TÍNH** tại các ô lưới có chứa vật thể thật (`cls_target[:, :, 0] == 0` hay `object_mask == 1`).
- Phân tách rõ ràng giữa phân loại nền vs tiền cảnh và phân loại 7 lớp vật thể.

**Why this priority**: Đây là nguyên nhân trực tiếp làm mAP@0.5 = 0.00%. Nếu không mask box loss, mô hình không thể học được tọa độ bounding box.

**Acceptance Scenarios**:
1. **Scenario 1.1 (Masked Box Loss Invariant)**:
   - **Given**: Grid $20 \times 20$ với 3 vật thể (3 ô positive, 397 ô background).
   - **When**: Tính loss hồi quy bounding box.
   - **Then**: 397 ô background không đóng góp bất kỳ giá trị loss nào vào box loss; box loss chỉ được chia trung bình cho 3 ô positive (hoặc $\ge 1.0$).
2. **Scenario 1.2 (Focal Loss Foreground Calibration)**:
   - **Given**: Ma trận ground truth one-hot 8 class.
   - **When**: Tính classification loss.
   - **Then**: Tín hiệu phạt nền rỗng bị triệt tiêu thích ứng qua $\gamma=2.0$ và $\alpha_t$, ngăn chặn tình trạng tràn gradient từ nền.

---

### User Story 2 - Calibrated Decoding & Background Suppression (Priority: P1)

Nâng cấp thuật toán giải mã tensor đầu ra trong `qualifiers/src/detection/infer.py`:
- Chỉ kích hoạt đề xuất vật thể khi xác suất vật thể vượt trội hơn xác suất nền: $P(\text{class}) > P(\text{background})$ VÀ $P(\text{class}) \ge \text{conf\_thresh}$.
- Chuẩn hóa lại tọa độ bounding box trong không gian $320 \times 240$ và áp dụng Non-Maximum Suppression (NMS) với ngưỡng IoU chuẩn $0.45$.

**Why this priority**: Loại bỏ hoàn toàn hiện tượng bùng nổ 27 box ảo/ảnh, đưa số lượng box về dải hợp lý (1 đến 5 box/ảnh).

**Acceptance Scenarios**:
1. **Scenario 2.1 (Background Rejection)**:
   - **Given**: Một ô lưới có $P(\text{bg}) = 0.60$ và $P(\text{mouse}) = 0.25$.
   - **When**: Chạy hàm giải mã `infer.py` với `conf_thresh = 0.20`.
   - **Then**: Ô lưới này BẮT BUỘC bị loại bỏ vì $P(\text{bg}) > P(\text{mouse})$.
2. **Scenario 2.2 (Density Sanity Check)**:
   - **Given**: Tập 24 ảnh validation.
   - **When**: Chạy inference toàn bộ tập validation.
   - **Then**: Tổng số bounding box sinh ra nằm trong khoảng $20 \le N \le 120$ (trung bình 1–5 box/ảnh), loại bỏ hoàn toàn 647 box rác.

---

### User Story 3 - Dataset Balancing & Occlusion Augmentation (Priority: P2)

Tích hợp trực tiếp module `qualifiers/src/data_tools/balance_and_occlude_dataset.py` vào `QualifierDetectionDataset`:
- Cơ chế che khuất ngẫu nhiên (Cutout, Shadow, Table Blur) trên các bounding box để mô hình học nhận dạng vật thể bị che khuất (theo đúng kịch bản đề thi thực tế).
- Cân bằng trọng số loss theo tần suất hiệu dụng mẫu (CVPR 2019 Effective Number of Samples) hoặc nhân bản copy-paste cho các lớp thiểu số (`mouse`, `box`, `phone`).

**Why this priority**: Dữ liệu thực tế vòng bảng và chung kết sẽ có vật thể bị che khuất một phần (occlusion) trên bàn.

**Acceptance Scenarios**:
1. **Scenario 3.1 (Occlusion Augmentation Pipeline)**:
   - **Given**: Ảnh huấn luyện có bounding box vật thể.
   - **When**: Nạp qua `QualifierDetectionDataset(augment=True)`.
   - **Then**: Áp dụng che khuất 15%–45% diện tích vật thể với xác suất thích hợp, nhãn tọa độ gốc vẫn được bảo toàn.
2. **Scenario 3.2 (Class Weight Integration)**:
   - **Given**: Tỷ lệ mẫu mất cân bằng giữa `mouse` (25) và `toy_car` (68).
   - **When**: Huấn luyện detector.
   - **Then**: Trọng số phạt của `mouse` và `box` được tăng cường theo hệ số Smooth Sqrt / ENS ($\ge 1.4\times$).

---

### User Story 4 - Full Integer INT8 PTQ & Validation mAP Convergence (Priority: P1)

Xuất mô hình đã hiệu chỉnh sang TensorFlow Lite Micro INT8 (`qualifiers/weights/mobilenet_bbox_160_int8.tflite`):
- Giữ nguyên số tham số $< 200,000$ (94,284 params) và kích thước tệp $\le 1.5$ MB (~120 KB).
- Toàn bộ toán tử tương thích 100% với TFLite Micro trên ESP32-S3.
- Đo lường và nghiệm thu chỉ số mAP@0.5 trên tập validation: Đạt tối thiểu **$\ge 40 - 60\%$** (vượt bậc so với 0.00% ban đầu).
- Tích hợp vào `build_submission.py` để sinh ra file nộp hợp lệ.

**Why this priority**: Đảm bảo toàn bộ giải pháp thỏa mãn Hiến chương và sẵn sàng nộp bài hoặc triển khai phần cứng.

**Acceptance Scenarios**:
1. **Scenario 4.1 (Hardware Limits)**:
   - **Given**: File `mobilenet_bbox_160_int8.tflite`.
   - **When**: Kiểm tra tham số và kích thước.
   - **Then**: Tham số < 200,000, dung lượng file $\le 1,500$ KB, kiểu dữ liệu `int8`.
2. **Scenario 4.2 (mAP Benchmark)**:
   - **Given**: File dự đoán validation `val_detection_preds.csv`.
   - **When**: Chạy `calculate_map50(gt_det, pred_det)`.
   - **Then**: mAP@0.5 đạt $\ge 40.0\%$, tất cả 7 class đều có phát hiện hợp lệ.
