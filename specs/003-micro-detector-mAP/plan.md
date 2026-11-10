# Implementation Plan: Robust ESP32-S3 Micro-Scale Detector Calibration

**Branch**: `003-micro-detector-mAP` | **Date**: 2026-11-08 | **Spec**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/003-micro-detector-mAP/spec.md)

---

## 1. Summary

Cải tiến toàn diện mô hình Object Detection MobileNetV2-BiFPN (< 200k params) trên ESP32-S3:
1. Thay thế Unmasked MSE Loss bằng **Masked Smooth-L1 / CIoU Loss** chỉ tính toán trên các ô lưới chứa vật thể thật (`cls_target[:, :, 0] == 0`), loại bỏ triệt để hiện tượng gradient nền kéo tọa độ về 0.
2. Hiệu chỉnh **Balanced Focal Loss** với $\alpha_t$ phân tách rõ ràng giữa nền ($1 - \alpha$) và 7 lớp vật thể ($\alpha \cdot W_c$).
3. Nâng cấp thuật toán giải mã tại `infer.py` với **Explicit Background Rejection** ($P(\text{fg}) > P(\text{bg})$) và chuẩn hóa NMS, triệt tiêu 600+ box rác.
4. Tích hợp trực tiếp engine che khuất bàn ăn (`balance_and_occlude_dataset.py`) vào `QualifierDetectionDataset`.
5. Huấn luyện lại detector, lượng tử hóa Full Integer INT8 PTQ, và đo lường mAP@0.5 trên tập validation đạt mục tiêu $\ge 40 - 60\%$.

---

## 2. Technical Context & Invariants

- **Language / Runtime**: Python 3.11 in `.venv/bin/python`.
- **Deep Learning Framework**: TensorFlow 2.19 / Keras 3.15 (TFLite Micro target).
- **Target Microcontroller**: ESP32-S3 N16R8 (Xtensa LX7 @ 240MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash).
- **Parameter Ceiling**: Strictly $< 200,000$ parameters (Current: 94,284).
- **Model Storage Budget**: $\le 1.5$ MB TFLite binary.
- **Quantization Standard**: Full Integer INT8 PTQ (`inference_input_type=int8`, `inference_output_type=int8`).
- **Operators**: Conv2D, DepthwiseConv2D, Add, UpSampling2D (nearest), Concatenate, Softmax, Sigmoid (Zero unsupported ops).
- **Evaluation Metric**: Official competition formula:
  $$S = 0.70 \cdot (\text{mAP50} \times 100) + 0.30 \cdot (\text{Accuracy} \times 100)$$

---

## 3. Architecture & Mathematical Formulations

### 3.1 Masked Bounding Box Loss
Tại mỗi cell $(g_y, g_x)$ trên lưới $20 \times 20$, ground truth phân loại $Y_{\text{cls}} \in \mathbb{R}^{20 \times 20 \times 8}$.
Vật thể tồn tại khi $Y_{\text{cls}}[g_y, g_x, 0] = 0$ (nghĩa là không phải nền).
Ta định nghĩa mask nhị phân:
$$M_{\text{obj}}[g_y, g_x] = 1 - Y_{\text{cls}}[g_y, g_x, 0] \in \{0, 1\}$$
Smooth L1 Box Loss cho mỗi cell:
$$\mathcal{L}_{\text{box\_cell}} = \text{SmoothL1}(Y_{\text{box}} - \hat{Y}_{\text{box}})$$
Box Loss chuẩn hóa trên toàn batch:
$$\mathcal{L}_{\text{box}} = \frac{\sum_{b, y, x} M_{\text{obj}}[b, y, x] \cdot \mathcal{L}_{\text{box\_cell}}[b, y, x]}{\max(1.0, \sum_{b, y, x} M_{\text{obj}}[b, y, x])}$$

### 3.2 Balanced Focal Classification Loss
Với xác suất dự đoán $\hat{p} = \text{softmax}(Z)$:
$$p_t = \sum_{c=0}^7 Y_{\text{cls}}[c] \cdot \hat{p}[c]$$
$$\alpha_t = \begin{cases} 1 - \alpha & \text{nếu } c = 0 \text{ (background)} \\ \alpha \cdot W_c & \text{nếu } c \in [1..7] \text{ (foreground)} \end{cases}$$
$$\mathcal{L}_{\text{cls}} = - \frac{1}{N_{\text{cells}}} \sum_{b, y, x} \alpha_t (1 - p_t)^\gamma \log(p_t + \epsilon)$$

### 3.3 Calibrated Decoder Logic (`infer.py`)
Cho mỗi cell $(g_y, g_x)$:
1. Xác suất nền: $P_{\text{bg}} = \hat{p}[0]$.
2. Lớp tiền cảnh có điểm cao nhất: $c^* = \text{argmax}_{k \in [1..7]}(\hat{p}[k])$, điểm số $S = \hat{p}[c^* + 1]$.
3. Điều kiện chấp nhận box ứng viên:
   $$S \ge \text{conf\_thresh} \quad \text{VÀ} \quad S > P_{\text{bg}}$$
4. Nếu thỏa mãn, giải mã tọa độ và đưa vào danh sách ứng viên NMS:
   $$\hat{c}_x = \frac{g_x + \text{offset}_x}{20} \times 320, \quad \hat{c}_y = \frac{g_y + \text{offset}_y}{20} \times 240$$
   $$\hat{w} = w_{\text{norm}} \times 320, \quad \hat{h} = h_{\text{norm}} \times 240$$

---

## 4. Execution Workflow

1. **Step 1: Loss & Dataset Engineering**
   - Cập nhật `qualifiers/src/detection/losses.py` với `masked_box_loss` và `calibrated_focal_loss`.
   - Tích hợp `balance_and_occlude_dataset.py` vào `QualifierDetectionDataset` trong `dataset.py`.
2. **Step 2: Decoder Calibration**
   - Sửa đổi logic trích xuất bounding box trong `qualifiers/src/detection/infer.py` với background rejection và NMS.
3. **Step 3: Model Retraining & Calibration**
   - Chạy huấn luyện detector trên bộ dữ liệu với loss mới trong `train.py`.
   - Lưu checkpoint tốt nhất `best_weights.keras`.
4. **Step 4: INT8 Quantization Export**
   - Xuất mô hình sang TFLite INT8 qua `qualifiers/src/detection/quantize.py`.
5. **Step 5: End-to-End Evaluation & Verification**
   - Đo lường mAP@0.5 trên `qualifiers/data/val/images` và đối chiếu ground truth `qualifiers/data/val/labels`.
   - Sinh file dự đoán hợp lệ và chạy `validate_submission`.
