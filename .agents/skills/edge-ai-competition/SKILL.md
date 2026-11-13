---
name: edge-ai-competition
description: >-
  Runbook for Edge AI Challenge 2026 (Lần 2): 7-class object detection (<200k params)
  + Vietnamese intent TextCNN on ESP32-S3. Covers Kaggle T4x2 remote training,
  full-INT8 post-training quantization for TFLite Micro, single-model threshold
  calibration (no ensemble/TTA), and the pre-submission gate for the
  task_type,sample_id,prediction CSV. Use for any training, export, or submission step.
---

# Edge AI Competition Runbook

Rules live in `AGENTS.md` (source: `Edge AI Challenge 2026 - Training 1.pdf`). Re-read them before changing architecture or submitting.

Key numbers: score = 70 x mAP50 + 30 x intent accuracy; detection <= 200k params, INT8 <= 1.5 MB, input <= 320x240; no ensemble / TTA; 4 submissions / 24 h; finals latency <= 10 s on ESP32-S3.

---

## 1. Remote Training on Kaggle (T4x2)

1. **Credentials**: Kaggle CLI configured on this machine (`~/.kaggle/kaggle.json` or access token). Check with `kaggle competitions list -s edge-ai` or the `kaggle` skill's `doctor`.
2. **Kernel metadata**: `kernel-metadata.json` `id` must be `<your-username>/<kernel-name>`, with `"competition_sources": ["edge-ai-challenge-2026-lan-2"]`, `enable_gpu: true`.
3. **Runner**: `python warmup/kaggle_runner/run_remote_train.py auto` (push -> monitor -> pull). Call `kaggle` from PATH, not a hardcoded `/home/bim/...` path.
4. Pulled weights go to `warmup/weights/kaggle_t4/`; always evaluate locally on `data/val_split.txt` afterwards.
5. Starting a kernel spends weekly GPU hours: confirm with the user first.

---

## 2. INT8 Post-Training Quantization (PTQ) Standard

> [!WARNING]
> Do NOT use `tfmot` QAT on Keras 3 checkpoints; partial layer matching uninitializes ~30% of layers. Use native PTQ.

```python
import numpy as np
import tensorflow as tf

def rep_dataset_gen():
    for img in sample_imgs[:200]:          # 100+ train images, SAME preprocessing as training
        yield [img[None, ...].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open(out_path, "wb").write(tflite_model)
```

After export, verify:
- File size <= 1.5 MB; parameter count of the Keras model < 200,000 (`model.count_params()`).
- Op list contains only TFLM builtins (`tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)`).
- Input/output dtype, scale, zero-point read from the interpreter (never from the file name). For TextCNN check token-ID input dtype.
- FP32 vs INT8 mAP50 / accuracy drop on local validation.

---

## 3. Single-Model Calibration (no ensemble, no TTA)

Ensembles, WBF across models/resolutions, and flip/multi-scale TTA are **banned**. Allowed post-processing on a single forward pass:
1. Decode head outputs -> boxes in original image coordinates.
2. Per-class confidence thresholds and NMS IoU tuned by coordinate descent on `data/val_split.txt` only (e.g. `warmup/tune_per_class_conf.py`, adapted to the 7 classes).
3. Max detections per image cap.
Never tune on, train on, or pseudo-label the hidden test set.

### 3.1. Full-Recall Decoding Invariant (Tránh bẫy Score Truncation)
> [!CAUTION]
> Tuyệt đối KHÔNG lọc bỏ dự đoán bằng ngưỡng score cao (>= 0.03 - 0.07) khi xuất file nộp mAP.
> Việc cắt cụt score sẽ triệt tiêu đuôi Recall của đường cong PR, vứt bỏ các True Positive độ tự tin thấp và trực tiếp làm tụt 0.8% - 1.0% mAP trên Leaderboard.

* **Cấu hình chuẩn khi xuất file nộp**:
  - `score_thr = 0.001` (giữ nguyên dải Recall).
  - `max_det = 100` (giới hạn tối đa 100 box có confidence cao nhất mỗi ảnh).
  - `nms_iou = 0.58 - 0.60` (loại bỏ box trùng lặp).
  - Tổng số box trên 4.000 ảnh test phải đạt khoảng **180.000 - 400.000 box** (45 - 100 box/ảnh). Nếu chỉ có ~50.000 box (~13 box/ảnh), chắc chắn file đang bị lỗi cắt cụt ngưỡng score.

---

## 4. Pre-Submission Gate

Before any `kaggle competitions submit` (user must approve each one; max 4 / 24 h):
1. **Format**: columns exactly `task_type,sample_id,prediction`; matches `sample_submission.csv` IDs and row count; each `sample_id` once.
   - `DETECTION`: `prediction` is a valid JSON list of boxes, class IDs in 0..6 (`bottle, cup, phone, book, mouse, box, toy_car`).
   - `INTENT`: `prediction` in `{COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE}`.
2. **Provenance**: predictions produced by the INT8 `.tflite` that satisfies the constraints; log its path, size, and param count with the CSV. Declare any pretrained weights used.
3. **Box-count sanity**: report total boxes and boxes/image and compare with the validation-calibrated expectation. (The warm-up 2,600-2,800 rule does not apply to this dataset.)
4. **Local score**: report local `70 x mAP50 + 30 x acc` next to the file name.

---

## 5. Target Assignment, Loss & Augmentation Guardrails

1. **Bẫy Nhãn Lờ Đi `-2` (Border Ignore Label Trap)**:
   - Nếu đánh dấu toàn bộ ô viền trong box ngoài tâm là `-2` (trọng số loss nền = 0.0), hơn 40% lưới bị mất giám sát âm tính. Logit nền sẽ trôi lên cao và sinh False Positive hàng loạt ở ngưỡng thấp.
   - **Quy tắc**: Dùng gán nhãn Area-Adaptive gốc (bảo toàn 100% giám sát nền), hoặc nếu làm mịn viền thì chỉ dùng **Soft Border Weight = 0.25**, tuyệt đối không zero-out hoàn toàn (0.0).

2. **Quy tắc Không Xoay Bounding Box (`rotate = 0.0`)**:
   - Xoay box chữ nhật $\pm 20^\circ$ làm diện tích box phình to $1.3\times$, làm loãng nhãn của vật thể nhỏ (mouse, toy_car, phone).
   - **Quy tắc**: Giữ `rotate = 0.0`. Chỉ dùng scale zoom-out `(0.60, 1.25)` để tăng tần suất xuất hiện vật thể nhỏ và đóng mosaic 25-35 epoch cuối.

3. **Phạt Nặng Vật Thể Nhỏ (Small-Object Scale Loss)**:
   - Áp dụng hệ số tỷ lệ nghịch với căn diện tích:
     $$w_{\text{size}} = \text{clip}\left(\frac{24.0}{\sqrt{w \times h}}, 1.0, 2.5\right)$$
   - Áp dụng vào EIoU box loss để ép gradient tập trung chính xác vào vật thể $< 16$ px và $16-24$ px.

4. **Quy tắc Log Huấn luyện & Đánh giá**:
   - Đánh giá validation loss & `val_mAP50` trên từng epoch (`eval_every = 1`).
   - Chỉ in bảng chi tiết AP của toàn bộ 7 lớp (`bottle`, `cup`, `phone`, `book`, `mouse`, `box`, `toy_car`) khi đạt kỷ lục mới (`is_best == True`). Các epoch bình thường chỉ in tóm tắt 1 dòng để log tinh gọn, trực quan.

