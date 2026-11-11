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

---

## 4. Pre-Submission Gate

Before any `kaggle competitions submit` (user must approve each one; max 4 / 24 h):
1. **Format**: columns exactly `task_type,sample_id,prediction`; matches `sample_submission.csv` IDs and row count; each `sample_id` once.
   - `DETECTION`: `prediction` is a valid JSON list of boxes, class IDs in 0..6 (`bottle, cup, phone, book, mouse, box, toy_car`).
   - `INTENT`: `prediction` in `{COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE}`.
2. **Provenance**: predictions produced by the INT8 `.tflite` that satisfies the constraints; log its path, size, and param count with the CSV. Declare any pretrained weights used.
3. **Box-count sanity**: report total boxes and boxes/image and compare with the validation-calibrated expectation. (The warm-up 2,600-2,800 rule does not apply to this dataset.)
4. **Local score**: report local `70 x mAP50 + 30 x acc` next to the file name.
