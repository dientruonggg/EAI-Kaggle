---
name: edge-ai-competition
description: >-
  Runbook for Edge AI competition workflows: Kaggle T4x2 remote training,
  INT8 post-training quantization, WBF ensemble calibration with coordinate descent,
  and pre-submission density gating for ESP32-S3 deployment.
---

# Edge AI Competition Runbook

## Overview
This skill operationalizes the end-to-end pipeline for microcontroller-bounded object detection competitions (specifically targeting ESP32-S3 N16R8, TensorFlow Lite Micro, and Kaggle evaluations).

---

## 1. Remote Training on Kaggle (T4x2)
When training on large datasets (e.g. 11,000 images), avoid slow local CPU training by executing on dual T4 GPUs on Kaggle:

1. **Verify Kernel Metadata**:
   Ensure `warmup/kaggle_runner/kernel-metadata.json` points to the active competition slug:
   ```json
   {
     "id": "dientruong/edge-ai-2026-t4-trainer",
     "title": "Edge AI 2026 T4x2 High-Speed Trainer",
     "code_file": "kaggle_train_pipeline.py",
     "language": "python",
     "kernel_type": "script",
     "is_private": "true",
     "enable_gpu": "true",
     "enable_internet": "true",
     "competition_sources": ["edge-ai-challenge-2026-lan-2"]
   }
   ```
2. **Execute 1-Click Push, Monitor & Pull**:
   ```bash
   python warmup/kaggle_runner/run_remote_train.py auto
   ```
3. The runner automatically pushes `kaggle_train_pipeline.py` (which runs `MirroredStrategy` across 2x T4 GPUs, applies EIoU + Focal Loss, and exports INT8 quantized `.tflite`), monitors the run, and pulls weights to `warmup/weights/kaggle_t4/`.

---

## 2. INT8 Post-Training Quantization (PTQ) Standard
For Keras 3 models targeting ESP32-S3:
> [!WARNING]
> Do NOT use `tfmot` QAT on Keras 3 checkpoints; partial layer matching uninitializes ~30% of layers. Always use native PTQ.

```python
import tensorflow as tf

def rep_dataset_gen():
    for img in sample_imgs[:100]:
        yield [img[None, ...].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)
```

---

## 3. Multi-Scale WBF & Coordinate Descent Calibration
1. **Pyramid Models**: Combine distinct spatial resolutions (128x128 with 16x16 grid, 144x144 with 18x18 grid, 160x160 with 20x20 grid).
2. **Per-Class Threshold Calibration**:
   Run coordinate descent on the local validation split (`data/val_split.txt`) using `warmup/tune_per_class_conf.py` to optimize confidence per class while holding IoU threshold at 0.28.
3. **Geometric Prior Weighting**: Apply higher alpha weights to high-resolution feature maps for small objects (e.g. traffic signs) and balanced weights for large vehicles.

---

## 4. Pre-Submission Sanity Gate
Before uploading any CSV to Kaggle via `/home/bim/.local/bin/kaggle competitions submit`:
1. **Format Check**:
   Confirm columns: `image_id,prediction_string`
2. **Intent Score Check**:
   Verify intent predictions are merged from `submissions/intent_preds.csv` (ensuring 30.00/30.00 points).
3. **Box Count Sanity Check**:
   - For 60 test images, verify total predicted boxes $N$ satisfies:
     $$2,600 \le N \le 2,800 \quad (43 - 47\text{ boxes/image})$$
   - $N > 3,000$ indicates excessive false positives on pavement/background.
   - $N < 2,500$ indicates missed true positives.
