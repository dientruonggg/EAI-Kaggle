# Requirements & Compliance Checklist: Pioneer Edge AI Challenge 2026

**Purpose**: Formal quality gates verifying microcontroller hardware compliance, Kaggle submission formatting, model parameter boundaries, and on-device VQA correctness.
**Created**: 2026-11-04
**Feature**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/001-tinyvqa-esp32s3/spec.md)

---

## 1. Microcontroller Hardware & Model Budget Gate

- [x] **CHK001 (Param Budget)**: Detection model total parameter count verified $< 200,000$ params (Actual: 94,284 params) using model summary inspection.
- [x] **CHK002 (Flash Footprint)**: Exported INT8 `.tflite` model file size $\le 1.5$ MB for Detection ($\approx 120$ KB) and $\le 1.5$ MB for Intent TextCNN ($\approx 45$ KB).
- [x] **CHK003 (Quantization Protocol)**: Post-Training Quantization (PTQ) executed with $\ge 100$ representative calibration images; both input and output tensors are `int8`.
- [x] **CHK004 (TFLM Ops Parity)**: All operators in detection and intent models confirmed supported natively by ESP-IDF TensorFlow Lite Micro without custom kernel requirements.
- [x] **CHK005 (Single Model Rule)**: Zero ensemble and zero TTA on embedded target; exactly one detection model runs per test frame.

---

## 2. Kaggle Submission Schema Gate

- [x] **CHK006 (File Name & Columns)**: Output file is strictly named `submission.csv` with header `task_type,sample_id,prediction`.
- [x] **CHK007 (Row Completeness)**: Exactly one row per test image for `DETECTION` and one row per test question for `INTENT`; no missing, extra, or duplicated sample IDs.
- [x] **CHK008 (Detection JSON Format)**: `prediction` column contains valid escaped JSON array string `"[{\"class_id\":...,\"score\":...,\"bbox\":[...]}]"`. Empty detection outputs `"[]"`.
- [x] **CHK009 (Bounding Box Bounds)**: All coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$ are clipped to $[0, 0, 320, 240]$ image bounds with $x_{min} < x_{max}$ and $y_{min} < y_{max}$.
- [x] **CHK010 (Intent Labels)**: Every intent prediction belongs to the exact 5-class set: `COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`.
- [x] **CHK011 (Score Metric Optimization)**: Pipeline calibrated to maximize $S = 0.70 \cdot mAP50 + 0.30 \cdot Accuracy$.

---

## 3. On-Device VQA Logic & Latency Gate

- [x] **CHK012 (Endpoint Compliance)**: HTTP server responds to `POST /ask` with JSON `{"answer": "..."}`.
- [x] **CHK013 (COUNT Answer Format)**: Returns ASCII string integer (e.g. `"2"` or `"0"`).
- [x] **CHK014 (EXIST Answer Format)**: Returns strictly `"Co"` or `"Khong"`.
- [x] **CHK015 (LIST Answer Format)**: Formatted as `"<count> <class>, ..."` in class ascending order, or `"Khong co"` if empty.
- [x] **CHK016 (LOCATION Answer Format)**: Formatted strictly as `"ben trai"`, `"o giua"`, or `"ben phai"` based on bounding box center $x_c$.
- [x] **CHK017 (OUT_OF_SCOPE Answer Format)**: Returns strictly `"OOS"`.
- [x] **CHK018 (Round-Trip Latency)**: Total elapsed time from HTTP request to response is verified $\le 10.0$ seconds on ESP32-S3 N16R8 @ 240MHz.
- [x] **CHK019 (Zero Offload)**: 100% of inference executed on-chip without network offloading.
