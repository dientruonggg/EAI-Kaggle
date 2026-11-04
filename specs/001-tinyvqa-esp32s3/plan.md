# Implementation Plan: Pioneer Edge AI Challenge 2026 - TinyVQA Pipeline

**Branch**: `001-tinyvqa-esp32s3` | **Date**: 2026-11-04 | **Spec**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/001-tinyvqa-esp32s3/spec.md)

**Input**: Official Rulebook (`docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf`) and Specification (`specs/001-tinyvqa-esp32s3/spec.md`).

---

## 1. Summary

This plan outlines the architecture and phased implementation to compete and qualify in Edge AI Challenge 2026:
1. **Kaggle Phase (Vòng bảng)**: A high-accuracy dual-task prediction pipeline delivering $S = 0.70 \cdot mAP50 + 0.30 \cdot Accuracy$ formatted as a valid `submission.csv` across 7 classes (`bottle, cup, phone, book, mouse, box, toy_car`) and 5 intents (`COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`).
2. **ESP32-S3 Firmware Phase (Vòng chung kết)**: A standalone ESP-IDF C/C++ project embedding quantized INT8 detection (MobileNetV2-BiFPN, 94,284 params) and INT8 TextCNN (11,301 params) behind an HTTP POST `/ask` endpoint, executing inference and deterministic reasoning within $\le 10$ seconds on live OV7670 camera frames.

---

## 2. Technical Context

- **Languages**: Python 3.11 (Training, Evaluation, Submission, Quantization), C/C++ (ESP-IDF v5.x Firmware).
- **Core ML Frameworks**: TensorFlow 2.16+ / Keras 3, PyTorch 2.x, TensorFlow Lite Micro (TFLM).
- **Target Microcontroller**: ESP32-S3 N16R8 (Xtensa dual-core LX7 @ 240MHz, 512KB SRAM, 8MB Octal PSRAM, 16MB Flash).
- **Camera Sensor**: OmniVision OV7670 (SCCB / DVP interface, $320 \times 240$ QVGA raw).
- **Embedded Budgets**:
  - Detection Model: 94,284 params ($< 200,000$ limit), file size $\sim 120$ KB ($\le 1.5$ MB limit).
  - Intent Model: 11,301 params, file size $\sim 45$ KB ($\le 1.5$ MB limit).
  - Latency: Full query cycle $\le 10.0$ seconds.
- **Testing**: `pytest`, synthetic test benches, submission schema validator, mock HTTP `/ask` server.

---

## 3. Constitution Gate Verification

| Constitution Gate | Invariant Requirement | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Param Limit** | $< 200,000$ parameters | ✅ PASS | `build_mobilenet_bbox_model()` has 94,284 params; TextCNN has 11,301 params. |
| **Model Flash** | $\le 1.5$ MB per model | ✅ PASS | Detection TFLite $\approx 120$ KB; TextCNN TFLite $\approx 45$ KB. |
| **Quantization** | Full Integer INT8 PTQ | ✅ PASS | Post-Training Quantization with $\ge 100$ images; int8 input/output tensors. |
| **Kaggle Schema** | `task_type,sample_id,prediction` | ✅ PASS | Verified format in `qualifiers/submit.py` with escaped JSON quotes. |
| **No Server Offload** | 100% on-chip inference | ✅ PASS | Embedded TFLM runtime executing locally on ESP32-S3 LX7 core. |
| **Single Model Finals**| No ensemble/TTA on MCU | ✅ PASS | Single model forward pass on embedded firmware. |

---

## 4. Architectural Components

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Kaggle & Training Pipeline                      │
│                                                                        │
│   Images (320x240) ──► MobileNetV2-BiFPN (INT8, 94k) ──► Bounding Boxes│
│                                                                        │
│   Questions (VN)   ──► TextCNN INT8 (11k params)     ──► Intent Label  │
│                                                                        │
│                                          │                             │
│                                          ▼                             │
│                     [ Schema & Boundary Validator ]                    │
│                                          │                             │
│                                          ▼                             │
│                              submission.csv (Kaggle)                   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                 ESP32-S3 On-Device Firmware Architecture               │
│                                                                        │
│   HTTP POST /ask ──► [ Intent Parser ] (TextCNN INT8)                  │
│                                │ (Intent & Slot)                       │
│                                ▼                                       │
│   OV7670 Frame   ──► [ Vision Perception ] (MobileNetV2 INT8)          │
│                                │ (7-Class BBoxes)                      │
│                                ▼                                       │
│                      [ Symbolic Reasoning ]                            │
│                        - COUNT (int string)                            │
│                        - EXIST ("Co" / "Khong")                        │
│                        - LIST ("<n> <class>, ...")                     │
│                        - LOCATION ("ben trai" / "o giua" / "ben phai") │
│                        - OUT_OF_SCOPE ("OOS")                          │
│                                │                                       │
│                                ▼                                       │
│                      {"answer": "<result>"} (<= 10s)                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase-by-Phase Roadmap

### Phase 0: Research & Ground-Truth Verification
- Verify official class order from `qualifiers/data/classes.txt` (`bottle, cup, phone, book, mouse, box, toy_car`).
- Verify intent dictionary and TextCNN accuracy on `qualifiers/data/intent_val.csv`.
- Verify TFLite INT8 conversion compatibility with TFLite Micro runtime.

### Phase 1: Kaggle Remote Workflow & Submission Gating
- Ensure `qualifiers/submit.py` enforces Kaggle scoring formula $S = 0.70 \cdot mAP50 + 0.30 \cdot Accuracy$.
- Validate that all detections on $320 \times 240$ images clip bounding boxes to $[0, 0, 320, 240]$ with valid class IDs $0..6$.
- Validate that intent predictions strictly adhere to the 5 valid labels.

### Phase 2: On-Device Neuro-Symbolic VQA Engine
- Standardize the question parser and slot extractor in `qualifiers/src/vqa_engine/`.
- Validate spatial reasoning thresholds: $x_c < 0.35$ ("ben trai"), $0.35 \le x_c \le 0.65$ ("o giua"), $x_c > 0.65$ ("ben phai").
- Validate answer formatting for all 5 intent types against the test rubric.
- Create mock test suite running 20 representative queries through the full pipeline to guarantee $< 10.0$ s response latency.
