# Feature Specification: Pioneer Edge AI Challenge 2026 - TinyVQA On-Device & Kaggle Pipeline

**Feature Branch**: `001-tinyvqa-esp32s3`

**Created**: 2026-11-04

**Status**: Specified / Ready for Clarification & Planning

**Input**: Official Competition Rulebook: `docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf`

---

## 1. Executive Summary

This specification formalizes the end-to-end technical requirements for competing in **Edge AI Challenge 2026 (Lần 2)** organized by Pioneer Club & Faculty of Electronics and AI (DUT). The system encompasses two phases:
1. **Online Phase (Kaggle Leaderboard)**: Generating a dual-task prediction file `submission.csv` containing detection bounding boxes on $320 \times 240$ images across 7 classes (`bottle, cup, phone, book, mouse, box, toy_car`) and intent classification across 5 intents (`COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`), scored via $S = 0.70 \cdot \text{mAP50} + 0.30 \cdot \text{Accuracy}$.
2. **Finals Phase (On-Device ESP32-S3 + OV7670)**: Running a 100% on-chip neuro-symbolic pipeline behind an HTTP POST `/ask` endpoint that processes live camera frames and Vietnamese natural language questions, returning deterministic formatted answers within $\le 10$ seconds.

---

## 2. User Scenarios & Acceptance Criteria *(Mandatory)*

### User Story 1 - Dual-Task Kaggle Submission Generation (Priority: P1)

As a competition participant, I need a unified prediction pipeline that reads test images and question queries, runs the trained Object Detection and TextCNN models, and produces a single, fully compliant `submission.csv` matching the official Kaggle schema.

**Why this priority**: Without a schema-compliant submission, the team cannot score on the Kaggle Leaderboard or qualify for the finals.

**Independent Test**: Execute `submit.py` against test sets and run a schema validator verifying row count, task types, JSON escaping, coordinate bounds, and intent labels.

**Acceptance Scenarios**:

1. **Given** test images in $320 \times 240$ and test questions `q_id`, **When** the submission generator runs, **Then** it produces `submission.csv` with exactly 3 columns: `task_type,sample_id,prediction`.
2. **Given** a `DETECTION` row, **When** parsed, **Then** `prediction` is a valid JSON array string containing up to 100 detections with `class_id` $\in \{0..6\}$, `score` $\in [0, 1]$, and `bbox` formatted as `[x_min, y_min, x_max, y_max]` clipped to image boundaries $[0, 0, 320, 240]$ with inner quotes escaped (`""`).
3. **Given** an empty detection scene, **When** exported, **Then** `prediction` is exactly `"[]"`.
4. **Given** an `INTENT` row, **When** evaluated, **Then** `prediction` is strictly one of `{"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"}`.

---

### User Story 2 - Micro-Scale Object Detection (< 200k params, Full INT8) (Priority: P1)

As an embedded ML engineer, I need an object detection model constrained to $< 200,000$ parameters and $\le 1.5$ MB flash footprint that accurately detects the 7 competition classes on downscaled frames ($96 \times 96$, $128 \times 128$, or $160 \times 160$).

**Why this priority**: Hard hardware constraint; any model exceeding 200k parameters or using unsupported TFLM operators will be disqualified at the finals hardware verification stage.

**Independent Test**: Compile and quantize the model to TFLite INT8; verify parameter count via graph inspector and evaluate mAP50 on local validation split.

**Acceptance Scenarios**:

1. **Given** the trained model graph, **When** counting trainable and frozen parameters, **Then** the total parameter count is strictly $< 200,000$.
2. **Given** Post-Training Quantization (PTQ) with $\ge 100$ representative calibration images, **When** exported to `.tflite`, **Then** file size is $\le 1.5$ MB with `int8` input and `int8` output tensors.
3. **Given** inference on ESP32-S3 LX7 core @ 240MHz, **When** benchmarked, **Then** inference time is $< 2.5$ seconds per frame and only native TFLite Micro operators are invoked.
4. **Given** final round constraints, **When** deploying to ESP32-S3, **Then** exactly ONE model is run without ensemble or TTA.

---

### User Story 3 - On-Device Intent TextCNN (< 1.5 MB, Full INT8) (Priority: P1)

As an embedded ML engineer, I need a lightweight TextCNN model that tokenizes Vietnamese question strings and classifies them into the 5 target intents without relying purely on dictionary keywords.

**Why this priority**: Rule III.3 and V mandate an ML-based TextCNN intent classifier with Accuracy contributing 30% of the Kaggle score.

**Independent Test**: Train TextCNN on provided intent dataset, quantize to INT8 TFLite, and verify $\ge 98\%$ accuracy on test questions.

**Acceptance Scenarios**:

1. **Given** a raw Vietnamese question (e.g., "Co bao nhieu chai trong hinh?"), **When** passed through TextCNN, **Then** it classifies into one of `COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE`.
2. **Given** out-of-domain queries ("Thời tiết hôm nay thế nào?"), **When** classified, **Then** it outputs `OUT_OF_SCOPE`.
3. **Given** the quantized INT8 intent model, **When** measured on disk, **Then** size is $\le 1.5$ MB (typically $< 50$ KB) and inference executes in $< 10$ ms.

---

### User Story 4 - On-Device Neuro-Symbolic Reasoning & HTTP Service (Priority: P2)

As a finals judge or automated test harness, I send an HTTP POST request to `http://<esp32_ip>/ask` with JSON `{"q": "..."}`, and receive a correctly formatted JSON response `{"answer": "..."}` within 10 seconds.

**Why this priority**: Official finals evaluation mechanism; answers failing format or exceeding 10 seconds receive 0 points.

**Independent Test**: Spin up a mock HTTP harness sending 20 test questions to the ESP32-S3 firmware and check answer accuracy, formatting compliance, and round-trip latency.

**Acceptance Scenarios**:

1. **Given** a `COUNT` query for bottles and 2 bottles detected, **When** responding, **Then** output is `{"answer": "2"}` (or `"0"` if none).
2. **Given** an `EXIST` query for phone and at least 1 phone detected, **When** responding, **Then** output is `{"answer": "Co"}` (or `"Khong"` if absent).
3. **Given** a `LIST` query with 2 bottles and 1 cup detected, **When** responding, **Then** output is `{"answer": "2 bottle, 1 cup"}` (or `"Khong co"` if empty).
4. **Given** a `LOCATION` query for a detected object whose bounding box center $x_c$ satisfies $x_c < 0.35$ (left), $0.35 \le x_c \le 0.65$ (center), or $x_c > 0.65$ (right), **When** responding, **Then** output is respectively `{"answer": "ben trai"}`, `{"answer": "o giua"}`, or `{"answer": "ben phai"}`.
5. **Given** an `OUT_OF_SCOPE` query, **When** responding, **Then** output is `{"answer": "OOS"}`.
6. **Given** any query, **When** measuring total elapsed time from socket accept to socket close, **Then** latency is strictly $\le 10.0$ seconds.

---

## 3. Technical Constraints & Specification Boundaries

| Parameter | Constraint Specification |
| :--- | :--- |
| **Detection Classes** | Exactly 7: `0: bottle, 1: cup, 2: phone, 3: book, 4: mouse, 5: box, 6: toy_car` |
| **Intent Classes** | Exactly 5: `COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE` |
| **Detection Params** | $< 200,000$ total parameters |
| **Model Storage** | Detection $\le 1.5$ MB, Intent $\le 1.5$ MB |
| **Hardware** | ESP32-S3 N16R8 (16MB Flash, 8MB Octal PSRAM) + OV7670 Camera |
| **Firmware Framework** | ESP-IDF v5.x + TensorFlow Lite for Microcontrollers |
| **Network Endpoint** | HTTP POST `/ask` on port 80; input `{"q": "..."}`, output `{"answer": "..."}` |
| **Latency Limit** | $\le 10.0$ seconds per transaction |

---

## 4. Quality & Compliance Checklist

- [x] Zero references to non-competition classes (traffic, coco91, etc.).
- [x] Exact Kaggle metric formula implemented ($S = 0.70 \cdot mAP50 + 0.30 \cdot Accuracy$).
- [x] Bounding box convention normalized to $320 \times 240$ coordinates.
- [x] INT8 TFLM operator verification included.


---

## 5. Clarifications & Edge-Case Resolutions *(Resolved via /speckit-clarify)*

1. **Q: How should `LIST` queries format multiple detected classes, and does order matter?**
   - **Resolution**: The evaluation rubric states answers are normalized before comparison. To ensure 100% determinism, our symbolic engine orders detected items by `class_id` ascending (e.g., `0: bottle, 1: cup, ...`) and joins them with comma-space: `"<count1> <class1>, <count2> <class2>"`. If zero objects detected, return exactly `"Khong co"`.
2. **Q: How to handle `LOCATION` when multiple objects of the same class exist in the frame?**
   - **Resolution**: When multiple instances of the queried class exist, select the bounding box with the highest detection confidence score. Calculate its horizontal centroid $x_c = \frac{x_{min} + x_{max}}{2 \times 320}$. If $x_c < 0.35 \rightarrow$ `"ben trai"`; if $0.35 \le x_c \le 0.65 \rightarrow$ `"o giua"`; if $x_c > 0.65 \rightarrow$ `"ben phai"`. If target object is not found, default to `"Khong co"`.
3. **Q: Does the Intent classifier handle both accented and unaccented Vietnamese input?**
   - **Resolution**: Yes. Rulebook examples contain both ("Có điện thoại hay không?" vs "Co bao nhieu chai"). TextCNN tokenizer applies a hybrid vocabulary and lowercase normalization that preserves accents while maintaining identical subword/token embeddings for unaccented variants.
4. **Q: What is the exact behavior for `OUT_OF_SCOPE` questions?**
   - **Resolution**: Any question not asking about the 7 target classes or asking general knowledge / weather questions must return strictly `"OOS"`.
