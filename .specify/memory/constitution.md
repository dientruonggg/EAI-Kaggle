<!--
Sync Impact Report:
- Version change: 1.0.0 -> 1.1.0 (MINOR: Ratified official rules from Pioneer Edge AI Challenge 2026 Lan 2 PDF)
- Modified principles:
  - Principle I: Formally updated to 7 competition classes (bottle, cup, phone, book, mouse, box, toy_car) and 320x240 source resolution.
  - Principle II: Replaced informal text with Kaggle composite score formula S = 0.70 * mAP50 + 0.30 * Accuracy and single submission.csv schema.
  - Principle III: Standardized TFLite Micro INT8 constraints for both Detection (< 200k params, <= 1.5MB) and Intent TextCNN (<= 1.5MB).
  - Principle IV: Added 5 specific Intent types (COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE) and response formatting rules.
  - Principle V: Added on-device HTTP POST /ask endpoint specification and 20-query final evaluation criteria (COUNT/EXIST 35%, LIST 25%, LOCATION 30%, OOS 10%).
  - Principle VI: Specified max response latency <= 10.0 seconds per query on ESP32-S3 N16R8 + OV7670.
- Added sections:
  - Section II: Kaggle Submission Schema & Scoring Mechanics
  - Section III: On-Device ESP-IDF & TFLM Deployment Specifications
- Removed sections:
  - Deprecated vague references to generic COCO classes or informal box count sweet spots.
-->

# Edge AI Challenge 2026 Constitution
<!-- Official Project Constitution - Spec Kit SDD Governance -->
<!-- Ratified directly against 'docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf' -->

## Core Principles

### I. Target Microcontroller & Sensor Constraints (ESP32-S3 N16R8 + OV7670)
- **Target Chipset**: ESP32-S3 (Xtensa dual-core 32-bit LX7 @ 240MHz, 512KB SRAM, 8MB Octal PSRAM, 16MB Flash).
- **Vision Sensor**: Omnivision OV7670 Camera module.
- **Source Resolution**: $320 \times 240$ pixels (QVGA).
- **Detection Model Input Resolution**: Each dimension $\le 320 \times 240$, recommended $96 \times 96$, $128 \times 128$, or $160 \times 160$.
- **Detection Model Parameters**: Strictly **$< 200,000$ parameters** (Hard limit).
- **Model Flash Footprint**: Detection model $\le 1.5$ MB; Intent model $\le 1.5$ MB.
- **Quantization Standard**: Full Integer INT8 quantization (`inference_input_type: INT8`, `inference_output_type: INT8`) compatible with TensorFlow Lite for Microcontrollers (TFLM).
- **Operator Invariant**: Zero unsupported TFLM operators (no custom CUDA ops, dilated convolutions with non-unit strides, or large multi-head attention blocks).
- **Single Inference Constraint**: In the final evaluation on ESP32-S3, exactly ONE detection model is run once per image; **NO ensemble** and **NO Test-Time Augmentation (TTA)** are permitted.

### II. Official Label Space & Intent Taxonomy
- **Object Detection (7 Classes)**:
  - `0: bottle` (Chai)
  - `1: cup` (Cốc)
  - `2: phone` (Điện thoại)
  - `3: book` (Sách)
  - `4: mouse` (Chuột máy tính)
  - `5: box` (Hộp)
  - `6: toy_car` (Ô tô đồ chơi)
- **Question Intent Taxonomy (5 Classes - TextCNN Required)**:
  - `COUNT`: Count instances (e.g. "Có bao nhiêu chai?" $\rightarrow$ integer answer string `"2"`, or `"0"` if none).
  - `EXIST`: Binary existence query (e.g. "Có điện thoại không?" $\rightarrow$ `"Co"` or `"Khong"`).
  - `LIST`: List all detected classes (e.g. "Trong ảnh có gì?" $\rightarrow$ `"2 bottle, 1 cup"`, or `"Khong co"` if empty).
  - `LOCATION`: Spatial position of target object (e.g. "Điện thoại ở đâu?" $\rightarrow$ `"ben trai"`, `"o giua"`, or `"ben phai"` based on bounding box center).
  - `OUT_OF_SCOPE`: Irrelevant query (e.g. "Thời tiết hôm nay thế nào?" $\rightarrow$ `"OOS"`).
- **Intent Classifier Constraint**: Intent classification MUST be performed by an INT8 machine learning model (TextCNN); purely rule-based or dictionary-only replacement is strictly prohibited.

### III. Kaggle Submission Schema & Scoring Mechanics
- **Single Submission File**: Exactly one `submission.csv` containing both `DETECTION` and `INTENT` task rows.
  ```csv
  task_type,sample_id,prediction
  DETECTION,img_0001,"[{""class_id"":0,""score"":0.91,""bbox"":[12,40,98,220]}]"
  DETECTION,img_0002,"[]"
  INTENT,q_0001,COUNT
  INTENT,q_0002,LOCATION
  ```
- **Coordinate Conventions**: `bbox` is `[x_min, y_min, x_max, y_max]` relative to original $320 \times 240$ image space.
- **Max Predictions**: At most 100 bounding boxes per image; empty scene must output `[]`.
- **Score Formula**:
  $$S = 0.70 \times (\text{mAP50} \times 100) + 0.30 \times (\text{Accuracy} \times 100)$$
  Tie-breaker: higher mAP50 ranks above; if equal, higher Intent Accuracy wins.
- **Fair Play & Integrity**: Max 4 submissions per 24h. No manual test labelling, no pseudo-labeling from test set, no test data snooping.

### IV. On-Device Final Round Protocol (ESP32-S3 Firmware)
- **Communication Protocol**: HTTP Server running on ESP32-S3 over Wi-Fi AP / local subnet.
- **Query Endpoint**: `POST /ask` with JSON payload `{"q": "<question_text>"}`.
- **Response Format**: JSON response `{"answer": "<answer_string>"}`.
- **Latency Ceiling**: Strictly **$\le 10.0$ seconds** per query turn from receiving HTTP request to completing response.
- **Scoring Weights in Finals (20 test scenarios)**:
  - `COUNT` & `EXIST`: 35% of final score.
  - `LIST`: 25% of final score (standardized class names and quantities).
  - `LOCATION`: 30% of final score (centroid horizontal region: left / center / right).
  - `OUT_OF_SCOPE`: 10% of final score (correctly returning `"OOS"`).
- **Zero Offload Invariant**: 100% of inference must happen on-chip. Any offload to server/phone results in immediate disqualification.

### V. Code Modification & Invariant Guardrails
- **Never alter working code**: Preserve 100% logic, function names, imports, and parameters when synthesizing or refactoring validated scripts.
- **Diff Verification First**: Always verify diffs before running or committing.
- **Single Source of Truth**: All documentation and specifications must align with `docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf`.

---

## Governance & Quality Gates

1. **Constitution Authority**: This Constitution supersedes all ad-hoc guidelines and is the supreme project law.
2. **Amendments**: Amendments require updating this document with an impact report and verified compatibility.
3. **Automated Pipeline**: Any new feature or module must run through Spec Kit SDD skills (`/speckit-*`).

**Version**: 1.1.0 | **Ratified**: 2026-11-04 | **Last Amended**: 2026-11-04
