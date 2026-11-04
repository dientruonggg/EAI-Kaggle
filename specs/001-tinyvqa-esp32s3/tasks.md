# Implementation Tasks: Pioneer Edge AI Challenge 2026 - TinyVQA Pipeline

**Branch**: `001-tinyvqa-esp32s3` | **Spec**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/001-tinyvqa-esp32s3/spec.md) | **Plan**: [plan.md](file:///home/bim/Projects/EAI-2026/specs/001-tinyvqa-esp32s3/plan.md)

---

## Phase 1: Setup & Ground Truth Alignment

- [x] **TSK-001** [Setup]: Extract and ratify official rules from `docs/The Le Cuoc Thi EdgeAIChallenge2026 Lan2 - Pioneer.pdf` into `.specify/memory/constitution.md`.
- [x] **TSK-002** [Setup]: Verify 7-class definition (`bottle, cup, phone, book, mouse, box, toy_car`) in `qualifiers/data/classes.txt` and model architecture files.
- [x] **TSK-003** [Setup]: Confirm Python 3.11 environment in `.venv` with TensorFlow, PyTorch, and Keras 3.

---

## Phase 2: User Story 1 - Dual-Task Kaggle Submission & Gating (P1)

- [x] **TSK-010** [US1]: Implement and verify `qualifiers/submit.py` to enforce the dual-task schema:
  - Header: `task_type,sample_id,prediction`
  - `DETECTION` row formatting: `[{"class_id": 0, "score": 0.91, "bbox": [x_min, y_min, x_max, y_max]}]` with escaped CSV quotes (`""`).
  - `INTENT` row formatting: exactly one of `COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`.
- [x] **TSK-011** [US1]: Implement automated pre-submission gating validator in `qualifiers/submit.py` & `qualifiers/build_submission.py`:
  - Verified every `bbox` coordinate is strictly within $[0, 0, 320, 240]$.
  - Assert zero missing or duplicate sample IDs.
  - Test run passed with verified sample submission.
- [x] **TSK-012** [US1]: Verify score calculation according to official formula:
  $$S = 0.70 \cdot (\text{mAP50} \times 100) + 0.30 \cdot (\text{Accuracy} \times 100)$$

---

## Phase 3: User Story 2 - Micro-Scale Object Detection (< 200k params, INT8) (P1)

- [x] **TSK-020** [US2]: Inspect and verify MobileNetV2-BiFPN architecture in `qualifiers/src/detection/model.py`:
  - Parameter count: 94,284 params (strictly $< 200,000$).
  - Output grid: Softmax for 7 classes + 1 background, Sigmoid for bounding box offsets.
- [x] **TSK-021** [US2]: Verify Post-Training Quantization (PTQ) pipeline in `qualifiers/src/detection/quantize.py`:
  - Calibration using representative images from dataset.
  - Enforce `inference_input_type = tf.int8` and `inference_output_type = tf.int8`.
  - Exported model size $\le 1.5$ MB (Actual: $\sim 120$ KB).
- [x] **TSK-022** [US2]: Verify TFLite Micro operator compatibility (Conv2D, DepthwiseConv2D, Add, Reshape, Softmax, Sigmoid).

---

## Phase 4: User Story 3 - On-Device Intent TextCNN (< 1.5 MB, INT8) (P1)

- [x] **TSK-030** [US3]: Validate TextCNN architecture in `qualifiers/src/intent/text_cnn.py`:
  - Parameter count: 11,301 parameters.
  - Vocabulary mapping covering accented and unaccented Vietnamese query tokens.
  - Output classes: exactly 5 (`COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE`).
- [x] **TSK-031** [US3]: Export and quantize TextCNN to INT8 TFLite via `qualifiers/src/intent/export_tflite.py`:
  - Verified model size $\le 1.5$ MB (Actual: $\sim 45$ KB).
  - Accuracy validation on `qualifiers/data/intent_val.csv`.

---

## Phase 5: User Story 4 - On-Device Neuro-Symbolic VQA Engine (P2)

- [x] **TSK-040** [US4]: Standardize question parsing in `qualifiers/src/vqa_engine/question_parser.py`:
  - Extract queried target class and slot entities from natural language query.
  - Flag out-of-scope questions immediately.
- [x] **TSK-041** [US4]: Standardize spatial reasoning in `qualifiers/src/vqa_engine/spatial_reasoner.py`:
  - Bounding box horizontal center $x_c = \frac{x_{min} + x_{max}}{2 \times 320}$.
  - $x_c < 0.35 \rightarrow$ `"ben trai"`; $0.35 \le x_c \le 0.65 \rightarrow$ `"o giua"`; $x_c > 0.65 \rightarrow$ `"ben phai"`.
- [x] **TSK-042** [US4]: Standardize answer formatting in `qualifiers/src/vqa_engine/answer_generator.py`:
  - `COUNT`: ASCII integer string (e.g., `"2"` or `"0"`).
  - `EXIST`: `"Co"` or `"Khong"`.
  - `LIST`: comma-separated `"<count> <class>"` sorted by class_id ascending, or `"Khong co"`.
  - `LOCATION`: `"ben trai"`, `"o giua"`, or `"ben phai"`.
  - `OUT_OF_SCOPE`: `"OOS"`.
- [x] **TSK-043** [US4]: Build mock HTTP POST `/ask` test harness in `qualifiers/src/vqa_engine/mock_ask_server.py` verifying full round-trip execution within $\le 10.0$ seconds.

---

## Phase 6: Convergence & End-to-End Verification

- [x] **TSK-050** [Verify]: Run cross-artifact consistency check (`speckit-analyze`) across `spec.md`, `plan.md`, `tasks.md`.
- [x] **TSK-051** [Verify]: Run test suite on all modules: detection inference, intent classifier, VQA reasoning, and submission builder.
- [x] **TSK-052** [Verify]: Confirm all 19 checklist items in `checklists.md` are satisfied.

---

## Convergence Status: CONVERGED

All 16 tasks defined across Phases 1 through 6 are fully implemented, verified with automated unit tests, and cross-checked against the official Pioneer Edge AI Challenge 2026 rulebook.
