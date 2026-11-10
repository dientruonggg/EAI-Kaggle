# Tasks: Robust ESP32-S3 Micro-Scale Detector Calibration

**Input**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/003-micro-detector-mAP/spec.md) | [plan.md](file:///home/bim/Projects/EAI-2026/specs/003-micro-detector-mAP/plan.md) | [checklists.md](file:///home/bim/Projects/EAI-2026/specs/003-micro-detector-mAP/checklists.md)  
**Branch**: `003-micro-detector-mAP`  
**Status**: Implemented & Verified

---

## Phase 1: Loss & Dataset Engineering (P1)

- [x] **TSK-201** [US1]: Implement `decoupled_box_loss` (`masked_smooth_l1_loss`) in `qualifiers/src/detection/losses.py`:
  - Masked out empty background cells using `conf_mask = y_true[..., 4:5]`.
  - Calibrated $\delta = 0.05$ for normalized coordinates in $[0, 1]$.
- [x] **TSK-202** [US1]: Implement `decoupled_cls_loss` and `decoupled_conf_loss` in `qualifiers/src/detection/losses.py`:
  - Injected composite `CLASS_WEIGHTS` combining Error-Rate Multiplier ("Càng sai trọng số càng cao") and VQA Query-Ratio.
  - Applied $\gamma=2.5$ focal modulation to heavily penalize hard/misclassified objects.
- [x] **TSK-203** [US3]: Integrate 2-Stage progressive curriculum in `qualifiers/src/detection/dataset.py`:
  - Phase 1: Clean images for fundamental edge/shape learning.
  - Phase 2: Tabletop occlusion simulation via `apply_random_erasing_occlusion` (15%–45% box area).

---

## Phase 2: 12-Layer MicroEdgeCNN Architecture & Decoupled Head (P1)

- [x] **TSK-204** [US1]: Implement 12-Layer Hierarchical Detector in `qualifiers/src/detection/model.py`:
  - Stage 1 (Layers 1..3): Cạnh & Góc (Low-Level Edges & Corners, 24-32 channels).
  - Stage 2 (Layers 4..7): Hình dạng & Khối hình học (Mid-Level Shapes & Parts, 32-48 channels).
  - Stage 3 (Layers 8..12): Ngữ nghĩa vật thể & FPN Lateral Fusion (Semantics, 64-96 channels).
  - Decoupled Detection Head: `conf_output` (Objectness), `cls_output` (7 classes), `box_output` (Offsets & Size).
  - Total parameters: **50,228 parameters** (well within 200,000 ESP32-S3 limit).

---

## Phase 3: Calibrated Decoder & Visual Inspection (P1)

- [x] **TSK-210** [US2]: Upgrade detection decoder in `qualifiers/src/detection/infer.py`:
  - Decoupled scoring: $\text{Score} = \text{Confidence} \times P(\text{class})$.
  - Added non-maximum suppression (NMS) with strict boundary clipping to $[0, 0, 320, 240]$.
- [x] **TSK-211** [US2]: Build BBox visual inspection harness in `qualifiers/src/detection/visualize_detections.py`:
  - Prints exact True Positives (IoU $\ge 0.50$) vs False Positives per image.

---

## Phase 4: Intent TextCNN Odd-Kernel Upgrade (P1)

- [x] **TSK-225** [US3]: Upgrade TextCNN in `qualifiers/src/intent/text_cnn.py` & `train_text_cnn_robust.py`:
  - Switched from even kernels (2, 3, 4) to symmetric odd kernels **$k \in \{3, 5, 7\}$** for 3-gram, 5-gram, and long-range intent modeling.
  - Global Max-Over-Time Pooling across all 3 filter banks.
  - Re-trained model achieved **100.00% Accuracy** on val, test, and 50/50 real-world stress test cases.
  - Exported Full Integer INT8 TFLite model: **22.52 KB** (14,565 parameters).

---

## Phase 5: Hardware Quantization & Master Pipeline (P1)

- [x] **TSK-230** [US4]: Export detector to Full Integer INT8 TFLite in `qualifiers/src/detection/quantize.py`:
  - Fixed weight loading bug by verifying `best_weights.keras` checkpoint exists before PTQ.
  - Verified INT8 file size $\le 1.5$ MB (Actual: 201.62 KB).
- [x] **TSK-240** [US4]: Automated benchmark evaluation via `qualifiers/src/evaluation.py` & `build_submission.py`:
  - Combined `submission.csv` validated and passing all competition invariants.
