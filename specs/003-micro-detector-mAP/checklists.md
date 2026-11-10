# Quality & Testing Checklist: Robust ESP32-S3 Micro-Scale Detector Calibration

**Purpose**: Requirements and testing quality checklist for Object Detection calibration and training under ESP32-S3 constraints.  
**Created**: 2026-11-08  
**Feature**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/003-micro-detector-mAP/spec.md)

---

## 1. Loss & Architectural Integrity

- [ ] **CHK-001**: Box regression loss computes gradients strictly on positive object cells (`cls_target[:, :, 0] == 0`), masked out from 95% background cells.
- [ ] **CHK-002**: Classification loss properly downweights easy negative background cells via focal weight $\alpha_t$ and $\gamma=2.0$.
- [ ] **CHK-003**: Model parameter count remains strictly below 200,000 parameters (MobileNetV2-BiFPN: ~94,284).
- [ ] **CHK-004**: Zero unsupported TFLite Micro operators used in architecture (Conv2D, DepthwiseConv2D, Add, UpSampling2D/Reshape, Softmax, Sigmoid).

## 2. Decoding & Post-Processing Calibrations

- [ ] **CHK-010**: `infer.py` explicitly rejects grid cells where background probability $P(\text{bg}) \ge P(\text{class})$.
- [ ] **CHK-011**: NMS threshold set to $0.45$, confidence threshold calibrated between $0.20$ and $0.35$.
- [ ] **CHK-012**: Bounding box coordinates accurately scale from normalized grid cell space back to original $320 \times 240$ frame with clipping to $[0, 0, 320, 240]$.
- [ ] **CHK-013**: Total detected boxes on 24-image validation set drops from 647 boxes down to realistic sweet spot (20 to 120 boxes total, ~1–5 boxes/image).

## 3. Data Augmentation & Imbalance Mitigation

- [ ] **CHK-020**: Tabletop occlusion engine (`balance_and_occlude_dataset.py`) integrated into dataset loading for training.
- [ ] **CHK-021**: Random erasing / cutout masks $15\% - 45\%$ of box area without corrupting original bounding box annotations.
- [ ] **CHK-022**: Minority classes (`mouse`, `box`, `phone`) receive augmented sample frequency or weighted focal loss multiplier.

## 4. Hardware Quantization & Evaluation Convergence

- [ ] **CHK-030**: Exported model quantized to Full Integer INT8 with representative calibration dataset.
- [ ] **CHK-031**: TFLite file size $\le 1.5$ MB (target $\sim 120$ KB).
- [ ] **CHK-032**: Validation mAP@0.5 improves from 0.00% to $\ge 40.0\%$.
- [ ] **CHK-033**: Generated `submission.csv` strictly validated by `submit.py` schema checker with zero formatting errors.
