#!/usr/bin/env python3
"""
calibrate.py
================================================================================
Post-Quantization Single-Model Threshold Calibration on Validation Split
--------------------------------------------------------------------------------
1. Runs direct INT8 TFLite inference using TensorFlow Lite Interpreter.
2. Accurately decodes INT8 input/output quantization scale and zero-point.
3. Sweeps confidence threshold and NMS IoU to identify the optimal peak mAP50.
4. Generates calibrated_thresholds.json for production deployment and Kaggle submission.
5. Strictly adheres to single-model forward pass (No ensemble, no TTA).
================================================================================
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtnet_from_scratch_ultra.dtnet_max import ORIG_W, ORIG_H, IMG_W, IMG_H, NUM_CLASSES, CLASS_NAMES
from dtnet_from_scratch_ultra.evaluation import decode_batch_np, evaluate_map


def run_calibration(tflite_path, val_data_dir, output_json_path):
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    val_data_dir = Path(val_data_dir)
    img_dir = val_data_dir / "images" if (val_data_dir / "images").exists() else val_data_dir
    lbl_dir = val_data_dir / "labels" if (val_data_dir / "labels").exists() else val_data_dir.parent / "labels"

    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    assert len(img_paths) > 0, f"No validation images found in {img_dir}"

    gts = []
    for p in img_paths:
        stem = Path(p).stem
        lbl_p = lbl_dir / f"{stem}.txt"
        rows = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2.0) * ORIG_W
                        y1 = (cy - h / 2.0) * ORIG_H
                        x2 = (cx + w / 2.0) * ORIG_W
                        y2 = (cy + h / 2.0) * ORIG_H
                        rows.append([c, x1, y1, x2, y2])
        gts.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))

    print(f"[*] Loaded {len(img_paths)} validation images and labels.")
    print(f"[*] Initializing TFLite Interpreter: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    in_scale, in_zp = input_details["quantization"]
    cls_idx = 0 if output_details[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx = 1 - cls_idx

    cls_scale, cls_zp = output_details[cls_idx]["quantization"]
    box_scale, box_zp = output_details[box_idx]["quantization"]

    print(f"[Quantization Metadata] Input: scale={in_scale}, zp={in_zp}")
    print(f"[Quantization Metadata] Cls Output: scale={cls_scale}, zp={cls_zp}")
    print(f"[Quantization Metadata] Box Output: scale={box_scale}, zp={box_zp}")

    # Run inference across validation images
    print("[*] Running INT8 inference across validation set...")
    raw_cls_list, raw_box_list = [], []
    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        # Quantize float [0, 1] to int8
        f_norm = img.astype(np.float32) / 255.0
        q_in = np.round(f_norm / in_scale + in_zp).clip(-128, 127).astype(np.int8)[np.newaxis, ...]

        interpreter.set_tensor(input_details["index"], q_in)
        interpreter.invoke()

        q_cls = interpreter.get_tensor(output_details[cls_idx]["index"])
        q_box = interpreter.get_tensor(output_details[box_idx]["index"])

        # Dequantize to float logits
        f_cls = (q_cls.astype(np.float32) - cls_zp) * cls_scale
        f_box = (q_box.astype(np.float32) - box_zp) * box_scale

        # Transpose from (B, gh, gw, C) to (B, C, gh, gw)
        raw_cls_list.append(f_cls.transpose(0, 3, 1, 2))
        raw_box_list.append(f_box.transpose(0, 3, 1, 2))

    all_cls = np.concatenate(raw_cls_list, axis=0)
    all_box = np.concatenate(raw_box_list, axis=0)

    # Threshold Sweeper (Includes full-recall thresholds 0.001..0.05 for VOC mAP curve integration)
    print("\n[*] Sweeping confidence & NMS thresholds to find optimal sweet spot...")
    conf_candidates = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    nms_candidates = [0.50, 0.55, 0.58, 0.60, 0.62]

    best_map50 = 0.0
    best_conf = 0.25
    best_nms = 0.58
    best_per_class = {}

    for nms_thr in nms_candidates:
        for conf_thr in conf_candidates:
            dets = decode_batch_np(all_cls, all_box, score_thr=conf_thr, nms_iou=nms_thr)
            m_ap, per_c = evaluate_map(dets, gts)
            if m_ap > best_map50:
                best_map50 = m_ap
                best_conf = conf_thr
                best_nms = nms_thr
                best_per_class = per_c
                print(f"  -> [Candidate] conf={conf_thr:.2f}, nms={nms_thr:.2f} | mAP50: {m_ap*100:.2f}%")

    print("\n" + "=" * 60)
    print(f"[★ CALIBRATION WINNER] Peak INT8 mAP50: {best_map50*100:.2f}%")
    print(f"    Optimal Confidence Threshold: {best_conf}")
    print(f"    Optimal NMS IoU Threshold:    {best_nms}")
    print("    Per-Class AP:")
    for c in range(NUM_CLASSES):
        print(f"      - {CLASS_NAMES[c]:<10}: {best_per_class.get(c, 0)*100:.2f}%")
    print("=" * 60)

    calibration_result = {
        "model": str(tflite_path),
        "val_mAP50": best_map50,
        "optimal_conf_thr": best_conf,
        "optimal_nms_thr": best_nms,
        "per_class_ap": {CLASS_NAMES[c]: best_per_class.get(c, 0) for c in range(NUM_CLASSES)}
    }

    with open(output_json_path, "w") as f:
        json.dump(calibration_result, f, indent=2)
    print(f"[✔] Calibration parameters saved to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Single-Model Thresholds for INT8 TFLite")
    parser.add_argument("--tflite", type=str, required=True, help="Path to .tflite model")
    parser.add_argument("--val_dir", type=str, default="qualifiers/data/val", help="Path to validation directory")
    parser.add_argument("--output", type=str, default="dtnet_from_scratch_ultra/calibrated_thresholds.json", help="Output JSON path")
    args = parser.parse_args()

    run_calibration(args.tflite, args.val_dir, args.output)


if __name__ == "__main__":
    main()
