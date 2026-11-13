#!/usr/bin/env python3
"""
calibrate.py
================================================================================
Post-Quantization Calibration & Box Density Optimizer on Validation Split
================================================================================
"""
from pathlib import Path
import sys
CURRENT_PKG_DIR = Path(__file__).resolve().parent
if str(CURRENT_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_PKG_DIR))


import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtnet_max import ORIG_W, ORIG_H, IMG_W, IMG_H, NUM_CLASSES, CLASS_NAMES
from evaluation import decode_batch_np, evaluate_map


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

        f_norm = img.astype(np.float32) / 255.0
        q_in = np.round(f_norm / in_scale + in_zp).clip(-128, 127).astype(np.int8)[np.newaxis, ...]

        interpreter.set_tensor(input_details["index"], q_in)
        interpreter.invoke()

        q_cls = interpreter.get_tensor(output_details[cls_idx]["index"])
        q_box = interpreter.get_tensor(output_details[box_idx]["index"])

        f_cls = (q_cls.astype(np.float32) - cls_zp) * cls_scale
        f_box = (q_box.astype(np.float32) - box_zp) * box_scale

        raw_cls_list.append(f_cls.transpose(0, 3, 1, 2))
        raw_box_list.append(f_box.transpose(0, 3, 1, 2))

    print("\n" + "=" * 80)
    print("   NMS IoU & THRESHOLD & BOX DENSITY CALIBRATION SWEEP")
    print("=" * 80)

    best_overall_map = 0.0
    best_iou_thr = 0.65
    best_score_thr = 0.030
    best_aps = {}

    for iou_thr in [0.55, 0.60, 0.65, 0.70]:
        for s_thr in [0.001, 0.010, 0.020, 0.030, 0.040, 0.050]:
            all_dets = []
            for c_f, b_f in zip(raw_cls_list, raw_box_list):
                d = decode_batch_np(c_f, b_f, score_thr=s_thr, nms_iou=iou_thr, max_det=100)
                all_dets.append(d[0])

            m_val, aps_val = evaluate_map(all_dets, gts)
            total_b = sum(len(x) for x in all_dets)
            avg_b = total_b / len(all_dets)
            print(f"NMS IoU: {iou_thr:.2f} | Score Thr: {s_thr:.3f} -> Val mAP50: {m_val*100:.2f}% (Boxes/Img: {avg_b:4.1f})")
            if m_val > best_overall_map:
                best_overall_map = m_val
                best_iou_thr = iou_thr
                best_score_thr = s_thr
                best_aps = aps_val

    print("\n" + "=" * 80)
    print(f"[✔] Best Quantized INT8 mAP50: {best_overall_map*100:.2f}%")
    print(f"    Optimal NMS IoU: {best_iou_thr}")
    print(f"    Optimal Score Threshold: {best_score_thr}")
    per_cls_str = " | ".join([f"{CLASS_NAMES[c]}: {best_aps.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)])
    print(f"    Per-class AP50: {per_cls_str}")
    print("=" * 80)

    calib_result = {
        "best_mAP50": best_overall_map,
        "optimal_nms_iou": best_iou_thr,
        "optimal_score_thr": best_score_thr,
        "per_class_ap50": {CLASS_NAMES[c]: best_aps.get(c, 0.0) for c in range(NUM_CLASSES)},
        "tflite_path": str(tflite_path)
    }

    with open(output_json_path, "w") as f:
        json.dump(calib_result, f, indent=2)
    print(f"[✔] Calibration metadata saved to: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate INT8 TFLite model on validation data")
    parser.add_argument("--tflite", type=str, default="qualifiers/weights/dtnet_nwd_tal_full6k/dtnet_nwd_tal_full6k_int8.tflite",
                        help="Path to INT8 .tflite model")
    parser.add_argument("--val_dir", type=str, default="qualifiers/data/val",
                        help="Path to validation directory")
    parser.add_argument("--output", type=str, default="dtnet_nwd_tal_full6k/calibrated_thresholds.json",
                        help="Output JSON path")
    args = parser.parse_args()

    run_calibration(args.tflite, args.val_dir, args.output)


if __name__ == "__main__":
    main()
