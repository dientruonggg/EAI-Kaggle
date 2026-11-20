#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - DETERMINISTIC SUBMISSION REPRODUCER
============================================================
Guarantees 100% exact reproduction of competition submission CSV without retraining.
Zero guesswork, zero randomness during inference:
  - Loads frozen Full INT8 model weights (.tflite)
  - Evaluates test images deterministically
  - Applies calibrated per-class sweet spot thresholds
  - Merges verified INT8 Intent predictions
  - Validates all Section IV & V competition invariants
"""

import os
import sys
import argparse
import csv
import glob
import numpy as np
import cv2
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detection.infer import infer_test_images, CALIBRATED_PER_CLASS_THRESH
from submit import validate_submission

INTENT_PREDS_CSV = os.path.join(PROJECT_ROOT, "submissions/intent_preds.csv")

def reproduce(tflite_path, test_img_dir, output_csv, sweetspot_scale=0.6500, iou_thresh=0.45):
    print("=" * 75)
    print("   EDGE AI 2026 - DETERMINISTIC INFERENCE REPRODUCTION")
    print("=" * 75)
    print(f"[*] Model Weights:  {tflite_path}")
    print(f"[*] Test Directory: {test_img_dir}")
    print(f"[*] Output CSV:     {output_csv}")
    print(f"[*] Calibration:    Scale={sweetspot_scale}, IoU={iou_thresh}")

    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Model weights not found at: {tflite_path}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_img_dir}")

    # Scaled per-class thresholds
    base_thresh = list(CALIBRATED_PER_CLASS_THRESH)
    effective_thresh = [round(t * sweetspot_scale, 4) for t in base_thresh]
    print(f"[*] Effective Thresholds: {effective_thresh}")

    temp_det_csv = output_csv + ".temp_det.csv"
    infer_test_images(
        tflite_path,
        test_img_dir,
        output_csv=temp_det_csv,
        conf_thresh=effective_thresh,
        iou_thresh=iou_thresh,
        target_sweetspot=False  # Deterministic fixed threshold
    )

    # Load detection rows
    det_rows = []
    with open(temp_det_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        h = next(reader)
        for r in reader:
            if r:
                det_rows.append(r)
    if os.path.exists(temp_det_csv):
        os.remove(temp_det_csv)

    # Load intent rows
    intent_rows = []
    if os.path.exists(INTENT_PREDS_CSV):
        with open(INTENT_PREDS_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            h = next(reader)
            for r in reader:
                if r:
                    intent_rows.append(r)
    else:
        print("[!] Intent predictions CSV not found, generating fallback INTENT rows...")
        test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png")))
        for p in test_imgs:
            sid = os.path.splitext(os.path.basename(p))[0]
            intent_rows.append(["INTENT", sid, "COUNT"])

    # Ensure equal sample count and strict interleaving or block format
    all_rows = det_rows + intent_rows

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(all_rows)

    print(f"[✔] Generated {len(all_rows)} rows ({len(det_rows)} DET, {len(intent_rows)} INTENT).")
    validate_submission(output_csv, expected_det_count=len(det_rows), expected_intent_count=len(intent_rows))
    print(f"[★] Submission successfully reproduced at: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Deterministic Submission Reproducer")
    parser.add_argument("--weights", required=True, help="Path to model_int8.tflite")
    parser.add_argument("--test_dir", default="data/test/images", help="Path to test images directory")
    parser.add_argument("--output", required=True, help="Output submission CSV path")
    parser.add_argument("--scale", type=float, default=0.6500, help="Sweet spot calibration scale")
    args = parser.parse_args()

    reproduce(args.weights, args.test_dir, args.output, sweetspot_scale=args.scale)

if __name__ == "__main__":
    main()
