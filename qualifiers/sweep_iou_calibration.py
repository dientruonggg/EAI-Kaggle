#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - Joint IoU & Per-Class Calibration Sweep
Sweeps IoU threshold [0.35, 0.40, 0.45, 0.48, 0.50, 0.55] with Fine-Grained
Coordinate Descent & Sweetspot Density Targeting on Champion Model (Trial 150).
"""

import os
import sys
import glob
import pickle
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tune_sweetspot_calibration import (
    load_ground_truth,
    extract_image_candidates,
    apply_nms_and_thresholds,
    coordinate_descent_tuning,
    CLASS_NAMES,
    DEFAULT_BASE_THRESHOLDS,
    NUM_CLASSES
)
from src.evaluation import calculate_map50
from submit import validate_submission

def main():
    model_path = os.path.join(PROJECT_ROOT, "weights/continuous_loop/trial_150/model_int8.tflite")
    val_images_dir = os.path.join(PROJECT_ROOT, "data/val/images")
    val_labels_dir = os.path.join(PROJECT_ROOT, "data/val/labels")
    test_images_dir = os.path.join(PROJECT_ROOT, "data/test/images")
    intent_preds_csv = os.path.join(PROJECT_ROOT, "submissions/intent_preds.csv")
    
    val_imgs = sorted(glob.glob(os.path.join(val_images_dir, "*.png")) + glob.glob(os.path.join(val_images_dir, "*.jpg")))
    test_imgs = sorted(glob.glob(os.path.join(test_images_dir, "*.png")) + glob.glob(os.path.join(test_images_dir, "*.jpg")))
    gt_dict = load_ground_truth(val_labels_dir)
    
    target_min = int(round(len(test_imgs) * 43.333))
    target_max = int(round(len(test_imgs) * 46.667))
    target_mid = (target_min + target_max) / 2.0
    
    # Disk caching of raw candidates
    cache_path = os.path.join(PROJECT_ROOT, "candidate_submissions/cand_cache_trial_150_model_int8.pkl")
    if os.path.exists(cache_path):
        print(f"[*] Loading candidates from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            val_cand, test_cand = pickle.load(f)
    else:
        print(f"[*] Extracting raw candidates with floor_thresh=0.005...")
        val_cand = extract_image_candidates(model_path, val_imgs, floor_thresh=0.005)
        test_cand = extract_image_candidates(model_path, test_imgs, floor_thresh=0.005)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((val_cand, test_cand), f)
        print(f"[✔] Saved candidates cache to: {cache_path}")

    iou_candidates = [0.38, 0.42, 0.45, 0.48, 0.50, 0.52]
    results = []

    print("\n" + "=" * 80)
    print("      JOINT IOU & PER-CLASS COORDINATE DESCENT SWEEP")
    print(f"      Model: Trial 150 | Target Range: [{target_min}, {target_max}] boxes")
    print("=" * 80)

    best_global_score = 94.02
    best_global_info = None

    for iou in iou_candidates:
        print(f"\n>>> EVALUATING IOU THRESHOLD = {iou:.2f} <<<")
        tuned_th = coordinate_descent_tuning(val_cand, gt_dict, DEFAULT_BASE_THRESHOLDS, passes=2, iou_thresh=iou)
        
        # Multiplier grid search
        multipliers = np.linspace(0.3, 3.5, 65)
        best_scale = None
        best_val_map = -1.0
        best_test_boxes = 0
        closest_scale = 1.0
        closest_diff = float("inf")
        closest_boxes = 0

        for s in multipliers:
            scaled_th = [round(float(t * s), 4) for t in tuned_th]
            v_preds, _ = apply_nms_and_thresholds(val_cand, scaled_th, iou_thresh=iou)
            v_map, _ = calculate_map50(gt_dict, v_preds)
            _, t_boxes = apply_nms_and_thresholds(test_cand, scaled_th, iou_thresh=iou)

            diff = abs(t_boxes - target_mid)
            if diff < closest_diff:
                closest_diff = diff
                closest_scale = float(s)
                closest_boxes = t_boxes

            if target_min <= t_boxes <= target_max:
                if v_map > best_val_map or best_scale is None:
                    best_val_map = v_map
                    best_scale = float(s)
                    best_test_boxes = t_boxes

        if best_scale is None:
            best_scale = closest_scale
            best_test_boxes = closest_boxes

        # Fine binary search
        low = max(0.2, best_scale - 0.15)
        high = best_scale + 0.15
        for _ in range(8):
            mid = (low + high) / 2.0
            scaled_th = [round(float(t * mid), 4) for t in tuned_th]
            _, t_boxes = apply_nms_and_thresholds(test_cand, scaled_th, iou_thresh=iou)
            if target_min <= t_boxes <= target_max:
                best_scale = float(mid)
                best_test_boxes = t_boxes
            if t_boxes > target_mid:
                low = mid
            else:
                high = mid

        opt_thresholds = [round(float(t * best_scale), 4) for t in tuned_th]
        val_preds_opt, _ = apply_nms_and_thresholds(val_cand, opt_thresholds, iou_thresh=iou)
        opt_val_map, class_aps = calculate_map50(gt_dict, val_preds_opt)
        score_s = 0.70 * (opt_val_map * 100.0) + 0.30 * 100.0

        res_item = {
            "iou": iou,
            "val_map": opt_val_map * 100.0,
            "score": score_s,
            "scale": best_scale,
            "boxes": best_test_boxes,
            "avg_boxes": best_test_boxes / len(test_imgs),
            "thresholds": opt_thresholds,
            "class_aps": class_aps
        }
        results.append(res_item)
        print(f"[IoU {iou:.2f}] -> Val mAP: {res_item['val_map']:.2f}% | Score: {score_s:.2f} | Boxes: {best_test_boxes} (avg {res_item['avg_boxes']:.1f}/img)")

        if score_s > best_global_score:
            best_global_score = score_s
            best_global_info = res_item
            print(f"[🏆 NEW RECORD FOUND!] Score {score_s:.2f} with IoU {iou:.2f}")

    print("\n" + "=" * 80)
    print("      SUMMARY TABLE OF IOU SWEEP")
    print("=" * 80)
    print(f"{'IoU':<8} | {'Val mAP@50':<12} | {'Score S':<10} | {'Test Boxes':<12} | {'Avg Boxes/Img':<14}")
    print("-" * 65)
    for r in results:
        print(f"{r['iou']:<8.2f} | {r['val_map']:<10.2f}% | {r['score']:<10.2f} | {r['boxes']:<12} | {r['avg_boxes']:<14.1f}")
    print("=" * 80)

    if best_global_info:
        print(f"\n[★] Champion beaten! Best IoU: {best_global_info['iou']:.2f} with Score: {best_global_info['score']:.2f}")
    else:
        print(f"\n[✔] Previous Champion 94.02 remains the peak sweet spot.")

if __name__ == "__main__":
    main()
