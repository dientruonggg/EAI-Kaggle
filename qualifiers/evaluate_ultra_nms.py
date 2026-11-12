#!/usr/bin/env python3
"""
Evaluate DTNet-P2-Ultra with NMS tuning & Threshold Calibration on Validation Set
"""
import os
import sys
import glob
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn

from qualifiers.train_dtnet_p2_ultra import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    DTNetMax, decode_batch_np, evaluate_map
)

def load_val_data(data_dir):
    val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) +
                      glob.glob(str(data_dir / "val/images/*.jpg")))
    gts = []
    for p in val_imgs:
        stem = Path(p).stem
        lp = Path(p).parent.parent / "labels" / f"{stem}.txt"
        boxes = []
        if lp.exists():
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2.0) * ORIG_W
                        y1 = (cy - h / 2.0) * ORIG_H
                        x2 = (cx + w / 2.0) * ORIG_W
                        y2 = (cy + h / 2.0) * ORIG_H
                        boxes.append([cid, x1, y1, x2, y2])
        gts.append(np.array(boxes, dtype=np.float32).reshape(-1, 5))
    return val_imgs, gts

def main():
    repo_root = Path(__file__).resolve().parent.parent
    weights_path = repo_root / "qualifiers/weights/dtnet_p2_ultra/best_p2_ultra.pt"
    data_dir = repo_root / "qualifiers/data"

    print("=" * 80)
    print("   DTNet-P2-ULTRA NMS SWEEP & CALIBRATION EVALUATOR")
    print(f"   Checkpoint: {weights_path}")
    print("=" * 80)

    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    spec = ck["spec"]
    print(f"[*] Checkpoint loaded: Epoch {ck.get('epoch')} | Stored val_mAP50: {ck.get('map50', 0)*100:.2f}%")

    model = DTNetMax(spec)
    model.load_state_dict(ck["ema"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[*] Running inference on device: {device}")

    val_imgs, gts = load_val_data(data_dir)
    print(f"[*] Loaded {len(val_imgs)} validation images and {sum(len(b) for b in gts)} ground truth boxes.")

    # 1. Forward pass all images to extract raw logits
    raw_logits_cls = []
    raw_boxes = []

    print("[*] Extracting model predictions across validation set...")
    t0 = time.time()
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(val_imgs), batch_size):
            batch_paths = val_imgs[i:i+batch_size]
            batch_tensors = []
            for p in batch_paths:
                bgr = cv2.imread(p)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if (rgb.shape[1], rgb.shape[0]) != (IMG_W, IMG_H):
                    rgb = cv2.resize(rgb, (IMG_W, IMG_H))
                batch_tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            
            x = torch.from_numpy(np.array(batch_tensors)).to(device)
            co, bo = model(x)
            raw_logits_cls.append(co.float().cpu().numpy())
            raw_boxes.append(bo.float().cpu().numpy())

    raw_logits_cls = np.concatenate(raw_logits_cls, axis=0)
    raw_boxes = np.concatenate(raw_boxes, axis=0)
    inf_time = time.time() - t0
    print(f"[✔] Forward pass finished in {inf_time:.2f}s ({inf_time/len(val_imgs)*1000:.1f} ms/img)")

    # 2. NMS IoU Sweep (default score_thr=0.001)
    print("\n" + "=" * 80)
    print("   STAGE 1: NMS IoU THRESHOLD SWEEP (score_thr=0.001)")
    print("=" * 80)

    best_nms_iou = 0.55
    best_raw_map = 0.0
    best_per_class = None

    for nms_iou in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        dets = decode_batch_np(raw_logits_cls, raw_boxes, score_thr=0.001, nms_iou=nms_iou)
        m, per_c = evaluate_map(dets, gts, iou_thr=0.5)
        print(f"   NMS IoU = {nms_iou:.2f} -> mAP50 = {m*100:.2f}% | Projected Score: {70.0*m + 30.0:.2f}")
        if m > best_raw_map:
            best_raw_map = m
            best_nms_iou = nms_iou
            best_per_class = per_c

    print(f"\n[★] Best NMS IoU: {best_nms_iou:.2f} with Base mAP50 = {best_raw_map*100:.2f}%")
    print("    Per-class base AP50:")
    for c in range(NUM_CLASSES):
        print(f"       {CLASS_NAMES[c]:8s}: {best_per_class[c]*100:.2f}%")

    # 3. Sweetspot Confidence Threshold Calibration (Coordinate Descent)
    print("\n" + "=" * 80)
    print(f"   STAGE 2: PER-CLASS CONFIDENCE CALIBRATION (at best NMS IoU = {best_nms_iou:.2f})")
    print("=" * 80)

    base_dets = decode_batch_np(raw_logits_cls, raw_boxes, score_thr=0.001, nms_iou=best_nms_iou)
    thresh_cands = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
    opt_thresh = {c: 0.01 for c in range(NUM_CLASSES)}

    for c in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c]
        best_c_thr = opt_thresh[c]
        best_c_ap = 0.0
        for thr in thresh_cands:
            cur_thrs = dict(opt_thresh)
            cur_thrs[c] = thr
            # Filter detections by per-class thresholds
            filtered = []
            for d_img in base_dets:
                if len(d_img) == 0:
                    filtered.append(np.zeros((0, 6), dtype=np.float32))
                    continue
                mask = np.ones(len(d_img), dtype=bool)
                for ci, th in cur_thrs.items():
                    c_mask = (d_img[:, 5] == ci)
                    mask[c_mask] = (d_img[c_mask, 4] >= th)
                filtered.append(d_img[mask])
            _, per_c = evaluate_map(filtered, gts, iou_thr=0.5)
            if per_c[c] > best_c_ap:
                best_c_ap = per_c[c]
                best_c_thr = thr
        opt_thresh[c] = best_c_thr
        print(f"   - Optimized {c_name:8s}: thr={best_c_thr:.3f} -> AP50: {best_c_ap*100:.2f}%")

    # Final evaluation with all calibrated thresholds
    final_dets = []
    for d_img in base_dets:
        if len(d_img) == 0:
            final_dets.append(np.zeros((0, 6), dtype=np.float32))
            continue
        mask = np.ones(len(d_img), dtype=bool)
        for ci, th in opt_thresh.items():
            c_mask = (d_img[:, 5] == ci)
            mask[c_mask] = (d_img[c_mask, 4] >= th)
        final_dets.append(d_img[mask])

    final_map, final_per_c = evaluate_map(final_dets, gts, iou_thr=0.5)
    final_score = 70.0 * final_map + 30.0

    print("\n" + "=" * 80)
    print("   [FINAL ULTRA RESULT AFTER NMS & THRESHOLD CALIBRATION]")
    print(f"   Calibrated Val mAP50 : {final_map*100:.2f}%")
    print(f"   Projected Contest Score: {final_score:.2f} / 100.00")
    print(f"   Best NMS IoU         : {best_nms_iou:.2f}")
    print("   Calibrated Thresholds:")
    for c in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c]
        print(f"       {c_name:8s}: {opt_thresh[c]:.3f} (AP50: {final_per_c[c]*100:.2f}%)")
    print("=" * 80)

    # Save summary
    res_path = repo_root / "qualifiers/weights/dtnet_p2_ultra/nms_calibration_summary.json"
    summary = {
        "model": "DTNet-P2-Ultra",
        "epoch": ck.get("epoch"),
        "best_nms_iou": best_nms_iou,
        "calibrated_val_map50": final_map,
        "projected_contest_score": final_score,
        "calibrated_thresholds": {CLASS_NAMES[c]: opt_thresh[c] for c in range(NUM_CLASSES)},
        "per_class_ap50": {CLASS_NAMES[c]: final_per_c[c] for c in range(NUM_CLASSES)}
    }
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✔] Summary written to: {res_path}")

if __name__ == "__main__":
    main()
