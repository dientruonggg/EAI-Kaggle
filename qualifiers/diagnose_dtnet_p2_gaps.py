#!/usr/bin/env python3
"""
diagnose_dtnet_p2_gaps.py
Deep error diagnosis for DTNet-P2 Full INT8 model on 600 validation images:
1. Quantifies exact FN (False Negatives) vs FP (False Positives) per class.
2. Identifies 'Near-Misses' (IoU between 0.35 and 0.49).
3. Evaluates NMS IoU sweep (0.50 -> 0.70) and 2D grid calibration.
4. Pinpoints high-impact actions to reach 99.0%+ contest score.
"""

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ORIG_W, ORIG_H = 320, 240
IMG_W, IMG_H = 320, 240
STRIDE = 8
NUM_CLASSES = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def nms_np(boxes, scores, thr, max_keep=100):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[r] - inter + 1e-9)
        order = r[iou <= thr]
    return np.asarray(keep, dtype=np.int64)


def decode_batch_np(cls_logit, box_raw, score_thr=0.001, nms_iou=0.60, pre_nms=1000, max_det=100):
    B, C, gh, gw = cls_logit.shape
    scores = _sigmoid(cls_logit)
    d = _sigmoid(box_raw) * (20.0 * STRIDE)
    gx = (np.arange(gw) + 0.5) * STRIDE
    gy = (np.arange(gh) + 0.5) * STRIDE
    out = []
    for b in range(B):
        sc = scores[b].reshape(-1)
        k = min(pre_nms, sc.size)
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[sc[idx] > score_thr]
        if idx.size == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        c = idx // (gh * gw)
        rem = idx % (gh * gw)
        j, i = rem // gw, rem % gw
        boxes = np.stack([gx[i] - d[b, 0, j, i], gy[j] - d[b, 1, j, i],
                          gx[i] + d[b, 2, j, i], gy[j] + d[b, 3, j, i]], 1)
        s = sc[idx]
        off = boxes + (c[:, None] * 4096.0)
        keep = nms_np(off, s, nms_iou, max_det)
        bx = boxes[keep]
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, ORIG_W)
        bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, ORIG_H)
        out.append(np.concatenate([bx, s[keep, None], c[keep, None]], 1).astype(np.float32))
    return out


def evaluate_map(dets, gts, num_classes=NUM_CLASSES, iou_thr=0.5):
    aps = {}
    details = {}
    for c in range(num_classes):
        gt_c = [g[g[:, 0] == c][:, 1:5] for g in gts]
        n_gt = sum(len(x) for x in gt_c)
        if n_gt == 0:
            continue
        used = [np.zeros(len(x), dtype=bool) for x in gt_c]
        recs = []
        for k, d in enumerate(dets):
            if d is None or len(d) == 0:
                continue
            for row in d[d[:, 5] == c]:
                recs.append((float(row[4]), k, row[:4]))
        recs.sort(key=lambda r: -r[0])
        tp, fp = np.zeros(len(recs)), np.zeros(len(recs))
        near_misses = 0
        for n, (s, k, bx) in enumerate(recs):
            g = gt_c[k]
            if len(g) == 0:
                fp[n] = 1
                continue
            xx1 = np.maximum(bx[0], g[:, 0])
            yy1 = np.maximum(bx[1], g[:, 1])
            xx2 = np.minimum(bx[2], g[:, 2])
            yy2 = np.minimum(bx[3], g[:, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a = max(0.0, (bx[2] - bx[0])) * max(0.0, (bx[3] - bx[1]))
            ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
            ious = inter / np.maximum(a + ag - inter, 1e-9)
            j = int(ious.argmax())
            max_iou = ious[j]
            if max_iou >= iou_thr and not used[k][j]:
                tp[n] = 1
                used[k][j] = True
            else:
                fp[n] = 1
                if 0.35 <= max_iou < iou_thr:
                    near_misses += 1

        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ii = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))
        aps[c] = ap
        total_tp = int(ctp[-1]) if len(ctp) > 0 else 0
        total_fp = int(cfp[-1]) if len(cfp) > 0 else 0
        fn = n_gt - total_tp
        details[c] = {
            "n_gt": n_gt, "tp": total_tp, "fp": total_fp, "fn": fn,
            "near_misses": near_misses, "ap50": ap
        }
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps, details


def main():
    tflite_path = PROJECT_ROOT / "qualifiers/weights/dtnet_p2_champion/dtnet_p2_int8.tflite"
    val_imgs = sorted(glob.glob(str(PROJECT_ROOT / "qualifiers/data/val/images/*.png")) +
                      glob.glob(str(PROJECT_ROOT / "qualifiers/data/val/images/*.jpg")))
    print(f"[*] Analyzing DTNet-P2 errors across {len(val_imgs)} validation images...")

    val_lbls = []
    for p in val_imgs:
        stem = Path(p).stem
        lbl_p = Path(p).parent.parent / "labels" / f"{stem}.txt"
        rows = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                    rows.append([c, (cx - w / 2) * ORIG_W, (cy - h / 2) * ORIG_H,
                                    (cx + w / 2) * ORIG_W, (cy + h / 2) * ORIG_H])
        val_lbls.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    cls_idx_tflite = 0 if output_details[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx_tflite = 1 - cls_idx_tflite
    in_scale, in_zp = input_details["quantization"]
    cls_scale, cls_zp = output_details[cls_idx_tflite]["quantization"]
    box_scale, box_zp = output_details[box_idx_tflite]["quantization"]

    raw_preds = []
    for p in val_imgs:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        quant_in = np.clip(np.round((img.astype(np.float32) / 255.0) / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quant_in[np.newaxis, ...])
        interpreter.invoke()
        q_cls = interpreter.get_tensor(output_details[cls_idx_tflite]["index"])
        q_box = interpreter.get_tensor(output_details[box_idx_tflite]["index"])
        f_cls = (q_cls.astype(np.float32) - cls_zp) * cls_scale
        f_box = (q_box.astype(np.float32) - box_zp) * box_scale
        raw_preds.append((f_cls.transpose(0, 3, 1, 2), f_box.transpose(0, 3, 1, 2)))

    # NMS IoU Sweep
    print("\n" + "=" * 80)
    print("   NMS IoU THRESHOLD SENSITIVITY SWEEP")
    print("=" * 80)
    best_nms_iou = 0.60
    best_map_nms = 0.0
    for nms_iou in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        dets = [decode_batch_np(c, b, score_thr=0.001, nms_iou=nms_iou)[0] for c, b in raw_preds]
        m, _, _ = evaluate_map(dets, val_lbls)
        print(f"   NMS IoU = {nms_iou:.2f} -> Base mAP50 = {m*100:.2f}%")
        if m > best_map_nms:
            best_map_nms = m
            best_nms_iou = nms_iou

    print(f"\n[✔] Optimal NMS IoU threshold: {best_nms_iou:.2f} (mAP50: {best_map_nms*100:.2f}%)")

    # Error breakdown with optimal NMS IoU
    opt_dets = [decode_batch_np(c, b, score_thr=0.001, nms_iou=best_nms_iou)[0] for c, b in raw_preds]
    cal_path = PROJECT_ROOT / "qualifiers/weights/dtnet_p2_champion/champion_summary.json"
    with open(cal_path) as f:
        meta = json.load(f)
    opt_thresh = meta["evaluation_metrics"]["calibrated_thresholds"]

    filtered_dets = []
    for d in opt_dets:
        if len(d) == 0:
            filtered_dets.append(d)
        else:
            keep = np.array([row[4] >= opt_thresh[CLASS_NAMES[int(row[5])]] for row in d])
            filtered_dets.append(d[keep])

    overall_map, aps, details = evaluate_map(filtered_dets, val_lbls)

    print("\n" + "=" * 80)
    print("   DETAILED ERROR BREAKDOWN PER CLASS (INT8 MODEL)")
    print("=" * 80)
    print(f" {'Class':<10} | {'AP@50':<8} | {'GT Count':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Near-Miss (0.35-0.49)':<22}")
    print("-" * 80)
    total_gt, total_tp, total_fp, total_fn, total_nm = 0, 0, 0, 0, 0
    for c in range(NUM_CLASSES):
        d = details[c]
        print(f" {CLASS_NAMES[c]:<10} | {d['ap50']*100:6.2f}% | {d['n_gt']:<8} | {d['tp']:<6} | {d['fp']:<6} | {d['fn']:<6} | {d['near_misses']:<22}")
        total_gt += d["n_gt"]
        total_tp += d["tp"]
        total_fp += d["fp"]
        total_fn += d["fn"]
        total_nm += d["near_misses"]
    print("-" * 80)
    print(f" {'TOTAL':<10} | {overall_map*100:6.2f}% | {total_gt:<8} | {total_tp:<6} | {total_fp:<6} | {total_fn:<6} | {total_nm:<22}")
    print("=" * 80)


if __name__ == "__main__":
    main()
