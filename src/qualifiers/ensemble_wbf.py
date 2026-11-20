#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - WEIGHTED BOX FUSION (WBF) ENSEMBLE ENGINE
===================================================================
Fuses detection predictions from multiple ESP32-S3 compliant INT8 models.
Pure NumPy / Python implementation - zero external C/Cython dependencies.

Algorithm:
1. Normalize and group bounding boxes by class.
2. For each class, cluster overlapping candidate boxes across models (IoU >= iou_thr).
3. Compute fused box coordinates:
       x = sum(weight_i * score_i * coord_i) / sum(weight_i * score_i)
4. Compute fused box confidence score:
       fused_score = sum(weight_i * score_i) / sum(weight_i)
5. Enforces sweet spot density gating and competition formatting invariants.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from submit import validate_submission

NUM_CLASSES = 7
ORIG_W = 320.0
ORIG_H = 240.0

def bb_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)

def weighted_box_fusion_image(boxes_list, scores_list, classes_list, weights=None, iou_thr=0.55, skip_box_thr=0.01):
    """
    Fuses predictions for a single image across multiple models.
    boxes_list: list of lists of [x1, y1, x2, y2]
    scores_list: list of lists of scores
    classes_list: list of lists of class_ids
    weights: list of model weights (float)
    """
    num_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * num_models
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    # Group candidate boxes by class
    cls_candidates = {c: [] for c in range(NUM_CLASSES)}
    for m in range(num_models):
        m_weight = weights[m]
        for b, s, c in zip(boxes_list[m], scores_list[m], classes_list[m]):
            if s < skip_box_thr:
                continue
            cls_candidates[int(c)].append({
                "box": np.array(b, dtype=np.float32),
                "score": float(s),
                "weight": float(m_weight),
                "model_idx": m
            })

    fused_boxes = []
    fused_scores = []
    fused_classes = []

    for c in range(NUM_CLASSES):
        cands = cls_candidates[c]
        if not cands:
            continue

        # Sort candidates descending by confidence score
        cands.sort(key=lambda x: x["score"], reverse=True)

        clusters = []  # Each cluster: list of candidate dicts
        for cand in cands:
            matched = False
            for cl in clusters:
                # Check IoU with average box in cluster
                cl_boxes = [item["box"] for item in cl]
                cl_weights = [item["weight"] * item["score"] for item in cl]
                sum_w = sum(cl_weights)
                avg_box = sum(w * b for w, b in zip(cl_weights, cl_boxes)) / max(sum_w, 1e-6)

                if bb_iou(cand["box"], avg_box) >= iou_thr:
                    cl.append(cand)
                    matched = True
                    break
            if not matched:
                clusters.append([cand])

        for cl in clusters:
            cl_boxes = [item["box"] for item in cl]
            cl_scores = [item["score"] for item in cl]
            cl_weights = [item["weight"] for item in cl]

            # Weighted average coordinates
            box_weights = [w * s for w, s in zip(cl_weights, cl_scores)]
            sum_bw = sum(box_weights)
            f_box = sum(w * b for w, b in zip(box_weights, cl_boxes)) / max(sum_bw, 1e-6)

            # Score: weighted average boosted by consensus across models
            models_present = len(set(item["model_idx"] for item in cl))
            consensus_factor = min(1.0, 0.75 + 0.25 * (models_present / num_models))
            f_score = (sum_bw / max(sum(cl_weights), 1e-6)) * consensus_factor

            # Clip coordinates
            x1 = max(0.0, min(ORIG_W, float(f_box[0])))
            y1 = max(0.0, min(ORIG_H, float(f_box[1])))
            x2 = max(0.0, min(ORIG_W, float(f_box[2])))
            y2 = max(0.0, min(ORIG_H, float(f_box[3])))

            if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
                fused_boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                fused_scores.append(round(float(f_score), 4))
                fused_classes.append(int(c))

    # Sort final predictions by score descending
    order = np.argsort(fused_scores)[::-1]
    res = []
    for idx in order[:100]:  # strictly max 100 boxes per image
        res.append({
            "class_id": fused_classes[idx],
            "score": fused_scores[idx],
            "bbox": fused_boxes[idx]
        })
    return res

def parse_submission_csv(csv_path):
    """Parses a submission CSV into {sample_id: [boxes]} and {sample_id: intent}."""
    det_dict = {}
    intent_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 3:
                continue
            task_type, sid, pred_str = row[0].strip(), row[1].strip(), row[2].strip()
            if task_type == "DETECTION":
                try:
                    boxes = json.loads(pred_str)
                    det_dict[sid] = boxes
                except Exception:
                    det_dict[sid] = []
            elif task_type == "INTENT":
                intent_dict[sid] = pred_str
    return det_dict, intent_dict

def fuse_submissions(sub_paths, weights=None, iou_thr=0.55, skip_box_thr=0.02, output_csv="wbf_ensemble.csv"):
    """Fuses multiple submission CSVs using Weighted Box Fusion."""
    print("=" * 80)
    print("   ⚡ WEIGHTED BOX FUSION (WBF) ENSEMBLE ENGINE ⚡")
    print(f"   Merging {len(sub_paths)} Submissions:")
    for i, p in enumerate(sub_paths):
        w = weights[i] if weights else 1.0
        print(f"     [{i+1}] Weight {w:.2f}: {p}")
    print(f"   IoU Threshold: {iou_thr} | Skip Box Threshold: {skip_box_thr}")
    print("=" * 80)

    parsed = [parse_submission_csv(p) for p in sub_paths]
    all_det = [p[0] for p in parsed]
    all_intent = [p[1] for p in parsed]

    # Shared sample IDs for detection
    sample_ids = sorted(list(all_det[0].keys()))
    print(f"[*] Found {len(sample_ids)} detection sample IDs.")

    fused_det_rows = []
    total_boxes = 0

    for sid in sample_ids:
        boxes_list = []
        scores_list = []
        classes_list = []
        for m in range(len(sub_paths)):
            m_preds = all_det[m].get(sid, [])
            boxes_list.append([item["bbox"] for item in m_preds])
            scores_list.append([item["score"] for item in m_preds])
            classes_list.append([item["class_id"] for item in m_preds])

        fused = weighted_box_fusion_image(
            boxes_list, scores_list, classes_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        total_boxes += len(fused)
        fused_det_rows.append(["DETECTION", sid, json.dumps(fused)])

    avg_boxes = total_boxes / max(len(sample_ids), 1)
    print(f"[✔] WBF Fusion complete: {total_boxes} total boxes (Avg: {avg_boxes:.1f} boxes/img)")

    # Intent rows (take from first submission, 100% locked)
    intent_rows = [["INTENT", sid, all_intent[0][sid]] for sid in sorted(all_intent[0].keys())]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(fused_det_rows)
        writer.writerows(intent_rows)

    print(f"[*] Validating output CSV: {output_csv}")
    validate_submission(output_csv)
    print(f"[✔] Successfully created verified ensemble: {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+", help="Paths to submission CSVs to fuse")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Weights for each submission")
    parser.add_argument("--iou-thr", type=float, default=0.55, help="WBF IoU match threshold")
    parser.add_argument("--skip-thr", type=float, default=0.02, help="Skip box threshold")
    parser.add_argument("--output", type=str, default="wbf_ensemble.csv", help="Output CSV path")
    args = parser.parse_args()

    fuse_submissions(
        args.submissions,
        weights=args.weights,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_thr,
        output_csv=args.output
    )
