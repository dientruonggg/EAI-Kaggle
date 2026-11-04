#!/usr/bin/env python3
"""
ENSEMBLE YOLOV8N + YOLO11N WITH WEIGHTED BOX FUSION (WBF)
Merges multi-architecture YOLO predictions on Warmup test set:
  - Model 1: YOLOv8n (trained with SGD/Adam)
  - Model 2: YOLO11n (trained with AdamW + Cosine LR)
Outputs:
  warmup/submissions/sub_yolo_ensemble_wbf.csv
"""

import os
import sys
import glob
import json
import csv
from ultralytics import YOLO

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "test", "images")
INTENT_PREDS = os.path.join(PROJECT_ROOT, "submissions", "intent_preds.csv")

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def weighted_box_fusion(c_boxes, iou_thresh=0.50, boost_factor=0.04):
    sorted_boxes = sorted(c_boxes, key=lambda b: b["score"], reverse=True)
    clusters = []

    for b in sorted_boxes:
        matched = False
        for cluster in clusters:
            leader = cluster[0]["bbox"]
            if compute_iou(b["bbox"], leader) >= iou_thresh:
                cluster.append(b)
                matched = True
                break
        if not matched:
            clusters.append([b])

    fused = []
    for cluster in clusters:
        tot_w = sum(b["score"] for b in cluster)
        ax1 = sum(b["score"] * b["bbox"][0] for b in cluster) / tot_w
        ay1 = sum(b["score"] * b["bbox"][1] for b in cluster) / tot_w
        ax2 = sum(b["score"] * b["bbox"][2] for b in cluster) / tot_w
        ay2 = sum(b["score"] * b["bbox"][3] for b in cluster) / tot_w

        max_sc = cluster[0]["score"]
        boosted = min(1.0, max_sc * (1.0 + boost_factor * (len(cluster) - 1)))
        cid = cluster[0]["class_id"]
        fused.append({
            "class_id": cid,
            "score": round(boosted, 4),
            "bbox": [round(ax1, 1), round(ay1, 1), round(ax2, 1), round(ay2, 1)]
        })
    return fused

def run_ensemble(
    v8_weights=None,
    v11_weights=None,
    conf=0.12,
    imgsz=640,
    iou_thresh=0.50,
    output_csv="/home/bim/Projects/EAI-2026/warmup/submissions/sub_yolo_ensemble_wbf.csv"
):
    if v8_weights is None:
        v8_weights = os.path.join(PROJECT_ROOT, "runs", "detect", "yolov8n_warmup", "weights", "best.pt")
    if v11_weights is None:
        v11_weights = os.path.join(PROJECT_ROOT, "runs", "detect", "yolo11n_warmup", "weights", "best.pt")

    print(f"[*] Loading Model 1: {v8_weights}")
    m_v8 = YOLO(v8_weights)
    print(f"[*] Loading Model 2: {v11_weights}")
    m_v11 = YOLO(v11_weights)

    test_imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"[*] Running WBF ensemble on {len(test_imgs)} test images (conf={conf}, iou={iou_thresh})...")

    results = []
    total_boxes = 0

    for img_p in test_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]

        raw_boxes = []
        # Predict V8
        r_v8 = m_v8.predict(img_p, conf=conf, imgsz=imgsz, verbose=False)[0]
        for b in r_v8.boxes:
            raw_boxes.append({
                "class_id": int(b.cls[0].item()),
                "score": float(b.conf[0].item()),
                "bbox": b.xyxy[0].tolist()
            })

        # Predict V11
        r_v11 = m_v11.predict(img_p, conf=conf, imgsz=imgsz, verbose=False)[0]
        for b in r_v11.boxes:
            raw_boxes.append({
                "class_id": int(b.cls[0].item()),
                "score": float(b.conf[0].item()),
                "bbox": b.xyxy[0].tolist()
            })

        # Per-class WBF fusion
        fused_img = []
        for c in range(10):
            c_boxes = [b for b in raw_boxes if b["class_id"] == c]
            if c_boxes:
                fused = weighted_box_fusion(c_boxes, iou_thresh=iou_thresh, boost_factor=0.04)
                fused_img.extend(fused)

        fused_img.sort(key=lambda b: b["score"], reverse=True)
        total_boxes += len(fused_img)
        results.append(["DETECTION", sid, json.dumps(fused_img[:100])])

    print(f"[✔] Total fused detections: {total_boxes} across {len(test_imgs)} images ({total_boxes / len(test_imgs):.1f} boxes/img)")

    intent_rows = []
    if os.path.exists(INTENT_PREDS):
        with open(INTENT_PREDS, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row: intent_rows.append(row)

    all_rows = intent_rows + results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[🎉] Successfully created ensemble submission: {output_csv} ({len(all_rows)} rows)")
    return output_csv, total_boxes

if __name__ == "__main__":
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
    out_csv = sys.argv[2] if len(sys.argv) > 2 else f"/home/bim/Projects/EAI-2026/warmup/submissions/sub_yolo_ensemble_c{int(conf*100)}.csv"
    run_ensemble(conf=conf, output_csv=out_csv)
