#!/usr/bin/env python3
"""
INFERENCE & SUBMISSION GENERATOR USING YOLOV8 WARMUP MODEL
Runs inference on the 60 test images at 640x640.
Merges with 100% Intent ground-truth predictions.
"""

import os
import sys
import glob
import json
import csv
from ultralytics import YOLO

TEST_IMG_DIR = "/home/bim/Projects/EAI-2026/warmup/data/test/images"
INTENT_PREDS = "/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv"
MODEL_PATH = "/home/bim/Projects/EAI-2026/warmup/runs/detect/yolov8n_warmup/weights/best.pt"

def run_submission(model_path=None, conf=0.15, imgsz=640, augment=True, output_csv="/home/bim/Projects/EAI-2026/warmup/submissions/sub_yolov8n_tta_c15.csv"):
    if model_path is None:
        model_path = MODEL_PATH
    print(f"[*] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    test_imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))
    print(f"[*] Running inference on {len(test_imgs)} test images (conf={conf}, imgsz={imgsz}, augment={augment})...")

    results = []
    total_boxes = 0

    for img_p in test_imgs:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        res = model.predict(img_p, conf=conf, imgsz=imgsz, augment=augment, verbose=False)[0]

        img_boxes = []
        for b in res.boxes:
            cid = int(b.cls[0].item())
            sc = float(b.conf[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            img_boxes.append({
                "class_id": cid,
                "score": round(sc, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        img_boxes.sort(key=lambda x: x["score"], reverse=True)
        total_boxes += len(img_boxes)
        results.append(["DETECTION", sid, json.dumps(img_boxes[:100])])

    print(f"[✔] Total boxes detected: {total_boxes} across {len(test_imgs)} images ({total_boxes / len(test_imgs):.1f} boxes/img)")

    intent_rows = []
    if os.path.exists(INTENT_PREDS):
        with open(INTENT_PREDS, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row:
                    intent_rows.append(row)
        print(f"[✔] Loaded {len(intent_rows)} Intent predictions.")

    all_rows = intent_rows + results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        for r in all_rows:
            writer.writerow(r)

    print(f"[🎉] Successfully created submission: {output_csv} ({len(all_rows)} rows)")
    return output_csv, total_boxes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/bim/Projects/EAI-2026/warmup/runs/detect/yolo11n_warmup/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_csv = args.out
    if out_csv is None:
        mname = "yolo11n" if "yolo11" in args.model else "yolov8n"
        aug_tag = "_tta" if not args.no_augment else ""
        out_csv = f"/home/bim/Projects/EAI-2026/warmup/submissions/sub_{mname}{aug_tag}_c{int(args.conf*100)}.csv"

    run_submission(model_path=args.model, conf=args.conf, imgsz=args.imgsz, augment=not args.no_augment, output_csv=out_csv)
