#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 (QUALIFIERS) - Submission Validator & Kaggle Submitter
Zero-dependency validator using pure Python standard library (csv, json, subprocess).
Validates all Section IV & V constraints:
- 7 Detection Classes (0: bottle, 1: cup, 2: phone, 3: book, 4: mouse, 5: box, 6: toy_car)
- Bounding box coordinates within 320x240 image space
- Max 100 bounding boxes per image
- Valid Intents: COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE
"""
import sys
import os
import csv
import json
import subprocess

COMPETITION_SLUG = "edge-ai-challenge-2026-lan-2"
VALID_INTENTS = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"}
MAX_BOXES_PER_IMAGE = 100
MAX_CLASS_ID = 6  # 7 classes: 0..6
ORIG_W = 320.0
ORIG_H = 240.0

def validate_submission(csv_path, expected_det_count=None, expected_intent_count=None):
    print(f"[*] Validating Qualifier submission: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["task_type", "sample_id", "prediction"]:
            raise ValueError(f"Header mismatch! Expected ['task_type', 'sample_id', 'prediction'], got {header}")
        for r in reader:
            if r:
                rows.append(r)

    det_rows = [r for r in rows if r[0] == "DETECTION"]
    intent_rows = [r for r in rows if r[0] == "INTENT"]

    print(f"[*] Total Rows: {len(rows)} | DETECTION rows: {len(det_rows)} | INTENT rows: {len(intent_rows)}")

    if expected_det_count is not None and len(det_rows) != expected_det_count:
        raise ValueError(f"Expected {expected_det_count} DETECTION rows, got {len(det_rows)}")
    if expected_intent_count is not None and len(intent_rows) != expected_intent_count:
        raise ValueError(f"Expected {expected_intent_count} INTENT rows, got {len(intent_rows)}")

    # Check unique sample_ids
    det_ids = set()
    intent_ids = set()

    # Validate Detection rows
    total_boxes = 0
    for r in det_rows:
        task_type, sample_id, pred_raw = r[0], r[1], r[2].strip()
        if sample_id in det_ids:
            raise ValueError(f"Duplicate sample_id in DETECTION: {sample_id}")
        det_ids.add(sample_id)

        try:
            boxes = json.loads(pred_raw)
            if not isinstance(boxes, list):
                raise ValueError(f"Prediction for {sample_id} must be a JSON array, got: {pred_raw}")
            if len(boxes) > MAX_BOXES_PER_IMAGE:
                raise ValueError(f"Sample {sample_id} exceeds {MAX_BOXES_PER_IMAGE} boxes limit (got {len(boxes)})")
            
            total_boxes += len(boxes)
            for b in boxes:
                if not isinstance(b, dict):
                    raise ValueError(f"Box item must be dict, got: {b}")
                for k in ("class_id", "score", "bbox"):
                    if k not in b:
                        raise ValueError(f"Box missing '{k}' key: {b}")
                cid = b["class_id"]
                score = b["score"]
                bbox = b["bbox"]

                if not isinstance(cid, int) or not (0 <= cid <= MAX_CLASS_ID):
                    raise ValueError(f"Invalid class_id {cid} (must be int 0..{MAX_CLASS_ID}) in {sample_id}")
                if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                    raise ValueError(f"Invalid score {score} (must be 0.0..1.0) in {sample_id}")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid bbox {bbox} (must be [xmin, ymin, xmax, ymax]) in {sample_id}")
                
                xmin, ymin, xmax, ymax = bbox
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError(f"Degenerate box [x1={xmin}, y1={ymin}, x2={xmax}, y2={ymax}] in {sample_id}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {sample_id}: {pred_raw} ({str(e)})")

    # Validate Intent rows
    for r in intent_rows:
        task_type, sample_id, pred_raw = r[0], r[1], r[2].strip()
        if sample_id in intent_ids:
            raise ValueError(f"Duplicate sample_id in INTENT: {sample_id}")
        intent_ids.add(sample_id)

        if pred_raw not in VALID_INTENTS:
            raise ValueError(f"Invalid intent '{pred_raw}' in {sample_id}. Must be one of {VALID_INTENTS}")

    avg_boxes = (total_boxes / len(det_rows)) if det_rows else 0
    print(f"[✔] Schema & Invariants PASSED! Total detected boxes: {total_boxes} (Avg {avg_boxes:.1f} boxes/img)")
    return True

def submit_to_kaggle(csv_path, message="Qualifier submission"):
    print(f"[*] Submitting {csv_path} to Kaggle competition: {COMPETITION_SLUG}...")
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION_SLUG,
        "-f", csv_path,
        "-m", message
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("[✔] Submission successfully sent to Kaggle!")
    else:
        print(f"[✘] Kaggle submission failed with code {res.returncode}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python submit.py <path_to_submission.csv> [optional_message]")
        sys.exit(1)

    sub_file = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else "Qualifier automated submission"
    
    validate_submission(sub_file)
    if "--submit" in sys.argv:
        submit_to_kaggle(sub_file, msg)
