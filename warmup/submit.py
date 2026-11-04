#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 (WARM-UP) - Submission Helper & Validator
Zero-dependency validator (uses pure Python standard library: csv, json, subprocess).
"""
import sys
import os
import csv
import json
import subprocess

COMPETITION_SLUG = "edge-ai-challenge-2026-lan-2"
EXPECTED_DETECTION_COUNT = 60
EXPECTED_INTENT_COUNT = 28
VALID_INTENTS = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"}

def validate_submission(csv_path):
    print(f"[*] Validating submission file: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["task_type", "sample_id", "prediction"]:
            raise ValueError(f"Columns mismatch: expected ['task_type', 'sample_id', 'prediction'], got {header}")
        for r in reader:
            if r:
                rows.append(r)

    total_expected = EXPECTED_DETECTION_COUNT + EXPECTED_INTENT_COUNT
    if len(rows) != total_expected:
        raise ValueError(f"Expected {total_expected} rows, got {len(rows)}")

    det_rows = [r for r in rows if r[0] == "DETECTION"]
    intent_rows = [r for r in rows if r[0] == "INTENT"]

    if len(det_rows) != EXPECTED_DETECTION_COUNT:
        raise ValueError(f"Expected {EXPECTED_DETECTION_COUNT} DETECTION rows, got {len(det_rows)}")
    if len(intent_rows) != EXPECTED_INTENT_COUNT:
        raise ValueError(f"Expected {EXPECTED_INTENT_COUNT} INTENT rows, got {len(intent_rows)}")

    # Verify detection rows
    for i, r in enumerate(det_rows):
        task_type, sample_id, pred_raw = r[0], r[1], r[2].strip()
        try:
            boxes = json.loads(pred_raw)
            if not isinstance(boxes, list):
                raise ValueError("Prediction must be a JSON array")
            for b in boxes:
                if not isinstance(b, dict):
                    raise ValueError(f"Box entry must be dict: {b}")
                for key in ("class_id", "score", "bbox"):
                    if key not in b:
                        raise ValueError(f"Box missing '{key}' key")
                if not isinstance(b["class_id"], int) or not (0 <= b["class_id"] <= 9):
                    raise ValueError(f"Invalid class_id: {b.get('class_id')}")
                if not isinstance(b["score"], (int, float)) or not (0.0 <= b["score"] <= 1.0):
                    raise ValueError(f"Invalid score: {b.get('score')}")
                bbox = b["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"bbox must be [xmin, ymin, xmax, ymax]: {bbox}")
        except Exception as e:
            raise ValueError(f"DETECTION row {i+1} ({sample_id}) error: {e}")

    # Verify intent rows
    for i, r in enumerate(intent_rows):
        task_type, sample_id, pred = r[0], r[1], r[2].strip()
        if pred not in VALID_INTENTS:
            raise ValueError(f"INTENT row {i+1} ({sample_id}) invalid intent '{pred}'. Must be one of {VALID_INTENTS}")

    print("[\u2714] Validation PASSED! File format is 100% compliant.")
    return True

def submit_to_kaggle(csv_path, message="baseline submission"):
    validate_submission(csv_path)
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION_SLUG,
        "-f", csv_path,
        "-m", message
    ]
    print(f"[*] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode == 0:
        print("[\u2714] Submitted successfully!")
        print(f"[*] Check status with: kaggle competitions submissions -c {COMPETITION_SLUG}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 submit.py <path_to_submission.csv> [message]")
        sys.exit(1)
    csv_file = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else "auto submission"
    submit_to_kaggle(csv_file, msg)
