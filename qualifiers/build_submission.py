#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 (QUALIFIERS) - Combine Detection & Intent Predictions
Merges detection_preds.csv and intent_preds.csv into official submission.csv.
Runs validation before saving and optional direct submission to Kaggle.
"""
import sys
import os
import time
import csv
from submit import validate_submission, submit_to_kaggle

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.dirname(__file__)))

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader:
            if r:
                rows.append(r)
    return rows

def combine_and_create_submission(detection_csv, intent_csv, output_csv=None, submit=False, message=""):
    print(f"[*] Reading detection predictions from: {detection_csv}")
    det_rows = read_csv_rows(detection_csv)
    
    print(f"[*] Reading intent predictions from: {intent_csv}")
    intent_rows = read_csv_rows(intent_csv)

    all_rows = det_rows + intent_rows

    if output_csv is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(PROJECT_ROOT, f"submissions/submission_{timestamp}.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(all_rows)
        
    print(f"[✔] Combined submission ({len(all_rows)} rows) written to: {output_csv}")

    # Validate submission against 7 classes, 320x240 box rules, 5 intents
    validate_submission(output_csv)

    if submit:
        submit_to_kaggle(output_csv, message=message or f"Qualifier submission {time.strftime('%Y-%m-%d %H:%M')}")

    return output_csv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine Detection & Intent predictions into final submission.csv")
    parser.add_argument("--detection", default=os.path.join(PROJECT_ROOT, "submissions/detection_preds.csv"))
    parser.add_argument("--intent", default=os.path.join(PROJECT_ROOT, "submissions/intent_preds.csv"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--submit", action="store_true", help="Submit directly to Kaggle after validation")
    parser.add_argument("--message", default="Qualifiers automated pipeline submission")
    args = parser.parse_args()

    combine_and_create_submission(args.detection, args.intent, args.output, args.submit, args.message)
