#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - Combine Detection & Intent Predictions into final submission.csv
Zero-dependency implementation using Python standard library (csv, json, time).
"""
import sys
import os
import time
import csv
from submit import validate_submission, submit_to_kaggle

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
        output_csv = f"/home/bim/Projects/EAI-2026/warmup/submissions/submission_{timestamp}.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "sample_id", "prediction"])
        writer.writerows(all_rows)
        
    print(f"[✔] Combined submission ({len(all_rows)} data rows) written to: {output_csv}")

    # Validate schema
    validate_submission(output_csv)

    # Submit if requested
    if submit:
        submit_to_kaggle(output_csv, message=message or f"Submission {time.strftime('%Y-%m-%d %H:%M')}")

    return output_csv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and validate submission CSV")
    parser.add_argument("--detection", default="/home/bim/Projects/EAI-2026/warmup/submissions/detection_preds.csv")
    parser.add_argument("--intent", default="/home/bim/Projects/EAI-2026/warmup/submissions/intent_preds.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--submit", action="store_true", help="Submit directly to Kaggle after validation")
    parser.add_argument("--message", default="auto pipeline submission")

    args = parser.parse_args()
    combine_and_create_submission(args.detection, args.intent, args.output, submit=args.submit, message=args.message)
