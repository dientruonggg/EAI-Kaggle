import csv
import os
import json

def main():
    root = "D:/EAI-Kaggle"
    det_csv = os.path.join(root, "submission_detection_int8.csv")
    intent_csv = os.path.join(root, "qualifiers/weights_pure_ascii_v2/pure_ascii_v2_intent_preds.csv")
    sample_csv = os.path.join(root, "qualifiers/data/sample_submission.csv")
    out_csv = os.path.join(root, "submission_v2_ascii.csv")

    if not os.path.exists(det_csv):
        print(f"Error: {det_csv} does not exist.")
        return
    if not os.path.exists(intent_csv):
        print(f"Error: {intent_csv} does not exist.")
        return

    # Load detection predictions
    det_map = {}
    with open(det_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["task_type"] == "DETECTION":
                det_map[r["sample_id"]] = r["prediction"]

    # Load intent predictions
    intent_map = {}
    with open(intent_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["task_type"] == "INTENT":
                intent_map[r["sample_id"]] = r["prediction"]

    print(f"Loaded {len(det_map):,} DETECTION rows and {len(intent_map):,} INTENT rows.")

    # Read sample_submission order
    out_rows = []
    with open(sample_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ttype = r["task_type"]
            sid = r["sample_id"]
            if ttype == "DETECTION":
                pred = det_map.get(sid, "[]")
            elif ttype == "INTENT":
                pred = intent_map.get(sid, "OUT_OF_SCOPE")
            else:
                pred = r.get("prediction", "")
            out_rows.append({"task_type": ttype, "sample_id": sid, "prediction": pred})

    # Write clean UTF-8 without BOM
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[✔] Successfully exported {len(out_rows):,} rows to: {out_csv}")

    # Validation
    assert len(out_rows) == 8000, f"Expected 8000 rows, got {len(out_rows)}"
    det_cnt = sum(1 for r in out_rows if r["task_type"] == "DETECTION")
    int_cnt = sum(1 for r in out_rows if r["task_type"] == "INTENT")
    assert det_cnt == 4000 and int_cnt == 4000, "Count mismatch!"
    print(f"[✔] Verification PASSED: 4,000 DETECTION + 4,000 INTENT = 8,000 rows.")

if __name__ == "__main__":
    main()
