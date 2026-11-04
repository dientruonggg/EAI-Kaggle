#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - Master End-to-End Pipeline
Executes: Training -> Inference -> Submission Creation -> Kaggle Scoring
"""
import os
import sys
import time
import subprocess

PROJECT_ROOT = "/home/bim/Projects/EAI-2026"
WARMUP_DIR = os.path.join(PROJECT_ROOT, "warmup")
PYTHON_BIN = os.path.join(PROJECT_ROOT, ".venv/bin/python")

def run_step(step_name, cmd):
    print("\n" + "="*70)
    print(f"[*] STARTING STEP: {step_name}")
    print("="*70)
    start_t = time.time()
    res = subprocess.run(cmd, cwd=WARMUP_DIR)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"[✘] STEP FAILED: {step_name} with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✔] COMPLETED STEP: {step_name} in {elapsed:.1f}s")

def main():
    print("="*70)
    print("   EDGE AI CHALLENGE 2026 - MASTER PIPELINE")
    print("="*70)

    # 1. Intent Model Training & Inference
    run_step("1. Train Intent TextCNN & Infer Test Questions", [
        PYTHON_BIN, "src/intent/text_cnn.py"
    ])

    # 2. Object Detection Inference
    weights_path = os.path.join(WARMUP_DIR, "runs/detect/yolov8n_warmup/weights/best.pt")
    if not os.path.exists(weights_path):
        run_step("2a. Train YOLOv8n Object Detector", [
            PYTHON_BIN, "src/detection/train_detection.py"
        ])
    else:
        print(f"[*] Found trained detector weights at: {weights_path}")

    run_step("2b. Infer Test Images with YOLO", [
        PYTHON_BIN, "-c",
        f"""
import sys; sys.path.append('src/detection')
from infer_detection import infer_test_images
df = infer_test_images('{weights_path}', 'data/test/images', conf_threshold=0.20)
df.to_csv('submissions/detection_preds.csv', index=False)
print('Detection predictions saved!')
"""
    ])

    # 3. Build & Validate Submission
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_sub = os.path.join(WARMUP_DIR, f"submissions/submission_{ts}.csv")
    run_step("3. Combine Predictions into Final Submission", [
        PYTHON_BIN, "build_submission.py",
        "--detection", "submissions/detection_preds.csv",
        "--intent", "submissions/intent_preds.csv",
        "--output", final_sub
    ])

    # 4. Submit to Kaggle & Get Score
    run_step("4. Submit to Kaggle & Await Score", [
        PYTHON_BIN, "submit.py", final_sub, f"Pipeline run {ts}"
    ])

    print("\n[*] Checking Kaggle leaderboard...")
    subprocess.run(["kaggle", "competitions", "submissions", "-c", "edge-ai-challenge-2026-lan-2"])

if __name__ == "__main__":
    main()
