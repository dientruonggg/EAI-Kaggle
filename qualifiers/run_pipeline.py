#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - QUALIFIERS MASTER PIPELINE
End-to-End Execution: Intent Training -> Detection Inference -> Submission Combination -> Verification
"""
import os
import sys
import time
import subprocess

PROJECT_ROOT = "/home/bim/Projects/EAI-2026"
QUALIFIERS_DIR = os.path.join(PROJECT_ROOT, "qualifiers")
PYTHON_BIN = os.path.join(PROJECT_ROOT, ".venv/bin/python")

def run_step(step_name, cmd):
    print("\n" + "=" * 70)
    print(f"[*] STARTING STEP: {step_name}")
    print("=" * 70)
    start_t = time.time()
    res = subprocess.run(cmd, cwd=QUALIFIERS_DIR)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"[✘] STEP FAILED: {step_name} with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✔] COMPLETED STEP: {step_name} in {elapsed:.1f}s")

def main():
    print("=" * 70)
    print("   EDGE AI CHALLENGE 2026 - QUALIFIERS MASTER PIPELINE")
    print("=" * 70)

    # 1. Intent Model Training & Inference
    train_intent_csv = os.path.join(QUALIFIERS_DIR, "data/intent_train.csv")
    test_intent_csv = os.path.join(QUALIFIERS_DIR, "data/intent_val.csv")
    intent_preds_csv = os.path.join(QUALIFIERS_DIR, "submissions/intent_preds.csv")
    model_save = os.path.join(QUALIFIERS_DIR, "weights/text_cnn.pth")
    vocab_save = os.path.join(QUALIFIERS_DIR, "weights/text_cnn_vocab.json")

    run_step("1. Train Intent TextCNN & Infer Questions", [
        PYTHON_BIN, "src/intent/text_cnn.py",
        "--train_csv", train_intent_csv,
        "--test_csv", test_intent_csv,
        "--model_save", model_save,
        "--vocab_save", vocab_save,
        "--output_csv", intent_preds_csv,
        "--epochs", "50"
    ])

    # 2. Object Detection Model & Quantization Check
    tflite_weights = os.path.join(QUALIFIERS_DIR, "weights/mobilenet_bbox_160_int8.tflite")
    if not os.path.exists(tflite_weights):
        run_step("2a. Export Full INT8 TFLite Detector Model", [
            PYTHON_BIN, "src/detection/quantize.py",
            "--img_size", "160",
            "--output", tflite_weights
        ])

    # 2b. Run Single-Model Detection Inference
    det_preds_csv = os.path.join(QUALIFIERS_DIR, "submissions/detection_preds.csv")
    run_step("2b. Infer Test Images with INT8 Detector (Single Model)", [
        PYTHON_BIN, "src/detection/infer.py",
        "--model", tflite_weights,
        "--images_dir", "data/test/images",
        "--output", det_preds_csv,
        "--conf", "0.20",
        "--iou", "0.45"
    ])

    # 3. Combine Predictions into Final Submission
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_sub = os.path.join(QUALIFIERS_DIR, f"submissions/submission_{ts}.csv")
    run_step("3. Combine Predictions into Final Submission", [
        PYTHON_BIN, "build_submission.py",
        "--detection", det_preds_csv,
        "--intent", intent_preds_csv,
        "--output", final_sub
    ])

    print("\n" + "=" * 70)
    print(f"[✔] QUALIFIER PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"[*] Final validated submission file: {final_sub}")
    print("=" * 70)

if __name__ == "__main__":
    main()
