#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 (QUALIFIERS) - DTNet-NWD-TAL 197k Detection Trainer
-----------------------------------------------------------------------------
Unified training script for DTNetMax (197,647 parameters < 200,000 hard limit).
Features:
  1. MobileNetV3 backbone with High-Res P2 (Stride-4) skip connection
  2. Native Rectangular 320x240 OV7670 camera aspect ratio
  3. Task-Aligned Assigner (TAL) with dynamic soft-label quality metrics
  4. Normalized Wasserstein Distance (NWD) loss for tiny objects (phones, mice, cups)
  5. Cross-class mutual exclusion loss to prevent overlapping false positives
  6. Multi-scale Mosaic augmentation with close-mosaic clean phase
  7. Full INT8 Post-Training Quantization (PTQ) via representative dataset
  8. Sweetspot threshold calibration for maximum precision mAP50

Usage:
  python reproduce/train_detection.py --epochs 300 --batch_size 64
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CURRENT_DIR / "training_detection"
PROJECT_ROOT = CURRENT_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


def main():
    parser = argparse.ArgumentParser(description="Train DTNet-NWD-TAL Detection Champion Model")
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Learning rate (default: 1.2e-4)")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "weights"), help="Weights output directory")
    parser.add_argument("--scratch", action="store_true", help="Train strictly from scratch (random weights, no checkpoint)")
    parser.add_argument("--init_weights", type=str, default="", help="Initial checkpoint path if resuming/finetuning")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run only INT8 export + calibration")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        candidates = [
            PROJECT_ROOT / "qualifiers/data",
            PROJECT_ROOT / "data",
            CURRENT_DIR / "data"
        ]
        for c in candidates:
            if c.exists():
                data_dir = str(c)
                break
    if not data_dir:
        raise FileNotFoundError("Could not locate training dataset directory!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_pt_path = output_dir / "best_dtnet_nwd_tal_full6k.pt"
    tflite_path = output_dir / "dtnet_nwd_tal_full6k_int8.tflite"

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - DTNet-NWD-TAL FULL-6K DETECTION TRAINER")
    print(f"   Architecture : DTNetMax (197,647 params <= 200,000 limit)")
    print(f"   Epochs       : {args.epochs} (Batch Size: {args.batch_size})")
    print(f"   Data Dir     : {data_dir}")
    print(f"   Output Dir   : {output_dir}")
    print("=" * 80)

    if not args.skip_train:
        train_cmd = [
            sys.executable, str(TRAINING_DIR / "train.py"),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--data_dir", str(data_dir),
            "--output_dir", str(output_dir)
        ]
        if args.scratch:
            train_cmd.append("--scratch")
        elif args.init_weights:
            train_cmd.extend(["--init_weights", args.init_weights])
        print(f"\n[*] Executing training: {' '.join(train_cmd)}")
        res = subprocess.run(train_cmd)
        if res.returncode != 0:
            print(f"[✘] Training exited with error code {res.returncode}")
            sys.exit(res.returncode)

    if not best_pt_path.exists():
        print(f"[!] Warning: {best_pt_path} not found. Checking existing weights...")
        fallback_pt = PROJECT_ROOT / "qualifiers/weights/dtnet_nwd_tal_full6k/best_dtnet_nwd_tal_full6k.pt"
        if fallback_pt.exists():
            import shutil
            shutil.copy2(fallback_pt, best_pt_path)
            print(f"[✔] Copied pre-trained weights from {fallback_pt} to {best_pt_path}")

    # Export INT8
    print("\n[*] Exporting to Full INT8 TFLite for ESP32-S3...")
    calib_dir = os.path.join(data_dir, "train/images")
    export_cmd = [
        sys.executable, str(TRAINING_DIR / "export_int8.py"),
        "--checkpoint", str(best_pt_path),
        "--calib_dir", calib_dir,
        "--output", str(tflite_path),
        "--num_calib", "200"
    ]
    res = subprocess.run(export_cmd)
    if res.returncode != 0:
        print(f"[✘] INT8 export failed with code {res.returncode}")
        sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print(" [★ SUCCESS] Detection training & INT8 export completed successfully!")
    print(f" PyTorch Checkpoint : {best_pt_path}")
    print(f" TFLite INT8 Model  : {tflite_path} ({os.path.getsize(tflite_path):,} bytes)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
