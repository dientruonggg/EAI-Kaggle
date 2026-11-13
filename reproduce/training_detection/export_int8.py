#!/usr/bin/env python3
"""
export_int8.py
================================================================================
Full Integer INT8 Export & Hardware Verification for DTNet-NWD-TAL on ESP32-S3
================================================================================
"""
from pathlib import Path
import sys
CURRENT_PKG_DIR = Path(__file__).resolve().parent
if str(CURRENT_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_PKG_DIR))


import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf
import keras

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtnet_max import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, NUM_CLASSES, MAX_ALLOWED_PARAMS,
    DTNetMax, count_params_strict, make_spec_max,
    build_keras_dtnet_max, transfer_weights_to_keras
)

MAX_ALLOWED_FILE_SIZE = 1.5 * 1024 * 1024  # 1.5 MB in bytes


def export_full_int8(pt_checkpoint_path, calib_img_dir, output_tflite_path, num_calib=200):
    output_tflite_path = Path(output_tflite_path)
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    print(f"[*] Loading PyTorch checkpoint from: {pt_checkpoint_path}")
    ck = torch.load(pt_checkpoint_path, map_location="cpu", weights_only=False)
    spec = ck.get("spec", make_spec_max())
    net_pt = DTNetMax(spec)
    weights = ck.get("ema", ck.get("model", ck))
    net_pt.load_state_dict(weights)
    net_pt.eval()

    total_params = count_params_strict(net_pt)
    print(f"[Hardware Check] PyTorch Parameters: {total_params:,} / {MAX_ALLOWED_PARAMS:,}")
    assert total_params <= MAX_ALLOWED_PARAMS, f"Exceeded parameter limit: {total_params} > {MAX_ALLOWED_PARAMS}"

    # 2. Build Keras 3 Model and Transfer Weights
    print("[*] Synthesizing matching Keras 3 model...")
    kmodel = build_keras_dtnet_max(spec, IMG_H, IMG_W)
    transfer_weights_to_keras(net_pt, kmodel, spec)

    # 3. Numerical Parity Check
    print("[*] Verifying PyTorch <-> Keras numerical parity...")
    dummy_x = np.random.uniform(0, 1, (2, IMG_H, IMG_W, 3)).astype(np.float32)
    with torch.no_grad():
        pt_in = torch.from_numpy(dummy_x.transpose(0, 3, 1, 2))
        pt_cls, pt_box = net_pt(pt_in)
    kc, kb = kmodel(dummy_x, training=False)
    diff_c = float(np.abs(pt_cls.numpy() - np.asarray(kc).transpose(0, 3, 1, 2)).max())
    diff_b = float(np.abs(pt_box.numpy() - np.asarray(kb).transpose(0, 3, 1, 2)).max())
    print(f"[Parity] Max difference: Classification={diff_c:.2e}, Box={diff_b:.2e}")
    assert diff_c < 0.02 and diff_b < 0.02, f"Numerical parity failed! diff_c={diff_c}, diff_b={diff_b}"
    print("[✔] PyTorch <-> Keras parity verified successfully (< 0.02 tolerance).")

    # 4. Calibration Dataset
    calib_img_paths = sorted(glob.glob(str(Path(calib_img_dir) / "*.jpg")) + glob.glob(str(Path(calib_img_dir) / "*.png")))
    assert len(calib_img_paths) >= 50, f"Insufficient calibration images in {calib_img_dir} ({len(calib_img_paths)} found)"
    calib_samples = calib_img_paths[:num_calib]
    print(f"[*] Post-Training Quantization with {len(calib_samples)} representative calibration samples...")

    def representative_dataset():
        for p in calib_samples:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
                img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            arr = (img.astype(np.float32) / 255.0)[np.newaxis, ...]
            yield [arr]

    # 5. Full Integer INT8 Converter Configuration
    converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[*] Performing Full INT8 PTQ Quantization...")
    tflite_quant_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    file_size_bytes = os.path.getsize(output_tflite_path)
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)
    print(f"\n[Hardware Check] INT8 TFLite Size: {file_size_mb:.3f} MB ({file_size_bytes:,} bytes)")
    print(f"[Hardware Check] Limit: {MAX_ALLOWED_FILE_SIZE / (1024*1024):.2f} MB")
    assert file_size_bytes <= MAX_ALLOWED_FILE_SIZE, f"FATAL: File size exceeded limit: {file_size_mb:.3f} MB > 1.5 MB"
    print(f"[✔] Successfully exported Full INT8 model to: {output_tflite_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DTNetMax PyTorch to Full INT8 TFLite")
    parser.add_argument("--checkpoint", type=str, default="qualifiers/weights/dtnet_nwd_tal_full6k/best_dtnet_nwd_tal_full6k.pt",
                        help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--calib_dir", type=str, default="qualifiers/data/train/images",
                        help="Directory with calibration images")
    parser.add_argument("--output", type=str, default="qualifiers/weights/dtnet_nwd_tal_full6k/dtnet_nwd_tal_full6k_int8.tflite",
                        help="Path to output .tflite model")
    parser.add_argument("--num_calib", type=int, default=200, help="Number of calibration samples")
    args = parser.parse_args()

    export_full_int8(args.checkpoint, args.calib_dir, args.output, args.num_calib)


if __name__ == "__main__":
    main()
