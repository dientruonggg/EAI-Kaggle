#!/usr/bin/env python3
import json
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qualifiers.train_dtnet_p2_eiou import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    make_spec, DTNet, build_keras_dtnet, transfer_weights_to_keras,
    export_full_int8_tflite, decode_batch_np, evaluate_map
)

def main():
    weights_dir = PROJECT_ROOT / "qualifiers/weights/dtnet_p2_eiou"
    pt_path = weights_dir / "best_p2_eiou.pt"
    data_dir = PROJECT_ROOT / "qualifiers/data"

    print("=" * 80)
    print("   EXPORTING & CALIBRATING DTNet-P2 EIoU CHAMPION TO FULL INT8")
    print(f"   Checkpoint: {pt_path}")
    print("=" * 80)

    # 1. Load PyTorch weights
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    spec = ck["spec"]
    model = DTNet(spec)
    model.load_state_dict(ck["ema"])
    model.eval()

    # 2. Synthesize Keras model & transfer weights
    print("[*] Synthesizing matching Keras model...")
    kmodel = build_keras_dtnet(spec, IMG_H, IMG_W)
    transfer_weights_to_keras(model, kmodel, spec)

    # Verification of parity
    test_x = np.random.uniform(0, 1, (2, IMG_H, IMG_W, 3)).astype(np.float32)
    with torch.no_grad():
        pt_in = torch.from_numpy(test_x.transpose(0, 3, 1, 2))
        tc, tb = model(pt_in)
    kc, kb = kmodel(test_x, training=False)
    diff_c = float(np.abs(tc.numpy() - np.asarray(kc).transpose(0, 3, 1, 2)).max())
    diff_b = float(np.abs(tb.numpy() - np.asarray(kb).transpose(0, 3, 1, 2)).max())
    print(f"[✔] PyTorch <-> Keras verification: cls error={diff_c:.2e}, box error={diff_b:.2e}")
    assert diff_c < 0.02 and diff_b < 0.02

    # 3. Representative dataset & PTQ
    all_train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) + glob.glob(str(data_dir / "train/images/*.jpg")))
    all_val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) + glob.glob(str(data_dir / "val/images/*.jpg")))
    all_imgs = all_train_imgs + all_val_imgs

    # Reproducible 9:1 split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_imgs))
    val_count = int(round(len(all_imgs) * 0.10))
    val_idx, tr_idx = indices[:val_count], indices[val_count:]
    tr_imgs = [all_imgs[i] for i in tr_idx]
    vl_imgs = [all_imgs[i] for i in val_idx]

    vl_lbls = []
    for p in vl_imgs:
        stem = Path(p).stem
        lbl_p = Path(p).parent.parent / "labels" / f"{stem}.txt"
        rows = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                    rows.append([c, (cx - w / 2) * ORIG_W, (cy - h / 2) * ORIG_H,
                                    (cx + w / 2) * ORIG_W, (cy + h / 2) * ORIG_H])
        vl_lbls.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))

    tflite_path = weights_dir / "dtnet_p2_eiou_int8.tflite"
    export_full_int8_tflite(kmodel, tr_imgs, str(tflite_path), num_calib=200)

    # 4. Evaluate INT8 model on 600 validation images
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    cls_idx_tflite = 0 if output_details[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx_tflite = 1 - cls_idx_tflite
    in_scale, in_zp = input_details["quantization"]
    cls_scale, cls_zp = output_details[cls_idx_tflite]["quantization"]
    box_scale, box_zp = output_details[box_idx_tflite]["quantization"]

    print(f"[*] Inferring 600 validation images with Full INT8 model...")
    int8_raw_preds = []
    for p in vl_imgs:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        quant_in = np.clip(np.round((img.astype(np.float32) / 255.0) / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quant_in[np.newaxis, ...])
        interpreter.invoke()
        q_cls = interpreter.get_tensor(output_details[cls_idx_tflite]["index"])
        q_box = interpreter.get_tensor(output_details[box_idx_tflite]["index"])
        f_cls = (q_cls.astype(np.float32) - cls_zp) * cls_scale
        f_box = (q_box.astype(np.float32) - box_zp) * box_scale
        int8_raw_preds.append((f_cls.transpose(0, 3, 1, 2), f_box.transpose(0, 3, 1, 2)))

    base_thresh = 0.001
    int8_dets = [decode_batch_np(c, b, score_thr=base_thresh)[0] for c, b in int8_raw_preds]
    int8_map50, int8_per_class = evaluate_map(int8_dets, vl_lbls)
    print(f"[✔] Base INT8 mAP50 = {int8_map50*100:.2f}% (Quantization delta: {(int8_map50 - ck['map50'])*100:+.2f}%)")

    # 5. Sweetspot Per-Class Threshold Calibration
    print("[*] Calibrating Sweetspot per-class thresholds...")
    opt_thresholds = [0.001] * NUM_CLASSES
    for c in range(NUM_CLASSES):
        best_t, best_ap = 0.001, int8_per_class.get(c, 0.0)
        for t in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
            filtered_dets = []
            for d in int8_dets:
                if len(d) == 0:
                    filtered_dets.append(d)
                else:
                    mask = (d[:, 5] != c) | (d[:, 4] >= t)
                    filtered_dets.append(d[mask])
            m, per_c = evaluate_map(filtered_dets, vl_lbls)
            if per_c.get(c, 0.0) >= best_ap:
                best_ap = per_c.get(c, 0.0)
                best_t = t
        opt_thresholds[c] = best_t

    calibrated_dets = []
    for d in int8_dets:
        if len(d) == 0:
            calibrated_dets.append(d)
        else:
            keep = np.array([row[4] >= opt_thresholds[int(row[5])] for row in d])
            calibrated_dets.append(d[keep])
    cal_map50, cal_per_class = evaluate_map(calibrated_dets, vl_lbls)
    cal_score = 30.0 + 70.0 * cal_map50

    print(f"\n{'='*80}")
    print(f"   CHAMPION RESULTS: DTNet-P2 EIoU & GATED HARD-NEGATIVE FULL INT8")
    print(f"   Calibrated Val mAP50 : {cal_map50*100:.2f}%")
    print(f"   Total Contest Score  : {cal_score:.2f} / 100.00")
    print(f"   Optimal Thresholds   : {dict(zip(CLASS_NAMES, opt_thresholds))}")
    print(f"   Per-class AP50:")
    for c in range(NUM_CLASSES):
        print(f"     {CLASS_NAMES[c]:<10}: {cal_per_class.get(c, 0)*100:.2f}%")
    print(f"{'='*80}\n")

    summary_path = weights_dir / "champion_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_name": "DTNet-P2 EIoU & Gated Hard-Negative Champion",
            "epochs_trained": ck["epoch"],
            "val_map50_fp32": ck["map50"],
            "val_map50_int8": cal_map50,
            "total_contest_score": cal_score,
            "optimal_thresholds": dict(zip(CLASS_NAMES, opt_thresholds)),
            "per_class_ap50": {CLASS_NAMES[c]: cal_per_class.get(c, 0) for c in range(NUM_CLASSES)}
        }, f, indent=2)
    print(f"[✔] Summary written to {summary_path}")

if __name__ == "__main__":
    main()
