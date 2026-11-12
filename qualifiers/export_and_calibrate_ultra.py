#!/usr/bin/env python3
"""
EXPORT & CALIBRATE DTNet-P2-ULTRA TO FULL INTEGER INT8 FOR ESP32-S3
Evaluates on qualifiers/data/val/images, tunes per-class thresholds, and exports champion INT8 model.
"""
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

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from qualifiers.train_dtnet_p2_ultra import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    DTNetMax as DTNet, build_keras_dtnet_max as build_keras_dtnet,
    export_full_int8_tflite, decode_batch_np, evaluate_map
)

def main():
    weights_dir = PROJECT_ROOT / "qualifiers/weights/dtnet_p2_ultra"
    pt_path = weights_dir / "best_p2_ultra.pt"
    data_dir = PROJECT_ROOT / "qualifiers/data"

    print("=" * 80)
    print("   EXPORTING & CALIBRATING DTNet-P2 ULTRA TO FULL INT8 (TARGET 99.5%)")
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

    # Transfer weights
    npy = lambda x: x.detach().cpu().numpy()
    for op in spec:
        if op["t"] not in ("conv", "dw"):
            continue
        m = model.mods[op["name"]]
        w = npy(m.weight)
        ws = [w.transpose(2, 3, 1, 0) if op["t"] == "conv" else w.transpose(2, 3, 0, 1)]
        if op["bias"]:
            ws.append(npy(m.bias))
        kmodel.get_layer(op["name"]).set_weights(ws)
        if op["bn"]:
            bn = model.mods[op["name"] + "_bn"]
            kmodel.get_layer(op["name"] + "_bn").set_weights(
                [npy(bn.weight), npy(bn.bias), npy(bn.running_mean), npy(bn.running_var)])

    # Verification of parity
    test_x = np.random.uniform(0, 1, (2, IMG_H, IMG_W, 3)).astype(np.float32)
    with torch.no_grad():
        pt_in = torch.from_numpy(test_x.transpose(0, 3, 1, 2))
        tc, tb = model(pt_in)
    kc, kb = kmodel(test_x, training=False)
    diff_c = float(np.abs(tc.numpy() - np.asarray(kc).transpose(0, 3, 1, 2)).max())
    diff_b = float(np.abs(tb.numpy() - np.asarray(kb).transpose(0, 3, 1, 2)).max())
    print(f"[✔] PyTorch <-> Keras verification max error: cls={diff_c:.2e}, box={diff_b:.2e}")
    assert diff_c < 0.02 and diff_b < 0.02, "Keras model does not match PyTorch weights!"

    # 3. Export to Full Integer INT8 TFLite Micro
    tflite_path = weights_dir / "dtnet_p2_ultra_int8.tflite"
    train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) +
                        glob.glob(str(data_dir / "train/images/*.jpg")))
    export_full_int8_tflite(kmodel, train_imgs, str(tflite_path), num_calib=200)

    # 4. INT8 TFLite Inference & Evaluation on Validation Set
    print("\n[*] Running INT8 TFLite inference across validation set...")
    val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) +
                      glob.glob(str(data_dir / "val/images/*.jpg")))
    vl_lbls = []
    for p in val_imgs:
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

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()

    cls_idx = 0 if out_dets[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx = 1 - cls_idx
    cls_det = out_dets[cls_idx]
    box_det = out_dets[box_idx]

    in_scale, in_zp = in_det["quantization"]
    cls_scale, cls_zp = cls_det["quantization"]
    box_scale, box_zp = box_det["quantization"]

    raw_preds = []
    print(f"[*] Inferencing {len(val_imgs)} validation images...")
    t0 = time.time()
    for img_path in val_imgs:
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (IMG_W, IMG_H):
            rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        norm_img = rgb.astype(np.float32) / 255.0

        # Quantize input
        q_in = np.clip(np.round(norm_img / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], q_in[None, ...])
        interp.invoke()

        # Dequantize output
        q_cls = interp.get_tensor(cls_det["index"])[0].astype(np.float32)
        q_box = interp.get_tensor(box_det["index"])[0].astype(np.float32)
        dq_cls = (q_cls - cls_zp) * cls_scale
        dq_box = (q_box - box_zp) * box_scale

        # Transpose to (1, C, gh, gw)
        raw_preds.append((dq_cls.transpose(2, 0, 1)[None, ...], dq_box.transpose(2, 0, 1)[None, ...]))

    inf_time = time.time() - t0
    print(f"[✔] INT8 Inference completed in {inf_time:.1f}s ({inf_time / len(val_imgs) * 1000:.1f} ms/image)")

    # 5. Grid Search for Sweetspot Per-Class Thresholds
    print("\n[*] Optimizing per-class calibrated thresholds & NMS IoU...")
    best_map = 0.0
    best_nms_iou = 0.50
    for nms_iou in [0.45, 0.50, 0.55, 0.60]:
        dets = [decode_batch_np(c, b, score_thr=0.001, nms_iou=nms_iou)[0] for c, b in raw_preds]
        m, per_c = evaluate_map(dets, vl_lbls)
        print(f"    Raw base mAP @ NMS IoU {nms_iou:.2f}: {m * 100:.2f}%")
        if m > best_map:
            best_map = m
            best_nms_iou = nms_iou

    # Per-class coordinate descent
    base_dets = [decode_batch_np(c, b, score_thr=0.001, nms_iou=best_nms_iou)[0] for c, b in raw_preds]
    thresh_cands = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    opt_thresh = [0.001] * NUM_CLASSES
    _, base_per_c = evaluate_map(base_dets, vl_lbls)

    for c in range(NUM_CLASSES):
        best_t, best_ap = 0.001, base_per_c.get(c, 0.0)
        for t in thresh_cands:
            filtered = []
            for d in base_dets:
                if len(d) == 0:
                    filtered.append(d)
                else:
                    mask = (d[:, 5] != c) | (d[:, 4] >= t)
                    filtered.append(d[mask])
            m, per_c = evaluate_map(filtered, vl_lbls)
            if per_c.get(c, 0.0) >= best_ap:
                best_ap = per_c.get(c, 0.0)
                best_t = t
        opt_thresh[c] = best_t
        print(f"    - Optimized {CLASS_NAMES[c]:8s}: thr={best_t:.3f} -> AP50: {best_ap * 100:.2f}%")

    # Final evaluation with calibrated thresholds
    final_dets = []
    for d in base_dets:
        if len(d) == 0:
            final_dets.append(d)
        else:
            mask = np.ones(len(d), dtype=bool)
            for c in range(NUM_CLASSES):
                mask &= (d[:, 5] != c) | (d[:, 4] >= opt_thresh[c])
            final_dets.append(d[mask])
    final_map, final_per_c = evaluate_map(final_dets, vl_lbls)

    print("\n" + "=" * 80)
    print(f"   [★ DTNet-P2-ULTRA INT8 EVALUATION RESULT]")
    print(f"   Calibrated Val mAP50 : {final_map * 100:.2f}%")
    print(f"   Contest Score (Local): {70.0 * final_map + 30.0:.2f} (assuming 100% intent)")
    print("   Per-Class AP50:")
    for c in range(NUM_CLASSES):
        print(f"       {CLASS_NAMES[c]:8s}: {final_per_c.get(c, 0.0) * 100:.2f}% (thr={opt_thresh[c]:.3f})")
    print("=" * 80)

    # Save summary json
    cal_dict = {CLASS_NAMES[c]: opt_thresh[c] for c in range(NUM_CLASSES)}
    summary = {
        "model": "DTNet-P2-Ultra",
        "int8_tflite_path": str(tflite_path),
        "int8_size_kb": len(open(tflite_path, "rb").read()) / 1024.0,
        "best_nms_iou": best_nms_iou,
        "calibrated_thresholds": cal_dict,
        "val_map50": final_map,
        "per_class_ap50": {CLASS_NAMES[c]: final_per_c.get(c, 0.0) for c in range(NUM_CLASSES)}
    }
    with open(weights_dir / "ultra_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✔] Saved calibration summary: {weights_dir / 'ultra_summary.json'}")

if __name__ == "__main__":
    main()
