#!/usr/bin/env python3
"""
train.py
================================================================================
Clean-Phase 200-Epoch Fine-Tuning Pipeline for DTNetMax
================================================================================
Designed to push mAP50 from 98.42% towards 99.2%+ under strict contest invariants:
1. Architecture: DTNetMax (197,647 strict parameters <= 200,000 hardware limit).
2. Clean Resolution Training: Native 320x240 RGB with Zero-Mosaic (no border crop).
3. Anti-False-Positive Loss (AFPL):
   - Scale-Adaptive EIoU for small objects (mouse, toy_car).
   - Hard-Class Priority Multipliers (toy_car 1.40x, book 1.35x, mouse 1.25x).
   - Cross-Class Mutual Exclusion Penalty (w_mutual = 3.0) to eliminate ghost predictions.
4. ModelEMA (decay 0.9998) + Cosine LR Schedule (1.5e-4 -> 2e-6).
================================================================================
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtnet_clean_p2_200ep.dtnet_max import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, count_params_strict, make_spec_max
)
from dtnet_clean_p2_200ep.dataset import CleanDetDataset
from dtnet_clean_p2_200ep.loss import det_loss_clean_afpl
from dtnet_clean_p2_200ep.evaluation import decode_batch_np, evaluate_map, ModelEMA


def load_dataset_split(data_dir):
    """Loads images and YOLO labels from pooled dataset with deterministic 9:1 split."""
    data_dir = Path(data_dir)
    all_train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) + glob.glob(str(data_dir / "train/images/*.jpg")))
    all_val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) + glob.glob(str(data_dir / "val/images/*.jpg")))
    all_imgs = all_train_imgs + all_val_imgs

    all_lbls = []
    for p in all_imgs:
        stem = Path(p).stem
        parent = Path(p).parent.parent
        lbl_p = parent / "labels" / f"{stem}.txt"
        rows = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2.0) * ORIG_W
                        y1 = (cy - h / 2.0) * ORIG_H
                        x2 = (cx + w / 2.0) * ORIG_W
                        y2 = (cy + h / 2.0) * ORIG_H
                        rows.append([c, x1, y1, x2, y2])
        all_lbls.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_imgs))
    n_val = int(len(all_imgs) * 0.1)  # Exactly 600 validation images
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    tr_imgs = [all_imgs[i] for i in train_indices]
    tr_lbls = [all_lbls[i] for i in train_indices]
    vl_imgs = [all_imgs[i] for i in val_indices]
    vl_lbls = [all_lbls[i] for i in val_indices]

    print(f"[✔] Deterministic 9:1 Partition: {len(tr_imgs)} Train images | {len(vl_imgs)} Val images from {data_dir}")
    return tr_imgs, tr_lbls, vl_imgs, vl_lbls


def main():
    parser = argparse.ArgumentParser(description="DTNet Clean-Phase 200-Epoch Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate (default: 1.5e-4)")
    parser.add_argument("--min_lr", type=float, default=2e-6, help="Minimum learning rate (default: 2e-6)")
    parser.add_argument("--w_box", type=float, default=5.5, help="Base box loss weight (default: 5.5)")
    parser.add_argument("--w_neg", type=float, default=1.5, help="Negative background multiplier (default: 1.5)")
    parser.add_argument("--w_mutual", type=float, default=3.0, help="Cross-class mutual exclusion multiplier (default: 3.0)")
    parser.add_argument("--iou_gate", type=float, default=0.48, help="IoU gate for QFL (default: 0.48)")
    parser.add_argument("--init_weights", type=str, default="qualifiers/weights/dtnet_p2_ultra_finetune/best_p2_ultra_finetune.pt",
                        help="Initial checkpoint to fine-tune from")
    parser.add_argument("--data_dir", type=str, default="qualifiers/data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="qualifiers/weights/dtnet_clean_p2_200ep", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = output_dir / "best_dtnet_clean_200ep.pt"
    latest_pt_path = output_dir / "latest_dtnet_clean_200ep.pt"
    config_path = output_dir / "training_config.json"

    # Save configuration for full reproducibility
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[✔] Configuration saved to {config_path}")

    # 1. Dataset & Loaders
    tr_imgs, tr_lbls, vl_imgs, vl_lbls = load_dataset_split(args.data_dir)
    train_ds = CleanDetDataset(tr_imgs, tr_lbls, IMG_W, IMG_H, STRIDE, train=True, mosaic_p=0.0, seed=42)
    val_ds = CleanDetDataset(vl_imgs, vl_lbls, IMG_W, IMG_H, STRIDE, train=False, mosaic_p=0.0, seed=42)

    num_workers = max(1, min(args.num_workers, os.cpu_count() or 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # 2. Model & Hardware Invariant Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec_max()
    model = DTNetMax(spec).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params = count_params_strict(model)
    print(f"\n[Hardware Gate] Strict Parameters: {total_params:,} / {MAX_ALLOWED_PARAMS:,}")
    assert total_params <= MAX_ALLOWED_PARAMS, f"Fatal: Exceeded parameter limit {total_params} > {MAX_ALLOWED_PARAMS}"

    ema = ModelEMA(model, decay=0.9998)

    # 3. Load Initial Weights
    if args.init_weights and os.path.exists(args.init_weights):
        print(f"[*] Loading initial checkpoint from: {args.init_weights}")
        ck = torch.load(args.init_weights, map_location=device, weights_only=False)
        state = ck.get("ema", ck.get("model_state_dict", ck))
        model.load_state_dict(state)
        ema.ema.load_state_dict(state)
        print(f"[✔] Checkpoint loaded successfully!")
    else:
        print(f"[!] Warning: Initial weights {args.init_weights} not found. Training from scratch.")

    # 4. Optimizer & Schedule
    lr_eff = args.lr * (args.batch_size / 32.0)
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if p.ndim > 1 else no_decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 5e-5},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr_eff, betas=(0.9, 0.999)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_epochs = 2
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def get_lr(step):
        if step < warmup_steps:
            return args.min_lr + (lr_eff - args.min_lr) * (step / warmup_steps)
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return args.min_lr + 0.5 * (lr_eff - args.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))

    # Initial Baseline Validation
    print("\n[*] Evaluating Initial Baseline mAP50...")
    ema.ema.eval()
    val_preds = []
    with torch.no_grad():
        for imgs, _, _, _ in val_loader:
            x = imgs.to(device, non_blocking=True).float().div_(255.0)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                c_pred, b_pred = ema.ema(x)
            dets = decode_batch_np(c_pred.cpu().numpy(), b_pred.cpu().numpy(), score_thr=0.001, nms_iou=0.65, max_det=100)
            val_preds.extend(dets)
    base_mAP50, base_aps = evaluate_map(val_preds, vl_lbls)
    print(f"[✔] Initial Baseline mAP50: {base_mAP50*100:.2f}%")
    per_cls_str = " | ".join([f"{CLASS_NAMES[c]}: {base_aps.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)])
    print(f"    Per-class: {per_cls_str}\n")

    best_map50 = base_mAP50
    best_epoch = 0

    print(f"[*] Commencing Clean-Phase 200-Epoch Fine-Tuning on {device}...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)

        loss_cls_accum, loss_box_accum, total_loss_accum = 0.0, 0.0, 0.0
        for step, (imgs, cls_idx, box_t, _) in enumerate(train_loader):
            current_step = (epoch - 1) * steps_per_epoch + step
            current_lr = get_lr(current_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            x = imgs.to(device, non_blocking=True).float().div_(255.0)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            cls_idx = cls_idx.to(device, non_blocking=True)
            box_t = box_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                cls_logit, box_raw = model(x)
                loss, l_cls, l_box = det_loss_clean_afpl(
                    cls_logit, box_raw, cls_idx, box_t,
                    lam=1.0, w_cls=1.0, w_box=args.w_box,
                    qfl_beta_pos=2.0, qfl_beta_neg=1.2,
                    w_neg=args.w_neg, w_mutual=args.w_mutual,
                    iou_gate=args.iou_gate
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            total_loss_accum += loss.item()
            loss_cls_accum += l_cls.item()
            loss_box_accum += l_box.item()

        tr_total = total_loss_accum / steps_per_epoch
        tr_cls = loss_cls_accum / steps_per_epoch
        tr_box = loss_box_accum / steps_per_epoch

        # Validation at epoch end
        ema.ema.eval()
        val_preds = []
        with torch.no_grad():
            for imgs, _, _, _ in val_loader:
                x = imgs.to(device, non_blocking=True).float().div_(255.0)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    c_pred, b_pred = ema.ema(x)
                dets = decode_batch_np(c_pred.cpu().numpy(), b_pred.cpu().numpy(), score_thr=0.001, nms_iou=0.65, max_det=100)
                val_preds.extend(dets)

        val_mAP50, val_aps = evaluate_map(val_preds, vl_lbls)
        dt = time.time() - t0
        ms_step = (dt / steps_per_epoch) * 1000.0

        is_best = val_mAP50 > best_map50
        if is_best:
            best_map50 = val_mAP50
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "ema": ema.ema.state_dict(),
                "spec": spec,
                "map50": best_map50,
                "per_class_ap": val_aps
            }, best_pt_path)

        torch.save({
            "epoch": epoch,
            "ema": ema.ema.state_dict(),
            "spec": spec,
            "map50": val_mAP50,
            "per_class_ap": val_aps
        }, latest_pt_path)

        best_mark = f"[★ NEW BEST] Val mAP50: {val_mAP50*100:.2f}% (Epoch {epoch})\n    " if is_best else ""
        per_cls_str = " | ".join([f"{CLASS_NAMES[c]}: {val_aps.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)])
        print(f"Epoch {epoch:03d}/{args.epochs} [{dt:.0f}s, {ms_step:.0f}ms/step] loss: {tr_total:.4f} (cls: {tr_cls:.4f}, box: {tr_box:.4f}) | val_mAP50: {val_mAP50*100:.2f}% | lr: {current_lr:.4e}")
        if is_best:
            print(f"{best_mark}Per-class: {per_cls_str}")

    total_time = (time.time() - start_time) / 60.0
    print(f"\n[✔] Fine-Tuning completed in {total_time:.1f} minutes. Best mAP50: {best_map50*100:.2f}% at Epoch {best_epoch}.")
    print(f"[✔] Best checkpoint saved to: {best_pt_path.resolve()}")


if __name__ == "__main__":
    main()
