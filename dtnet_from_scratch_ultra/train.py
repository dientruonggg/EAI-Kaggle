#!/usr/bin/env python3
"""
train.py
================================================================================
Full-Topping From-Scratch Trainer for DTNetMax (ESP32-S3 Edge AI Challenge 2026)
--------------------------------------------------------------------------------
Incorporates every allowable technique from BTC guidelines:
1. Architecture: DTNetMax (197,647 strict parameters < 200,000 hardware limit)
2. Target Assignment: Anti-Nerf Soft Border (0.25 border weight) + Area Adaptive Radius
3. Loss: Small-Object Scale Weighting + Hard-Class Priority (book, toy_car, mouse) + EIoU + QFL
4. Augmentation: Mosaic (p=0.65) + HSV Jitter + Zoom-out Affine + Sensor Noise Simulation
5. Schedule: 2-Phase Training with Close-Mosaic at the final 35 epochs
6. Stability: ModelEMA (0.9998) + Cosine Annealing with Warmup + Gradient Clipping
================================================================================
"""

import argparse
import glob
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

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dtnet_from_scratch_ultra.dtnet_max import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, count_params_strict, make_spec_max
)
from dtnet_from_scratch_ultra.dataset import UltraDetDataset
from dtnet_from_scratch_ultra.loss import det_loss_ultra
from dtnet_from_scratch_ultra.evaluation import decode_batch_np, evaluate_map, ModelEMA


def load_dataset_split(data_dir):
    """Loads images and YOLO labels from pooled dataset with deterministic 9:1 split (Exact Parity)."""
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

    print(f"[✔] Pooled 6,000 images & 9:1 Partition: {len(tr_imgs)} Train images | {len(vl_imgs)} Val images from {data_dir}")
    return tr_imgs, tr_lbls, vl_imgs, vl_lbls


def main():
    parser = argparse.ArgumentParser(description="DTNetMax From-Scratch Full-Topping Training")
    parser.add_argument("--epochs", type=int, default=400, help="Total training epochs (default: 400)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak learning rate for batch 32 (default: 2e-3)")
    parser.add_argument("--w_box", type=float, default=5.5, help="Base box loss weight (default: 5.5)")
    parser.add_argument("--w_neg", type=float, default=1.5, help="Negative background multiplier (default: 1.5)")
    parser.add_argument("--iou_gate", type=float, default=0.48, help="IoU gate for QFL (default: 0.48)")
    parser.add_argument("--close_mosaic", type=int, default=35, help="Epochs before end to turn off Mosaic (default: 35)")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 to disable, default: 0)")
    parser.add_argument("--data_dir", type=str, default="qualifiers/data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="qualifiers/weights/dtnet_from_scratch_ultra", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = output_dir / "best_dtnet_ultra.pt"
    latest_pt_path = output_dir / "latest_dtnet_ultra.pt"

    # 1. Dataset & Loaders
    tr_imgs, tr_lbls, vl_imgs, vl_lbls = load_dataset_split(args.data_dir)
    train_ds = UltraDetDataset(tr_imgs, tr_lbls, IMG_W, IMG_H, STRIDE, train=True, mosaic_p=0.65, seed=42)
    val_ds = UltraDetDataset(vl_imgs, vl_lbls, IMG_W, IMG_H, STRIDE, train=False, mosaic_p=0.0, seed=42)

    num_workers = max(1, min(args.num_workers, os.cpu_count() or 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # 2. Model & Strict Hardware Invariant Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec_max()
    model = DTNetMax(spec).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params = count_params_strict(model)
    print(f"\n[Hardware Gate] DTNetMax Strict Parameters: {total_params:,} / {MAX_ALLOWED_PARAMS:,}")
    print(f"[Hardware Gate] Margin to limit: {MAX_ALLOWED_PARAMS - total_params:,} parameters")
    assert total_params <= MAX_ALLOWED_PARAMS, f"Fatal: Exceeded parameter limit {total_params} > {MAX_ALLOWED_PARAMS}"

    ema = ModelEMA(model, decay=0.9998)

    # 3. Optimizer & LR Scheduler
    lr_eff = args.lr * (args.batch_size / 32.0)
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if p.ndim > 1 else no_decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 1e-4},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr_eff, betas=(0.9, 0.999)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_epochs = 3
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    close_mosaic_epoch = args.epochs - args.close_mosaic

    def get_lr(step):
        if step < warmup_steps:
            return lr_eff * (0.01 + 0.99 * step / warmup_steps)
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_eff * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    start_epoch = 0
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming checkpoint from: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["ema"])
        ema.ema.load_state_dict(ck["ema"])
        start_epoch = ck.get("epoch", 0)
        best_map50 = ck.get("map50", 0.0)
        best_epoch = ck.get("epoch", 0)
        print(f"[✔] Resumed: Start Epoch {start_epoch}, Baseline mAP50: {best_map50*100:.2f}%")

    print(f"\n[*] Commencing Training on {device} (Epochs {start_epoch}->{args.epochs}, Close-Mosaic at Ep {close_mosaic_epoch})...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        if epoch >= close_mosaic_epoch:
            train_ds.set_mosaic(0.0)  # Close Mosaic for native resolution fine-tuning

        w_box_curr = args.w_box + (1.0 if epoch >= close_mosaic_epoch else 0.0)
        lam = min(1.0, (epoch + 1) / 20.0)

        loss_cls_accum, loss_box_accum, total_loss_accum = 0.0, 0.0, 0.0
        for step, (imgs, cls_idx, box_t, _) in enumerate(train_loader):
            current_step = epoch * steps_per_epoch + step
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
                loss, l_cls, l_box = det_loss_ultra(
                    cls_logit, box_raw, cls_idx, box_t,
                    lam=lam, w_cls=1.0, w_box=w_box_curr,
                    qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=args.w_neg,
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

        # Validation with EMA weights
        ema.ema.eval()
        val_cls_accum, val_box_accum, val_loss_accum = 0.0, 0.0, 0.0
        dets_val = []
        with torch.no_grad():
            for imgs, cls_idx, box_t, _ in val_loader:
                x = imgs.to(device, non_blocking=True).float().div_(255.0)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                cls_idx = cls_idx.to(device, non_blocking=True)
                box_t = box_t.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    co, bo = ema.ema(x)
                    vl, v_cls, v_box = det_loss_ultra(
                        co, bo, cls_idx, box_t,
                        lam=1.0, w_cls=1.0, w_box=w_box_curr,
                        qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=args.w_neg,
                        iou_gate=args.iou_gate
                    )
                val_cls_accum += v_cls.item()
                val_box_accum += v_box.item()
                val_loss_accum += vl.item()
                outs = decode_batch_np(co.float().cpu().numpy(), bo.float().cpu().numpy(), score_thr=0.001)
                dets_val.extend(outs)

        val_map50, per_class = evaluate_map(dets_val, vl_lbls)
        val_cls = val_cls_accum / len(val_loader)
        val_box = val_box_accum / len(val_loader)
        val_loss = val_loss_accum / len(val_loader)

        is_best = val_map50 > best_map50
        if is_best:
            best_map50 = val_map50
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "ema": ema.ema.state_dict(),
                "spec": spec,
                "map50": val_map50,
                "per_class_ap": per_class,
            }, best_pt_path)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch + 1,
            "ema": ema.ema.state_dict(),
            "spec": spec,
            "map50": val_map50,
            "per_class_ap": per_class,
        }, latest_pt_path)

        dt = time.time() - t0
        ms_per_step = (dt / steps_per_epoch) * 1000.0
        print(f"Epoch {epoch+1}/{args.epochs} [{int(dt)}s, {int(ms_per_step)}ms/step] "
              f"loss: {tr_total:.4f} (cls: {tr_cls:.4f}, box: {tr_box:.4f}) | "
              f"val_loss: {val_loss:.4f} | val_mAP50: {val_map50*100:.2f}% | lr: {current_lr:.4e}", flush=True)

        if is_best:
            print(f"[★ NEW BEST] Val mAP50: {val_map50*100:.2f}% (Epoch {best_epoch})", flush=True)
            ap_strs = [f"{CLASS_NAMES[c]}: {per_class.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)]
            print(f"    Per-class: {' | '.join(ap_strs)}", flush=True)

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[!] Early stopping after {args.patience} epochs without improvement.", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"\n[✔] Training completed in {elapsed/60:.1f} minutes. Best mAP50: {best_map50*100:.2f}% at Epoch {best_epoch}.")
    print(f"[✔] Best checkpoint saved to: {best_pt_path}")


if __name__ == "__main__":
    main()
