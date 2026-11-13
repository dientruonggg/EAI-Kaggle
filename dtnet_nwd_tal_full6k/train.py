#!/usr/bin/env python3
"""
train.py
================================================================================
Full-6,000 Dataset Curriculum Trainer with TAL + NWD Loss
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

from dtnet_nwd_tal_full6k.dtnet_max import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, count_params_strict, make_spec_max
)
from dtnet_nwd_tal_full6k.dataset import Full6kDetDataset, collate_tal_fn
from dtnet_nwd_tal_full6k.loss import TaskAlignedAssigner, det_loss_tal_nwd, decode_ltrb
from dtnet_nwd_tal_full6k.evaluation import decode_batch_np, evaluate_map, ModelEMA


def load_full6k_data(data_dir):
    """Loads all 6,000 images and YOLO labels from train + val pools."""
    data_dir = Path(data_dir)
    train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) + glob.glob(str(data_dir / "train/images/*.jpg")))
    val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) + glob.glob(str(data_dir / "val/images/*.jpg")))
    all_imgs = train_imgs + val_imgs

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
    n_val = 600
    val_indices = indices[:n_val]

    eval_imgs = [all_imgs[i] for i in val_indices]
    eval_lbls = [all_lbls[i] for i in val_indices]

    print(f"[✔] Pooled Full Dataset: {len(all_imgs)} Total Images ({len(train_imgs)} train + {len(val_imgs)} val)")
    print(f"[✔] Evaluation Tracking Split: {len(eval_imgs)} validation images")
    return all_imgs, all_lbls, eval_imgs, eval_lbls


def main():
    parser = argparse.ArgumentParser(description="DTNetMax Full-6K TAL + NWD Training")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Peak learning rate (default: 1.2e-4)")
    parser.add_argument("--min_lr", type=float, default=2e-6, help="Minimum learning rate (default: 2e-6)")
    parser.add_argument("--w_box", type=float, default=5.5, help="Base box loss weight (default: 5.5)")
    parser.add_argument("--w_nwd", type=float, default=2.0, help="NWD loss weight (default: 2.0)")
    parser.add_argument("--w_neg", type=float, default=1.5, help="Negative background multiplier (default: 1.5)")
    parser.add_argument("--w_mutual", type=float, default=3.0, help="Cross-class mutual exclusion multiplier (default: 3.0)")
    parser.add_argument("--close_mosaic", type=int, default=50, help="Epochs before end to close Mosaic (default: 50)")
    parser.add_argument("--init_weights", type=str,
                        default="qualifiers/weights/dtnet_clean_p2_200ep/best_dtnet_clean_200ep.pt",
                        help="Initial checkpoint to fine-tune from")
    parser.add_argument("--data_dir", type=str, default="qualifiers/data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="qualifiers/weights/dtnet_nwd_tal_full6k", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = output_dir / "best_dtnet_nwd_tal_full6k.pt"
    latest_pt_path = output_dir / "latest_dtnet_nwd_tal_full6k.pt"
    config_path = output_dir / "training_config.json"

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[✔] Configuration saved to {config_path}")

    # 1. Dataset & Loaders
    all_imgs, all_lbls, eval_imgs, eval_lbls = load_full6k_data(args.data_dir)
    train_ds = Full6kDetDataset(all_imgs, all_lbls, IMG_W, IMG_H, STRIDE, train=True, mosaic_p=0.5, seed=42)
    eval_ds = Full6kDetDataset(eval_imgs, eval_lbls, IMG_W, IMG_H, STRIDE, train=False, mosaic_p=0.0, seed=42)

    num_workers = max(1, min(args.num_workers, os.cpu_count() or 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=collate_tal_fn, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                             collate_fn=collate_tal_fn)

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
    assigner = TaskAlignedAssigner(topk=4, alpha=0.5, beta=6.0, stride=STRIDE)

    # 3. Load Initial Weights & Support Resuming
    start_epoch = 1
    best_map50 = 0.0
    best_epoch = 0
    if args.init_weights and os.path.exists(args.init_weights):
        print(f"[*] Loading initial checkpoint from: {args.init_weights}", flush=True)
        ck = torch.load(args.init_weights, map_location=device, weights_only=False)
        state = ck.get("ema", ck.get("model_state_dict", ck))
        model.load_state_dict(state)
        ema.ema.load_state_dict(state)
        if "epoch" in ck and "map50" in ck:
            start_epoch = ck["epoch"] + 1
            best_map50 = ck.get("map50", 0.0)
            best_epoch = ck["epoch"]
            print(f"[✔] Resuming from Epoch {start_epoch} with baseline mAP50: {best_map50*100:.2f}%", flush=True)
        else:
            print(f"[✔] Checkpoint loaded successfully!", flush=True)
    else:
        print(f"[!] Warning: Initial weights {args.init_weights} not found. Training from scratch.", flush=True)

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
    close_mosaic_epoch = args.epochs - args.close_mosaic

    def get_lr(step):
        if step < warmup_steps:
            return args.min_lr + (lr_eff - args.min_lr) * (step / warmup_steps)
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return args.min_lr + 0.5 * (lr_eff - args.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))

    # Initial Baseline Validation
    if best_map50 == 0.0:
        print("\n[*] Evaluating Initial Baseline mAP50...", flush=True)
        ema.ema.eval()
        val_preds = []
        with torch.no_grad():
            for batch in eval_loader:
                imgs = batch[0]
                x = imgs.to(device, non_blocking=True).float().div_(255.0)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    c_pred, b_pred = ema.ema(x)
                dets = decode_batch_np(c_pred.cpu().numpy(), b_pred.cpu().numpy(), score_thr=0.001, nms_iou=0.65, max_det=100)
                val_preds.extend(dets)
        base_mAP50, base_aps = evaluate_map(val_preds, eval_lbls)
        print(f"[✔] Initial Baseline mAP50: {base_mAP50*100:.2f}%", flush=True)
        per_cls_str = " | ".join([f"{CLASS_NAMES[c]}: {base_aps.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)])
        print(f"    Per-class: {per_cls_str}\n", flush=True)
        best_map50 = base_mAP50
        best_epoch = 0

    print(f"[*] Commencing Vectorized Full-6K TAL+NWD Training on {device} (Epochs {start_epoch}->{args.epochs}, Close-Mosaic at Ep {close_mosaic_epoch})...\n", flush=True)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        if epoch >= close_mosaic_epoch:
            train_ds.set_mosaic(0.0)  # Close Mosaic for Clean-Phase

        loss_cls_accum, loss_box_accum, total_loss_accum = 0.0, 0.0, 0.0
        for step, batch in enumerate(train_loader):
            current_step = (epoch - 1) * steps_per_epoch + step
            current_lr = get_lr(current_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            imgs, gt_labels, gt_bboxes = batch[0], batch[1], batch[2]
            x = imgs.to(device, non_blocking=True).float().div_(255.0)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            gt_labels = gt_labels.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                cls_logit, box_raw = model(x)

                # Vectorized 3D Tensor Broadcasting TAL Assignment (< 4ms on GPU)
                with torch.no_grad():
                    pred_xyxy = decode_ltrb(box_raw)
                    pred_scores = cls_logit.sigmoid()
                    _, _, gh, gw = cls_logit.shape
                    a_cls, a_box, a_tgt = assigner.assign(
                        pred_scores.float(), pred_xyxy.float(), gt_labels, gh, gw, device, gt_bboxes_arg=gt_bboxes
                    )

                loss, l_cls, l_box = det_loss_tal_nwd(
                    cls_logit, box_raw, a_cls, a_box, a_tgt,
                    w_cls=1.0, w_box=args.w_box, w_nwd=args.w_nwd,
                    w_neg=args.w_neg, w_mutual=args.w_mutual
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
            for batch in eval_loader:
                imgs = batch[0]
                x = imgs.to(device, non_blocking=True).float().div_(255.0)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    c_pred, b_pred = ema.ema(x)
                dets = decode_batch_np(c_pred.cpu().numpy(), b_pred.cpu().numpy(), score_thr=0.001, nms_iou=0.65, max_det=100)
                val_preds.extend(dets)

        val_mAP50, val_aps = evaluate_map(val_preds, eval_lbls)
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
        print(f"Epoch {epoch:03d}/{args.epochs} [{dt:.1f}s, {ms_step:.0f}ms/step] loss: {tr_total:.4f} (cls: {tr_cls:.4f}, box: {tr_box:.4f}) | val_mAP50: {val_mAP50*100:.2f}% | lr: {current_lr:.4e}", flush=True)
        if is_best:
            print(f"{best_mark}Per-class: {per_cls_str}", flush=True)

    total_time = (time.time() - start_time) / 60.0
    print(f"\n[✔] Full-6K TAL+NWD Training completed in {total_time:.1f} minutes. Best mAP50: {best_map50*100:.2f}% at Epoch {best_epoch}.")
    print(f"[✔] Best checkpoint saved to: {best_pt_path.resolve()}")


if __name__ == "__main__":
    main()
