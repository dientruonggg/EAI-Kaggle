#!/usr/bin/env python3
"""
DTNet-P2-Ultra Hard-Class & Small-Object Adaptive Fine-Tuning Pipeline (100 Epochs)
- Loads best_p2_ultra.pt (98.18% mAP50 base)
- New Loss: Scale-Adaptive Box Loss (boosts small objects) + Class-Weighted QFL (boosts book, toy_car, mouse)
- Close-mosaic at Epoch 20 for 80 epochs of pure native resolution convergence
- Fine-tuning Cosine LR: 3.5e-4 -> 2e-6
"""
import argparse
import copy
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
import torch.nn.functional as F

# Ensure root import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qualifiers.train_dtnet_p2_ultra import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, ModelEMA, count_params_strict,
    FastDetDataset, decode_ltrb, bbox_eiou, decode_batch_np, evaluate_map
)

# Class weight vector prioritizes lower mAP50 classes (book: 96.73%, toy_car: 97.11%, mouse: 97.31%)
# 0: bottle (1.00), 1: cup (1.00), 2: phone (1.05), 3: book (1.35), 4: mouse (1.25), 5: box (0.95), 6: toy_car (1.30)
CLASS_PRIORITY_WEIGHTS = [1.00, 1.00, 1.05, 1.35, 1.25, 0.95, 1.30]


def det_loss_small_and_hard_adaptive(cls_logit, box_raw, cls_idx, box_t,
                                     lam=1.0, w_cls=1.0, w_box=6.0,
                                     qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.5,
                                     iou_gate=0.48, stride=STRIDE, max_dist=20.0,
                                     class_weights=None):
    """
    Hard-Class & Small-Object Adaptive Detection Loss:
    1. Small-Object Scale Weight: w_scale = 1.0 + 1.2 * (1.0 - sqrt(area_norm))
       Forces heavy gradients on small objects where localization error hurts mAP50 most.
    2. Hard-Class Priority Weight: scales both box loss and positive focal classification
       by class_weights[c] (book: 1.35x, toy_car: 1.30x, mouse: 1.25x).
    """
    cls_logit, box_raw = cls_logit.float(), box_raw.float()
    device = cls_logit.device
    pos = cls_idx >= 0
    n_pos = pos.sum().clamp(min=1).float()
    tgt = torch.zeros_like(cls_logit)

    if class_weights is None:
        class_weights = torch.tensor(CLASS_PRIORITY_WEIGHTS, device=device, dtype=torch.float32)

    if pos.any():
        pred_xyxy = decode_ltrb(box_raw, stride, max_dist)
        pb = pred_xyxy.permute(0, 2, 3, 1)[pos]
        gb = box_t.permute(0, 2, 3, 1)[pos]
        eiou, iou = bbox_eiou(pb, gb)

        # 1. Scale-Adaptive Weight (boosts small objects)
        w_obj = (gb[:, 2] - gb[:, 0]).clamp(min=1.0)
        h_obj = (gb[:, 3] - gb[:, 1]).clamp(min=1.0)
        area_norm = (w_obj * h_obj) / (ORIG_W * ORIG_H)
        w_scale = 1.0 + 1.2 * (1.0 - torch.sqrt(area_norm.clamp(min=1e-6, max=1.0)))

        # 2. Hard-Class Priority Weight
        pos_classes = cls_idx[pos]
        w_class = class_weights[pos_classes]
        combined_box_weight = w_scale * w_class

        loss_box = (combined_box_weight * (1.0 - eiou)).sum() / n_pos

        # Quality Focal Target
        iou_clamped = iou.detach().clamp(0, 1)
        valid_mask = iou_clamped >= iou_gate
        quality_score = torch.zeros_like(iou_clamped)
        quality_score[valid_mask] = ((iou_clamped[valid_mask] - iou_gate) / (1.0 - iou_gate)).pow(1.5)

        q = lam * quality_score + (1.0 - lam)
        b_i, j_i, i_i = pos.nonzero(as_tuple=True)
        tgt[b_i, pos_classes, j_i, i_i] = q
    else:
        loss_box = box_raw.sum() * 0.0

    p = cls_logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(cls_logit, tgt, reduction="none")

    is_pos_target = tgt > 0
    # Apply class priority weighting to positive classification loss
    pos_weight_grid = class_weights.view(1, NUM_CLASSES, 1, 1)
    loss_cls_pos = pos_weight_grid * ce * (tgt - p).abs().pow(qfl_beta_pos)
    loss_cls_neg = w_neg * ce * p.pow(qfl_beta_neg)

    loss_cls = torch.where(is_pos_target, loss_cls_pos, loss_cls_neg).sum() / n_pos
    total = w_cls * loss_cls + w_box * loss_box
    return total, loss_cls.detach(), loss_box.detach()


def main():
    parser = argparse.ArgumentParser(description="DTNet-P2-Ultra Hard-Class & Small-Object Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=100, help="Total fine-tuning epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Peak fine-tuning LR (default: 3.5e-4)")
    parser.add_argument("--w_box", type=float, default=6.0, help="Base box loss weight (default: 6.0)")
    parser.add_argument("--w_neg", type=float, default=1.5, help="Negative classification penalty multiplier (default: 1.5)")
    parser.add_argument("--iou_gate", type=float, default=0.48, help="IoU gate threshold (default: 0.48)")
    parser.add_argument("--resume", type=str, required=True, help="Path to best_p2_ultra.pt base checkpoint")
    parser.add_argument("--data_dir", default=None, help="Path to data directory")
    parser.add_argument("--output_dir", default="qualifiers/weights/dtnet_p2_ultra_finetune", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        candidates = [
            repo_root / "qualifiers/data",
            Path("/home/dutai/EAI-2026/qualifiers/data"),
            repo_root / "data",
        ]
        data_dir = next((c for c in candidates if c.exists()), candidates[0])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - DTNet-P2-ULTRA 100-EPOCH ADAPTIVE FINE-TUNING")
    print(f"   Base Checkpoint : {args.resume}")
    print(f"   Target Classes  : Boosted book (1.35x), toy_car (1.30x), mouse (1.25x)")
    print(f"   Small Objects   : Scale-Adaptive Box Weighting (1.0 + 1.2*(1 - sqrt(area)))")
    print(f"   Resolution      : {IMG_W} x {IMG_H} (Native 4:3)")
    print(f"   Epochs & Batch  : {args.epochs} Epochs | Batch Size {args.batch_size} | Close-Mosaic Ep 20")
    print(f"   Peak LR         : {args.lr} (Cosine Annealing -> 2e-6)")
    print(f"   Data Directory  : {data_dir}")
    print(f"   Output Directory: {output_dir}")
    print(f"   Device          : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # 1. Dataset Pool & Reproducible 9:1 Partition
    all_train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) + glob.glob(str(data_dir / "train/images/*.jpg")))
    all_val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) + glob.glob(str(data_dir / "val/images/*.jpg")))
    all_imgs = all_train_imgs + all_val_imgs
    total_imgs = len(all_imgs)
    print(f"[*] Total dataset images pooled: {total_imgs}")

    all_lbls = []
    for p in all_imgs:
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
        all_lbls.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))

    rng = np.random.default_rng(42)
    indices = rng.permutation(total_imgs)
    val_count = int(round(total_imgs * 0.10))
    val_idx, tr_idx = indices[:val_count], indices[val_count:]

    tr_imgs = [all_imgs[i] for i in tr_idx]
    tr_lbls = [all_lbls[i] for i in tr_idx]
    vl_imgs = [all_imgs[i] for i in val_idx]
    vl_lbls = [all_lbls[i] for i in val_idx]

    print(f"[✔] 9:1 Split created: {len(tr_imgs)} Train images | {len(vl_imgs)} Val images")

    # 2. DataLoaders (Gentle mosaic 0.2 for 20 epochs, then close mosaic)
    num_workers = max(1, min(args.num_workers, os.cpu_count() or 1))
    train_ds = FastDetDataset(tr_imgs, tr_lbls, IMG_W, IMG_H, STRIDE, train=True, mosaic_p=0.20, seed=42)
    val_ds = FastDetDataset(vl_imgs, vl_lbls, IMG_W, IMG_H, STRIDE, train=False, mosaic_p=0.0, seed=42)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # 3. Model & Checkpoint Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    spec = ck["spec"]

    model = DTNetMax(spec).to(device)
    model.load_state_dict(ck["ema"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    param_strict = count_params_strict(model)
    print(f"[✔] DTNet-P2-Max loaded: {param_strict:,} parameters (< {MAX_ALLOWED_PARAMS:,} LIMIT)")
    print(f"[*] Base Checkpoint mAP50: {ck.get('map50', 0)*100:.2f}% (Epoch {ck.get('epoch', '?')})")

    ema = ModelEMA(model, decay=0.9995)
    ema.ema.load_state_dict(ck["ema"])

    # 4. Optimizer & Fine-tuning LR Schedule
    lr_eff = args.lr
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if p.ndim > 1 else no_decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.01},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr_eff, betas=(0.9, 0.999)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_epochs = 2
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    close_mosaic_epoch = 20

    def get_lr(step):
        if step < warmup_steps:
            return lr_eff * (0.1 + 0.9 * step / warmup_steps)
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return 2e-6 + (lr_eff - 2e-6) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    # 5. Training Loop
    best_map50 = ck.get("map50", 0.0)
    best_epoch = 0
    best_pt_path = output_dir / "best_p2_ultra_finetune.pt"
    class_weights_t = torch.tensor(CLASS_PRIORITY_WEIGHTS, device=device, dtype=torch.float32)

    print(f"\n[*] Starting Fine-Tuning: 100 Epochs on {device} (Close-mosaic at Ep {close_mosaic_epoch})...\n")
    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        if epoch >= close_mosaic_epoch:
            train_ds.set_mosaic(0.0)

        w_box_curr = args.w_box + (1.0 if epoch >= close_mosaic_epoch else 0.0)
        lam = 1.0  # Quality gate fully active for fine-tuning

        loss_cls_accum, loss_box_accum, total_loss_accum = 0.0, 0.0, 0.0
        for step, (imgs, cls_idx, box_t, _) in enumerate(train_loader):
            current_step = epoch * steps_per_epoch + step
            current_lr = get_lr(current_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
            if device.type == "cuda":
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            cls_idx = cls_idx.to(device, non_blocking=True)
            box_t = box_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                co, bo = model(imgs)
                loss, l_cls, l_box = det_loss_small_and_hard_adaptive(
                    co, bo, cls_idx, box_t,
                    lam=lam, w_cls=1.0, w_box=w_box_curr,
                    qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=args.w_neg,
                    iou_gate=args.iou_gate,
                    class_weights=class_weights_t
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            loss_cls_accum += l_cls.item()
            loss_box_accum += l_box.item()
            total_loss_accum += loss.item()

        tr_cls = loss_cls_accum / steps_per_epoch
        tr_box = loss_box_accum / steps_per_epoch
        tr_total = total_loss_accum / steps_per_epoch

        # Validation evaluation every epoch
        ema.ema.eval()
        dets_val = []
        with torch.no_grad():
            for imgs, _, _, _ in val_loader:
                x = imgs.to(device, non_blocking=True).float().div_(255.0)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    co, bo = ema.ema(x)
                outs = decode_batch_np(co.float().cpu().numpy(), bo.float().cpu().numpy(), score_thr=0.001, nms_iou=0.60)
                dets_val.extend(outs)

        val_map50, per_class = evaluate_map(dets_val, vl_lbls)
        ep_time = time.time() - t0

        is_best = val_map50 > best_map50
        status_str = f"Epoch {epoch+1:03d}/{args.epochs:03d} - {ep_time:.1f}s - loss: {tr_total:.4f} (cls: {tr_cls:.4f}, box: {tr_box:.4f}) - val_mAP50: {val_map50*100:.2f}% - lr: {current_lr:.4e}"
        print(status_str)

        if is_best:
            best_map50 = val_map50
            best_epoch = epoch + 1
            print(f"[★ NEW RECORD] Val mAP50: {val_map50*100:.2f}% (Epoch {epoch+1})")
            per_c_str = " | ".join(f"{CLASS_NAMES[c]}: {per_class[c]*100:.1f}%" for c in range(NUM_CLASSES))
            print(f"    Per-class: {per_c_str}")
            torch.save({
                "epoch": epoch + 1,
                "ema": ema.ema.state_dict(),
                "spec": spec,
                "map50": val_map50,
                "per_class_ap": per_class,
            }, best_pt_path)

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 80)
    print(f"[✔] Fine-Tuning Complete! Best val_mAP50: {best_map50*100:.2f}% (Epoch {best_epoch}) in {total_time:.1f} minutes")
    print(f"[✔] Best weights saved to: {best_pt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
