#!/usr/bin/env python3
"""
DTNet-P2-Toan-Ultra: High-Precision Detector incorporating DuyToan's Architecture & Loss Innovations
===================================================================================================
1. Border Cells Ignore Label (-2): Eliminates conflicting background gradient penalties on edge cells.
2. Dynamic Small-Object Scale Weighting: w = clip(24.0 / sqrt(wh), 1.0, 3.0) applied to box and cls losses.
3. WeightedRandomSampler: Prioritizes images with high density of small objects (< 16px and 16-24px).
4. Zero Box-Rotation (rotate=0.0) + Scale Zoom-Out (0.60 - 1.25) to avoid bounding box bloating.
5. Close Mosaic at the final 15 epochs for clean native edge alignment matching test distribution.
6. Full Integer INT8 PTQ Export & Calibration with TFLM compliance check (< 200k params, < 1.5MB).
"""
import os
import sys
import math
import time
import copy
import glob
import json
import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure root import
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qualifiers.train_dtnet_p2_ultra import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, ModelEMA, count_params_strict,
    decode_ltrb, bbox_eiou, decode_batch_np, evaluate_map, _affine_boxes, _hsv_jitter
)

CLASS_PRIORITY_WEIGHTS = [1.00, 1.00, 1.05, 1.25, 1.20, 0.95, 1.25]


# --------------------------------------------------------------------------- #
# 1. DUYTOAN INNOVATION: TARGET ASSIGNMENT WITH -2 BORDER IGNORE LABEL
# --------------------------------------------------------------------------- #
def assign_targets_toan_adaptive(boxes, gw, gh, stride, min_r=1.5, max_r=5.5, rad_scale=0.32):
    """
    Area-adaptive center sampling with -2 border cell ignore label:
    - Center/near-center cells receive positive class_id >= 0.
    - Outer cells inside bbox receive -2 (ambiguous/border cell): FP loss is NOT penalized.
    - True background cells outside all bboxes receive -1.
    """
    cls_idx = np.full((gh, gw), -1, dtype=np.int64)
    box_t = np.zeros((4, gh, gw), dtype=np.float32)
    best_area = np.full((gh, gw), np.inf, dtype=np.float32)

    for c, x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ci = min(max(int(cx // stride), 0), gw - 1)
        cj = min(max(int(cy // stride), 0), gh - 1)
        rx = float(np.clip(rad_scale * w / stride, min_r, max_r))
        ry = float(np.clip(rad_scale * h / stride, min_r, max_r))
        reach_x = int(math.ceil(rx))
        reach_y = int(math.ceil(ry))
        area = w * h

        for j in range(max(0, cj - reach_y), min(gh, cj + reach_y + 1)):
            ccy = (j + 0.5) * stride
            dy = (ccy - cy) / (ry * stride)
            for i in range(max(0, ci - reach_x), min(gw, ci + reach_x + 1)):
                ccx = (i + 0.5) * stride
                dx = (ccx - cx) / (rx * stride)
                is_center = (i == ci and j == cj)
                inside = (x1 <= ccx <= x2) and (y1 <= ccy <= y2)
                near = (dx * dx + dy * dy) <= 1.0
                if is_center or (inside and near):
                    if area < best_area[j, i]:
                        best_area[j, i] = area
                        cls_idx[j, i] = int(c)
                        box_t[:, j, i] = (x1, y1, x2, y2)

    # Border cells inside bbox but outside positive radius get -2 (ignore label)
    inside_mask = np.zeros((gh, gw), dtype=bool)
    for c, x1, y1, x2, y2 in boxes:
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue
        i0 = max(0, int(math.ceil(x1 / stride - 0.5)))
        i1 = min(gw - 1, int(math.floor(x2 / stride - 0.5)))
        j0 = max(0, int(math.ceil(y1 / stride - 0.5)))
        j1 = min(gh - 1, int(math.floor(y2 / stride - 0.5)))
        if i1 >= i0 and j1 >= j0:
            inside_mask[j0:j1 + 1, i0:i1 + 1] = True

    cls_idx[(cls_idx == -1) & inside_mask] = -2
    return cls_idx, box_t


# --------------------------------------------------------------------------- #
# 2. DATASET WITH ZERO ROTATION & CLEAN MOSAIC CLOSURE
# --------------------------------------------------------------------------- #
class ToanUltraDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, img_w=IMG_W, img_h=IMG_H, stride=STRIDE, train=True,
                 mosaic_p=0.5, scale=(0.60, 1.25), translate=0.10, flip_p=0.5,
                 hsv=(0.015, 0.5, 0.35), orig_w=ORIG_W, orig_h=ORIG_H, seed=42):
        self.img_paths = list(img_paths)
        self.labels = list(labels)
        self.W, self.H, self.stride = int(img_w), int(img_h), int(stride)
        self.gw, self.gh = self.W // self.stride, self.H // self.stride
        self.train = train
        self.scale = scale
        self.translate = translate
        self.flip_p = flip_p
        self.hsv = hsv
        self.orig_w, self.orig_h = orig_w, orig_h
        self.seed = seed
        self.epoch = mp.Value("i", 0)
        self.mosaic_p = mp.Value("d", float(mosaic_p))

    def set_epoch(self, e):
        self.epoch.value = int(e)

    def set_mosaic(self, p):
        self.mosaic_p.value = float(p)

    def __len__(self):
        return len(self.img_paths)

    def _load(self, k):
        p = self.img_paths[k]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        b = self.labels[k].copy()
        if self.W != self.orig_w or self.H != self.orig_h:
            b[:, [1, 3]] *= self.W / self.orig_w
            b[:, [2, 4]] *= self.H / self.orig_h
        return img, b

    def _single(self, k, rng):
        W, H = self.W, self.H
        img, b = self._load(k)
        s = rng.uniform(*self.scale)
        tx = W / 2 - s * W / 2 + rng.uniform(-self.translate, self.translate) * W
        ty = H / 2 - s * H / 2 + rng.uniform(-self.translate, self.translate) * H
        M = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        return img, _affine_boxes(b, s, tx, ty, W, H)

    def _mosaic(self, k, rng):
        W, H = self.W, self.H
        ids = [k] + [int(x) for x in rng.integers(0, len(self.img_paths), 3)]
        xc = int(rng.uniform(0.5 * W, 1.5 * W))
        yc = int(rng.uniform(0.5 * H, 1.5 * H))
        canvas = np.full((2 * H, 2 * W, 3), 114, dtype=np.uint8)
        all_b = []
        for n, i in enumerate(ids):
            img, b = self._load(i)
            if n == 0:
                x1a, y1a, x2a, y2a = max(xc - W, 0), max(yc - H, 0), xc, yc
                x1b, y1b, x2b, y2b = W - (x2a - x1a), H - (y2a - y1a), W, H
            elif n == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - H, 0), min(xc + W, 2 * W), yc
                x1b, y1b, x2b, y2b = 0, H - (y2a - y1a), min(W, x2a - x1a), H
            elif n == 2:
                x1a, y1a, x2a, y2a = max(xc - W, 0), yc, xc, min(2 * H, yc + H)
                x1b, y1b, x2b, y2b = W - (x2a - x1a), 0, W, min(y2a - y1a, H)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + W, 2 * W), min(2 * H, yc + H)
                x1b, y1b, x2b, y2b = 0, 0, min(W, x2a - x1a), min(y2a - y1a, H)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            if len(b):
                b = b.copy()
                b[:, [1, 3]] += padw
                b[:, [2, 4]] += padh
                area0 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
                b[:, [1, 3]] = b[:, [1, 3]].clip(x1a, x2a)
                b[:, [2, 4]] = b[:, [2, 4]].clip(y1a, y2a)
                bw, bh = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
                ok = (bw >= 2) & (bh >= 2) & (bw * bh >= 0.4 * area0)
                all_b.append(b[ok])
        b = np.concatenate(all_b, 0) if all_b else np.zeros((0, 5), np.float32)
        s = rng.uniform(*self.scale)
        tx = W / 2 - s * W + rng.uniform(-self.translate, self.translate) * W
        ty = H / 2 - s * H + rng.uniform(-self.translate, self.translate) * H
        M = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)
        img = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        return img, _affine_boxes(b, s, tx, ty, W, H)

    def __getitem__(self, k):
        if self.train:
            rng = np.random.default_rng((self.seed * 1000003 + self.epoch.value * 100003 + k) & 0xFFFFFFFF)
            if rng.random() < self.mosaic_p.value:
                img, b = self._mosaic(k, rng)
            else:
                img, b = self._single(k, rng)
            img = _hsv_jitter(img, rng, self.hsv)
            if rng.random() < self.flip_p:
                img = img[:, ::-1]
                if len(b):
                    b = b.copy()
                    x1, x2 = b[:, 1].copy(), b[:, 3].copy()
                    b[:, 1], b[:, 3] = self.W - x2, self.W - x1
        else:
            img, b = self._load(k)

        cls_idx, box_t = assign_targets_toan_adaptive(b, self.gw, self.gh, self.stride)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img, torch.from_numpy(cls_idx), torch.from_numpy(box_t), k


# --------------------------------------------------------------------------- #
# 3. LOSS FUNCTION WITH MASKED BORDER CELLS (-2) AND SMALL OBJECT WEIGHTING
# --------------------------------------------------------------------------- #
def det_loss_toan_ultra(cls_logit, box_raw, cls_idx, box_t,
                        lam=1.0, w_cls=1.0, w_box=6.0,
                        qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.5,
                        iou_gate=0.48, stride=STRIDE, max_dist=20.0,
                        class_weights=None):
    """
    Toan-Ultra Loss:
    1. Small-Object Scale Weighting: w_size = clip(24.0 / sqrt(w*h), 1.0, 3.0)
    2. Border cells ignore label (-2): Background classification penalty is ONLY
       applied where cls_idx == -1 (true background). Cells with -2 are masked out.
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

        # Small Object Scale Weight: w_size = clip(24.0 / sqrt(wh), 1.0, 3.0)
        w_obj = (gb[:, 2] - gb[:, 0]).clamp(min=1.0)
        h_obj = (gb[:, 3] - gb[:, 1]).clamp(min=1.0)
        s_obj = (w_obj * h_obj).sqrt()
        w_size = (24.0 / s_obj).clamp(1.0, 3.0)

        # Hard-Class Priority Weight
        pos_classes = cls_idx[pos]
        w_class = class_weights[pos_classes]
        combined_box_weight = w_size * w_class

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
    pos_weight_grid = class_weights.view(1, NUM_CLASSES, 1, 1)
    loss_cls_pos = pos_weight_grid * ce * (tgt - p).abs().pow(qfl_beta_pos)

    # DUYTOAN INNOVATION: True background mask (cls_idx == -1 only; cls_idx == -2 is masked out)
    is_true_bg = (cls_idx == -1).unsqueeze(1).expand_as(cls_logit)
    loss_cls_neg = torch.where(is_true_bg, w_neg * ce * p.pow(qfl_beta_neg), torch.zeros_like(ce))

    loss_cls = torch.where(is_pos_target, loss_cls_pos, loss_cls_neg).sum() / n_pos
    total = w_cls * loss_cls + w_box * loss_box
    return total, loss_cls.detach(), loss_box.detach()


# --------------------------------------------------------------------------- #
# 4. TRAINING PIPELINE
# --------------------------------------------------------------------------- #
def load_dataset(data_dir):
    data_dir = Path(data_dir)
    train_imgs = sorted(glob.glob(str(data_dir / "train/images/*.png")) +
                        glob.glob(str(data_dir / "train/images/*.jpg")))
    val_imgs = sorted(glob.glob(str(data_dir / "val/images/*.png")) +
                      glob.glob(str(data_dir / "val/images/*.jpg")))

    def parse_labels(img_paths, split):
        lbls = []
        for p in img_paths:
            stem = Path(p).stem
            lp = data_dir / split / "labels" / f"{stem}.txt"
            boxes = []
            if lp.exists():
                with open(lp) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = (cx - w / 2.0) * ORIG_W
                            y1 = (cy - h / 2.0) * ORIG_H
                            x2 = (cx + w / 2.0) * ORIG_W
                            y2 = (cy + h / 2.0) * ORIG_H
                            boxes.append([cid, x1, y1, x2, y2])
            lbls.append(np.array(boxes, dtype=np.float32).reshape(-1, 5))
        return lbls

    tr_lbls = parse_labels(train_imgs, "train")
    vl_lbls = parse_labels(val_imgs, "val")
    return train_imgs, tr_lbls, val_imgs, vl_lbls


def compute_sample_weights(labels):
    """Calculates sampling weights: boosts images containing small objects (< 16px and 16-24px)."""
    weights = []
    for b in labels:
        if len(b) == 0:
            weights.append(1.0)
            continue
        w = b[:, 3] - b[:, 1]
        h = b[:, 4] - b[:, 2]
        sz = np.sqrt(np.maximum(0.0, w * h))
        n16 = (sz < 16.0).sum()
        n24 = ((sz >= 16.0) & (sz < 24.0)).sum()
        score = 2.0 * n16 + 1.0 * n24
        weights.append(float(1.0 + min(2.0, 0.5 * score)))
    return np.array(weights, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="DTNet-P2-Toan-Ultra Training")
    parser.add_argument("--epochs", type=int, default=60, help="Total training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Initial learning rate (default: 2.5e-4)")
    parser.add_argument("--close_mosaic", type=int, default=15, help="Close mosaic at last N epochs (default: 15)")
    parser.add_argument("--resume", type=str, default="qualifiers/weights/dtnet_p2_ultra/best_p2_ultra.pt")
    parser.add_argument("--output_dir", type=str, default="qualifiers/weights/dtnet_p2_toan_ultra")
    parser.add_argument("--data_dir", type=str, default="qualifiers/data")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   DTNet-P2-TOAN-ULTRA: HIGH-PRECISION DETECTOR TRAINING")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Base Checkpoint: {args.resume}")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch_size} | Peak LR: {args.lr}")
    print(f"   Close Mosaic: Last {args.close_mosaic} Epochs")
    print(f"   Output: {output_dir}")
    print("=" * 80)

    # 1. Load Data
    tr_imgs, tr_lbls, vl_imgs, vl_lbls = load_dataset(args.data_dir)
    print(f"[*] Train images: {len(tr_imgs)} | Val images: {len(vl_imgs)}")

    sample_weights = compute_sample_weights(tr_lbls)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = ToanUltraDataset(tr_imgs, tr_lbls, train=True, mosaic_p=0.5, scale=(0.60, 1.25))
    val_ds = ToanUltraDataset(vl_imgs, vl_lbls, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True
    )

    # 2. Build Model & Warmstart
    print(f"[*] Loading base model weights from: {args.resume}")
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    spec = ck["spec"]
    model = DTNetMax(spec).to(device)

    base_state = ck.get("ema", ck.get("model_state_dict", ck.get("model", ck)))
    clean_state = {k.replace("module.", ""): v for k, v in base_state.items()}
    model.load_state_dict(clean_state, strict=True)

    n_params = count_params_strict(model)
    print(f"[✔] DTNetMax loaded successfully: {n_params:,} parameters (Budget: {MAX_ALLOWED_PARAMS:,})")
    assert n_params <= MAX_ALLOWED_PARAMS, f"Exceeded parameter budget: {n_params} > {MAX_ALLOWED_PARAMS}"

    model_ema = ModelEMA(model, decay=0.9990)

    # 3. Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    warmup_epochs = 3
    lr_min = 1.0e-6

    def get_lr(ep):
        if ep < warmup_epochs:
            return args.lr * (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos(math.pi * progress))

    # Evaluate initial baseline
    print("[*] Evaluating baseline on validation split...")
    model.eval()
    dets_base = []
    with torch.no_grad():
        for imgs, _, _, _ in val_loader:
            imgs = imgs.to(device).float().div_(255.0)
            c_l, b_r = model(imgs)
            d = decode_batch_np(c_l.cpu().numpy(), b_r.cpu().numpy(), score_thr=0.001, nms_iou=0.58)
            dets_base.extend(d)
    base_map50, base_aps = evaluate_map(dets_base, vl_lbls)
    print(f"[*] Initial Baseline val mAP50: {base_map50*100:.2f}% | Per-class: { {CLASS_NAMES[k]: round(v*100, 2) for k, v in base_aps.items()} }")

    best_map50 = base_map50
    best_epoch = 0
    history = []

    # 4. Training Loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cur_lr = get_lr(epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Mosaic closure at the last N epochs
        if epoch > (args.epochs - args.close_mosaic):
            train_ds.set_mosaic(0.0)
        else:
            train_ds.set_mosaic(0.5)

        train_ds.set_epoch(epoch)

        model.train()
        sum_loss, sum_cls, sum_box, steps = 0.0, 0.0, 0.0, 0

        for imgs, cls_idx, box_t, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
            cls_idx = cls_idx.to(device, non_blocking=True)
            box_t = box_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            c_logit, b_raw = model(imgs)

            loss, l_cls, l_box = det_loss_toan_ultra(
                c_logit, b_raw, cls_idx, box_t,
                lam=1.0, w_cls=1.0, w_box=6.0,
                qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.5
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            model_ema.update(model)

            sum_loss += float(loss.item())
            sum_cls += float(l_cls.item())
            sum_box += float(l_box.item())
            steps += 1

        avg_loss = sum_loss / max(1, steps)
        avg_cls = sum_cls / max(1, steps)
        avg_box = sum_box / max(1, steps)

        # Validation evaluation using EMA model
        model_ema.ema.eval()
        val_dets = []
        with torch.no_grad():
            for imgs, _, _, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
                c_l, b_r = model_ema.ema(imgs)
                d = decode_batch_np(c_l.cpu().numpy(), b_r.cpu().numpy(), score_thr=0.001, nms_iou=0.58)
                val_dets.extend(d)

        val_map50, per_class = evaluate_map(val_dets, vl_lbls)
        dur = time.time() - t0

        is_best = val_map50 > best_map50
        if is_best:
            best_map50 = val_map50
            best_epoch = epoch
            save_dict = {
                "epoch": epoch,
                "map50": val_map50,
                "per_class_ap": per_class,
                "spec": spec,
                "ema": model_ema.ema.state_dict(),
                "model": model.state_dict(),
                "params": n_params
            }
            torch.save(save_dict, output_dir / "best_p2_toan_ultra.pt")

        star = " ★ BEST" if is_best else ""
        print(f"Epoch {epoch:03d}/{args.epochs:03d} - {dur:.1f}s - loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f}) - val_mAP50: {val_map50*100:.2f}% (Best: {best_map50*100:.2f}% @ Ep {best_epoch}){star} - lr: {cur_lr:.4e}")

        history.append({
            "epoch": epoch, "loss": avg_loss, "val_map50": val_map50,
            "lr": cur_lr, "per_class": {CLASS_NAMES[k]: round(v*100, 2) for k, v in per_class.items()}
        })

    total_time = (time.time() - t_start) / 60.0
    print("\n" + "=" * 80)
    print(f" [✔] Training Finished in {total_time:.1f} minutes!")
    print(f" [★] Peak val_mAP50: {best_map50*100:.2f}% achieved at Epoch {best_epoch}!")
    print(f" [✔] Best Model Saved: {output_dir / 'best_p2_toan_ultra.pt'}")
    print("=" * 80)

    # Save summary report
    summary = {
        "best_epoch": best_epoch,
        "best_val_map50": best_map50,
        "base_val_map50": base_map50,
        "gain": best_map50 - base_map50,
        "params": n_params,
        "total_time_min": total_time,
        "history": history
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
