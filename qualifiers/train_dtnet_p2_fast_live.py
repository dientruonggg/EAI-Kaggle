#!/usr/bin/env python3
"""
DTNet-P2 Fast Fine-Tuning Pipeline (Live Per-Class Logging & Rapid Convergence)
==============================================================================
- Warmstart: Loads best_p2_ultra.pt (98.18% base)
- Epochs: 30 epochs (completes in ~5 minutes on Tesla V100)
- Validation: Evaluated EVERY SINGLE EPOCH with full per-class mAP50 breakdown
- Anti-Nerf: Soft border penalty (0.25) to preserve background learning
- Small-object boost: EIoU loss + dynamic scale weight for small objects
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qualifiers.train_dtnet_p2_ultra import (
    ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES, CLASS_NAMES,
    MAX_ALLOWED_PARAMS, DTNetMax, ModelEMA, count_params_strict,
    decode_ltrb, bbox_eiou, decode_batch_np, evaluate_map, _affine_boxes, _hsv_jitter
)

CLASS_PRIORITY_WEIGHTS = [1.00, 1.00, 1.05, 1.30, 1.25, 0.95, 1.25]


def assign_targets_soft_border(boxes, gw, gh, stride, min_r=1.5, max_r=4.0, rad_scale=0.30):
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

    bg_weight = np.ones((gh, gw), dtype=np.float32)
    for c, x1, y1, x2, y2 in boxes:
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue
        i0 = max(0, int(math.ceil(x1 / stride - 0.5)))
        i1 = min(gw - 1, int(math.floor(x2 / stride - 0.5)))
        j0 = max(0, int(math.ceil(y1 / stride - 0.5)))
        j1 = min(gh - 1, int(math.floor(y2 / stride - 0.5)))
        if i1 >= i0 and j1 >= j0:
            bg_weight[j0:j1 + 1, i0:i1 + 1] = 0.25

    bg_weight[cls_idx >= 0] = 0.0
    return cls_idx, box_t, bg_weight


class FastLiveDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, img_w=IMG_W, img_h=IMG_H, stride=STRIDE, train=True,
                 mosaic_p=0.20, scale=(0.75, 1.25), translate=0.08, flip_p=0.5,
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

    def __getitem__(self, k):
        if self.train:
            rng = np.random.default_rng((self.seed * 1000003 + self.epoch.value * 100003 + k) & 0xFFFFFFFF)
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

        cls_idx, box_t, bg_weight = assign_targets_soft_border(b, self.gw, self.gh, self.stride)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img, torch.from_numpy(cls_idx), torch.from_numpy(box_t), torch.from_numpy(bg_weight), k


def det_loss_soft_finetune(cls_logit, box_raw, cls_idx, box_t, bg_weight,
                           lam=1.0, w_cls=1.0, w_box=5.0,
                           qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.2,
                           iou_gate=0.48, stride=STRIDE, max_dist=20.0,
                           class_weights=None):
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

        w_obj = (gb[:, 2] - gb[:, 0]).clamp(min=1.0)
        h_obj = (gb[:, 3] - gb[:, 1]).clamp(min=1.0)
        s_obj = (w_obj * h_obj).sqrt()
        w_size = (24.0 / s_obj).clamp(1.0, 2.5)

        pos_classes = cls_idx[pos]
        w_class = class_weights[pos_classes]
        combined_box_weight = w_size * w_class

        loss_box = (combined_box_weight * (1.0 - eiou)).sum() / n_pos

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

    bg_w = bg_weight.unsqueeze(1).expand_as(cls_logit)
    loss_cls_neg = bg_w * w_neg * ce * p.pow(qfl_beta_neg)

    loss_cls = torch.where(is_pos_target, loss_cls_pos, loss_cls_neg).sum() / n_pos
    total = w_cls * loss_cls + w_box * loss_box
    return total, loss_cls.detach(), loss_box.detach()


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


def main():
    parser = argparse.ArgumentParser(description="DTNet-P2 Fast Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=6.0e-5, help="Peak fine-tuning LR (default: 6.0e-5)")
    parser.add_argument("--resume", type=str, default="qualifiers/weights/dtnet_p2_ultra/best_p2_ultra.pt")
    parser.add_argument("--output_dir", type=str, default="qualifiers/weights/dtnet_p2_fast_live")
    parser.add_argument("--data_dir", type=str, default="qualifiers/data")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   DTNet-P2 FAST LIVE FINE-TUNING (RAPID CONVERGENCE)")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Base Checkpoint: {args.resume}")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch_size} | Peak LR: {args.lr}")
    print(f"   Output Directory: {output_dir}")
    print("=" * 80)

    tr_imgs, tr_lbls, vl_imgs, vl_lbls = load_dataset(args.data_dir)
    print(f"[*] Train images: {len(tr_imgs)} | Val images: {len(vl_imgs)}")

    train_ds = FastLiveDataset(tr_imgs, tr_lbls, train=True)
    val_ds = FastLiveDataset(vl_imgs, vl_lbls, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True
    )

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    spec = ck["spec"]
    model = DTNetMax(spec).to(device)

    base_state = ck.get("ema", ck.get("model_state_dict", ck.get("model", ck)))
    clean_state = {k.replace("module.", ""): v for k, v in base_state.items()}
    model.load_state_dict(clean_state, strict=True)

    n_params = count_params_strict(model)
    print(f"[✔] DTNetMax loaded: {n_params:,} parameters (< {MAX_ALLOWED_PARAMS:,})")

    model_ema = ModelEMA(model, decay=0.9995)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Initial evaluation
    print("\n[*] Evaluating baseline checkpoint before training...")
    model.eval()
    dets_init = []
    with torch.no_grad():
        for imgs, _, _, _, _ in val_loader:
            imgs = imgs.to(device).float().div_(255.0)
            c_l, b_r = model(imgs)
            d = decode_batch_np(c_l.cpu().numpy(), b_r.cpu().numpy(), score_thr=0.001, nms_iou=0.58)
            dets_init.extend(d)
    best_map50, init_aps = evaluate_map(dets_init, vl_lbls)
    best_epoch = 0
    print(f"[★ Baseline Epoch 0] val_mAP50: {best_map50*100:.2f}%")
    per_c_init = " | ".join(f"{CLASS_NAMES[c]}: {init_aps[c]*100:.2f}%" for c in range(NUM_CLASSES))
    print(f"    Per-class: {per_c_init}\n")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    def get_lr(step):
        prog = step / total_steps
        return 1e-6 + (args.lr - 1e-6) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)

        sum_loss, sum_cls, sum_box, steps = 0.0, 0.0, 0.0, 0

        for step, (imgs, cls_idx, box_t, bg_weight, _) in enumerate(train_loader):
            global_step = (epoch - 1) * steps_per_epoch + step
            cur_lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
            cls_idx = cls_idx.to(device, non_blocking=True)
            box_t = box_t.to(device, non_blocking=True)
            bg_weight = bg_weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                c_logit, b_raw = model(imgs)
                loss, l_cls, l_box = det_loss_soft_finetune(
                    c_logit, b_raw, cls_idx, box_t, bg_weight,
                    lam=1.0, w_cls=1.0, w_box=5.0,
                    qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.2
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            model_ema.update(model)

            sum_loss += float(loss.item())
            sum_cls += float(l_cls.item())
            sum_box += float(l_box.item())
            steps += 1

        avg_loss = sum_loss / max(1, steps)
        avg_cls = sum_cls / max(1, steps)
        avg_box = sum_box / max(1, steps)
        dur = time.time() - t0

        # LIVE VALIDATION EVERY SINGLE EPOCH WITH FULL PER-CLASS BREAKDOWN
        model_ema.ema.eval()
        val_dets = []
        with torch.no_grad():
            for imgs, _, _, _, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    c_l, b_r = model_ema.ema(imgs)
                d = decode_batch_np(c_l.float().cpu().numpy(), b_r.float().cpu().numpy(), score_thr=0.001, nms_iou=0.58)
                val_dets.extend(d)

        val_map50, per_class = evaluate_map(val_dets, vl_lbls)
        is_best = val_map50 > best_map50
        star = " ★ NEW BEST RECORD!" if is_best else ""

        print(f"Epoch {epoch:03d}/{args.epochs:03d} - {dur:.1f}s - loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f}) - val_mAP50: {val_map50*100:.2f}% (Best: {best_map50*100:.2f}% @ Ep {best_epoch}){star} - lr: {cur_lr:.4e}")
        per_c_str = " | ".join(f"{CLASS_NAMES[c]}: {per_class[c]*100:.2f}%" for c in range(NUM_CLASSES))
        print(f"    Per-class: {per_c_str}")

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
            torch.save(save_dict, output_dir / "best_p2_live.pt")

    print("\n" + "=" * 80)
    print(f" [★] Fast Live Fine-Tuning Complete! Peak val_mAP50: {best_map50*100:.2f}% @ Epoch {best_epoch}")
    print(f" [✔] Saved to: {output_dir / 'best_p2_live.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
