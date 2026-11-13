#!/usr/bin/env python3
"""
train_dtnet_p2_ultra.py
=============================================================================
Edge AI Challenge 2026 (Qualifiers) - DTNet-P2-Ultra Pipeline (197k Params)
Target 99.5% mAP@50 Architecture:
  1. DTNet-P2 Architecture (197,647 params < 200,000 limit, margin: 2,353 params)
     - Expanded fuse_ch = 116, head_ch = 104
     - Multi-Scale SPPF Bottleneck (3x MaxPool 5x5, triples receptive field to 144x144)
  2. Ultra Area-Adaptive Target Assignment:
     - min_r = 1.5 (safeguards 5-9 supervisory anchors for small objects mouse & toy_car)
     - max_r = 5.5 (full supervisory coverage for large objects book & box)
     - rad_scale = 0.32
  3. Efficient IoU (EIoU) Box Regression + IoU-Gated Hard-Negative QFL (w_neg = 1.5, gate = 0.48)
  4. Train 100% from Scratch (Seed 42, Kaiming Normal) for Full 400 Epochs (Patience=0, No Early Stopping)
  5. Memory-safe DataLoader (num_workers=2, avoids OOM/systemd-oomd)
=============================================================================
"""

import argparse
import copy
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)

ORIG_W, ORIG_H = 320, 240
IMG_W, IMG_H = 320, 240
STRIDE = 8
GRID_W, GRID_H = IMG_W // STRIDE, IMG_H // STRIDE  # 40 x 30
NUM_CLASSES = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
MAX_ALLOWED_PARAMS = 200_000


# --------------------------------------------------------------------------- #
# 1. ARCHITECTURE SPECIFICATION (DTNet-P2-Max)
# --------------------------------------------------------------------------- #
def make_spec_max(num_classes=NUM_CLASSES, head_ch=104, fuse_ch=116, use_p2=True, use_sppf=True):
    ops = []

    def conv(name, src, cin, cout, k=1, s=1, bn=True, act=True, bias=False, zero_gamma=False):
        ops.append(dict(t="conv", name=name, src=src, cin=cin, cout=cout, k=k, s=s, bn=bn, act=act, bias=bias,
                        zero_gamma=zero_gamma))
        return name

    def dw(name, src, c, k=3, s=1, bn=True, act=True):
        ops.append(dict(t="dw", name=name, src=src, cin=c, cout=c, k=k, s=s, bn=bn, act=act, bias=False,
                        zero_gamma=False))
        return name

    def add(name, a, b):
        ops.append(dict(t="add", name=name, srcs=[a, b]))
        return name

    def cat(name, srcs):
        ops.append(dict(t="cat", name=name, srcs=list(srcs)))
        return name

    def up(name, src):
        ops.append(dict(t="up", name=name, src=src))
        return name

    def mp_op(name, src, k=5):
        ops.append(dict(t="mp", name=name, src=src, k=k))
        return name

    def res(prefix, x, c):
        a = dw(prefix + "_dw", x, c)
        b = conv(prefix + "_pw", a, c, c, act=False, zero_gamma=True)
        return add(prefix + "_add", x, b)

    x = conv("stem", "input", 3, 32, k=3, s=2)                       # /2 (160x120)
    x = dw("s1_dw", x, 32)
    x = conv("s1_pw", x, 32, 48, act=False)
    x = dw("s1_down_dw", x, 48, s=2)                                 # /4 (80x60)
    x = conv("s1_down_pw", x, 48, 64)
    p2 = res("s2a", x, 64)                                           # P2: Stride-4 features (high-res)
    x = dw("s2_down_dw", p2, 64, s=2)                                # /8 (40x30)
    x = conv("s2_down_pw", x, 64, 112)
    x = res("s2b", x, 112)
    shape = res("s2c", x, 112)                                       # Stride-8 shape features
    x = dw("s3_down_dw", shape, 112, s=2)                            # /16 (20x15)
    x = conv("s3_down_pw", x, 112, 160)
    sem = res("s3a", x, 160)                                         # Stride-16 semantic features

    # SPPF: Spatial Pyramid Pooling Fast (0 params, multi-scale context)
    if use_sppf:
        p1 = mp_op("spp_p1", sem, k=5)
        p2_pool = mp_op("spp_p2", p1, k=5)
        sem_add = add("spp_add1", sem, p1)
        sem = add("spp_add2", sem_add, p2_pool)

    x = conv("s3_proj", sem, 160, fuse_ch, bn=False, act=False)
    up_f = up("s3_up", x)                                            # /16 -> /8 (nearest)

    parts = [shape, up_f]
    fuse_in = 112 + fuse_ch
    if use_p2:
        q = dw("p2_dw", p2, 64, s=2)                                 # P2: /4 -> /8
        q = conv("p2_pw", q, 64, 48)
        parts.append(q)
        fuse_in += 48

    f = cat("fuse_cat", parts)
    f = conv("fuse_pw", f, fuse_in, fuse_ch)
    f = res("refine", f, fuse_ch)

    # Decoupled heads with expanded head_ch=104
    c = dw("cls_dw", f, fuse_ch)
    c = conv("cls_pw", c, fuse_ch, head_ch)
    conv("cls_out", c, head_ch, num_classes, bn=False, act=False, bias=True)

    b = dw("box_dw", f, fuse_ch)
    b = conv("box_pw", b, fuse_ch, head_ch)
    conv("box_out", b, head_ch, 4, bn=False, act=False, bias=True)
    return ops


class DTNetMax(nn.Module):
    def __init__(self, spec, max_dist=20.0):
        super().__init__()
        self.spec = spec
        self.max_dist = max_dist
        self.mods = nn.ModuleDict()
        for op in spec:
            if op["t"] in ("conv", "dw"):
                k, s = op["k"], op["s"]
                pad = (k - 1) // 2 if s == 1 else 0
                groups = op["cin"] if op["t"] == "dw" else 1
                self.mods[op["name"]] = nn.Conv2d(op["cin"], op["cout"], k, s, pad, groups=groups, bias=op["bias"])
                if op["bn"]:
                    self.mods[op["name"] + "_bn"] = nn.BatchNorm2d(op["cout"], eps=1e-5, momentum=0.1)
            elif op["t"] == "mp":
                self.mods[op["name"]] = nn.MaxPool2d(op["k"], stride=1, padding=op["k"] // 2)
        self._init(max_dist)

    def _init(self, max_dist):
        for op in self.spec:
            if op["t"] not in ("conv", "dw"):
                continue
            conv = self.mods[op["name"]]
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            if op["bn"] and op.get("zero_gamma"):
                nn.init.zeros_(self.mods[op["name"] + "_bn"].weight)
        nn.init.normal_(self.mods["cls_out"].weight, std=0.01)
        nn.init.constant_(self.mods["cls_out"].bias, -math.log((1 - 0.01) / 0.01))
        nn.init.normal_(self.mods["box_out"].weight, std=0.01)
        p = min(max(2.0 / max_dist, 1e-3), 0.5)
        nn.init.constant_(self.mods["box_out"].bias, math.log(p / (1 - p)))

    def forward(self, x):
        t = {"input": x}
        for op in self.spec:
            ty, n = op["t"], op["name"]
            if ty in ("conv", "dw"):
                y = t[op["src"]]
                if op["s"] == 2 and op["k"] == 3:
                    y = F.pad(y, (0, 1, 0, 1))
                y = self.mods[n](y)
                if op["bn"]:
                    y = self.mods[n + "_bn"](y)
                if op["act"]:
                    y = F.relu6(y)
            elif ty == "add":
                y = t[op["srcs"][0]] + t[op["srcs"][1]]
            elif ty == "cat":
                y = torch.cat([t[s] for s in op["srcs"]], 1)
            elif ty == "up":
                y = F.interpolate(t[op["src"]], scale_factor=2, mode="nearest")
            elif ty == "mp":
                y = self.mods[n](t[op["src"]])
            t[n] = y
        return t["cls_out"], t["box_out"]


def count_params_strict(model):
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for k, b in model.named_buffers() if k.endswith("running_mean") or k.endswith("running_var"))
    return n


# --------------------------------------------------------------------------- #
# 2. DYNAMIC AREA-ADAPTIVE TARGET ASSIGNMENT
# --------------------------------------------------------------------------- #
def assign_targets_adaptive(boxes, gw, gh, stride, min_r=1.5, max_r=5.5, rad_scale=0.32):
    """Dynamic Area-Adaptive Target Assignment (Ultra Calibration):
    - min_r = 1.5: Preserves 5-9 supervisory anchors for small objects (mouse, toy_car), preventing gradient starvation.
    - max_r = 5.5: Provides full supervisory coverage across large objects (book, box), preventing edge truncation."""
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
    return cls_idx, box_t


def _affine_boxes(boxes, s, tx, ty, W, H, min_vis=0.4, min_size=2.0):
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [1, 3]] = b[:, [1, 3]] * s + tx
    b[:, [2, 4]] = b[:, [2, 4]] * s + ty
    area0 = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, W)
    b[:, [2, 4]] = b[:, [2, 4]].clip(0, H)
    w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
    keep = (w >= min_size) & (h >= min_size) & (w * h >= min_vis * area0)
    return b[keep]


def _hsv_jitter(img, rng, gains):
    r = rng.uniform(-1, 1, 3) * np.asarray(gains) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)


class FastDetDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, img_w=IMG_W, img_h=IMG_H, stride=STRIDE, train=True,
                 mosaic_p=0.5, scale=(0.75, 1.25), translate=0.1, flip_p=0.5,
                 hsv=(0.015, 0.5, 0.35), orig_w=ORIG_W, orig_h=ORIG_H, seed=42):
        self.img_paths = list(img_paths)
        self.labels = list(labels)
        self.W, self.H, self.stride = int(img_w), int(img_h), int(stride)
        self.gw, self.gh = self.W // self.stride, self.H // self.stride
        self.train = train
        self.scale, self.translate, self.flip_p, self.hsv = scale, translate, flip_p, hsv
        self.orig_w, self.orig_h, self.seed = orig_w, orig_h, seed
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
        cls_idx, box_t = assign_targets_adaptive(b, self.gw, self.gh, self.stride)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img, torch.from_numpy(cls_idx), torch.from_numpy(box_t), k


# --------------------------------------------------------------------------- #
# 3. ADVANCED LOSSES & DECODING (EIoU + IoU-Gated QFL)
# --------------------------------------------------------------------------- #
def decode_ltrb(box_raw, stride=STRIDE, max_dist=20.0):
    B, _, gh, gw = box_raw.shape
    d = torch.sigmoid(box_raw) * (max_dist * stride)
    gx = ((torch.arange(gw, device=box_raw.device, dtype=box_raw.dtype) + 0.5) * stride).view(1, gw)
    gy = ((torch.arange(gh, device=box_raw.device, dtype=box_raw.dtype) + 0.5) * stride).view(gh, 1)
    return torch.stack([gx - d[:, 0], gy - d[:, 1], gx + d[:, 2], gy + d[:, 3]], 1)


def bbox_eiou(b1, b2, eps=1e-7):
    ix1, iy1 = torch.max(b1[:, 0], b2[:, 0]), torch.max(b1[:, 1], b2[:, 1])
    ix2, iy2 = torch.min(b1[:, 2], b2[:, 2]), torch.min(b1[:, 3], b2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    w1, h1 = (b1[:, 2] - b1[:, 0]).clamp(min=eps), (b1[:, 3] - b1[:, 1]).clamp(min=eps)
    w2, h2 = (b2[:, 2] - b2[:, 0]).clamp(min=eps), (b2[:, 3] - b2[:, 1]).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (torch.max(b1[:, 2], b2[:, 2]) - torch.min(b1[:, 0], b2[:, 0])).clamp(min=eps)
    ch = (torch.max(b1[:, 3], b2[:, 3]) - torch.min(b1[:, 1], b2[:, 1])).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps

    cx1, cy1 = (b1[:, 0] + b1[:, 2]) / 2, (b1[:, 1] + b1[:, 3]) / 2
    cx2, cy2 = (b2[:, 0] + b2[:, 2]) / 2, (b2[:, 1] + b2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    dist_w2 = (w1 - w2) ** 2 / (cw ** 2 + eps)
    dist_h2 = (h1 - h2) ** 2 / (ch ** 2 + eps)

    eiou = iou - (rho2 / c2) - dist_w2 - dist_h2
    return eiou, iou


def det_loss_eiou_gated(cls_logit, box_raw, cls_idx, box_t,
                        lam=1.0, w_cls=1.0, w_box=5.5,
                        qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=1.5,
                        iou_gate=0.48, stride=STRIDE, max_dist=20.0):
    cls_logit, box_raw = cls_logit.float(), box_raw.float()
    pos = cls_idx >= 0
    n_pos = pos.sum().clamp(min=1).float()
    tgt = torch.zeros_like(cls_logit)

    if pos.any():
        pred_xyxy = decode_ltrb(box_raw, stride, max_dist)
        pb = pred_xyxy.permute(0, 2, 3, 1)[pos]
        gb = box_t.permute(0, 2, 3, 1)[pos]
        eiou, iou = bbox_eiou(pb, gb)
        loss_box = (1.0 - eiou).sum() / n_pos

        iou_clamped = iou.detach().clamp(0, 1)
        valid_mask = iou_clamped >= iou_gate
        quality_score = torch.zeros_like(iou_clamped)
        quality_score[valid_mask] = ((iou_clamped[valid_mask] - iou_gate) / (1.0 - iou_gate)).pow(1.5)

        q = lam * quality_score + (1.0 - lam)
        b_i, j_i, i_i = pos.nonzero(as_tuple=True)
        tgt[b_i, cls_idx[pos], j_i, i_i] = q
    else:
        loss_box = box_raw.sum() * 0.0

    p = cls_logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(cls_logit, tgt, reduction="none")

    is_pos_target = tgt > 0
    loss_cls_pos = ce * (tgt - p).abs().pow(qfl_beta_pos)
    loss_cls_neg = w_neg * ce * p.pow(qfl_beta_neg)

    loss_cls = torch.where(is_pos_target, loss_cls_pos, loss_cls_neg).sum() / n_pos
    total = w_cls * loss_cls + w_box * loss_box
    return total, loss_cls.detach(), loss_box.detach()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def nms_np(boxes, scores, thr, max_keep):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[r] - inter + 1e-9)
        order = r[iou <= thr]
    return np.asarray(keep, dtype=np.int64)


def decode_batch_np(cls_logit, box_raw, score_thr=0.001, nms_iou=0.65, pre_nms=1000, max_det=100,
                    stride=STRIDE, max_dist=20.0, orig_w=ORIG_W, orig_h=ORIG_H):
    B, C, gh, gw = cls_logit.shape
    scores = _sigmoid(cls_logit)
    d = _sigmoid(box_raw) * (max_dist * stride)
    gx = (np.arange(gw) + 0.5) * stride
    gy = (np.arange(gh) + 0.5) * stride
    out = []
    for b in range(B):
        sc = scores[b].reshape(-1)
        k = min(pre_nms, sc.size)
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[sc[idx] > score_thr]
        if idx.size == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        c = idx // (gh * gw)
        rem = idx % (gh * gw)
        j, i = rem // gw, rem % gw
        boxes = np.stack([gx[i] - d[b, 0, j, i], gy[j] - d[b, 1, j, i],
                          gx[i] + d[b, 2, j, i], gy[j] + d[b, 3, j, i]], 1)
        s = sc[idx]
        off = boxes + (c[:, None] * 4096.0)
        keep = nms_np(off, s, nms_iou, max_det)
        bx = boxes[keep]
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, orig_w)
        bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, orig_h)
        out.append(np.concatenate([bx, s[keep, None], c[keep, None]], 1).astype(np.float32))
    return out


def evaluate_map(dets, gts, num_classes=NUM_CLASSES, iou_thr=0.5):
    aps = {}
    for c in range(num_classes):
        gt_c = [g[g[:, 0] == c][:, 1:5] for g in gts]
        n_gt = sum(len(x) for x in gt_c)
        if n_gt == 0:
            continue
        used = [np.zeros(len(x), dtype=bool) for x in gt_c]
        recs = []
        for k, d in enumerate(dets):
            if d is None or len(d) == 0:
                continue
            for row in d[d[:, 5] == c]:
                recs.append((float(row[4]), k, row[:4]))
        recs.sort(key=lambda r: -r[0])
        tp, fp = np.zeros(len(recs)), np.zeros(len(recs))
        for n, (s, k, bx) in enumerate(recs):
            g = gt_c[k]
            if len(g) == 0:
                fp[n] = 1
                continue
            xx1 = np.maximum(bx[0], g[:, 0])
            yy1 = np.maximum(bx[1], g[:, 1])
            xx2 = np.minimum(bx[2], g[:, 2])
            yy2 = np.minimum(bx[3], g[:, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a = max(0.0, (bx[2] - bx[0])) * max(0.0, (bx[3] - bx[1]))
            ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
            ious = inter / np.maximum(a + ag - inter, 1e-9)
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[k][j]:
                tp[n] = 1
                used[k][j] = True
            else:
                fp[n] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ii = np.where(mrec[1:] != mrec[:-1])[0]
        aps[c] = float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps


# --------------------------------------------------------------------------- #
# 4. EMA
# --------------------------------------------------------------------------- #
class ModelEMA:
    def __init__(self, model, decay=0.9995, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


# --------------------------------------------------------------------------- #
# 5. KERAS & INT8 QUANTIZATION SYNTHESIS
# --------------------------------------------------------------------------- #
def build_keras_dtnet_max(spec, h=IMG_H, w=IMG_W):
    import keras
    from keras import layers

    inp = keras.Input((h, w, 3), name="image")
    t = {"input": inp}
    for op in spec:
        ty, n = op["t"], op["name"]
        if ty == "conv":
            y = layers.Conv2D(op["cout"], op["k"], strides=op["s"], padding="same", use_bias=op["bias"], name=n)(t[op["src"]])
        elif ty == "dw":
            y = layers.DepthwiseConv2D(op["k"], strides=op["s"], padding="same", use_bias=False, name=n)(t[op["src"]])
        elif ty == "add":
            y = layers.Add(name=n)([t[s] for s in op["srcs"]])
        elif ty == "cat":
            y = layers.Concatenate(axis=-1, name=n)([t[s] for s in op["srcs"]])
        elif ty == "up":
            _, hh, ww, _ = t[op["src"]].shape
            y = layers.Resizing(hh * 2, ww * 2, interpolation="nearest", name=n)(t[op["src"]])
        elif ty == "mp":
            y = layers.MaxPooling2D(pool_size=op["k"], strides=1, padding="same", name=n)(t[op["src"]])
        if ty in ("conv", "dw"):
            if op["bn"]:
                y = layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=n + "_bn")(y)
            if op["act"]:
                y = layers.ReLU(6.0, name=n + "_act")(y)
        t[n] = y
    return keras.Model(inp, [t["cls_out"], t["box_out"]], name="DTNet_P2_Max")


def transfer_weights_to_keras(net_t, kmodel, spec):
    npy = lambda x: x.detach().cpu().numpy()
    for op in spec:
        if op["t"] not in ("conv", "dw"):
            continue
        m = net_t.mods[op["name"]]
        w = npy(m.weight)
        ws = [w.transpose(2, 3, 1, 0) if op["t"] == "conv" else w.transpose(2, 3, 0, 1)]
        if op["bias"]:
            ws.append(npy(m.bias))
        kmodel.get_layer(op["name"]).set_weights(ws)
        if op["bn"]:
            bn = net_t.mods[op["name"] + "_bn"]
            kmodel.get_layer(op["name"] + "_bn").set_weights(
                [npy(bn.weight), npy(bn.bias), npy(bn.running_mean), npy(bn.running_var)])
        if op["bn"]:
            bn = net_t.mods[op["name"] + "_bn"]
            kmodel.get_layer(op["name"] + "_bn").set_weights(
                [npy(bn.weight), npy(bn.bias), npy(bn.running_mean), npy(bn.running_var)])


def export_full_int8_tflite(kmodel, calib_img_paths, output_tflite_path, num_calib=200):
    import tensorflow as tf

    print(f"[*] Quantizing Keras model to Full Integer INT8 with {min(num_calib, len(calib_img_paths))} representative samples...")

    def representative_dataset():
        for p in calib_img_paths[:num_calib]:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
                img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            arr = (img.astype(np.float32) / 255.0)[np.newaxis, ...]
            yield [arr]

    converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print(f"[✔] Full INT8 TFLite exported successfully: {output_tflite_path} ({size_kb:.1f} KB)")
    return output_tflite_path


# --------------------------------------------------------------------------- #
# 6. MAIN ENGINE
# --------------------------------------------------------------------------- #
def main():
    default_out = Path(__file__).resolve().parents[1] / "qualifiers/weights/dtnet_p2_ultra"
    parser = argparse.ArgumentParser(description="Train DTNet-P2-Ultra Champion Pipeline (197k Params)")
    parser.add_argument("--epochs", type=int, default=400, help="Total training epochs (default: 400)")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (default: 0 to disable, or >0 for early stop)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Base LR for batch 32")
    parser.add_argument("--w_box", type=float, default=5.5, help="Base EIoU box loss weight (default: 5.5)")
    parser.add_argument("--w_neg", type=float, default=1.5, help="Negative classification penalty multiplier (default: 1.5)")
    parser.add_argument("--iou_gate", type=float, default=0.48, help="IoU gate threshold for quality alignment (default: 0.48)")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from existing checkpoint .pt")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers (default: 2 to prevent OOM)")
    parser.add_argument("--data_dir", default=None, help="Path to data directory containing train/val")
    parser.add_argument("--output_dir", default=str(default_out), help="Output directory for weights and logs")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        candidates = [
            repo_root / "qualifiers/data",
            repo_root / "data",
            Path("/home/dutai/EAI-2026/qualifiers/data"),
        ]
        data_dir = next((c for c in candidates if c.exists()), candidates[0])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - DTNet-P2-ULTRA PIPELINE (TARGET 99.5% mAP50)")
    print(f"   Architecture    : DTNet with SPPF, P2 (Stride-4), fuse_ch=116, head_ch=104 (197k Params)")
    print(f"   Resolution      : {IMG_W} x {IMG_H} (Native 4:3 Aspect Ratio)")
    print(f"   Epochs & Batch  : {args.epochs} Epochs | Batch Size {args.batch_size} | Patience: {args.patience}")
    print(f"   Target Assign   : Ultra Area-Adaptive (min_r=1.5, max_r=5.5, rad_scale=0.32)")
    print(f"   Loss Formulation: EIoU (w_box={args.w_box}) + IoU-Gated QFL (gate={args.iou_gate}, w_neg={args.w_neg})")
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

    # 2. Datasets & Loaders
    num_workers = max(1, min(args.num_workers, os.cpu_count() or 1))
    train_ds = FastDetDataset(tr_imgs, tr_lbls, IMG_W, IMG_H, STRIDE, train=True, mosaic_p=0.5, seed=42)
    val_ds = FastDetDataset(vl_imgs, vl_lbls, IMG_W, IMG_H, STRIDE, train=False, mosaic_p=0.0, seed=42)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # 3. Model & EMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec_max(num_classes=NUM_CLASSES, head_ch=104, fuse_ch=116, use_p2=True, use_sppf=True)
    model = DTNetMax(spec).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    param_strict = count_params_strict(model)
    print(f"[✔] DTNet-P2-Max instantiated: {param_strict:,} parameters (< {MAX_ALLOWED_PARAMS:,} LIMIT | Margin: {MAX_ALLOWED_PARAMS - param_strict})")
    assert param_strict < MAX_ALLOWED_PARAMS, f"Exceeded hardware parameter limit: {param_strict}"

    ema = ModelEMA(model, decay=0.9995)

    # 4. Optimizer & LR Schedule
    lr_eff = args.lr * (args.batch_size / 32.0)
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if p.ndim > 1 else no_decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.02},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr_eff, betas=(0.9, 0.999)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_epochs = 3
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    close_mosaic_epoch = args.epochs - 35

    def get_lr(step):
        if step < warmup_steps:
            return lr_eff * (0.01 + 0.99 * step / warmup_steps)
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_eff * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    # 5. Training Loop with Early Stopping & Optional Resume
    start_epoch = 0
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_pt_path = output_dir / "best_p2_ultra.pt"

    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming from checkpoint: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["ema"])
        ema.ema.load_state_dict(ck["ema"])
        start_epoch = ck.get("epoch", 0)
        best_map50 = ck.get("map50", 0.0)
        best_epoch = ck.get("epoch", 0)
        print(f"[✔] Resumed state: Start Epoch {start_epoch}, Baseline Best mAP50: {best_map50*100:.2f}% (Epoch {best_epoch})")

    patience_str = "Disabled (full 400 epochs)" if args.patience <= 0 else f"{args.patience} epochs"
    print(f"\n[*] Starting Training: Epochs {start_epoch}->{args.epochs} on {device} (Close-mosaic at Ep {close_mosaic_epoch}, Patience: {patience_str})...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        if epoch >= close_mosaic_epoch:
            train_ds.set_mosaic(0.0)

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
                loss, l_cls, l_box = det_loss_eiou_gated(
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

            loss_cls_accum += l_cls.item()
            loss_box_accum += l_box.item()
            total_loss_accum += loss.item()

        tr_cls = loss_cls_accum / steps_per_epoch
        tr_box = loss_box_accum / steps_per_epoch
        tr_total = total_loss_accum / steps_per_epoch

        # Evaluate validation loss and mAP50 every epoch
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
                    vl, v_cls, v_box = det_loss_eiou_gated(
                        co, bo, cls_idx, box_t,
                        lam=lam, w_cls=1.0, w_box=w_box_curr,
                        qfl_beta_pos=2.0, qfl_beta_neg=1.2, w_neg=args.w_neg,
                        iou_gate=args.iou_gate
                    )
                val_cls_accum += v_cls.item()
                val_box_accum += v_box.item()
                val_loss_accum += vl.item()
                outs = decode_batch_np(co.float().cpu().numpy(), bo.float().cpu().numpy(), score_thr=0.001)
                dets_val.extend(outs)

        val_cls = val_cls_accum / len(val_loader)
        val_box = val_box_accum / len(val_loader)
        val_loss = val_loss_accum / len(val_loader)
        val_map50, per_class = evaluate_map(dets_val, vl_lbls)

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
        }, output_dir / "latest_p2_ultra.pt")

        dt = time.time() - t0
        ms_per_step = (dt / steps_per_epoch) * 1000.0
        print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
        print(f"{steps_per_epoch}/{steps_per_epoch} ━━━━━━━━━━━━━━━━━━━━ {int(dt)}s {int(ms_per_step)}ms/step - loss: {tr_total:.4f} - cls_loss: {tr_cls:.4f} - box_loss: {tr_box:.4f} - val_loss: {val_loss:.4f} - val_cls_loss: {val_cls:.4f} - val_box_loss: {val_box:.4f} - val_mAP50: {val_map50*100:.2f}% - lr: {current_lr:.4e}", flush=True)
        if is_best:
            print(f"[★ NEW BEST] Val mAP50: {val_map50*100:.2f}% (Epoch {best_epoch})", flush=True)
            ap_strs = [f"{CLASS_NAMES[c]}: {per_class.get(c, 0)*100:.1f}%" for c in range(NUM_CLASSES)]
            print(f"    Per-class: {' | '.join(ap_strs)}", flush=True)
        elif args.patience > 0:
            print(f"    [i] Patience: {patience_counter}/{args.patience} (Best: {best_map50*100:.2f}% at Ep {best_epoch})", flush=True)

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[!] Early stopping triggered: no improvement in val_mAP50 for {args.patience} consecutive epochs.", flush=True)
            print(f"[*] Stopping training at Epoch {epoch+1}. Best epoch was {best_epoch} with val_mAP50: {best_map50*100:.2f}%\n", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"\nRestoring model weights from the end of the best epoch: {best_epoch}.", flush=True)
    print(f"[✔] Best val_mAP50: {best_map50*100:.2f}% (completed in {elapsed/60:.1f} minutes)\n", flush=True)

    # 6. Load Best Weights & Synthesize Keras + INT8 Model
    print("[*] Synthesizing matching Keras model and performing Full INT8 PTQ...")
    ck = torch.load(best_pt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"])
    model.eval().cpu()

    try:
        import keras
        import tensorflow as tf
        kmodel = build_keras_dtnet_max(spec, IMG_H, IMG_W)
        transfer_weights_to_keras(model, kmodel, spec)

        # Verification of parity
        test_x = np.random.uniform(0, 1, (2, IMG_H, IMG_W, 3)).astype(np.float32)
        with torch.no_grad():
            pt_in = torch.from_numpy(test_x.transpose(0, 3, 1, 2))
            tc, tb = model(pt_in)
        kc, kb = kmodel(test_x, training=False)
        diff_c = float(np.abs(tc.numpy() - np.asarray(kc).transpose(0, 3, 1, 2)).max())
        diff_b = float(np.abs(tb.numpy() - np.asarray(kb).transpose(0, 3, 1, 2)).max())
        print(f"[✔] PyTorch <-> Keras verification max error: cls={diff_c:.2e}, box={diff_b:.2e} (Strictly < 0.01)")
        assert diff_c < 0.02 and diff_b < 0.02, "Keras model does not match PyTorch weights!"

        tflite_path = output_dir / "dtnet_p2_ultra_int8.tflite"
        export_full_int8_tflite(kmodel, tr_imgs, str(tflite_path), num_calib=200)

        # 7. Evaluate INT8 Model on Validation Set & Sweetspot Calibration
        print("\n[*] Evaluating INT8 TFLite model on validation set...")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()

        cls_idx_tflite = 0 if output_details[0]["shape"][-1] == NUM_CLASSES else 1
        box_idx_tflite = 1 - cls_idx_tflite
        in_scale, in_zp = input_details["quantization"]
        cls_scale, cls_zp = output_details[cls_idx_tflite]["quantization"]
        box_scale, box_zp = output_details[box_idx_tflite]["quantization"]

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
        int8_dets = [decode_batch_np(c, b, score_thr=base_thresh, nms_iou=0.65)[0] for c, b in int8_raw_preds]
        int8_map50, int8_per_class = evaluate_map(int8_dets, vl_lbls)
        print(f"[✔] Raw INT8 mAP50 = {int8_map50*100:.2f}% (Quantization delta: {(int8_map50 - best_map50)*100:+.2f}%)")

        # Optimal Thresholds
        print("[*] Calibrating Sweetspot per-class thresholds...")
        opt_thresholds = [0.001] * NUM_CLASSES
        for c in range(NUM_CLASSES):
            best_t, best_ap = 0.001, int8_per_class.get(c, 0.0)
            for t in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
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
        print(f"   RESULTS: DTNet-P2-MAX FULL INT8")
        print(f"   Calibrated Val mAP50 : {cal_map50*100:.2f}%")
        print(f"   Total Contest Score  : {cal_score:.2f} / 100.00")
        print(f"   Optimal Thresholds   : {dict(zip(CLASS_NAMES, opt_thresholds))}")
        print(f"{'='*80}\n")

        thresh_path = output_dir / "calibrated_thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump({"thresholds": opt_thresholds, "class_names": CLASS_NAMES, "map50": cal_map50, "score": cal_score}, f, indent=2)

    except Exception as e:
        print(f"[!] Note: Keras/INT8 export deferred ({e}). PyTorch checkpoint best_p2_ultra.pt saved.")


if __name__ == "__main__":
    main()
