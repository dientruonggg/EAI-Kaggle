#!/usr/bin/env python3
"""
dataset.py
================================================================================
Clean-Phase Native Resolution Dataset & Target Assignment
--------------------------------------------------------------------------------
Key Features:
1. Native 320x240 Geometry:
   - Zero-Mosaic (mosaic_p = 0.0 default for clean fine-tuning).
   - Eliminates mosaic border artifacts and preserves natural camera perspective.
2. Fine-Tuned Data Augmentation:
   - Random horizontal flip (p = 0.5).
   - Subtle scale jitter (0.94 - 1.06) & translation (+/- 6px).
   - Camera OV7670 sensor noise simulation (mild Gaussian noise + HSV jitter).
3. Collision-Free Adaptive Anchor Radius:
   - Small objects (< 24px: mouse, toy_car): Tight 1-4 anchor radius to prevent
     collision/overwrite between nearby objects.
   - Medium/Large objects: Proportional radius up to 3.5 grid cells.
================================================================================
"""

import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dtnet_clean_p2_200ep.dtnet_max import ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES


def assign_targets_clean(boxes, gw, gh, stride=STRIDE):
    """
    Assigns targets with collision-free adaptive radius.
    boxes: (N, 5) with [c, x1, y1, x2, y2]
    Returns:
    - cls_idx: (gh, gw) class_id >= 0 for positive cells, -1 for background
    - box_t: (4, gh, gw) ground-truth coordinates [x1, y1, x2, y2]
    """
    cls_idx = np.full((gh, gw), -1, dtype=np.int64)
    box_t = np.zeros((4, gh, gw), dtype=np.float32)
    best_area = np.full((gh, gw), np.inf, dtype=np.float32)

    # Sort boxes by area descending so smaller objects overwrite larger ones
    if len(boxes) > 0:
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        order = np.argsort(-areas)
        boxes = boxes[order]

    for c, x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ci = min(max(int(cx // stride), 0), gw - 1)
        cj = min(max(int(cy // stride), 0), gh - 1)

        # Collision-free adaptive radius
        min_dim = min(w, h)
        if min_dim < 24.0:
            rx = 0.9
            ry = 0.9
        else:
            rx = float(np.clip(0.28 * w / stride, 1.0, 3.5))
            ry = float(np.clip(0.28 * h / stride, 1.0, 3.5))

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


def _hsv_jitter(img, rng, gains=(0.015, 0.20, 0.20)):
    r = rng.uniform(-1, 1, 3) * np.asarray(gains) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge([cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class CleanDetDataset(Dataset):
    """
    Clean-Phase Native Resolution Dataset with subtle augmentations for fine-tuning.
    """
    def __init__(self, img_paths, label_arrays, img_w=IMG_W, img_h=IMG_H, stride=STRIDE,
                 train=True, mosaic_p=0.0, seed=42):
        self.img_paths = list(img_paths)
        self.labels = [np.asarray(x, dtype=np.float32).reshape(-1, 5) for x in label_arrays]
        self.img_w = img_w
        self.img_h = img_h
        self.stride = stride
        self.gw = img_w // stride
        self.gh = img_h // stride
        self.train = train
        self.mosaic_p = mosaic_p
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_mosaic(self, p):
        self.mosaic_p = p

    def __len__(self):
        return len(self.img_paths)

    def _load_sample(self, idx):
        path = self.img_paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing image at {path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.img_w or img.shape[0] != self.img_h:
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        boxes = self.labels[idx].copy()
        return img, boxes

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + self.epoch * 100000 + idx)
        img, boxes = self._load_sample(idx)

        if self.train:
            # Subtle Affine Translation & Scale Jitter
            s = rng.uniform(0.94, 1.06)
            tx = rng.uniform(-6.0, 6.0)
            ty = rng.uniform(-6.0, 6.0)

            M = np.array([
                [s, 0.0, (1.0 - s) * self.img_w * 0.5 + tx],
                [0.0, s, (1.0 - s) * self.img_h * 0.5 + ty]
            ], dtype=np.float32)
            img = cv2.warpAffine(img, M, (self.img_w, self.img_h), borderMode=cv2.BORDER_REFLECT101)

            if len(boxes) > 0:
                b = boxes.copy()
                b[:, [1, 3]] = b[:, [1, 3]] * s + (1.0 - s) * self.img_w * 0.5 + tx
                b[:, [2, 4]] = b[:, [2, 4]] * s + (1.0 - s) * self.img_h * 0.5 + ty
                b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.img_w)
                b[:, [2, 4]] = b[:, [2, 4]].clip(0, self.img_h)
                w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
                keep = (w >= 3.0) & (h >= 3.0)
                boxes = b[keep]

            # Horizontal Flip (p = 0.5)
            if rng.random() < 0.5:
                img = np.fliplr(img).copy()
                if len(boxes) > 0:
                    x1 = self.img_w - boxes[:, 3]
                    x2 = self.img_w - boxes[:, 1]
                    boxes[:, 1] = x1
                    boxes[:, 3] = x2

            # HSV Color Jitter
            img = _hsv_jitter(img, rng)

            # Camera Sensor Gaussian Noise Simulation (p = 0.3)
            if rng.random() < 0.3:
                noise = rng.normal(0, 3.5, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        cls_idx, box_t = assign_targets_clean(boxes, self.gw, self.gh, self.stride)

        # PyTorch format (C, H, W)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        cls_idx_t = torch.from_numpy(cls_idx).long()
        box_t_t = torch.from_numpy(box_t).float()

        return img_t, cls_idx_t, box_t_t, idx
