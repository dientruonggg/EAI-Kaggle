#!/usr/bin/env python3
"""
dataset.py
================================================================================
Full-6,000 Dataset Loader & 2-Phase Curriculum for TAL + NWD Training
================================================================================
"""

import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dtnet_nwd_tal_full6k.dtnet_max import ORIG_W, ORIG_H, IMG_W, IMG_H, STRIDE, NUM_CLASSES


def _hsv_jitter(img, rng, gains=(0.015, 0.25, 0.25)):
    r = rng.uniform(-1, 1, 3) * np.asarray(gains) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge([cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class Full6kDetDataset(Dataset):
    """
    Loads all 6,000 images with 2-Phase Curriculum:
    Phase 1: Dynamic Mosaic + Affine Scale + Sensor Noise
    Phase 2: Clean-Phase (Zero-Mosaic, Native 320x240)
    """
    def __init__(self, img_paths, label_arrays, img_w=IMG_W, img_h=IMG_H, stride=STRIDE,
                 train=True, mosaic_p=0.5, seed=42):
        self.img_paths = list(img_paths)
        self.labels = [np.asarray(x, dtype=np.float32).reshape(-1, 5) for x in label_arrays]
        self.img_w = img_w
        self.img_h = img_h
        self.stride = stride
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

    def _mosaic4(self, idx, rng):
        W, H = self.img_w, self.img_h
        xc = int(rng.uniform(W * 0.35, W * 0.65))
        yc = int(rng.uniform(H * 0.35, H * 0.65))

        indices = [idx] + list(rng.choice(len(self.img_paths), 3, replace=False))
        out_img = np.full((H, W, 3), 114, dtype=np.uint8)
        out_boxes = []

        for i, idx_i in enumerate(indices):
            im, bx = self._load_sample(idx_i)
            h, w, _ = im.shape

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, W), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(H, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, W), min(H, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            out_img[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(bx) > 0:
                b = bx.copy()
                b[:, 1] += padw
                b[:, 2] += padh
                b[:, 3] += padw
                b[:, 4] += padh
                out_boxes.append(b)

        if len(out_boxes) > 0:
            boxes = np.concatenate(out_boxes, axis=0)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, W)
            boxes[:, [2, 4]] = np.clip(boxes[:, [2, 4]], 0, H)
            w = boxes[:, 3] - boxes[:, 1]
            h = boxes[:, 4] - boxes[:, 2]
            keep = (w >= 3.0) & (h >= 3.0)
            boxes = boxes[keep]
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)

        return out_img, boxes

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + self.epoch * 100000 + idx)

        if self.train and rng.random() < self.mosaic_p:
            img, boxes = self._mosaic4(idx, rng)
        else:
            img, boxes = self._load_sample(idx)

        if self.train:
            # Subtle Affine Translation & Scale Jitter
            s = rng.uniform(0.92, 1.08)
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

            # HSV Jitter
            img = _hsv_jitter(img, rng)

            # Camera Sensor Gaussian Noise Simulation (p = 0.25)
            if rng.random() < 0.25:
                noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # PyTorch format (C, H, W)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        boxes_t = torch.from_numpy(boxes).float()

        return img_t, boxes_t, idx


def collate_tal_fn(batch):
    """
    Optimized custom collate function for Vectorized TAL.
    Pre-packs variable-length bounding boxes into uniform tensors on CPU:
    - imgs: (B, 3, H, W) tensor
    - gt_labels: (B, M) long tensor (-1 for padding)
    - gt_bboxes: (B, M, 4) float tensor
    - indices: (B,) tensor
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    boxes_list = [item[1] for item in batch]
    indices = torch.tensor([item[2] for item in batch], dtype=torch.long)

    max_m = max([len(b) for b in boxes_list] + [1])
    B = len(batch)
    gt_labels = torch.full((B, max_m), -1, dtype=torch.long)
    gt_bboxes = torch.zeros((B, max_m, 4), dtype=torch.float32)

    for b, bx in enumerate(boxes_list):
        if len(bx) > 0:
            gt_labels[b, :len(bx)] = bx[:, 0].long()
            gt_bboxes[b, :len(bx)] = bx[:, 1:5].float()

    return imgs, gt_labels, gt_bboxes, indices
