# ============================================================================
# dt_data.py  -  Dataset + augmentation + gan target (CELL 3a ghi file nay)
# Viet thanh module rieng de DataLoader(num_workers>0) chay duoc tren Windows (spawn).
# ============================================================================
import math
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)


def assign_targets(boxes, gw, gh, stride, min_r=1.5, max_r=3.0, rad_scale=0.3):
    """Gan target tren luoi (gh, gw) cho cac box (N,5)=[cls,x1,y1,x2,y2] (pixel, he toa do input).
    - Moi vat the: o chua tam + cac o co TAM nam trong box va gan tam (center sampling ~3x3, vat to toi da 7x7).
    - o bi nhieu vat tranh nhau -> uu tien vat NHO HON (bao ve vat nho).
    Tra ve cls_idx (gh,gw) int64 (-1 = nen) va box_t (4,gh,gw) float32 xyxy pixel.
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
        r = float(np.clip(rad_scale * min(w, h) / stride, min_r, max_r))
        reach = int(math.ceil(r))
        area = w * h
        for j in range(max(0, cj - reach), min(gh, cj + reach + 1)):
            ccy = (j + 0.5) * stride
            for i in range(max(0, ci - reach), min(gw, ci + reach + 1)):
                ccx = (i + 0.5) * stride
                is_center = (i == ci and j == cj)
                inside = (x1 <= ccx <= x2) and (y1 <= ccy <= y2)
                near = abs(ccx - cx) <= r * stride and abs(ccy - cy) <= r * stride
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
    keep = (w >= min_size) & (h >= min_size) & (w * h >= min_vis * area0)   # bo box bi cat qua nhieu
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


class DetDataset(Dataset):
    """Tra ve (img uint8 [3,H,W], cls_idx [gh,gw], box_t [4,gh,gw], idx).
    train=True: mosaic + scale/translate + HSV + lat ngang. train=False: chi resize (neu can)."""

    def __init__(self, img_dir, ids, labels, img_w, img_h, stride, train,
                 mosaic_p=0.5, scale=(0.75, 1.25), translate=0.1, flip_p=0.5,
                 hsv=(0.015, 0.5, 0.35), orig_w=320, orig_h=240, seed=0):
        self.img_dir = Path(img_dir)
        self.ids, self.labels = list(ids), list(labels)
        self.W, self.H, self.stride = int(img_w), int(img_h), int(stride)
        self.gw, self.gh = self.W // self.stride, self.H // self.stride
        self.train = train
        self.scale, self.translate, self.flip_p, self.hsv = scale, translate, flip_p, hsv
        self.orig_w, self.orig_h, self.seed = orig_w, orig_h, seed
        self.epoch = mp.Value("i", 0)             # dung chung voi workers (persistent_workers)
        self.mosaic_p = mp.Value("d", float(mosaic_p))

    def set_epoch(self, e):
        self.epoch.value = int(e)

    def set_mosaic(self, p):
        self.mosaic_p.value = float(p)

    def __len__(self):
        return len(self.ids)

    # -------------------------------------------------------------------- #
    def _load(self, k):
        img = cv2.imread(str(self.img_dir / f"{self.ids[k]}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.img_dir / f"{self.ids[k]}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        b = self.labels[k].copy()
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
        ids = [k] + [int(x) for x in rng.integers(0, len(self.ids), 3)]
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
                # cat theo vung anh duoc dan vao canvas; bo box con thay < 40% dien tich goc
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

    # -------------------------------------------------------------------- #
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
        cls_idx, box_t = assign_targets(b, self.gw, self.gh, self.stride)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img, torch.from_numpy(cls_idx), torch.from_numpy(box_t), k


class ImageOnlyDataset(Dataset):
    """Cho du doan (test/calibration): tra ve (img uint8 [3,H,W], idx)."""

    def __init__(self, img_dir, ids, img_w, img_h):
        self.img_dir, self.ids, self.W, self.H = Path(img_dir), list(ids), int(img_w), int(img_h)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        img = cv2.imread(str(self.img_dir / f"{self.ids[k]}.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), k
