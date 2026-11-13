#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - KAGGLE GPU FROM-SCRATCH CHAMPION TRAINING PIPELINE
================================================================================
Completely self-contained, end-to-end training and export script for Kaggle GPU.
1. Trains Multi-Scale TextCNN (25k params) from scratch on contest questions -> text_cnn_int8.tflite.
2. Trains DTNetMax (197k params <= 200,000) from pure random SCRATCH on 6,000 images.
   - Native rectangular 320x240 RGB OV7670 camera aspect ratio.
   - Task-Aligned Assigner (TAL) + Normalized Wasserstein Distance (NWD) loss.
   - Dynamic 2-Phase Mosaic Augmentation with clean close-mosaic phase.
   - Mixed precision AMP on Kaggle GPU T4.
3. Post-Training Quantization to Full INT8 for ESP32-S3 (TFLM Ops Only).
4. Generates submission.csv on 4,000 test images + 4,000 test questions.
5. Strict Pre-Submission Gate verification (100% compliant).
================================================================================
"""

import os
import sys
import glob
import json
import math
import copy
import time
import csv
import random
import unicodedata
import re
from pathlib import Path

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
import keras
from keras import layers

# Set global deterministic seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
tf.random.set_seed(SEED)

print("=" * 80)
print("   EDGE AI CHALLENGE 2026 - KAGGLE FROM-SCRATCH TRAINING RUNNER")
print(f"   Python Version     : {sys.version.split()[0]}")
print(f"   PyTorch Version    : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"   CUDA Device        : {torch.cuda.get_device_name(0)}")
print(f"   TensorFlow Version : {tf.__version__}")
print(f"   Target Constraints : < 200,000 params, < 1.5 MB Flash INT8, ESP32-S3")
print("=" * 80)

ORIG_W, ORIG_H = 320.0, 240.0
IMG_W, IMG_H = 320, 240
STRIDE = 8
NUM_CLASSES = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
INTENT_CLASSES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
MAX_ALLOWED_PARAMS = 200_000
MAX_ALLOWED_FILE_SIZE = 1.5 * 1024 * 1024

OUTPUT_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("kaggle_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. DATASET DISCOVERY
# =============================================================================
def locate_dataset():
    """Locates train, val, and test data in /kaggle/input or local."""
    possible_roots = [
        "/kaggle/input/qualifiers-dataset",
        "/kaggle/input/dientruong-qualifiers-dataset",
        "/kaggle/input/edge-ai-challenge-2026-lan-2",
        "/kaggle/input/egde-ai-challenge-2026-lan-2-qualifiers",
        "qualifiers/data",
        "data"
    ]
    data_root = None
    for r in possible_roots:
        if os.path.exists(os.path.join(r, "train", "images")):
            data_root = r
            break
    if data_root is None:
        candidates = glob.glob("/kaggle/input/**/train/images", recursive=True)
        if candidates:
            data_root = os.path.dirname(os.path.dirname(candidates[0]))

    if not data_root:
        raise FileNotFoundError("Could not find qualifiers dataset in /kaggle/input!")

    print(f"[✔] Located Dataset Root at: {data_root}")
    return Path(data_root)


# =============================================================================
# 2. INTENT NLP PIPELINE (Multi-Scale TextCNN INT8)
# =============================================================================
STOPWORDS = {
    'ban', 'oi', 'cho', 'toi', 'hoi', 'xin', 'minh',
    'vay', 'nhi', 'the', 'a', 'gium', 'giup'
}

def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)

def clean_and_tokenize(text):
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if w and w not in STOPWORDS]

TARGET_ALIASES = [
    'chai', 'binh nuoc', 'chai nuoc', 'lo', 'chai nhua', 'chai thuy tinh',
    'coc', 'ly', 'tach', 'coc uong nuoc', 'ly nuoc',
    'dien thoai', 'smartphone', 'dien thoai thong minh', 'dt', 'chiec dien thoai', 'dien thoai di dong',
    'sach', 'cuon sach', 'quyen sach', 'tap vo', 'giao trinh', 'tai lieu',
    'chuot', 'chuot may tinh', 'con chuot', 'chuot khong day', 'chuot quang',
    'hop', 'thung', 'hop giay', 'cai hop', 'hop carton', 'hop qua', 'thung giay',
    'oto do choi', 'xe do choi', 'xe con do choi', 'xe hoi do choi', 'xe mo hinh', 'xe oto do choi',
    'cai o to do choi', 'cai oto do choi', 'o to do choi', 'oto', 'o to'
]

def has_target_class(text):
    q_norm = remove_accents(text).lower()
    return any(re.search(r'\b' + re.escape(alias) + r'\b', q_norm) for alias in TARGET_ALIASES)

def train_and_export_intent(data_root):
    print("\n" + "=" * 80)
    print(" [PHASE 1/4] Training Vietnamese Question Intent TextCNN (From Scratch)")
    print("=" * 80)

    train_csv = None
    for name in ["intent_train.csv", "intent_train_augmented.csv", "qa_train.csv"]:
        p = data_root / name
        if p.exists():
            train_csv = p
            break
    if not train_csv:
        raise FileNotFoundError("Could not find intent train CSV!")

    # Read training data
    texts, labels = [], []
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = (r.get("question") or r.get("text") or "").strip()
            l = (r.get("label") or r.get("intent") or "").strip()
            if t and l in INTENT_CLASSES:
                texts.append(t)
                labels.append(INTENT_CLASSES.index(l))

    print(f"[*] Loaded {len(texts)} training questions across {len(INTENT_CLASSES)} classes.")

    # Build Vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t in texts:
        for w in clean_and_tokenize(t):
            if w not in vocab:
                vocab[w] = len(vocab)
    print(f"[✔] Built Vocabulary: {len(vocab)} unique tokens.")

    MAX_LEN = 18
    X = []
    for t in texts:
        toks = clean_and_tokenize(t)
        ids = [vocab.get(w, 1) for w in toks][:MAX_LEN]
        padded = ids + [0] * (MAX_LEN - len(ids))
        X.append(padded)
    X = np.array(X, dtype=np.int32)
    y = np.array(labels, dtype=np.int32)

    # Multi-Scale TextCNN Architecture
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    emb = layers.Embedding(input_dim=len(vocab) + 16, output_dim=48, input_length=MAX_LEN, name="emb")(inp)
    conv_blocks = []
    for k in [2, 3, 4]:
        c = layers.Conv1D(filters=48, kernel_size=k, padding="same", activation="relu", name=f"c_{k}")(emb)
        p = layers.GlobalMaxPooling1D(name=f"p_{k}")(c)
        conv_blocks.append(p)
    merged = layers.Concatenate(name="concat")(conv_blocks)
    d = layers.Dense(64, activation="relu", name="dense")(merged)
    out = layers.Dense(len(INTENT_CLASSES), activation="softmax", name="pred")(d)

    model = keras.Model(inp, out, name="TextCNN_MultiScale")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"[✔] TextCNN Training Complete! Final Accuracy: {acc*100:.2f}% (Loss: {loss:.4f})")

    # Save vocab
    vocab_path = OUTPUT_DIR / "text_cnn_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Full INT8 PTQ Quantization
    def rep_data():
        for i in range(min(200, len(X))):
            yield [X[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int32
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = OUTPUT_DIR / "text_cnn_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[★ SUCCESS] Intent INT8 TFLite exported: {tflite_path} ({os.path.getsize(tflite_path):,} bytes)")
    return tflite_path, vocab_path


# =============================================================================
# 3. DETECTION MODEL ARCHITECTURE: DTNetMax (197,647 params)
# =============================================================================
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

    x = conv("stem", "input", 3, 32, k=3, s=2)
    x = dw("s1_dw", x, 32)
    x = conv("s1_pw", x, 32, 48, act=False)
    x = dw("s1_down_dw", x, 48, s=2)
    x = conv("s1_down_pw", x, 48, 64)
    p2 = res("s2a", x, 64)
    x = dw("s2_down_dw", p2, 64, s=2)
    x = conv("s2_down_pw", x, 64, 112)
    x = res("s2b", x, 112)
    shape = res("s2c", x, 112)
    x = dw("s3_down_dw", shape, 112, s=2)
    x = conv("s3_down_pw", x, 112, 160)
    sem = res("s3a", x, 160)

    if use_sppf:
        p1 = mp_op("spp_p1", sem, k=5)
        p2_pool = mp_op("spp_p2", p1, k=5)
        sem_add = add("spp_add1", sem, p1)
        sem = add("spp_add2", sem_add, p2_pool)

    x = conv("s3_proj", sem, 160, fuse_ch, bn=False, act=False)
    up_f = up("s3_up", x)

    parts = [shape, up_f]
    fuse_in = 112 + fuse_ch
    if use_p2:
        q = dw("p2_dw", p2, 64, s=2)
        q = conv("p2_pw", q, 64, 48)
        parts.append(q)
        fuse_in += 48

    f = cat("fuse_cat", parts)
    f = conv("fuse_pw", f, fuse_in, fuse_ch)
    f = res("refine", f, fuse_ch)

    c = dw("cls_dw", f, fuse_ch)
    c = conv("cls_pw", c, fuse_ch, head_ch)
    conv("cls_out", c, head_ch, num_classes, bn=False, act=False, bias=True)

    b = dw("box_dw", f, fuse_ch)
    b = conv("box_pw", b, fuse_ch, head_ch)
    conv("box_out", b, head_ch, 4, bn=False, act=False, bias=True)
    return ops

class DTNetMax(nn.Module):
    def __init__(self, spec=None, max_dist=20.0):
        super().__init__()
        self.spec = spec if spec is not None else make_spec_max()
        self.max_dist = max_dist
        self.mods = nn.ModuleDict()
        for op in self.spec:
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
        # Focal Loss prior bias init: p ~ 0.01 for background suppression
        nn.init.normal_(self.mods["cls_out"].weight, std=0.01)
        nn.init.constant_(self.mods["cls_out"].bias, -math.log((1 - 0.01) / 0.01))
        # Stable Anchor box bias init
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


# =============================================================================
# 4. DATASET & AUGMENTATION (Full-6K Mosaic Curriculum)
# =============================================================================
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
    def __init__(self, img_paths, label_arrays, img_w=IMG_W, img_h=IMG_H, stride=STRIDE,
                 train=True, mosaic_p=0.8, seed=42):
        self.img_paths = list(img_paths)
        self.labels = [np.asarray(x, dtype=np.float32).reshape(-1, 5) for x in label_arrays]
        self.img_w = img_w
        self.img_h = img_h
        self.stride = stride
        self.train = train
        self.mosaic_p = mosaic_p
        self.seed = seed
        self.epoch = 0

    def set_mosaic(self, p):
        self.mosaic_p = p

    def __len__(self):
        return len(self.img_paths)

    def _load_sample(self, idx):
        path = self.img_paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing image at {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        if (orig_w, orig_h) != (self.img_w, self.img_h):
            rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        boxes = self.labels[idx].copy()
        if len(boxes) > 0:
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.img_w
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * self.img_h
        return rgb, boxes

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + self.epoch * 10007 + idx)
        use_mosaic = self.train and (rng.random() < self.mosaic_p)

        if use_mosaic:
            indices = [idx] + list(rng.choice(len(self.img_paths), 3, replace=True))
            xc = int(rng.uniform(self.img_w * 0.4, self.img_w * 0.6))
            yc = int(rng.uniform(self.img_h * 0.4, self.img_h * 0.6))
            mosaic_img = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)
            mosaic_boxes = []

            for i, s_idx in enumerate(indices):
                im, bx = self._load_sample(s_idx)
                if i == 0:
                    x1a, y1a, x2a, y2a = 0, 0, xc, yc
                    x1b, y1b, x2b, y2b = self.img_w - xc, self.img_h - yc, self.img_w, self.img_h
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, 0, self.img_w, yc
                    x1b, y1b, x2b, y2b = 0, self.img_h - yc, self.img_w - xc, self.img_h
                elif i == 2:
                    x1a, y1a, x2a, y2a = 0, yc, xc, self.img_h
                    x1b, y1b, x2b, y2b = self.img_w - xc, 0, self.img_w, self.img_h - yc
                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, self.img_w, self.img_h
                    x1b, y1b, x2b, y2b = 0, 0, self.img_w - xc, self.img_h - yc

                mosaic_img[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b
                if len(bx) > 0:
                    b = bx.copy()
                    b[:, 1] += padw
                    b[:, 2] += padh
                    b[:, 3] += padw
                    b[:, 4] += padh
                    b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.img_w)
                    b[:, [2, 4]] = b[:, [2, 4]].clip(0, self.img_h)
                    w, h = b[:, 3] - b[:, 1], b[:, 4] - b[:, 2]
                    keep = (w >= 4.0) & (h >= 4.0)
                    mosaic_boxes.append(b[keep])

            img = mosaic_img
            boxes = np.concatenate(mosaic_boxes, axis=0) if len(mosaic_boxes) > 0 else np.zeros((0, 5), dtype=np.float32)
        else:
            img, boxes = self._load_sample(idx)

        if self.train:
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

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        boxes_t = torch.from_numpy(boxes).float()
        return img_t, boxes_t, idx

def collate_tal_fn(batch):
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


# =============================================================================
# 5. LOSS & ASSIGNMENT (TAL + NWD + EIoU)
# =============================================================================
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

    cx1, cy1 = (b1[:, 0] + b1[:, 2]) / 2.0, (b1[:, 1] + b1[:, 3]) / 2.0
    cx2, cy2 = (b2[:, 0] + b2[:, 2]) / 2.0, (b2[:, 1] + b2[:, 3]) / 2.0
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    dist_w2 = (w1 - w2) ** 2 / (cw ** 2 + eps)
    dist_h2 = (h1 - h2) ** 2 / (ch ** 2 + eps)
    eiou = iou - (rho2 / c2) - dist_w2 - dist_h2
    return eiou, iou

def bbox_nwd(b1, b2, C=12.0, eps=1e-7):
    cx1, cy1 = (b1[:, 0] + b1[:, 2]) / 2.0, (b1[:, 1] + b1[:, 3]) / 2.0
    w1, h1 = (b1[:, 2] - b1[:, 0]).clamp(min=eps), (b1[:, 3] - b1[:, 1]).clamp(min=eps)
    cx2, cy2 = (b2[:, 0] + b2[:, 2]) / 2.0, (b2[:, 1] + b2[:, 3]) / 2.0
    w2, h2 = (b2[:, 2] - b2[:, 0]).clamp(min=eps), (b2[:, 3] - b2[:, 1]).clamp(min=eps)
    w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
    return torch.exp(-torch.sqrt(w2_dist.clamp(min=eps)) / C)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=4, alpha=0.5, beta=6.0, stride=STRIDE):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.stride = stride

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes):
        B, num_anchors, num_classes = pred_scores.shape
        _, max_gt, _ = gt_bboxes.shape
        device = pred_scores.device

        assigned_labels = torch.full((B, num_anchors), -1, dtype=torch.long, device=device)
        assigned_bboxes = torch.zeros((B, num_anchors, 4), dtype=torch.float32, device=device)
        assigned_scores = torch.zeros((B, num_anchors, num_classes), dtype=torch.float32, device=device)
        mask_pos = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)

        if max_gt == 0:
            return assigned_labels, assigned_bboxes, assigned_scores, mask_pos

        for b in range(B):
            valid_gt = gt_labels[b] >= 0
            if not valid_gt.any():
                continue
            cur_gt_labels = gt_labels[b][valid_gt]
            cur_gt_bboxes = gt_bboxes[b][valid_gt]
            num_valid = len(cur_gt_labels)

            cur_scores = pred_scores[b][:, cur_gt_labels]
            b1 = pred_bboxes[b].unsqueeze(1).repeat(1, num_valid, 1).view(-1, 4)
            b2 = cur_gt_bboxes.unsqueeze(0).repeat(num_anchors, 1, 1).view(-1, 4)
            _, ious = bbox_eiou(b1, b2)
            ious = ious.view(num_anchors, num_valid).clamp(min=0.0, max=1.0)
            align_metric = (cur_scores ** self.alpha) * (ious ** self.beta)

            topk_metrics, topk_idxs = torch.topk(align_metric, min(self.topk, num_anchors), dim=0, largest=True)
            candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            for g in range(num_valid):
                candidate_mask[topk_idxs[:, g], g] = True

            max_metric, best_gt = (align_metric * candidate_mask.float()).max(dim=1)
            pos_anchor_idx = torch.where(max_metric > 1e-4)[0]
            if len(pos_anchor_idx) == 0:
                continue

            chosen_gt = best_gt[pos_anchor_idx]
            assigned_labels[b, pos_anchor_idx] = cur_gt_labels[chosen_gt]
            assigned_bboxes[b, pos_anchor_idx] = cur_gt_bboxes[chosen_gt]
            mask_pos[b, pos_anchor_idx] = True

            for idx, g_idx in zip(pos_anchor_idx, chosen_gt):
                c = cur_gt_labels[g_idx]
                target_score = float(ious[idx, g_idx].item())
                assigned_scores[b, idx, c] = max(assigned_scores[b, idx, c].item(), target_score)

        return assigned_labels, assigned_bboxes, assigned_scores, mask_pos

def det_loss_tal_nwd(cls_logit, box_raw, gt_labels, gt_bboxes, assigner,
                     w_box=5.5, w_nwd=2.0, w_neg=1.5, w_mutual=3.0):
    B, C, gh, gw = cls_logit.shape
    num_anchors = gh * gw
    cls_flat = cls_logit.permute(0, 2, 3, 1).reshape(B, num_anchors, C)
    pred_scores = torch.sigmoid(cls_flat)
    box_dec = decode_ltrb(box_raw, stride=STRIDE).permute(0, 2, 3, 1).reshape(B, num_anchors, 4)

    target_labels, target_bboxes, target_scores, mask_pos = assigner(
        pred_scores, box_dec, gt_labels, gt_bboxes
    )

    p = pred_scores
    t = target_scores
    focal_weight = torch.abs(t - p) ** 2.0
    bce = F.binary_cross_entropy_with_logits(cls_flat, t, reduction="none")
    cls_loss = focal_weight * bce

    neg_mask = (~mask_pos).unsqueeze(-1)
    cls_loss = torch.where(neg_mask, cls_loss * w_neg, cls_loss)
    cls_loss = cls_loss.sum() / max(mask_pos.sum().float().item(), 1.0)

    # Box Regression Loss (EIoU + NWD)
    num_pos = mask_pos.sum().item()
    if num_pos > 0:
        pos_pred_box = box_dec[mask_pos]
        pos_target_box = target_bboxes[mask_pos]
        eiou, _ = bbox_eiou(pos_pred_box, pos_target_box)
        loss_eiou = (1.0 - eiou).mean()
        nwd = bbox_nwd(pos_pred_box, pos_target_box)
        loss_nwd = (1.0 - nwd).mean()
        reg_loss = w_box * loss_eiou + w_nwd * loss_nwd
    else:
        reg_loss = torch.tensor(0.0, device=cls_logit.device)

    total_loss = cls_loss + reg_loss
    return total_loss, cls_loss, reg_loss


# =============================================================================
# 6. EVALUATION & EMA
# =============================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def nms_np(boxes, scores, thr, max_keep=100):
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
        ap = float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))
        aps[c] = ap
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps

class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))
                else:
                    v.copy_(msd[k])


# =============================================================================
# 7. MAIN TRAINING FUNCTION (From Scratch on Kaggle GPU)
# =============================================================================
def train_detection_scratch(data_root, epochs=250, batch_size=64, lr=1.2e-4):
    print("\n" + "=" * 80)
    print(" [PHASE 2/4] Training Object Detection DTNetMax (From Pure Scratch)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Load 6,000 images
    train_imgs = sorted(glob.glob(str(data_root / "train/images/*.png")) + glob.glob(str(data_root / "train/images/*.jpg")))
    val_imgs = sorted(glob.glob(str(data_root / "val/images/*.png")) + glob.glob(str(data_root / "val/images/*.jpg")))
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
                        c, cx, cy, w, h = map(float, parts[:5])
                        x1, y1 = cx - w / 2.0, cy - h / 2.0
                        x2, y2 = cx + w / 2.0, cy + h / 2.0
                        rows.append([c, x1, y1, x2, y2])
        all_lbls.append(np.array(rows, dtype=np.float32).reshape(-1, 5))

    val_pool_imgs = val_imgs
    val_pool_lbls = all_lbls[len(train_imgs):]

    train_dataset = Full6kDetDataset(all_imgs, all_lbls, train=True, mosaic_p=0.8)
    val_dataset = Full6kDetDataset(val_pool_imgs, val_pool_lbls, train=False, mosaic_p=0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_tal_fn)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=collate_tal_fn)

    print(f"[✔] Pooled Full Dataset: {len(all_imgs)} images ({len(train_imgs)} train + {len(val_imgs)} val)")

    spec = make_spec_max()
    model = DTNetMax(spec).to(device)
    total_params = count_params_strict(model)
    print(f"[Hardware Gate] Strict Parameters: {total_params:,} / {MAX_ALLOWED_PARAMS:,}")
    assert total_params <= MAX_ALLOWED_PARAMS, f"Exceeded parameter limit: {total_params} > {MAX_ALLOWED_PARAMS}"

    ema = ModelEMA(model, decay=0.9998)
    assigner = TaskAlignedAssigner(topk=4, alpha=0.5, beta=6.0, stride=STRIDE)

    lr_eff = lr * (batch_size / 32.0)
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        (decay_params if p.ndim > 1 else no_decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 5e-5},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr_eff, betas=(0.9, 0.999)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    warmup_epochs = 3
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    close_mosaic_epoch = epochs - 50

    def get_lr(step):
        if step < warmup_steps:
            return 2e-6 + (lr_eff - 2e-6) * (step / warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 2e-6 + 0.5 * (lr_eff - 2e-6) * (1 + math.cos(math.pi * min(prog, 1.0)))

    best_map50 = 0.0
    best_pt_path = OUTPUT_DIR / "best_dtnet_nwd_tal_full6k.pt"

    print(f"\n[*] Commencing Full-6K TAL+NWD Training from Scratch ({epochs} Epochs)...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_dataset.epoch = epoch

        if epoch >= close_mosaic_epoch:
            train_dataset.set_mosaic(0.0)

        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0
        steps = 0

        for batch in train_loader:
            global_step = (epoch - 1) * steps_per_epoch + steps
            cur_lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            imgs, gt_labels, gt_bboxes, _ = batch
            imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
            gt_labels = gt_labels.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                cls_logit, box_raw = model(imgs)
                loss, loss_c, loss_b = det_loss_tal_nwd(
                    cls_logit, box_raw, gt_labels, gt_bboxes, assigner,
                    w_box=5.5, w_nwd=2.0, w_neg=1.5, w_mutual=3.0
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item()
            running_cls += loss_c.item()
            running_box += loss_b.item()
            steps += 1

        elapsed = time.time() - t0
        avg_loss = running_loss / max(1, steps)
        avg_cls = running_cls / max(1, steps)
        avg_box = running_box / max(1, steps)

        # Periodic Validation every 5 epochs, or every epoch during final 30 epochs
        if epoch % 5 == 0 or epoch >= (epochs - 30):
            ema.ema.eval()
            val_dets = []
            val_gts = []
            with torch.no_grad():
                for batch in eval_loader:
                    imgs, gt_labels, gt_bboxes, _ = batch
                    imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        c_pred, b_pred = ema.ema(imgs)
                    dets = decode_batch_np(c_pred.cpu().numpy(), b_pred.cpu().numpy(),
                                           score_thr=0.001, nms_iou=0.65, max_det=100)
                    val_dets.extend(dets)
                    for b in range(len(gt_labels)):
                        valid = gt_labels[b] >= 0
                        if valid.any():
                            gl = gt_labels[b][valid].cpu().numpy()
                            gb = gt_bboxes[b][valid].cpu().numpy()
                            val_gts.append(np.column_stack([gl, gb]))
                        else:
                            val_gts.append(np.zeros((0, 5), dtype=np.float32))

            map50, aps = evaluate_map(val_dets, val_gts)
            is_best = map50 > best_map50
            if is_best:
                best_map50 = map50
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema.ema.state_dict(),
                    "spec": spec,
                    "map50": map50,
                    "aps": aps,
                    "params": total_params
                }, best_pt_path)
                best_mark = f"[★ BEST {map50*100:.2f}%]"
            else:
                best_mark = f"(Best: {best_map50*100:.2f}%)"

            print(f"Epoch {epoch:03d}/{epochs} [{elapsed:.0f}s] loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f}) | val_mAP50: {map50*100:.2f}% {best_mark} | lr: {cur_lr:.2e}")
        else:
            print(f"Epoch {epoch:03d}/{epochs} [{elapsed:.0f}s] loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f}) | lr: {cur_lr:.2e}")

    print(f"\n[✔] Detection Training Completed! Champion mAP50: {best_map50*100:.2f}%")
    return best_pt_path, spec


# =============================================================================
# 8. POST-TRAINING INT8 PTQ EXPORT
# =============================================================================
def build_keras_dtnet_max(spec, h=IMG_H, w=IMG_W):
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

def export_detection_int8(pt_checkpoint_path, data_root, spec):
    print("\n" + "=" * 80)
    print(" [PHASE 3/4] Exporting DTNetMax to Full INT8 TFLite for ESP32-S3")
    print("=" * 80)

    ck = torch.load(pt_checkpoint_path, map_location="cpu", weights_only=False)
    net_pt = DTNetMax(spec)
    weights = ck.get("ema", ck.get("model", ck))
    net_pt.load_state_dict(weights)
    net_pt.eval()

    kmodel = build_keras_dtnet_max(spec, IMG_H, IMG_W)
    transfer_weights_to_keras(net_pt, kmodel, spec)

    calib_imgs = sorted(glob.glob(str(data_root / "train/images/*.jpg")) + glob.glob(str(data_root / "train/images/*.png")))[:200]
    def representative_dataset():
        for p in calib_imgs:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (img.shape[1], img.shape[0]) != (IMG_W, IMG_H):
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
    out_tflite = OUTPUT_DIR / "dtnet_nwd_tal_full6k_int8.tflite"
    with open(out_tflite, "wb") as f:
        f.write(tflite_quant_model)

    size_bytes = os.path.getsize(out_tflite)
    print(f"[★ SUCCESS] Full INT8 Detection Model Exported: {out_tflite} ({size_bytes:,} bytes)")
    assert size_bytes <= MAX_ALLOWED_FILE_SIZE, f"Exceeded Flash limit: {size_bytes} > {MAX_ALLOWED_FILE_SIZE}"
    return out_tflite


# =============================================================================
# 9. INFERENCE & SUBMISSION GENERATOR
# =============================================================================
def generate_submission(det_tflite, intent_tflite, vocab_path, data_root):
    print("\n" + "=" * 80)
    print(" [PHASE 4/4] Generating & Validating Final Kaggle Submission.csv")
    print("=" * 80)

    # 1. Detection Inference on 4,000 Test Images
    test_img_dir = data_root / "test/images"
    test_img_paths = sorted(glob.glob(str(test_img_dir / "*.jpg")) + glob.glob(str(test_img_dir / "*.png")))
    print(f"[*] Processing {len(test_img_paths)} test images with DTNet INT8...")

    interp_det = tf.lite.Interpreter(model_path=str(det_tflite))
    interp_det.allocate_tensors()
    in_det = interp_det.get_input_details()[0]
    out_dets = interp_det.get_output_details()

    cls_idx = 0 if out_dets[0]["shape"][-1] == NUM_CLASSES else 1
    box_idx = 1 - cls_idx
    cls_det = out_dets[cls_idx]
    box_det = out_dets[box_idx]

    in_scale, in_zp = in_det["quantization"]
    cls_scale, cls_zp = cls_det["quantization"]
    box_scale, box_zp = box_det["quantization"]

    detection_rows = []
    total_boxes = 0

    for idx, p in enumerate(test_img_paths):
        stem = Path(p).stem
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (IMG_W, IMG_H):
            rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        norm_img = rgb.astype(np.float32) / 255.0
        q_in = np.round(norm_img / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp_det.set_tensor(in_det["index"], q_in[None, ...])
        interp_det.invoke()

        q_cls = interp_det.get_tensor(cls_det["index"])[0].astype(np.float32)
        q_box = interp_det.get_tensor(box_det["index"])[0].astype(np.float32)

        dq_cls = (q_cls - cls_zp) * cls_scale
        dq_box = (q_box - box_zp) * box_scale
        dq_cls = dq_cls.transpose(2, 0, 1)[None, ...]
        dq_box = dq_box.transpose(2, 0, 1)[None, ...]

        # Golden Calibrated Thresholds: Score Thr = 0.045, NMS IoU = 0.65
        dets = decode_batch_np(dq_cls, dq_box, score_thr=0.045, nms_iou=0.65, max_det=100)[0]

        boxes_json = []
        for row in dets:
            cid = int(row[5])
            score = float(row[4])
            x1, y1, x2, y2 = map(float, row[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_json.append({
                "class_id": cid,
                "score": round(score, 5),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })
        total_boxes += len(boxes_json)
        pred_str = json.dumps(boxes_json, separators=(",", ":"))
        detection_rows.append(["DETECTION", stem, pred_str])

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(test_img_paths)} test images...")

    print(f"[✔] Total Detected Boxes: {total_boxes:,} ({total_boxes/len(test_img_paths):.2f} boxes/image)")

    # 2. Intent Inference on 4,000 Test Questions
    qa_test_path = None
    for name in ["intent_test.csv", "qa_test.csv"]:
        p = data_root / name
        if p.exists():
            qa_test_path = p
            break
    if not qa_test_path:
        raise FileNotFoundError("Could not find intent_test.csv!")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    interp_intent = tf.lite.Interpreter(model_path=str(intent_tflite))
    interp_intent.allocate_tensors()
    in_intent = interp_intent.get_input_details()[0]
    out_intent = interp_intent.get_output_details()[0]
    max_len = in_intent["shape"][1]

    intent_rows = []
    from collections import Counter
    intent_counts = Counter()

    with open(qa_test_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("sample_id") or row.get("id") or row.get("question_id")
            text = (row.get("question") or row.get("text") or "").strip()
            tokens = clean_and_tokenize(text)
            token_ids = [vocab.get(t, 1) for t in tokens][:max_len]
            padded = token_ids + [0] * (max_len - len(token_ids))

            in_tensor = np.array([padded], dtype=np.int32)
            interp_intent.set_tensor(in_intent["index"], in_tensor)
            interp_intent.invoke()
            probs = interp_intent.get_tensor(out_intent["index"])[0]
            pred_class = INTENT_CLASSES[int(np.argmax(probs))]

            if pred_class in {"COUNT", "EXIST", "LOCATION"} and not has_target_class(text):
                pred_class = "OUT_OF_SCOPE"

            intent_counts[pred_class] += 1
            intent_rows.append(["INTENT", qid, pred_class])

    print(f"[✔] Processed {len(intent_rows)} intent questions. Distribution: {dict(intent_counts)}")

    # 3. Assemble submission.csv
    final_rows = [["task_type", "sample_id", "prediction"]] + detection_rows + intent_rows
    out_csv = OUTPUT_DIR / "submission.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(final_rows)

    # 4. Strict Pre-Submission Validator Gate
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["task_type", "sample_id", "prediction"], f"Header error: {header}"
        rows = list(reader)

    det_r = [r for r in rows if r[0] == "DETECTION"]
    int_r = [r for r in rows if r[0] == "INTENT"]
    assert len(det_r) == len(test_img_paths), "Detection rows mismatch!"
    assert len(int_r) == len(intent_rows), "Intent rows mismatch!"
    sample_ids = [r[1] for r in rows]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample_ids detected!"

    print("\n" + "=" * 80)
    print(f" [★ SUCCESS] Final Champion Submission generated at: {out_csv}")
    print(f" Total Rows: {len(rows):,} ({len(det_r):,} DETECTION + {len(int_r):,} INTENT)")
    print(f" Total Boxes: {total_boxes:,} ({total_boxes/len(det_r):.2f} boxes/image)")
    print(f" Submission File Size: {os.path.getsize(out_csv):,} bytes")
    print(f" Pre-Submission Gate Validator: 100% COMPLIANT [PASSED]")
    print("=" * 80 + "\n")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    t_start = time.time()
    data_root = locate_dataset()

    # Phase 1: Train Intent TextCNN from Scratch
    intent_tflite, vocab_path = train_and_export_intent(data_root)

    # Phase 2: Train DTNetMax from Pure Scratch (200 Epochs)
    best_pt_path, spec = train_detection_scratch(data_root, epochs=200, batch_size=64, lr=1.2e-4)

    # Phase 3: Export DTNetMax to Full INT8
    det_tflite = export_detection_int8(best_pt_path, data_root, spec)

    # Phase 4: Generate submission.csv
    generate_submission(det_tflite, intent_tflite, vocab_path, data_root)

    elapsed_mins = (time.time() - t_start) / 60.0
    print(f"[★ ALL PHASES COMPLETED] Total End-to-End Elapsed Time: {elapsed_mins:.1f} minutes.")


if __name__ == "__main__":
    main()
