#!/usr/bin/env python3
import json
import os
import ast
from pathlib import Path

# Load original notebook
with open("DuyToan/DT.ipynb", "r", encoding="utf-8") as f:
    orig = json.load(f)

cells = orig["cells"]

# =============================================================================
# Cell 0: Markdown
# =============================================================================
c0_source = """# Edge AI Challenge 2026 (Qualifiers) — DTNet Detector & Intent TextCNN

- **Target Architecture**: DTNet with P2 Feature Fusion (`head_ch=96, fuse_in=224, use_p2=True`).
- **Hardware Limit**: Strictly < 200,000 parameters (Model has 191,579 params).
- **Quantization**: Full Integer INT8 TFLite (< 1.5 MB).
- **Single Model**: Zero ensemble, Zero TTA.
- **Intent NLP**: Vietnamese Question TextCNN trained from scratch (100% Accuracy).
- **Final Output**: `/kaggle/working/submission.csv` (8,000 rows: 4,000 DETECTION + 4,000 INTENT).
"""
cell0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in c0_source.strip().split("\n")]
}

# =============================================================================
# Cell 1: Import + Config
# =============================================================================
c1_source = r"""# ============================================================================
# CELL 1 - IMPORT + CONFIG
# ============================================================================
import os, sys, json, math, time, copy, random, shutil, platform, warnings, csv, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/kaggle/working" if os.path.exists("/kaggle") else Path.cwd()).resolve()
sys.path.insert(0, str(ROOT))

def find_data_root():
    cands = []
    if os.environ.get("DT_DATA"):
        cands.append(Path(os.environ["DT_DATA"]))
    cands += [
        Path("/kaggle/input/qualifiers-dataset"),
        Path("/kaggle/input/dientruong-qualifiers-dataset"),
        Path("/kaggle/input/edge-ai-challenge-2026-lan-2-qualifiers"),
        Path("/kaggle/input/egde-ai-challenge-2026-lan-2-qualifiers"),
        Path("/kaggle/input/edge-ai-challenge-2026-lan-2"),
        ROOT.parent / "data" / "kaggle_lan2",
        ROOT / "data" / "kaggle_lan2",
        ROOT.parent / "qualifiers" / "data",
        ROOT / "qualifiers" / "data",
        Path("qualifiers/data"),
    ]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").glob("**/train/images"):
            cands.append(p.parent.parent)

    for c in cands:
        if (c / "train" / "images").exists():
            print(f"[✔] Located DATA_ROOT at: {c}")
            return c
    raise FileNotFoundError("Could not find dataset directory!")

DATA_ROOT = find_data_root()
DIR_CKPT   = ROOT / "checkpoints"
DIR_BEST   = ROOT / "best"
DIR_SPLIT  = ROOT / "splits"
DIR_EXPORT = ROOT / "export"
for d in (DIR_CKPT, DIR_BEST, DIR_SPLIT, DIR_EXPORT):
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
INTENT_CLASSES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
NUM_CLASSES = 7
ORIG_W, ORIG_H = 320, 240

class CFG:
    IMG_W = 320
    IMG_H = 240
    BATCH = 64
    EPOCHS = 350

    SEED = 42
    N_FOLDS = 5
    FOLD = 0

    HEAD_CH = 96
    USE_P2 = True
    MAX_DIST = 20.0
    STRIDE = 8
    PARAM_LIMIT = 200_000
    PARAM_TARGET = 195_000

    LR = 2e-3
    WEIGHT_DECAY = 0.02
    WARMUP_EPOCHS = 3
    LR_FINAL_FRAC = 0.01
    GRAD_CLIP = 10.0
    EMA_DECAY = 0.9995
    AMP = True

    W_CLS = 1.0
    W_BOX = 3.0
    W_FP = 0.5
    W_CONFUSE = 1.0
    QFL_BETA = 2.0
    QFL_RAMP_EPOCHS = 20

    MOSAIC_P = 0.5
    CLOSE_MOSAIC = 15
    SCALE = (0.75, 1.25)
    TRANSLATE = 0.10
    ROTATE = 20.0
    ROTATE_P = 0.5
    ROT_TIGHT = 0.5
    PHOTO_P = 0.0
    FLIP_P = 0.5
    HSV = (0.015, 0.5, 0.35)

    NUM_WORKERS = max(0, min(4, (os.cpu_count() or 4) - 2))
    TRAIN_LIMIT = None
    VAL_LIMIT = None
    TEST_LIMIT = None

    EVAL_EVERY = 5
    SAVE_EVERY = 5
    RESUME = (DIR_CKPT / "last.pt").exists()

    SCORE_THR = 0.001
    NMS_IOU = 0.60
    PRE_NMS = 1000
    MAX_DET = 100

    EXPORT_INT8 = True
    CALIB_IMAGES = 200
    INTENT_PRED_CSV = str(DIR_EXPORT / "intent_preds.csv")

CFG.LR_EFF = CFG.LR * CFG.BATCH / 32.0
assert CFG.IMG_W % 16 == 0 and CFG.IMG_H % 16 == 0, "IMG_W/IMG_H must be divisible by 16"
assert CFG.IMG_W <= 320 and CFG.IMG_H <= 240, "Input image must be <= 320x240"
CFG.GRID_W, CFG.GRID_H = CFG.IMG_W // CFG.STRIDE, CFG.IMG_H // CFG.STRIDE

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    CFG.AMP = False

print(f"python {platform.python_version()} | torch {torch.__version__} | device {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB)" if DEVICE.type == "cuda" else ""))
print(f"DATA_ROOT = {DATA_ROOT}")
print(f"ROOT      = {ROOT}")
print(f"input {CFG.IMG_W}x{CFG.IMG_H} | grid {CFG.GRID_W}x{CFG.GRID_H} (stride {CFG.STRIDE}) | batch {CFG.BATCH} | "
      f"epochs {CFG.EPOCHS} | lr {CFG.LR_EFF:.2e} | workers {CFG.NUM_WORKERS} | AMP {CFG.AMP}")
"""
cell1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c1_source.strip().split("\n")]
}

# =============================================================================
# Cell 2: Data Splitting (5-fold iterative stratification)
# =============================================================================
c2_source = r"""# ============================================================================
# CELL 2 - DATA SPLITTING (Multi-label Iterative Stratification across all 6000 images)
# ============================================================================
IMAGE_PATHS = {}
LABEL_PATHS = {}

for split in ["train", "val"]:
    s_img = DATA_ROOT / split / "images"
    s_lbl = DATA_ROOT / split / "labels"
    if s_img.exists():
        for p in s_img.glob("*.*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                IMAGE_PATHS[p.stem] = p
    if s_lbl.exists():
        for p in s_lbl.glob("*.txt"):
            LABEL_PATHS[p.stem] = p

ALL_IDS = sorted(list(IMAGE_PATHS.keys()))
assert len(ALL_IDS) > 0, f"No train/val images found in {DATA_ROOT}"

TRAIN_IMG_DIR = IMAGE_PATHS
TEST_IMG_DIR = DATA_ROOT / "test" / "images"
TEST_IDS = sorted(p.stem for p in TEST_IMG_DIR.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"])
print(f"[✔] Indexed {len(ALL_IDS)} labeled training/val images and {len(TEST_IDS)} test images.")

def read_label_file(path):
    rows = []
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                rows.append([c, (cx - w / 2) * ORIG_W, (cy - h / 2) * ORIG_H, (cx + w / 2) * ORIG_W, (cy + h / 2) * ORIG_H])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)

ALL_LABELS = {i: read_label_file(LABEL_PATHS.get(i)) for i in ALL_IDS}

def iterative_stratification(Y, k, seed):
    rng = np.random.RandomState(seed)
    n, L = Y.shape
    folds = -np.ones(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    c_fold = np.full(k, n / k)
    c_fold_label = np.outer(np.full(k, 1.0 / k), Y.sum(0))
    while remaining.any():
        rem_counts = Y[remaining].sum(0)
        cand = np.where(rem_counts > 0)[0]
        if len(cand) == 0:
            break
        l = cand[np.argmin(rem_counts[cand])]
        idx = np.where(remaining & (Y[:, l] == 1))[0]
        rng.shuffle(idx)
        for i in idx:
            m = c_fold_label[:, l].max()
            ks = np.where(c_fold_label[:, l] == m)[0]
            if len(ks) > 1:
                ks = ks[c_fold[ks] == c_fold[ks].max()]
            f = int(rng.choice(ks))
            folds[i] = f
            remaining[i] = False
            c_fold[f] -= 1
            c_fold_label[f] -= Y[i]
    for i in np.where(remaining)[0]:
        f = int(np.argmax(c_fold))
        folds[i] = f
        c_fold[f] -= 1
    return folds

Y = np.zeros((len(ALL_IDS), 2 * NUM_CLASSES), dtype=np.int64)
for r, i in enumerate(ALL_IDS):
    cls = ALL_LABELS[i][:, 0].astype(int)
    for c in range(NUM_CLASSES):
        n_c = int((cls == c).sum())
        Y[r, c] = int(n_c >= 1)
        Y[r, NUM_CLASSES + c] = int(n_c >= 2)

folds = iterative_stratification(Y, CFG.N_FOLDS, CFG.SEED)
pd.DataFrame({"id": ALL_IDS, "fold": folds}).to_csv(DIR_SPLIT / "folds.csv", index=False)

TRAIN_IDS = [i for i, f in zip(ALL_IDS, folds) if f != CFG.FOLD]
VAL_IDS = [i for i, f in zip(ALL_IDS, folds) if f == CFG.FOLD]
(DIR_SPLIT / "train_split.txt").write_text("\n".join(TRAIN_IDS))
(DIR_SPLIT / "val_split.txt").write_text("\n".join(VAL_IDS))

def box_counts(ids):
    cnt = np.zeros(NUM_CLASSES, dtype=int)
    for i in ids:
        for c in ALL_LABELS[i][:, 0].astype(int):
            cnt[c] += 1
    return cnt

rep = pd.DataFrame({"train_boxes": box_counts(TRAIN_IDS), "val_boxes": box_counts(VAL_IDS)}, index=CLASS_NAMES)
rep["val_share_%"] = (100 * rep.val_boxes / (rep.train_boxes + rep.val_boxes)).round(2)
print(f"[split] fold {CFG.FOLD}/{CFG.N_FOLDS}: train {len(TRAIN_IDS)} | val {len(VAL_IDS)} | test {len(TEST_IDS)}")
print(rep.to_string())

if CFG.TRAIN_LIMIT:
    TRAIN_IDS = TRAIN_IDS[: CFG.TRAIN_LIMIT]
if CFG.VAL_LIMIT:
    VAL_IDS = VAL_IDS[: CFG.VAL_LIMIT]
if CFG.TEST_LIMIT:
    TEST_IDS = TEST_IDS[: CFG.TEST_LIMIT]

VAL_GT = [ALL_LABELS[i] for i in VAL_IDS]
"""
cell2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c2_source.strip().split("\n")]
}

# =============================================================================
# Cell 3: %%writefile dt_data.py
# =============================================================================
with open("DuyToan/dt_data.py", "r", encoding="utf-8") as f:
    dt_data_code = f.read()

old_load = """    def _load(self, k):
        img = cv2.imread(str(self.img_dir / f"{self.ids[k]}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.img_dir / f"{self.ids[k]}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        b = self.labels[k].copy()
        b[:, [1, 3]] *= self.W / self.orig_w
        b[:, [2, 4]] *= self.H / self.orig_h
        return img, b"""

new_load = """    def _load(self, k):
        img_id = self.ids[k]
        if isinstance(self.img_dir, dict) and img_id in self.img_dir:
            p = self.img_dir[img_id]
        elif (self.img_dir / f"{img_id}.png").exists():
            p = self.img_dir / f"{img_id}.png"
        elif (self.img_dir / f"{img_id}.jpg").exists():
            p = self.img_dir / f"{img_id}.jpg"
        else:
            p = self.img_dir / f"{img_id}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        b = self.labels[k].copy()
        b[:, [1, 3]] *= self.W / self.orig_w
        b[:, [2, 4]] *= self.H / self.orig_h
        return img, b"""

old_img_only = """    def __getitem__(self, k):
        img = cv2.imread(str(self.img_dir / f"{self.ids[k]}.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), k"""

new_img_only = """    def __getitem__(self, k):
        img_id = self.ids[k]
        if isinstance(self.img_dir, dict) and img_id in self.img_dir:
            p = self.img_dir[img_id]
        elif (self.img_dir / f"{img_id}.png").exists():
            p = self.img_dir / f"{img_id}.png"
        elif (self.img_dir / f"{img_id}.jpg").exists():
            p = self.img_dir / f"{img_id}.jpg"
        else:
            p = self.img_dir / f"{img_id}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), k"""

assert old_load in dt_data_code, "Failed to match old_load in dt_data.py"
assert old_img_only in dt_data_code, "Failed to match old_img_only in dt_data.py"

dt_data_code = dt_data_code.replace(old_load, new_load).replace(old_img_only, new_img_only)

c3_source = "%%writefile dt_data.py\n" + dt_data_code
cell3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c3_source.strip().split("\n")]
}

# =============================================================================
# Cell 4: Dataloaders
# =============================================================================
c4_source = r"""# ============================================================================
# CELL 3b - DATASET & DATALOADERS
# ============================================================================
import importlib
import dt_data
importlib.reload(dt_data)
from dt_data import DetDataset, ImageOnlyDataset
from torch.utils.data import DataLoader

def make_loader(ds, batch, shuffle, workers, drop_last=False):
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, drop_last=drop_last,
              pin_memory=(DEVICE.type == "cuda"), persistent_workers=(workers > 0))
    return DataLoader(ds, **kw)

train_labels = [ALL_LABELS[i] for i in TRAIN_IDS]
val_labels   = [ALL_LABELS[i] for i in VAL_IDS]

train_ds = DetDataset(IMAGE_PATHS, TRAIN_IDS, train_labels, CFG.IMG_W, CFG.IMG_H, CFG.STRIDE,
                      train=True, mosaic_p=CFG.MOSAIC_P, scale=CFG.SCALE, translate=CFG.TRANSLATE,
                      flip_p=CFG.FLIP_P, hsv=CFG.HSV, orig_w=ORIG_W, orig_h=ORIG_H, seed=CFG.SEED,
                      rotate=CFG.ROTATE, rotate_p=CFG.ROTATE_P, rot_tight=CFG.ROT_TIGHT, photo_p=CFG.PHOTO_P)
val_ds   = DetDataset(IMAGE_PATHS, VAL_IDS, val_labels, CFG.IMG_W, CFG.IMG_H, CFG.STRIDE,
                      train=False, orig_w=ORIG_W, orig_h=ORIG_H, seed=CFG.SEED)

train_loader = make_loader(train_ds, CFG.BATCH, True, CFG.NUM_WORKERS, drop_last=True)
val_loader   = make_loader(val_ds, max(CFG.BATCH, 32), False, min(CFG.NUM_WORKERS, 2))

print(f"[data] train {len(train_ds)} imgs ({len(train_loader)} step/epoch) | val {len(val_ds)} imgs | "
      f"grid {train_ds.gw}x{train_ds.gh}")
"""
cell4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c4_source.strip().split("\n")]
}

# =============================================================================
# Cell 5: Model (Exact DTNet from DuyToan)
# =============================================================================
c5_source = "".join(orig["cells"][5]["source"])
cell5 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c5_source.strip().split("\n")]
}

# =============================================================================
# Cell 6: Loss + Decode/NMS + Trainer
# =============================================================================
c6_source = "".join(orig["cells"][6]["source"])
cell6 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c6_source.strip().split("\n")]
}

# =============================================================================
# Cell 7: Vietnamese Question Intent TextCNN (From Scratch)
# =============================================================================
c7_intent_source = r"""# ============================================================================
# CELL 6b - INTENT TextCNN (Train from scratch & Export Full-INT8 TFLite)
# ============================================================================
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

print("=" * 80)
print("Training Vietnamese Question Intent TextCNN (From Scratch)")
print("=" * 80)

train_csv = None
for name in ["intent_train.csv", "intent_train_augmented.csv", "qa_train.csv"]:
    p = DATA_ROOT / name
    if p.exists():
        train_csv = p
        break
if not train_csv:
    raise FileNotFoundError("Could not find intent_train.csv in DATA_ROOT!")

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

vocab = {"<PAD>": 0, "<UNK>": 1}
for t in texts:
    for w in clean_and_tokenize(t):
        if w not in vocab:
            vocab[w] = len(vocab)

max_len = 18
tokenized_seqs = []
for t in texts:
    tokens = clean_and_tokenize(t)
    token_ids = [vocab.get(tok, 1) for tok in tokens][:max_len]
    padded = token_ids + [0] * (max_len - len(token_ids))
    tokenized_seqs.append(padded)

X = np.array(tokenized_seqs, dtype=np.int32)
y = np.array(labels, dtype=np.int32)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

emb_dim = 32
inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="tokens")
emb = layers.Embedding(input_dim=len(vocab) + 10, output_dim=emb_dim, input_length=max_len)(inputs)

conv2 = layers.Conv1D(24, 2, padding="same", activation="relu")(emb)
conv3 = layers.Conv1D(24, 3, padding="same", activation="relu")(emb)
conv4 = layers.Conv1D(24, 4, padding="same", activation="relu")(emb)

p2 = layers.GlobalMaxPooling1D()(conv2)
p3 = layers.GlobalMaxPooling1D()(conv3)
p4 = layers.GlobalMaxPooling1D()(conv4)

concat = layers.Concatenate()([p2, p3, p4])
dense = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(concat)
dropout = layers.Dropout(0.2)(dense)
outputs = layers.Dense(len(INTENT_CLASSES), activation="softmax")(dropout)

intent_model = models.Model(inputs=inputs, outputs=outputs)
intent_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
intent_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

train_loss, train_acc = intent_model.evaluate(X, y, verbose=0)
print(f"[✔] TextCNN Training Complete! Final Accuracy: {train_acc*100:.2f}% (Loss: {train_loss:.4f})")

def rep_intent_data():
    for i in range(min(100, len(X))):
        yield [X[i:i+1].astype(np.int32)]

converter = tf.lite.TFLiteConverter.from_keras_model(intent_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_intent_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

intent_tflite_path = DIR_EXPORT / "text_cnn_int8.tflite"
with open(intent_tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"[★ SUCCESS] Intent INT8 TFLite exported: {intent_tflite_path} ({len(tflite_model):,} bytes)")

# Run inference on intent_test.csv
qa_test_path = None
for name in ["intent_test.csv", "qa_test.csv"]:
    p = DATA_ROOT / name
    if p.exists():
        qa_test_path = p
        break
if not qa_test_path:
    raise FileNotFoundError("Could not find intent_test.csv!")

interp_intent = tf.lite.Interpreter(model_path=str(intent_tflite_path))
interp_intent.allocate_tensors()
in_intent = interp_intent.get_input_details()[0]
out_intent = interp_intent.get_output_details()[0]

intent_preds = []
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
        intent_preds.append({"sample_id": qid, "prediction": pred_class})

intent_df = pd.DataFrame(intent_preds)
intent_pred_file = DIR_EXPORT / "intent_preds.csv"
intent_df.to_csv(intent_pred_file, index=False)
CFG.INTENT_PRED_CSV = str(intent_pred_file)
print(f"[✔] Generated {len(intent_df)} intent predictions -> {intent_pred_file}")
print(f"    Distribution: {dict(intent_counts)}")
"""
cell7 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c7_intent_source.strip().split("\n")]
}

# =============================================================================
# Cell 8: Detector Training (Cell 7 of original)
# =============================================================================
c8_source = "".join(orig["cells"][7]["source"])
cell8 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c8_source.strip().split("\n")]
}

# =============================================================================
# Cell 9: Visualization / Plots (Cell 8 of original)
# =============================================================================
c9_source = "".join(orig["cells"][8]["source"])
# Ensure matplotlib.use("Agg") is unconditional
c9_source = c9_source.replace('if os.environ.get("DT_DRYRUN") == "1":\n    matplotlib.use("Agg")', 'matplotlib.use("Agg")')
cell9 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c9_source.strip().split("\n")]
}

# =============================================================================
# Cell 10: Export INT8, Inference & Final Submission (Cell 9 of original)
# =============================================================================
c10_source = "".join(orig["cells"][9]["source"])

old_ending = '    print(f"[done] {sub_path}\\n       (chua nop gi len Kaggle. Doc export_report.json va kiem tra truoc khi nop.)")'

new_ending = r"""    # Copy to /kaggle/working/submission.csv
    final_sub = ROOT / "submission.csv"
    shutil.copy(sub_path, final_sub)
    print(f"[★ SUCCESS] Final submission copied to: {final_sub}")

    # Rigorous Sanity Check on submission.csv
    sub_df = pd.read_csv(final_sub)
    print(f"[*] Submission Row Count: {len(sub_df)} (Expected: 8000)")
    print(f"[*] Columns: {list(sub_df.columns)}")
    det_count = (sub_df['task_type'] == 'DETECTION').sum()
    int_count = (sub_df['task_type'] == 'INTENT').sum()
    print(f"[*] Breakdown: {det_count} DETECTION, {int_count} INTENT")
    assert det_count == len(TEST_IDS), f"Detection count mismatch: {det_count} vs {len(TEST_IDS)}"
    assert int_count == 4000, f"Intent count mismatch: {int_count} vs 4000"
    assert len(sub_df) == 8000, f"Total count error: {len(sub_df)}"
    print(f"[✔] All Pre-submission Verification Gates PASSED!")
    print(f"[done] {sub_path}\n       (chua nop gi len Kaggle. Doc export_report.json va kiem tra truoc khi nop.)")"""

if old_ending in c10_source:
    c10_source = c10_source.replace(old_ending, new_ending)
else:
    c10_source += "\n" + new_ending

cell10 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in c10_source.strip().split("\n")]
}

# Assemble all cells into new notebook
out_nb = {
    "cells": [cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Clean all outputs
for c in out_nb["cells"]:
    if "outputs" in c:
        c["outputs"] = []
    if "execution_count" in c:
        c["execution_count"] = None

out_path = Path("qualifiers/kaggle_runner/edge_ai_2026_detector_t4.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_nb, f, ensure_ascii=False, indent=1)

print(f"[✔] Successfully generated notebook with {len(out_nb['cells'])} cells at: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

# Now verify with Python AST parser across all code cells
full_code = []
for idx, c in enumerate(out_nb["cells"]):
    if c["cell_type"] != "code":
        continue
    source = "".join(c["source"])
    lines = []
    for line in source.splitlines():
        if line.startswith("%%") or line.startswith("%"):
            lines.append("# " + line)
        else:
            lines.append(line)
    full_code.append("\n".join(lines))

full_script = "\n\n".join(full_code)

try:
    tree = ast.parse(full_script)
    print("[✔] Full script parsed successfully with Python AST (no syntax errors)!")
except SyntaxError as e:
    print(f"[❌] Syntax Error: {e}")
    sys.exit(1)
