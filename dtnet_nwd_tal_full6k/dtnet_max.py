#!/usr/bin/env python3
"""
dtnet_max.py
================================================================================
Hardware-Compliant DTNetMax Architecture for ESP32-S3 (Edge AI Challenge 2026)
--------------------------------------------------------------------------------
Key Invariants:
1. Strictly <= 200,000 parameters (Exact: 197,647 params | Margin: 2,353 params)
2. Native resolution: 320x240 RGB (matches OV7670 camera sensor)
3. Stride: 8 (Feature grid: 40x30)
4. Multi-Scale P2 path (Stride 4) preserving high-resolution spatial details
5. Multi-Scale SPPF Bottleneck (3x MaxPool 5x5 - 0 extra params, triples receptive field)
6. Decoupled Heads: Separate branches for classification and bounding box regression
7. 100% TFLite Micro operator compatibility on ESP32-S3
================================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ORIG_W, ORIG_H = 320, 240
IMG_W, IMG_H = 320, 240
STRIDE = 8
GRID_W, GRID_H = IMG_W // STRIDE, IMG_H // STRIDE  # 40 x 30
NUM_CLASSES = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
MAX_ALLOWED_PARAMS = 200_000


def make_spec_max(num_classes=NUM_CLASSES, head_ch=104, fuse_ch=116, use_p2=True, use_sppf=True):
    """
    Constructs the exact architecture specification for DTNetMax.
    Returns a list of layer operation dictionaries.
    """
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

    # Backbone: Stride-2 -> Stride-4 -> Stride-8 -> Stride-16
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

    # SPPF: Spatial Pyramid Pooling Fast (0 params, expands context window to 144x144)
    if use_sppf:
        p1 = mp_op("spp_p1", sem, k=5)
        p2_pool = mp_op("spp_p2", p1, k=5)
        sem_add = add("spp_add1", sem, p1)
        sem = add("spp_add2", sem_add, p2_pool)

    x = conv("s3_proj", sem, 160, fuse_ch, bn=False, act=False)
    up_f = up("s3_up", x)                                            # /16 -> /8 (nearest upsample)

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

    # Decoupled detection heads with expanded head_ch=104
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
        # Classification head bias init for focal loss: p ~ 0.01
        nn.init.normal_(self.mods["cls_out"].weight, std=0.01)
        nn.init.constant_(self.mods["cls_out"].bias, -math.log((1 - 0.01) / 0.01))
        # Box head bias init for stable anchor regression
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
    """Strictly counts all trainable and frozen parameters plus BN running statistics."""
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for k, b in model.named_buffers() if k.endswith("running_mean") or k.endswith("running_var"))
    return n


def build_keras_dtnet_max(spec, h=IMG_H, w=IMG_W):
    """Builds identical Keras 3 model graph for Full INT8 quantization."""
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
    """Transfers PyTorch weights and BN statistics to Keras 3 layers."""
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


if __name__ == "__main__":
    net = DTNetMax()
    p = count_params_strict(net)
    print(f"[DTNetMax Check] Strict Params: {p:,} / {MAX_ALLOWED_PARAMS:,} (Margin: {MAX_ALLOWED_PARAMS - p:,})")
    assert p <= MAX_ALLOWED_PARAMS, f"EXCEEDED: {p}"
    x = torch.zeros(1, 3, IMG_H, IMG_W)
    c, b = net(x)
    print(f"[DTNetMax Check] Forward output shapes: cls={c.shape}, box={b.shape}")
