import torch
import torch.nn as nn
import math

def count_params_strict(model):
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for k, b in model.named_buffers() if k.endswith("running_mean") or k.endswith("running_var"))
    return n

def test_config(head_ch=96, fuse_ch=112, use_sppf=True, extra_refine=False):
    ops = []
    def conv(name, src, cin, cout, k=1, s=1, bn=True, act=True, bias=False, zero_gamma=False):
        ops.append(dict(t="conv", name=name, src=src, cin=cin, cout=cout, k=k, s=s, bn=bn, act=act, bias=bias, zero_gamma=zero_gamma))
        return name
    def dw(name, src, c, k=3, s=1, bn=True, act=True):
        ops.append(dict(t="dw", name=name, src=src, cin=c, cout=c, k=k, s=s, bn=bn, act=act, bias=False, zero_gamma=False))
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
    def mp(name, src, k=5):
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
        p1 = mp("spp_p1", sem, k=5)
        p2_pool = mp("spp_p2", p1, k=5)
        sem_add = add("spp_add1", sem, p1)
        sem = add("spp_add2", sem_add, p2_pool)

    x = conv("s3_proj", sem, 160, fuse_ch, bn=False, act=False)
    up_f = up("s3_up", x)
    
    q = dw("p2_dw", p2, 64, s=2)
    q = conv("p2_pw", q, 64, 48)
    
    parts = [shape, up_f, q]
    fuse_in = 112 + fuse_ch + 48
    f = cat("fuse_cat", parts)
    f = conv("fuse_pw", f, fuse_in, fuse_ch)
    f = res("refine", f, fuse_ch)
    if extra_refine:
        f = res("refine2", f, fuse_ch)

    c = dw("cls_dw", f, fuse_ch)
    c = conv("cls_pw", c, fuse_ch, head_ch)
    conv("cls_out", c, head_ch, 7, bn=False, act=False, bias=True)

    b = dw("box_dw", f, fuse_ch)
    b = conv("box_pw", b, fuse_ch, head_ch)
    conv("box_out", b, head_ch, 4, bn=False, act=False, bias=True)

    # Build torch model to count params strictly
    class Net(nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.spec = spec
            self.mods = nn.ModuleDict()
            for op in spec:
                if op["t"] in ("conv", "dw"):
                    k, s = op["k"], op["s"]
                    pad = (k - 1) // 2 if s == 1 else 0
                    groups = op["cin"] if op["t"] == "dw" else 1
                    self.mods[op["name"]] = nn.Conv2d(op["cin"], op["cout"], k, s, pad, groups=groups, bias=op["bias"])
                    if op["bn"]:
                        self.mods[op["name"] + "_bn"] = nn.BatchNorm2d(op["cout"])
                elif op["t"] == "mp":
                    self.mods[op["name"]] = nn.MaxPool2d(op["k"], stride=1, padding=op["k"]//2)
    m = Net(ops)
    return count_params_strict(m), ops

for head_ch in [96, 100, 104, 108, 112]:
    for fuse_ch in [112, 116, 120]:
        for extra in [False, True]:
            cnt, _ = test_config(head_ch, fuse_ch, use_sppf=True, extra_refine=extra)
            if cnt <= 200_000 and cnt >= 195_000:
                print(f"head_ch={head_ch:<3} | fuse_ch={fuse_ch:<3} | extra_refine={str(extra):<5} -> {cnt:,} params (Margin: {200000 - cnt} params)")
