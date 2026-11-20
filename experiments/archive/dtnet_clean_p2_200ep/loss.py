#!/usr/bin/env python3
"""
loss.py
================================================================================
Anti-False-Positive Loss (AFPL) for Clean-Phase 200-Epoch Fine-Tuning
--------------------------------------------------------------------------------
Key Formulations to Push from 98.4% to 99.2%+:
1. Scale-Adaptive EIoU Box Loss:
   - Penalizes center point distance and aspect ratio deviations.
   - Boosts gradient magnitude by up to 2.2x for tiny objects (mouse, toy_car).
2. Hard-Class Priority Multipliers:
   - toy_car: 1.40x | book: 1.35x | mouse: 1.25x | phone: 1.05x | cup: 1.05x | bottle: 1.00x | box: 0.95x
3. Cross-Class Mutual Exclusion Penalty:
   - At every positive anchor cell for class c*, the other 6 classes are strictly 0.
   - Enforces a high-gradient linear penalty (w_mutual = 3.0, beta = 1.0) on any ghost class
     predicting score > 0.05 at that location, eradicating cross-class false positives.
4. Quality Focal Target:
   - Continuous IoU-gated target for smooth precision-recall ranking.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtnet_clean_p2_200ep.dtnet_max import ORIG_W, ORIG_H, STRIDE, NUM_CLASSES

# Hard-Class Priority Weights vector (classes 0..6)
# 0: bottle, 1: cup, 2: phone, 3: book, 4: mouse, 5: box, 6: toy_car
CLASS_WEIGHTS = torch.tensor([1.00, 1.05, 1.05, 1.35, 1.25, 0.95, 1.40], dtype=torch.float32)


def decode_ltrb(box_raw, stride=STRIDE, max_dist=20.0):
    """
    Decodes raw network predictions into absolute xyxy bounding boxes.
    box_raw: (B, 4, gh, gw)
    Returns: (B, 4, gh, gw) with [x1, y1, x2, y2]
    """
    B, _, gh, gw = box_raw.shape
    d = torch.sigmoid(box_raw) * (max_dist * stride)
    gx = ((torch.arange(gw, device=box_raw.device, dtype=box_raw.dtype) + 0.5) * stride).view(1, gw)
    gy = ((torch.arange(gh, device=box_raw.device, dtype=box_raw.dtype) + 0.5) * stride).view(gh, 1)
    return torch.stack([gx - d[:, 0], gy - d[:, 1], gx + d[:, 2], gy + d[:, 3]], 1)


def bbox_eiou(b1, b2, eps=1e-7):
    """
    Calculates Efficient IoU (EIoU) and standard IoU between two sets of bboxes (N, 4).
    """
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


def det_loss_clean_afpl(cls_logit, box_raw, cls_idx, box_t,
                        lam=1.0, w_cls=1.0, w_box=5.5,
                        qfl_beta_pos=2.0, qfl_beta_neg=1.2,
                        w_neg=1.5, w_mutual=3.0,
                        iou_gate=0.48, stride=STRIDE, max_dist=20.0):
    """
    Anti-False-Positive Loss with Cross-Class Mutual Exclusion & Scale-Adaptive EIoU.
    """
    device = cls_logit.device
    cls_weights = CLASS_WEIGHTS.to(device)

    cls_logit, box_raw = cls_logit.float(), box_raw.float()
    B, C, gh, gw = cls_logit.shape
    pos = cls_idx >= 0
    n_pos = pos.sum().clamp(min=1).float()
    tgt = torch.zeros_like(cls_logit)

    if pos.any():
        pred_xyxy = decode_ltrb(box_raw, stride, max_dist)
        pb = pred_xyxy.permute(0, 2, 3, 1)[pos]
        gb = box_t.permute(0, 2, 3, 1)[pos]
        pos_classes = cls_idx[pos]

        eiou, iou = bbox_eiou(pb, gb)

        # Scale adaptive weighting for small objects (mouse, toy_car)
        gw_box = (gb[:, 2] - gb[:, 0]).clamp(min=1.0)
        gh_box = (gb[:, 3] - gb[:, 1]).clamp(min=1.0)
        area_norm = (gw_box * gh_box) / (ORIG_W * ORIG_H)
        w_scale = 1.0 + 1.2 * (1.0 - torch.sqrt(area_norm)).clamp(0, 1.0)

        # Hard-class weighting on box loss
        w_cls_box = cls_weights[pos_classes]
        loss_box = ((1.0 - eiou) * w_scale * w_cls_box).sum() / n_pos

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

    # 1. Positive loss with class priority weighting
    class_indices = torch.arange(C, device=device).view(1, C, 1, 1).expand(B, C, gh, gw)
    weight_map = cls_weights[class_indices]
    loss_cls_pos = weight_map * ce * (tgt - p).abs().pow(qfl_beta_pos)

    # 2. Identify Positive Anchor locations across batch
    # pos is (B, gh, gw). Expand to (B, C, gh, gw)
    pos_expanded = pos.unsqueeze(1).expand(-1, C, -1, -1)
    is_mutual_exclusion = pos_expanded & (~is_pos_target)

    # 3. Cross-Class Mutual Exclusion Loss (Linear penalty, w_mutual = 3.0)
    loss_cls_mutual = w_mutual * ce * p.pow(1.0)

    # 4. Standard Background Loss
    loss_cls_bg = w_neg * ce * p.pow(qfl_beta_neg)

    # Combine negative loss: mutual exclusion gets high penalty, pure bg gets standard penalty
    loss_cls_neg = torch.where(is_mutual_exclusion, loss_cls_mutual, loss_cls_bg)

    loss_cls = torch.where(is_pos_target, loss_cls_pos, loss_cls_neg).sum() / n_pos
    total = w_cls * loss_cls + w_box * loss_box
    return total, loss_cls.detach(), loss_box.detach()
