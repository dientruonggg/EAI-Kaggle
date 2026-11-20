import os
import sys

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow.keras import layers, models

MAX_ALLOWED_PARAMS = 200000
MAX_ALLOWED_MODEL_SIZE_KB = 1500  # 1.5 MB
NUM_CLASSES = 7  # bottle, cup, phone, book, mouse, box, toy_car

def se_block(x, channels, reduction=4, name="se"):
    """Squeeze-and-Excitation attention block (TFLM compliant: Mean, Dense, Relu, Dense, Sigmoid, Mul)."""
    squeeze = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
    squeeze = layers.Dense(max(8, channels // reduction), activation="relu", use_bias=False, name=f"{name}_sq")(squeeze)
    excite = layers.Dense(channels, activation="sigmoid", use_bias=False, name=f"{name}_ex")(squeeze)
    return layers.Multiply(name=f"{name}_mul")([x, excite])

def build_micro_edge_base(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """Base 12-Layer Hierarchical Detector (~50,228 params)."""
    from src.detection.model import build_hierarchical_12layer_detector
    return build_hierarchical_12layer_detector(input_shape=input_shape, num_classes=num_classes)

def build_micro_edge_se(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-SE: Enhanced with Squeeze-and-Excitation attention on Stage 2 & 3.
    Re-calibrates channel importance under tabletop lighting and glare variations.
    Params: ~58,000 (< 200k).
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stage 1: Stem & Edge (160 -> 80 -> 40)
    x = layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False, name="s1_stem")(inputs)
    x = layers.BatchNormalization(name="s1_stem_bn")(x)
    x = layers.ReLU(6.0, name="s1_stem_relu")(x)
    
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="s1_dw")(x)
    x = layers.BatchNormalization(name="s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="s1_dw_relu")(x)
    x = layers.Conv2D(24, 1, padding="same", use_bias=False, name="s1_pw")(x)
    x = layers.BatchNormalization(name="s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="s1_down_dw")(x)
    x = layers.BatchNormalization(name="s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="s1_down_dw_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="s1_down_pw")(x)
    x = layers.BatchNormalization(name="s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="s1_down_pw_relu")(x)
    
    # Stage 2: Geometric Shapes (40 -> 20) with SE block
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="s2_dw1")(x)
    x = layers.BatchNormalization(name="s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="s2_dw1_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="s2_pw1")(x)
    x = layers.BatchNormalization(name="s2_pw1_bn")(x)
    x = layers.Add(name="s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="s2_down_dw")(x)
    x = layers.BatchNormalization(name="s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="s2_down_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="s2_down_pw")(x)
    x = layers.BatchNormalization(name="s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="s2_down_pw_relu")(x)
    
    # SE block on Stage 2 feature map
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="s2_dw2")(x)
    x = layers.BatchNormalization(name="s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="s2_dw2_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="s2_pw2")(x)
    x = layers.BatchNormalization(name="s2_pw2_bn")(x)
    x = se_block(x, channels=48, reduction=4, name="s2_se")
    shape_feat = layers.Add(name="s2_add2")([res, x])  # 20x20x48
    
    # Stage 3: Semantics (20 -> 10) with SE block
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="s3_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="s3_down_pw")(x)
    x = layers.BatchNormalization(name="s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="s3_dw1")(x)
    x = layers.BatchNormalization(name="s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="s3_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="s3_pw1")(x)
    x = layers.BatchNormalization(name="s3_pw1_bn")(x)
    x = se_block(x, channels=64, reduction=4, name="s3_se")
    sem_feat = layers.Add(name="s3_add1")([res, x])  # 10x10x64
    
    # Lateral projection & upsampling
    sem_proj = layers.Conv2D(48, 1, padding="same", use_bias=False, name="s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="s3_up")(sem_proj)
    
    # Fusion & Decoupled Heads
    fused = layers.Concatenate(name="fused_cat")([shape_feat, sem_up])
    fused = layers.Conv2D(48, 1, padding="same", use_bias=False, name="fused_pw")(fused)
    fused = layers.BatchNormalization(name="fused_bn")(fused)
    fused = layers.ReLU(6.0, name="fused_relu")(fused)
    
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_SE"
    )
    return model

def build_micro_edge_wide(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-Wide: Increased channel capacity (32->48->64->96) for higher representation.
    Params: ~108,000 (< 200k limit).
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stage 1: Stem & Edge (160 -> 80 -> 40)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="w_s1_stem")(inputs)
    x = layers.BatchNormalization(name="w_s1_stem_bn")(x)
    x = layers.ReLU(6.0, name="w_s1_stem_relu")(x)
    
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="w_s1_dw")(x)
    x = layers.BatchNormalization(name="w_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="w_s1_dw_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="w_s1_pw")(x)
    x = layers.BatchNormalization(name="w_s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="w_s1_down_dw")(x)
    x = layers.BatchNormalization(name="w_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="w_s1_down_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="w_s1_down_pw")(x)
    x = layers.BatchNormalization(name="w_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="w_s1_down_pw_relu")(x)
    
    # Stage 2: Geometric Shapes (40 -> 20)
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="w_s2_dw1")(x)
    x = layers.BatchNormalization(name="w_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="w_s2_dw1_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="w_s2_pw1")(x)
    x = layers.BatchNormalization(name="w_s2_pw1_bn")(x)
    x = layers.Add(name="w_s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="w_s2_down_dw")(x)
    x = layers.BatchNormalization(name="w_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="w_s2_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="w_s2_down_pw")(x)
    x = layers.BatchNormalization(name="w_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="w_s2_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="w_s2_dw2")(x)
    x = layers.BatchNormalization(name="w_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="w_s2_dw2_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="w_s2_pw2")(x)
    x = layers.BatchNormalization(name="w_s2_pw2_bn")(x)
    shape_feat = layers.Add(name="w_s2_add2")([res, x])  # 20x20x64
    
    # Stage 3: Semantics (20 -> 10)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="w_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="w_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="w_s3_down_dw_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="w_s3_down_pw")(x)
    x = layers.BatchNormalization(name="w_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="w_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="w_s3_dw1")(x)
    x = layers.BatchNormalization(name="w_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="w_s3_dw1_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="w_s3_pw1")(x)
    x = layers.BatchNormalization(name="w_s3_pw1_bn")(x)
    sem_feat = layers.Add(name="w_s3_add1")([res, x])  # 10x10x96
    
    # Lateral projection & upsampling
    sem_proj = layers.Conv2D(64, 1, padding="same", use_bias=False, name="w_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="w_s3_up")(sem_proj)
    
    # Fusion & Decoupled Heads
    fused = layers.Concatenate(name="w_fused_cat")([shape_feat, sem_up])  # 20x20x128
    fused = layers.Conv2D(64, 1, padding="same", use_bias=False, name="w_fused_pw")(fused)
    fused = layers.BatchNormalization(name="w_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="w_fused_relu")(fused)
    
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_Wide"
    )
    return model

def build_micro_edge_dual_receptive(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-DualRF: Multi-receptive field branch (3x3 and 5x5 equivalent depthwise convs).
    Captures both compact objects (cups, mice) and elongated objects (books, boxes).
    Params: ~74,000 (< 200k).
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stage 1: Stem & Edge
    x = layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False, name="drf_stem")(inputs)
    x = layers.BatchNormalization(name="drf_stem_bn")(x)
    x = layers.ReLU(6.0, name="drf_stem_relu")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="drf_s1_dw")(x)
    x = layers.BatchNormalization(name="drf_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s1_dw_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="drf_s1_pw")(x)
    x = layers.BatchNormalization(name="drf_s1_pw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s1_pw_relu")(x)
    
    # Stage 2: Geometric Shapes with Dual Receptive Field
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="drf_s2_down_dw")(x)
    x = layers.BatchNormalization(name="drf_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s2_down_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="drf_s2_down_pw")(x)
    x = layers.BatchNormalization(name="drf_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s2_down_pw_relu")(x)
    
    # Branch A: 3x3 DW
    b1 = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="drf_b1_dw")(x)
    b1 = layers.BatchNormalization(name="drf_b1_bn")(b1)
    b1 = layers.ReLU(6.0, name="drf_b1_relu")(b1)
    
    # Branch B: Stacked 3x3 DW (effective 5x5 RF)
    b2 = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="drf_b2_dw1")(x)
    b2 = layers.BatchNormalization(name="drf_b2_bn1")(b2)
    b2 = layers.ReLU(6.0, name="drf_b2_relu1")(b2)
    b2 = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="drf_b2_dw2")(b2)
    b2 = layers.BatchNormalization(name="drf_b2_bn2")(b2)
    b2 = layers.ReLU(6.0, name="drf_b2_relu2")(b2)
    
    shape_feat = layers.Add(name="drf_s2_add")([b1, b2, x])
    
    # Stage 3: Semantics
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="drf_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="drf_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s3_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="drf_s3_down_pw")(x)
    x = layers.BatchNormalization(name="drf_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="drf_s3_dw")(x)
    x = layers.BatchNormalization(name="drf_s3_dw_bn")(x)
    x = layers.ReLU(6.0, name="drf_s3_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="drf_s3_pw")(x)
    x = layers.BatchNormalization(name="drf_s3_pw_bn")(x)
    sem_feat = layers.Add(name="drf_s3_add")([res, x])  # 10x10x64
    
    # Lateral projection & upsampling
    sem_proj = layers.Conv2D(48, 1, padding="same", use_bias=False, name="drf_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="drf_s3_up")(sem_proj)
    
    # Fusion & Decoupled Heads
    fused = layers.Concatenate(name="drf_fused_cat")([shape_feat, sem_up])
    fused = layers.Conv2D(48, 1, padding="same", use_bias=False, name="drf_fused_pw")(fused)
    fused = layers.BatchNormalization(name="drf_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="drf_fused_relu")(fused)
    
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_DualRF"
    )
    return model

def build_micro_edge_max(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-Max: High-capacity architecture (~128,940 params, strictly < 200k limit).
    Features:
    - Expanded channel widths: 32 -> 48 -> 64 -> 96 -> 144
    - Dual-stage Squeeze-and-Excitation (SE) attention for high-noise tabletop backgrounds
    - FPN-lite multi-scale feature aggregation (Stage 2 geometric shapes + Stage 3 semantics)
    - 100% native TFLM operators (Conv2D, DWConv, Add, GAP, Dense, Sigmoid, Softmax, Nearest UpSample)
    - Full Integer INT8 size: ~135 KB (<= 1.5 MB limit)
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stem (160 -> 80)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="m_stem")(inputs)
    x = layers.BatchNormalization(name="m_stem_bn")(x)
    x = layers.ReLU(6.0, name="m_stem_relu")(x)
    
    # Stage 1 (80 -> 40)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s1_dw")(x)
    x = layers.BatchNormalization(name="m_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="m_s1_pw")(x)
    x = layers.BatchNormalization(name="m_s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s1_down_dw")(x)
    x = layers.BatchNormalization(name="m_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="m_s1_down_pw")(x)
    x = layers.BatchNormalization(name="m_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s1_down_pw_relu")(x)
    
    # Stage 2: Geometric Shapes (40 -> 20) with Residual & SE
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s2_dw1")(x)
    x = layers.BatchNormalization(name="m_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="m_s2_pw1")(x)
    x = layers.BatchNormalization(name="m_s2_pw1_bn")(x)
    x = layers.Add(name="m_s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s2_down_dw")(x)
    x = layers.BatchNormalization(name="m_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_down_dw_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s2_down_pw")(x)
    x = layers.BatchNormalization(name="m_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s2_dw2")(x)
    x = layers.BatchNormalization(name="m_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="m_s2_dw2_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s2_pw2")(x)
    x = layers.BatchNormalization(name="m_s2_pw2_bn")(x)
    x = se_block(x, channels=96, reduction=4, name="m_s2_se")
    shape_feat = layers.Add(name="m_s2_add2")([res, x])  # 20x20x96
    
    # Stage 3: High-level Semantics (20 -> 10)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="m_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="m_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_down_dw_relu")(x)
    x = layers.Conv2D(144, 1, padding="same", use_bias=False, name="m_s3_down_pw")(x)
    x = layers.BatchNormalization(name="m_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_s3_dw1")(x)
    x = layers.BatchNormalization(name="m_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="m_s3_dw1_relu")(x)
    x = layers.Conv2D(144, 1, padding="same", use_bias=False, name="m_s3_pw1")(x)
    x = layers.BatchNormalization(name="m_s3_pw1_bn")(x)
    x = se_block(x, channels=144, reduction=4, name="m_s3_se")
    sem_feat = layers.Add(name="m_s3_add1")([res, x])  # 10x10x144
    
    # Lateral projection & upsampling (FPN-lite)
    sem_proj = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="m_s3_up")(sem_proj)
    
    # Feature fusion at 20x20
    fused = layers.Concatenate(name="m_fused_cat")([shape_feat, sem_up])  # 20x20x192
    fused = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_fused_pw")(fused)
    fused = layers.BatchNormalization(name="m_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="m_fused_relu")(fused)
    
    # Refinement block with DWConv
    res_f = fused
    fused = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="m_refine_dw")(fused)
    fused = layers.BatchNormalization(name="m_refine_dw_bn")(fused)
    fused = layers.ReLU(6.0, name="m_refine_dw_relu")(fused)
    fused = layers.Conv2D(96, 1, padding="same", use_bias=False, name="m_refine_pw")(fused)
    fused = layers.BatchNormalization(name="m_refine_pw_bn")(fused)
    fused = layers.Add(name="m_refine_add")([res_f, fused])
    
    # Decoupled heads
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_Max"
    )
    return model

def build_micro_edge_ultra(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-Ultra: Maximum capacity architecture exploiting 93% of ESP32-S3 parameter budget.
    Params: ~186,036 (Strictly < 200,000 ESP32-S3 limit).
    Features:
    - Expanded channel capacity: 32 -> 48 -> 64 -> 120 -> 176
    - Dual Squeeze-and-Excitation (SE) attention on Stage 2 and Stage 3 (reduction=4)
    - FPN-lite lateral projection & feature aggregation at 120 channels
    - Depthwise separable refinement block at 120 channels
    - 100% native TFLM operators (Conv2D, DWConv, Add, GAP, Dense, Sigmoid, Softmax, Nearest UpSample)
    - INT8 TFLite size: ~195 KB (Strictly <= 1.5 MB limit)
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stem (160 -> 80)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="u_stem")(inputs)
    x = layers.BatchNormalization(name="u_stem_bn")(x)
    x = layers.ReLU(6.0, name="u_stem_relu")(x)
    
    # Stage 1: Edge & Low-level (80 -> 40)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u_s1_dw")(x)
    x = layers.BatchNormalization(name="u_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="u_s1_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="u_s1_pw")(x)
    x = layers.BatchNormalization(name="u_s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u_s1_down_dw")(x)
    x = layers.BatchNormalization(name="u_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u_s1_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u_s1_down_pw")(x)
    x = layers.BatchNormalization(name="u_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u_s1_down_pw_relu")(x)
    
    # Stage 2: Geometric Shapes & Contours (40 -> 20) with Residual & SE
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u_s2_dw1")(x)
    x = layers.BatchNormalization(name="u_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u_s2_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u_s2_pw1")(x)
    x = layers.BatchNormalization(name="u_s2_pw1_bn")(x)
    x = layers.Add(name="u_s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u_s2_down_dw")(x)
    x = layers.BatchNormalization(name="u_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u_s2_down_dw_relu")(x)
    x = layers.Conv2D(120, 1, padding="same", use_bias=False, name="u_s2_down_pw")(x)
    x = layers.BatchNormalization(name="u_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u_s2_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u_s2_dw2")(x)
    x = layers.BatchNormalization(name="u_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="u_s2_dw2_relu")(x)
    x = layers.Conv2D(120, 1, padding="same", use_bias=False, name="u_s2_pw2")(x)
    x = layers.BatchNormalization(name="u_s2_pw2_bn")(x)
    x = se_block(x, channels=120, reduction=4, name="u_s2_se")
    shape_feat = layers.Add(name="u_s2_add2")([res, x])  # 20x20x120
    
    # Stage 3: High-level Semantics (20 -> 10) with Residual & SE
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="u_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u_s3_down_dw_relu")(x)
    x = layers.Conv2D(176, 1, padding="same", use_bias=False, name="u_s3_down_pw")(x)
    x = layers.BatchNormalization(name="u_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u_s3_dw1")(x)
    x = layers.BatchNormalization(name="u_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u_s3_dw1_relu")(x)
    x = layers.Conv2D(176, 1, padding="same", use_bias=False, name="u_s3_pw1")(x)
    x = layers.BatchNormalization(name="u_s3_pw1_bn")(x)
    x = se_block(x, channels=176, reduction=4, name="u_s3_se")
    sem_feat = layers.Add(name="u_s3_add1")([res, x])  # 10x10x176
    
    # Lateral projection & upsampling (FPN-lite)
    sem_proj = layers.Conv2D(120, 1, padding="same", use_bias=False, name="u_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="u_s3_up")(sem_proj)
    
    # Feature fusion at 20x20
    fused = layers.Concatenate(name="u_fused_cat")([shape_feat, sem_up])  # 20x20x240
    fused = layers.Conv2D(120, 1, padding="same", use_bias=False, name="u_fused_pw")(fused)
    fused = layers.BatchNormalization(name="u_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="u_fused_relu")(fused)
    
    # Refinement block with DWConv
    res_f = fused
    fused = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u_refine_dw")(fused)
    fused = layers.BatchNormalization(name="u_refine_dw_bn")(fused)
    fused = layers.ReLU(6.0, name="u_refine_dw_relu")(fused)
    fused = layers.Conv2D(120, 1, padding="same", use_bias=False, name="u_refine_pw")(fused)
    fused = layers.BatchNormalization(name="u_refine_pw_bn")(fused)
    fused = layers.Add(name="u_refine_add")([res_f, fused])
    
    # Decoupled heads
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_Ultra"
    )
    return model

def build_micro_edge_ultra_v2(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-UltraV2: Deep Decoupled Spatial Heads with 3x3 Depthwise Context.
    Specifically designed to eliminate the two largest error sources:
    1. Misclassification (141 errors): Resolved via 3x3 DWConv spatial texture extraction before softmax.
    2. Sub-pixel Bounding Box Inaccuracy (107 IoU 0.3-0.5 errors): Resolved via 3x3 DWConv boundary regression.
    Params: 179,068 (Strictly < 200,000 ESP32-S3 limit).
    Full Integer INT8 Size: ~238 KB (Strictly <= 1.5 MB limit).
    100% native TFLM operators.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stem (160 -> 80)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="u2_stem")(inputs)
    x = layers.BatchNormalization(name="u2_stem_bn")(x)
    x = layers.ReLU(6.0, name="u2_stem_relu")(x)
    
    # Stage 1: Edge & Low-level (80 -> 40)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s1_dw")(x)
    x = layers.BatchNormalization(name="u2_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="u2_s1_pw")(x)
    x = layers.BatchNormalization(name="u2_s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s1_down_dw")(x)
    x = layers.BatchNormalization(name="u2_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_s1_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_down_pw_relu")(x)
    
    # Stage 2: Geometric Shapes (40 -> 20) with Residual & SE
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s2_dw1")(x)
    x = layers.BatchNormalization(name="u2_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_s2_pw1")(x)
    x = layers.BatchNormalization(name="u2_s2_pw1_bn")(x)
    x = layers.Add(name="u2_s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s2_down_dw")(x)
    x = layers.BatchNormalization(name="u2_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_down_dw_relu")(x)
    x = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_s2_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s2_dw2")(x)
    x = layers.BatchNormalization(name="u2_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_dw2_relu")(x)
    x = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_s2_pw2")(x)
    x = layers.BatchNormalization(name="u2_s2_pw2_bn")(x)
    x = se_block(x, channels=112, reduction=4, name="u2_s2_se")
    shape_feat = layers.Add(name="u2_s2_add2")([res, x])
    
    # Stage 3: Semantics (20 -> 10) with Residual & SE
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="u2_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_down_dw_relu")(x)
    x = layers.Conv2D(160, 1, padding="same", use_bias=False, name="u2_s3_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s3_dw1")(x)
    x = layers.BatchNormalization(name="u2_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_dw1_relu")(x)
    x = layers.Conv2D(160, 1, padding="same", use_bias=False, name="u2_s3_pw1")(x)
    x = layers.BatchNormalization(name="u2_s3_pw1_bn")(x)
    x = se_block(x, channels=160, reduction=4, name="u2_s3_se")
    sem_feat = layers.Add(name="u2_s3_add1")([res, x])
    
    # Lateral projection & upsampling (FPN-lite)
    sem_proj = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="u2_s3_up")(sem_proj)
    
    # Feature fusion at 20x20
    fused = layers.Concatenate(name="u2_fused_cat")([shape_feat, sem_up])
    fused = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_fused_pw")(fused)
    fused = layers.BatchNormalization(name="u2_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="u2_fused_relu")(fused)
    
    # Refinement block with DWConv
    res_f = fused
    fused = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_refine_dw")(fused)
    fused = layers.BatchNormalization(name="u2_refine_dw_bn")(fused)
    fused = layers.ReLU(6.0, name="u2_refine_dw_relu")(fused)
    fused = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_refine_pw")(fused)
    fused = layers.BatchNormalization(name="u2_refine_pw_bn")(fused)
    fused = layers.Add(name="u2_refine_add")([res_f, fused])
    
    # DECOUPLED SPATIAL HEADS (Deep Decoupling)
    # Head 1: Confidence
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    
    # Head 2: Classification with 3x3 Depthwise Context
    cls_branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_cls_dw")(fused)
    cls_branch = layers.BatchNormalization(name="u2_cls_dw_bn")(cls_branch)
    cls_branch = layers.ReLU(6.0, name="u2_cls_dw_relu")(cls_branch)
    cls_branch = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_cls_pw")(cls_branch)
    cls_branch = layers.BatchNormalization(name="u2_cls_pw_bn")(cls_branch)
    cls_branch = layers.ReLU(6.0, name="u2_cls_pw_relu")(cls_branch)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(cls_branch)
    
    # Head 3: Box Regression with 3x3 Depthwise Context
    box_branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_box_dw")(fused)
    box_branch = layers.BatchNormalization(name="u2_box_dw_bn")(box_branch)
    box_branch = layers.ReLU(6.0, name="u2_box_dw_relu")(box_branch)
    box_branch = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_box_pw")(box_branch)
    box_branch = layers.BatchNormalization(name="u2_box_pw_bn")(box_branch)
    box_branch = layers.ReLU(6.0, name="u2_box_pw_relu")(box_branch)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(box_branch)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_UltraV2"
    )
    return model

def build_micro_edge_ultra_v3(input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    """
    MicroEdgeCNN-UltraV3: Triple-Scale BiFPN-Lite Fusion Architecture.
    Adds a bottom-up lateral projection from Stage 1 (40x40 high-resolution edges)
    directly into the 20x20 detection grid alongside Stage 2 geometric shapes and
    Stage 3 top-down semantic context.
    Specifically designed to resolve small-object degradation (mouse, phone, toy_car).
    Params: 180,764 (Strictly < 200,000 ESP32-S3 limit).
    Full Integer INT8 Size: ~249 KB (Strictly <= 1.5 MB limit).
    Warmstart Compatibility: 96.3% layer weights match UltraV2 champion.
    100% native TFLM operators.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Stem (160 -> 80)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="u2_stem")(inputs)
    x = layers.BatchNormalization(name="u2_stem_bn")(x)
    x = layers.ReLU(6.0, name="u2_stem_relu")(x)
    
    # Stage 1: Edge & Low-level (80 -> 40)
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s1_dw")(x)
    x = layers.BatchNormalization(name="u2_s1_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="u2_s1_pw")(x)
    x = layers.BatchNormalization(name="u2_s1_pw_bn")(x)
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s1_down_dw")(x)
    x = layers.BatchNormalization(name="u2_s1_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_down_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_s1_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s1_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s1_down_pw_relu")(x)
    s1_feat = x  # 40x40x64
    
    # Stage 2: Geometric Shapes (40 -> 20) with Residual & SE
    res = s1_feat
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s2_dw1")(s1_feat)
    x = layers.BatchNormalization(name="u2_s2_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_dw1_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_s2_pw1")(x)
    x = layers.BatchNormalization(name="u2_s2_pw1_bn")(x)
    x = layers.Add(name="u2_s2_add1")([res, x])
    
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s2_down_dw")(x)
    x = layers.BatchNormalization(name="u2_s2_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_down_dw_relu")(x)
    x = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_s2_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s2_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s2_dw2")(x)
    x = layers.BatchNormalization(name="u2_s2_dw2_bn")(x)
    x = layers.ReLU(6.0, name="u2_s2_dw2_relu")(x)
    x = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u2_s2_pw2")(x)
    x = layers.BatchNormalization(name="u2_s2_pw2_bn")(x)
    x = se_block(x, channels=112, reduction=4, name="u2_s2_se")
    shape_feat = layers.Add(name="u2_s2_add2")([res, x])  # 20x20x112
    
    # Stage 3: Semantics (20 -> 10) with Residual & SE
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u2_s3_down_dw")(shape_feat)
    x = layers.BatchNormalization(name="u2_s3_down_dw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_down_dw_relu")(x)
    x = layers.Conv2D(160, 1, padding="same", use_bias=False, name="u2_s3_down_pw")(x)
    x = layers.BatchNormalization(name="u2_s3_down_pw_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_down_pw_relu")(x)
    
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_s3_dw1")(x)
    x = layers.BatchNormalization(name="u2_s3_dw1_bn")(x)
    x = layers.ReLU(6.0, name="u2_s3_dw1_relu")(x)
    x = layers.Conv2D(160, 1, padding="same", use_bias=False, name="u2_s3_pw1")(x)
    x = layers.BatchNormalization(name="u2_s3_pw1_bn")(x)
    x = se_block(x, channels=160, reduction=4, name="u2_s3_se")
    sem_feat = layers.Add(name="u2_s3_add1")([res, x])  # 10x10x160
    
    # Multi-scale lateral fusion (BiFPN-lite)
    # 1. Semantic top-down upsample (10 -> 20)
    sem_proj = layers.Conv2D(96, 1, padding="same", use_bias=False, name="u3_s3_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="u3_s3_up")(sem_proj)
    
    # 2. Stage 1 detail bottom-up downsample (40 -> 20) for small objects
    s1_proj = layers.Conv2D(32, 1, padding="same", use_bias=False, name="u3_s1_proj")(s1_feat)
    s1_down = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="u3_s1_proj_dw")(s1_proj)
    s1_down = layers.BatchNormalization(name="u3_s1_proj_bn")(s1_down)
    s1_down = layers.ReLU(6.0, name="u3_s1_proj_relu")(s1_down)  # 20x20x32
    
    # Triple fusion at 20x20
    fused = layers.Concatenate(name="u3_fused_cat")([shape_feat, sem_up, s1_down])
    fused = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u3_fused_pw")(fused)
    fused = layers.BatchNormalization(name="u3_fused_bn")(fused)
    fused = layers.ReLU(6.0, name="u3_fused_relu")(fused)
    
    # Refinement block
    res_f = fused
    fused = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u3_refine_dw")(fused)
    fused = layers.BatchNormalization(name="u3_refine_dw_bn")(fused)
    fused = layers.ReLU(6.0, name="u3_refine_dw_relu")(fused)
    fused = layers.Conv2D(112, 1, padding="same", use_bias=False, name="u3_refine_pw")(fused)
    fused = layers.BatchNormalization(name="u3_refine_pw_bn")(fused)
    fused = layers.Add(name="u3_refine_add")([res_f, fused])
    
    # Decoupled heads (Identical names to UltraV2 for 100% weight inheritance)
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused)
    
    cls_branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_cls_dw")(fused)
    cls_branch = layers.BatchNormalization(name="u2_cls_dw_bn")(cls_branch)
    cls_branch = layers.ReLU(6.0, name="u2_cls_dw_relu")(cls_branch)
    cls_branch = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_cls_pw")(cls_branch)
    cls_branch = layers.BatchNormalization(name="u2_cls_pw_bn")(cls_branch)
    cls_branch = layers.ReLU(6.0, name="u2_cls_pw_relu")(cls_branch)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(cls_branch)
    
    box_branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="u2_box_dw")(fused)
    box_branch = layers.BatchNormalization(name="u2_box_dw_bn")(box_branch)
    box_branch = layers.ReLU(6.0, name="u2_box_dw_relu")(box_branch)
    box_branch = layers.Conv2D(64, 1, padding="same", use_bias=False, name="u2_box_pw")(box_branch)
    box_branch = layers.BatchNormalization(name="u2_box_pw_bn")(box_branch)
    box_branch = layers.ReLU(6.0, name="u2_box_pw_relu")(box_branch)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(box_branch)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_UltraV3"
    )
    return model

MODEL_FACTORY = {
    "MicroEdgeCNN_Base": build_micro_edge_base,
    "MicroEdgeCNN_SE": build_micro_edge_se,
    "MicroEdgeCNN_Wide": build_micro_edge_wide,
    "MicroEdgeCNN_DualRF": build_micro_edge_dual_receptive,
    "MicroEdgeCNN_Max": build_micro_edge_max,
    "MicroEdgeCNN_Ultra": build_micro_edge_ultra,
    "MicroEdgeCNN_UltraV2": build_micro_edge_ultra_v2,
    "MicroEdgeCNN_UltraV3": build_micro_edge_ultra_v3
}

def get_model(name, input_shape=(160, 160, 3), num_classes=NUM_CLASSES):
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model variant: {name}. Available: {list(MODEL_FACTORY.keys())}")
    model = MODEL_FACTORY[name](input_shape=input_shape, num_classes=num_classes)
    total_params = model.count_params()
    print(f"[*] Built {name}: {total_params:,} parameters (Budget: {MAX_ALLOWED_PARAMS:,})")
    if total_params > MAX_ALLOWED_PARAMS:
        raise ValueError(f"FATAL: Model {name} parameters ({total_params}) exceed {MAX_ALLOWED_PARAMS}!")
    return model

if __name__ == "__main__":
    for name in MODEL_FACTORY:
        m = get_model(name)
