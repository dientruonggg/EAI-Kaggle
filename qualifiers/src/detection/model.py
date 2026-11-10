import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

MAX_ALLOWED_PARAMS = 200000
MAX_ALLOWED_MODEL_SIZE_KB = 1500  # 1.5 MB
NUM_CLASSES = 7  # 7 classes: bottle, cup, phone, book, mouse, box, toy_car

def build_hierarchical_12layer_detector(input_shape=(160, 160, 3), num_classes=NUM_CLASSES, **kwargs):
    """
    MicroEdgeCNN: Dedicated 12-Layer Hierarchical Detector for ESP32-S3 Edge AI.
    
    Structure:
      - Stage 1: Cạnh - Góc (Layers 1..3: Low-level Edges & Corners, Stride 2 & 1)
      - Stage 2: Hình dạng (Layers 4..7: Geometric Shapes & Object Parts, Stride 2)
      - Stage 3: Ngữ nghĩa (Layers 8..12: High-level Semantics & FPN Lateral Fusion)
      - Decoupled Detection Head:
          1. conf_output: (Grid, Grid, 1) with Sigmoid (Objectness)
          2. cls_output:  (Grid, Grid, 7) with Softmax (7-Class Distribution)
          3. box_output:  (Grid, Grid, 4) with Sigmoid (cx_offset, cy_offset, w, h)
          
    Parameters: ~50,228 (Strictly < 200,000 budget).
    100% TensorFlow Lite Micro compatible.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # =========================================================================
    # STAGE 1: CẠNH & GÓC (Low-Level Edges, Textures & Corner Gradients)
    # =========================================================================
    # Layer 1: Stem Convolution (3x3, stride=2) -> 80x80x24
    x = layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False, name="stg1_l1_stem")(inputs)
    x = layers.BatchNormalization(name="stg1_l1_bn")(x)
    x = layers.ReLU(6.0, name="stg1_l1_relu")(x)
    
    # Layer 2: Depthwise Separable Edge Filter -> 80x80x24
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg1_l2_dw")(x)
    x = layers.BatchNormalization(name="stg1_l2_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg1_l2_dw_relu")(x)
    x = layers.Conv2D(24, 1, padding="same", use_bias=False, name="stg1_l2_pw")(x)
    x = layers.BatchNormalization(name="stg1_l2_pw_bn")(x)
    
    # Layer 3: Corner & Transition Downsample (stride=2) -> 40x40x32
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="stg1_l3_dw")(x)
    x = layers.BatchNormalization(name="stg1_l3_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg1_l3_dw_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="stg1_l3_pw")(x)
    x = layers.BatchNormalization(name="stg1_l3_pw_bn")(x)
    x = layers.ReLU(6.0, name="stg1_l3_pw_relu")(x)
    
    # =========================================================================
    # STAGE 2: HÌNH DẠNG & KHỐI HÌNH HỌC (Mid-Level Geometric Shapes & Parts)
    # =========================================================================
    # Layer 4: Residual Shape Refinement -> 40x40x32
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg2_l4_dw")(x)
    x = layers.BatchNormalization(name="stg2_l4_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg2_l4_dw_relu")(x)
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="stg2_l4_pw")(x)
    x = layers.BatchNormalization(name="stg2_l4_pw_bn")(x)
    x = layers.Add(name="stg2_l4_add")([res, x])
    
    # Layer 5: Downsample to Grid (stride=2) -> 20x20x48
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="stg2_l5_dw")(x)
    x = layers.BatchNormalization(name="stg2_l5_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg2_l5_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="stg2_l5_pw")(x)
    x = layers.BatchNormalization(name="stg2_l5_pw_bn")(x)
    x = layers.ReLU(6.0, name="stg2_l5_pw_relu")(x)
    
    # Layer 6: In-grid Part Association -> 20x20x48
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg2_l6_dw")(x)
    x = layers.BatchNormalization(name="stg2_l6_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg2_l6_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="stg2_l6_pw")(x)
    x = layers.BatchNormalization(name="stg2_l6_pw_bn")(x)
    x = layers.Add(name="stg2_l6_add")([res, x])
    
    # Layer 7: Final Shape Representation Layer -> 20x20x48
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg2_l7_dw")(x)
    x = layers.BatchNormalization(name="stg2_l7_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg2_l7_dw_relu")(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False, name="stg2_l7_pw")(x)
    x = layers.BatchNormalization(name="stg2_l7_pw_bn")(x)
    shape_feat = layers.Add(name="stg2_l7_add")([res, x])  # 20x20x48
    
    # =========================================================================
    # STAGE 3: NGỮ NGHĨA VẬT THỂ (High-Level Semantic Representation & FPN)
    # =========================================================================
    # Layer 8: Semantic Downsample (stride=2) -> 10x10x64
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False, name="stg3_l8_dw")(shape_feat)
    x = layers.BatchNormalization(name="stg3_l8_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l8_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="stg3_l8_pw")(x)
    x = layers.BatchNormalization(name="stg3_l8_pw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l8_pw_relu")(x)
    
    # Layer 9: Semantic Context Refinement -> 10x10x64
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg3_l9_dw")(x)
    x = layers.BatchNormalization(name="stg3_l9_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l9_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False, name="stg3_l9_pw")(x)
    x = layers.BatchNormalization(name="stg3_l9_pw_bn")(x)
    x = layers.Add(name="stg3_l9_add")([res, x])
    
    # Layer 10: High-Level Feature Expansion -> 10x10x96
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg3_l10_dw")(x)
    x = layers.BatchNormalization(name="stg3_l10_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l10_dw_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="stg3_l10_pw")(x)
    x = layers.BatchNormalization(name="stg3_l10_pw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l10_pw_relu")(x)
    
    # Layer 11: Deep Semantic Association -> 10x10x96
    res = x
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="stg3_l11_dw")(x)
    x = layers.BatchNormalization(name="stg3_l11_dw_bn")(x)
    x = layers.ReLU(6.0, name="stg3_l11_dw_relu")(x)
    x = layers.Conv2D(96, 1, padding="same", use_bias=False, name="stg3_l11_pw")(x)
    x = layers.BatchNormalization(name="stg3_l11_pw_bn")(x)
    sem_feat = layers.Add(name="stg3_l11_add")([res, x])  # 10x10x96
    
    # Layer 12: FPN Semantic Projection & Upsampling back to 20x20
    sem_proj = layers.Conv2D(48, 1, padding="same", use_bias=False, name="stg3_l12_proj")(sem_feat)
    sem_up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="stg3_l12_up")(sem_proj)  # 20x20x48
    
    # =========================================================================
    # FUSED FEATURE MAP (Kết hợp Cạnh-Góc + Hình dạng + Ngữ nghĩa)
    # =========================================================================
    fused_feat = layers.Concatenate(name="fused_feature_map")([shape_feat, sem_up])  # 20x20x96
    fused_feat = layers.Conv2D(48, 1, padding="same", use_bias=False, name="fused_compress")(fused_feat)
    fused_feat = layers.BatchNormalization(name="fused_bn")(fused_feat)
    fused_feat = layers.ReLU(6.0, name="fused_relu")(fused_feat)  # 20x20x48
    
    # =========================================================================
    # DECOUPLED DETECTION HEAD (Conf, Class, BBox)
    # =========================================================================
    # 1. Confidence / Objectness (sigmoid: 0..1)
    conf_out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conf_output")(fused_feat)
    
    # 2. 7-Class Classification (softmax over 7 foreground classes)
    cls_out = layers.Conv2D(num_classes, 1, padding="same", activation="softmax", name="cls_output")(fused_feat)
    
    # 3. Bounding Box Offset & Size (sigmoid: 0..1)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(fused_feat)
    
    model = models.Model(
        inputs=inputs,
        outputs={"conf_output": conf_out, "cls_output": cls_out, "box_output": box_out},
        name="MicroEdgeCNN_12Layer"
    )
    
    total_params = model.count_params()
    print(f"[*] MicroEdgeCNN (12-Layer) total parameters: {total_params:,} (Limit: {MAX_ALLOWED_PARAMS:,})")
    if total_params > MAX_ALLOWED_PARAMS:
        raise ValueError(f"FATAL: Model parameters ({total_params}) exceed competition limit ({MAX_ALLOWED_PARAMS})!")
        
    return model

# Legacy alias
build_mobilenet_bbox_model = build_hierarchical_12layer_detector

if __name__ == "__main__":
    m = build_hierarchical_12layer_detector(input_shape=(160, 160, 3), num_classes=7)
    m.summary()
