import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

MAX_ALLOWED_PARAMS = 200000
MAX_ALLOWED_MODEL_SIZE_KB = 1500  # 1.5 MB
NUM_CLASSES = 7  # 7 classes: bottle, cup, phone, book, mouse, box, toy_car

def build_mobilenet_bbox_model(input_shape=(160, 160, 3), num_classes=NUM_CLASSES, alpha=0.35):
    """
    Ultra-lightweight MobileNetV2 with Bi-feature FPN Head for ESP32-S3.
    Input: (H, W, 3) - typically 160x160, 144x144, 128x128, or 96x96.
    Output:
      - 'cls_output': (Grid, Grid, num_classes + 1) with Softmax (1 background + 7 classes)
      - 'box_output': (Grid, Grid, 4) with Sigmoid (cx_offset, cy_offset, w, h)
    Parameters: ~50,000 - 90,000 (Strictly < 200,000).
    TFLite Micro 100% compatible.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    
    # Stride 8 feature map (e.g. 20x20 for 160x160 input)
    feat_stride8 = base.get_layer("block_5_add").output
    # Stride 16 feature map (e.g. 10x10 for 160x160 input)
    feat_stride16 = base.get_layer("block_12_add").output
    
    # Multi-scale feature fusion (FPN via nearest neighbor upsampling)
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    # Lightweight Depthwise-Separable Detection Head
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="head_dw")(fused)
    x = layers.BatchNormalization(name="head_dw_bn")(x)
    x = layers.ReLU(6.0, name="head_dw_relu")(x)
    
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="head_pw")(x)
    x = layers.BatchNormalization(name="head_pw_bn")(x)
    x = layers.ReLU(6.0, name="head_pw_relu")(x)
    
    # Dual Head: Classification (8 channels) + Regression (4 channels)
    cls_out = layers.Conv2D(num_classes + 1, 1, padding="same", activation="softmax", name="cls_output")(x)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out},
                         name="MobileNetV2_BBox_Qualifiers")
    
    total_params = model.count_params()
    print(f"[*] MobileNetV2 BBox total parameters: {total_params:,} (Limit: {MAX_ALLOWED_PARAMS:,})")
    if total_params > MAX_ALLOWED_PARAMS:
        raise ValueError(f"FATAL: Model parameters ({total_params}) exceed competition limit ({MAX_ALLOWED_PARAMS})!")
        
    return model

if __name__ == "__main__":
    m = build_mobilenet_bbox_model(input_shape=(160, 160, 3), num_classes=7, alpha=0.35)
    m.summary()
