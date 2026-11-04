import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

MAX_ALLOWED_PARAMS = 200000
MAX_ALLOWED_MODEL_SIZE_KB = 1500  # 1.5 MB

def create_pretrained_micro_fomo(input_shape=(96, 96, 3), num_classes=10):
    """
    Micro-FOMO with ImageNet pretrained MobileNetV2 (alpha=0.35) truncated at block 5.
    Output: 12x12 grid heatmap with 11 channels (1 bg + 10 classes).
    Parameters: ~16,500 (< 200,000).
    TFLite Micro 100% compatible.
    """
    # Load pretrained MobileNetV2 0.35 without top
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.35,
        include_top=False,
        weights="imagenet"
    )
    
    # Truncate at block 5 add (downsampled 8x -> 12x12x16)
    feature_layer = base.get_layer("block_5_add")
    features = feature_layer.output
    
    # Lightweight FOMO Head:
    # 1. DepthwiseConv2D 3x3
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="fomo_head_dw")(features)
    x = layers.BatchNormalization(name="fomo_head_dw_bn")(x)
    x = layers.ReLU(6.0, name="fomo_head_dw_relu")(x)
    
    # 2. Pointwise Conv 1x1 to 32 channels
    x = layers.Conv2D(32, 1, padding="same", use_bias=False, name="fomo_head_pw")(x)
    x = layers.BatchNormalization(name="fomo_head_pw_bn")(x)
    x = layers.ReLU(6.0, name="fomo_head_pw_relu")(x)
    
    # 3. Final classification 1x1 Conv with Softmax
    outputs = layers.Conv2D(num_classes + 1, 1, padding="same", activation="softmax", name="fomo_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs=outputs, name="Pretrained_Micro_FOMO")
    
    total_params = model.count_params()
    print(f"[*] Pretrained Micro-FOMO total parameters: {total_params:,}")
    if total_params > MAX_ALLOWED_PARAMS:
        raise ValueError(f"CRITICAL: Param count {total_params} exceeds limit of {MAX_ALLOWED_PARAMS}!")
        
    return model

def export_to_tflite_int8(model, representative_data, output_tflite_path):
    """
    Exports Keras model to Full INT8 TFLite model optimized for TFLite Micro on ESP32-S3.
    """
    def rep_gen():
        for sample in representative_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant = converter.convert()
    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] Exported Full INT8 TFLite model: {size_kb:.2f} KB (Limit: {MAX_ALLOWED_MODEL_SIZE_KB} KB)")
    
    if size_kb > MAX_ALLOWED_MODEL_SIZE_KB:
        raise ValueError(f"CRITICAL: TFLite size {size_kb:.1f} KB exceeds {MAX_ALLOWED_MODEL_SIZE_KB} KB!")
        
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant)
    print(f"[✔] Saved TFLite model to: {output_tflite_path}")
    return tflite_quant

if __name__ == "__main__":
    m = create_pretrained_micro_fomo()
    m.summary()
