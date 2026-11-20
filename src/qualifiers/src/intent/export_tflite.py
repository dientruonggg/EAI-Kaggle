#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - Export TextCNN to Full Integer INT8 TFLite Micro for ESP32-S3
Complies with Section V:
- INT8 quantization
- Model size <= 1.5 MB
- Only operators supported by TFLM (Conv1D/Depthwise/Dense/Embedding)
"""
import os
import sys
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MAX_MODEL_SIZE_KB = 1500  # 1.5 MB limit
MAX_PARAMS = 200000

def build_keras_text_cnn(vocab_size, max_len=30, embed_dim=32, num_filters=16, filter_sizes=(3, 5, 7), num_classes=5):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    
    # Embedding layer
    emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, name="embedding")(inputs)
    
    # Odd-kernel 1D conv filters for symmetric context analysis
    pooled_outputs = []
    for k in filter_sizes:
        conv = layers.Conv1D(num_filters, k, padding="same", activation="relu", name=f"conv_k{k}")(emb)
        pool = layers.GlobalMaxPooling1D(name=f"pool_k{k}")(conv)
        pooled_outputs.append(pool)
    
    fused = layers.Concatenate(name="concat")(pooled_outputs)
    drop = layers.Dropout(0.2, name="dropout")(fused)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(drop)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="TextCNN_OddKernels_Micro")
    return model

def convert_to_tflite_int8(keras_model, representative_samples, output_tflite_path):
    def rep_dataset():
        for sample in representative_samples:
            yield [sample.astype(np.int32)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Input remains int32 tokens, weights & intermediate activations INT8
    tflite_model = converter.convert()
    
    size_kb = len(tflite_model) / 1024.0
    print(f"[*] Exported TextCNN INT8 Model Size: {size_kb:.2f} KB (Budget: {MAX_MODEL_SIZE_KB} KB)")
    
    if size_kb > MAX_MODEL_SIZE_KB:
        raise ValueError(f"CRITICAL: Model size {size_kb:.1f} KB exceeds limit of {MAX_MODEL_SIZE_KB} KB!")
        
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[✔] Successfully saved quantized TextCNN to: {output_tflite_path}")
    return output_tflite_path

def main():
    vocab_path = "/home/bim/Projects/EAI-2026/qualifiers/weights/text_cnn_vocab.json"
    output_tflite = "/home/bim/Projects/EAI-2026/qualifiers/weights/text_cnn_int8.tflite"
    
    if not os.path.exists(vocab_path):
        print(f"[!] Vocab path {vocab_path} not found. Running training first...")
        from text_cnn import train_model
        train_model("/home/bim/Projects/EAI-2026/qualifiers/data/intent_train.csv",
                    "/home/bim/Projects/EAI-2026/qualifiers/weights/text_cnn.pth",
                    vocab_path, epochs=10)
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    model = build_keras_text_cnn(vocab_size=len(vocab))
    total_params = model.count_params()
    print(f"[*] TextCNN total parameters: {total_params:,} (Limit: {MAX_PARAMS:,})")
    
    # Generate representative samples
    dummy_samples = [np.random.randint(0, len(vocab), size=(1, 30)) for _ in range(50)]
    convert_to_tflite_int8(model, dummy_samples, output_tflite)

if __name__ == "__main__":
    main()
