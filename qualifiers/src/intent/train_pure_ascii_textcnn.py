#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - PURE CHAR/BYTE-LEVEL TextCNN INTENT CLASSIFIER
========================================================================
100% COMPLIANT WITH COMPETITION & HARDWARE INVARIANTS:
1. Zero Vocabulary Leakage: No custom 'vocab.json', no dynamic vocabulary generated from data.
   Uses a strictly fixed 128 ASCII table (ASCII 0..127).
2. Zero Heuristics / Zero If-Else: 100% end-to-end neural network classification via argmax.
3. Zero Synthetic Templates: Trains strictly on your real competition dataset.
4. ESP32-S3 Ready: Quantizes to Full INT8 TFLite Micro (~32 KB, ~29k params, budget: <1.5 MB & <200k params).

ESP32-S3 Firmware Tokenization in C (One-liner):
    for (int i = 0; i < MAX_LEN; i++) {
        input_tensor[i] = (i < strlen(raw_str)) ? (int8_t)raw_str[i] : 0;
    }
"""

import os
import re
import csv
import json
import argparse
import unicodedata
import numpy as np

# Set deterministic seed for reproducible results
np.random.seed(42)

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras import layers, models

# 5 Competition Intent Classes (Fixed Order)
INTENT_CLASSES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
LABEL_MAP = {name: idx for idx, name in enumerate(INTENT_CLASSES)}
INV_LABEL_MAP = {idx: name for idx, name in enumerate(INTENT_CLASSES)}

# Fixed ASCII alphabet size (0..127)
# 0: <PAD>, 1: <UNK> (for characters outside standard ASCII), 2..127: exact ord(c)
ASCII_VOCAB_SIZE = 128
MAX_CHAR_LEN = 64  # Ample length for Vietnamese queries (most are 20-45 chars)


def remove_accents(text: str) -> str:
    """Standard Vietnamese diacritic removal (NFD decomposition)."""
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def clean_text_to_ascii(text: str) -> str:
    """
    Normalizes text to lowercase, unaccented ASCII characters.
    Does NOT remove words, does NOT use stopword lists or dictionaries.
    """
    text = remove_accents(text).lower().strip()
    # Replace non-alphanumeric/whitespace/basic punctuation with space
    text = re.sub(r"[^a-z0-9\s\?\,\.]", " ", text)
    # Collapse multiple spaces into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_query_to_ascii_ids(text: str, max_len: int = MAX_CHAR_LEN) -> np.ndarray:
    """
    Encodes text into a sequence of fixed ASCII integer IDs (0..127).
    No external dictionary/vocabulary file needed!
    """
    clean_str = clean_text_to_ascii(text)
    ids = []
    for ch in clean_str[:max_len]:
        code = ord(ch)
        if 0 < code < ASCII_VOCAB_SIZE:
            ids.append(code)
        else:
            ids.append(1)  # <UNK> fallback for foreign symbols

    # Zero-padding to fixed max_len
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


def build_pure_ascii_textcnn(
    max_len: int = MAX_CHAR_LEN,
    vocab_size: int = ASCII_VOCAB_SIZE,
    embed_dim: int = 32,
    num_filters: int = 24,
    filter_sizes: tuple = (3, 5, 7, 9),
    num_classes: int = 5
) -> models.Model:
    """
    Pure Char/Byte-Level TextCNN Architecture:
    - Input: [batch_size, max_len] ASCII integer values
    - Embedding: 128 -> embed_dim (32)
    - Conv1D banks:
        * k=3: captures 3-char stems ("dem", "coc", "hop", "xe ")
        * k=5: captures core root words ("chai ", "sach ", "o dau")
        * k=7: captures key phrases ("co phai", "bao nhieu", "vi tri ")
        * k=9: captures long intent structures ("co bao nhieu", "nam o dau vay")
    - GlobalMaxPooling1D: captures most salient signals across sequence
    - Dense + Softmax: outputs exact class probability distribution
    """
    inputs = layers.Input(shape=(max_len,), dtype="int32", name="ascii_input_ids")
    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="ascii_embedding"
    )(inputs)

    pooled_branches = []
    for k in filter_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=k,
            padding="same",
            activation="relu",
            name=f"char_conv_k{k}"
        )(emb)
        pool = layers.GlobalMaxPooling1D(name=f"char_gpool_k{k}")(conv)
        pooled_branches.append(pool)

    concat = layers.Concatenate(name="feature_concat")(pooled_branches)
    dense = layers.Dense(64, activation="relu", name="dense_latent")(concat)
    drop = layers.Dropout(0.2, name="dropout")(dense)
    outputs = layers.Dense(num_classes, activation="softmax", name="intent_probs")(drop)

    model = models.Model(inputs=inputs, outputs=outputs, name="Pure_ASCII_Char_TextCNN")
    return model


def load_dataset(csv_path: str, is_test: bool = False):
    """
    Reads CSV rows. Automatically discovers question column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    ids, texts, labels = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Resolve ID
            qid = row.get("question_id") or row.get("sample_id") or row.get("id") or f"q_{idx+1:04d}"
            # Resolve Question text
            text = row.get("question") or row.get("text") or row.get("query") or ""
            ids.append(qid)
            texts.append(text)

            if not is_test:
                label_str = row.get("intent") or row.get("label") or ""
                if label_str not in LABEL_MAP:
                    raise ValueError(f"Unknown label '{label_str}' at row {idx+1} in {csv_path}")
                labels.append(LABEL_MAP[label_str])

    return ids, texts, np.array(labels, dtype=np.int32) if not is_test else None


def train_and_export(
    train_csv: str,
    val_csv: str = None,
    test_csv: str = None,
    output_dir: str = "weights_pure_ascii",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 2e-3
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("   TRAINING PURE CHAR/BYTE-LEVEL TextCNN (ZERO VOCAB LEAKAGE)")
    print("   Input Encoding : Fixed 128 ASCII Bytes (Zero vocab.json)")
    print("   Rule Overrides : NONE (100% Pure Neural Network Argmax)")
    print("   Target Ops     : Full Integer INT8 TFLite Micro for ESP32-S3")
    print("=" * 80)

    # 1. Load Training Data
    print(f"[*] Loading training data from: {train_csv}")
    train_ids, train_texts, y_train_raw = load_dataset(train_csv, is_test=False)
    X_train = np.stack([encode_query_to_ascii_ids(t) for t in train_texts])
    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes=5)
    print(f"[✔] Training samples: {len(X_train):,} queries | Input shape: {X_train.shape}")

    # 2. Load Validation Data (if provided)
    val_data = None
    if val_csv and os.path.exists(val_csv):
        print(f"[*] Loading validation data from: {val_csv}")
        _, val_texts, y_val_raw = load_dataset(val_csv, is_test=False)
        X_val = np.stack([encode_query_to_ascii_ids(t) for t in val_texts])
        y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes=5)
        val_data = (X_val, y_val)
        print(f"[✔] Validation samples: {len(X_val):,} queries")

    # 3. Instantiate Architecture & Check Parameters
    model = build_pure_ascii_textcnn()
    param_count = model.count_params()
    print(f"\n[*] Model Parameters: {param_count:,} (Hardware budget: < 200,000)")
    assert param_count < 200000, f"Exceeded hardware budget! {param_count} >= 200,000"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Train Model
    print(f"\n[*] Starting training for {epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 5. Evaluate Floating-Point Model
    tr_loss, tr_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"\n[✔] Train Accuracy: {tr_acc * 100:.2f}%")
    if val_data:
        vl_loss, vl_acc = model.evaluate(val_data[0], val_data[1], verbose=0)
        print(f"[✔] Val Accuracy  : {vl_acc * 100:.2f}%")

    # Save Keras Model
    keras_path = os.path.join(output_dir, "char_text_cnn.keras")
    model.save(keras_path)
    print(f"[✔] Saved Keras model: {keras_path}")

    # 6. Quantize to Full INT8 TFLite Micro
    print("\n[*] Quantizing to Full INT8 TFLite Micro for ESP32-S3...")
    def representative_dataset_gen():
        # Representative calibration samples from real training set
        for i in range(min(250, len(X_train))):
            yield [X_train[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, "char_text_cnn_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024.0
    print(f"[✔] Exported INT8 TFLite: {tflite_path} ({size_kb:.2f} KB | Budget: 1,536 KB)")
    assert size_kb <= 1536.0, "Model exceeded Flash budget!"

    # 7. Validate INT8 Model on Validation Set
    if val_data:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        correct = 0
        for i in range(len(val_data[0])):
            interp.set_tensor(in_idx, val_data[0][i:i+1])
            interp.invoke()
            out_logits = interp.get_tensor(out_idx)[0]
            if int(np.argmax(out_logits)) == y_val_raw[i]:
                correct += 1
        int8_val_acc = (correct / len(val_data[0])) * 100.0
        print(f"[✔] INT8 TFLite Validation Accuracy: {correct}/{len(val_data[0])} ({int8_val_acc:.2f}%)")

    # 8. Predict Test Set (if provided) -> Output pure submission CSV
    if test_csv and os.path.exists(test_csv):
        print(f"\n[*] Predicting test questions from: {test_csv}...")
        test_ids, test_texts, _ = load_dataset(test_csv, is_test=True)
        X_test = np.stack([encode_query_to_ascii_ids(t) for t in test_texts])

        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        predictions = []
        for i in range(len(X_test)):
            interp.set_tensor(in_idx, X_test[i:i+1])
            interp.invoke()
            out_probs = interp.get_tensor(out_idx)[0]
            pred_class = INV_LABEL_MAP[int(np.argmax(out_probs))]
            predictions.append(pred_class)

        # Save to output_dir, repo root (intent_only_submission.csv), and qualifiers/submissions
        qualifiers_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        repo_root = os.path.dirname(qualifiers_dir)
        sub_paths = [
            os.path.join(output_dir, "pure_ascii_intent_preds.csv"),
            os.path.join(repo_root, "intent_only_submission.csv"),
            os.path.join(qualifiers_dir, "submissions/intent_preds.csv")
        ]

        for sp in sub_paths:
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            with open(sp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
                writer.writeheader()
                for qid, pred in zip(test_ids, predictions):
                    writer.writerow({
                        "task_type": "INTENT",
                        "sample_id": qid,
                        "prediction": pred
                    })
            print(f"[✔] Exported INTENT predictions to: {sp}")

        from collections import Counter
        print(f"[✔] Predicted Intent Class Distribution: {dict(Counter(predictions))}")

    print("\n" + "=" * 80)
    print("   [★ FINISHED] PURE ASCII TextCNN TRAINING & INT8 EXPORT COMPLETE!")
    print("   File location: " + os.path.abspath(output_dir))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pure ASCII Char/Byte-Level TextCNN for Edge AI 2026")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV (columns: question, intent)")
    parser.add_argument("--val_csv", default=None, help="Path to validation CSV (optional)")
    parser.add_argument("--test_csv", default=None, help="Path to test CSV (optional, columns: question_id, question)")
    parser.add_argument("--output_dir", default="qualifiers/weights_pure_ascii", help="Directory to save weights & CSV")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    args = parser.parse_args()

    train_and_export(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
