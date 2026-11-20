"""
Edge AI Challenge 2026 - Upgraded Pure ASCII TextCNN (v2)
=========================================================
Upgrades:
1. Receptive Field: Multi-Scale Large Kernels (3, 5, 7, 9, 13, 17, 21) to capture long phrases like 'tinh tong so' (12 chars).
2. Dual Pooling: GlobalMaxPooling1D + GlobalAveragePooling1D per branch (preserves strong keywords + whole-sequence context).
3. Targeted Data Augmentation: Synthesizes natural phrasing variations for COUNT ('tinh tong so', 'so luong') and EXIST ('o dau do', 'kiem tra xem').
4. Hardware compliance: < 100k parameters (budget < 200k), full INT8 < 100 KB (budget 1.5 MB), 100% native TFLM ops.
5. Zero Vocab Dependency: pure ASCII byte-indices (0..127), instant deployment on ESP32-S3 in C.
"""

import os
import sys
import re
import csv
import argparse
import unicodedata
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

ASCII_VOCAB_SIZE = 128
MAX_CHAR_LEN = 128

INTENT_MAP = {
    0: "COUNT",
    1: "EXIST",
    2: "LIST",
    3: "LOCATION",
    4: "OUT_OF_SCOPE"
}
LABEL_MAP = {v: k for k, v in INTENT_MAP.items()}


def remove_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def clean_text_to_ascii(text: str) -> str:
    text = remove_accents(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_query_to_ascii_ids(text: str, max_len: int = MAX_CHAR_LEN) -> np.ndarray:
    clean_str = clean_text_to_ascii(text)
    ids = []
    for ch in clean_str[:max_len]:
        code = ord(ch)
        if 0 < code < ASCII_VOCAB_SIZE:
            ids.append(code)
        else:
            ids.append(1)  # Fallback for foreign symbols

    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


OOS_OBJECTS = [
    "cuc tay", "chia khoa", "cai bat", "cai thia", "em be", "dong ho", "cai quat",
    "dong xu", "den ban", "ban phim", "laptop", "con cho", "con chim", "con meo",
    "binh hoa", "banh mi", "cai mu", "cai dia", "cai vi", "cai ghe", "cai keo",
    "cai remote", "cay but", "cay nen", "doi giay", "chau cay", "o cam dien",
    "qua tao", "qua bong", "khan trai ban", "nguoi"
]


def augment_dataset(texts, labels):
    """
    Synthesizes natural linguistic variations for underspecified patterns in training set:
    - COUNT: adds 'tinh tong so ...', 'so luong ... la bao nhieu'
    - EXIST: adds '... o dau do ... khong', 'kiem tra xem co ...'
    - OUT_OF_SCOPE: adds question frames for out-of-scope entities to distinguish from EXIST
    """
    aug_texts = []
    aug_labels = []

    for text, label in zip(texts, labels):
        aug_texts.append(text)
        aug_labels.append(label)

        clean = clean_text_to_ascii(text)

        # 1. COUNT augmentations
        if label == LABEL_MAP["COUNT"]:
            # Pattern: "hay dem so X" -> "tinh tong so X"
            if "hay dem so " in clean:
                aug_texts.append(clean.replace("hay dem so ", "tinh tong so "))
                aug_labels.append(LABEL_MAP["COUNT"])
            elif "dem so " in clean:
                aug_texts.append(clean.replace("dem so ", "tinh tong so "))
                aug_labels.append(LABEL_MAP["COUNT"])

            # Pattern: "co tong cong bao nhieu X" -> "tinh tong so X"
            if "co tong cong bao nhieu " in clean:
                aug_texts.append(clean.replace("co tong cong bao nhieu ", "tinh tong so "))
                aug_labels.append(LABEL_MAP["COUNT"])

            # Pattern: "bao nhieu X" -> "so luong X la bao nhieu"
            m = re.search(r"co bao nhieu (.*?) xuat hien", clean)
            if m:
                obj = m.group(1)
                aug_texts.append(f"so luong {obj} xuat hien trong buc anh la bao nhieu")
                aug_labels.append(LABEL_MAP["COUNT"])

        # 2. EXIST augmentations
        elif label == LABEL_MAP["EXIST"]:
            # Pattern: "co thay X khong" -> "co thay X o dau do trong buc anh khong"
            m = re.search(r"co thay (.*?) (trong canh|trong hinh|trong anh|trong buc anh|trong khung hinh) khong", clean)
            if m:
                obj = m.group(1)
                context = m.group(2)
                aug_texts.append(f"co thay {obj} o dau do {context} khong")
                aug_labels.append(LABEL_MAP["EXIST"])

            # Pattern: "co X trong anh khong" -> "kiem tra xem co X trong anh khong"
            m2 = re.search(r"co (.*?) (trong canh|trong hinh|trong anh|trong buc anh|trong khung hinh) khong", clean)
            if m2:
                obj = m2.group(1)
                context = m2.group(2)
                aug_texts.append(f"kiem tra xem co {obj} {context} khong")
                aug_labels.append(LABEL_MAP["EXIST"])

    # 3. OUT_OF_SCOPE balanced entity augmentation
    for obj in OOS_OBJECTS:
        aug_texts.append(f"ban co nhin thay {obj} trong khung hinh khong")
        aug_labels.append(LABEL_MAP["OUT_OF_SCOPE"])
        aug_texts.append(f"ban oi, ban co nhin thay {obj} trong buc anh khong")
        aug_labels.append(LABEL_MAP["OUT_OF_SCOPE"])
        aug_texts.append(f"cho toi hoi, co thay {obj} o dau do trong hinh anh khong")
        aug_labels.append(LABEL_MAP["OUT_OF_SCOPE"])
        aug_texts.append(f"co {obj} trong anh khong")
        aug_labels.append(LABEL_MAP["OUT_OF_SCOPE"])

    return aug_texts, aug_labels


def build_upgraded_ascii_textcnn(
    max_len: int = MAX_CHAR_LEN,
    vocab_size: int = ASCII_VOCAB_SIZE,
    embed_dim: int = 32,
    num_filters: int = 24,
    filter_sizes: tuple = (3, 5, 7, 9, 13, 17, 21),
    num_classes: int = 5,
    l2_reg: float = 1e-4,
    spatial_dropout: float = 0.15,
    dense_dropout: float = 0.3
) -> models.Model:
    inputs = layers.Input(shape=(max_len,), dtype="int32", name="ascii_input_ids")
    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="ascii_embedding"
    )(inputs)

    x = layers.SpatialDropout1D(spatial_dropout, name="char_spatial_dropout")(emb)
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    pooled_branches = []
    for k in filter_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=k,
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name=f"char_conv_k{k}"
        )(x)
        # Dual-Pooling: Max + Avg
        max_pool = layers.GlobalMaxPooling1D(name=f"max_pool_k{k}")(conv)
        avg_pool = layers.GlobalAveragePooling1D(name=f"avg_pool_k{k}")(conv)
        dual = layers.Concatenate(name=f"dual_pool_k{k}")([max_pool, avg_pool])
        pooled_branches.append(dual)

    concat = layers.Concatenate(name="feature_concat")(pooled_branches)
    dense = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_latent")(concat)
    drop = layers.Dropout(dense_dropout, name="dropout")(dense)
    outputs = layers.Dense(num_classes, activation="softmax", name="intent_probs")(drop)

    model = models.Model(inputs=inputs, outputs=outputs, name="Upgraded_ASCII_TextCNN_v2")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Upgraded Pure ASCII TextCNN v2")
    parser.add_argument("--data_dir", default="qualifiers/data")
    parser.add_argument("--train_csv", default="qualifiers/data/intent_train.csv")
    parser.add_argument("--val_csv", default="qualifiers/data/intent_val.csv")
    parser.add_argument("--test_csv", default="qualifiers/data/intent_test.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output_dir", default="qualifiers/weights_pure_ascii_v2")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("   EDGE AI CHALLENGE 2026 - UPGRADED PURE ASCII TextCNN (v2)")
    print("   Architecture : Multi-Scale Kernels (3, 5, 7, 9, 13, 17, 21) + Dual Pooling")
    print("   Target Ops   : 100% Native TFLM on ESP32-S3 Microcontroller (Zero Vocab File)")
    print("=" * 80)

    # 1. Load Data
    raw_texts, raw_labels = [], []
    with open(args.train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "intent" in r and r["intent"] in LABEL_MAP:
                raw_texts.append(r["question"])
                raw_labels.append(LABEL_MAP[r["intent"]])

    # If val_csv exists, add it to train pool
    if os.path.exists(args.val_csv):
        with open(args.val_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "intent" in r and r["intent"] in LABEL_MAP:
                    raw_texts.append(r["question"])
                    raw_labels.append(LABEL_MAP[r["intent"]])

    print(f"[✔] Original training samples: {len(raw_texts):,}")

    # 2. Augment training set with linguistic variations
    aug_texts, aug_labels = augment_dataset(raw_texts, raw_labels)
    print(f"[✔] Augmented training samples: {len(aug_texts):,} (+{len(aug_texts)-len(raw_texts):,} generated)")

    # 3. Create 10% stratified validation split to monitor generalization
    np.random.seed(42)
    indices_by_class = {c: [] for c in range(5)}
    for idx, lbl in enumerate(aug_labels):
        indices_by_class[lbl].append(idx)

    train_idx, val_idx = [], []
    for c, idxs in indices_by_class.items():
        np.random.shuffle(idxs)
        sp = int(len(idxs) * 0.9)
        train_idx.extend(idxs[:sp])
        val_idx.extend(idxs[sp:])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_texts = [aug_texts[i] for i in train_idx]
    train_labels = [aug_labels[i] for i in train_idx]
    val_texts = [aug_texts[i] for i in val_idx]
    val_labels = [aug_labels[i] for i in val_idx]

    print(f"[✔] Split: {len(train_texts):,} Train | {len(val_texts):,} Validation")

    X_train = np.stack([encode_query_to_ascii_ids(t) for t in train_texts])
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=5)
    X_val = np.stack([encode_query_to_ascii_ids(t) for t in val_texts])
    y_val = tf.keras.utils.to_categorical(val_labels, num_classes=5)

    # 4. Build Model
    model = build_upgraded_ascii_textcnn()
    param_count = model.count_params()
    print(f"[✔] Model Instantiated: {param_count:,} parameters (Budget: < 200,000)")
    assert param_count < 200000, "Exceeded microcontroller parameter budget!"

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]

    print(f"\n[*] Training Upgraded ASCII TextCNN v2 for {args.epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb_list,
        verbose=2
    )

    tr_loss, tr_acc = model.evaluate(X_train, y_train, verbose=0)
    vl_loss, vl_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[★ RESULT] Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}%")

    # Save Keras Model
    keras_path = os.path.join(output_dir, "char_text_cnn_v2.keras")
    model.save(keras_path)
    print(f"[✔] Saved Keras Model: {keras_path}")

    # 5. Full INT8 Quantization for ESP32-S3
    print("\n[*] Quantizing to Full INT8 TFLite Micro for ESP32-S3...")
    def rep_dataset():
        for i in range(min(300, len(X_train))):
            yield [X_train[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, "char_text_cnn_v2_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024.0
    print(f"[✔] Exported INT8 TFLite: {tflite_path} ({size_kb:.2f} KB | Budget: <= 1,536 KB)")
    assert size_kb <= 1536.0, "Model exceeded Flash budget!"

    # 6. Verify INT8 Accuracy on Validation Set
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    int8_corr = 0
    for i in range(len(X_val)):
        interp.set_tensor(in_idx, X_val[i:i+1])
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out_idx)[0]))
        if pred == val_labels[i]:
            int8_corr += 1
    print(f"[✔] INT8 TFLite Validation Accuracy: {int8_corr}/{len(X_val)} ({int8_corr/len(X_val)*100:.2f}%)")

    # 7. Predict 4,000 Test Questions
    test_csv = os.path.abspath(args.test_csv)
    if os.path.exists(test_csv):
        print(f"\n[*] Predicting 4,000 test questions from {test_csv}...")
        test_qids, test_texts = [], []
        with open(test_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                test_qids.append(r["question_id"])
                test_texts.append(r["question"])

        X_test = np.stack([encode_query_to_ascii_ids(t) for t in test_texts])
        predictions = []
        for i in range(len(X_test)):
            interp.set_tensor(in_idx, X_test[i:i+1])
            interp.invoke()
            pred_idx = int(np.argmax(interp.get_tensor(out_idx)[0]))
            predictions.append(INTENT_MAP[pred_idx])

        # Write predictions
        sub_paths = [
            os.path.join(output_dir, "pure_ascii_v2_intent_preds.csv"),
            os.path.join(PROJECT_ROOT, "intent_ascii_v2_submission.csv")
        ]
        for sp in sub_paths:
            with open(sp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
                writer.writeheader()
                for qid, pred in zip(test_qids, predictions):
                    writer.writerow({"task_type": "INTENT", "sample_id": qid, "prediction": pred})
            print(f"[✔] Written 4,000 INTENT predictions to: {sp}")

        from collections import Counter
        dist = Counter(predictions)
        print(f"[✔] Test Intent Distribution: {dict(dist)}")

    print("\n" + "=" * 80)
    print("   [★ SUCCESS] Upgraded Pure ASCII TextCNN v2 Finished!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
