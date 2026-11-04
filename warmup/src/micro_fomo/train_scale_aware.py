import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

GRID_SIZE = 16
NUM_CLASSES = 10
TOTAL_CATEGORIES = 21  # 0: bg, 1..10: small classes 0..9, 11..20: large classes 0..9
INPUT_SIZE = 128
ORIG_W = 1672.0
ORIG_H = 941.0

def build_scale_aware_fomo(input_shape=(128, 128, 3), alpha=0.35):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    feature_layer = base.get_layer("block_5_add")
    features = feature_layer.output  # (None, 16, 16, 16) for alpha=0.35

    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="fomo_dw")(features)
    x = tf.keras.layers.BatchNormalization(name="fomo_dw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="fomo_dw_relu")(x)

    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=False, name="fomo_pw")(x)
    x = tf.keras.layers.BatchNormalization(name="fomo_pw_bn")(x)
    x = tf.keras.layers.ReLU(6.0, name="fomo_pw_relu")(x)
    x = tf.keras.layers.Dropout(0.2, name="fomo_dropout")(x)

    outputs = tf.keras.layers.Conv2D(TOTAL_CATEGORIES, 1, padding="same", activation="softmax", name="fomo_output")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs, name="ScaleAware_Micro_FOMO")
    return model

def calc_iou(w1, h1, w2, h2):
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def load_scale_aware_dataset(split_file, labels_dir, priors_path, img_size=128):
    with open(priors_path, "r") as f:
        priors = json.load(f)

    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]

    X = []
    Y = []

    for p in img_paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(labels_dir, f"{sid}.txt")

        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        norm = img_resized.astype(np.float32) / 255.0

        target = np.zeros((GRID_SIZE, GRID_SIZE, TOTAL_CATEGORIES), dtype=np.float32)
        target[:, :, 0] = 1.0  # Background

        if os.path.exists(lbl):
            boxes = []
            with open(lbl, "r") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cid, cx, cy, w, h, w * h))

            # Larger boxes first, smaller take priority
            boxes.sort(key=lambda b: b[5], reverse=True)

            for cid, cx, cy, w, h, _ in boxes:
                gx = min(GRID_SIZE - 1, max(0, int(cx * GRID_SIZE)))
                gy = min(GRID_SIZE - 1, max(0, int(cy * GRID_SIZE)))

                # Determine best anchor (small vs large)
                cls_p = priors[str(cid)]
                iou_small = calc_iou(w, h, cls_p["small"]["w"], cls_p["small"]["h"])
                iou_large = calc_iou(w, h, cls_p["large"]["w"], cls_p["large"]["h"])

                if iou_small >= iou_large:
                    cat_id = cid + 1        # Small category: 1..10
                else:
                    cat_id = cid + 11       # Large category: 11..20

                target[gy, gx, 0] = 0.0
                target[gy, gx, 1:] = 0.0
                target[gy, gx, cat_id] = 1.0

        X.append(norm)
        Y.append(target)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def weighted_scale_aware_loss(gamma=2.0, fg_weight=15.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        is_bg = y_true[:, :, :, 0:1]
        is_fg = 1.0 - is_bg

        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)

        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        class_weight = is_bg * 1.0 + is_fg * fg_weight
        total_loss = ce * focal_weight * class_weight
        return tf.reduce_mean(total_loss)
    return loss

def train_and_export():
    train_split = "data/train_split.txt"
    val_split = "data/val_split.txt"
    labels_dir = "data/train/labels"
    priors_path = "data/scale_aware_priors.json"
    save_dir = "weights/scale_aware_fomo"
    os.makedirs(save_dir, exist_ok=True)

    print("[*] Loading Scale-Aware dataset (128x128, 21 categories)...")
    X_train, Y_train = load_scale_aware_dataset(train_split, labels_dir, priors_path, img_size=INPUT_SIZE)
    X_val, Y_val = load_scale_aware_dataset(val_split, labels_dir, priors_path, img_size=INPUT_SIZE)
    print(f"[*] Train set: {len(X_train)} images, Val set: {len(X_val)} images")

    # Data Augmentation on train set: horizontal flips
    X_aug, Y_aug = [], []
    for i in range(len(X_train)):
        img, tgt = X_train[i], Y_train[i]
        X_aug.append(img)
        Y_aug.append(tgt)
        # Flip horizontal
        img_flip = cv2.flip(img, 1)
        tgt_flip = np.flip(tgt, axis=1)
        X_aug.append(img_flip)
        Y_aug.append(tgt_flip)

    X_train = np.array(X_aug, dtype=np.float32)
    Y_train = np.array(Y_aug, dtype=np.float32)
    print(f"[✔] Augmented Train set size: {len(X_train)} images")

    model = build_scale_aware_fomo(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), alpha=0.35)
    print(f"[✔] Model parameters: {model.count_params():,} (Limit: 200,000)")

    loss_fn = weighted_scale_aware_loss(fg_weight=15.0)
    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    keras_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "scale_aware_fomo_int8.tflite")

    cb_list = [
        callbacks.ModelCheckpoint(keras_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    epochs = 40
    print(f"[*] Training Scale-Aware Micro-FOMO for {epochs} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=cb_list,
        verbose=1
    )

    # Export INT8 TFLite
    print(f"\n[*] Reloading best model from {keras_path} for INT8 export...")
    best_model = tf.keras.models.load_model(keras_path, custom_objects={"loss": loss_fn})

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(min(len(X_train), 50)):
            yield [np.expand_dims(X_train[i], axis=0)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024.0
    print(f"[✔] Successfully exported {tflite_path}: {size_kb:.2f} KB (Limit: 1,500 KB)")
    return tflite_path

if __name__ == "__main__":
    train_and_export()
