import os
import glob
import json
import cv2
import numpy as np
import tensorflow as tf

GRID_SIZE = 16
NUM_CLASSES = 10
INPUT_SIZE = 128
ORIG_W = 1672.0
ORIG_H = 941.0

def build_centernet_micro(input_shape=(128, 128, 3), alpha=0.5):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )
    # Output of block 5 is (None, 16, 16, 16 for alpha=0.5)
    feat = base.get_layer('block_5_add').output

    # Branch 1: Classification logits (11 channels: 0=bg, 1..10=objects)
    cls_logits = tf.keras.layers.Conv2D(11, (1, 1), padding='same', name='cls_logits')(feat)

    # Branch 2: Bounding box regression (4 channels: dx, dy, w, h with sigmoid)
    box_raw = tf.keras.layers.Conv2D(4, (1, 1), padding='same', name='box_conv')(feat)
    box_pred = tf.keras.layers.Activation('sigmoid', name='box_preds')(box_raw)

    model = tf.keras.Model(inputs=base.input, outputs=[cls_logits, box_pred], name='centernet_micro')
    return model

def load_dataset(split_file, labels_dir, is_train=True):
    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]

    images = []
    cls_targets = []
    box_targets = []
    mask_targets = []

    for img_p in img_paths:
        sid = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(labels_dir, f"{sid}.txt")

        img = cv2.imread(img_p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        norm_img = img_resized.astype(np.float32) / 255.0

        # Shapes:
        # cls: (16, 16, 11)
        # box: (16, 16, 4)
        # mask: (16, 16, 1)
        c_target = np.zeros((GRID_SIZE, GRID_SIZE, 11), dtype=np.float32)
        c_target[:, :, 0] = 1.0  # background default
        b_target = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
        m_target = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)

        if os.path.exists(lbl_p):
            boxes = []
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cid, cx, cy, w, h, w * h))

            # Larger boxes first, smaller overwrite
            boxes.sort(key=lambda b: b[5], reverse=True)

            for cid, cx, cy, w, h, _ in boxes:
                gx = int(cx * GRID_SIZE)
                gy = int(cy * GRID_SIZE)
                gx = min(max(gx, 0), GRID_SIZE - 1)
                gy = min(max(gy, 0), GRID_SIZE - 1)

                dx = cx * GRID_SIZE - gx
                dy = cy * GRID_SIZE - gy

                c_target[gy, gx, 0] = 0.0
                c_target[gy, gx, 1:11] = 0.0
                c_target[gy, gx, cid + 1] = 1.0

                b_target[gy, gx, 0] = dx
                b_target[gy, gx, 1] = dy
                b_target[gy, gx, 2] = w
                b_target[gy, gx, 3] = h
                m_target[gy, gx, 0] = 1.0

        images.append(norm_img)
        cls_targets.append(c_target)
        box_targets.append(b_target)
        mask_targets.append(m_target)

    return (
        np.array(images, dtype=np.float32),
        np.array(cls_targets, dtype=np.float32),
        np.array(box_targets, dtype=np.float32),
        np.array(mask_targets, dtype=np.float32)
    )

@tf.function
def compute_loss(y_true_cls, y_true_box, mask, pred_cls, pred_box, gamma=2.0, alpha=0.25, lambda_box=5.0):
    # 1. Focal classification loss
    probs = tf.nn.softmax(pred_cls, axis=-1)
    ce = tf.keras.losses.categorical_crossentropy(y_true_cls, pred_cls, from_logits=True)
    pt = tf.reduce_sum(y_true_cls * probs, axis=-1)
    focal_weight = tf.pow(1.0 - pt, gamma)

    is_bg = y_true_cls[:, :, :, 0]
    alpha_weight = is_bg * (1.0 - alpha) + (1.0 - is_bg) * alpha
    cls_loss = tf.reduce_mean(alpha_weight * focal_weight * ce)

    # 2. Box regression loss (Smooth L1) on positive cells
    diff = tf.abs(y_true_box - pred_box)
    huber = tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5)
    box_loss_per_cell = tf.reduce_sum(huber, axis=-1)
    pos_mask = tf.squeeze(mask, axis=-1)
    pos_count = tf.reduce_sum(pos_mask)
    box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss_per_cell * pos_mask), pos_count)

    total_loss = cls_loss + lambda_box * box_loss
    return total_loss, cls_loss, box_loss

def train_and_export():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    out_dir = "/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo"
    os.makedirs(out_dir, exist_ok=True)

    print("[*] Loading training data...")
    X_train, Yc_train, Yb_train, Ym_train = load_dataset(train_split, labels_dir, is_train=True)
    X_val, Yc_val, Yb_val, Ym_val = load_dataset(val_split, labels_dir, is_train=False)
    print(f"[✔] Loaded {len(X_train)} train images, {len(X_val)} val images.")

    model = build_centernet_micro(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), alpha=0.5)
    print(f"[✔] Model parameters: {model.count_params():,} (Limit: 200,000)")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    batch_size = 16
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Yc_train, Yb_train, Ym_train)).shuffle(200).batch(batch_size)

    epochs = 80
    print(f"[*] Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_cls = 0.0
        epoch_box = 0.0
        num_batches = 0

        for x_b, yc_b, yb_b, ym_b in train_ds:
            with tf.GradientTape() as tape:
                pred_c, pred_b = model(x_b, training=True)
                loss, c_loss, b_loss = compute_loss(yc_b, yb_b, ym_b, pred_c, pred_b)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_total += float(loss)
            epoch_cls += float(c_loss)
            epoch_box += float(b_loss)
            num_batches += 1

        if epoch % 10 == 0 or epoch == epochs:
            val_c_pred, val_b_pred = model(X_val, training=False)
            val_loss, val_c, val_b = compute_loss(Yc_val, Yb_val, Ym_val, val_c_pred, val_b_pred)
            print(f"[Epoch {epoch:02d}/{epochs}] Train: {epoch_total/num_batches:.4f} (cls: {epoch_cls/num_batches:.4f}, box: {epoch_box/num_batches:.4f}) | Val Loss: {float(val_loss):.4f}")

    # Export Full INT8 TFLite with separate outputs
    print("\n[*] Exporting to Full INT8 TFLite with independent quantization scales...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_dataset():
        for i in range(min(len(X_train), 50)):
            yield [np.expand_dims(X_train[i], axis=0)]

    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()

    tflite_path = os.path.join(out_dir, "fomo_bbox_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024.0
    print(f"[✔] Successfully exported {tflite_path}!")
    print(f"[✔] Size: {size_kb:.2f} KB (Limit: 1,500 KB)")
    return tflite_path

if __name__ == "__main__":
    train_and_export()
