import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
import random
import cv2
import numpy as np
import tensorflow as tf

GRID_SIZE = 16
NUM_CLASSES = 10
INPUT_SIZE = 128
ORIG_W = 1672.0
ORIG_H = 941.0

def build_augmented_fomo_bbox(input_shape=(128, 128, 3), alpha=0.5):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )
    # Output of block 5 is (None, 16, 16, 16 for alpha=0.5)
    feat = base.get_layer('block_5_add').output

    # Spatial context fusion: 3x3 Depthwise Conv + 1x1 Pointwise Conv + BN + ReLU6
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False, name='head_dw')(feat)
    x = tf.keras.layers.BatchNormalization(name='head_dw_bn')(x)
    x = tf.keras.layers.ReLU(6.0, name='head_dw_relu')(x)

    x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=False, name='head_pw')(x)
    x = tf.keras.layers.BatchNormalization(name='head_pw_bn')(x)
    x = tf.keras.layers.ReLU(6.0, name='head_pw_relu')(x)
    x = tf.keras.layers.Dropout(0.2, name='head_dropout')(x)

    # Classification output (Softmax over 11 classes: 0=bg, 1..10=objects)
    cls_out = tf.keras.layers.Conv2D(11, 1, padding='same', activation='softmax', name='cls_output')(x)

    # Box regression output (Sigmoid over 4 channels: dx, dy, w, h)
    box_raw = tf.keras.layers.Conv2D(4, 1, padding='same', name='box_conv')(x)
    box_out = tf.keras.layers.Activation('sigmoid', name='box_output')(box_raw)

    model = tf.keras.Model(inputs=base.input, outputs=[cls_out, box_out], name='fomo_bbox_augmented')
    return model

def load_raw_samples(split_file, labels_dir):
    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]

    samples = []
    for p in img_paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        lbl_p = os.path.join(labels_dir, f"{sid}.txt")
        boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cid, cx, cy, w, h, w * h))
        samples.append((p, boxes))
    return samples

def augment_sample(img_rgb, boxes):
    """
    Applies on-the-fly random horizontal flip and photometric jitter
    """
    img = img_rgb.copy()
    aug_boxes = [b for b in boxes]

    # 1. Random horizontal flip (p=0.5)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        flipped_boxes = []
        for cid, cx, cy, w, h, area in aug_boxes:
            flipped_boxes.append((cid, 1.0 - cx, cy, w, h, area))
        aug_boxes = flipped_boxes

    # 2. Random brightness / contrast
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.uniform(-20, 20)      # brightness
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img, aug_boxes

def encode_targets(boxes):
    c_target = np.zeros((GRID_SIZE, GRID_SIZE, 11), dtype=np.float32)
    c_target[:, :, 0] = 1.0  # background default
    b_target = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    m_target = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)

    # Sort boxes by area descending so smaller boxes overwrite
    sorted_boxes = sorted(boxes, key=lambda b: b[5], reverse=True)

    for cid, cx, cy, w, h, _ in sorted_boxes:
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

    return c_target, b_target, m_target

def data_generator(samples, batch_size=16, is_train=True):
    num_samples = len(samples)
    indices = list(range(num_samples))

    # Pre-cache raw images in memory to speed up training
    cached_images = []
    cached_boxes = []
    for p, b in samples:
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        cached_images.append(img)
        cached_boxes.append(b)

    while True:
        if is_train:
            random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            b_imgs = []
            b_cls = []
            b_box = []
            b_mask = []

            for idx in batch_indices:
                img = cached_images[idx]
                boxes = cached_boxes[idx]

                if is_train:
                    img_aug, boxes_aug = augment_sample(img, boxes)
                else:
                    img_aug, boxes_aug = img, boxes

                c_t, b_t, m_t = encode_targets(boxes_aug)
                norm_img = img_aug.astype(np.float32) / 255.0

                b_imgs.append(norm_img)
                b_cls.append(c_t)
                b_box.append(b_t)
                b_mask.append(m_t)

            yield (
                np.array(b_imgs, dtype=np.float32),
                np.array(b_cls, dtype=np.float32),
                np.array(b_box, dtype=np.float32),
                np.array(b_mask, dtype=np.float32)
            )

@tf.function
def compute_loss(y_true_cls, y_true_box, mask, pred_cls, pred_box, gamma=2.0, fg_weight=15.0, lambda_box=5.0):
    # Pred cls is Softmax probabilities
    probs = tf.clip_by_value(pred_cls, 1e-7, 1.0 - 1e-7)

    # 1. Weighted Focal Cross Entropy
    is_bg = y_true_cls[:, :, :, 0:1]
    is_fg = 1.0 - is_bg

    pt = tf.reduce_sum(y_true_cls * probs, axis=-1, keepdims=True)
    focal_weight = tf.pow(1.0 - pt, gamma)

    ce = -tf.reduce_sum(y_true_cls * tf.math.log(probs), axis=-1, keepdims=True)
    class_weight = is_bg * 1.0 + is_fg * fg_weight
    cls_loss = tf.reduce_mean(ce * focal_weight * class_weight)

    # 2. Smooth L1 Box Loss on positive cells only
    diff = tf.abs(y_true_box - pred_box)
    huber = tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5)
    box_loss_per_cell = tf.reduce_sum(huber, axis=-1, keepdims=True)

    pos_count = tf.reduce_sum(mask)
    box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss_per_cell * mask), pos_count)

    total_loss = cls_loss + lambda_box * box_loss
    return total_loss, cls_loss, box_loss

def train_augmented():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    out_dir = "/home/bim/Projects/EAI-2026/warmup/weights/micro_fomo"
    os.makedirs(out_dir, exist_ok=True)

    train_samples = load_raw_samples(train_split, labels_dir)
    val_samples = load_raw_samples(val_split, labels_dir)
    print(f"[*] Loaded {len(train_samples)} train samples, {len(val_samples)} val samples.")

    model = build_augmented_fomo_bbox(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), alpha=0.5)
    print(f"[✔] Model parameters: {model.count_params():,} (Limit: 200,000)")

    batch_size = 16
    train_steps = int(np.ceil(len(train_samples) / batch_size))
    val_steps = int(np.ceil(len(val_samples) / batch_size))

    train_gen = data_generator(train_samples, batch_size=batch_size, is_train=True)
    val_gen = data_generator(val_samples, batch_size=batch_size, is_train=False)

    # Optimization
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=60 * train_steps,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    best_val_loss = float("inf")
    best_weights_path = os.path.join(out_dir, "best_augmented.weights.h5")

    epochs = 60
    print(f"[*] Training FOMO-BBox with Data Augmentation for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        tr_total, tr_cls, tr_box = 0.0, 0.0, 0.0
        for _ in range(train_steps):
            x_b, yc_b, yb_b, ym_b = next(train_gen)
            with tf.GradientTape() as tape:
                pred_c, pred_b = model(x_b, training=True)
                loss, c_loss, b_loss = compute_loss(yc_b, yb_b, ym_b, pred_c, pred_b)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            tr_total += float(loss)
            tr_cls += float(c_loss)
            tr_box += float(b_loss)

        # Validation evaluation
        va_total, va_cls, va_box = 0.0, 0.0, 0.0
        for _ in range(val_steps):
            vx, vyc, vyb, vym = next(val_gen)
            pred_vc, pred_vb = model(vx, training=False)
            vloss, vc_l, vb_l = compute_loss(vyc, vyb, vym, pred_vc, pred_vb)
            va_total += float(vloss)
            va_cls += float(vc_l)
            va_box += float(vb_l)

        epoch_val_loss = va_total / val_steps
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            model.save_weights(best_weights_path)

        if epoch % 10 == 0 or epoch == epochs or is_best:
            star = " ★ BEST" if is_best else ""
            print(f"[Epoch {epoch:02d}/{epochs}] Train Loss: {tr_total/train_steps:.4f} (cls: {tr_cls/train_steps:.4f}, box: {tr_box/train_steps:.4f}) | Val Loss: {epoch_val_loss:.4f}{star}")

    # Load best weights for export
    print(f"\n[*] Reloading best model weights (val_loss: {best_val_loss:.4f}) for INT8 export...")
    model.load_weights(best_weights_path)

    # Export INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Pre-cache calibration data from training images
    calib_images = []
    for p, _ in train_samples[:50]:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            calib_images.append(img.astype(np.float32) / 255.0)

    def rep_gen():
        for img in calib_images:
            yield [np.expand_dims(img, axis=0)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()

    tflite_path = os.path.join(out_dir, "fomo_bbox_augmented_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024.0
    print(f"[✔] Successfully exported {tflite_path}!")
    print(f"[✔] Size: {size_kb:.2f} KB (Limit: 1,500 KB)")
    return tflite_path

if __name__ == "__main__":
    train_augmented()
