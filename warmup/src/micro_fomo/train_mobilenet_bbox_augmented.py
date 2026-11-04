import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models

ORIG_W = 1672.0
ORIG_H = 941.0
NUM_CLASSES = 10
GRID_SIZE = 16

grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float32), (GRID_SIZE, 1))
grid_y = np.tile(np.arange(GRID_SIZE, dtype=np.float32)[:, None], (1, GRID_SIZE))
tf_gx = tf.constant(grid_x[None, :, :, None], dtype=tf.float32)
tf_gy = tf.constant(grid_y[None, :, :, None], dtype=tf.float32)

def build_model(pretrained_path=None, img_size=128, alpha=0.50):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )
    
    feat_stride8 = base.get_layer("block_5_add").output
    feat_stride16 = base.get_layer("block_12_add").output
    
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="fomo_refine")(fused)
    x = layers.BatchNormalization(name="fomo_refine_bn")(x)
    x = layers.ReLU(6.0, name="fomo_refine_relu")(x)
    
    cls_out = layers.Conv2D(NUM_CLASSES + 1, 1, padding="same", activation="softmax", name="cls_output")(x)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_Augmented")
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[*] Loading pretrained weights from: {pretrained_path}")
        src = tf.keras.models.load_model(pretrained_path, compile=False)
        for l in model.layers:
            try:
                l.set_weights(src.get_layer(l.name).get_weights())
            except Exception:
                pass
        print("[✔] Successfully transferred weights!")
        
    return model

def load_samples(split_file, labels_dir):
    with open(split_file, "r") as f:
        img_paths = [l.strip() for l in f if l.strip()]
        
    samples = []
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_file = os.path.join(labels_dir, f"{base}.txt")
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cid, cx, cy, w, h, w * h))
        samples.append((img, boxes))
    return samples

def augment_sample(img, boxes):
    aug_img = img.copy()
    aug_boxes = list(boxes)
    
    # 1. Random horizontal flip (p=0.5)
    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        flipped = []
        for cid, cx, cy, w, h, area in aug_boxes:
            flipped.append((cid, 1.0 - cx, cy, w, h, area))
        aug_boxes = flipped
        
    # 2. Random contrast and brightness
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    return aug_img, aug_boxes

def encode_targets(boxes, grid_size=16):
    target_cls = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
    target_cls[:, :, 0] = 1.0
    target_box = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
    
    # Sort boxes by area descending so smaller boxes overwrite larger ones
    sorted_b = sorted(boxes, key=lambda b: b[5], reverse=True)
    for cid, cx, cy, w, h, _ in sorted_b:
        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
        
        target_cls[gy, gx, 0] = 0.0
        target_cls[gy, gx, 1:11] = 0.0
        target_cls[gy, gx, cid + 1] = 1.0
        target_box[gy, gx] = [1.0, cx, cy, w, h]
        
    return target_cls, target_box

class AugmentedDataGen(tf.keras.utils.Sequence):
    def __init__(self, samples, batch_size=16, img_size=(128, 128), is_train=True):
        self.samples = samples
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_train = is_train
        self.indices = list(range(len(samples)))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))
        
    def on_epoch_end(self):
        if self.is_train:
            random.shuffle(self.indices)
            
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        Y_cls = []
        Y_box = []
        
        for i in batch_idx:
            img, boxes = self.samples[i]
            if self.is_train:
                img_aug, boxes_aug = augment_sample(img, boxes)
            else:
                img_aug, boxes_aug = img, boxes
                
            resized = cv2.resize(img_aug, self.img_size, interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            tc, tb = encode_targets(boxes_aug, grid_size=GRID_SIZE)
            
            X.append(norm)
            Y_cls.append(tc)
            Y_box.append(tb)
            
        return np.array(X, dtype=np.float32), {"cls_output": np.array(Y_cls, dtype=np.float32), "box_output": np.array(Y_box, dtype=np.float32)}

def make_losses():
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        is_bg = y_true[:, :, :, 0:1]
        is_fg = 1.0 - is_bg
        weights = is_bg * 0.2 + is_fg * 3.0
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = tf.pow(1.0 - pt, 2.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        return tf.reduce_mean(weights * focal * ce)
        
    def giou_box_loss(y_true, y_pred):
        is_obj = y_true[..., 0:1]
        gt_cx = y_true[..., 1:2]
        gt_cy = y_true[..., 2:3]
        gt_w  = y_true[..., 3:4]
        gt_h  = y_true[..., 4:5]

        pred_dx = y_pred[..., 0:1]
        pred_dy = y_pred[..., 1:2]
        pred_w  = y_pred[..., 2:3]
        pred_h  = y_pred[..., 3:4]

        pred_cx = (tf_gx + pred_dx) / 16.0
        pred_cy = (tf_gy + pred_dy) / 16.0

        px1 = pred_cx - pred_w / 2.0
        py1 = pred_cy - pred_h / 2.0
        px2 = pred_cx + pred_w / 2.0
        py2 = pred_cy + pred_h / 2.0

        tx1 = gt_cx - gt_w / 2.0
        ty1 = gt_cy - gt_h / 2.0
        tx2 = gt_cx + gt_w / 2.0
        ty2 = gt_cy + gt_h / 2.0

        ix1 = tf.maximum(px1, tx1)
        iy1 = tf.maximum(py1, ty1)
        ix2 = tf.minimum(px2, tx2)
        iy2 = tf.minimum(py2, ty2)
        iw = tf.maximum(0.0, ix2 - ix1)
        ih = tf.maximum(0.0, iy2 - iy1)
        inter = iw * ih

        area_p = tf.maximum(0.0, px2 - px1) * tf.maximum(0.0, py2 - py1)
        area_t = tf.maximum(0.0, tx2 - tx1) * tf.maximum(0.0, ty2 - ty1)
        union = area_p + area_t - inter + 1e-7
        iou = inter / union

        cx1 = tf.minimum(px1, tx1)
        cy1 = tf.minimum(py1, ty1)
        cx2 = tf.maximum(px2, tx2)
        cy2 = tf.maximum(py2, ty2)
        c_area = tf.maximum(0.0, cx2 - cx1) * tf.maximum(0.0, cy2 - cy1) + 1e-7

        giou = iou - (c_area - union) / c_area
        giou_l = (1.0 - giou) * is_obj

        l1_loss = (tf.abs(pred_w - gt_w) + tf.abs(pred_h - gt_h)) * is_obj

        total = giou_l + 1.0 * l1_loss
        return tf.reduce_sum(total) / (tf.reduce_sum(is_obj) + 1e-6)
        
    return focal_loss, giou_box_loss

def export_dual_head_tflite(model, rep_data, out_path):
    def rep_gen():
        for sample in rep_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()
    
    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] Dual-head INT8 model exported: {size_kb:.2f} KB (Limit: 1500 KB)")
    with open(out_path, "wb") as f:
        f.write(tflite_quant)
    print(f"[✔] Saved to: {out_path}")
    return tflite_quant

def train():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    pretrained = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_giou_128/model_best.keras"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_128"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading Raw Samples for Online Augmentation...")
    train_samples = load_samples(train_split, labels_dir)
    val_samples = load_samples(val_split, labels_dir)
    
    train_gen = AugmentedDataGen(train_samples, batch_size=16, is_train=True)
    val_gen = AugmentedDataGen(val_samples, batch_size=16, is_train=False)
    
    model = build_model(pretrained_path=pretrained, img_size=128, alpha=0.50)
    print(f"[*] Total Model Parameters: {model.count_params():,}")
    
    focal, giou_l = make_losses()
    
    # Unfreeze backbone from block_4 onwards for deeper adaptation with augmented data
    for layer in model.layers:
        if any(b in layer.name for b in ["block_0", "block_1", "block_2", "block_3", "Conv1"]):
            layer.trainable = False
        else:
            layer.trainable = True
            
    best_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "mobilenet_bbox_augmented_128_int8.tflite")
    
    cb = [
        callbacks.ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    ]
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1.5e-4),
        loss={"cls_output": focal, "box_output": giou_l},
        loss_weights={"cls_output": 1.0, "box_output": 3.0}
    )
    
    print("\n--- TRAINING WITH ONLINE DATA AUGMENTATION (50 epochs) ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=cb,
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting INT8 quantized TFLite...")
    best_model = tf.keras.models.load_model(best_path, custom_objects={"focal_loss": focal, "giou_box_loss": giou_l})
    
    # Representative data from val samples
    rep_X = [cv2.resize(s[0], (128, 128)).astype(np.float32) / 255.0 for s in val_samples]
    export_dual_head_tflite(best_model, rep_X, tflite_path)

if __name__ == "__main__":
    train()
