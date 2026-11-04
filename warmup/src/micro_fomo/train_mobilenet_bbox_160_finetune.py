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
IMG_SIZE = 160
GRID_SIZE = 20  # 160 // 8

grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float32), (GRID_SIZE, 1))
grid_y = np.tile(np.arange(GRID_SIZE, dtype=np.float32)[:, None], (1, GRID_SIZE))
tf_gx = tf.constant(grid_x[None, :, :, None], dtype=tf.float32)
tf_gy = tf.constant(grid_y[None, :, :, None], dtype=tf.float32)

def build_model_160(pretrained_128_path=None, alpha=0.50):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=alpha,
        include_top=False,
        weights=None
    )
    
    feat_stride8 = base.get_layer("block_5_add").output   # 20x20x16
    feat_stride16 = base.get_layer("block_12_add").output # 10x10x32
    
    up = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="fpn_upsample")(feat_stride16)
    fused = layers.Concatenate(name="fpn_fused")([feat_stride8, up])
    
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="fomo_refine")(fused)
    x = layers.BatchNormalization(name="fomo_refine_bn")(x)
    x = layers.ReLU(6.0, name="fomo_refine_relu")(x)
    
    cls_out = layers.Conv2D(NUM_CLASSES + 1, 1, padding="same", activation="softmax", name="cls_output")(x)
    box_out = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="box_output")(x)
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_Augmented_160")
    
    if pretrained_128_path and os.path.exists(pretrained_128_path):
        print(f"[*] Progressive Transfer: Loading trained weights from 128x128 checkpoint: {pretrained_128_path}")
        src = tf.keras.models.load_model(pretrained_128_path, compile=False)
        transferred = 0
        for l in model.layers:
            try:
                src_layer = src.get_layer(l.name)
                w = src_layer.get_weights()
                if len(w) > 0:
                    l.set_weights(w)
                    transferred += 1
            except Exception:
                pass
        print(f"[✔] Successfully transferred weights for {transferred} layers to 160x160 architecture!")
        
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
    
    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        flipped = []
        for cid, cx, cy, w, h, area in aug_boxes:
            flipped.append((cid, 1.0 - cx, cy, w, h, area))
        aug_boxes = flipped
        
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    if random.random() < 0.5:
        scale = random.uniform(0.96, 1.04)
        h, w = aug_img.shape[:2]
        nw, nh = int(w * scale), int(h * scale)
        res = cv2.resize(aug_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if scale > 1.0:
            x1 = (nw - w) // 2
            y1 = (nh - h) // 2
            aug_img = res[y1:y1+h, x1:x1+w]
        else:
            aug_img = np.zeros_like(img)
            x1 = (w - nw) // 2
            y1 = (h - nh) // 2
            aug_img[y1:y1+nh, x1:x1+nw] = res
            
    return aug_img, aug_boxes

def encode_targets_160(img, boxes):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    
    target_cls = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CLASSES + 1), dtype=np.float32)
    target_cls[:, :, 0] = 1.0  # Background
    target_box = np.zeros((GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
    
    # Sort boxes descending by area so smaller objects overwrite larger ones
    sorted_boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    
    for cid, cx, cy, w, h, _ in sorted_boxes:
        gx = min(GRID_SIZE - 1, max(0, int(cx * GRID_SIZE)))
        gy = min(GRID_SIZE - 1, max(0, int(cy * GRID_SIZE)))
        
        target_cls[gy, gx, 0] = 0.0
        target_cls[gy, gx, 1:] = 0.0
        target_cls[gy, gx, cid + 1] = 1.0
        
        target_box[gy, gx] = [1.0, cx, cy, w, h]
        
    return img_norm, target_cls, target_box

class AugmentedDataGen160(tf.keras.utils.Sequence):
    def __init__(self, samples, batch_size=16, is_train=True):
        self.samples = samples
        self.batch_size = batch_size
        self.is_train = is_train
        self.indices = list(range(len(samples)))
        if self.is_train:
            random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))
        
    def on_epoch_end(self):
        if self.is_train:
            random.shuffle(self.indices)
            
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = []
        Y_cls = []
        Y_box = []
        
        for i in batch_idx:
            img, boxes = self.samples[i]
            if self.is_train:
                img_aug, boxes_aug = augment_sample(img, boxes)
            else:
                img_aug, boxes_aug = img, boxes
                
            norm_img, t_cls, t_box = encode_targets_160(img_aug, boxes_aug)
            X.append(norm_img)
            Y_cls.append(t_cls)
            Y_box.append(t_box)
            
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

        pred_cx = (tf_gx + pred_dx) / float(GRID_SIZE)
        pred_cy = (tf_gy + pred_dy) / float(GRID_SIZE)

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
        cw = tf.maximum(0.0, cx2 - cx1)
        ch = tf.maximum(0.0, cy2 - cy1)
        c_area = cw * ch + 1e-7

        giou = iou - ((c_area - union) / c_area)
        giou_loss = 1.0 - giou

        l1_loss = (tf.abs(pred_cx - gt_cx) + tf.abs(pred_cy - gt_cy) +
                   tf.abs(pred_w - gt_w) + tf.abs(pred_h - gt_h))

        total_cell_loss = is_obj * (giou_loss + 0.5 * l1_loss)
        num_pos = tf.reduce_sum(is_obj) + 1e-7
        return tf.reduce_sum(total_cell_loss) / num_pos

    return focal_loss, giou_box_loss

def export_dual_head_tflite_160(model, rep_data, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def rep_gen():
        for x in rep_data:
            yield [np.expand_dims(x, axis=0).astype(np.float32)]
            
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant = converter.convert()
    size_kb = len(tflite_quant) / 1024.0
    print(f"[✔] 160x160 Dual-head INT8 model exported: {size_kb:.2f} KB (Limit: 1500 KB)")
    with open(out_path, "wb") as f:
        f.write(tflite_quant)
    print(f"[✔] Saved to: {out_path}")
    return tflite_quant

def train():
    train_split = "/home/bim/Projects/EAI-2026/warmup/data/train_split.txt"
    val_split = "/home/bim/Projects/EAI-2026/warmup/data/val_split.txt"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    pretrained = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_128/model_best.keras"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_augmented_160"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading Raw Samples for 160x160 Progressive Fine-Tuning...")
    train_samples = load_samples(train_split, labels_dir)
    val_samples = load_samples(val_split, labels_dir)
    
    train_gen = AugmentedDataGen160(train_samples, batch_size=16, is_train=True)
    val_gen = AugmentedDataGen160(val_samples, batch_size=16, is_train=False)
    
    model = build_model_160(pretrained_128_path=pretrained, alpha=0.50)
    print(f"[*] Total Model Parameters: {model.count_params():,}")
    
    focal, giou_l = make_losses()
    
    # Trainable from block_4 onwards
    for layer in model.layers:
        if any(b in layer.name for b in ["block_0", "block_1", "block_2", "block_3", "Conv1"]):
            layer.trainable = False
        else:
            layer.trainable = True
            
    best_path = os.path.join(save_dir, "model_best.keras")
    tflite_path = os.path.join(save_dir, "mobilenet_bbox_augmented_160_int8.tflite")
    
    cb = [
        callbacks.ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # Small learning rate 8e-5 for smooth resolution adaptation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=8e-5),
        loss={"cls_output": focal, "box_output": giou_l},
        loss_weights={"cls_output": 1.0, "box_output": 3.0}
    )
    
    print("\n--- PROGRESSIVE FINE-TUNING AT 160x160 (40 epochs) ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=cb,
        verbose=1
    )
    
    print("\n[✔] 160x160 Training completed! Exporting INT8 quantized TFLite...")
    best_model = tf.keras.models.load_model(best_path, custom_objects={"focal_loss": focal, "giou_box_loss": giou_l})
    
    rep_X = [cv2.resize(s[0], (160, 160)).astype(np.float32) / 255.0 for s in val_samples]
    export_dual_head_tflite_160(best_model, rep_X, tflite_path)

if __name__ == "__main__":
    train()
