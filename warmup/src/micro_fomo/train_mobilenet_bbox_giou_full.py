import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import json
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

def load_full_bbox_giou_data(images_dir, labels_dir, img_size=(128, 128)):
    grid_size = img_size[0] // 8  # 16
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        
    X = []
    Y_cls = []
    Y_box = []
    
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_file = os.path.join(labels_dir, f"{base}.txt")
        
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        target_cls = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
        target_cls[:, :, 0] = 1.0  # Background
        
        target_box = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                        
                        target_cls[gy, gx, 0] = 0.0
                        target_cls[gy, gx, cls_id + 1] = 1.0
                        target_box[gy, gx] = [1.0, cx, cy, w, h]
                        
        X.append(img_norm)
        Y_cls.append(target_cls)
        Y_box.append(target_box)
        
    return np.array(X, dtype=np.float32), np.array(Y_cls, dtype=np.float32), np.array(Y_box, dtype=np.float32)

def build_mobilenet_dual_head(fomo_weights_path=None, img_size=128, alpha=0.50):
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
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name=f"MobileNetV2_{alpha}_DualHead_128_Full")
    
    if fomo_weights_path and os.path.exists(fomo_weights_path):
        print(f"[*] Transferring pretrained weights from: {fomo_weights_path}")
        fomo_m = tf.keras.models.load_model(fomo_weights_path, compile=False)
        for layer in model.layers:
            if layer.name not in ["box_output", "cls_output"]:
                try:
                    fomo_layer = fomo_m.get_layer(layer.name)
                    layer.set_weights(fomo_layer.get_weights())
                except Exception:
                    pass
        try:
            model.get_layer("cls_output").set_weights(fomo_m.get_layer("fomo_output").get_weights())
            print("[✔] Successfully transferred backbone + cls_output weights!")
        except Exception as e:
            print(f"[!] Warning transferring cls_output: {e}")
            
    return model

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

def train_full_and_export():
    images_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/images"
    labels_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    fomo_pretrained = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_fomo_128/model_best.keras"
    save_dir = "/home/bim/Projects/EAI-2026/warmup/weights/mobilenet_bbox_giou_128_full"
    os.makedirs(save_dir, exist_ok=True)
    
    print("[*] Loading All 140 Training Images with GIoU Targets...")
    X_train, Y_cls_train, Y_box_train = load_full_bbox_giou_data(images_dir, labels_dir, img_size=(128, 128))
    print(f"[*] Dataset: {X_train.shape[0]} images loaded.")
    
    model = build_mobilenet_dual_head(fomo_weights_path=fomo_pretrained, img_size=128, alpha=0.50)
    print(f"[*] Total Model Parameters: {model.count_params():,}")
    
    focal, giou_l = make_losses()
    
    # Phase 1: Train ONLY box_output head (15 epochs)
    print("\n--- PHASE 1: WARMUP DYNAMIC BBOX HEAD (15 epochs) ---")
    for layer in model.layers:
        if layer.name == "box_output":
            layer.trainable = True
        else:
            layer.trainable = False
            
    model.compile(
        optimizer=optimizers.Adam(learning_rate=2e-3),
        loss={"cls_output": focal, "box_output": giou_l},
        loss_weights={"cls_output": 0.0, "box_output": 1.0}
    )
    
    model.fit(
        X_train, {"cls_output": Y_cls_train, "box_output": Y_box_train},
        epochs=15,
        batch_size=16,
        verbose=1
    )
    
    # Phase 2: Joint fine-tuning (backbone unfreeze from block_6, 35 epochs)
    print("\n--- PHASE 2: JOINT FINE-TUNING ON ALL 140 IMAGES (35 epochs) ---")
    for layer in model.layers:
        if any(b in layer.name for b in ["block_0", "block_1", "block_2", "block_3", "block_4", "block_5", "Conv1"]):
            layer.trainable = False
        else:
            layer.trainable = True
            
    tflite_path = os.path.join(save_dir, "mobilenet_bbox_giou_128_full_int8.tflite")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1.5e-4),
        loss={"cls_output": focal, "box_output": giou_l},
        loss_weights={"cls_output": 1.0, "box_output": 3.0}
    )
    
    model.fit(
        X_train, {"cls_output": Y_cls_train, "box_output": Y_box_train},
        epochs=35,
        batch_size=16,
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting INT8 quantized TFLite...")
    export_dual_head_tflite(model, X_train[:50], tflite_path)

if __name__ == "__main__":
    train_full_and_export()
