import os
import sys
import glob
import json
import random
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, models

print(f"[*] TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"[*] Available GPUs: {gpus}")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"[✔] MirroredStrategy initialized across {strategy.num_replicas_in_sync} GPUs!")
else:
    strategy = tf.distribute.get_strategy()
    print(f"[✔] Running on single device: {gpus[0] if gpus else 'CPU'}")

ORIG_W = 1672.0
ORIG_H = 941.0
NUM_CLASSES = 10
GRID_SIZE = 20
IMG_SIZE = 160
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

grid_x = np.tile(np.arange(GRID_SIZE, dtype=np.float32), (GRID_SIZE, 1))
grid_y = np.tile(np.arange(GRID_SIZE, dtype=np.float32)[:, None], (1, GRID_SIZE))
tf_gx = tf.constant(grid_x[None, :, :, None], dtype=tf.float32)
tf_gy = tf.constant(grid_y[None, :, :, None], dtype=tf.float32)

def locate_data():
    input_base = "/kaggle/input"
    comp_dirs = glob.glob(os.path.join(input_base, "*"))
    print(f"[*] Found input directories: {comp_dirs}")
    
    train_imgs = []
    labels_dir = None
    
    for c_dir in comp_dirs:
        possible_imgs = glob.glob(os.path.join(c_dir, "**", "train", "images", "*.png"), recursive=True) + \
                        glob.glob(os.path.join(c_dir, "**", "train", "images", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(c_dir, "**", "train", "*.png"), recursive=True)
        if possible_imgs:
            train_imgs = sorted(possible_imgs)
            # Find labels directory
            for root, dirs, files in os.walk(c_dir):
                if "labels" in dirs:
                    labels_dir = os.path.join(root, "labels")
                    break
            print(f"[✔] Detected {len(train_imgs)} images and labels in {c_dir}")
            break
            
    if not train_imgs:
        raise FileNotFoundError(f"Could not find training images in {input_base}!")
    return train_imgs, labels_dir

def build_model(img_size=160, alpha=0.50):
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
    
    model = models.Model(inputs=base.inputs, outputs={"cls_output": cls_out, "box_output": box_out}, name="MobileNetV2_EdgeAI_160")
    return model

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        focal_weight = alpha_t * tf.pow(1.0 - pt, gamma)
        bce = -tf.math.log(pt)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * bce, axis=-1))
    return loss

def eiou_loss():
    def loss(y_true, y_pred):
        obj_mask = y_true[..., 4:5]
        
        pred_dx = y_pred[..., 0:1]
        pred_dy = y_pred[..., 1:2]
        pred_w = tf.clip_by_value(y_pred[..., 2:3], 1e-4, 1.0)
        pred_h = tf.clip_by_value(y_pred[..., 3:4], 1e-4, 1.0)
        
        pred_cx = (tf_gx + pred_dx) / float(GRID_SIZE)
        pred_cy = (tf_gy + pred_dy) / float(GRID_SIZE)
        
        true_dx = y_true[..., 0:1]
        true_dy = y_true[..., 1:2]
        true_w = tf.clip_by_value(y_true[..., 2:3], 1e-4, 1.0)
        true_h = tf.clip_by_value(y_true[..., 3:4], 1e-4, 1.0)
        
        true_cx = (tf_gx + true_dx) / float(GRID_SIZE)
        true_cy = (tf_gy + true_dy) / float(GRID_SIZE)
        
        b1_x1 = pred_cx - pred_w / 2.0
        b1_y1 = pred_cy - pred_h / 2.0
        b1_x2 = pred_cx + pred_w / 2.0
        b1_y2 = pred_cy + pred_h / 2.0
        
        b2_x1 = true_cx - true_w / 2.0
        b2_y1 = true_cy - true_h / 2.0
        b2_x2 = true_cx + true_w / 2.0
        b2_y2 = true_cy + true_h / 2.0
        
        inter_x1 = tf.maximum(b1_x1, b2_x1)
        inter_y1 = tf.maximum(b1_y1, b2_y1)
        inter_x2 = tf.minimum(b1_x2, b2_x2)
        inter_y2 = tf.minimum(b1_y2, b2_y2)
        inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
        inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        b1_area = pred_w * pred_h
        b2_area = true_w * true_h
        union_area = b1_area + b2_area - inter_area + 1e-7
        iou = inter_area / union_area
        
        enc_x1 = tf.minimum(b1_x1, b2_x1)
        enc_y1 = tf.minimum(b1_y1, b2_y1)
        enc_x2 = tf.maximum(b1_x2, b2_x2)
        enc_y2 = tf.maximum(b1_y2, b2_y2)
        c_w = enc_x2 - enc_x1 + 1e-7
        c_h = enc_y2 - enc_y1 + 1e-7
        c2 = tf.square(c_w) + tf.square(c_h)
        
        rho2_center = tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy)
        rho2_w = tf.square(pred_w - true_w)
        rho2_h = tf.square(pred_h - true_h)
        
        eiou = iou - (rho2_center / c2) - (rho2_w / tf.square(c_w)) - (rho2_h / tf.square(c_h))
        loss_val = 1.0 - eiou
        
        num_objs = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        return tf.reduce_sum(loss_val * obj_mask) / num_objs
    return loss

def encode_targets(boxes, grid_size=GRID_SIZE):
    cls_target = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
    cls_target[..., 0] = 1.0
    
    box_target = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
    sorted_boxes = sorted(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]), reverse=True)
    
    for cid, xmin, ymin, xmax, ymax in sorted_boxes:
        bw = xmax - xmin
        bh = ymax - ymin
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        gx_f = cx * grid_size
        gy_f = cy * grid_size
        gx = int(np.clip(gx_f, 0, grid_size - 1))
        gy = int(np.clip(gy_f, 0, grid_size - 1))
        
        dx = float(gx_f - gx)
        dy = float(gy_f - gy)
        
        cls_target[gy, gx, 0] = 0.0
        cls_target[gy, gx, cid + 1] = 1.0
        
        box_target[gy, gx, 0] = dx
        box_target[gy, gx, 1] = dy
        box_target[gy, gx, 2] = bw
        box_target[gy, gx, 3] = bh
        box_target[gy, gx, 4] = 1.0
        
    return cls_target, box_target

class FastDataset(tf.keras.utils.Sequence):
    def __init__(self, samples, batch_size=32, img_size=160, grid_size=20, augment=True):
        self.samples = samples
        self.batch_size = batch_size
        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment
        self.indices = np.arange(len(self.samples))
        
    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        Y_cls = []
        Y_box = []
        
        for i in batch_idx:
            img_p, boxes = self.samples[i]
            img = cv2.imread(img_p)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.augment and random.random() < 0.5:
                hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
                hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
                img_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
            resized = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            
            c_tar, b_tar = encode_targets(boxes, self.grid_size)
            X.append(norm)
            Y_cls.append(c_tar)
            Y_box.append(b_tar)
            
        return np.array(X, dtype=np.float32), {"cls_output": np.array(Y_cls, dtype=np.float32), "box_output": np.array(Y_box, dtype=np.float32)}

def export_tflite(model, val_samples, output_path):
    print(f"[*] Exporting Full INT8 TFLite model to {output_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def rep_gen():
        rep_count = min(100, len(val_samples))
        for i in range(rep_count):
            img_p, _ = val_samples[i]
            img = cv2.imread(img_p)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                norm = resized.astype(np.float32) / 255.0
                yield [np.expand_dims(norm, axis=0)]
                
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[✔] Successfully exported INT8 TFLite: {output_path} ({len(tflite_model)/1024:.1f} KB)")

def main():
    print("=== HIGH-SPEED T4x2 TRAINING PIPELINE STARTED ===")
    start_time = time.time()
    
    train_imgs, labels_dir = locate_data()
    samples = []
    
    for img_p in train_imgs:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(labels_dir, f"{stem}.txt") if labels_dir else None
        boxes = []
        if lbl_p and os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        cid = int(p[0])
                        cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                        boxes.append((cid, cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0))
        samples.append((img_p, boxes))
        
    random.seed(42)
    random.shuffle(samples)
    val_split = max(20, int(len(samples) * 0.15))
    val_samples = samples[:val_split]
    train_samples = samples[val_split:]
    print(f"[✔] Prepared {len(train_samples)} Train samples, {len(val_samples)} Val samples.")
    
    batch_size = 32 * max(1, len(gpus))
    train_gen = FastDataset(train_samples, batch_size=batch_size, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=True)
    val_gen = FastDataset(val_samples, batch_size=batch_size, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=False)
    
    with strategy.scope():
        model = build_model(img_size=IMG_SIZE, alpha=0.50)
        print(f"[✔] Model Parameters: {model.count_params():,} (Limit: 200,000)")
        
        focal = focal_loss(gamma=2.0, alpha=0.25)
        eiou_l = eiou_loss()
        
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=3e-4,
            decay_steps=40 * len(train_gen),
            alpha=0.05
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss={"cls_output": focal, "box_output": eiou_l},
            loss_weights={"cls_output": 1.0, "box_output": 3.0}
        )
        
    best_keras = os.path.join(OUTPUT_DIR, "model_best.keras")
    tflite_path = os.path.join(OUTPUT_DIR, "mobilenet_bbox_160_int8.tflite")
    
    ckpt = callbacks.ModelCheckpoint(
        filepath=best_keras,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    epochs = 40
    print(f"[*] Starting Multi-GPU Training for {epochs} epochs (Batch Size: {batch_size})...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[ckpt],
        verbose=1
    )
    
    print("\n[✔] Training completed! Exporting best checkpoint...")
    best_model = tf.keras.models.load_model(best_keras, compile=False)
    export_tflite(best_model, val_samples, tflite_path)
    
    elapsed = time.time() - start_time
    print(f"\n🎉 PIPELINE FINISHED IN {elapsed/60:.2f} MINUTES! Output ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
