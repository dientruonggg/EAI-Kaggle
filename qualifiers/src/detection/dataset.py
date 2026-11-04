import os
import cv2
import numpy as np
import tensorflow as tf

ORIG_W = 320.0
ORIG_H = 240.0
NUM_CLASSES = 7

class QualifierDetectionDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence dataset loader for Qualifier Object Detection.
    Loads images and YOLO labels, maps them to grid targets:
      - cls_target: (batch, grid, grid, NUM_CLASSES + 1)
      - box_target: (batch, grid, grid, 4) -> [cx_offset, cy_offset, w, h]
    """
    def __init__(self, image_paths, labels_dir, img_size=160, grid_size=20,
                 batch_size=16, augment=True, num_classes=NUM_CLASSES):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.augment = augment
        self.num_classes = num_classes
        self.indices = np.arange(len(self.image_paths))
        if self.augment:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs = []
        batch_cls = []
        batch_box = []

        for b_i in batch_idx:
            img_path = self.image_paths[b_i]
            sample_id = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, f"{sample_id}.txt")

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((int(ORIG_H), int(ORIG_W), 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Read labels
            boxes = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            if 0 <= cid < self.num_classes:
                                cx, cy, w, h = map(float, parts[1:5])
                                boxes.append((cid, cx, cy, w, h))

            # Augmentation (Horizontal Flip)
            flip = self.augment and (np.random.rand() > 0.5)
            if flip:
                img = np.fliplr(img)
                boxes = [(cid, 1.0 - cx, cy, w, h) for (cid, cx, cy, w, h) in boxes]

            # Resize to model input shape
            resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            norm_img = resized.astype(np.float32) / 255.0

            # Target grids
            cls_grid = np.zeros((self.grid_size, self.grid_size, self.num_classes + 1), dtype=np.float32)
            cls_grid[:, :, 0] = 1.0  # Background class by default
            box_grid = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)

            for cid, cx, cy, w, h in boxes:
                # Find grid cell
                gx = int(cx * self.grid_size)
                gy = int(cy * self.grid_size)
                gx = min(max(gx, 0), self.grid_size - 1)
                gy = min(max(gy, 0), self.grid_size - 1)

                # Offset within cell
                offset_x = (cx * self.grid_size) - gx
                offset_y = (cy * self.grid_size) - gy

                # One-hot class (0 is bg, cid+1 is foreground)
                cls_grid[gy, gx, 0] = 0.0
                cls_grid[gy, gx, cid + 1] = 1.0
                box_grid[gy, gx] = [offset_x, offset_y, w, h]

            batch_imgs.append(norm_img)
            batch_cls.append(cls_grid)
            batch_box.append(box_grid)

        return np.array(batch_imgs, dtype=np.float32), {
            "cls_output": np.array(batch_cls, dtype=np.float32),
            "box_output": np.array(batch_box, dtype=np.float32)
        }
