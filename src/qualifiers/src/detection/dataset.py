import os
import sys
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_tools.balance_and_occlude_dataset import apply_random_erasing_occlusion

ORIG_W = 320.0
ORIG_H = 240.0
NUM_CLASSES = 7


def apply_photometric_jitter(image, prob=0.6, h_range=(-8, 8), s_range=(0.75, 1.25), v_range=(0.75, 1.25)):
    """
    Applies photometric HSV color jitter to RGB image.
    Prevents false associations between surface color and object identity
    (resolving phone vs book, mouse vs toy_car confusion).
    """
    if np.random.rand() > prob or image is None or image.size == 0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Hue shift (wrapped mod 180)
    dh = np.random.uniform(h_range[0], h_range[1])
    hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180.0

    # Saturation scale
    ds = np.random.uniform(s_range[0], s_range[1])
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * ds, 0.0, 255.0)

    # Brightness / Value scale
    dv = np.random.uniform(v_range[0], v_range[1])
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * dv, 0.0, 255.0)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_tabletop_perspective(image, boxes, pitch_range=(30.0, 60.0), prob=0.5):
    """
    Simulates tabletop perspective warp (30°-60° camera pitch).
    Warps all 4 corners of each bounding box using cv2.perspectiveTransform
    and computes the enclosing axis-aligned bounding box.
    Clamps coordinates strictly within [0, 1] with zero inverted coordinates.
    Filters out degenerate boxes smaller than 4 pixels in width or height.

    Args:
        image: np.ndarray, input image (H, W, C) in uint8 or float32.
        boxes: list of tuples (cid, cx, cy, w, h) in normalized YOLO format [0, 1].
        pitch_range: tuple of (min_pitch_deg, max_pitch_deg).
        prob: float, probability of applying the transformation.

    Returns:
        warped_img: np.ndarray of same shape and dtype as image.
        warped_boxes: list of tuples (cid, cx, cy, w, h) in normalized format [0, 1].
    """
    if image is None or image.size == 0 or len(boxes) == 0 or np.random.rand() > prob:
        return image, boxes

    img_h, img_w = image.shape[:2]
    pitch = np.random.uniform(pitch_range[0], pitch_range[1])
    pitch_norm = (pitch - pitch_range[0]) / max(1e-5, (pitch_range[1] - pitch_range[0]))

    # Perspective parameters: top edge squeeze and downward shift
    top_squeeze = 0.15 + 0.20 * pitch_norm
    top_y_shift = 0.05 + 0.15 * pitch_norm
    roll_skew = np.random.uniform(-0.03, 0.03)

    src = np.array([
        [0.0, 0.0],
        [float(img_w), 0.0],
        [float(img_w), float(img_h)],
        [0.0, float(img_h)]
    ], dtype=np.float32)

    dst = np.array([
        [(top_squeeze + roll_skew) * float(img_w), top_y_shift * float(img_h)],
        [(1.0 - top_squeeze + roll_skew) * float(img_w), top_y_shift * float(img_h)],
        [float(img_w), float(img_h)],
        [0.0, float(img_h)]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, M, (img_w, img_h), borderMode=cv2.BORDER_REFLECT101)

    warped_boxes = []
    for cid, cx, cy, bw, bh in boxes:
        # Convert normalized YOLO (cx, cy, bw, bh) to absolute pixel corners
        x1 = (cx - bw / 2.0) * float(img_w)
        y1 = (cy - bh / 2.0) * float(img_h)
        x2 = (cx + bw / 2.0) * float(img_w)
        y2 = (cy + bh / 2.0) * float(img_h)

        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32).reshape(4, 1, 2)

        p_warped = cv2.perspectiveTransform(corners, M).reshape(4, 2)

        if not np.all(np.isfinite(p_warped)):
            continue

        # Axis-aligned bounding box enclosing all 4 warped corners
        wx1 = float(np.min(p_warped[:, 0]))
        wy1 = float(np.min(p_warped[:, 1]))
        wx2 = float(np.max(p_warped[:, 0]))
        wy2 = float(np.max(p_warped[:, 1]))

        # Guarantee non-inversion
        if wx2 < wx1:
            wx1, wx2 = wx2, wx1
        if wy2 < wy1:
            wy1, wy2 = wy2, wy1

        # Clamping coordinates strictly within image bounds [0, img_w / img_h]
        wx1 = max(0.0, min(wx1, float(img_w)))
        wy1 = max(0.0, min(wy1, float(img_h)))
        wx2 = max(0.0, min(wx2, float(img_w)))
        wy2 = max(0.0, min(wy2, float(img_h)))

        wbw = wx2 - wx1
        wbh = wy2 - wy1

        # Filter out degenerate boxes smaller than 4 pixels in width or height
        if wbw >= 4.0 and wbh >= 4.0:
            wcx = (wx1 + wx2) / (2.0 * float(img_w))
            wcy = (wy1 + wy2) / (2.0 * float(img_h))
            new_bw = wbw / float(img_w)
            new_bh = wbh / float(img_h)

            # Strict normalization clamping to [0.0, 1.0]
            wcx = max(0.0, min(1.0, float(wcx)))
            wcy = max(0.0, min(1.0, float(wcy)))
            new_bw = max(0.0, min(1.0, float(new_bw)))
            new_bh = max(0.0, min(1.0, float(new_bh)))

            warped_boxes.append((cid, wcx, wcy, new_bw, new_bh))

    return warped_img, warped_boxes


class QualifierDetectionDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence dataset loader for 12-Layer Decoupled Object Detection:
      - conf_output: (batch, grid, grid, 1) -> objectness (1.0 = object, 0.0 = bg)
      - cls_output:  (batch, grid, grid, 8) -> [7 classes one-hot, 1 conf_mask]
      - box_output:  (batch, grid, grid, 5) -> [offset_x, offset_y, w, h, conf_mask]
    """
    def __init__(self, image_paths, labels_dir, img_size=160, grid_size=20,
                 batch_size=16, augment=True, augment_occlusion=False,
                 augment_perspective=True, augment_photometric=True,
                 num_classes=NUM_CLASSES):
        self.image_paths = image_paths
        self.labels_dir = labels_dir

        if isinstance(img_size, (tuple, list)):
            self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
            self.img_size = img_size
        else:
            self.img_h, self.img_w = int(img_size), int(img_size)
            self.img_size = int(img_size)

        if isinstance(grid_size, (tuple, list)):
            self.grid_h, self.grid_w = int(grid_size[0]), int(grid_size[1])
            self.grid_size = grid_size
        elif grid_size is None:
            self.grid_h = self.img_h // 8
            self.grid_w = self.img_w // 8
            self.grid_size = (self.grid_h, self.grid_w)
        else:
            self.grid_h = int(grid_size)
            self.grid_w = int(grid_size)
            self.grid_size = int(grid_size)

        self.batch_size = batch_size
        self.augment = augment
        self.augment_occlusion = augment_occlusion
        self.augment_perspective = augment_perspective
        self.augment_photometric = augment_photometric
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
        batch_conf = []
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

            # Augmentation Stage 3: Tabletop Perspective Warp
            if self.augment and self.augment_perspective and len(boxes) > 0:
                img, boxes = apply_tabletop_perspective(img, boxes, prob=0.5)

            # Augmentation Stage 2: Tabletop Occlusion (only if explicitly enabled)
            if self.augment and self.augment_occlusion and len(boxes) > 0:
                img = apply_random_erasing_occlusion(img, boxes)

            # Augmentation Stage 1.5: Photometric Color Jitter (HSV)
            if self.augment and self.augment_photometric:
                img = apply_photometric_jitter(img, prob=0.6)

            # Augmentation Stage 1: Horizontal Flip
            flip = self.augment and (np.random.rand() > 0.5)
            if flip:
                img = np.fliplr(img)
                boxes = [(cid, 1.0 - cx, cy, w, h) for (cid, cx, cy, w, h) in boxes]

            # Resize to model input shape (cv2.resize expects (width, height))
            if img.shape[0] != self.img_h or img.shape[1] != self.img_w:
                resized = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            else:
                resized = img
            norm_img = resized.astype(np.float32) / 255.0

            # Target grids: (grid_h, grid_w, channels)
            conf_grid = np.zeros((self.grid_h, self.grid_w, 1), dtype=np.float32)
            cls_grid = np.zeros((self.grid_h, self.grid_w, self.num_classes + 1), dtype=np.float32)
            box_grid = np.zeros((self.grid_h, self.grid_w, 5), dtype=np.float32)

            for cid, cx, cy, w, h in boxes:
                gx = int(cx * self.grid_w)
                gy = int(cy * self.grid_h)
                gx = min(max(gx, 0), self.grid_w - 1)
                gy = min(max(gy, 0), self.grid_h - 1)

                offset_x = (cx * self.grid_w) - gx
                offset_y = (cy * self.grid_h) - gy

                conf_grid[gy, gx, 0] = 1.0
                cls_grid[gy, gx, cid] = 1.0
                cls_grid[gy, gx, self.num_classes] = 1.0  # conf_mask
                box_grid[gy, gx] = [offset_x, offset_y, w, h, 1.0]

            batch_imgs.append(norm_img)
            batch_conf.append(conf_grid)
            batch_cls.append(cls_grid)
            batch_box.append(box_grid)

        return np.array(batch_imgs, dtype=np.float32), {
            "conf_output": np.array(batch_conf, dtype=np.float32),
            "cls_output": np.array(batch_cls, dtype=np.float32),
            "box_output": np.array(batch_box, dtype=np.float32)
        }
