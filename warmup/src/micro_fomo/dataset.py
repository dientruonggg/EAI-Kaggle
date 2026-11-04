import os
import glob
import cv2
import numpy as np

CLASS_NAMES = [
    "person", "bicycle", "motorcycle", "car", "bus",
    "truck", "traffic_light", "stop_sign", "crossing_warning_sign", "zebra_crossing"
]
NUM_CLASSES = 10
GRID_SIZE = 12

def load_fomo_dataset(images_dir, labels_dir, img_size=(96, 96)):
    """
    Loads images and creates (img_size//8, img_size//8, 11) target grid for Micro-FOMO.
    Target:
      channel 0: background
      channel 1..10: class_id 0..9
    """
    grid_size = img_size[0] // 8
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    X = []
    Y = []
    
    for p in image_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        label_file = os.path.join(labels_dir, f"{base}.txt")
        
        # Read image
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Grid target initialized to background (channel 0 = 1.0)
        target = np.zeros((grid_size, grid_size, NUM_CLASSES + 1), dtype=np.float32)
        target[:, :, 0] = 1.0
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy = float(parts[1]), float(parts[2])
                        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
                        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
                        # Assign class (0=bg, cls_id + 1)
                        target[gy, gx, 0] = 0.0
                        target[gy, gx, cls_id + 1] = 1.0
                        
        X.append(img_norm)
        Y.append(target)
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

if __name__ == "__main__":
    img_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/images"
    lbl_dir = "/home/bim/Projects/EAI-2026/warmup/data/train/labels"
    X, Y = load_fomo_dataset(img_dir, lbl_dir)
    print(f"Loaded X: {X.shape}, Y: {Y.shape}")
    bg_cells = np.sum(Y[:, :, :, 0] == 1.0)
    fg_cells = np.sum(Y[:, :, :, 1:] == 1.0)
    print(f"Background cells: {bg_cells}, Foreground object cells: {fg_cells}")
