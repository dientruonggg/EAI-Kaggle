import os
import glob
import json
import cv2
import numpy as np
import tensorflow as tf

GRID_SIZE = 16
NUM_CLASSES = 10
ORIG_W = 1672.0
ORIG_H = 941.0

def load_data_samples(split_file, labels_dir):
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
                        boxes.append((cid, cx, cy, w, h))
        samples.append((p, boxes))
    return samples

print("Data loader helper ready.")
