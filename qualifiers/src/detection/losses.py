import tensorflow as tf
import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss for multiclass classification on dense grids.
    y_true: (batch, grid, grid, num_classes + 1) one-hot
    y_pred: (batch, grid, grid, num_classes + 1) softmax
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # Focal weight
    weight = alpha * tf.math.pow(1.0 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def ciou_loss(boxes1, boxes2):
    """
    Complete IoU (CIoU) Loss between predicted boxes and ground truth boxes.
    Both boxes formatted as [x1, y1, x2, y2].
    """
    x1_1, y1_1, x2_1, y2_1 = tf.split(boxes1, 4, axis=-1)
    x1_2, y1_2, x2_2, y2_2 = tf.split(boxes2, 4, axis=-1)
    
    # Inter area
    inter_x1 = tf.maximum(x1_1, x1_2)
    inter_y1 = tf.maximum(y1_1, y1_2)
    inter_x2 = tf.minimum(x2_1, x2_2)
    inter_y2 = tf.minimum(y2_1, y2_2)
    
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Area
    area1 = tf.maximum(0.0, x2_1 - x1_1) * tf.maximum(0.0, y2_1 - y1_1)
    area2 = tf.maximum(0.0, x2_2 - x1_2) * tf.maximum(0.0, y2_2 - y1_2)
    union_area = area1 + area2 - inter_area + 1e-7
    
    iou = inter_area / union_area
    
    # Enclosing smallest enclosing box
    enc_x1 = tf.minimum(x1_1, x1_2)
    enc_y1 = tf.minimum(y1_1, y1_2)
    enc_x2 = tf.maximum(x2_1, x2_2)
    enc_y2 = tf.maximum(y2_1, y2_2)
    c_diag_sq = tf.square(enc_x2 - enc_x1) + tf.square(enc_y2 - enc_y1) + 1e-7
    
    # Center distance
    center1_x = (x1_1 + x2_1) / 2.0
    center1_y = (y1_1 + y2_1) / 2.0
    center2_x = (x1_2 + x2_2) / 2.0
    center2_y = (y1_2 + y2_2) / 2.0
    rho_sq = tf.square(center1_x - center2_x) + tf.square(center1_y - center2_y)
    
    # Aspect ratio consistency
    w1 = tf.maximum(1e-7, x2_1 - x1_1)
    h1 = tf.maximum(1e-7, y2_1 - y1_1)
    w2 = tf.maximum(1e-7, x2_2 - x1_2)
    h2 = tf.maximum(1e-7, y2_2 - y1_2)
    
    atan_diff = tf.atan(w2 / h2) - tf.atan(w1 / h1)
    v = (4.0 / (np.pi ** 2)) * tf.square(atan_diff)
    alpha = v / (1.0 - iou + v + 1e-7)
    
    ciou = iou - (rho_sq / c_diag_sq) - (alpha * v)
    return 1.0 - tf.clip_by_value(ciou, -1.0, 1.0)
