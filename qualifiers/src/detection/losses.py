import tensorflow as tf
import numpy as np

# Composite class weights combining:
# 1. Error Rate Multiplier ("Càng sai trọng số càng cao") - heavily boosted weakest classes:
#    - mouse (Class 4): 86.46% (weakest by 5-11% -> boosted to 3.80)
#    - toy_car (Class 6): 89.68% (boosted to 2.50)
#    - phone (Class 2): 89.81% (boosted to 2.20)
#    - Saturated classes reduced: box (97.23% -> 0.90), cup (93.62% -> 1.20)
# Order: [0: bottle, 1: cup, 2: phone, 3: book, 4: mouse, 5: box, 6: toy_car]
CLASS_WEIGHTS = [1.80, 1.20, 2.20, 1.60, 3.80, 0.90, 2.50]
NUM_CLASSES = 7

# Dynamic / Learnable variable for adaptive hard-class gradient rebalancing
DYNAMIC_CLASS_WEIGHTS_VAR = tf.Variable(
    CLASS_WEIGHTS,
    trainable=False,
    dtype=tf.float32,
    name="dynamic_class_weights"
)

def get_current_class_weights():
    return DYNAMIC_CLASS_WEIGHTS_VAR.numpy().tolist()

def update_dynamic_class_weights(new_weights):
    """
    Updates the class weights vector dynamically.
    Ensures total weight across classes is conserved: sum(w) == NUM_CLASSES.
    """
    arr = np.array(new_weights, dtype=np.float32)
    arr = np.maximum(0.5, arr)  # floor at 0.5 to prevent class starvation
    arr = arr * (float(NUM_CLASSES) / np.sum(arr))
    DYNAMIC_CLASS_WEIGHTS_VAR.assign(arr)
    return arr

class AdaptiveHardClassWeightCallback(tf.keras.callbacks.Callback):
    """
    Online Learnable Hard-Class Gradient Balancer.
    At the end of each epoch, evaluates per-class cross-entropy error on validation data,
    and dynamically rebalances class weights: classes with higher error automatically receive
    increased gradient penalty, while saturated classes transfer gradient focus.
    """
    def __init__(self, val_dataset, beta=2.0, momentum=0.5, class_names=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.beta = beta
        self.momentum = momentum
        self.class_names = class_names or ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]

    def on_train_begin(self, logs=None):
        update_dynamic_class_weights(CLASS_WEIGHTS)
        curr = get_current_class_weights()
        weight_str = ", ".join(f"{self.class_names[c]}: {curr[c]:.2f}" for c in range(NUM_CLASSES))
        print(f"\n[*] Initialized Prior Class Weights (Weak-Class Boosted): {weight_str}\n", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        # Accumulate per-class loss on validation set
        class_losses = [[] for _ in range(NUM_CLASSES)]
        for x_batch, y_batch in self.val_dataset:
            preds = self.model(x_batch, training=False)
            p_cls = tf.clip_by_value(preds["cls_output"], 1e-7, 1.0 - 1e-7)
            t_cls = y_batch["cls_output"][..., :7]
            mask = y_batch["cls_output"][..., 7:8]

            ce_per_cell = -tf.reduce_sum(t_cls * tf.math.log(p_cls), axis=-1, keepdims=True)
            for c in range(NUM_CLASSES):
                c_mask = mask * t_cls[..., c:c+1]
                num_c = tf.reduce_sum(c_mask)
                if num_c > 0:
                    c_loss = float(tf.reduce_sum(ce_per_cell * c_mask) / num_c)
                    class_losses[c].append(c_loss)

        avg_losses = np.array([np.mean(cl) if cl else 1.0 for cl in class_losses], dtype=np.float32)
        
        # Softmax exponential weighting modulated by prior weak-class boost
        target_w = np.exp(self.beta * (avg_losses - np.mean(avg_losses)))
        prior = np.array(CLASS_WEIGHTS, dtype=np.float32)
        prior_norm = prior / np.mean(prior)
        target_w = target_w * prior_norm
        target_w = target_w * (float(NUM_CLASSES) / np.sum(target_w))
        
        # Smooth update with momentum
        curr_w = np.array(get_current_class_weights(), dtype=np.float32)
        new_w = (1.0 - self.momentum) * curr_w + self.momentum * target_w
        final_w = update_dynamic_class_weights(new_w)

        weight_str = ", ".join(f"{self.class_names[c]}: {final_w[c]:.2f}" for c in range(NUM_CLASSES))
        print(f"\n[★ Adaptive Class Weights (Epoch {epoch+1})] {weight_str}", flush=True)

class LearnableClassWeighting(tf.keras.layers.Layer):
    """
    Differentiable Backprop Layer for Learnable Class Weights & Focal Modulation.
    Can be inserted into Model computation graph for end-to-end gradient descent.
    """
    def __init__(self, num_classes=NUM_CLASSES, init_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        init_val = init_weights or [0.2, -0.1, 0.0, -0.1, 0.4, -0.1, 0.2]
        self.class_logits = self.add_weight(
            name="class_logits",
            shape=(num_classes,),
            initializer=tf.constant_initializer(init_val),
            trainable=True
        )
        self.focal_gamma = self.add_weight(
            name="focal_gamma",
            shape=(num_classes,),
            initializer=tf.constant_initializer(2.0),
            trainable=True
        )

    def get_normalized_weights(self):
        return tf.nn.softmax(self.class_logits) * float(self.num_classes)

    def call(self, inputs):
        y_true, y_pred, conf_mask = inputs
        norm_weights = self.get_normalized_weights()
        gamma = tf.nn.softplus(self.focal_gamma)
        
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_factor = tf.pow(1.0 - pt, tf.reduce_sum(y_true * gamma, axis=-1, keepdims=True))
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1, keepdims=True)
        sample_w = tf.reduce_sum(y_true * norm_weights, axis=-1, keepdims=True)
        
        loss = ce * focal_factor * sample_w * conf_mask
        num_pos = tf.maximum(1.0, tf.reduce_sum(conf_mask))
        return tf.reduce_sum(loss) / num_pos

def decoupled_conf_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
    """
    Focal Loss for Confidence / Objectness branch.
    Normalized by positive object count (RetinaNet standard) to prevent background dominance.
    y_true: (batch, grid, grid, 1) -> 1.0 = object, 0.0 = background
    y_pred: (batch, grid, grid, 1) -> sigmoid confidence score
    """
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_factor = alpha * y_true + (1.0 - alpha) * (1.0 - y_true)
    focal_weight = alpha_factor * tf.math.pow(1.0 - pt, gamma)
    bce = -tf.math.log(pt)
    
    num_pos = tf.maximum(1.0, tf.reduce_sum(y_true))
    return tf.reduce_sum(focal_weight * bce) / num_pos

def decoupled_cls_loss(y_true, y_pred):
    """
    Masked Multiclass Cross-Entropy with Error & Query-Ratio weights.
    Only computes loss on cells containing an object (conf_mask == 1).
    y_true: (batch, grid, grid, 8) -> [7 classes one-hot, 1 conf_mask]
    y_pred: (batch, grid, grid, 7) -> softmax probabilities over 7 classes
    """
    cls_true = y_true[..., 0:7]
    conf_mask = y_true[..., 7:8]
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    ce = -tf.reduce_sum(cls_true * tf.math.log(y_pred), axis=-1, keepdims=True)
    sample_w = tf.reduce_sum(cls_true * DYNAMIC_CLASS_WEIGHTS_VAR, axis=-1, keepdims=True)

    masked_loss = ce * sample_w * conf_mask
    num_obj = tf.maximum(1.0, tf.reduce_sum(conf_mask))
    return tf.reduce_sum(masked_loss) / num_obj

def decoupled_ciou_loss(y_true, y_pred, *args, **kwargs):
    """
    Masked Complete IoU (CIoU) Loss for bounding box regression.
    Penalizes overlap area (1 - IoU), normalized central point distance (rho^2 / c^2),
    and aspect ratio discrepancy (alpha * v).
    
    y_true: (batch, grid_h, grid_w, 5) -> [offset_x, offset_y, w, h, conf_mask]
    y_pred: (batch, grid_h, grid_w, 4) -> [pred_offset_x, pred_offset_y, pred_w, pred_h]
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    conf_mask = y_true[..., 4:5]
    grid_h = tf.cast(tf.shape(y_pred)[1], y_pred.dtype)
    grid_w = tf.cast(tf.shape(y_pred)[2], y_pred.dtype)

    gy, gx = tf.meshgrid(
        tf.range(grid_h, dtype=y_pred.dtype),
        tf.range(grid_w, dtype=y_pred.dtype),
        indexing="ij"
    )
    gx = tf.expand_dims(tf.expand_dims(gx, 0), -1)
    gy = tf.expand_dims(tf.expand_dims(gy, 0), -1)

    pred_dx = y_pred[..., 0:1]
    pred_dy = y_pred[..., 1:2]
    pred_w = tf.clip_by_value(y_pred[..., 2:3], 1e-4, 1.0)
    pred_h = tf.clip_by_value(y_pred[..., 3:4], 1e-4, 1.0)

    pred_cx = (gx + pred_dx) / grid_w
    pred_cy = (gy + pred_dy) / grid_h

    true_dx = y_true[..., 0:1]
    true_dy = y_true[..., 1:2]
    true_w = tf.clip_by_value(y_true[..., 2:3], 1e-4, 1.0)
    true_h = tf.clip_by_value(y_true[..., 3:4], 1e-4, 1.0)

    true_cx = (gx + true_dx) / grid_w
    true_cy = (gy + true_dy) / grid_h

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
    iou = tf.clip_by_value(inter_area / union_area, 0.0, 1.0)

    c_x1 = tf.minimum(b1_x1, b2_x1)
    c_y1 = tf.minimum(b1_y1, b2_y1)
    c_x2 = tf.maximum(b1_x2, b2_x2)
    c_y2 = tf.maximum(b1_y2, b2_y2)
    c_w = tf.maximum(0.0, c_x2 - c_x1)
    c_h = tf.maximum(0.0, c_y2 - c_y1)
    c2 = tf.square(c_w) + tf.square(c_h) + 1e-7

    rho2 = tf.square(pred_cx - true_cx) + tf.square(pred_cy - true_cy)

    pi = tf.constant(np.pi, dtype=y_pred.dtype)
    v = (4.0 / (pi ** 2)) * tf.square(tf.atan(true_w / true_h) - tf.atan(pred_w / pred_h))
    alpha = v / (1.0 - iou + v + 1e-7)
    alpha = tf.stop_gradient(alpha)

    ciou = iou - (rho2 / c2) - (alpha * v)
    loss = (1.0 - ciou) * conf_mask

    num_pos = tf.maximum(1.0, tf.reduce_sum(conf_mask))
    return tf.reduce_sum(loss) / num_pos


# Aliases
decoupled_box_loss = decoupled_ciou_loss
masked_smooth_l1_loss = decoupled_ciou_loss
calibrated_focal_loss = decoupled_cls_loss
