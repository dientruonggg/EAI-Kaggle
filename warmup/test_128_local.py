import os
import sys

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/warmup"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval_local import eval_micro_fomo

tflite_path = "weights/micro_fomo_128/micro_fomo_128_int8.tflite"
default_priors = "data/class_priors.json"
optimal_priors = "data/class_priors_optimal.json"

if os.path.exists(tflite_path):
    print("\n--- EVAL WITH DEFAULT PRIORS ---")
    for conf in [0.20, 0.25, 0.30, 0.35]:
        eval_micro_fomo(tflite_path, priors_path=default_priors, conf_thresh=conf)

    print("\n--- EVAL WITH OPTIMAL ANCHOR PRIORS ---")
    for conf in [0.20, 0.25, 0.30, 0.35]:
        eval_micro_fomo(tflite_path, priors_path=optimal_priors, conf_thresh=conf)
else:
    print(f"Model not found at {tflite_path}")
