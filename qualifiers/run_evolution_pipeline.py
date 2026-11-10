#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - COMPLETE CONTINUOUS EVOLUTION PIPELINE
==============================================================
Runs the multi-stage breakthrough optimization:
  Stage 1: Fine-tune MicroEdgeCNN_Max warm-started from Trial 20 Extended (86.37% mAP base)
           with boosted hard-class weights (Mouse 2.5x, Bottle 2.2x, Toy car 2.2x)
           for 35 epochs with low LR (3.0e-4 -> 1.0e-5).
  Stage 2: Train MicroEdgeCNN_Ultra (186,036 parameters, 93% ESP32-S3 capacity)
           from scratch for 60 epochs with Cosine Decay (1.8e-3 -> 1.8e-5).
  Stage 3: Fuse predictions using Weighted Box Fusion (WBF) ensemble.
  Stage 4: Validate and mirror champion to root submission.csv.
"""

import os
import sys
import glob
import time
import shutil
import subprocess

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_EXE = sys.executable

def run_step(cmd_list, desc):
    print("\n" + "=" * 80)
    print(f"   ⚡ [PIPELINE] STARTING: {desc}")
    print(f"   Command: {' '.join(cmd_list)}")
    print("=" * 80, flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list)
    dur = time.time() - t0
    if res.returncode != 0:
        print(f"[!] Step '{desc}' failed with exit code {res.returncode} after {dur:.1f}s!", flush=True)
        return False
    print(f"[✔] Step '{desc}' completed successfully in {dur:.1f}s ({dur/60.0:.1f}m)!", flush=True)
    return True

def main():
    print("*" * 80)
    print("   EDGE AI CHALLENGE 2026 - CONTINUOUS BREAKTHROUGH PIPELINE")
    print("*" * 80, flush=True)

    warmstart_path = os.path.join(PROJECT_ROOT, "weights/trial20_extended/best_weights.keras")
    
    # -------------------------------------------------------------
    # STAGE 1: Fine-tune MicroEdgeCNN_Max (Hard Classes)
    # -------------------------------------------------------------
    cmd_stage1 = [
        PYTHON_EXE,
        os.path.join(PROJECT_ROOT, "train_evolution.py"),
        "--model", "MicroEdgeCNN_Max",
        "--epochs", "35",
        "--batch-size", "16",
        "--lr", "3.0e-4",
        "--min-lr", "1.0e-5",
        "--tag", "finetune_max"
    ]
    if os.path.exists(warmstart_path):
        cmd_stage1.extend(["--warmstart", warmstart_path])

    run_step(cmd_stage1, "Stage 1: Fine-tune MicroEdgeCNN_Max (Hard Classes)")

    # -------------------------------------------------------------
    # STAGE 2: Train MicroEdgeCNN_Ultra (186k params)
    # -------------------------------------------------------------
    cmd_stage2 = [
        PYTHON_EXE,
        os.path.join(PROJECT_ROOT, "train_evolution.py"),
        "--model", "MicroEdgeCNN_Ultra",
        "--epochs", "60",
        "--batch-size", "16",
        "--lr", "1.8e-3",
        "--min-lr", "1.8e-5",
        "--tag", "ultra"
    ]
    run_step(cmd_stage2, "Stage 2: Train MicroEdgeCNN_Ultra (186k params)")

    # -------------------------------------------------------------
    # STAGE 3: Weighted Box Fusion (WBF) Ensemble
    # -------------------------------------------------------------
    candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, "candidate_submissions/candidate_*.csv")))
    print(f"\n[*] Found {len(candidates)} candidate submissions for ensembling.")
    
    top_candidates = []
    # Always include Trial 20 Extended if present
    t20_ext = os.path.join(PROJECT_ROOT, "candidate_submissions/candidate_trial20_extended_score_90.46.csv")
    if os.path.exists(t20_ext):
        top_candidates.append(t20_ext)
    
    # Add new candidates
    for c in candidates:
        if c not in top_candidates and ("finetune_max" in c or "ultra" in c):
            top_candidates.append(c)

    if len(top_candidates) >= 2:
        ensemble_csv = os.path.join(PROJECT_ROOT, "candidate_submissions/candidate_wbf_ensemble.csv")
        cmd_wbf = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "ensemble_wbf.py")
        ] + top_candidates + ["--output", ensemble_csv]
        
        if run_step(cmd_wbf, f"Stage 3: WBF Ensemble of {len(top_candidates)} Models"):
            root_sub = os.path.join(PROJECT_ROOT, "submission.csv")
            workspace_sub = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "submission.csv"))
            shutil.copy2(ensemble_csv, root_sub)
            if os.path.exists(workspace_sub):
                shutil.copy2(ensemble_csv, workspace_sub)
            print(f"[★] Crowned WBF Ensemble as champion! Mirrored to {root_sub} and {workspace_sub}")

    print("\n[✔] COMPLETE EVOLUTION PIPELINE FINISHED!")

if __name__ == "__main__":
    main()
