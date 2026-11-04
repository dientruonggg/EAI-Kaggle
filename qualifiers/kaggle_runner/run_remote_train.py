#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SLUG = "dientruong/edge-ai-2026-qualifiers-t4"
WEIGHTS_OUT = os.path.join(RUNNER_DIR, "..", "weights", "kaggle_t4")
os.makedirs(WEIGHTS_OUT, exist_ok=True)

def run_cmd(cmd):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] {res.stderr}")
    else:
        print(res.stdout)
    return res

def push():
    print(f"[*] Pushing Kaggle Kernel from {RUNNER_DIR}...")
    run_cmd(f"kaggle kernels push -p {RUNNER_DIR}")

def status():
    res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
    return res.stdout

def pull():
    print(f"[*] Pulling trained weights from Kaggle into {WEIGHTS_OUT}...")
    run_cmd(f"kaggle kernels output {KERNEL_SLUG} -p {WEIGHTS_OUT}")
    print("[✔] Downloaded files:")
    if os.path.exists(WEIGHTS_OUT):
        for f in os.listdir(WEIGHTS_OUT):
            print(f"  - {f} ({os.path.getsize(os.path.join(WEIGHTS_OUT, f))/1024:.1f} KB)")

def monitor_and_pull():
    print("[*] Monitoring Kaggle Kernel execution...")
    while True:
        res = subprocess.run(f"kaggle kernels status {KERNEL_SLUG}", shell=True, capture_output=True, text=True)
        out = res.stdout.strip()
        print(f"[{time.strftime('%H:%M:%S')}] Status: {out}")
        if "complete" in out.lower():
            print("[✔] Kernel execution completed successfully!")
            pull()
            break
        elif "error" in out.lower() or "failed" in out.lower():
            print("[❌] Kernel execution encountered an error!")
            break
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "status", "pull", "auto"], default="status", nargs="?")
    args = parser.parse_args()

    if args.action == "push":
        push()
    elif args.action == "status":
        status()
    elif args.action == "pull":
        pull()
    elif args.action == "auto":
        push()
        time.sleep(10)
        monitor_and_pull()
