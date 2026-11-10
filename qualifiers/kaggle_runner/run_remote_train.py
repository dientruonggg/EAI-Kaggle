#!/home/bim/.local/share/uv/tools/kaggle/bin/python
import os
import sys
import time
import json
import subprocess
import argparse

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(RUNNER_DIR, "kernel-metadata.json")
WEIGHTS_OUT = os.path.join(RUNNER_DIR, "..", "weights", "kaggle_t4")
os.makedirs(WEIGHTS_OUT, exist_ok=True)

# Ensure dientruong's active token is available
TOKEN_FILE = os.path.expanduser("~/.kaggle/profiles/dientruong/access_token")
if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "r") as f:
        tok = f.read().strip()
        if tok:
            os.environ["KAGGLE_API_TOKEN"] = tok

PYTHON_BIN = "/home/bim/.local/share/uv/tools/kaggle/bin/python"

def get_kernel_slug():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            data = json.load(f)
            return data.get("id", "dientruong/edge-ai-2026-detector-t4")
    return "dientruong/edge-ai-2026-detector-t4"

def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def push():
    print(f"[*] Pushing Kaggle Kernel from {RUNNER_DIR}...")
    api = get_api()
    res = api.kernels_push(RUNNER_DIR)
    if res.error:
        print(f"[ERROR] {res.error}")
    else:
        print(f"[✔] Successfully pushed version {res.versionNumber} to: {res.url}")

def status():
    slug = get_kernel_slug()
    api = get_api()
    st = api.kernels_status(slug)
    print(f"[*] Kernel Status ({slug}): {st}")
    return st

def pull():
    slug = get_kernel_slug()
    print(f"[*] Pulling trained weights from Kaggle ({slug}) into {WEIGHTS_OUT}...")
    api = get_api()
    api.kernels_output(slug, path=WEIGHTS_OUT)
    print("[✔] Downloaded files:")
    if os.path.exists(WEIGHTS_OUT):
        for f in os.listdir(WEIGHTS_OUT):
            print(f"  - {f} ({os.path.getsize(os.path.join(WEIGHTS_OUT, f))/1024:.1f} KB)")

def monitor_and_pull():
    slug = get_kernel_slug()
    print(f"[*] Monitoring Kaggle Kernel ({slug}) execution...")
    api = get_api()
    while True:
        st = api.kernels_status(slug)
        curr_status = str(getattr(st, "status", st)).upper()
        print(f"[{time.strftime('%H:%M:%S')}] Status: {curr_status}", flush=True)
        if "COMPLETE" in curr_status:
            print("[✔] Kernel execution completed successfully!", flush=True)
            pull()
            break
        elif "ERROR" in curr_status or "FAIL" in curr_status:
            print(f"[❌] Kernel execution encountered an error! Details: {st}", flush=True)
            break
        time.sleep(15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "status", "pull", "auto", "monitor"], default="status", nargs="?")
    args = parser.parse_args()

    if args.action == "push":
        push()
    elif args.action == "status":
        status()
    elif args.action == "pull":
        pull()
    elif args.action == "monitor":
        monitor_and_pull()
    elif args.action == "auto":
        push()
        monitor_and_pull()
