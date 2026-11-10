#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - MIDNIGHT COMPETITION DATASET AUTO-POLLER
=================================================================
Monitors https://www.kaggle.com/competitions/egde-ai-challenge-2026-lan-2-qualifiers
Polls the Kaggle API and web endpoints every 30 seconds until midnight launch.
As soon as the dataset becomes accessible:
  1. Downloads all competition data files to qualifiers/data_incoming/
  2. Automatically executes qualifiers/ingest_official_data.py
  3. Archives current COCO data to qualifiers/data_stress_test/
  4. Prepares train/val/test splits for immediate Kaggle Dual T4 and local training.
"""

import os
import sys
import time
import json
import glob
import subprocess
import requests
from datetime import datetime

PROJECT_ROOT = "/home/bim/Projects/EAI-2026/qualifiers"
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data_incoming")
STATUS_FILE = os.path.join(PROJECT_ROOT, "auto_download_status.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "auto_poller.log")
SLUG = "egde-ai-challenge-2026-lan-2-qualifiers"

# Ensure dientruong's active token is available
TOKEN_FILE = os.path.expanduser("~/.kaggle/profiles/dientruong/access_token")
if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "r") as f:
        tok = f.read().strip()
        if tok:
            os.environ["KAGGLE_API_TOKEN"] = tok

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def check_and_download():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    os.makedirs(INCOMING_DIR, exist_ok=True)
    
    try:
        # Check if competition files are accessible
        resp = api.competition_list_files(SLUG)
        files = getattr(resp, "files", getattr(resp, "data", resp))
        file_list = []
        if isinstance(files, (list, tuple)):
            file_list = [f.name if hasattr(f, "name") else str(f) for f in files]
        
        log(f"[🎉] DATASET DETECTED & ACCESSIBLE! Found {len(file_list)} files.")
        log(f"[*] Starting automatic download to {INCOMING_DIR}...")
        
        # Download files
        api.competition_download_files(SLUG, path=INCOMING_DIR, force=True, quiet=False)
        log("[✔] Competition dataset files downloaded successfully!")
        
        # Check downloaded files
        downloaded = os.listdir(INCOMING_DIR)
        log(f"[*] Downloaded items: {downloaded}")
        
        # Look for zip file
        zips = glob.glob(os.path.join(INCOMING_DIR, "*.zip"))
        if zips:
            target_zip = zips[0]
            log(f"[*] Executing automatic intake via ingest_official_data.py on {target_zip}...")
            ingest_script = os.path.join(PROJECT_ROOT, "ingest_official_data.py")
            python_bin = sys.executable
            res = subprocess.run([python_bin, ingest_script, "--input", target_zip], capture_output=True, text=True)
            log(f"[✔] Ingestion stdout:\n{res.stdout}")
            if res.returncode != 0:
                log(f"[!] Ingestion stderr:\n{res.stderr}")
        else:
            log(f"[*] Dataset unzipped directly in {INCOMING_DIR}. Running directory ingest...")
            ingest_script = os.path.join(PROJECT_ROOT, "ingest_official_data.py")
            res = subprocess.run([sys.executable, ingest_script, "--input", INCOMING_DIR], capture_output=True, text=True)
            log(f"[✔] Ingestion output:\n{res.stdout}")

        # Update status
        with open(STATUS_FILE, "w", encoding="utf-8") as sf:
            json.dump({
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
                "slug": SLUG,
                "downloaded_files": downloaded
            }, sf, indent=2)

        log("[🚀] ALL OFFICIAL DATASET PREPARATION COMPLETED! READY FOR DUAL T4 TRAINING.")
        return True

    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg:
            log(f"[*] Status: 403 Gated / Waiting for midnight launch (Competition: {SLUG})...")
        elif "404" in err_msg:
            log(f"[*] Status: 404 Competition URL not active yet ({SLUG})...")
        else:
            log(f"[!] Warning while checking: {err_msg}")
        return False

def main():
    log(f"Starting Kaggle Midnight Auto-Poller for: https://www.kaggle.com/competitions/{SLUG}")
    log(f"Poll interval: 30 seconds. Ingestion landing directory: {INCOMING_DIR}")

    with open(STATUS_FILE, "w", encoding="utf-8") as sf:
        json.dump({"status": "POLLING", "started_at": datetime.now().isoformat(), "slug": SLUG}, sf, indent=2)

    iteration = 0
    while True:
        iteration += 1
        success = check_and_download()
        if success:
            log("[✔] Task complete. Auto-poller terminating successfully.")
            break
        
        # Periodic reminder every 10 iterations (~5 mins)
        if iteration % 10 == 0:
            log("[TIP] Reminder: When midnight strikes, please make sure you accepted the rules on:")
            log(f"      https://www.kaggle.com/competitions/{SLUG}")
            log("      Once accepted on browser, the poller will immediately catch the 200 and pull data.")

        time.sleep(30)

if __name__ == "__main__":
    main()
