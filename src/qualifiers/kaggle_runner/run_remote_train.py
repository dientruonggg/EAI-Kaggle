#!/usr/bin/env python3
import os
import sys
import time
import json
import shutil
import argparse

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(RUNNER_DIR, "kernel-metadata.json")
WEIGHTS_OUT = os.path.join(RUNNER_DIR, "..", "weights", "kaggle_t4")
os.makedirs(WEIGHTS_OUT, exist_ok=True)

def ensure_kaggle_environment():
    try:
        import kaggle  # noqa: F401
        return
    except ImportError:
        pass

    # Search for an interpreter that has kaggle installed
    candidates = []
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli:
        try:
            with open(kaggle_cli, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    candidates.append(first_line[2:].strip())
        except Exception:
            pass

    candidates.append(os.path.expanduser("~/.local/share/uv/tools/kaggle/bin/python"))

    for py in candidates:
        if os.path.isfile(py) and os.access(py, os.X_OK) and py != sys.executable:
            if __name__ == "__main__":
                os.execv(py, [py] + sys.argv)

def configure_kaggle_auth(slug=None):
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return
    if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return
    
    # Try slug owner profile if slug provided
    if slug and "/" in slug:
        owner = slug.split("/")[0]
        tok_file = os.path.expanduser(f"~/.kaggle/profiles/{owner}/access_token")
        if os.path.isfile(tok_file):
            with open(tok_file, "r", encoding="utf-8") as f:
                tok = f.read().strip()
                if tok:
                    os.environ["KAGGLE_API_TOKEN"] = tok
                    return
    
    # Try default access_token
    default_tok = os.path.expanduser("~/.kaggle/access_token")
    if os.path.isfile(default_tok):
        with open(default_tok, "r", encoding="utf-8") as f:
            tok = f.read().strip()
            if tok:
                os.environ["KAGGLE_API_TOKEN"] = tok
                return

    # Try any profile directory
    profiles_dir = os.path.expanduser("~/.kaggle/profiles")
    if os.path.isdir(profiles_dir):
        for entry in os.listdir(profiles_dir):
            tok_f = os.path.join(profiles_dir, entry, "access_token")
            if os.path.isfile(tok_f):
                with open(tok_f, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                    if tok:
                        os.environ["KAGGLE_API_TOKEN"] = tok
                        return

def fix_mojibake(text):
    if not text:
        return text
    if "â" in text or "ð" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except Exception:
            text = (text.replace("â” ", "━")
                        .replace("â”\x81", "━")
                        .replace("âœ”", "✔")
                        .replace("â˜…", "★")
                        .replace("ð\x9f\x8f\x86", "🏆"))
    return text

def get_kernel_slug():
    if os.environ.get("KAGGLE_KERNEL_SLUG"):
        return os.environ["KAGGLE_KERNEL_SLUG"]
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("id"):
                    return data["id"]
        except Exception:
            pass
    raise RuntimeError(
        f"Kernel slug not specified. Set KAGGLE_KERNEL_SLUG env var or configure 'id' in {META_PATH}"
    )

def get_api(slug=None):
    ensure_kaggle_environment()
    configure_kaggle_auth(slug)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def push():
    slug = get_kernel_slug()
    print(f"[*] Pushing Kaggle Kernel ({slug}) from {RUNNER_DIR}...")
    api = get_api(slug)
    res = api.kernels_push(RUNNER_DIR)
    if res.error:
        print(f"[ERROR] {res.error}")
    else:
        print(f"[✔] Successfully pushed version {res.versionNumber} to: {res.url}")

def status():
    slug = get_kernel_slug()
    api = get_api(slug)
    try:
        st = api.kernels_status(slug)
        print(f"[*] Kernel Status ({slug}): {st}")
        return st
    except Exception as e:
        if "429" in str(e):
            print(f"[*] Kernel Status ({slug}): RUNNING (API 429 rate limit active - remote container is executing)")
            return "RUNNING"
        else:
            print(f"[!] Error fetching status: {e}")
            return None

def pull():
    slug = get_kernel_slug()
    print(f"[*] Pulling trained weights from Kaggle ({slug}) into {WEIGHTS_OUT}...")
    api = get_api(slug)
    api.kernels_output(slug, path=WEIGHTS_OUT)
    print("[✔] Downloaded files:")
    if os.path.exists(WEIGHTS_OUT):
        for f in os.listdir(WEIGHTS_OUT):
            print(f"  - {f} ({os.path.getsize(os.path.join(WEIGHTS_OUT, f))/1024:.1f} KB)")

def monitor_and_pull():
    slug = get_kernel_slug()
    print(f"[*] Monitoring Kaggle Kernel ({slug}) execution...")
    api = get_api(slug)
    while True:
        st = api.kernels_status(slug)
        curr_status = str(getattr(st, "status", st)).upper()
        print(f"[{time.strftime('%H:%M:%S')}] Status: {curr_status}", flush=True)
        if "COMPLETE" in curr_status:
            print("[✔] Kernel execution completed successfully!", flush=True)
            pull()
            break
        elif "ERROR" in curr_status or "FAIL" in curr_status or "CANCEL" in curr_status:
            print(f"[❌] Kernel execution stopped! Details: {st}", flush=True)
            break
        time.sleep(15)

def stream_logs():
    slug = get_kernel_slug()
    print(f"[*] Streaming live logs (tail -f) for {slug} (auto-reconnect enabled)...\n", flush=True)
    api = get_api(slug)
    owner_slug, kernel_slug, version = api.parse_kernel_string(slug)

    seen_count = 0
    consecutive_errors = 0

    last_status_check = 0
    while True:
        # Check kernel status conservatively (at most once every 60s) to avoid 429 rate limit
        now = time.time()
        if now - last_status_check > 60:
            last_status_check = now
            try:
                st_obj = api.kernels_status(slug)
                curr_status = str(getattr(st_obj, "status", st_obj)).upper()
                if "COMPLETE" in curr_status:
                    print(f"\n[✔] Kernel execution completed successfully! (Status: {curr_status})", flush=True)
                    break
                elif "ERROR" in curr_status or "FAIL" in curr_status or "CANCEL" in curr_status:
                    print(f"\n[❌] Kernel stopped with error/failure! Details: {st_obj}", flush=True)
                    break
            except Exception:
                pass

        try:
            with api.build_kaggle_client() as kaggle:
                http = kaggle._http_client
                http._init_session()
                url = f"{http._endpoint}/v1/kernels/logs/stream/{owner_slug}/{kernel_slug}"
                headers = dict(http._session.headers)
                headers["Accept"] = "text/event-stream, */*"
                headers.pop("Content-Type", None)
                res = http._session.get(url, stream=True, headers=headers, auth=http._session.auth, timeout=(15, 60))
                res.encoding = "utf-8"

                if res.status_code == 429:
                    print(f"[*] Kaggle API rate limited (HTTP 429). Backing off for 60s to allow cooldown...", flush=True)
                    time.sleep(60)
                    continue

                idx = 0
                for raw_line in res.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    payload = raw_line[len("data:") :].lstrip()
                    if payload == api._LOG_STREAM_END_SENTINEL:
                        break

                    if idx >= seen_count:
                        try:
                            ev = json.loads(payload)
                            d = ev.get("data", "")
                            sys.stdout.write(fix_mojibake(d))
                            sys.stdout.flush()
                        except Exception:
                            sys.stdout.write(fix_mojibake(payload) + "\n")
                            sys.stdout.flush()
                        seen_count = idx + 1
                    idx += 1

                consecutive_errors = 0

        except KeyboardInterrupt:
            print("\n[*] Stopped log streaming.", flush=True)
            break
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors > 1:
                print(f"\n[!] Log stream connection dropped ({e}), reconnecting...", flush=True)
            time.sleep(2)
            continue

        # Check kernel status conservatively (at most once every 60s)
        now = time.time()
        if now - last_status_check > 60:
            last_status_check = now
            try:
                st_obj = api.kernels_status(slug)
                curr_status = str(getattr(st_obj, "status", st_obj)).upper()
                if "COMPLETE" in curr_status:
                    print(f"\n[✔] Kernel execution completed successfully! (Status: {curr_status})", flush=True)
                    break
                elif "ERROR" in curr_status or "FAIL" in curr_status or "CANCEL" in curr_status:
                    print(f"\n[❌] Kernel stopped with error/failure! Details: {st_obj}", flush=True)
                    break
            except Exception:
                pass

        time.sleep(5)

if __name__ == "__main__":
    ensure_kaggle_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["push", "status", "pull", "auto", "monitor", "log", "tail"], default="status", nargs="?")
    args = parser.parse_args()

    if args.action == "push":
        push()
    elif args.action == "status":
        status()
    elif args.action == "pull":
        pull()
    elif args.action == "monitor":
        monitor_and_pull()
    elif args.action in ("log", "tail"):
        stream_logs()
    elif args.action == "auto":
        push()
        monitor_and_pull()
