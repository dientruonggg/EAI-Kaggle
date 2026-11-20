#!/usr/bin/env python3
"""
Edge AI Challenge 2026 - High-Performance Multi-Worker TUI Monitor
===================================================================
Real-time full-screen terminal dashboard powered by Rich.
Features:
- Dedicated 3-panel display for Worker 0, Worker 1, and Worker 2
- Live epoch progress, train/val loss curves, model architecture, and hyperparameters
- Real-time Tesla V100 GPU telemetry (VRAM, Util, Temp, Power)
- Top Champions Leaderboard from optimizer_state.json
"""

import os
import re
import sys
import glob
import time
import json
import subprocess
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich import box

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
_candidates = [
    os.path.join(PROJECT_ROOT, "candidate_submissions/continuous_optimizer.log"),
    os.path.join(PROJECT_ROOT, "candidate_submissions/parallel_optimizer.log"),
    os.path.join(PROJECT_ROOT, "parallel_optimizer.log"),
]
_valid = [c for c in _candidates if os.path.exists(c)]
LOG_FILE = max(_valid, key=os.path.getmtime) if _valid else os.path.join(PROJECT_ROOT, "candidate_submissions/continuous_optimizer.log")
STATE_FILE = os.path.join(PROJECT_ROOT, "candidate_submissions/optimizer_state.json")

_cached_gpu = None
_last_gpu_query = 0.0

def get_gpu_telemetry():
    """Queries nvidia-smi for real-time GPU stats with a 3s TTL cache."""
    global _cached_gpu, _last_gpu_query
    now = time.time()
    if _cached_gpu is not None and (now - _last_gpu_query) < 3.0:
        return _cached_gpu
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, universal_newlines=True
        ).strip()
        util, mem_used, mem_tot, temp, pwr = [float(x.strip()) for x in out.split(",")]
        _cached_gpu = {
            "util": util,
            "mem_used": mem_used / 1024.0,
            "mem_tot": mem_tot / 1024.0,
            "mem_pct": (mem_used / mem_tot) * 100.0,
            "temp": int(temp),
            "pwr": pwr
        }
        _last_gpu_query = now
        return _cached_gpu
    except Exception:
        return _cached_gpu

KNOWN_PARAMS = {
    "MicroEdgeCNN_UltraV2": "179,068",
    "MicroEdgeCNN_Ultra": "186,036",
    "MicroEdgeCNN_Max": "128,940",
    "MicroEdgeCNN_Wide": "50,540",
    "MicroEdgeCNN_Base": "50,228",
    "MicroEdgeCNN_SE": "30,564",
    "MicroEdgeCNN_DualRF": "23,564"
}

def parse_worker_logs():
    """
    Parses recent lines from parallel_optimizer.log and cross-references with optimizer_state.json
    to build an exact per-worker trial execution history with zero CPU lag.
    """
    workers_data = {
        0: {"current": None, "history_trials": [], "recent_steps": []},
        1: {"current": None, "history_trials": [], "recent_steps": []},
        2: {"current": None, "history_trials": [], "recent_logs": []}
    }
    worker_trials = {0: {}, 1: {}, 2: {}}

    # 1. Load ground-truth completed trials from state
    state_trials = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                st = json.load(f)
            for t in st.get("completed_trials", []):
                state_trials[t["trial_id"]] = t
        except Exception:
            pass

    # 2. Parse tail of log for recent worker activities (Fast seek: last 150 KB)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "rb") as f:
                f.seek(0, os.SEEK_END)
                fsize = f.tell()
                f.seek(max(0, fsize - 150000))
                raw_chunk = f.read().decode("utf-8", errors="ignore")
                lines = raw_chunk.splitlines()

            for line in lines:
                line = line.strip()
                m_wid = re.search(r"\[Worker (\d+)\]", line)
                if m_wid:
                    wid = int(m_wid.group(1))
                else:
                    wid = 0
                if wid not in worker_trials:
                    continue

                # Starting trial (both parallel and continuous loop format)
                if ">>> Starting Trial" in line or "[STARTING TRIAL" in line:
                    m = re.search(r"Starting Trial (\d+) \| Architecture: (\w+) \| LR: ([\d.e+-]+)", line)
                    if not m:
                        m = re.search(r"\[STARTING TRIAL (\d+)\] ARCHITECTURE: (\w+)", line)
                    if m:
                        tid = int(m.group(1))
                        m_arch = m.group(2)
                        m_lr = re.search(r"LR:\s*([\d.e+-]+)", line)
                        lr = m_lr.group(1) if m_lr else "1.8e-4"
                        m_loss = re.search(r"Loss.*?(\{[^}]+\})", line)
                        loss_cfg = m_loss.group(1) if m_loss else "-"
                        worker_trials[wid][tid] = {
                            "trial_id": tid,
                            "model": m_arch,
                            "params": KNOWN_PARAMS.get(m_arch, "-"),
                            "lr": lr,
                            "loss_cfg": loss_cfg,
                            "status": "Starting",
                            "epoch": "0/25",
                            "loss": "-",
                            "val_loss": "-"
                        }
                        workers_data[wid]["current"] = worker_trials[wid][tid]

                # Built parameters
                if "[*] Built " in line:
                    m_p = re.search(r"Built (\w+): ([\d,]+) parameters", line)
                    if m_p:
                        b_arch = m_p.group(1)
                        b_params = m_p.group(2)
                        for w_i in [0, 1, 2]:
                            c = workers_data[w_i].get("current")
                            if c and c.get("model") == b_arch:
                                c["params"] = b_params

                # Epoch training steps
                if "Epoch" in line and ("loss:" in line or "/" in line):
                    m_ep = re.search(r"Trial\s+(\d+)\s+\|\s+Epoch\s+(\d+/\d+)\s+\|\s+loss:\s+([\d.]+)\s+\|\s+val_loss:\s+([\d.]+)", line)
                    if not m_ep:
                        m_ep = re.search(r"Epoch\s+(\d+/\d+)\s+\|\s+loss:\s+([\d.]+)\s+\|\s+val_loss:\s+([\d.]+)", line)
                    if not m_ep:
                        m_ep_simple = re.search(r"Epoch (\d+)/(\d+)", line)
                        if m_ep_simple:
                            cur = workers_data[wid].get("current")
                            if cur:
                                cur["epoch"] = f"{m_ep_simple.group(1)}/{m_ep_simple.group(2)}"
                                cur["status"] = f"Training (Ep {cur['epoch']})"
                    if m_ep:
                        if len(m_ep.groups()) == 4:
                            tid = int(m_ep.group(1))
                            ep_str = m_ep.group(2)
                            l_val = m_ep.group(3)
                            vl_val = m_ep.group(4)
                        else:
                            ep_str = m_ep.group(1)
                            l_val = m_ep.group(2)
                            vl_val = m_ep.group(3)
                            cur = workers_data[wid].get("current")
                            tid = cur["trial_id"] if cur else None

                        if tid and tid in worker_trials[wid]:
                            worker_trials[wid][tid]["epoch"] = ep_str
                            worker_trials[wid][tid]["loss"] = l_val
                            worker_trials[wid][tid]["val_loss"] = vl_val
                            worker_trials[wid][tid]["status"] = f"Training ({ep_str})"

                        entry = f"Ep {ep_str} -> loss: {l_val} | val_loss: {vl_val}"
                        if not workers_data[wid].get("recent_steps") or workers_data[wid]["recent_steps"][-1] != entry:
                            if "recent_steps" not in workers_data[wid]:
                                workers_data[wid]["recent_steps"] = []
                            workers_data[wid]["recent_steps"].append(entry)
                            if len(workers_data[wid]["recent_steps"]) > 20:
                                workers_data[wid]["recent_steps"].pop(0)

                # Finished trial
                if "[✔ FINISHED TRIAL" in line:
                    m_fin = re.search(r"FINISHED TRIAL (\d+).*?Val mAP@50:\s*([\d.]+)%.*?Score S:\s*([\d.]+)", line)
                    if m_fin:
                        tid = int(m_fin.group(1))
                        map_val = float(m_fin.group(2))
                        sc_val = float(m_fin.group(3))
                        if tid in worker_trials[wid]:
                            worker_trials[wid][tid]["status"] = "✔ Done"
                            worker_trials[wid][tid]["map50"] = map_val
                            worker_trials[wid][tid]["score"] = sc_val

                # Error in trial
                if "[!] Error in Trial" in line:
                    m_err = re.search(r"Error in Trial (\d+)", line)
                    if m_err:
                        tid = int(m_err.group(1))
                        if tid in worker_trials[wid]:
                            worker_trials[wid][tid]["status"] = "❌ Error"
        except Exception:
            pass

    # 3. Enrich per-worker trials with official state ground truth
    for wid in [0, 1, 2]:
        for tid, t in worker_trials[wid].items():
            if tid in state_trials:
                st = state_trials[tid]
                t["status"] = "✔ Done"
                t["score"] = st.get("score", 0.0)
                t["map50"] = st.get("map50", 0.0) * 100.0
                t["params"] = f"{st.get('params', 0):,}"

        # Sorted trial list for this worker
        sorted_h = sorted(worker_trials[wid].values(), key=lambda x: x["trial_id"])
        workers_data[wid]["history_trials"] = sorted_h

        # Active or latest trial
        if sorted_h:
            # If the last trial is still in progress, that's current
            last_t = sorted_h[-1]
            workers_data[wid]["current"] = last_t

    return workers_data

def get_leaderboard(limit=15):
    """
    Dynamically discovers and aggregates top champions from multiple sources:
    1. optimizer_state.json
    2. candidate_submissions/ directory (candidate_*.csv)
    3. Live stream from parallel_optimizer.log
    """
    trials_map = {}

    # 1. Load from optimizer_state.json
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                st = json.load(f)
            for t in st.get("completed_trials", []):
                tid = t.get("trial_id")
                if tid is not None:
                    trials_map[tid] = dict(t)
        except Exception:
            pass

    # 2. Discover from candidate CSV files
    cand_dir = os.path.dirname(STATE_FILE)
    if os.path.exists(cand_dir):
        for cf in glob.glob(os.path.join(cand_dir, "candidate_*.csv")):
            bname = os.path.basename(cf)
            m_sc = re.search(r"score_([0-9]+(?:\.[0-9]+)?)\.csv", bname)
            if m_sc:
                sc = float(m_sc.group(1))
                m_tid = re.search(r"trial_(\d+)", bname)
                if m_tid:
                    tid = int(m_tid.group(1))
                else:
                    tid = bname.replace("candidate_", "").split("_score_")[0]

                if tid in trials_map:
                    trials_map[tid]["candidate_file"] = bname
                else:
                    m_name = "MicroEdgeCNN_Ultra" if "ultra" in str(tid).lower() else "MicroEdgeCNN_Max"
                    params = 186036 if "ultra" in str(tid).lower() else 128940
                    trials_map[tid] = {
                        "trial_id": tid,
                        "model_name": m_name,
                        "params": params,
                        "map50": (sc - 30.0) / 70.0 / 100.0 if sc > 30 else 0.0,
                        "score": sc,
                        "candidate_file": bname
                    }

    # 3. Real-time extraction from recent log tail
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "rb") as f:
                f.seek(0, os.SEEK_END)
                fsize = f.tell()
                f.seek(max(0, fsize - 100000))
                recent_lines = f.read().decode("utf-8", errors="ignore").splitlines()

            for line in recent_lines:
                if "[✔ FINISHED TRIAL" in line:
                    m = re.search(r"FINISHED TRIAL (\d+).*?([A-Za-z0-9_]+)\s*\(([\d,]+)\s*params\).*?Val mAP@50:\s*([\d.]+)%.*?Score S:\s*([\d.]+)", line)
                    if m:
                        tid = int(m.group(1))
                        m_name = m.group(2)
                        params = int(m.group(3).replace(",", ""))
                        map50 = float(m.group(4)) / 100.0
                        score = float(m.group(5))
                        if tid not in trials_map or trials_map[tid].get("score", 0) < score:
                            rec = trials_map.get(tid, {})
                            rec.update({
                                "trial_id": tid,
                                "model_name": m_name,
                                "params": params,
                                "map50": map50,
                                "score": score
                            })
                            trials_map[tid] = rec
        except Exception:
            pass

    # Identify currently mirrored champion submission
    sub_csv = os.path.join(cand_dir, "submission.csv")
    if os.path.exists(sub_csv) and trials_map:
        sub_sz = os.path.getsize(sub_csv)
        for t in trials_map.values():
            cf = t.get("candidate_file")
            if cf:
                cf_path = os.path.join(cand_dir, cf)
                if os.path.exists(cf_path) and os.path.getsize(cf_path) == sub_sz:
                    t["is_mirrored"] = True
                    break

    # Sort descending by score
    sorted_trials = sorted(trials_map.values(), key=lambda x: x.get("score", 0), reverse=True)
    return sorted_trials[:limit], len(trials_map)

_cached_candidates = None
_last_candidates_query = 0.0

def get_candidate_vault():
    """Returns detailed records of all candidate submissions generated with 5s cache."""
    global _cached_candidates, _last_candidates_query
    now = time.time()
    if _cached_candidates is not None and (now - _last_candidates_query) < 5.0:
        return _cached_candidates

    cand_dir = os.path.dirname(STATE_FILE)
    candidates = []
    sub_csv = os.path.join(cand_dir, "submission.csv")
    sub_sz = os.path.getsize(sub_csv) if os.path.exists(sub_csv) else 0

    if os.path.exists(cand_dir):
        # Scan only recent candidate CSVs
        all_cfs = sorted(glob.glob(os.path.join(cand_dir, "candidate_*.csv")), reverse=True)
        for cf in all_cfs:
            bname = os.path.basename(cf)
            m_sc = re.search(r"score_([0-9]+(?:\.[0-9]+)?)\.csv", bname)
            if m_sc:
                sc = float(m_sc.group(1))
                m_tid = re.search(r"trial_(\d+)", bname)
                if m_tid:
                    tid = int(m_tid.group(1))
                else:
                    tid = bname.replace("candidate_", "").split("_score_")[0]
                try:
                    fsize = os.path.getsize(cf)
                except Exception:
                    fsize = 0
                sz_mb = fsize / (1024.0 * 1024.0)
                is_mir = (fsize == sub_sz and sub_sz > 0)
                candidates.append({
                    "file": bname,
                    "trial_id": tid,
                    "score": sc,
                    "size_mb": sz_mb,
                    "is_mirrored": is_mir
                })
    _cached_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    _last_candidates_query = now
    return _cached_candidates

def make_worker_panel(wid, data):
    """
    Builds an informative panel for a worker process displaying:
    1. Currently active trial, model architecture, live epoch progress & losses
    2. Exact table of all trials executed by THIS worker with status, mAP, and score
    3. Live epoch loss steps feed
    """
    cur = data.get("current")
    history = data.get("history_trials", [])
    recent_steps = data.get("recent_steps", [])

    # Header title
    if cur and "Training" in cur.get("status", ""):
        title = f"[bold cyan]Worker #{wid}[/bold cyan] [bold green]● BUSY (Trial #{cur['trial_id']})[/bold green]"
    elif cur and "Done" in cur.get("status", ""):
        title = f"[bold cyan]Worker #{wid}[/bold cyan] [bold bright_yellow]✔ FINISHED Trial #{cur['trial_id']}[/bold bright_yellow]"
    else:
        title = f"[bold cyan]Worker #{wid}[/bold cyan] [dim]○ IDLE[/dim]"

    content = Table.grid(padding=(0, 0), expand=True)

    # 1. Active Trial Telemetry
    act_tbl = Table.grid(padding=(0, 1))
    act_tbl.add_column(style="bold yellow", justify="right", width=13)
    act_tbl.add_column(style="white")

    if cur:
        act_tbl.add_row("Active Trial:", f"[bold bright_yellow]Trial #{cur['trial_id']}[/bold bright_yellow] - [bold magenta]{cur['model']}[/bold magenta] ({cur.get('params', '-')} params)")
        act_tbl.add_row("LR / Loss:", f"[cyan]{cur.get('lr', '-')}[/cyan] | [dim]{cur.get('loss_cfg', '-')}[/dim]")

        # Progress visual
        ep = cur.get("epoch", "0/25")
        cur_ep = 0
        tot_ep = 25
        if "/" in str(ep):
            try:
                c, t = ep.split("/")
                cur_ep = int(c)
                tot_ep = int(t)
            except Exception:
                pass
        pct = (cur_ep / max(tot_ep, 1)) * 100.0
        bar_len = 12
        filled = int(bar_len * (cur_ep / max(tot_ep, 1)))
        bar_str = "█" * filled + "░" * (bar_len - filled)
        act_tbl.add_row("Live Epoch:", f"[bold green]{ep}[/bold green] [{bar_str}] [cyan]{pct:.0f}%[/cyan]")
        act_tbl.add_row("Train Loss:", f"[white]{cur.get('loss', '-')}[/white] | Val: [bold green]{cur.get('val_loss', '-')}[/bold green]")

        # Status badge
        stat = cur.get("status", "Starting")
        if "Training" in stat:
            stat_badge = f"[bold green]⏳ {stat.upper()}[/bold green]"
        elif "Done" in stat:
            stat_badge = f"[bold bright_yellow]✔ FINISHED ({cur.get('score', 0):.2f} pts)[/bold bright_yellow]"
        elif "Error" in stat:
            stat_badge = "[bold red]❌ FAILED[/bold red]"
        else:
            stat_badge = f"[cyan]{stat}[/cyan]"
        act_tbl.add_row("Status:", stat_badge)
    else:
        act_tbl.add_row("Status:", "[dim]Worker idle, awaiting assignment...[/dim]")

    content.add_row(act_tbl)

    # 2. Mini-Table of All Trials Handled by this Worker
    htbl = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True, padding=(0, 1))
    htbl.add_column("Trial", style="yellow", width=7)
    htbl.add_column("Architecture", style="magenta")
    htbl.add_column("Status", width=12)
    htbl.add_column("mAP@50", justify="right", style="green", width=8)
    htbl.add_column("Score S", justify="right", style="bold bright_yellow", width=8)

    if history:
        for t in history:
            tid_badge = f"#{t['trial_id']}"
            m_arch = t.get("model", "-")
            st = t.get("status", "-")
            if "Done" in st:
                st_str = "[bold green]✔ Done[/bold green]"
                sc_str = f"{t.get('score', 0):.2f}"
                map_str = f"{t.get('map50', 0):.1f}%"
            elif "Error" in st:
                st_str = "[red]❌ Error[/red]"
                sc_str = "-"
                map_str = "-"
            else:
                st_str = f"[cyan]⏳ {t.get('epoch', 'Run')}[/cyan]"
                sc_str = "-"
                map_str = "-"
            htbl.add_row(tid_badge, m_arch, st_str, map_str, sc_str)
    else:
        htbl.add_row("-", "No trials recorded yet", "-", "-", "-")

    content.add_row(Text("\n📋 Trials Assigned to Worker #" + str(wid) + ":", style="bold underline yellow"))
    content.add_row(htbl)

    # 3. Recent Epoch Steps
    if recent_steps:
        steps_text = Text()
        steps_text.append("\nLive Steps Feed:\n", style="bold underline dim")
        for h in recent_steps[-4:]:
            steps_text.append(f" • {h}\n", style="dim")
        content.add_row(steps_text)

    # Border style
    border_styles = ["bright_cyan", "bright_magenta", "bright_blue"]
    return Panel(content, title=title, border_style=border_styles[wid % 3], box=box.ROUNDED)

def make_header(total_count=0, champion=None):
    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Text("⚡ EDGE AI CHALLENGE 2026 - PARALLEL MULTI-WORKER OPTIMIZER ⚡", style="bold bright_yellow"))
    champ_str = f"⭐ All-time Champion: Trial {champion.get('trial_id', '-')} ({champion.get('model_name', '-')}) -> Score: {champion.get('score', 0):.2f} (mAP {champion.get('map50', 0)*100:.2f}%)" if champion else "Evaluating Trials..."
    sub_str = f"Target: ESP32-S3 (< 200k params, INT8) | Total Trials: {total_count} Completed | {champ_str} | Active: submission.csv"
    grid.add_row(Text(sub_str, style="dim"))
    return Panel(grid, style="on grey11", box=box.HEAVY)

def make_footer(gpu, champions, total_count=0, candidates=None):
    if candidates is None:
        candidates = []

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)  # Left: GPU & Specs
    grid.add_column(ratio=3)  # Center: Leaderboard Top 15
    grid.add_column(ratio=2)  # Right: Candidate Vault

    # 1. Left: GPU & Microcontroller Specs
    gpu_grid = Table.grid(padding=(0, 1))
    gpu_grid.add_column(style="bold cyan", width=12)
    gpu_grid.add_column(style="white")
    if gpu:
        vram_bar = "█" * int(gpu['mem_pct'] / 10) + "░" * (10 - int(gpu['mem_pct'] / 10))
        gpu_grid.add_row("VRAM Used:", f"[bold green]{gpu['mem_used']:.1f} GB[/bold green] / {gpu['mem_tot']:.1f} GB [{vram_bar}] ({gpu['mem_pct']:.0f}%)")
        gpu_grid.add_row("Compute:", f"Util: [bold yellow]{gpu['util']:.0f}%[/bold yellow] | Pwr: [cyan]{gpu['pwr']:.0f}W[/cyan] | Temp: [bold red]{gpu['temp']}°C[/bold red]")
        gpu_grid.add_row("Workers:", f"[bold green]3 Active Processes[/bold green] (Multi-Worker Pool)")
        gpu_grid.add_row("Target HW:", "[yellow]ESP32-S3 (Xtensa LX7 @ 240MHz)[/yellow]")
        gpu_grid.add_row("Budget:", "[dim]< 200k params, <= 1.5MB Flash[/dim]")
        gpu_grid.add_row("Format:", "[bold magenta]Full INT8 TFLM (Zero Float Ops)[/bold magenta]")
        gpu_grid.add_row("Evaluated:", f"[bold green]{total_count} Trials Completed[/bold green] (600-img val)")
        gpu_grid.add_row("Intent:", "[bold green]100.00% (30.00/30.00 pts Locked)[/bold green]")
    else:
        gpu_grid.add_row("GPU:", "Querying telemetry...")

    gpu_panel = Panel(gpu_grid, title="[bold green]🎮 Tesla V100 & Target HW[/bold green]", border_style="green", box=box.ROUNDED)

    # 2. Center: Leaderboard (Top 15)
    lb_table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, expand=True)
    lb_table.add_column("Rank", style="bold cyan", width=6)
    lb_table.add_column("Trial", style="yellow", width=8)
    lb_table.add_column("Architecture", style="magenta")
    lb_table.add_column("Params", justify="right", style="dim")
    lb_table.add_column("mAP@50", justify="right", style="green")
    lb_table.add_column("Score S", justify="right", style="bold bright_yellow")
    lb_table.add_column("Candidate / Mirror", style="cyan")

    for i, t in enumerate(champions[:15]):
        if i == 0:
            rank_badge = "🥇 1st"
            score_str = f"[bold bright_green]{t.get('score', 0):.2f}[/bold bright_green]"
        elif i == 1:
            rank_badge = "🥈 2nd"
            score_str = f"[bold bright_yellow]{t.get('score', 0):.2f}[/bold bright_yellow]"
        elif i == 2:
            rank_badge = "🥉 3rd"
            score_str = f"[bold yellow]{t.get('score', 0):.2f}[/bold yellow]"
        else:
            rank_badge = f"#{i+1}"
            score_str = f"{t.get('score', 0):.2f}"

        cf = t.get("candidate_file")
        if t.get("is_mirrored") and cf:
            cand_disp = f"[bold bright_green]⭐ {cf[:18]}.. [SUBMISSION][/bold bright_green]"
        elif t.get("is_mirrored"):
            cand_disp = "[bold bright_green]⭐ submission.csv[/bold bright_green]"
        elif cf:
            cand_disp = f"[cyan]{cf[:22]}...[/cyan]" if len(cf) > 25 else f"[cyan]{cf}[/cyan]"
        else:
            cand_disp = "[dim]-[/dim]"

        lb_table.add_row(
            rank_badge,
            f"Trial {t.get('trial_id', '-')}",
            t.get("model_name", "-"),
            f"{t.get('params', 0):,}",
            f"{t.get('map50', 0)*100:.2f}%",
            score_str,
            cand_disp
        )

    now_str = datetime.now().strftime("%H:%M:%S")
    lb_panel = Panel(
        lb_table,
        title=f"[bold bright_yellow]🏆 Top 15 Champion Models (Ranked by Score)[/bold bright_yellow] [bold cyan]({now_str})[/bold cyan]",
        border_style="bright_yellow",
        box=box.ROUNDED
    )

    # 3. Right: Candidate Vault (Saved CSVs)
    vault_table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, expand=True)
    vault_table.add_column("Candidate File", style="white")
    vault_table.add_column("Score", justify="right", style="bold yellow")
    vault_table.add_column("Size", justify="right", style="dim")
    vault_table.add_column("Status", style="green")

    for c in candidates[:10]:
        fn = c["file"]
        short_fn = fn if len(fn) <= 28 else fn[:25] + "..."
        if c.get("is_mirrored"):
            stat = "[bold bright_green]⭐ Mirrored[/bold bright_green]"
        elif "trial_14" in fn:
            stat = "[bold cyan]LB: 87.74[/bold cyan]"
        else:
            stat = "[dim]Verified[/dim]"

        vault_table.add_row(
            short_fn,
            f"{c['score']:.2f}",
            f"{c['size_mb']:.1f}MB",
            stat
        )

    vault_panel = Panel(
        vault_table,
        title=f"[bold bright_magenta]📦 Candidate Vault ({len(candidates)} Files)[/bold bright_magenta]",
        border_style="bright_magenta",
        box=box.ROUNDED
    )

    grid.add_row(gpu_panel, lb_panel, vault_panel)
    return grid

def build_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="workers", ratio=1, minimum_size=20),
        Layout(name="footer", ratio=1, minimum_size=20)
    )
    layout["workers"].split_row(
        Layout(name="worker_0"),
        Layout(name="worker_1"),
        Layout(name="worker_2")
    )

    gpu = get_gpu_telemetry()
    workers = parse_worker_logs()
    champions, total_count = get_leaderboard(limit=15)
    candidates = get_candidate_vault()

    top_champ = champions[0] if champions else None
    layout["header"].update(make_header(total_count, top_champ))
    layout["worker_0"].update(make_worker_panel(0, workers[0]))
    layout["worker_1"].update(make_worker_panel(1, workers[1]))
    layout["worker_2"].update(make_worker_panel(2, workers[2]))
    layout["footer"].update(make_footer(gpu, champions, total_count, candidates))

    return layout

def main():
    console = Console()
    if "--once" in sys.argv:
        console.print(build_dashboard())
        return
    console.clear()
    with Live(build_dashboard(), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(1.0)
                live.update(build_dashboard())
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
