#!/usr/bin/env python3
"""
execute_and_populate_notebook.py
================================================================================
Executes reproduce/reproduce_submission.ipynb and populates all cell outputs
with actual runtime logs and matplotlib visual plots.
"""

import sys
import io
import json
import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

nb_path = Path("reproduce/reproduce_submission.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

ctx = {
    "__name__": "__main__",
    "__file__": str(nb_path.resolve())
}

import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

exec_count = 1
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue

    code = "".join(cell["source"])
    print(f"[*] Running and capturing Cell {exec_count}...")

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    cell["execution_count"] = exec_count
    cell["outputs"] = []

    try:
        exec(code, ctx)
    except Exception as e:
        sys.stdout = old_stdout
        print(f"[ERROR] in Cell {exec_count}: {e}")
        raise e

    sys.stdout = old_stdout
    captured_text = stdout_capture.getvalue()

    if captured_text:
        # Split into lines preserving newlines
        lines = [l + "\n" for l in captured_text.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": lines
        })

    # If matplotlib plot was generated in this cell, capture it
    if "plt" in ctx and hasattr(ctx["plt"], "get_fignums") and ctx["plt"].get_fignums():
        fig = ctx["plt"].gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        ctx["plt"].close("all")

        cell["outputs"].append({
            "output_type": "display_data",
            "data": {
                "image/png": img_b64,
                "text/plain": ["<Figure size 1400x1000 with 4 Axes>"]
            },
            "metadata": {}
        })

    exec_count += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[★ SUCCESS] Notebook {nb_path} fully executed and saved with all outputs populated!")
