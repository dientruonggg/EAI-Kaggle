#!/usr/bin/env python3
"""
build_detector_notebook.py
Generates edge_ai_2026_detector_t4.ipynb matching DuyToan/DT.ipynb exactly,
with Kaggle dataset discovery, Intent TextCNN, and full INT8 submission export.
"""

import json
import os
from pathlib import Path

# Load DuyToan/DT.ipynb to extract exact cells
with open("DuyToan/DT.ipynb", "r", encoding="utf-8") as f:
    orig_nb = json.load(f)

orig_cells = orig_nb["cells"]
print(f"Loaded DuyToan/DT.ipynb with {len(orig_cells)} cells.")

