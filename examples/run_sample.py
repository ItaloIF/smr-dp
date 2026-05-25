"""Minimal SMR-DP sample inference.

Run from the repository root:

    python3 examples/run_sample.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_matrix import run_demo


if __name__ == "__main__":
    run_demo(trace_index=3, save_figures=False)
