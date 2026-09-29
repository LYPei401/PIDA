#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pida_ml.full_baseline import run


if __name__ == "__main__":
    run("B", "B: residual correction from prior + sensor innovation")
