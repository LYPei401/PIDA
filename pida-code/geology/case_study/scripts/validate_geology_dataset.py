#!/usr/bin/env python3
"""Validate the Kimberlina-CO2 directory layout without loading it into RAM."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def files(path: Path, prefix: str) -> list[Path]:
    return sorted(path.glob(f"{prefix}*.npz"))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True, type=Path)
    p.add_argument("--train-label", required=True, type=Path)
    p.add_argument("--test-data", required=True, type=Path)
    p.add_argument("--test-label", required=True, type=Path)
    p.add_argument("--target", default="sim0878")
    a = p.parse_args()
    roots = ((a.train_data, "data"), (a.train_label, "label"), (a.test_data, "data"), (a.test_label, "label"))
    for root, prefix in roots:
        found = files(root, prefix)
        if not found:
            raise SystemExit(f"No {prefix}_*.npz files in {root}")
        with np.load(found[0]) as sample:
            array = sample[sample.files[0]]
        print(f"OK {root}: {len(found)} files; sample={array.shape} {array.dtype}")
    for step in (60, 100, 140, 180):
        suffix = f"{a.target}_t{step}"
        data = list(a.test_data.glob(f"data_{suffix}*.npz"))
        label = list(a.test_label.glob(f"label_{suffix}*.npz"))
        print(f"{suffix}: data={len(data)} label={len(label)}")
        if len(data) != 1 or len(label) != 1:
            raise SystemExit(f"Missing or ambiguous Case-5 pair for {suffix}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
