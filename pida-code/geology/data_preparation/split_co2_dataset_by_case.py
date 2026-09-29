#!/usr/bin/env python3
"""Split downloaded Kimberlina-CO2 NPZ files by simulation case.

All time steps of one ``simXXXX`` case go to the same split, preventing
temporal leakage.  The default is a dry run; add ``--apply`` to write files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

PATTERN = re.compile(r"^(data|label)_sim(\d+)_t\d+\.npz$")


def destination(root: Path, kind: str, split: str) -> Path:
    """Return the standardized folder for one split and file type."""
    return root / f"kimberlina_co2_{split}_{'data' if kind == 'data' else 'label'}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, action="append", required=True,
        help="raw download directory; repeat for every downloaded folder",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--train-max-case", type=int, default=800,
        help="simulation IDs <= this value are training cases",
    )
    p.add_argument(
        "--test-case", action="append", default=[], type=int,
        help="hold out this simulation case for testing; repeat as needed",
    )
    p.add_argument("--mode", choices=("copy", "move", "symlink"), default="symlink")
    p.add_argument(
        "--apply", action="store_true",
        help="perform writes; otherwise only print the planned split",
    )
    args = p.parse_args()

    # Key by filename because the release naming convention encodes both case
    # and time. This detects accidental duplicate downloads before writing.
    files: dict[str, Path] = {}
    ignored = 0
    for source in args.source:
        if not source.is_dir():
            raise NotADirectoryError(f"source is not a directory: {source}")
        for path in source.rglob("*.npz"):
            match = PATTERN.fullmatch(path.name)
            if not match:
                ignored += 1
                continue
            if path.name in files and files[path.name].resolve() != path.resolve():
                raise ValueError(f"duplicate filename: {path.name}")
            files[path.name] = path

    counts = Counter()
    for name, path in sorted(files.items()):
        kind, case_text = PATTERN.fullmatch(name).groups()
        case = int(case_text)
        split = "test" if case in args.test_case else ("train" if case <= args.train_max_case else "test")
        out = destination(args.output_root, kind, split) / name
        counts[(split, kind)] += 1
        if args.apply:
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() or out.is_symlink():
                out.unlink()
            if args.mode == "copy":
                shutil.copy2(path, out)
            elif args.mode == "move":
                shutil.move(path, out)
            else:
                os.symlink(path.resolve(), out)

    manifest = {
        "sources": [str(x) for x in args.source],
        "train_max_case": args.train_max_case,
        "held_out_test_cases": sorted(args.test_case),
        "mode": args.mode,
        "applied": args.apply,
        "counts": {f"{split}_{kind}": count for (split, kind), count in sorted(counts.items())},
        "ignored_npz": ignored,
    }
    print(json.dumps(manifest, indent=2))
    if args.apply:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
