#!/usr/bin/env python3
"""Check that the geological source tree is complete and contains no model weights or data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CASE_ROOT.parent


def required_files(mode: str) -> list[Path]:
    required = [
        APP_ROOT.parent / "LICENSE",
        APP_ROOT / "run_full_workflow.py",
        APP_ROOT / "interfaces" / "backend_request.schema.json",
        APP_ROOT / "interfaces" / "backend_result.schema.json",
        APP_ROOT / "pida_pred" / "infer_next_velocity_and_range.py",
        APP_ROOT / "pida_logn" / "pida_logn.py",
        CASE_ROOT / "configs" / "backend_manifest.json",
        CASE_ROOT / "configs" / "deployments" / "pida_one" / "sim0878_t60.json",
        CASE_ROOT / "configs" / "deployments" / "pida_logn" / "sim0878_t60.json",
    ]
    if mode == "quick":
        required.extend(
            [
                CASE_ROOT / "configs" / "predicted_ranges.csv",
                CASE_ROOT / "results" / "replay" / "sim0878_t60" / "metrics.json",
            ]
        )
    return required


def forbidden_files() -> list[Path]:
    forbidden: list[Path] = []
    for directory in (APP_ROOT / "pida_ml", CASE_ROOT / "checkpoints"):
        if directory.exists():
            forbidden.append(directory)
    binary_suffixes = {".pth", ".pt", ".npy", ".npz", ".pyc"}
    forbidden.extend(path for path in APP_ROOT.parent.rglob("*") if path.is_file() and path.suffix.lower() in binary_suffixes)
    return forbidden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "quick"), default="source")
    args = parser.parse_args()
    missing = [path for path in required_files(args.mode) if not path.is_file() or path.stat().st_size == 0]
    forbidden = forbidden_files()
    try:
        manifest = json.loads((CASE_ROOT / "configs" / "backend_manifest.json").read_text())
        valid_manifest = manifest.get("schema_version") == 1 and manifest.get("backend") == "external"
    except (OSError, json.JSONDecodeError):
        valid_manifest = False
    if not valid_manifest:
        missing.append(CASE_ROOT / "configs" / "backend_manifest.json")

    if missing or forbidden:
        print("SOURCE RELEASE CHECK: FAILED")
        for path in missing:
            print(f"  missing/invalid: {path.relative_to(APP_ROOT.parent)}")
        for path in forbidden:
            print(f"  forbidden vendored artifact: {path.relative_to(APP_ROOT.parent)}")
        return 1
    print(f"SOURCE RELEASE CHECK: PASSED ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
