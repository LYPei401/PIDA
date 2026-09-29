#!/usr/bin/env python3
"""Run the released deployment through an external InversionNet backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = CASE_ROOT.parent
TARGET = "sim0878_t60"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=(TARGET,), default=TARGET)
    parser.add_argument("--input-data", required=True, type=Path, help="external Kimberlina case NPZ")
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="external InversionNet checkpoint")
    parser.add_argument(
        "--backend-command",
        default=os.environ.get("PIDA_INVERSION_BACKEND"),
        help="external executable prefix; it will receive --request-json PATH",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=CASE_ROOT / "results" / "external_replay")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.backend_command and not args.dry_run:
        parser.error("set --backend-command or PIDA_INVERSION_BACKEND")
    command = [
        sys.executable,
        str(APP_ROOT / "run_full_workflow.py"),
        "--method",
        "pida_one",
        "--target",
        args.target,
        "--skip-train",
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--case-input",
        str(args.input_data),
        "--device",
        args.device,
        "--output-root",
        str(args.output_dir),
    ]
    if args.backend_command:
        command.extend(["--backend-command", args.backend_command])
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, cwd=APP_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
