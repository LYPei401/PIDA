#!/usr/bin/env python3
"""Stage launcher for the reproducible atmospheric PIDA workflow.

Usage: python run_full_workflow.py <stage> -- <stage arguments>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP = ROOT
PACKAGE = APP

STAGES = {
    "download-station-sample": ["-m", "data_preparation.download_wb2_metar_docs_sample"],
    "analyze-stations": ["-m", "data_preparation.analyze_wb2_metar_stations"],
    "prepare-field": ["-m", "data_preparation.prepare_wb2_weekly_range"],
    "prepare-sensors": ["-m", "data_preparation.prepare_station_location_sensor_xy_range"],
    "train-full-baseline": [str(ROOT / "pida_ml" / "train_full_baseline.py")],
    "target-adaptive": [str(ROOT / "pida_pred" / "target_adaptive_pida" / "run_target_adaptive.py")],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER, help="Arguments for the selected stage; prefix with -- when needed.")
    args = parser.parse_args()
    forwarded = args.stage_args[1:] if args.stage_args[:1] == ["--"] else args.stage_args
    command = [sys.executable, *STAGES[args.stage], *forwarded]
    env = {**os.environ, "PYTHONPATH": str(APP) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
