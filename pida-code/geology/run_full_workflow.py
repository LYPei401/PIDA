#!/usr/bin/env python3
"""Create a PIDA request and run a separately installed inversion backend."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_POSITIONS = (5, 16, 27, 38, 50, 61, 72, 83, 95)
DATA_ENV = {
    "train_data_root": "PIDA_TRAIN_DATA_ROOT",
    "train_label_root": "PIDA_TRAIN_LABEL_ROOT",
    "test_data_root": "PIDA_TEST_DATA_ROOT",
    "test_label_root": "PIDA_TEST_LABEL_ROOT",
    "annotation_path": "PIDA_ANNO_PATH",
}


def read_deployment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    required = ("monitoring_range", "source_indices", "selected_receiver_count")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"deployment {path} is missing: {', '.join(missing)}")
    monitoring = payload["monitoring_range"]
    if not isinstance(monitoring, dict) or not {"start", "end"} <= set(monitoring):
        raise ValueError(f"deployment {path} has an invalid monitoring_range")
    return payload


def default_deployment(method: str, target: str) -> Path | None:
    if method == "pida_one":
        return APP_ROOT / "case_study" / "configs" / "deployments" / "pida_one" / f"{target}.json"
    return None


def build_deployment(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if args.method == "full":
        deployment: dict[str, Any] = {
            "method": "full",
            "target": args.target,
            "monitoring_range": {"start": 0, "end": 400},
            "source_indices": list(range(len(DEFAULT_SOURCE_POSITIONS))),
            "selected_receiver_count": 101,
        }
    else:
        path = args.deployment_json or default_deployment(args.method, args.target)
        if path is None:
            parser.error(f"--method {args.method} requires --deployment-json")
        if not path.is_file():
            parser.error(f"deployment JSON does not exist: {path}")
        try:
            deployment = read_deployment(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        deployment = dict(deployment)
        deployment["method"] = args.method
        deployment["target"] = args.target

    if args.source_number is not None:
        current = list(deployment["source_indices"])
        if args.source_number > len(current):
            parser.error("--source-number cannot exceed the deployment's source-index count")
        deployment["source_indices"] = current[: args.source_number]
    if args.sensor_number is not None:
        deployment["selected_receiver_count"] = args.sensor_number
    return deployment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("pida_one", "pida_logn", "full"), required=True)
    parser.add_argument("--target", required=True, help="target such as sim0878_t60")
    parser.add_argument("--deployment-json", type=Path, help="PIDA deployment record; required for pida_logn")
    parser.add_argument("--source-number", type=int, help="use the first N source indices from the deployment")
    parser.add_argument("--sensor-number", type=int, help="override the selected receiver count")
    parser.add_argument("--epoch-block", type=int, default=40)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=APP_ROOT / "results" / "full_runs")
    parser.add_argument("--skip-train", action="store_true", help="request inference only")
    parser.add_argument("--checkpoint-path", type=Path, help="external checkpoint; valid with --skip-train")
    parser.add_argument("--case-input", type=Path, help="optional single-case NPZ passed to the backend")
    parser.add_argument(
        "--backend-command",
        default=os.environ.get("PIDA_INVERSION_BACKEND"),
        help="external executable prefix; it will receive --request-json PATH",
    )
    parser.add_argument("--dry-run", action="store_true", help="write and print the request without running a backend")
    args = parser.parse_args()

    if not re.fullmatch(r"sim\d+_t\d+", args.target):
        parser.error("--target must look like sim0878_t60")
    if args.checkpoint_path is not None and not args.skip_train:
        parser.error("--checkpoint-path is only valid with --skip-train")
    if args.skip_train and args.checkpoint_path is None:
        parser.error("--skip-train requires an external --checkpoint-path")
    if not args.backend_command and not args.dry_run:
        parser.error("set --backend-command or PIDA_INVERSION_BACKEND")
    if not args.dry_run:
        for label, path in (("checkpoint", args.checkpoint_path), ("case input", args.case_input)):
            if path is not None and not path.is_file():
                parser.error(f"external {label} does not exist: {path}")
        if not args.skip_train:
            missing = [env_name for env_name in DATA_ENV.values() if not os.environ.get(env_name)]
            if missing:
                parser.error("training requires external data paths: " + ", ".join(missing))

    deployment = build_deployment(args, parser)
    run_dir = args.output_root.resolve() / args.method / args.target
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "backend_request.json"
    request = {
        "schema": "pida.inversion-backend-request",
        "schema_version": 1,
        "operation": "infer" if args.skip_train else "train_and_infer",
        "model_family": "InversionNet",
        "method": args.method,
        "target": args.target,
        "deployment": deployment,
        "data": {key: os.environ.get(env_name) for key, env_name in DATA_ENV.items()},
        "case_input": str(args.case_input.resolve()) if args.case_input else None,
        "checkpoint": str(args.checkpoint_path.resolve()) if args.checkpoint_path else None,
        "training": {"epoch_block": args.epoch_block, "num_block": args.num_block, "seed": args.seed},
        "runtime": {"device": args.device},
        "output_dir": str(run_dir),
        "expected_result": str(run_dir / "backend_result.json"),
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"request": str(request_path), "payload": request}, indent=2, sort_keys=True))

    if args.dry_run:
        return 0
    command = [*shlex.split(args.backend_command), "--request-json", str(request_path)]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return completed.returncode
    result_path = run_dir / "backend_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"backend succeeded but did not write {result_path}")
    result = json.loads(result_path.read_text())
    if result.get("schema") != "pida.inversion-backend-result" or result.get("schema_version") != 1:
        raise RuntimeError(f"backend wrote an incompatible result schema: {result_path}")
    if result.get("status") not in {"ok", "failed"} or not isinstance(result.get("metrics"), dict):
        raise RuntimeError(f"backend result is missing status/metrics: {result_path}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
