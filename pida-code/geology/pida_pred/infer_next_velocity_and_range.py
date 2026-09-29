#!/usr/bin/env python3
"""Predict a future velocity map and its leakage region with PIDA-Pred.

Given V(t-2) and V(t-1), PIDA-Pred estimates V_hat(t).  A pixel belongs to
the predicted leakage region when its velocity change from a no-leak/reference
map is at least ``--delta-threshold``.  The script writes both the full binary
mask and the horizontal interval needed by PIDA-Opt.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

try:  # Package import for orchestration/tests; fallback supports direct execution.
    from .train_groundtruth_pida_pred import LABEL_MAX, LABEL_MIN, PidaPredVelocityUNet
except ImportError:
    from train_groundtruth_pida_pred import LABEL_MAX, LABEL_MIN, PidaPredVelocityUNet


def read_velocity(path: Path) -> np.ndarray:
    """Read one velocity map from an NPZ label or an NPY PIDA-ML output."""
    loaded = np.load(path)
    if isinstance(loaded, np.ndarray):
        return loaded.astype(np.float32)
    with loaded as archive:
        return archive[archive.files[0]].astype(np.float32)


def resolve_input_paths(args) -> tuple[Path, Path, Path]:
    """Allow either explicit files or a concise ``--label-root --target`` call."""
    if args.label_root is not None or args.target is not None:
        if args.label_root is None or args.target is None:
            raise ValueError("use --label-root and --target together")
        match = re.fullmatch(r"(sim\d+)_t(\d+)", args.target)
        if not match:
            raise ValueError("--target must look like sim0787_t60")
        case, time_text = match.groups()
        target_time = int(time_text)
        if target_time < 30:
            raise ValueError("PIDA-Pred requires a target time of at least t30")
        root = args.label_root
        paths = (
            root / f"label_{case}_t{target_time - 20}.npz",
            root / f"label_{case}_t{target_time - 10}.npz",
            root / f"label_{case}_t{args.reference_time}.npz",
        )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing PIDA-Pred label inputs: " + ", ".join(missing))
        return paths
    if None in (args.t_minus_2, args.t_minus_1, args.reference):
        raise ValueError("provide explicit --t-minus-2/--t-minus-1/--reference or --label-root --target")
    return args.t_minus_2, args.t_minus_1, args.reference


def derive_leakage_region(
    prediction: np.ndarray, reference: np.ndarray, delta_threshold: float
) -> tuple[np.ndarray, dict]:
    """Threshold the physical velocity change and summarize its support.

    ``mask`` has the same ``[depth, horizontal]`` shape as the predicted
    velocity map.  The horizontal interval is deliberately inclusive because
    it is consumed as a physical search range by the following PIDA-Opt stage.
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} differs from reference shape {reference.shape}"
        )
    if delta_threshold <= 0:
        raise ValueError("delta_threshold must be positive")

    change = np.abs(prediction - reference)
    mask = change >= delta_threshold
    rows, columns = np.where(mask)
    if len(columns) == 0:
        interval = None
        bounding_box = None
    else:
        interval = [int(columns.min()), int(columns.max())]
        bounding_box = {
            "row_start": int(rows.min()), "row_end": int(rows.max()),
            "column_start": int(columns.min()), "column_end": int(columns.max()),
        }
    summary = {
        "delta_threshold": float(delta_threshold),
        "changed_pixels": int(mask.sum()),
        "horizontal_interval_columns": interval,
        "bounding_box": bounding_box,
        "max_absolute_velocity_change": float(change.max()),
    }
    return mask.astype(np.uint8), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--t-minus-2", type=Path)
    parser.add_argument("--t-minus-1", type=Path)
    parser.add_argument(
        "--reference", type=Path,
        help="no-leak or baseline velocity map used to define leakage",
    )
    parser.add_argument("--label-root", type=Path, help="organized label directory; use with --target")
    parser.add_argument("--target", help="target such as sim0787_t60; derives t-20/t-10 input files")
    parser.add_argument("--reference-time", type=int, default=10, help="baseline label time used with --label-root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--delta-threshold", required=True, type=float,
        help="minimum absolute physical velocity change for one leakage pixel",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = PidaPredVelocityUNet(int(state["base_channels"])).to(args.device)
    model.load_state_dict(state["model_state"])
    model.eval()

    t_minus_2, t_minus_1, reference_path = resolve_input_paths(args)
    previous2 = read_velocity(t_minus_2)
    previous1 = read_velocity(t_minus_1)
    reference = read_velocity(reference_path)
    normalized = 2.0 * np.stack([previous2, previous1])[None] / (LABEL_MAX - LABEL_MIN)
    normalized -= (LABEL_MAX + LABEL_MIN) / (LABEL_MAX - LABEL_MIN)
    with torch.no_grad():
        predicted_normalized = model(torch.from_numpy(normalized).to(args.device)).cpu().numpy()[0, 0]
    prediction = ((predicted_normalized + 1.0) * (LABEL_MAX - LABEL_MIN) / 2.0 + LABEL_MIN).astype(np.float32)

    mask, summary = derive_leakage_region(prediction, reference, args.delta_threshold)
    provenance = {
        **summary,
        "reference": str(reference_path),
        "checkpoint": str(args.checkpoint),
        "inputs": {"t_minus_2": str(t_minus_2), "t_minus_1": str(t_minus_1)},
    }
    np.save(args.output_dir / "predicted_velocity.npy", prediction)
    np.save(args.output_dir / "predicted_leakage_mask.npy", mask)
    (args.output_dir / "predicted_leakage_region.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Compatibility alias for downstream code that only expects the range.
    (args.output_dir / "predicted_range.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
