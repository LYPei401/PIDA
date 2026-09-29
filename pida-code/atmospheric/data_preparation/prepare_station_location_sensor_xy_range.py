#!/usr/bin/env python3
"""Create X sensors from dense WeatherBench Y at real METAR station grid cells."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_Y_DIR = os.environ.get("PIDA_ATMOSPHERIC_Y_DIR", "data/weekly")
DEFAULT_STATION_CATALOG = os.environ.get("PIDA_ATMOSPHERIC_STATION_CATALOG", "data/station_catalog.csv")
DEFAULT_OUTPUT_DIR = os.environ.get("PIDA_ATMOSPHERIC_OUTPUT_DIR", "data/station_case")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_dir", default=DEFAULT_Y_DIR)
    parser.add_argument("--station_catalog", default=DEFAULT_STATION_CATALOG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid_height", type=int, default=721)
    parser.add_argument("--grid_width", type=int, default=1440)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample limit. Default uses all Y samples.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of time samples to convert per chunk.",
    )
    return parser.parse_args()


def load_sensor_grid_catalog(catalog_path: Path, height: int, width: int) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    grid_col = f"grid_id_{height}x{width}"
    lat_col = f"lat_idx_{height}x{width}"
    lon_col = f"lon_idx_{height}x{width}"
    required = {grid_col, lat_col, lon_col}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Station catalog missing required columns: {sorted(missing)}")

    sensors = (
        catalog[[grid_col, lat_col, lon_col]]
        .drop_duplicates()
        .sort_values(grid_col)
        .reset_index(drop=True)
    )
    sensors["sensor_index"] = np.arange(len(sensors), dtype=np.int64)
    sensors = sensors.rename(
        columns={
            grid_col: "grid_id",
            lat_col: "lat_idx",
            lon_col: "lon_idx",
        }
    )
    return sensors[["sensor_index", "grid_id", "lat_idx", "lon_idx"]]


def subset_splits(splits: dict, num_samples: int) -> dict:
    output = {}
    for split_name, values in splits.items():
        arr = np.asarray(values, dtype=np.int64)
        arr = arr[arr < num_samples]
        output[split_name] = arr.astype(int).tolist()
    return output


def write_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_x(
    y: np.ndarray,
    output_path: Path,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    num_samples: int,
    chunk_size: int,
) -> None:
    x = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, len(lat_idx)),
    )
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        x[start:end] = y[start:end, lat_idx, lon_idx].astype(np.float32)
        print(f"Wrote X rows {start}:{end}", flush=True)
    x.flush()


def main() -> int:
    args = parse_args()
    y_dir = Path(args.y_dir)
    output_dir = Path(args.output_dir)
    station_catalog = Path(args.station_catalog)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_path = y_dir / "Y.npy"
    times_path = y_dir / "times.npy"
    splits_path = y_dir / "splits.json"
    metadata_path = y_dir / "metadata.json"
    for path in [y_path, times_path, splits_path, metadata_path, station_catalog]:
        if not path.exists():
            raise FileNotFoundError(path)

    y = np.load(y_path, mmap_mode="r")
    times = np.load(times_path)
    if y.ndim != 3:
        raise ValueError(f"Expected Y shape [T,H,W], got {y.shape}")
    if y.shape[1:] != (args.grid_height, args.grid_width):
        raise ValueError(
            f"Y grid shape {y.shape[1:]} does not match requested "
            f"{(args.grid_height, args.grid_width)}"
        )

    num_samples = int(y.shape[0])
    if args.max_samples is not None:
        num_samples = min(num_samples, int(args.max_samples))
    times = times[:num_samples]

    sensors = load_sensor_grid_catalog(station_catalog, args.grid_height, args.grid_width)
    sensor_locations = sensors[["grid_id", "lat_idx", "lon_idx"]].to_numpy(dtype=np.int64)
    lat_idx = sensor_locations[:, 1].astype(np.int64)
    lon_idx = sensor_locations[:, 2].astype(np.int64)

    build_x(
        y=y,
        output_path=output_dir / "X.npy",
        lat_idx=lat_idx,
        lon_idx=lon_idx,
        num_samples=num_samples,
        chunk_size=args.chunk_size,
    )

    x_mask = np.lib.format.open_memmap(
        output_dir / "X_mask.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(num_samples, len(sensor_locations)),
    )
    x_mask[:] = 1
    x_mask.flush()

    np.save(output_dir / "times.npy", times)
    np.save(output_dir / "sensor_locations.npy", sensor_locations)
    np.save(output_dir / "y_source_indices.npy", np.arange(num_samples, dtype=np.int64))
    sensors.to_csv(output_dir / "sensor_grid_catalog.csv", index=False)
    write_symlink_or_copy(y_path, output_dir / "Y_source.npy")

    source_splits = json.loads(splits_path.read_text())
    splits = subset_splits(source_splits, num_samples)
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True))
    for split_name, values in splits.items():
        np.save(output_dir / f"{split_name}_indices.npy", np.asarray(values, dtype=np.int64))

    source_metadata = json.loads(metadata_path.read_text())
    metadata = {
        "description": (
            "X is simulated sensor data sampled from Y at real METAR station grid-cell "
            "locations. Sensor stations are assumed to exist for the full Y time range."
        ),
        "y_dir": str(y_dir),
        "y_source": str(y_path),
        "station_catalog": str(station_catalog),
        "grid_shape": [args.grid_height, args.grid_width],
        "num_samples": num_samples,
        "num_sensor_grid_cells": int(len(sensor_locations)),
        "X_shape": [num_samples, int(len(sensor_locations))],
        "X_mask_shape": [num_samples, int(len(sensor_locations))],
        "Y_shape": [num_samples, args.grid_height, args.grid_width],
        "Y_storage": "Y_source.npy symlink/copy plus y_source_indices.npy",
        "first_time": str(times[0]),
        "last_time": str(times[-1]),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "source_metadata": source_metadata,
        "sensor_value_definition": "X[t, k] = Y[t, lat_idx[k], lon_idx[k]]",
        "X_mask_definition": "All ones; every assumed station has a value at every sampled time.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    # Small correctness check.
    x = np.load(output_dir / "X.npy", mmap_mode="r")
    check_rows = [0, num_samples - 1] if num_samples > 1 else [0]
    max_abs_error = 0.0
    for row in check_rows:
        expected = y[row, lat_idx, lon_idx]
        err = np.max(np.abs(x[row].astype(np.float32) - expected.astype(np.float32)))
        max_abs_error = max(max_abs_error, float(err))
    metadata["x_y_sampling_check_max_abs_error"] = max_abs_error
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print("Wrote station-location sensor X/Y dataset to", output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
