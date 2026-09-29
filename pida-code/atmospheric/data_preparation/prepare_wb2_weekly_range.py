#!/usr/bin/env python3
"""
Prepare multi-year weekly WeatherBench2 / ERA5 data for PIDA experiments.

Output layout:
    Y.npy                [T, H, W] float32, memory-mappable dense weekly maps
    times.npy            [T] datetime64[ns]
    metadata.json
    splits.json          chronological 40-year train / 10-year test by default

The script writes Y.npy incrementally in time chunks so the 50-year high-res
case does not need to fit fully in RAM.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


DEFAULT_DATASET = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weekly WB2 range for PIDA.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--start_year", type=int, default=1970)
    parser.add_argument("--train_years", type=int, default=40)
    parser.add_argument("--test_years", type=int, default=10)
    parser.add_argument("--variable", default="temperature")
    parser.add_argument("--level", type=float, default=500)
    parser.add_argument("--sample_weekday", type=int, default=0, help="0=Monday, 6=Sunday.")
    parser.add_argument("--sample_hour", type=int, default=0, choices=[0, 6, 12, 18])
    parser.add_argument("--time_chunk", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def open_dataset(dataset_path: str):
    import xarray as xr

    if dataset_path.startswith("gs://"):
        import fsspec

        mapper = fsspec.get_mapper(dataset_path, token="anon")
        return xr.open_zarr(mapper, consolidated=True)
    return xr.open_zarr(dataset_path, consolidated=True)


def normalize_dim_names(da):
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    return da


def select_variable(ds, variable: str, level: Optional[float]):
    if variable not in ds:
        available = ", ".join(list(ds.data_vars)[:50])
        raise KeyError(f"Variable {variable!r} not found. Available examples: {available}")
    da = ds[variable]
    level_dim = next((dim for dim in ("level", "pressure_level") if dim in da.dims), None)
    if level_dim is not None and level is not None:
        da = da.sel({level_dim: level}, method=None)
    da = normalize_dim_names(da)
    missing = set(("time", "lat", "lon")) - set(da.dims)
    if missing:
        raise ValueError(f"Expected time/lat/lon dimensions, missing {sorted(missing)} from {da.dims}")
    return da.transpose("time", "lat", "lon").astype("float32")


def weekly_times(start_year: int, end_year_exclusive: int, sample_weekday: int, sample_hour: int) -> np.ndarray:
    if sample_weekday < 0 or sample_weekday > 6:
        raise ValueError("sample_weekday must be in [0, 6]")
    start = np.datetime64(f"{start_year:04d}-01-01T{sample_hour:02d}:00:00", "h")
    end = np.datetime64(f"{end_year_exclusive:04d}-01-01T{sample_hour:02d}:00:00", "h")
    start_day = start.astype("datetime64[D]")
    monday_ref = np.datetime64("1970-01-05", "D")
    start_weekday = int((start_day - monday_ref).astype(int) % 7)
    offset = (sample_weekday - start_weekday) % 7
    first = start + np.timedelta64(offset, "D")
    return np.arange(first, end, np.timedelta64(7, "D"), dtype="datetime64[h]").astype("datetime64[ns]")


def split_indices(times: np.ndarray, start_year: int, train_years: int) -> dict:
    train_end = np.datetime64(f"{start_year + train_years:04d}-01-01T00:00:00", "ns")
    train_idx = np.flatnonzero(times < train_end).astype(np.int64)
    test_idx = np.flatnonzero(times >= train_end).astype(np.int64)
    return {"train": train_idx.tolist(), "test": test_idx.tolist()}


def year_counts(times: np.ndarray) -> dict:
    years = [int(str(t.astype("datetime64[Y]")).split("-")[0]) for t in times]
    out = {}
    for year in years:
        out[str(year)] = out.get(str(year), 0) + 1
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    end_year = args.start_year + args.train_years + args.test_years
    requested_times = weekly_times(args.start_year, end_year, args.sample_weekday, args.sample_hour)

    print(f"Opening dataset: {args.dataset_path}", flush=True)
    ds = open_dataset(args.dataset_path)
    print("Dataset dimensions:", ds.dims, flush=True)
    da = select_variable(ds, args.variable, args.level)
    available_times = np.intersect1d(da["time"].values.astype("datetime64[ns]"), requested_times)
    if available_times.size == 0:
        raise RuntimeError("No requested weekly sample times exist in the selected dataset.")
    da = da.sel(time=available_times)
    height = int(da.sizes["lat"])
    width = int(da.sizes["lon"])
    total = int(available_times.size)

    print(f"Requested weekly samples: {len(requested_times)}", flush=True)
    print(f"Available weekly samples: {total}", flush=True)
    print(
        "First/last:",
        np.datetime_as_string(available_times[0], unit="h"),
        np.datetime_as_string(available_times[-1], unit="h"),
        flush=True,
    )
    print(f"Writing Y.npy with shape {(total, height, width)}", flush=True)

    y_path = out / "Y.npy"
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype="float32", shape=(total, height, width))
    for start in range(0, total, args.time_chunk):
        stop = min(start + args.time_chunk, total)
        chunk = da.isel(time=slice(start, stop)).compute().values.astype("float32")
        y_out[start:stop] = chunk
        y_out.flush()
        print(f"  wrote samples {start}:{stop}", flush=True)
    del y_out

    times = da["time"].values.astype("datetime64[ns]")
    np.save(out / "times.npy", times)
    splits = split_indices(times, args.start_year, args.train_years)
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    metadata = {
        "dataset_path": args.dataset_path,
        "start_year": args.start_year,
        "end_year_exclusive": end_year,
        "train_years": args.train_years,
        "test_years": args.test_years,
        "variable": args.variable,
        "level": args.level,
        "sample_weekday": args.sample_weekday,
        "sample_hour": args.sample_hour,
        "time_chunk": args.time_chunk,
        "num_samples": total,
        "resolution": [height, width],
        "Y_shape": [total, height, width],
        "times_shape": list(times.shape),
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
        "year_counts": year_counts(times),
        "first_time": np.datetime_as_string(times[0], unit="h"),
        "last_time": np.datetime_as_string(times[-1], unit="h"),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved weekly range data to:", out, flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
