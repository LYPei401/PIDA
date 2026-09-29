#!/usr/bin/env python3
"""Analyze local WeatherBench2 METAR samples and map stations to WB2 grids."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = "data/metar_sample"
DEFAULT_OUTPUT_DIR = "results/station_analysis"

STATION_COLUMNS = [
    "stationName",
    "locationName",
    "latitude",
    "longitude",
    "elevation",
    "timeObs",
    "timeNominal",
    "temperature",
    "reportType",
    "autoStationType",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parquet_glob",
        default="metar/**/*.parquet",
        help="Glob under input_dir. Default analyzes all downloaded METAR parquet files.",
    )
    parser.add_argument(
        "--lat_order",
        choices=("ascending", "descending"),
        default="ascending",
        help=(
            "Grid row orientation for generated indices. Use ascending for row 0=-90, "
            "descending for row 0=90."
        ),
    )
    return parser.parse_args()


def normalize_lon_0_360(lon: pd.Series) -> pd.Series:
    return lon.mod(360.0)


def map_to_grid(df: pd.DataFrame, height: int, width: int, lat_order: str) -> pd.DataFrame:
    lat = df["latitude"].astype(float).clip(-90.0, 90.0)
    lon = normalize_lon_0_360(df["longitude"].astype(float))

    if height == 1:
        lat_idx = pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)
    elif lat_order == "ascending":
        lat_idx = np.rint((lat + 90.0) / 180.0 * (height - 1)).astype(np.int64)
    else:
        lat_idx = np.rint((90.0 - lat) / 180.0 * (height - 1)).astype(np.int64)

    lon_idx = np.floor(lon / 360.0 * width).astype(np.int64).clip(0, width - 1)
    lat_idx = pd.Series(lat_idx, index=df.index).clip(0, height - 1).astype(np.int64)

    mapped = pd.DataFrame(index=df.index)
    mapped[f"lat_idx_{height}x{width}"] = lat_idx
    mapped[f"lon_idx_{height}x{width}"] = lon_idx
    mapped[f"grid_id_{height}x{width}"] = lat_idx * width + lon_idx
    return mapped


def read_station_frame(path: Path) -> pd.DataFrame:
    schema_cols = set(pq.read_schema(path).names)
    use_cols = [col for col in STATION_COLUMNS if col in schema_cols]
    if not use_cols:
        # Fall back to reading the full file if a future sample uses unexpected
        # field names. The downloaded samples are small enough for this job.
        full = pd.read_parquet(path, engine="pyarrow")
        use_cols = [col for col in STATION_COLUMNS if col in full.columns]
        return full[use_cols].copy()
    return pd.read_parquet(path, engine="pyarrow", columns=use_cols)


def file_summary(path: Path, df: pd.DataFrame) -> dict:
    temp = df["temperature"] if "temperature" in df.columns else pd.Series(dtype=float)
    valid_temp = temp.notna() if len(temp) else pd.Series(dtype=bool)
    return {
        "file": str(path),
        "rows": int(len(df)),
        "unique_stations": int(df["stationName"].nunique()) if "stationName" in df else 0,
        "temperature_non_null": int(valid_temp.sum()) if len(temp) else 0,
        "temperature_non_null_fraction": float(valid_temp.mean()) if len(temp) else None,
        "timeObs_min": str(df["timeObs"].min()) if "timeObs" in df and len(df) else None,
        "timeObs_max": str(df["timeObs"].max()) if "timeObs" in df and len(df) else None,
        "timeNominal_min": str(df["timeNominal"].min()) if "timeNominal" in df and len(df) else None,
        "timeNominal_max": str(df["timeNominal"].max()) if "timeNominal" in df and len(df) else None,
        "lat_min": float(df["latitude"].min()) if "latitude" in df and len(df) else None,
        "lat_max": float(df["latitude"].max()) if "latitude" in df and len(df) else None,
        "lon_min": float(df["longitude"].min()) if "longitude" in df and len(df) else None,
        "lon_max": float(df["longitude"].max()) if "longitude" in df and len(df) else None,
    }


def build_station_catalog(frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.dropna(subset=["stationName", "latitude", "longitude"])
    all_rows["has_temperature"] = all_rows["temperature"].notna() if "temperature" in all_rows else False

    agg_spec = {
        "locationName": lambda x: x.dropna().iloc[0] if len(x.dropna()) else "",
        "latitude": "median",
        "longitude": "median",
        "elevation": "median",
        "has_temperature": "sum",
    }
    if "timeObs" in all_rows:
        agg_spec["timeObs"] = ["min", "max", "count"]
    if "timeNominal" in all_rows:
        agg_spec["timeNominal"] = ["min", "max", "count"]

    grouped = all_rows.groupby("stationName", dropna=True).agg(agg_spec)
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns
    ]
    grouped = grouped.reset_index()
    grouped = grouped.rename(
        columns={
            "locationName_<lambda>": "locationName",
            "latitude_median": "latitude",
            "longitude_median": "longitude",
            "elevation_median": "elevation",
            "has_temperature_sum": "temperature_obs_count",
            "timeObs_count": "rows_with_timeObs",
            "timeNominal_count": "rows_with_timeNominal",
        }
    )
    if "timeObs_min" in grouped:
        grouped["timeObs_min"] = grouped["timeObs_min"].astype(str)
        grouped["timeObs_max"] = grouped["timeObs_max"].astype(str)
    if "timeNominal_min" in grouped:
        grouped["timeNominal_min"] = grouped["timeNominal_min"].astype(str)
        grouped["timeNominal_max"] = grouped["timeNominal_max"].astype(str)
    return grouped


def add_grid_mappings(catalog: pd.DataFrame, lat_order: str) -> pd.DataFrame:
    mapped = catalog.copy()
    for height, width in [(721, 1440), (32, 64)]:
        grid_cols = map_to_grid(mapped, height=height, width=width, lat_order=lat_order)
        mapped = pd.concat([mapped, grid_cols], axis=1)
    return mapped


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob(args.parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched {input_dir / args.parquet_glob}")

    summaries = []
    frames = []
    for path in parquet_files:
        print(f"Reading {path}", flush=True)
        df = read_station_frame(path)
        summaries.append(file_summary(path, df))
        frames.append(df)

    catalog = build_station_catalog(frames)
    catalog = add_grid_mappings(catalog, args.lat_order)
    catalog = catalog.sort_values(
        ["temperature_obs_count", "stationName"], ascending=[False, True]
    ).reset_index(drop=True)

    per_file = pd.DataFrame(summaries)
    grid_summary = {}
    for height, width in [(721, 1440), (32, 64)]:
        grid_col = f"grid_id_{height}x{width}"
        grid_summary[f"{height}x{width}"] = {
            "unique_grid_cells_with_station": int(catalog[grid_col].nunique()),
            "stations": int(len(catalog)),
            "max_stations_in_one_cell": int(catalog.groupby(grid_col).size().max()),
            "cells_with_collisions": int((catalog.groupby(grid_col).size() > 1).sum()),
        }

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "parquet_glob": args.parquet_glob,
        "lat_order": args.lat_order,
        "num_parquet_files": len(parquet_files),
        "total_rows": int(per_file["rows"].sum()),
        "unique_stations": int(len(catalog)),
        "temperature_station_count": int((catalog["temperature_obs_count"] > 0).sum()),
        "grid_summary": grid_summary,
        "files": summaries,
    }

    catalog_path = output_dir / "station_catalog_mapped.csv"
    per_file_path = output_dir / "per_file_summary.csv"
    summary_path = output_dir / "summary.json"

    catalog.to_csv(catalog_path, index=False)
    per_file.to_csv(per_file_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("Wrote:", catalog_path, flush=True)
    print("Wrote:", per_file_path, flush=True)
    print("Wrote:", summary_path, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("Top stations by temperature_obs_count:", flush=True)
    print(catalog.head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
