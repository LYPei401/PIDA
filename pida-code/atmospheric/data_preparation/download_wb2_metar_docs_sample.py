#!/usr/bin/env python3
"""Download WeatherBench2 documentation and small METAR station samples.

This script intentionally does not parse Parquet files. It only mirrors the
official docs and a few raw METAR partitions so later analysis can run locally.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DOC_URLS = {
    "weatherbench2_data_guide.html": "https://weatherbench2.readthedocs.io/en/latest/data-guide.html",
    "weatherbench2_index.html": "https://weatherbench2.readthedocs.io/en/latest/",
}


DEFAULT_METAR_FILES = [
    "metar-timeNominal-by-month/year=2020/month=1/2020-01.parquet",
    "metar-timeNominal-by-month/year=2020/month=7/2020-07.parquet",
    "metar-timeNominal-by-month/year=2010/month=1/2010-01.parquet",
    "metar-timeNominal-by-day/year=2020/month=1/day=1/2020-01-01.parquet",
    "metar-timeObs-by-hour_2001-07-01T00_2024-01-01T00/year=2020/month=1/day=1/hour=0/2020-01-01T00.parquet",
]


def download_url(url: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "openFWI-weatherbench2-metar-sampler"})
    started = time.time()
    with urlopen(request, timeout=120) as response:
        with output_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    return {
        "url": url,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "seconds": round(time.time() - started, 3),
    }


def download_gcs_file(fs, gcs_path: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    info = fs.info(gcs_path)
    fs.get(gcs_path, str(output_path))
    return {
        "gcs_path": "gs://" + gcs_path,
        "path": str(output_path),
        "gcs_size": info.get("size"),
        "bytes": output_path.stat().st_size,
        "seconds": round(time.time() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="data/metar_sample",
        help="Local output directory for downloaded documentation and METAR samples.",
    )
    parser.add_argument(
        "--metar_file",
        action="append",
        default=[],
        help=(
            "METAR path relative to gs://weatherbench2/datasets/metar/. "
            "Can be repeated. Defaults to a small fixed sample."
        ),
    )
    parser.add_argument(
        "--skip_metar",
        action="store_true",
        help="Only download documentation, not GCS METAR Parquet files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    docs_dir = output_dir / "docs"
    metar_dir = output_dir / "metar"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_dir": str(output_dir),
        "official_sources": {
            "docs": DOC_URLS,
            "metar_bucket": "gs://weatherbench2/datasets/metar/",
        },
        "metar_fields_from_official_docs": [
            "stationName",
            "locationName",
            "latitude",
            "longitude",
            "elevation",
            "timeObs",
            "timeNominal",
            "timeReceived",
            "reportType",
            "autoStationType",
            "temperature",
        ],
        "docs": [],
        "metar_files": [],
        "errors": [],
    }

    for filename, url in DOC_URLS.items():
        try:
            result = download_url(url, docs_dir / filename)
            manifest["docs"].append(result)
            print("Downloaded doc:", result, flush=True)
        except (OSError, URLError) as exc:
            error = {"url": url, "error": repr(exc)}
            manifest["errors"].append(error)
            print("Failed doc:", error, flush=True)

    if not args.skip_metar:
        try:
            import gcsfs
        except ImportError as exc:
            error = {"step": "import_gcsfs", "error": repr(exc)}
            manifest["errors"].append(error)
            print("Failed:", error, flush=True)
        else:
            fs = gcsfs.GCSFileSystem(token="anon")
            metar_files = args.metar_file or DEFAULT_METAR_FILES
            for rel_path in metar_files:
                gcs_path = "weatherbench2/datasets/metar/" + rel_path
                local_path = metar_dir / rel_path
                try:
                    result = download_gcs_file(fs, gcs_path, local_path)
                    manifest["metar_files"].append(result)
                    print("Downloaded METAR:", result, flush=True)
                except Exception as exc:  # noqa: BLE001 - keep batch download going.
                    error = {"gcs_path": "gs://" + gcs_path, "error": repr(exc)}
                    manifest["errors"].append(error)
                    print("Failed METAR:", error, flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("Wrote manifest:", manifest_path, flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 1 if manifest["errors"] and not manifest["docs"] else 0


if __name__ == "__main__":
    sys.exit(main())
