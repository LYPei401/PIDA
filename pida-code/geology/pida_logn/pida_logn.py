#!/usr/bin/env python3
"""Run the paper's PIDA-Opt-logN receiver binary search.

The input is a predicted leakage interval from PIDA-Pred. The interval is
expanded when needed, source indices are selected to cover it, and the minimum
receiver count meeting an SSIM threshold is found in O(log N) evaluations.
Each evaluation is external because it trains/evaluates a deployment-specific
PIDA-ML model. Supply either a JSON map of already measured SSIM values or an
evaluator command that prints a JSON object containing ``ssim``.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_SOURCE_POSITIONS = (5, 16, 27, 38, 50, 61, 72, 83, 95)


@dataclass(frozen=True)
class MonitoringRange:
    start: int
    end: int
    expanded: bool


def expand_monitoring_range(start: int, end: int, minimum_width: int, domain_end: int) -> MonitoringRange:
    """Expand a narrow inclusive range symmetrically and keep it in-domain."""
    if not 0 <= start <= end <= domain_end:
        raise ValueError(f"invalid interval [{start}, {end}] for domain 0..{domain_end}")
    current_width = end - start + 1
    if current_width >= minimum_width:
        return MonitoringRange(start, end, False)
    missing = minimum_width - current_width
    left = missing // 2
    right = missing - left
    start, end = max(0, start - left), min(domain_end, end + right)
    # If one boundary clipped, use the remaining space at the other boundary.
    if end - start + 1 < minimum_width:
        if start == 0:
            end = min(domain_end, start + minimum_width - 1)
        else:
            start = max(0, end - minimum_width + 1)
    return MonitoringRange(start, end, True)


def select_sources_covering_range(start: int, end: int, source_positions: tuple[int, ...], coordinate_scale: int) -> list[int]:
    """Select physical source channels covering the predicted monitoring range.

    This is the explicit, reusable version of the source rule embedded in the
    original PIDA-one scripts. At least two nearest sources are returned so a
    narrow leakage range still has a usable acquisition.
    """
    scaled_start, scaled_end = start // coordinate_scale, end // coordinate_scale
    inside = [index for index, pos in enumerate(source_positions) if scaled_start <= pos <= scaled_end]
    if len(inside) >= 2:
        outside = [index for index in range(len(source_positions)) if index not in inside]
        nearby = [index for index in outside if min(abs(source_positions[index] - scaled_start), abs(source_positions[index] - scaled_end)) < 6]
        if nearby:
            inside.append(min(nearby, key=lambda index: min(abs(source_positions[index] - scaled_start), abs(source_positions[index] - scaled_end))))
        return sorted(inside)
    midpoint = (scaled_start + scaled_end) / 2
    nearest = sorted(range(len(source_positions)), key=lambda index: abs(source_positions[index] - midpoint))[:2]
    return sorted(nearest)


def binary_search_min_receivers(max_receivers: int, threshold: float, evaluate: Callable[[int], float]) -> tuple[int | None, list[dict]]:
    """Return the smallest count with SSIM >= threshold and its evaluation trace."""
    low, high, best = 1, max_receivers, None
    trace: list[dict] = []
    while low <= high:
        count = (low + high) // 2
        ssim = float(evaluate(count))
        passed = ssim >= threshold
        trace.append({"receiver_count": count, "ssim": ssim, "passed_threshold": passed})
        if passed:
            best = count
            high = count - 1
        else:
            low = count + 1
    return best, trace


def interval_from_json(path: Path) -> tuple[int, int]:
    """Read PIDA-Pred's JSON output, accepting its current interval field."""
    payload = json.loads(path.read_text())
    interval = payload.get("horizontal_interval_columns") or payload.get("interval_columns")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"no predicted interval in {path}")
    return int(interval[0]), int(interval[1])


def command_evaluator(template: str, context: dict[str, int]) -> float:
    """Run a no-shell evaluator command and read its JSON/number SSIM stdout."""
    command = [token.format(**context) for token in shlex.split(template)]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    text = completed.stdout.strip().splitlines()[-1]
    try:
        payload = json.loads(text)
        return float(payload["ssim"] if isinstance(payload, dict) else payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"evaluator must end stdout with JSON {{'ssim': ...}} or a number; got {text!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    range_group = parser.add_mutually_exclusive_group(required=True)
    range_group.add_argument("--predicted-region-json", type=Path)
    range_group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--ssim-threshold", type=float, required=True)
    parser.add_argument("--minimum-range-width", type=int, default=80)
    parser.add_argument("--domain-end", type=int, default=400)
    parser.add_argument("--receiver-spacing", type=int, default=4)
    parser.add_argument("--source-positions", default=",".join(map(str, DEFAULT_SOURCE_POSITIONS)))
    parser.add_argument("--source-coordinate-scale", type=int, default=4)
    evaluator_group = parser.add_mutually_exclusive_group(required=True)
    evaluator_group.add_argument("--candidate-ssim-json", type=Path, help="JSON object mapping receiver count to measured SSIM")
    evaluator_group.add_argument("--evaluator-command", help="command template; supports {receiver_count}, {range_start}, {range_end}, {source_count}, {source_indices}")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    raw_start, raw_end = interval_from_json(args.predicted_region_json) if args.predicted_region_json else args.range
    monitoring = expand_monitoring_range(raw_start, raw_end, args.minimum_range_width, args.domain_end)
    source_positions = tuple(int(value) for value in args.source_positions.split(","))
    sources = select_sources_covering_range(monitoring.start, monitoring.end, source_positions, args.source_coordinate_scale)
    max_receivers = max(1, math.ceil((monitoring.end - monitoring.start + 1) / args.receiver_spacing))

    if args.candidate_ssim_json:
        values = {int(key): float(value) for key, value in json.loads(args.candidate_ssim_json.read_text()).items()}
        def evaluate(count: int) -> float:
            if count not in values:
                raise KeyError(f"candidate-ssim JSON has no result for receiver count {count}")
            return values[count]
    else:
        context = {"range_start": monitoring.start, "range_end": monitoring.end, "source_count": len(sources), "source_indices": ",".join(map(str, sources))}
        def evaluate(count: int) -> float:
            return command_evaluator(args.evaluator_command, {**context, "receiver_count": count})

    best, trace = binary_search_min_receivers(max_receivers, args.ssim_threshold, evaluate)
    result = {
        "algorithm": "PIDA-Opt-logN",
        "input_predicted_range": [raw_start, raw_end],
        "monitoring_range": {"start": monitoring.start, "end": monitoring.end, "expanded": monitoring.expanded},
        "source_indices": sources,
        "candidate_receiver_count": max_receivers,
        "ssim_threshold": args.ssim_threshold,
        "selected_receiver_count": best,
        "search_trace": trace,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
