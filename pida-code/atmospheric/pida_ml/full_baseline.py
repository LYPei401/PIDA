#!/usr/bin/env python3
"""Train and evaluate the full-sensor residual PIDA baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pida_ml.models.sensor_only_unet import SensorOnlyUNet


DEFAULT_DATA = os.environ.get("PIDA_ATMOSPHERIC_DATA_DIR", "data/station_case")
DEFAULT_RESULTS = os.environ.get("PIDA_ATMOSPHERIC_RESULTS_DIR", "results/full_residual")


def parse_args(method: str, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--output_dir", default=str(Path(DEFAULT_RESULTS) / method))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--drop_ratios", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--min_keep_ratio", type=float, default=0.6)
    parser.add_argument("--max_keep_ratio", type=float, default=1.0)
    parser.add_argument("--calibration_samples", type=int, default=12)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.set_defaults(method=method)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return raw


def y_path_for(data_dir: Path) -> Path:
    return data_dir / "Y.npy" if (data_dir / "Y.npy").exists() else data_dir / "Y_source.npy"


def load_splits(data_dir: Path, total: int) -> Dict[str, np.ndarray]:
    raw = json.loads((data_dir / "splits.json").read_text())
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}
    splits["all"] = np.arange(total, dtype=np.int64)
    return splits


def load_stats(y_path: Path, train_idx: np.ndarray) -> Dict[str, float]:
    y = np.load(y_path, mmap_mode="r")
    total = 0.0
    total_sq = 0.0
    count = 0
    for idx in train_idx:
        arr = np.asarray(y[int(idx)], dtype=np.float64)
        total += float(arr.sum())
        total_sq += float((arr * arr).sum())
        count += int(arr.size)
    mean = total / count
    std = float(np.sqrt(max(total_sq / count - mean * mean, 0.0)))
    return {"mean": float(mean), "std": std if std > 1e-6 else 1.0}


def norm01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def gradient(field: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(field.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


class PriorSensorPatchDataset(Dataset):
    def __init__(self, data_dir: Path, indices: np.ndarray, args: argparse.Namespace, stats: Dict[str, float]):
        self.y = np.load(y_path_for(data_dir), mmap_mode="r")
        self.sensor_locations = np.load(data_dir / "sensor_locations.npy")
        self.indices = indices[indices > 0].astype(np.int64)
        self.mean = float(stats["mean"])
        self.std = float(stats["std"])
        self.patch = int(args.patch_size)
        self.length = int(args.steps_per_epoch * args.batch_size)
        self.min_keep = float(args.min_keep_ratio)
        self.max_keep = float(args.max_keep_ratio)
        self.method = args.method
        self.rng = np.random.default_rng(args.seed)
        self.h = int(self.y.shape[1])
        self.w = int(self.y.shape[2])
        self.lat = self.sensor_locations[:, 1].astype(np.int64)
        self.lon = self.sensor_locations[:, 2].astype(np.int64)

    def __len__(self) -> int:
        return self.length

    def _patch_with_sensors(self) -> tuple[int, int, np.ndarray]:
        for _ in range(30):
            top = int(self.rng.integers(0, self.h - self.patch + 1))
            left = int(self.rng.integers(0, self.w - self.patch + 1))
            ids = np.flatnonzero(
                (self.lat >= top)
                & (self.lat < top + self.patch)
                & (self.lon >= left)
                & (self.lon < left + self.patch)
            )
            if len(ids):
                return top, left, ids
        return top, left, ids

    def __getitem__(self, _: int):
        t = int(self.rng.choice(self.indices))
        top, left, ids = self._patch_with_sensors()
        y_t = np.asarray(self.y[t, top : top + self.patch, left : left + self.patch], dtype=np.float32)
        prior = np.asarray(self.y[t - 1, top : top + self.patch, left : left + self.patch], dtype=np.float32)
        prior_n = ((prior - self.mean) / self.std).astype(np.float32)
        mask = np.zeros_like(y_t, dtype=np.float32)
        sensor_channel = np.zeros_like(y_t, dtype=np.float32)
        if len(ids):
            keep = max(1, int(round(len(ids) * float(self.rng.uniform(self.min_keep, self.max_keep)))))
            chosen = self.rng.choice(ids, size=keep, replace=False)
            rr = self.lat[chosen] - top
            cc = self.lon[chosen] - left
            mask[rr, cc] = 1.0
            if self.method == "A":
                sensor_channel[rr, cc] = ((y_t[rr, cc] - self.mean) / self.std).astype(np.float32)
            else:
                sensor_channel[rr, cc] = ((y_t[rr, cc] - prior[rr, cc]) / self.std).astype(np.float32)
        inputs = np.stack([prior_n, sensor_channel, mask], axis=0)
        if self.method == "A":
            target = ((y_t - self.mean) / self.std).astype(np.float32)
        else:
            target = ((y_t - prior) / self.std).astype(np.float32)
        return torch.from_numpy(inputs), torch.from_numpy(target[None])


def train_model(args: argparse.Namespace, out: Path, device: str) -> dict:
    data_dir = Path(args.data_dir)
    y = np.load(y_path_for(data_dir), mmap_mode="r")
    splits = load_splits(data_dir, int(y.shape[0]))
    stats = load_stats(y_path_for(data_dir), splits["train"])
    dataset = PriorSensorPatchDataset(data_dir, splits["train"], args, stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = SensorOnlyUNet(in_channels=3, base_channels=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = float("inf")
    history = []
    ckpt = out / "model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            count += int(x.shape[0])
        loss_value = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss": loss_value})
        if loss_value < best:
            best = loss_value
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": args.base_channels,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "method": args.method,
                    "data_dir": str(data_dir),
                },
                ckpt,
            )
        print(f"epoch={epoch} train_loss={loss_value:.6f} best={best:.6f}", flush=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return {"checkpoint": str(ckpt), "stats": stats, "best_loss": best}


def metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    rel = rmse / (float(np.sqrt(np.mean(target * target))) + 1e-12)
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    data_range = max(float(y.max() - y.min()), 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(((x - mx) ** 2).mean()), float(((y - my) ** 2).mean())
    cov = float(((x - mx) * (y - my)).mean())
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12)
    return {"rmse": rmse, "mae": mae, "relative_rmse": rel, "ssim": float(ssim)}


def make_groups(lat: np.ndarray, lon: np.ndarray, h: int, w: int, block_h: int = 90, block_w: int = 180) -> np.ndarray:
    return (lat // block_h) * ((w + block_w - 1) // block_w) + (lon // block_w)


def base_score(y: np.ndarray, t: int, lat: np.ndarray, lon: np.ndarray, mode: str) -> np.ndarray:
    prior_i = max(t - 1, 0)
    prior2_i = max(t - 2, 0)
    temporal = np.abs(y[prior_i, lat, lon] - y[prior2_i, lat, lon]).astype(np.float64)
    grad = gradient(np.asarray(y[prior_i], dtype=np.float32))[lat, lon].astype(np.float64)
    if mode == "temporal":
        return temporal
    if mode == "gradient":
        return grad
    return 0.5 * norm01(temporal) + 0.5 * norm01(grad)


def build_input(field: np.ndarray, prior: np.ndarray, values: np.ndarray, mask: np.ndarray, lat: np.ndarray, lon: np.ndarray, drop: np.ndarray, stats: Dict[str, float], method: str) -> np.ndarray:
    mean, std = float(stats["mean"]), float(stats["std"])
    keep = mask.astype(bool).copy()
    if len(drop):
        keep[drop] = False
    prior_n = ((prior - mean) / std).astype(np.float32)
    sensor_ch = np.zeros_like(field, dtype=np.float32)
    mask_ch = np.zeros_like(field, dtype=np.float32)
    if method == "A":
        sensor_ch[lat[keep], lon[keep]] = ((values[keep] - mean) / std).astype(np.float32)
    else:
        sensor_ch[lat[keep], lon[keep]] = ((values[keep] - prior[lat[keep], lon[keep]]) / std).astype(np.float32)
    mask_ch[lat[keep], lon[keep]] = 1.0
    return np.stack([prior_n, sensor_ch, mask_ch], axis=0)


@torch.no_grad()
def predict(model, inp: np.ndarray, prior: np.ndarray, stats: Dict[str, float], method: str, device: str) -> np.ndarray:
    out = model(torch.from_numpy(inp[None]).to(device)).detach().cpu().numpy()[0, 0]
    if method == "A":
        return out * stats["std"] + stats["mean"]
    return prior + out * stats["std"]


def calibrate_group_sensitivity(model, y, x, xmask, indices, lat, lon, groups, stats, method, device, n: int) -> np.ndarray:
    unique = np.unique(groups)
    score = np.zeros(int(unique.max()) + 1, dtype=np.float64)
    count = np.zeros_like(score)
    for t in indices[indices > 0][:n]:
        t = int(t)
        field = np.asarray(y[t], dtype=np.float32)
        prior = np.asarray(y[t - 1], dtype=np.float32)
        vals = np.asarray(x[t], dtype=np.float32)
        mask = np.asarray(xmask[t], dtype=np.uint8)
        full = predict(model, build_input(field, prior, vals, mask, lat, lon, np.array([], dtype=np.int64), stats, method), prior, stats, method, device)
        for g in unique:
            drop = np.flatnonzero(groups == g)
            pred = predict(model, build_input(field, prior, vals, mask, lat, lon, drop, stats, method), prior, stats, method, device)
            score[int(g)] += float(np.mean((pred - full) ** 2))
            count[int(g)] += 1
    return score / np.maximum(count, 1)


def choose_drop(args, y, t, lat, lon, ratio, rng, groups, group_sens, learned_w=None) -> np.ndarray:
    n = len(lat)
    k = int(round(n * ratio))
    if k <= 0:
        return np.array([], dtype=np.int64)
    if args.method in ("A", "B"):
        scores = base_score(y, t, lat, lon, "hybrid")
    elif args.method == "C":
        scores = group_sens[groups]
    elif args.method == "D":
        scores = base_score(y, t, lat, lon, "hybrid")
        selected = []
        for g in np.unique(groups):
            ids = np.flatnonzero(groups == g)
            local_k = int(round(len(ids) * ratio))
            if local_k:
                selected.extend(ids[np.argpartition(scores[ids], min(local_k, len(ids) - 1))[:local_k]].tolist())
        selected = np.asarray(selected, dtype=np.int64)
        if len(selected) > k:
            selected = selected[np.argpartition(scores[selected], k)[:k]]
        return np.sort(selected)
    elif args.method == "E":
        feature = np.stack(
            [
                norm01(base_score(y, t, lat, lon, "temporal")),
                norm01(base_score(y, t, lat, lon, "gradient")),
                lat / max(float(np.max(lat)), 1.0),
                lon / max(float(np.max(lon)), 1.0),
                np.ones(n),
            ],
            axis=1,
        )
        scores = feature @ learned_w
    else:
        scores = base_score(y, t, lat, lon, "hybrid")
    return np.sort(np.argpartition(scores, min(k, n - 1))[:k].astype(np.int64))


def evaluate(args: argparse.Namespace, out: Path, train_info: dict, device: str) -> None:
    data = Path(args.data_dir)
    ckpt = torch.load(train_info["checkpoint"], map_location="cpu", weights_only=False)
    model = SensorOnlyUNet(in_channels=3, base_channels=int(ckpt["base_channels"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    stats = train_info["stats"]
    y = np.load(y_path_for(data), mmap_mode="r")
    x = np.load(data / "X.npy", mmap_mode="r")
    xmask = np.load(data / "X_mask.npy", mmap_mode="r")
    sensors = np.load(data / "sensor_locations.npy")
    lat, lon = sensors[:, 1].astype(np.int64), sensors[:, 2].astype(np.int64)
    splits = load_splits(data, int(y.shape[0]))
    test = splits["test"]
    if args.max_eval_samples > 0:
        test = test[: args.max_eval_samples]
    ratios = [float(v) for v in args.drop_ratios.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)
    groups = make_groups(lat, lon, int(y.shape[1]), int(y.shape[2]))
    group_sens = np.ones(int(groups.max()) + 1, dtype=np.float64)
    learned_w = None
    if args.method in ("C", "E"):
        group_sens = calibrate_group_sensitivity(model, y, x, xmask, splits["train"], lat, lon, groups, stats, args.method, device, args.calibration_samples)
    if args.method == "E":
        feat = []
        labels = []
        calib_t = int(splits["train"][max(1, min(len(splits["train"]) - 1, args.calibration_samples))])
        for g in np.unique(groups):
            ids = np.flatnonzero(groups == g)
            f = np.stack(
                [
                    norm01(base_score(y, calib_t, lat[ids], lon[ids], "temporal")),
                    norm01(base_score(y, calib_t, lat[ids], lon[ids], "gradient")),
                    lat[ids] / max(float(np.max(lat)), 1.0),
                    lon[ids] / max(float(np.max(lon)), 1.0),
                    np.ones(len(ids)),
                ],
                axis=1,
            ).mean(axis=0)
            feat.append(f)
            labels.append(group_sens[int(g)])
        feat = np.asarray(feat, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        learned_w = np.linalg.solve(feat.T @ feat + 1e-3 * np.eye(feat.shape[1]), feat.T @ labels)
    rows = []
    for order, t_raw in enumerate(test):
        t = int(t_raw)
        field = np.asarray(y[t], dtype=np.float32)
        prior = np.asarray(y[t - 1], dtype=np.float32) if t > 0 else field
        vals = np.asarray(x[t], dtype=np.float32)
        mask = np.asarray(xmask[t], dtype=np.uint8)
        prior_only = metrics(prior, field)
        rows.append({"sample_index": t, "method": "prior_only", "drop_ratio": 1.0, "sensors_kept": 0, **prior_only})
        full = predict(model, build_input(field, prior, vals, mask, lat, lon, np.array([], dtype=np.int64), stats, args.method), prior, stats, args.method, device)
        rows.append({"sample_index": t, "method": "full_mask", "drop_ratio": 0.0, "sensors_kept": int(mask.sum()), **metrics(full, field)})
        observed = np.flatnonzero(mask.astype(bool))
        for ratio in ratios:
            k = int(round(len(observed) * ratio))
            random_drop = np.sort(rng.choice(observed, size=k, replace=False).astype(np.int64))
            random_pred = predict(model, build_input(field, prior, vals, mask, lat, lon, random_drop, stats, args.method), prior, stats, args.method, device)
            rows.append({"sample_index": t, "method": "random_drop", "drop_ratio": ratio, "sensors_kept": int(mask.sum() - len(random_drop)), **metrics(random_pred, field)})
            pida_drop = choose_drop(args, y, t, lat, lon, ratio, rng, groups, group_sens, learned_w)
            pida_pred = predict(model, build_input(field, prior, vals, mask, lat, lon, pida_drop, stats, args.method), prior, stats, args.method, device)
            rows.append({"sample_index": t, "method": f"pida_{args.method}", "drop_ratio": ratio, "sensors_kept": int(mask.sum() - len(pida_drop)), **metrics(pida_pred, field)})
        if (order + 1) % 25 == 0 or order + 1 == len(test):
            print(f"evaluated {order + 1}/{len(test)}", flush=True)
    fieldnames = ["sample_index", "method", "drop_ratio", "sensors_kept", "rmse", "mae", "relative_rmse", "ssim"]
    with (out / "per_time_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for key in sorted({(r["method"], r["drop_ratio"]) for r in rows}, key=lambda v: (v[1], v[0])):
        group = [r for r in rows if (r["method"], r["drop_ratio"]) == key]
        item = {"method": key[0], "drop_ratio": key[1], "num_samples": len(group)}
        for m in ["rmse", "mae", "relative_rmse", "ssim", "sensors_kept"]:
            vals = np.asarray([float(r[m]) for r in group])
            item[f"{m}_mean"] = float(vals.mean())
            item[f"{m}_std"] = float(vals.std())
        summary.append(item)
    with (out / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def run(method: str, description: str) -> None:
    args = parse_args(method, description)
    set_seed(args.seed)
    device = choose_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"method={method} device={device} output={out}", flush=True)
    train_info = train_model(args, out, device)
    config = vars(args)
    config.update({"train_info": train_info})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    evaluate(args, out, train_info, device)
