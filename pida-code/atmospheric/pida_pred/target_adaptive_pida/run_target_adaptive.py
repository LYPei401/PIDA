#!/usr/bin/env python3
"""Train one target-adaptive fixed-mask PIDA residual U-Net."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pida_ml.models.sensor_only_unet import SensorOnlyUNet


DEFAULT_DATA = os.environ.get("PIDA_ATMOSPHERIC_DATA_DIR", "data/station_case")
DEFAULT_FULL_CKPT = os.environ.get("PIDA_ATMOSPHERIC_FULL_CHECKPOINT", "results/full_residual/model.pt")
DEFAULT_OUTPUT = os.environ.get("PIDA_ATMOSPHERIC_OUTPUT_DIR", "results/target_adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--full_checkpoint", default=DEFAULT_FULL_CKPT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--target_rank", type=int, required=True, help="Rank inside test split.")
    parser.add_argument("--drop_ratio", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=256)
    parser.add_argument("--val_steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--no_warm_start",
        action="store_true",
        help="Disable initialization from the residual full-baseline checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
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


def load_splits(data_dir: Path) -> Dict[str, np.ndarray]:
    raw = json.loads((data_dir / "splits.json").read_text())
    return {k: np.asarray(v, dtype=np.int64) for k, v in raw.items()}


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


def select_pida_sensors(y: np.ndarray, target_idx: int, sensors: np.ndarray, drop_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    lat = sensors[:, 1].astype(np.int64)
    lon = sensors[:, 2].astype(np.int64)
    prior_i = max(target_idx - 1, 0)
    prior2_i = max(target_idx - 2, 0)
    temporal = np.abs(y[prior_i, lat, lon] - y[prior2_i, lat, lon]).astype(np.float64)
    grad = gradient(np.asarray(y[prior_i], dtype=np.float32))[lat, lon].astype(np.float64)
    score = 0.5 * norm01(temporal) + 0.5 * norm01(grad)
    drop_count = int(round(len(score) * drop_ratio))
    if drop_count <= 0:
        drop = np.array([], dtype=np.int64)
    else:
        drop = np.sort(np.argpartition(score, min(drop_count, len(score) - 1))[:drop_count].astype(np.int64))
    keep_mask = np.ones(len(score), dtype=bool)
    keep_mask[drop] = False
    keep = np.flatnonzero(keep_mask).astype(np.int64)
    return keep, drop


class FixedMaskPatchDataset(Dataset):
    def __init__(self, data_dir: Path, indices: np.ndarray, keep_ids: np.ndarray, stats: Dict[str, float], args: argparse.Namespace):
        self.y = np.load(y_path_for(data_dir), mmap_mode="r")
        self.sensors = np.load(data_dir / "sensor_locations.npy")
        self.indices = indices[indices > 0].astype(np.int64)
        self.keep_ids = keep_ids.astype(np.int64)
        self.mean = float(stats["mean"])
        self.std = float(stats["std"])
        self.patch = int(args.patch_size)
        self.length = int(args.steps_per_epoch * args.batch_size)
        self.rng = np.random.default_rng(args.seed)
        self.h = int(self.y.shape[1])
        self.w = int(self.y.shape[2])
        self.lat = self.sensors[:, 1].astype(np.int64)
        self.lon = self.sensors[:, 2].astype(np.int64)
        self.keep_lat = self.lat[self.keep_ids]
        self.keep_lon = self.lon[self.keep_ids]

    def __len__(self) -> int:
        return self.length

    def _patch_with_sensors(self) -> tuple[int, int, np.ndarray]:
        for _ in range(30):
            top = int(self.rng.integers(0, self.h - self.patch + 1))
            left = int(self.rng.integers(0, self.w - self.patch + 1))
            ids = np.flatnonzero(
                (self.keep_lat >= top)
                & (self.keep_lat < top + self.patch)
                & (self.keep_lon >= left)
                & (self.keep_lon < left + self.patch)
            )
            if len(ids):
                return top, left, ids
        return top, left, ids

    def __getitem__(self, _: int):
        t = int(self.rng.choice(self.indices))
        top, left, local_ids = self._patch_with_sensors()
        y_t = np.asarray(self.y[t, top : top + self.patch, left : left + self.patch], dtype=np.float32)
        prior = np.asarray(self.y[t - 1, top : top + self.patch, left : left + self.patch], dtype=np.float32)
        prior_n = ((prior - self.mean) / self.std).astype(np.float32)
        innovation = np.zeros_like(y_t, dtype=np.float32)
        mask = np.zeros_like(y_t, dtype=np.float32)
        if len(local_ids):
            rows = self.keep_lat[local_ids] - top
            cols = self.keep_lon[local_ids] - left
            innovation[rows, cols] = ((y_t[rows, cols] - prior[rows, cols]) / self.std).astype(np.float32)
            mask[rows, cols] = 1.0
        target = ((y_t - prior) / self.std).astype(np.float32)
        return torch.from_numpy(np.stack([prior_n, innovation, mask], axis=0)), torch.from_numpy(target[None])


class FixedMaskValPatchDataset(FixedMaskPatchDataset):
    def __init__(self, data_dir: Path, indices: np.ndarray, keep_ids: np.ndarray, stats: Dict[str, float], args: argparse.Namespace):
        super().__init__(data_dir, indices, keep_ids, stats, args)
        self.length = int(args.val_steps * args.batch_size)
        self.rng = np.random.default_rng(args.seed + 99991)


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


def build_input(field: np.ndarray, prior: np.ndarray, values: np.ndarray, keep_ids: np.ndarray, sensors: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    mean, std = float(stats["mean"]), float(stats["std"])
    lat = sensors[:, 1].astype(np.int64)
    lon = sensors[:, 2].astype(np.int64)
    prior_n = ((prior - mean) / std).astype(np.float32)
    innovation = np.zeros_like(field, dtype=np.float32)
    mask = np.zeros_like(field, dtype=np.float32)
    k_lat = lat[keep_ids]
    k_lon = lon[keep_ids]
    innovation[k_lat, k_lon] = ((values[keep_ids] - prior[k_lat, k_lon]) / std).astype(np.float32)
    mask[k_lat, k_lon] = 1.0
    return np.stack([prior_n, innovation, mask], axis=0)


@torch.no_grad()
def predict(model: torch.nn.Module, inputs: np.ndarray, prior: np.ndarray, stats: Dict[str, float], device: str) -> np.ndarray:
    out = model(torch.from_numpy(inputs[None]).to(device)).detach().cpu().numpy()[0, 0]
    return prior + out * float(stats["std"])


def run_validation(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = torch.nn.functional.mse_loss(model(inputs), target)
            total += float(loss.item()) * inputs.shape[0]
            count += int(inputs.shape[0])
    return total / max(count, 1)


def split_train_val(train_idx: np.ndarray, val_count: int = 104) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) <= val_count + 10:
        split = max(1, int(len(train_idx) * 0.9))
        return train_idx[:split], train_idx[split:]
    return train_idx[:-val_count], train_idx[-val_count:]


def train(args: argparse.Namespace, out: Path, keep_ids: np.ndarray, stats: Dict[str, float], train_idx: np.ndarray, device: str) -> Path:
    train_core, val_idx = split_train_val(train_idx)
    dataset = FixedMaskPatchDataset(Path(args.data_dir), train_core, keep_ids, stats, args)
    val_dataset = FixedMaskValPatchDataset(Path(args.data_dir), val_idx, keep_ids, stats, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = SensorOnlyUNet(in_channels=3, base_channels=args.base_channels).to(device)
    lr = float(args.lr)
    warm_start_used = False
    if not args.no_warm_start and Path(args.full_checkpoint).exists():
        full_ckpt = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
        if int(full_ckpt.get("base_channels", args.base_channels)) == int(args.base_channels):
            model.load_state_dict(full_ckpt["model_state"])
            lr = float(args.finetune_lr)
            warm_start_used = True
            print(f"warm-started from {args.full_checkpoint}; finetune_lr={lr}", flush=True)
        else:
            print("full checkpoint base_channels mismatch; training from scratch", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    history = []
    ckpt = out / "target_fixed_mask_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs), target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * inputs.shape[0]
            count += int(inputs.shape[0])
        loss_value = total / max(count, 1)
        val_loss = run_validation(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_value, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": args.base_channels,
                    "stats": stats,
                    "keep_ids": keep_ids,
                    "drop_ratio": args.drop_ratio,
                    "target_rank": args.target_rank,
                    "warm_start_used": warm_start_used,
                    "lr": lr,
                    "best_val_loss": best_val,
                },
                ckpt,
            )
        print(f"epoch={epoch} train_loss={loss_value:.6f} val_loss={val_loss:.6f} best_val={best_val:.6f}", flush=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "warm_start_used": warm_start_used,
                "lr": lr,
                "train_samples": int(len(train_core)),
                "val_samples": int(len(val_idx)),
            },
            indent=2,
        )
    )
    return ckpt


def evaluate(args: argparse.Namespace, out: Path, ckpt: Path, keep_ids: np.ndarray, drop_ids: np.ndarray, stats: Dict[str, float], target_idx: int, device: str) -> None:
    data_dir = Path(args.data_dir)
    y = np.load(y_path_for(data_dir), mmap_mode="r")
    x = np.load(data_dir / "X.npy", mmap_mode="r")
    sensors = np.load(data_dir / "sensor_locations.npy")
    field = np.asarray(y[target_idx], dtype=np.float32)
    prior = np.asarray(y[target_idx - 1], dtype=np.float32)
    values = np.asarray(x[target_idx], dtype=np.float32)

    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = SensorOnlyUNet(in_channels=3, base_channels=int(ckpt_obj["base_channels"]))
    model.load_state_dict(ckpt_obj["model_state"])
    model.to(device).eval()
    pred = predict(model, build_input(field, prior, values, keep_ids, sensors, stats), prior, stats, device)

    full_metrics = None
    no_finetune_metrics = None
    if Path(args.full_checkpoint).exists():
        full_ckpt = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
        full_model = SensorOnlyUNet(in_channels=3, base_channels=int(full_ckpt["base_channels"]))
        full_model.load_state_dict(full_ckpt["model_state"])
        full_model.to(device).eval()
        full_stats = {"mean": float(full_ckpt["mean"]), "std": float(full_ckpt["std"])}
        all_ids = np.arange(len(sensors), dtype=np.int64)
        full_pred = predict(full_model, build_input(field, prior, values, all_ids, sensors, full_stats), prior, full_stats, device)
        full_metrics = metrics(full_pred, field)
        no_finetune_pred = predict(full_model, build_input(field, prior, values, keep_ids, sensors, full_stats), prior, full_stats, device)
        no_finetune_metrics = metrics(no_finetune_pred, field)

    result = {
        "target_rank": int(args.target_rank),
        "target_index": int(target_idx),
        "drop_ratio": float(args.drop_ratio),
        "sensors_total": int(len(sensors)),
        "sensors_kept": int(len(keep_ids)),
        "sensors_dropped": int(len(drop_ids)),
        "prior_only": metrics(prior, field),
        "full_baseline_B": full_metrics,
        "no_finetune_selected_mask_B": no_finetune_metrics,
        "target_adaptive_pida": metrics(pred, field),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    np.save(out / "selected_sensor_indices.npy", keep_ids)
    np.save(out / "dropped_sensor_indices.npy", drop_ids)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed + args.target_rank * 1000 + int(round(args.drop_ratio * 1000)))
    device = choose_device(args.device)
    data_dir = Path(args.data_dir)
    y = np.load(y_path_for(data_dir), mmap_mode="r")
    splits = load_splits(data_dir)
    if args.target_rank < 0 or args.target_rank >= len(splits["test"]):
        raise ValueError(f"target_rank {args.target_rank} outside test split size {len(splits['test'])}")
    target_idx = int(splits["test"][args.target_rank])
    sensors = np.load(data_dir / "sensor_locations.npy")
    keep_ids, drop_ids = select_pida_sensors(y, target_idx, sensors, args.drop_ratio)
    stats = load_stats(y_path_for(data_dir), splits["train"])

    out = Path(args.output_dir) / f"target_{args.target_rank:03d}_drop_{int(round(args.drop_ratio * 100)):02d}"
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    config.update({"target_index": target_idx, "sensors_kept": int(len(keep_ids)), "sensors_dropped": int(len(drop_ids))})
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    ckpt = train(args, out, keep_ids, stats, splits["train"], device)
    evaluate(args, out, ckpt, keep_ids, drop_ids, stats, target_idx, device)


if __name__ == "__main__":
    main()
