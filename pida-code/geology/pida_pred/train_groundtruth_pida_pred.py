#!/usr/bin/env python3
"""Train PIDA-Pred from ground-truth velocity-map trajectories.

Inputs are two consecutive velocity maps, V(t-2) and V(t-1), stacked as two
channels.  The supervised target is V(t).  All three maps come from the same
geological case; only strictly consecutive 10-year steps are accepted.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

LABEL_MIN, LABEL_MAX = 951.0, 2546.0
NAME = re.compile(r"label_(sim\d+)_t(\d+)\.npz")


@dataclass(frozen=True)
class Triplet:
    case: str
    previous2: Path
    previous1: Path
    target: Path


def build_triplets(root: Path) -> list[Triplet]:
    """Build only (t-20, t-10, t) triplets belonging to one case."""
    by_case: dict[str, dict[int, Path]] = {}
    for path in root.glob("label_sim*_t*.npz"):
        match = NAME.fullmatch(path.name)
        if match:
            by_case.setdefault(match.group(1), {})[int(match.group(2))] = path
    result = []
    for case, sequence in sorted(by_case.items()):
        for time, target in sorted(sequence.items()):
            if time - 10 in sequence and time - 20 in sequence:
                result.append(Triplet(case, sequence[time - 20], sequence[time - 10], target))
    if not result:
        raise ValueError(f"no consecutive label triplets found under {root}")
    return result


def load_label(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        value = archive[archive.files[0]].astype(np.float32)
    return (2.0 * (value - LABEL_MIN) / (LABEL_MAX - LABEL_MIN) - 1.0).astype(np.float32)


class GroundTruthTrajectoryDataset(Dataset):
    """Lazy NPZ reader: avoids loading all velocity maps into host memory."""
    def __init__(self, triplets: list[Triplet]): self.triplets = triplets
    def __len__(self): return len(self.triplets)
    def __getitem__(self, index):
        item = self.triplets[index]
        inputs = np.stack([load_label(item.previous2), load_label(item.previous1)], axis=0)
        target = load_label(item.target)[None]
        return torch.from_numpy(inputs), torch.from_numpy(target)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.layers(x)


class PidaPredVelocityUNet(nn.Module):
    """Two-channel encoder--decoder predicting one same-size velocity map."""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.enc1, self.enc2 = ConvBlock(2, base_channels), ConvBlock(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2), ConvBlock(base_channels * 4, base_channels * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2), ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); mid = self.mid(self.pool(e2))
        d2 = F.interpolate(self.up2(mid), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        # Predict a residual over the latest observed digital state.
        return x[:, 1:2] + self.output(self.dec1(torch.cat([d1, e1], dim=1)))


def split_by_case(triplets: list[Triplet], validation_fraction: float, seed: int):
    cases = sorted({row.case for row in triplets}); rng = random.Random(seed); rng.shuffle(cases)
    count = max(1, round(len(cases) * validation_fraction)); val_cases = set(cases[:count])
    return [x for x in triplets if x.case not in val_cases], [x for x in triplets if x.case in val_cases], sorted(val_cases)


def evaluate(model, loader, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, target in loader:
            pred = model(inputs.to(device)); total += F.l1_loss(pred, target.to(device), reduction="sum").item()
    return total / (len(loader.dataset) * 401 * 141)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-label-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-case", action="append", default=[], help="case ID such as sim0787 to hold out entirely")
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    triplets = [row for row in build_triplets(args.train_label_root) if row.case not in set(args.exclude_case)]
    if not triplets:
        parser.error("no PIDA-Pred triplets remain after --exclude-case filtering")
    train_rows, val_rows, val_cases = split_by_case(triplets, args.validation_fraction, args.seed)
    train_loader = DataLoader(GroundTruthTrajectoryDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(GroundTruthTrajectoryDataset(val_rows), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    model = PidaPredVelocityUNet(args.base_channels).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for inputs, target in train_loader:
            optimizer.zero_grad(); prediction = model(inputs.to(device)); target = target.to(device)
            loss = F.l1_loss(prediction, target) + 0.1 * F.mse_loss(prediction, target); loss.backward(); optimizer.step(); loss_sum += loss.item()
        val_l1 = evaluate(model, val_loader, device); row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader), "val_l1": val_l1}; history.append(row); print(row, flush=True)
        if val_l1 < best:
            best = val_l1
            torch.save({"model_state": model.state_dict(), "base_channels": args.base_channels, "label_min": LABEL_MIN, "label_max": LABEL_MAX, "validation_cases": val_cases, "epoch": epoch, "val_l1": val_l1}, args.output_dir / "pida_pred_groundtruth.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps({**vars(args), "train_triplets": len(train_rows), "validation_triplets": len(val_rows), "validation_cases": val_cases}, default=str, indent=2) + "\n")
    return 0

if __name__ == "__main__": raise SystemExit(main())
