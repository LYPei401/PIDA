"""Sensor-only sparse-to-dense reconstruction model."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_crop_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    _, _, th, tw = target.shape
    if h == th and w == tw:
        return x
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    return x[:, :, top : top + th, left : left + tw]


class SensorOnlyUNet(nn.Module):
    """U-Net that maps [sparse_values, mask] to a dense weather map."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def padded_shape(height: int, width: int, multiple: int = 8) -> Tuple[int, int]:
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        return height + pad_h, width + pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_h, original_w = x.shape[-2:]
        padded_h, padded_w = self.padded_shape(original_h, original_w)
        if (padded_h, padded_w) != (original_h, original_w):
            x = F.pad(x, (0, padded_w - original_w, 0, padded_h - original_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        e3 = _center_crop_like(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        e2 = _center_crop_like(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        e1 = _center_crop_like(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out(d1)
        return out[:, :, :original_h, :original_w]
