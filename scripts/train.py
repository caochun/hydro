#!/usr/bin/env python3
"""
UNetSmall 水体语义分割训练脚本（oracle 监督版）。

- 输入: prepare_tiles_oracle.py 生成的 train/val 瓦片
- 验证: val 瓦片集直接算 IoU/F1
- 早停: IoU 连续 patience 次不提升就停
- 输出: best_model.pt
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── 配置 ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    tiles_dir:      Path = Path("/home/chun/hydro_data/tiles_oracle")
    out_dir:        Path = Path("/home/chun/hydro_data/runs")

    in_channels:    int   = 4
    base_channels:  int   = 16
    tile_size:      int   = 256
    image_scale:    float = 10000.0
    image_clip_max: float = 1.0

    epochs:         int   = 60
    batch_size:     int   = 32
    lr:             float = 3e-4
    pos_weight:     float = 40.0
    threshold:      float = 0.5
    patience:       int   = 5      # 早停: IoU 连续 N 次不升就停

    num_workers:    int   = 2
    seed:           int   = 42
# ─────────────────────────────────────────────────────────────────────────────


# ── 模型 ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class UNetSmall(nn.Module):
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        c = base_channels
        self.inc    = ConvBlock(in_channels, c)
        self.d1     = Down(c,      c*2)
        self.d2     = Down(c*2,    c*4)
        self.d3     = Down(c*4,    c*8)
        self.bridge = Down(c*8,    c*16)
        self.u3     = Up(c*16, c*8, c*8)
        self.u2     = Up(c*8,  c*4, c*4)
        self.u1     = Up(c*4,  c*2, c*2)
        self.u0     = Up(c*2,  c,   c)
        self.head   = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.d1(x0); x2 = self.d2(x1); x3 = self.d3(x2)
        xb = self.bridge(x3)
        x  = self.u3(xb, x3); x = self.u2(x, x2); x = self.u1(x, x1); x = self.u0(x, x0)
        return self.head(x)


# ── 数据集 ────────────────────────────────────────────────────────────────────

class TileDataset(Dataset):
    def __init__(self, tiles_dir: Path, augment: bool, clip_max: float):
        self.imgs    = sorted(tiles_dir.glob("img_*.npy"))
        self.augment = augment
        self.clip    = clip_max

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = img_path.parent / img_path.name.replace("img_", "lbl_")
        img = np.load(img_path).astype(np.float32)
        lbl = np.load(lbl_path).astype(np.int64)
        img = np.clip(img, 0.0, self.clip)

        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k).copy()

        return torch.from_numpy(img), torch.from_numpy(lbl)


# ── 验证 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device,
             threshold: float) -> dict:
    model.eval()
    tp = fp = fn = 0
    for imgs, lbls in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        pred = (torch.sigmoid(logits) >= threshold).long()
        valid = (lbls != 255)
        tp += int(((pred == 1) & (lbls == 1) & valid).sum())
        fp += int(((pred == 1) & (lbls == 0) & valid).sum())
        fn += int(((pred == 0) & (lbls == 1) & valid).sum())
    iou = tp / max(tp + fp + fn, 1)
    f1  = 2*tp / max(2*tp + fp + fn, 1)
    return {"iou": iou, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── 训练 ─────────────────────────────────────────────────────────────────────

def train():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  GPUs={torch.cuda.device_count()}")

    train_ds = TileDataset(cfg.tiles_dir / "train", augment=True,  clip_max=cfg.image_clip_max)
    val_ds   = TileDataset(cfg.tiles_dir / "val",   augment=False, clip_max=cfg.image_clip_max)
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = UNetSmall(cfg.in_channels, cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_iou = 0.0
    no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            logits = model(imgs).squeeze(1)
            target = lbls.float()
            ignore = (lbls == 255)

            loss = criterion(logits, target)
            loss[ignore] = 0.0
            n_valid = (~ignore).sum().clamp(min=1)
            loss = loss.sum() / n_valid

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        metrics = validate(model, val_loader, device, cfg.threshold)
        iou, f1 = metrics["iou"], metrics["f1"]
        elapsed = time.time() - t0
        improved = iou > best_iou
        marker = " *" if improved else ""
        print(f"[{epoch:03d}/{cfg.epochs}] loss={avg_loss:.4f}  "
              f"IoU={iou:.4f}  F1={f1:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={elapsed:.0f}s{marker}", flush=True)

        if improved:
            best_iou = iou
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "in_channels":    cfg.in_channels,
                    "base_channels":  cfg.base_channels,
                    "tile_size":      cfg.tile_size,
                    "image_scale":    cfg.image_scale,
                    "image_clip_max": cfg.image_clip_max,
                    "threshold":      cfg.threshold,
                    "band_order":     ["B02", "B03", "B04", "B08"],
                },
                "epoch": epoch,
                "iou":   best_iou,
            }, cfg.out_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[INFO] 早停: IoU 连续 {cfg.patience} 次未提升", flush=True)
                break

    print(f"\n[INFO] 训练完成，最佳 IoU={best_iou:.4f}")
    print(f"[INFO] 模型保存至: {cfg.out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train()
