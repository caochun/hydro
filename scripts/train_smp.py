#!/usr/bin/env python3
"""
UNet + ResNet34 (ImageNet 预训练) 训练脚本。

用 segmentation_models_pytorch 替换自写 UNetSmall：
- 编码器: ResNet34，加载 ImageNet 预训练权重
- 解码器: UNet 标准对称结构
- 输入 4 通道（smp 自动处理 RGB 权重扩展）

与 train.py 共享同一份 oracle 切片数据。
输出 best_model_smp.pt，与 UNetSmall 基线对比。
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# ── 配置 ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    tiles_dir:      Path = Path("/home/chun/hydro_data/tiles_oracle")
    out_dir:        Path = Path("/home/chun/hydro_data/runs")
    model_name:     str  = "best_model_smp.pt"

    encoder:        str  = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels:    int  = 4

    image_clip_max: float = 1.0

    epochs:         int   = 60
    batch_size:     int   = 16      # ResNet34 比小 UNet 大，降 batch
    lr:             float = 1e-4    # 预训练模型用更小 LR
    pos_weight:     float = 40.0
    threshold:      float = 0.5
    patience:       int   = 5

    num_workers:    int   = 2
    seed:           int   = 42
# ─────────────────────────────────────────────────────────────────────────────


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


def train():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  GPUs={torch.cuda.device_count()}")

    train_ds = TileDataset(cfg.tiles_dir / "train", augment=True,  clip_max=cfg.image_clip_max)
    val_ds   = TileDataset(cfg.tiles_dir / "val",   augment=False, clip_max=cfg.image_clip_max)
    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = smp.Unet(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] model=smp.Unet({cfg.encoder}, imagenet)  params={n_params:.1f}M")

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
                    "model":           "smp.Unet",
                    "encoder":         cfg.encoder,
                    "encoder_weights": cfg.encoder_weights,
                    "in_channels":     cfg.in_channels,
                    "image_clip_max":  cfg.image_clip_max,
                    "threshold":       cfg.threshold,
                    "band_order":      ["B02", "B03", "B04", "B08"],
                },
                "epoch": epoch,
                "iou":   best_iou,
            }, cfg.out_dir / cfg.model_name)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[INFO] 早停: IoU 连续 {cfg.patience} 次未提升", flush=True)
                break

    print(f"\n[INFO] 训练完成，最佳 IoU={best_iou:.4f}")
    print(f"[INFO] 模型保存至: {cfg.out_dir / cfg.model_name}")


if __name__ == "__main__":
    train()
