#!/usr/bin/env python3
"""
对比 val 瓦片在有/无 TTA 下的 IoU / F1。

TTA 方式：4 向翻转组合（原图 / hflip / vflip / hvflip），sigmoid 后平均。
可选 8 向（再加旋转），USE_ROTATIONS=True 开启。
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── 配置 ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    tiles_dir:  Path = Path("/home/chun/hydro_data/tiles_oracle")
    model_path: Path = Path("/home/chun/hydro_data/runs/best_model_smp.pt")
    model_kind: str  = "smp"     # "smp" 或 "unet_small"
    batch_size: int  = 16
    threshold:  float = 0.5
    num_workers: int = 2
    use_rotations: bool = True   # True=8 向, False=4 向
# ─────────────────────────────────────────────────────────────────────────────


class TileDataset(Dataset):
    def __init__(self, tiles_dir: Path):
        self.imgs = sorted(tiles_dir.glob("img_*.npy"))
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = img_path.parent / img_path.name.replace("img_", "lbl_")
        img = np.clip(np.load(img_path).astype(np.float32), 0.0, 1.0)
        lbl = np.load(lbl_path).astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(lbl)


def build_model(cfg: Config, device):
    ckpt = torch.load(cfg.model_path, map_location=device, weights_only=False)
    mc = ckpt["config"]
    if cfg.model_kind == "smp":
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=mc["encoder"],
            encoder_weights=None,          # 不再下载，只用 state_dict
            in_channels=mc["in_channels"],
            classes=1,
        )
    else:
        # 复用 train.py 里的 UNetSmall
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("train_mod",
               Path(__file__).parent / "train.py")
        mod = importlib.util.module_from_spec(spec); sys.modules["train_mod"] = mod
        spec.loader.exec_module(mod)
        model = mod.UNetSmall(mc["in_channels"], mc["base_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[INFO] loaded {cfg.model_path.name}  epoch={ckpt['epoch']}  iou(train-val)={ckpt['iou']:.4f}")
    return model


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, tta: bool,
                  use_rotations: bool) -> torch.Tensor:
    """返回 sigmoid 概率 (B, H, W)。tta=False 时只推一次原图。"""
    if not tta:
        return torch.sigmoid(model(x)).squeeze(1)

    variants = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            xa = x
            if hflip: xa = torch.flip(xa, dims=[3])
            if vflip: xa = torch.flip(xa, dims=[2])
            p = torch.sigmoid(model(xa)).squeeze(1)
            if hflip: p = torch.flip(p, dims=[2])
            if vflip: p = torch.flip(p, dims=[1])
            variants.append(p)

    if use_rotations:
        for k in [1, 2, 3]:
            xa = torch.rot90(x, k, dims=[2, 3])
            p  = torch.sigmoid(model(xa)).squeeze(1)
            p  = torch.rot90(p, -k, dims=[1, 2])
            variants.append(p)

    return torch.stack(variants).mean(dim=0)


@torch.no_grad()
def eval_model(model, loader, device, threshold, tta, use_rotations):
    tp = fp = fn = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        probs = predict_probs(model, imgs, tta=tta, use_rotations=use_rotations)
        pred  = (probs >= threshold).long()
        valid = (lbls != 255)
        tp += int(((pred == 1) & (lbls == 1) & valid).sum())
        fp += int(((pred == 1) & (lbls == 0) & valid).sum())
        fn += int(((pred == 0) & (lbls == 1) & valid).sum())
    iou = tp / max(tp + fp + fn, 1)
    f1  = 2*tp / max(2*tp + fp + fn, 1)
    return {"iou": iou, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    val_ds = TileDataset(cfg.tiles_dir / "val")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    print(f"[INFO] val tiles={len(val_ds)}")

    model = build_model(cfg, device)

    import time
    for tta, label in [(False, "no TTA"),
                       (True,  f"TTA ({'8向' if cfg.use_rotations else '4向'})")]:
        t0 = time.time()
        m = eval_model(model, val_loader, device, cfg.threshold, tta, cfg.use_rotations)
        dt = time.time() - t0
        print(f"[{label:10s}] IoU={m['iou']:.4f}  F1={m['f1']:.4f}  "
              f"TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  t={dt:.1f}s")


if __name__ == "__main__":
    main()
