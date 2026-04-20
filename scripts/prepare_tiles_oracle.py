#!/usr/bin/env python3
"""
S2B + oracle 真值切瓦片（监督学习用）。

- 只用 S2B_20260202（有 oracle 的那景）
- oracle shapefile 光栅化为 0/1 真值 mask
- 空间切分：按列分 train/val（左 80% / 右 20%）
- 云区（B02>CLOUD_THRESH）标 255 忽略
"""

from pathlib import Path
import numpy as np
import rasterio
import fiona
from rasterio.features import rasterize
from shapely.geometry import shape

# ── 配置 ──────────────────────────────────────────────────────────────────────
IMG_PATH    = Path("/home/chun/hydro_data/S2B_MSIL1C_20260202/S2B_MSIL1C_20260202.img")
ORACLE_SHP  = Path("/home/chun/hydro_data/S2B_MSIL1C_20260202-oracle/20260202_water_mask_0.1.shp")
OUT_DIR     = Path("/home/chun/hydro_data/tiles_oracle")
BANDS       = (1, 2, 3, 4)    # B02 B03 B04 B08
TILE        = 256
STRIDE      = 128
IMAGE_SCALE = 10000.0
NODATA      = 0
CLOUD_THRESH = 0.25
VAL_COL_FRAC = 0.8            # 左 80% train，右 20% val
MIN_VALID    = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def positions(length, tile, stride):
    xs = list(range(0, length - tile + 1, stride))
    if not xs or xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def build_oracle_mask(img_path: Path, shp_path: Path) -> np.ndarray:
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        transform = src.transform
    with fiona.open(shp_path) as f:
        geoms = [shape(feat["geometry"]) for feat in f]
    mask = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train").mkdir(exist_ok=True)
    (OUT_DIR / "val").mkdir(exist_ok=True)

    print("[INFO] 光栅化 oracle ...")
    oracle = build_oracle_mask(IMG_PATH, ORACLE_SHP)
    H, W = oracle.shape
    print(f"[INFO] oracle size=({W},{H})  水体像元={oracle.sum():,}")

    split_x = int(W * VAL_COL_FRAC)
    print(f"[INFO] 切分: train x<{split_x}, val x>={split_x}")

    train_idx = val_idx = 0
    skip_valid = skip_region = 0

    with rasterio.open(IMG_PATH) as src:
        xs = positions(W, TILE, STRIDE)
        ys = positions(H, TILE, STRIDE)
        total = len(xs) * len(ys)
        print(f"[INFO] 总瓦片数（过滤前）: {total}")

        for y0 in ys:
            win = rasterio.windows.Window(0, y0, W, min(TILE, H - y0))
            row = src.read(list(BANDS), window=win).astype(np.float32) / IMAGE_SCALE
            for x0 in xs:
                x1 = min(x0 + TILE, W)
                y1 = min(y0 + TILE, H)
                h_eff, w_eff = y1 - y0, x1 - x0

                patch = row[:, :h_eff, x0:x1]
                tile_img = np.zeros((4, TILE, TILE), dtype=np.float32)
                tile_img[:, :h_eff, :w_eff] = patch

                valid = ~np.all(tile_img == 0, axis=0)
                if valid.mean() < MIN_VALID:
                    skip_valid += 1
                    continue

                # 瓦片完整落在 train 或 val 区域
                if x1 <= split_x:
                    subset = "train"
                elif x0 >= split_x:
                    subset = "val"
                else:
                    skip_region += 1
                    continue

                # 标签 = oracle + 云掩膜忽略
                lbl = np.zeros((TILE, TILE), dtype=np.uint8)
                lbl[:h_eff, :w_eff] = oracle[y0:y0+h_eff, x0:x0+w_eff]
                b02 = tile_img[0]
                cloud = b02 > CLOUD_THRESH
                lbl[cloud] = 255
                lbl[~valid] = 255

                if subset == "train":
                    idx = train_idx; train_idx += 1
                else:
                    idx = val_idx; val_idx += 1
                np.save(OUT_DIR / subset / f"img_{idx:07d}.npy", tile_img)
                np.save(OUT_DIR / subset / f"lbl_{idx:07d}.npy", lbl)

    print(f"\n[INFO] 完成")
    print(f"  train: {train_idx}  val: {val_idx}")
    print(f"  丢弃(有效像元不足): {skip_valid}  跨边界: {skip_region}")

    # 抽样统计类别比例
    for subset, n in [("train", train_idx), ("val", val_idx)]:
        if n == 0: continue
        water = land = 0
        for i in range(min(n, 500)):
            l = np.load(OUT_DIR / subset / f"lbl_{i:07d}.npy")
            water += (l == 1).sum()
            land  += (l == 0).sum()
        ratio = water / max(land, 1)
        print(f"  {subset}: 水/陆 = {ratio:.3f}  pos_weight≈{1/max(ratio,1e-6):.1f}")


if __name__ == "__main__":
    main()
