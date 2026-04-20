#!/usr/bin/env python3
"""
从 Sentinel-2 ENVI 影像切瓦片，用 NDWI 高置信区自动生成弱监督标签。

标签规则：
  NDWI > WATER_THRESH  → 1（水体）
  NDWI < LAND_THRESH   → 0（非水体）
  中间区域             → 255（忽略，训练时跳过）

输出：
  tiles/
    img_XXXXXX.npy   float32, shape (4, TILE, TILE)，已归一化到 0~1
    lbl_XXXXXX.npy   uint8,   shape (TILE, TILE)，值 0/1/255
"""

from pathlib import Path
import numpy as np
import rasterio

# ── 配置 ──────────────────────────────────────────────────────────────────────
SOURCES = [
    {
        "img":    Path("/home/chun/hydro_data/S2A_MSIL1C_20250701.img"),
        "bands":  (7, 8, 9, 13),   # B02 B03 B04 B08（从 .hdr band names 确定）
    },
    {
        "img":    Path("/home/chun/hydro_data/S2B_MSIL1C_20260202/S2B_MSIL1C_20260202.img"),
        "bands":  (1, 2, 3, 4),    # B02 B03 B04 B08
    },
]
OUT_DIR      = Path("/home/chun/hydro_data/tiles")
TILE         = 256
STRIDE       = 128
IMAGE_SCALE  = 10000.0
NODATA       = 0
WATER_THRESH = 0.2    # NDWI > 此值 → 确定水体
LAND_THRESH  = -0.3   # NDWI < 此值 → 确定非水体
MIN_VALID    = 0.5    # 瓦片中有效像元比例下限
MIN_LABEL    = 0.01   # 瓦片中有标签像元（非255）比例下限
# ─────────────────────────────────────────────────────────────────────────────


def tile_positions(length: int, tile: int, stride: int):
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def process_source(src_cfg: dict, out_dir: Path, start_idx: int) -> int:
    img_path = src_cfg["img"]
    bands    = src_cfg["bands"]
    b_green, b_nir = bands[1], bands[3]  # B03, B08

    print(f"\n[INFO] 处理: {img_path.name}")
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        print(f"  size=({W}, {H})  bands={src.count}")

        xs = tile_positions(W, TILE, STRIDE)
        ys = tile_positions(H, TILE, STRIDE)
        total = len(xs) * len(ys)
        print(f"  瓦片数（含过滤前）: {total}")

        count = 0
        skip_valid = skip_label = 0
        idx = start_idx

        for y0 in ys:
            win = rasterio.windows.Window(0, y0, W, min(TILE, H - y0))
            # 读整行节省 IO
            row_data = src.read(list(bands), window=win).astype(np.float32) / IMAGE_SCALE

            for x0 in xs:
                x1 = min(x0 + TILE, W)
                y1 = min(y0 + TILE, H)
                h_eff, w_eff = y1 - y0, x1 - x0

                patch = row_data[:, :h_eff, x0:x1]

                # 填充到 TILE×TILE
                tile_img = np.zeros((4, TILE, TILE), dtype=np.float32)
                tile_img[:, :h_eff, :w_eff] = patch

                valid = ~np.all(tile_img == 0, axis=0)
                if valid.mean() < MIN_VALID:
                    skip_valid += 1
                    continue

                g = tile_img[1]  # B03 Green
                n = tile_img[3]  # B08 NIR
                eps = 1e-8
                ndwi = np.where(np.abs(g + n) > eps, (g - n) / (g + n), 0.0)

                lbl = np.full((TILE, TILE), 255, dtype=np.uint8)
                lbl[ndwi >  WATER_THRESH] = 1
                lbl[ndwi <  LAND_THRESH]  = 0
                lbl[~valid] = 255

                labeled_ratio = (lbl != 255).mean()
                if labeled_ratio < MIN_LABEL:
                    skip_label += 1
                    continue

                np.save(out_dir / f"img_{idx:07d}.npy", tile_img)
                np.save(out_dir / f"lbl_{idx:07d}.npy", lbl)
                idx += 1
                count += 1

        print(f"  保留: {count}  丢弃(有效像元不足): {skip_valid}  丢弃(标签太少): {skip_label}")
    return idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = 0
    for cfg in SOURCES:
        idx = process_source(cfg, OUT_DIR, idx)
    print(f"\n[INFO] 完成，共 {idx} 个瓦片 → {OUT_DIR}")

    # 统计水体/非水体比例
    lbls = list(OUT_DIR.glob("lbl_*.npy"))
    water = land = 0
    for p in lbls[:1000]:  # 抽样统计
        l = np.load(p)
        water += (l == 1).sum()
        land  += (l == 0).sum()
    ratio = water / max(land, 1)
    print(f"[INFO] 水体/非水体比（前1000瓦片抽样）: {ratio:.3f}  pos_weight≈{1/max(ratio,1e-6):.1f}")


if __name__ == "__main__":
    main()
