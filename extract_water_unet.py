#!/usr/bin/env python3
"""
用训练好的 UNetSmall 从 Sentinel-2 TOA ENVI 影像提取水体。

- 波段自动从 .hdr 检测（B02/B03/B04/B08）
- 滑窗推理 + 概率平均
- 输出: *_prob.img, *_water.img, *_water.shp, *_visual.png
"""

from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
import fiona
from shapely.geometry import mapping, shape
from PIL import Image


# ── 配置 ──────────────────────────────────────────────────────────────────────
IMAGE_PATH  = Path("/Users/chun/Develop/hydro/S2A_MSIL1C_20250701.img")
MODEL_PATH  = Path("/Users/chun/Develop/hydro/runs/best_model.pt")
OUT_DIR     = Path("/Users/chun/Develop/hydro/water_out_unet")

BAND_B02 = None  # None = 从 .hdr 自动检测
BAND_B03 = None
BAND_B04 = None
BAND_B08 = None

STRIDE_FRAC   = 0.5     # 瓦片步长 = tile * STRIDE_FRAC，重叠区取概率平均
BATCH_SIZE    = 16
MIN_AREA_M2   = 100000.0
WATER_COLOR   = (0, 100, 220)
WATER_ALPHA   = 0.55
PREVIEW_MAX_PX = 10000
# ─────────────────────────────────────────────────────────────────────────────


# ── 模型（须与 train.py 一致）────────────────────────────────────────────────
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


# ── 工具 ─────────────────────────────────────────────────────────────────────
def detect_bands(hdr_path: Path) -> dict:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"band names\s*=\s*\{([^}]+)\}", text, re.IGNORECASE | re.DOTALL)
    if not m: return {}
    names = [n.strip().lower() for n in m.group(1).split(",")]
    targets = {"rhos_492": "b02", "rhos_560": "b03", "rhos_665": "b04", "rhos_833": "b08"}
    result = {}
    for i, name in enumerate(names, start=1):
        for key, alias in targets.items():
            if key in name:
                result[alias] = i
    return result


def resolve_band(override, detected: dict, key: str):
    return override if override is not None else detected.get(key)


def remove_old(path: Path):
    for p in [path, path.with_suffix(".hdr"), Path(str(path) + ".aux.xml")]:
        if p.exists(): p.unlink()


def write_envi(path: Path, array: np.ndarray, profile: dict, description: str):
    prof = profile.copy()
    for k in ["blockxsize", "blockysize", "tiled", "compress", "predictor", "bigtiff"]:
        prof.pop(k, None)
    prof.update(driver="ENVI", count=1, dtype=str(array.dtype), nodata=None, interleave="band")
    remove_old(path)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array, 1)
        try: dst.set_band_description(1, description)
        except Exception: pass


def write_visual(path: Path, b04, b03, b02, water, valid, water_color, alpha, max_px):
    def stretch(band):
        v = band[valid].ravel()
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[~valid] = 0.0
        return (out * 255).astype(np.uint8)
    r = stretch(b04); g = stretch(b03); b = stretch(b02)
    rgb = np.stack([r, g, b], axis=-1)
    wm = water == 1
    for ch, c in enumerate(water_color):
        rgb[wm, ch] = np.clip(rgb[wm, ch] * (1 - alpha) + c * alpha, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    h, w = rgb.shape[:2]
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)
    return img.size


def write_shp(path: Path, mask: np.ndarray, transform, crs_wkt, min_area: float):
    for suf in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = path.with_suffix(suf)
        if p.exists(): p.unlink()
    schema = {"geometry": "Polygon", "properties": {"area_m2": "float:24.3"}}
    count = 0
    with fiona.open(path, "w", driver="ESRI Shapefile",
                    schema=schema, crs_wkt=crs_wkt, encoding="UTF-8") as sink:
        for geom_json, val in shapes(mask, mask=(mask == 1), transform=transform, connectivity=8):
            if int(val) != 1: continue
            geom = shape(geom_json)
            if geom.is_empty or geom.area < min_area: continue
            if geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    if part.area >= min_area:
                        sink.write({"geometry": mapping(part),
                                    "properties": {"area_m2": float(part.area)}})
                        count += 1
            else:
                sink.write({"geometry": mapping(geom),
                            "properties": {"area_m2": float(geom.area)}})
                count += 1
    return count


def positions(length, tile, stride):
    xs = list(range(0, length - tile + 1, stride))
    if not xs or xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = IMAGE_PATH.stem

    # 加载模型
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    mc = ckpt["config"]
    tile = mc["tile_size"]
    image_scale = mc["image_scale"]
    clip_max = mc["image_clip_max"]
    threshold = mc["threshold"]
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device={device}  tile={tile}  threshold={threshold}  epoch={ckpt['epoch']}  IoU={ckpt['iou']:.4f}")

    model = UNetSmall(mc["in_channels"], mc["base_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 波段检测
    hdr = IMAGE_PATH.with_suffix(".hdr")
    detected = detect_bands(hdr) if hdr.exists() else {}
    idx_b02 = resolve_band(BAND_B02, detected, "b02")
    idx_b03 = resolve_band(BAND_B03, detected, "b03")
    idx_b04 = resolve_band(BAND_B04, detected, "b04")
    idx_b08 = resolve_band(BAND_B08, detected, "b08")
    for name, idx in [("B02", idx_b02), ("B03", idx_b03), ("B04", idx_b04), ("B08", idx_b08)]:
        if idx is None: raise ValueError(f"找不到波段 {name}")
    print(f"[INFO] 波段: B02={idx_b02} B03={idx_b03} B04={idx_b04} B08={idx_b08}")

    bands = (idx_b02, idx_b03, idx_b04, idx_b08)
    stride = max(1, int(tile * STRIDE_FRAC))

    with rasterio.open(IMAGE_PATH) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None
        print(f"[INFO] size=({W},{H})  crs={src.crs}")

        prob_sum = np.zeros((H, W), dtype=np.float32)
        cnt      = np.zeros((H, W), dtype=np.float32)

        xs = positions(W, tile, stride)
        ys = positions(H, tile, stride)
        total = len(xs) * len(ys)
        print(f"[INFO] 瓦片数: {total}  stride={stride}")

        batch_patches = []
        batch_xy      = []
        done = 0

        def flush():
            nonlocal batch_patches, batch_xy
            if not batch_patches: return
            x = torch.from_numpy(np.stack(batch_patches)).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            for (x0, y0, h_eff, w_eff), p in zip(batch_xy, probs):
                prob_sum[y0:y0+h_eff, x0:x0+w_eff] += p[:h_eff, :w_eff]
                cnt     [y0:y0+h_eff, x0:x0+w_eff] += 1.0
            batch_patches = []
            batch_xy = []

        for y0 in ys:
            win = Window(0, y0, W, min(tile, H - y0))
            row = src.read(list(bands), window=win).astype(np.float32) / image_scale
            row = np.clip(row, 0.0, clip_max)
            for x0 in xs:
                x1 = min(x0 + tile, W); y1 = min(y0 + tile, H)
                h_eff, w_eff = y1 - y0, x1 - x0
                patch = np.zeros((4, tile, tile), dtype=np.float32)
                patch[:, :h_eff, :w_eff] = row[:, :h_eff, x0:x1]
                batch_patches.append(patch)
                batch_xy.append((x0, y0, h_eff, w_eff))
                if len(batch_patches) >= BATCH_SIZE:
                    flush()
                done += 1
            if done % 200 == 0 or done == total:
                print(f"  推理 {done}/{total}", flush=True)
        flush()

        # 读 RGB 用于可视化
        b02 = src.read(idx_b02).astype(np.float32) / image_scale
        b03 = src.read(idx_b03).astype(np.float32) / image_scale
        b04 = src.read(idx_b04).astype(np.float32) / image_scale
        b08 = src.read(idx_b08).astype(np.float32) / image_scale

    valid = ~((b02 == 0) & (b03 == 0) & (b04 == 0) & (b08 == 0))
    prob_avg = prob_sum / np.maximum(cnt, 1e-6)
    prob_avg[~valid] = 0.0
    water = ((prob_avg >= threshold) & valid).astype(np.uint8)

    print(f"[INFO] 水体像元={int(water.sum()):,}  面积≈{water.sum()*100:.0f} m²")

    prob_path  = OUT_DIR / f"{prefix}_prob.img"
    water_path = OUT_DIR / f"{prefix}_water.img"
    shp_path   = OUT_DIR / f"{prefix}_water.shp"
    vis_path   = OUT_DIR / f"{prefix}_visual.png"

    write_envi(prob_path,  prob_avg, profile, "water_prob")
    write_envi(water_path, water,    profile, "water_mask")
    n_poly = write_shp(shp_path, water, transform, crs_wkt, MIN_AREA_M2)
    vw = write_visual(vis_path, b04, b03, b02, water, valid,
                      WATER_COLOR, WATER_ALPHA, PREVIEW_MAX_PX)

    print(f"[INFO] prob  → {prob_path}")
    print(f"[INFO] mask  → {water_path}")
    print(f"[INFO] shp   → {shp_path}  ({n_poly} 面)")
    print(f"[INFO] 预览  → {vis_path}  {vw[0]}×{vw[1]} px")


if __name__ == "__main__":
    main()
