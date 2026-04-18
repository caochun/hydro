#!/usr/bin/env python3
"""
基于 NDWI / MNDWI 从 Sentinel-2 TOA ENVI 影像提取水体。
无需模型文件，直接运行。波段索引从 .hdr 自动检测。

输出：
  *_ndwi.img      NDWI 指数图，float32
  *_mndwi.img     MNDWI 指数图，float32（有 SWIR 时才输出）
  *_water.img     二值水体掩膜，uint8
  *_water.shp     水体矢量面
  *_visual.png    真彩色 + 水体高亮预览图（蓝色叠加）
"""

from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import mapping, shape
from PIL import Image


# ── 配置 ──────────────────────────────────────────────────────────────────────
IMAGE_PATH = Path(r"/Users/chun/Develop/hydro/S2B_MSIL1C_20260202/S2B_MSIL1C_20260202.img")
OUT_DIR    = Path(r"/Users/chun/Develop/hydro/water_out")

# 波段索引留空 = 从 .hdr 自动检测；也可手动指定 1-based 整数覆盖
BAND_B02 = None  # rhos_492  Blue
BAND_B03 = None  # rhos_560  Green
BAND_B04 = None  # rhos_665  Red
BAND_B08 = None  # rhos_833  NIR
BAND_B11 = None  # rhos_1614 SWIR1（无此波段时跳过 MNDWI）

IMAGE_SCALE   = 10000.0   # 原始值 ÷ 此值 = 反射率
NODATA_VALUE  = 0         # 无效像元标志
NDWI_THRESH   = 0.0       # NDWI  > 此值 判定为水体
MNDWI_THRESH  = 0.0       # MNDWI > 此值 判定为水体（两者取并集）
USE_MNDWI     = True      # True=用 MNDWI，False=只用 NDWI
MIN_AREA_M2   = 100000.0  # 最小水体面积（平方米），过滤噪声小斑块（0.1 km²）
CLOUD_B02_MAX = 0.20      # B02 反射率上限：超过此值视为云，排除（0~1，调小=更严格）
WATER_COLOR   = (0, 100, 220)   # 水体高亮颜色 RGB，默认蓝色
WATER_ALPHA   = 0.55            # 水体叠加透明度（0=完全透明，1=完全覆盖）
PREVIEW_MAX_PX = 10000          # 预览图最长边像素上限，超出则缩放
# ─────────────────────────────────────────────────────────────────────────────


def detect_bands(hdr_path: Path) -> dict:
    """从 ENVI .hdr 解析 band names，返回各波段的 1-based 索引（找不到则 None）。"""
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"band names\s*=\s*\{([^}]+)\}", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return {}
    names = [n.strip().lower() for n in m.group(1).split(",")]
    targets = {"rhos_492": "b02", "rhos_560": "b03", "rhos_665": "b04",
               "rhos_833": "b08", "rhos_1614": "b11"}
    result = {}
    for i, name in enumerate(names, start=1):
        for key, alias in targets.items():
            if key in name:
                result[alias] = i
    return result


def resolve_band(override, detected: dict, key: str) -> int | None:
    return override if override is not None else detected.get(key)


def remove_old(path: Path):
    for p in [path, path.with_suffix(".hdr"), Path(str(path) + ".aux.xml")]:
        if p.exists():
            p.unlink()


def write_envi(path: Path, array: np.ndarray, profile: dict, description: str):
    prof = profile.copy()
    for k in ["blockxsize", "blockysize", "tiled", "compress", "predictor", "bigtiff"]:
        prof.pop(k, None)
    prof.update(driver="ENVI", count=1, dtype=str(array.dtype),
                nodata=None, interleave="band")
    remove_old(path)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array, 1)
        try:
            dst.set_band_description(1, description)
        except Exception:
            pass


def write_visual(path: Path, b04: np.ndarray, b03: np.ndarray, b02: np.ndarray,
                 water: np.ndarray, valid: np.ndarray,
                 water_color: tuple, alpha: float, max_px: int):
    """输出真彩色 + 水体高亮 PNG。"""
    def stretch(band: np.ndarray) -> np.ndarray:
        v = band[valid].ravel()
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[~valid] = 0.0
        return (out * 255).astype(np.uint8)

    r = stretch(b04)
    g = stretch(b03)
    b = stretch(b02)
    rgb = np.stack([r, g, b], axis=-1)          # H×W×3

    # 叠加水体高亮
    wm = water == 1
    for ch, c in enumerate(water_color):
        rgb[wm, ch] = np.clip(
            rgb[wm, ch] * (1 - alpha) + c * alpha, 0, 255
        ).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")

    # 缩放到预览尺寸
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
        if p.exists():
            p.unlink()

    schema = {"geometry": "Polygon", "properties": {"area_m2": "float:24.3"}}
    count = 0
    with fiona.open(path, "w", driver="ESRI Shapefile",
                    schema=schema, crs_wkt=crs_wkt, encoding="UTF-8") as sink:
        for geom_json, val in shapes(mask, mask=(mask == 1),
                                     transform=transform, connectivity=8):
            if int(val) != 1:
                continue
            geom = shape(geom_json)
            if geom.is_empty or geom.area < min_area:
                continue
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = IMAGE_PATH.stem

    hdr_path = IMAGE_PATH.with_suffix(".hdr")
    detected = detect_bands(hdr_path) if hdr_path.exists() else {}
    idx_b02 = resolve_band(BAND_B02, detected, "b02")
    idx_b03 = resolve_band(BAND_B03, detected, "b03")
    idx_b04 = resolve_band(BAND_B04, detected, "b04")
    idx_b08 = resolve_band(BAND_B08, detected, "b08")
    idx_b11 = resolve_band(BAND_B11, detected, "b11")

    for name, idx in [("B02/Blue", idx_b02), ("B03/Green", idx_b03),
                      ("B04/Red", idx_b04), ("B08/NIR", idx_b08)]:
        if idx is None:
            raise ValueError(f"找不到波段 {name}，请在配置区手动指定索引")
    has_swir = idx_b11 is not None
    print(f"[INFO] 波段索引: B02={idx_b02} B03={idx_b03} B04={idx_b04} B08={idx_b08} B11={idx_b11}")
    print(f"[INFO] MNDWI={'启用' if has_swir else '跳过（无 SWIR 波段）'}")

    print(f"[INFO] 读取影像: {IMAGE_PATH}")
    with rasterio.open(IMAGE_PATH) as src:
        print(f"[INFO] size=({src.width}, {src.height})  bands={src.count}  crs={src.crs}")

        # 读波段并归一化
        raw_b02 = src.read(idx_b02)   # Blue
        raw_b03 = src.read(idx_b03)   # Green
        raw_b04 = src.read(idx_b04)   # Red
        raw_b08 = src.read(idx_b08)   # NIR
        raw_b11 = src.read(idx_b11) if has_swir else None  # SWIR1

        b02 = raw_b02.astype(np.float32) / IMAGE_SCALE
        b03 = raw_b03.astype(np.float32) / IMAGE_SCALE
        b04 = raw_b04.astype(np.float32) / IMAGE_SCALE
        b08 = raw_b08.astype(np.float32) / IMAGE_SCALE
        b11 = raw_b11.astype(np.float32) / IMAGE_SCALE if has_swir else None

        # 有效像元掩膜（任一波段为 nodata 则无效）
        valid = (raw_b03 != NODATA_VALUE) & (raw_b08 != NODATA_VALUE)
        if has_swir:
            valid &= (raw_b11 != NODATA_VALUE)

        profile  = src.profile
        transform = src.transform
        crs_wkt  = src.crs.to_wkt() if src.crs else None

    # ── 计算指数 ──────────────────────────────────────────────────────────────
    eps = 1e-8

    ndwi_num = b03 - b08
    ndwi_den = b03 + b08
    ndwi = np.where(np.abs(ndwi_den) > eps, ndwi_num / ndwi_den, 0.0).astype(np.float32)
    ndwi[~valid] = 0.0

    if has_swir:
        mndwi_num = b03 - b11
        mndwi_den = b03 + b11
        mndwi = np.where(np.abs(mndwi_den) > eps, mndwi_num / mndwi_den, 0.0).astype(np.float32)
        mndwi[~valid] = 0.0
    else:
        mndwi = None

    # ── 云掩膜（蓝光亮度过高 = 云）──────────────────────────────────────────
    not_cloud = b02 <= CLOUD_B02_MAX

    # ── 二值化 ────────────────────────────────────────────────────────────────
    water_ndwi  = (ndwi  > NDWI_THRESH) & valid & not_cloud

    if USE_MNDWI and has_swir:
        water_mndwi = (mndwi > MNDWI_THRESH) & valid & not_cloud
        water = (water_ndwi | water_mndwi).astype(np.uint8)
        method = "NDWI|MNDWI"
    else:
        water = water_ndwi.astype(np.uint8)
        method = "NDWI"

    water_px = int(water.sum())
    print(f"[INFO] 方法={method}  水体像元数={water_px:,}  "
          f"覆盖面积≈{water_px * 100:.0f} m²")

    # ── 写输出 ────────────────────────────────────────────────────────────────
    ndwi_path  = OUT_DIR / f"{prefix}_ndwi.img"
    mndwi_path = OUT_DIR / f"{prefix}_mndwi.img"
    mask_path  = OUT_DIR / f"{prefix}_water.img"
    shp_path   = OUT_DIR / f"{prefix}_water.shp"

    write_envi(ndwi_path, ndwi, profile, "NDWI")
    if has_swir and mndwi is not None:
        write_envi(mndwi_path, mndwi, profile, "MNDWI")
    write_envi(mask_path,  water, profile, "water_mask")
    poly_count = write_shp(shp_path, water, transform, crs_wkt, MIN_AREA_M2)

    vis_path = OUT_DIR / f"{prefix}_visual.png"
    vis_size = write_visual(
        vis_path, b04, b03, b02, water, valid,
        WATER_COLOR, WATER_ALPHA, PREVIEW_MAX_PX,
    )

    print(f"[INFO] NDWI  → {ndwi_path}")
    if has_swir:
        print(f"[INFO] MNDWI → {mndwi_path}")
    print(f"[INFO] 掩膜  → {mask_path}")
    print(f"[INFO] 矢量  → {shp_path}  ({poly_count} 个面要素)")
    print(f"[INFO] 预览图 → {vis_path}  {vis_size[0]}×{vis_size[1]} px")


if __name__ == "__main__":
    main()
