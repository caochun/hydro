#!/usr/bin/env python3
"""
将 UNetSmall 与 SMP+TTA 两个模型的概率图融合，输出最终水体矢量/掩膜/预览图。

融合方式（可选）：
  - mean : 概率平均后阈值化（默认，最平滑）
  - max  : 取两者最大概率（更激进，召回高）
  - union: 两者各自阈值化后取并集（最宽松）

输出文件后缀 _{FUSION}。输入 prob 由 extract_water_unet.py 预先产生。
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import mapping, shape
from PIL import Image


# ── 配置 ──────────────────────────────────────────────────────────────────────
SCENE_STEM    = "S2A_MSIL1C_20250701"    # 或 S2A_MSIL1C_20250701
OUT_DIR       = Path("/Users/chun/Develop/hydro/water_out_unet")
IMAGE_PATH    = None      # 用于写 RGB 预览；自动根据 SCENE_STEM 推断
FUSION        = "max"     # "mean" | "max" | "union"
THRESHOLD     = 0.5
MIN_AREA_M2   = 100000.0
WATER_COLOR   = (0, 100, 220)
WATER_ALPHA   = 0.55
PREVIEW_MAX_PX = 10000
# ─────────────────────────────────────────────────────────────────────────────


SCENE_TO_IMG = {
    "S2B_MSIL1C_20260202": Path("/Users/chun/Develop/hydro/S2B_MSIL1C_20260202/S2B_MSIL1C_20260202.img"),
    "S2A_MSIL1C_20250701": Path("/Users/chun/Develop/hydro/S2A_MSIL1C_20250701.img"),
}


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


def write_visual(path: Path, b04, b03, b02, water, valid, color, alpha, max_px):
    def stretch(band):
        v = band[valid].ravel()
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[~valid] = 0.0
        return (out * 255).astype(np.uint8)
    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)
    wm = water == 1
    for ch, c in enumerate(color):
        rgb[wm, ch] = np.clip(rgb[wm, ch] * (1 - alpha) + c * alpha, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    h, w = rgb.shape[:2]
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)
    return img.size


def main():
    prob_a = OUT_DIR / f"{SCENE_STEM}_prob.img"      # UNetSmall
    prob_b = OUT_DIR / f"{SCENE_STEM}_prob_smp.img"  # SMP+TTA
    for p in [prob_a, prob_b]:
        if not p.exists(): raise FileNotFoundError(p)

    print(f"[INFO] 读取 prob: {prob_a.name}, {prob_b.name}")
    with rasterio.open(prob_a) as sa, rasterio.open(prob_b) as sb:
        a = sa.read(1).astype(np.float32)
        b = sb.read(1).astype(np.float32)
        profile = sa.profile.copy()
        transform = sa.transform
        crs_wkt = sa.crs.to_wkt() if sa.crs else None

    print(f"[INFO] fusion={FUSION}  threshold={THRESHOLD}")
    if FUSION == "mean":
        prob = (a + b) / 2.0
        water = (prob >= THRESHOLD).astype(np.uint8)
    elif FUSION == "max":
        prob = np.maximum(a, b)
        water = (prob >= THRESHOLD).astype(np.uint8)
    elif FUSION == "union":
        water = ((a >= THRESHOLD) | (b >= THRESHOLD)).astype(np.uint8)
        prob = np.maximum(a, b)
    else:
        raise ValueError(FUSION)

    # 用原始影像的 RGB 做预览
    img_path = IMAGE_PATH or SCENE_TO_IMG.get(SCENE_STEM)
    if img_path and img_path.exists():
        import re
        hdr = img_path.with_suffix(".hdr")
        bands = {}
        if hdr.exists():
            text = hdr.read_text(errors="ignore")
            m = re.search(r"band names\s*=\s*\{([^}]+)\}", text, re.IGNORECASE | re.DOTALL)
            if m:
                names = [n.strip().lower() for n in m.group(1).split(",")]
                targets = {"rhos_492": "b02", "rhos_560": "b03", "rhos_665": "b04", "rhos_833": "b08"}
                for i, name in enumerate(names, 1):
                    for key, alias in targets.items():
                        if key in name: bands[alias] = i
        with rasterio.open(img_path) as src:
            b02 = src.read(bands.get("b02", 1)).astype(np.float32) / 10000.0
            b03 = src.read(bands.get("b03", 2)).astype(np.float32) / 10000.0
            b04 = src.read(bands.get("b04", 3)).astype(np.float32) / 10000.0
            b08 = src.read(bands.get("b08", 4)).astype(np.float32) / 10000.0
        valid = ~((b02 == 0) & (b03 == 0) & (b04 == 0) & (b08 == 0))
    else:
        valid = np.ones_like(water, dtype=bool)
        b02 = b03 = b04 = None

    water[~valid] = 0
    print(f"[INFO] 水体像元: {int(water.sum()):,}  面积≈{water.sum()*100:.0f} m²")

    prob_out  = OUT_DIR / f"{SCENE_STEM}_prob_{FUSION}.img"
    water_out = OUT_DIR / f"{SCENE_STEM}_water_{FUSION}.img"
    shp_out   = OUT_DIR / f"{SCENE_STEM}_water_{FUSION}.shp"
    vis_out   = OUT_DIR / f"{SCENE_STEM}_visual_{FUSION}.png"

    write_envi(prob_out,  prob,  profile, f"prob_{FUSION}")
    write_envi(water_out, water, profile, f"water_{FUSION}")
    n = write_shp(shp_out, water, transform, crs_wkt, MIN_AREA_M2)
    print(f"[INFO] prob → {prob_out}")
    print(f"[INFO] mask → {water_out}")
    print(f"[INFO] shp  → {shp_out}  ({n} 面)")
    if b02 is not None:
        vw = write_visual(vis_out, b04, b03, b02, water, valid,
                          WATER_COLOR, WATER_ALPHA, PREVIEW_MAX_PX)
        print(f"[INFO] 预览 → {vis_out}  {vw[0]}×{vw[1]} px")


if __name__ == "__main__":
    main()
