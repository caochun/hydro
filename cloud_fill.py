#!/usr/bin/env python3
"""
云掩膜 + 双时相填补：修复冰雪/薄云场景下的水体漏检。

步骤：
  1. 对每景原始影像计算云掩膜（薄卷云 + 厚云）
  2. 对 prob_max 施加云掩膜，得到"可信水体"
  3. 两景逻辑 OR 填补：景 A 云区 → 用景 B 的结果填补，反之亦然
  4. 输出填补后矢量 + 预览图
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import mapping, shape
from PIL import Image

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/chun/Develop/hydro/data/ice and snow")
PROB_DIR = Path("/Users/chun/Develop/hydro/water_out_ice")
OUT_DIR  = Path("/Users/chun/Develop/hydro/water_out_ice")

SCENES = [
    "S2A_MSIL1C_20241225",
    "S2A_MSIL1C_20250104",
]

# 云掩膜判据（TOA 反射率，已除以 10000）
CLOUD_B02_MIN   = 0.20   # 蓝光高
CLOUD_B11_MIN   = 0.10   # SWIR 也高 → 厚云/卷云
CIRRUS_NDSI_MAX = 0.10   # NDSI 低且蓝光偏高 → 卷云
# 云影：只在厚云周边 N 像素内查找异常暗区（避免把深水误判）
SHADOW_SEARCH_PX = 200   # 在云像素周边多少像素内搜索云影（冬季太阳角低，影子投射远）
SHADOW_B08_MAX   = 0.04  # NIR 极低（云影内）
CLOUD_DILATE_PX  = 8     # 云掩膜边缘膨胀，覆盖云边缘过渡带

MIN_AREA_M2 = 100_000.0
# ─────────────────────────────────────────────────────────────────────────────


def compute_cloud_mask(img_path: Path) -> np.ndarray:
    """返回 bool 数组，True = 云或云影（不可信）。"""
    from scipy.ndimage import binary_dilation
    with rasterio.open(img_path) as src:
        b02 = src.read(2).astype(np.float32) / 10000.0   # 492nm
        b03 = src.read(3).astype(np.float32) / 10000.0   # 560nm
        b08 = src.read(8).astype(np.float32) / 10000.0   # 833nm NIR
        b11 = src.read(11).astype(np.float32) / 10000.0  # 1614nm SWIR

    ndsi = (b03 - b11) / (b03 + b11 + 1e-6)

    thick_cloud  = (b02 > CLOUD_B02_MIN) & (b11 > CLOUD_B11_MIN)
    cirrus_cloud = (ndsi < CIRRUS_NDSI_MAX) & (b02 > 0.15) & (b11 < CLOUD_B11_MIN)
    cloud_raw = thick_cloud | cirrus_cloud

    # 云影：只在厚云周边搜索范围内寻找 NIR 异常暗的像素
    cloud_zone   = binary_dilation(cloud_raw, iterations=SHADOW_SEARCH_PX)
    cloud_shadow = cloud_zone & (b08 < SHADOW_B08_MAX) & ~cloud_raw

    # 最终掩膜：膨胀云体 + 云影
    cloud = binary_dilation(cloud_raw, iterations=CLOUD_DILATE_PX) | cloud_shadow

    pct = cloud.mean() * 100
    print(f"  云+云影覆盖率: {pct:.1f}%  (thick={thick_cloud.mean()*100:.1f}%  cirrus={cirrus_cloud.mean()*100:.1f}%  shadow={cloud_shadow.mean()*100:.1f}%)")
    return cloud


def load_water(stem: str) -> tuple[np.ndarray, dict, object, str]:
    prob_path = PROB_DIR / f"{stem}_prob_max.img"
    with rasterio.open(prob_path) as src:
        prob    = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None
    water = (prob >= 0.5).astype(np.uint8)
    return water, profile, transform, crs_wkt


def write_envi(path: Path, arr: np.ndarray, profile: dict, desc: str):
    prof = profile.copy()
    for k in ["blockxsize","blockysize","tiled","compress","predictor","bigtiff"]:
        prof.pop(k, None)
    prof.update(driver="ENVI", count=1, dtype=str(arr.dtype), nodata=None, interleave="band")
    for suf in ["", ".hdr", ".aux.xml"]:
        p = Path(str(path) + suf) if suf else path
        if p.exists(): p.unlink()
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        try: dst.set_band_description(1, desc)
        except: pass


def write_shp(path: Path, mask: np.ndarray, transform, crs_wkt) -> int:
    for suf in [".shp",".shx",".dbf",".prj",".cpg"]:
        p = path.with_suffix(suf)
        if p.exists(): p.unlink()
    schema = {"geometry": "Polygon", "properties": {"area_m2": "float:24.3"}}
    count = 0
    with fiona.open(path, "w", driver="ESRI Shapefile",
                    schema=schema, crs_wkt=crs_wkt, encoding="UTF-8") as sink:
        for geom_json, val in shapes(mask, mask=(mask==1), transform=transform, connectivity=8):
            if int(val) != 1: continue
            geom = shape(geom_json)
            if geom.is_empty or geom.area < MIN_AREA_M2: continue
            if geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    if part.area >= MIN_AREA_M2:
                        sink.write({"geometry": mapping(part),
                                    "properties": {"area_m2": float(part.area)}})
                        count += 1
            else:
                sink.write({"geometry": mapping(geom),
                            "properties": {"area_m2": float(geom.area)}})
                count += 1
    return count


def write_preview(path: Path, img_path: Path, water: np.ndarray, cloud: np.ndarray):
    with rasterio.open(img_path) as src:
        b02 = src.read(2).astype(np.float32) / 10000.0
        b03 = src.read(3).astype(np.float32) / 10000.0
        b04 = src.read(4).astype(np.float32) / 10000.0
    valid = ~((b02==0)&(b03==0)&(b04==0))

    def stretch(b):
        v = b[valid].ravel()
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out = np.clip((b-lo)/max(hi-lo,1e-6), 0, 1)
        out[~valid] = 0
        return (out*255).astype(np.uint8)

    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)
    # 水体：蓝色
    wm = water == 1
    rgb[wm] = (rgb[wm] * 0.4 + np.array([30,130,220]) * 0.6).clip(0,255).astype(np.uint8)
    # 云区：半透明红色标注
    cm = cloud & valid
    rgb[cm] = (rgb[cm] * 0.6 + np.array([220,60,60]) * 0.4).clip(0,255).astype(np.uint8)

    img = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    if max(h,w) > 10000:
        s = 10000/max(h,w)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    img.save(path, "PNG", optimize=True)
    print(f"  预览 → {path.name}  {img.size[0]}×{img.size[1]}")


def main():
    assert len(SCENES) == 2, "当前实现仅支持双时相填补"
    s0, s1 = SCENES

    print(f"[1] 计算云掩膜")
    img0 = DATA_DIR / f"{s0}.img"
    img1 = DATA_DIR / f"{s1}.img"
    print(f"  {s0}"); cloud0 = compute_cloud_mask(img0)
    print(f"  {s1}"); cloud1 = compute_cloud_mask(img1)

    print(f"\n[2] 加载水体概率图")
    water0, prof0, tr0, crs0 = load_water(s0)
    water1, prof1, tr1, crs1 = load_water(s1)
    print(f"  {s0}: 水体像元={water0.sum():,}")
    print(f"  {s1}: 水体像元={water1.sum():,}")

    print(f"\n[3] 云掩膜 + 双时相填补")
    # 可信水体：非云区域的结果
    trusted0 = water0.copy(); trusted0[cloud0] = 0
    trusted1 = water1.copy(); trusted1[cloud1] = 0

    # 填补：优先用对方的可信结果补云区
    filled0 = trusted0.copy()
    fill_from1 = cloud0 & (trusted1 == 1)
    filled0[fill_from1] = 1

    filled1 = trusted1.copy()
    fill_from0 = cloud1 & (trusted0 == 1)
    filled1[fill_from0] = 1

    # 合并两景填补结果（取并集）
    merged = ((filled0 == 1) | (filled1 == 1)).astype(np.uint8)

    print(f"  {s0} 填补前: {trusted0.sum():,}  填补后: {filled0.sum():,}  (+{fill_from1.sum():,})")
    print(f"  {s1} 填补前: {trusted1.sum():,}  填补后: {filled1.sum():,}  (+{fill_from0.sum():,})")
    print(f"  合并结果: {merged.sum():,} 像元  ≈{merged.sum()*100/1e6:.0f} km²")

    print(f"\n[4] 写出结果")
    for stem, water_filled, cloud, prof, tr, crs, img_path in [
        (s0, filled0, cloud0, prof0, tr0, crs0, img0),
        (s1, filled1, cloud1, prof1, tr1, crs1, img1),
    ]:
        out_mask = OUT_DIR / f"{stem}_water_cloudfill.img"
        out_shp  = OUT_DIR / f"{stem}_water_cloudfill.shp"
        out_vis  = OUT_DIR / f"{stem}_visual_cloudfill.png"
        write_envi(out_mask, water_filled, prof, "water_cloudfill")
        n = write_shp(out_shp, water_filled, tr, crs)
        write_preview(out_vis, img_path, water_filled, cloud)
        print(f"  {stem}: {n} 面 → {out_shp.name}")

    # 合并结果用 s0 的空间参考
    out_merged_shp = OUT_DIR / f"merged_{'_'.join(s[-8:] for s in SCENES)}_water.shp"
    out_merged_img = OUT_DIR / f"merged_{'_'.join(s[-8:] for s in SCENES)}_water.img"
    write_envi(out_merged_img, merged, prof0, "water_merged")
    n = write_shp(out_merged_shp, merged, tr0, crs0)
    print(f"  合并: {n} 面 → {out_merged_shp.name}")


if __name__ == "__main__":
    main()
