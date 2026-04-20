#!/usr/bin/env python3
"""
基于形态指标给水体 polygon 分类（lake / river / ambiguous）。

思路：
  把临近的水体 polygon 先 buffer + dissolve 成"连通组"（大型河段切片合并
  成整条河），再对组整体做形态分析。这样主干河道会被正确识别为细长的河，
  孤立湖泊还是紧凑的湖。

每个组指标：
  group_elongation = 最小外接矩形的 长 / 宽       （主判据）
  group_solidity   = Area / ConvexHull.Area      （辅助）
  n_members        = 组内原始 polygon 数

个体 polygon 继承组的分类。

读取 *_water_max.shp，写出 *_classified.shp（新增 type / compactness /
elongation / length_m / width_m 字段）。
"""

from pathlib import Path
import fiona
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree
import math


# ── 配置 ──────────────────────────────────────────────────────────────────────
IN_SHAPES = [
    Path("/Users/chun/Develop/hydro/water_out_unet/S2B_MSIL1C_20260202_water_max.shp"),
    Path("/Users/chun/Develop/hydro/water_out_unet/S2A_MSIL1C_20250701_water_max.shp"),
]
OUT_SUFFIX = "_classified"   # 输出名后缀

BUFFER_DIST       = 100.0 # 米，相距小于此距离的 polygon 归入同一连通组
# 判据优先级：先看 solidity（能识别河网），再看 elongation
RIVER_SOLIDITY    = 0.25  # 组 solidity ≤ 此值 → river（河网分散不填凸包）
RIVER_ELONG       = 6.0   # 组 elongation ≥ 此值 → river（直/弯的细长河段）
LAKE_SOLIDITY     = 0.40  # 组 solidity ≥ 此值 且 elong 小 → lake
LAKE_ELONG        = 2.5   # 组 elongation ≤ 此值 且 solidity 足够 → lake
# ─────────────────────────────────────────────────────────────────────────────


def morph_metrics(geom):
    """返回 (compactness, elongation, solidity, length_m, width_m)。"""
    area = geom.area
    perim = geom.length
    compactness = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0

    # 最小外接矩形
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        def dist(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])
        e1 = dist(coords[0], coords[1])
        e2 = dist(coords[1], coords[2])
        length_m = max(e1, e2)
        width_m  = min(e1, e2)
        elongation = length_m / width_m if width_m > 0 else 999.0
    except Exception:
        length_m = width_m = elongation = 0.0

    # 凸包填充率 solidity
    try:
        hull_area = geom.convex_hull.area
        solidity = area / hull_area if hull_area > 0 else 0.0
    except Exception:
        solidity = 0.0

    return compactness, elongation, solidity, length_m, width_m


def classify(elong: float, solidity: float) -> str:
    # 1) solidity 非常低 → 河网（分散/枝状/蜿蜒，不填凸包）
    if solidity <= RIVER_SOLIDITY:
        return "river"
    # 2) elongation 非常大 → 河（直或弯的细长河段）
    if elong >= RIVER_ELONG:
        return "river"
    # 3) 形状圆润 + 凸包充满 → 湖
    if elong <= LAKE_ELONG and solidity >= LAKE_SOLIDITY:
        return "lake"
    # 4) 其余：偏圆但 solidity 不高（破碎湖岸） → lake 宽松判
    if elong <= LAKE_ELONG:
        return "lake"
    # 5) 3 < elong < 6 + solidity 够高 → 宽河/水库，记 ambiguous
    return "ambiguous"


def build_groups(parts, buffer_dist):
    """buffer + dissolve 后返回连通组 polygons 列表。"""
    print(f"  buffer {buffer_dist}m + dissolve ...")
    buffered = [p.buffer(buffer_dist) for p in parts]
    dissolved = unary_union(buffered)
    if dissolved.is_empty:
        return []
    if dissolved.geom_type == "Polygon":
        return [dissolved]
    return list(dissolved.geoms)


def process(in_path: Path):
    out_path = in_path.with_name(in_path.stem + OUT_SUFFIX + in_path.suffix)
    for suf in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = out_path.with_suffix(suf)
        if p.exists(): p.unlink()

    schema = {
        "geometry": "Polygon",
        "properties": {
            "area_m2":     "float:24.3",
            "g_elong":     "float:10.2",   # 组 elongation
            "g_solidity":  "float:10.4",   # 组 solidity
            "g_members":   "int",           # 组内 polygon 数
            "g_area_km2":  "float:16.3",   # 组总面积 km²
            "elongation":  "float:10.2",   # 个体 elongation（保留参考）
            "solidity":    "float:10.4",
            "type":        "str:16",
        },
    }

    print(f"[{in_path.name}]")
    counts = {"lake": 0, "river": 0, "ambiguous": 0}

    with fiona.open(in_path) as src:
        crs_wkt = src.crs_wkt
        feats = list(src)
    parts = []
    part_props = []
    for feat in feats:
        geom = shape(feat["geometry"])
        if geom.is_empty: continue
        sub = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in sub:
            parts.append(p)
            part_props.append(feat["properties"])
    print(f"  原始 polygon 数: {len(parts)}")

    groups = build_groups(parts, BUFFER_DIST)
    print(f"  连通组数: {len(groups)}")

    # 为每个组预计算指标
    group_info = []
    for g in groups:
        _, g_elong, g_sol, _, _ = morph_metrics(g)
        t = classify(g_elong, g_sol)
        group_info.append({
            "geom":     g,
            "elong":    g_elong,
            "solidity": g_sol,
            "area":     g.area,
            "type":     t,
            "count":    0,
        })

    # STRtree 空间索引
    tree = STRtree([gi["geom"] for gi in group_info])

    with fiona.open(out_path, "w",
                    driver="ESRI Shapefile",
                    schema=schema,
                    crs_wkt=crs_wkt,
                    encoding="UTF-8") as dst:
        for part, props in zip(parts, part_props):
            # 找所属组（STRtree 返回索引）
            idxs = tree.query(part)
            group_idx = None
            for i in idxs:
                if group_info[i]["geom"].intersects(part):
                    group_idx = i
                    break
            if group_idx is None:
                # 兜底：最近的组
                group_idx = 0
            gi = group_info[group_idx]
            gi["count"] += 1

            # 个体 elongation/solidity 也算一下供参考
            _, p_elong, p_sol, _, _ = morph_metrics(part)
            t = gi["type"]
            counts[t] += 1
            dst.write({
                "geometry": mapping(part),
                "properties": {
                    "area_m2":    float(part.area),
                    "g_elong":    float(gi["elong"]),
                    "g_solidity": float(gi["solidity"]),
                    "g_members":  int(gi["count"]),  # 先占位，后面再补准确值
                    "g_area_km2": float(gi["area"] / 1e6),
                    "elongation": float(p_elong),
                    "solidity":   float(p_sol),
                    "type":       t,
                },
            })

    total = sum(counts.values())
    print(f"  总面: {total}")
    for t in ["lake", "river", "ambiguous"]:
        n = counts[t]
        pct = 100 * n / max(total, 1)
        print(f"  {t:10s}: {n:5d} ({pct:.1f}%)")

    # 打印最大的几个组信息
    big = sorted(group_info, key=lambda x: -x["area"])[:5]
    print(f"  最大 5 组:")
    for gi in big:
        print(f"    area={gi['area']/1e6:.1f}km²  elong={gi['elong']:.2f}  sol={gi['solidity']:.2f}  "
              f"members={gi['count']}  type={gi['type']}")
    print(f"  → {out_path.name}\n")


def main():
    for p in IN_SHAPES:
        if p.exists():
            process(p)
        else:
            print(f"[WARN] 不存在: {p}")


if __name__ == "__main__":
    main()
