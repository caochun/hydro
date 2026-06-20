#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Match experiment Swater polygons with external water priors.

JRC GSW and ESA WorldCover are handled as local GeoTIFF rasters. Dynamic World
requires Google Earth Engine authentication because it is published as an Earth
Engine ImageCollection rather than static downloadable tiles.
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask
from shapely.geometry import box, mapping
from shapely.ops import unary_union


DEFAULT_EXPERIMENT_ROOT = Path("data/实验数据")
DEFAULT_OUT_DIR = Path("data_download/downloads/experiment_external_water")
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"


@dataclass(frozen=True)
class ExperimentShape:
    sample_id: str
    scene: str
    date: str
    path: Path
    gdf: gpd.GeoDataFrame


def no_proxy_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.proxies = {}
    return session


def parse_scene_date(scene: str) -> str:
    match = re.search(r"(20\d{6}|201\d{5})", scene)
    if not match:
        return ""
    return match.group(1)


def iter_swater_shapes(root: Path) -> list[ExperimentShape]:
    shapes: list[ExperimentShape] = []
    for path in sorted(root.glob("*/*_Swater.shp")):
        gdf = gpd.read_file(path)
        if gdf.empty:
            continue
        scene = path.stem.replace("_Swater", "")
        shapes.append(
            ExperimentShape(
                sample_id=path.parent.name,
                scene=scene,
                date=parse_scene_date(scene),
                path=path,
                gdf=gdf,
            )
        )
    return shapes


def raster_values(src: rasterio.DatasetReader, geom) -> np.ndarray:
    try:
        arr, _ = mask(src, [mapping(geom)], crop=True, filled=True)
    except ValueError:
        return np.array([], dtype="float64")
    values = arr[0]
    valid = np.isfinite(values)
    if src.nodata is not None:
        valid &= values != src.nodata
    return values[valid]


def build_dynamic_world_manifest(
    shapes: list[ExperimentShape],
    out_path: Path,
    padding_deg: float,
) -> pd.DataFrame:
    rows = []
    for item in shapes:
        wgs84 = item.gdf.to_crs("EPSG:4326")
        geom = unary_union([geom for geom in wgs84.geometry if geom is not None and not geom.is_empty])
        if geom.is_empty:
            continue
        xmin, ymin, xmax, ymax = geom.bounds
        rows.append(
            {
                "sample_id": item.sample_id,
                "scene": item.scene,
                "date": item.date,
                "source_path": str(item.path),
                "xmin": xmin - padding_deg,
                "ymin": ymin - padding_deg,
                "xmax": xmax + padding_deg,
                "ymax": ymax + padding_deg,
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def compute_jrc_esa_stats(experiment_root: Path, out_dir: Path) -> None:
    shapes = iter_swater_shapes(experiment_root)
    rasters = {
        "jrc_occurrence": out_dir / "jrc_gsw/experiment_jrc_occurrence_clip.tif",
        "jrc_seasonality": out_dir / "jrc_gsw/experiment_jrc_seasonality_clip.tif",
        "esa_worldcover": out_dir / "esa_worldcover/experiment_esa_worldcover_clip.tif",
    }
    missing = [str(path) for path in rasters.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prior rasters: " + ", ".join(missing))

    rows = []
    with rasterio.open(rasters["jrc_occurrence"]) as jrc_occ, rasterio.open(
        rasters["jrc_seasonality"]
    ) as jrc_seas, rasterio.open(rasters["esa_worldcover"]) as esa:
        for item in shapes:
            for index, geom in enumerate(item.gdf.geometry):
                if geom is None or geom.is_empty:
                    continue
                row = {
                    "sample_id": item.sample_id,
                    "scene": item.scene,
                    "date": item.date,
                    "feature_index": index,
                    "source_path": str(item.path),
                }
                geom_jrc = gpd.GeoSeries([geom], crs=item.gdf.crs).to_crs(jrc_occ.crs).iloc[0]
                geom_esa = gpd.GeoSeries([geom], crs=item.gdf.crs).to_crs(esa.crs).iloc[0]
                for key, src, prior_geom in [
                    ("jrc_occurrence", jrc_occ, geom_jrc),
                    ("jrc_seasonality", jrc_seas, geom_jrc),
                    ("esa_worldcover", esa, geom_esa),
                ]:
                    values = raster_values(src, prior_geom)
                    row[f"{key}_valid_pixels"] = int(values.size)
                    if values.size == 0:
                        row[f"{key}_mean"] = np.nan
                        row[f"{key}_max"] = np.nan
                        if key == "esa_worldcover":
                            row["esa_class_counts"] = ""
                            row["esa_water_fraction"] = np.nan
                            row["esa_water_pixels"] = 0
                        continue
                    row[f"{key}_mean"] = float(np.mean(values))
                    row[f"{key}_max"] = float(np.max(values))
                    if key == "esa_worldcover":
                        unique, counts = np.unique(values.astype("int64"), return_counts=True)
                        row["esa_class_counts"] = ";".join(
                            f"{int(value)}:{int(count)}" for value, count in zip(unique, counts)
                        )
                        row["esa_water_fraction"] = float(np.mean(values == 80))
                        row["esa_water_pixels"] = int(np.sum(values == 80))
                rows.append(row)

    stats = pd.DataFrame(rows)
    stats_path = out_dir / "experiment_external_prior_stats.csv"
    stats.to_csv(stats_path, index=False)
    summary = (
        stats.groupby("sample_id", dropna=False)
        .agg(
            swater_features=("feature_index", "count"),
            mean_jrc_occurrence=("jrc_occurrence_mean", "mean"),
            mean_jrc_seasonality=("jrc_seasonality_mean", "mean"),
            mean_esa_water_fraction=("esa_water_fraction", "mean"),
            total_esa_water_pixels=("esa_water_pixels", "sum"),
            total_esa_valid_pixels=("esa_worldcover_valid_pixels", "sum"),
        )
        .reset_index()
    )
    summary_path = out_dir / "experiment_external_prior_summary_by_sample.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {stats_path} {stats.shape}")
    print(f"wrote {summary_path} {summary.shape}")
    print(summary.to_string(index=False))


def require_earth_engine():
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "Dynamic World 下载需要 earthengine-api。先运行："
            "./.venv/bin/python -m pip install earthengine-api"
        ) from exc
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Dynamic World 下载需要 Google Earth Engine 认证。先运行："
            "./.venv/bin/earthengine authenticate，然后重试本命令。"
        ) from exc
    return ee


def download_url(url: str, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{stem}.zip"
    session = no_proxy_session()
    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    extracted = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            target = out_dir / f"{stem}_{Path(name).name}"
            with archive.open(name) as source, target.open("wb") as dest:
                dest.write(source.read())
            extracted.append(target)
    return extracted


def download_dynamic_world(
    experiment_root: Path,
    out_dir: Path,
    padding_deg: float,
    window_days: int,
    limit: int | None,
) -> None:
    ee = require_earth_engine()
    shapes = iter_swater_shapes(experiment_root)
    manifest_path = out_dir / "dynamic_world/experiment_dynamic_world_manifest.csv"
    manifest = build_dynamic_world_manifest(shapes, manifest_path, padding_deg)
    dw_dir = out_dir / "dynamic_world"
    rows = []
    for _, row in manifest.iterrows():
        if limit is not None and len(rows) >= limit:
            break
        if not row["date"]:
            continue
        date = datetime.strptime(str(row["date"]), "%Y%m%d")
        start = (date - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")
        region = ee.Geometry.Rectangle(
            [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])],
            proj="EPSG:4326",
            geodesic=False,
        )
        collection = (
            ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
            .filterBounds(region)
            .filterDate(start, end)
        )
        count = int(collection.size().getInfo())
        if count == 0:
            rows.append({**row.to_dict(), "dw_image_count": 0, "downloaded_files": ""})
            print(f"no Dynamic World image: {row['sample_id']} {row['scene']}")
            continue
        water = collection.select("water").mean().rename("water_mean")
        label = collection.select("label").mode().rename("label_mode")
        image = water.addBands(label).clip(region)
        stem = f"{row['sample_id']}_{row['scene']}_dynamic_world"
        url = image.getDownloadURL(
            {
                "name": stem,
                "region": region,
                "scale": 10,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
                "filePerBand": False,
            }
        )
        files = download_url(url, dw_dir, stem)
        rows.append(
            {
                **row.to_dict(),
                "dw_image_count": count,
                "downloaded_files": ";".join(str(path) for path in files),
            }
        )
        print(f"downloaded Dynamic World: {row['sample_id']} {row['scene']} ({count} images)")
    pd.DataFrame(rows).to_csv(dw_dir / "experiment_dynamic_world_downloads.csv", index=False)


def write_dynamic_world_manifest(experiment_root: Path, out_dir: Path, padding_deg: float) -> None:
    shapes = iter_swater_shapes(experiment_root)
    manifest_path = out_dir / "dynamic_world/experiment_dynamic_world_manifest.csv"
    manifest = build_dynamic_world_manifest(shapes, manifest_path, padding_deg)
    print(f"wrote {manifest_path} {manifest.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--padding-deg", type=float, default=0.01)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="sample local JRC/ESA rasters by Swater polygons")
    subparsers.add_parser("dw-manifest", help="write Dynamic World scene/region manifest")
    dw = subparsers.add_parser("dw-download", help="download Dynamic World via Earth Engine")
    dw.add_argument("--window-days", type=int, default=0)
    dw.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.command == "stats":
        compute_jrc_esa_stats(args.experiment_root, args.out_dir)
    elif args.command == "dw-manifest":
        write_dynamic_world_manifest(args.experiment_root, args.out_dir, args.padding_deg)
    elif args.command == "dw-download":
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        download_dynamic_world(
            args.experiment_root,
            args.out_dir,
            args.padding_deg,
            args.window_days,
            args.limit,
        )


if __name__ == "__main__":
    main()
