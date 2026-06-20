#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:52:02 2026

@author: user
"""
########################################################
from __future__ import annotations
import warnings
import re
import stat
import os
import gc
import math
import xml.etree.ElementTree as ET
import time
import queue
import shutil
import ctypes
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Union
import threading
import numpy as np
import geopandas as gpd
import fiona
import shapefile
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform, aligned_target
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform, Window
from rasterio.features import geometry_mask, rasterize
from rasterio import features
from affine import Affine                     # 修正：使用 affine.Affine
from pyproj import CRS as PyprojCRS
from shapely.geometry import shape as shapely_shape, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid
from skimage.morphology import opening, square
import cv2
from PIL import Image, ImageDraw, ImageEnhance
from osgeo import gdal, osr, ogr
import concurrent.futures
from functools import partial
from natsort import natsorted
import matplotlib.pyplot as plt

try:
    from .sentinel2_download import (
        Day_path_row_files_can_be_full_test,
        Download_Sentinel2_using_IGA,
        Find_Sentinel2_using_IGA,
        Get_path_row_of_sentine2_by_shapefile,
        extract_unique_sorted_dates,
        get_copernicus_credentials,
        get_date_list,
        get_images_by_date,
        request_download_sentinel_file,
        write_copernicus_key_file,
    )
except ImportError:
    from sentinel2_download import (
        Day_path_row_files_can_be_full_test,
        Download_Sentinel2_using_IGA,
        Find_Sentinel2_using_IGA,
        Get_path_row_of_sentine2_by_shapefile,
        extract_unique_sorted_dates,
        get_copernicus_credentials,
        get_date_list,
        get_images_by_date,
        request_download_sentinel_file,
        write_copernicus_key_file,
    )

# 设置全局 PIL 选项
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
ImageDraw.LOAD_TRUNCATED_IMAGES = True
ImageDraw.MAX_IMAGE_PIXELS = None
ImageEnhance.LOAD_TRUNCATED_IMAGES = True
ImageEnhance.MAX_IMAGE_PIXELS = None

gdal.UseExceptions()
try:
    gdal.SetCacheMax(256 * 1024 * 1024)
except Exception:
    pass
try:
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", "256")
except Exception:
    pass

PathLike = Union[str, Path]

########################################################
def Calculate_coverage_ratio(
    image_path,
    vector_path,
    cloud_mask_path=None,
    all_touched=False,
    zero_as_nodata=True
):
    """
    计算影像在矢量范围内的影像覆盖率 coverage_ratio（仅此一项）。
    参数：
        image_path      : 影像路径
        vector_path     : 矢量文件路径（面要素）
        all_touched     : 栅格化时是否使用 all_touched 模式
        zero_as_nodata  : 是否将全波段均为 0 的像元视为无效
    返回：
        coverage_ratio  : 有效影像像元面积 / 矢量真实面积（0~1）
    """
    with rasterio.open(image_path) as src:
        if src.crs is None:
            raise ValueError("影像没有 CRS / 投影信息。")
        gdf = gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("矢量文件为空。")
        if gdf.crs is None:
            raise ValueError("矢量没有 CRS / 投影信息。")
        # 将矢量重投影到影像坐标系
        gdf = gdf.to_crs(src.crs)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        if gdf.empty:
            raise ValueError("矢量重投影后没有有效几何。")
        try:
            vector_geom = gdf.geometry.union_all()
        except Exception:
            vector_geom = gdf.geometry.unary_union
        if not vector_geom.is_valid:
            vector_geom = vector_geom.buffer(0)
        if vector_geom.is_empty:
            raise ValueError("矢量几何为空或无效。")
        vector_area = vector_geom.area
        if vector_area <= 0:
            raise ValueError("矢量面积为 0，请检查是否为面要素。")
        # 影像范围
        raster_geom = box(*src.bounds)
        # 矢量与影像范围的交集
        inter_geom = vector_geom.intersection(raster_geom)
        if inter_geom.is_empty:
            return 0.0
        # 只读取矢量交集范围，避免整景影像读入内存
        raw_window = from_bounds(
            *inter_geom.bounds,
            transform=src.transform
        )
        col_off = max(0, int(np.floor(raw_window.col_off)))
        row_off = max(0, int(np.floor(raw_window.row_off)))
        col_end = min(
            src.width,
            int(np.ceil(raw_window.col_off + raw_window.width))
        )
        row_end = min(
            src.height,
            int(np.ceil(raw_window.row_off + raw_window.height))
        )
        if col_end <= col_off or row_end <= row_off:
            return 0.0
        window = Window(
            col_off,
            row_off,
            col_end - col_off,
            row_end - row_off
        )
        win_transform = src.window_transform(window)
        out_shape = (int(window.height), int(window.width))
        # 栅格化矢量范围
        vector_mask = rasterize(
            [(vector_geom, 1)],
            out_shape=out_shape,
            transform=win_transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched
        ).astype(bool)
        if vector_mask.sum() == 0:
            return 0.0
        # 读取窗口内影像
        data = src.read(window=window)
        # 判断有效像元
        valid_mask = np.all(np.isfinite(data), axis=0)
        if src.nodata is not None:
            valid_mask = valid_mask & np.all(data != src.nodata, axis=0)
        # 全波段为 0 的像元视为无效
        if zero_as_nodata:
            valid_mask = valid_mask & np.any(data != 0, axis=0)
        valid_in_vector = vector_mask & valid_mask
        valid_pixel_count = int(valid_in_vector.sum())
        if valid_pixel_count == 0:
            return 0.0
        pixel_area = abs(src.transform.a * src.transform.e)
        # 影像覆盖率
        coverage_ratio = min(
            1.0,
            (valid_pixel_count * pixel_area) / vector_area
        )
        return coverage_ratio

########################################################
def _normalize_band_name(band: str) -> str:
    """标准化波段名称"""
    band = str(band).strip().upper()
    if band in ["8A", "B8A"]:
        return "B8A"
    m = re.fullmatch(r"B?(\d{1,2})", band)
    if m:
        return f"B{int(m.group(1)):02d}"
    return band

########################################################
def _output_band_name(band: str) -> str:
    """输出波段名称"""
    if band == "B8A":
        return "B8A"
    m = re.fullmatch(r"B(\d{2})", band)
    if m:
        return f"B{int(m.group(1))}"
    return band

########################################################
def _xml_tag_name(elem) -> str:
    """获取 XML 元素的本地标签名"""
    return elem.tag.split("}")[-1]

########################################################
def _make_target_crs(epsg_code, fallback_crs):
    """创建目标 CRS"""
    if epsg_code is None:
        return fallback_crs
    epsg_text = str(epsg_code).strip().upper()
    if epsg_text in ["", "NONE", "NULL", "ORIGINAL", "KEEP", "原始"]:
        return fallback_crs
    epsg_text = epsg_text.replace("EPSG:", "")
    try:
        crs = CRS.from_epsg(int(epsg_text))
    except Exception as e:
        raise ValueError(
            f"target_epsg 输入不正确：{epsg_code}。"
            f"请使用 None、32651、32650 或 'EPSG:32651' 这种形式。"
        ) from e
    if not crs.is_projected:
        raise ValueError(
            f"target_epsg 应该是米制投影坐标系，例如 32651；"
            f"当前为：{epsg_code}"
        )
    return crs

########################################################
def _read_l1c_reflectance_metadata(safe_dir: Path, band_id_to_name: dict):
    """读取 L1C 元数据中的反射率定标参数"""
    metadata_files = list(safe_dir.glob("MTD_MSIL1C.xml"))
    if not metadata_files:
        metadata_files = list(safe_dir.rglob("MTD_MSIL1C.xml"))
    if not metadata_files:
        raise FileNotFoundError(f"没有找到 L1C 元数据文件 MTD_MSIL1C.xml：{safe_dir}")
    metadata_file = metadata_files[0]
    tree = ET.parse(metadata_file)
    root = tree.getroot()
    quantification_value = None
    radio_offsets = {b: 0.0 for b in band_id_to_name.values()}
    for elem in root.iter():
        tag = _xml_tag_name(elem)
        text = elem.text.strip() if elem.text is not None else ""
        if tag == "QUANTIFICATION_VALUE" and text:
            quantification_value = float(text)
        elif tag == "RADIO_ADD_OFFSET" and text:
            band_id = elem.attrib.get("band_id")
            band_name = band_id_to_name.get(str(band_id))
            if band_name is not None:
                radio_offsets[band_name] = float(text)
    if quantification_value is None:
        raise ValueError(f"没有在元数据中找到 QUANTIFICATION_VALUE：{metadata_file}")
    return quantification_value, radio_offsets, metadata_file

########################################################
def _get_pixel_size(path: Path) -> float:
    """获取影像的像素大小（平均值）"""
    with rasterio.open(path) as src:
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)
        return (xres + yres) / 2.0

########################################################
def _find_band_files(image_files: list, band: str):
    """查找指定波段的所有候选文件"""
    pattern = re.compile(rf"(^|_){re.escape(band)}(_|\.|$)", re.IGNORECASE)
    matched = [p for p in image_files if pattern.search(p.name)]
    if len(matched) == 0:
        raise FileNotFoundError(f"没有找到波段 {band} 对应的文件")
    return matched

########################################################
def _choose_native_band_file(image_files: list, band: str, native_resolution: dict):
    """选择最匹配原始分辨率的波段文件"""
    candidates = _find_band_files(image_files, band)
    target_res = native_resolution.get(band)
    if target_res is None:
        return sorted(candidates)[0]
    scored = []
    for p in candidates:
        try:
            pix = _get_pixel_size(p)
            score = abs(pix - target_res)
            scored.append((score, str(p), p))
        except Exception:
            pass
    if not scored:
        return sorted(candidates)[0]
    scored.sort()
    return scored[0][2]

########################################################
def _find_10m_reference_file(image_files: list):
    """查找 10 米参考波段文件"""
    preferred = ["B02", "B03", "B04", "B08"]
    for band in preferred:
        try:
            candidates = _find_band_files(image_files, band)
        except FileNotFoundError:
            continue
        scored = []
        for p in candidates:
            try:
                pix = _get_pixel_size(p)
                score = abs(pix - 10)
                scored.append((score, str(p), p))
            except Exception:
                pass
        if scored:
            scored.sort()
            if scored[0][0] < 0.1:
                return scored[0][2]
    raise FileNotFoundError("没有找到 10 m 参考波段，例如 B02/B03/B04/B08")

########################################################
def _build_output_grid(ref_file: Path, target_crs):
    """构建输出网格"""
    with rasterio.open(ref_file) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
    if ref_crs == target_crs:
        return ref_crs, ref_transform, ref_width, ref_height, False
    dst_transform, dst_width, dst_height = calculate_default_transform(
        ref_crs,
        target_crs,
        ref_width,
        ref_height,
        *ref_bounds,
        resolution=(10, 10)
    )
    dst_transform, dst_width, dst_height = aligned_target(
        dst_transform,
        dst_width,
        dst_height,
        resolution=(10, 10)
    )
    return target_crs, dst_transform, dst_width, dst_height, True

########################################################
def _pyproj_crs_from_rasterio(dst_crs):
    """从 rasterio CRS 转为 pyproj CRS"""
    try:
        return PyprojCRS.from_wkt(dst_crs.to_wkt())
    except Exception:
        return PyprojCRS.from_user_input(dst_crs)

########################################################
def _read_clip_vector_geometries(vector_input, dst_crs, clip_all_touched):
    """读取裁剪矢量几何并转换到目标 CRS"""
    dst_pyproj_crs = _pyproj_crs_from_rasterio(dst_crs)
    if vector_input is None:
        return None, None
    if isinstance(vector_input, (str, Path)):
        vector_path = Path(vector_input)
        if not vector_path.exists():
            raise FileNotFoundError(f"裁剪矢量不存在：{vector_path}")
        gdf = gpd.read_file(vector_path)
    elif hasattr(vector_input, "geometry") and hasattr(vector_input, "crs"):
        if hasattr(vector_input, "copy"):
            gdf = vector_input.copy()
        else:
            gdf = vector_input
        if not hasattr(gdf, "geometry"):
            gdf = gpd.GeoDataFrame(geometry=gdf, crs=getattr(vector_input, "crs", None))
    elif isinstance(vector_input, (list, tuple)):
        if len(vector_input) == 0:
            raise ValueError("clip_vector 不能为空列表。")
        geometry_items = [
            shapely_shape(geom) if isinstance(geom, dict) else geom
            for geom in vector_input
        ]
        gdf = gpd.GeoDataFrame(geometry=geometry_items, crs=dst_pyproj_crs)
    elif isinstance(vector_input, dict):
        gdf = gpd.GeoDataFrame(
            geometry=[shapely_shape(vector_input)],
            crs=dst_pyproj_crs
        )
    elif hasattr(vector_input, "__geo_interface__"):
        gdf = gpd.GeoDataFrame(geometry=[vector_input], crs=dst_pyproj_crs)
    else:
        raise TypeError(
            "clip_vector 只支持矢量文件路径、GeoDataFrame/GeoSeries、"
            "shapely geometry 或 geometry 列表。"
        )
    if isinstance(gdf, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=gdf, crs=gdf.crs)
    if gdf.empty:
        raise ValueError("clip_vector 为空，无法裁剪。")
    if not hasattr(gdf, "geometry"):
        raise ValueError("clip_vector 中没有 geometry 列。")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("clip_vector 没有有效几何。")
    if gdf.crs is None:
        warnings.warn(
            "clip_vector 没有 CRS，已按输出投影处理；"
            "如果矢量实际是经纬度或其他投影，请先赋予正确 CRS。"
        )
        gdf = gdf.set_crs(dst_pyproj_crs, allow_override=True)
    else:
        src_pyproj_crs = PyprojCRS.from_user_input(gdf.crs)
        if not src_pyproj_crs.equals(dst_pyproj_crs):
            gdf = gdf.to_crs(dst_pyproj_crs)
    try:
        invalid_mask = ~gdf.geometry.is_valid
        if bool(invalid_mask.any()):
            if hasattr(gdf.geometry, "make_valid"):
                gdf.loc[invalid_mask, "geometry"] = (
                    gdf.loc[invalid_mask, "geometry"].make_valid()
                )
            else:
                gdf.loc[invalid_mask, "geometry"] = (
                    gdf.loc[invalid_mask, "geometry"].buffer(0)
                )
    except Exception:
        pass
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("clip_vector 修复后没有有效几何。")
    bounds = tuple(float(v) for v in gdf.total_bounds)
    if not np.isfinite(np.array(bounds)).all():
        raise ValueError(f"clip_vector 的范围无效：{bounds}")
    geometries = [geom.__geo_interface__ for geom in gdf.geometry]
    return geometries, bounds

########################################################
def _crop_grid_to_bounds(base_transform, base_width, base_height, bounds):
    """根据边界裁剪网格"""
    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"clip_vector 的外接矩形无效：{bounds}")
    raw_window = from_bounds(minx, miny, maxx, maxy, transform=base_transform)
    col_start = math.floor(raw_window.col_off)
    row_start = math.floor(raw_window.row_off)
    col_stop = math.ceil(raw_window.col_off + raw_window.width)
    row_stop = math.ceil(raw_window.row_off + raw_window.height)
    col_start = max(0, min(int(col_start), int(base_width)))
    row_start = max(0, min(int(row_start), int(base_height)))
    col_stop = max(0, min(int(col_stop), int(base_width)))
    row_stop = max(0, min(int(row_stop), int(base_height)))
    width = col_stop - col_start
    height = row_stop - row_start
    if width <= 0 or height <= 0:
        raise ValueError("clip_vector 与 Sentinel-2 影像输出网格没有重叠范围。")
    win = Window(col_start, row_start, width, height)
    return win, window_transform(win, base_transform), int(width), int(height)

########################################################
def _append_scale_info_to_hdr(hdr_path: Path, band: str, offset: float,
                              quantification_value: float, out_crs, scale_factor: float,
                              clip_vector, clip_all_touched: bool, clip_geometry_count: int):
    """向 ENVI 头文件追加定标信息"""
    if not hdr_path.exists():
        return
    real_scale = 1.0 / float(scale_factor)
    epsg = out_crs.to_epsg()
    with open(hdr_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"s2 band = {{{band}}}\n")
        f.write("s2 unit = {scaled TOA reflectance}\n")
        f.write(f"s2 stored value = {{round(TOA_reflectance * {scale_factor})}}\n")
        f.write(f"s2 reflectance scale factor = {real_scale}\n")
        f.write(f"s2 reflectance formula = {{TOA_reflectance = stored_value / {scale_factor}}}\n")
        f.write(f"s2 quantification value = {quantification_value}\n")
        f.write(f"s2 radio add offset = {offset}\n")
        if epsg is not None:
            f.write(f"s2 target epsg = {epsg}\n")
        f.write(f"s2 clip enabled = {bool(clip_vector is not None)}\n")
        if clip_vector is not None:
            f.write(f"s2 clip all touched = {bool(clip_all_touched)}\n")
            f.write(f"s2 clip geometry count = {clip_geometry_count}\n")

########################################################
def _parse_envi_band_names(img_path: Path) -> list:
    """从 ENVI 头文件解析波段名称"""
    band_names = []
    try:
        hdr_candidates = [
            img_path.with_suffix('.hdr'),
            Path(str(img_path) + '.hdr')
        ]
        hdr_path = None
        for candidate in hdr_candidates:
            if candidate.exists():
                hdr_path = candidate
                break
        if hdr_path is None:
            return band_names
        hdr_text = hdr_path.read_text(encoding='utf-8', errors='ignore')
        match = re.search(r'band names\s*=\s*\{(.*?)\}', hdr_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return band_names
        raw_text = match.group(1).replace('\n', ',')
        band_names = [x.strip().strip(',') for x in raw_text.split(',')]
        band_names = [x for x in band_names if len(x) > 0]
    except Exception:
        band_names = []
    return band_names

########################################################
def _get_gdal_band_names(ds, img_path: str) -> list:
    """获取 GDAL 数据集的波段名称"""
    names = []
    try:
        for band_i in range(1, ds.RasterCount + 1):
            desc = ds.GetRasterBand(band_i).GetDescription()
            names.append('' if desc is None else desc.strip())
    except Exception:
        names = []
    if not any(names):
        names = _parse_envi_band_names(Path(img_path))
    return names

########################################################
def _find_band_index(ds, band_names: list, candidate_names: list,
                     fallback_index=None, required=True) -> int:
    """查找波段索引（1-based）"""
    lower_names = [str(x).strip().lower() for x in band_names]
    for cname in candidate_names:
        cname_lower = str(cname).strip().lower()
        if cname_lower in lower_names:
            return lower_names.index(cname_lower) + 1
    if fallback_index is not None and fallback_index <= ds.RasterCount:
        return int(fallback_index)
    if required:
        raise ValueError(
            f"找不到必要波段：{candidate_names}，当前波段名为：{band_names}，总波段数：{ds.RasterCount}"
        )
    return None

########################################################
def _read_reflectance(ds, band_index: int) -> np.ndarray:
    """读取反射率波段并缩放到 0-1 范围"""
    arr = ds.GetRasterBand(int(band_index)).ReadAsArray(buf_type=gdal.GDT_Float32)
    arr = arr.astype(np.float32, copy=False)
    arr *= np.float32(1.0 / 10000.0)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

########################################################
def _write_single_band_tif(output_path: str, array: np.ndarray, projection, geotransform, gdal_type):
    """写入单波段 GeoTIFF"""
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    options = [
        "COMPRESS=LZW",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
        "NUM_THREADS=ALL_CPUS",
        "SPARSE_OK=YES",
    ]
    if gdal_type in (gdal.GDT_Float32, gdal.GDT_Float64):
        options.append("PREDICTOR=3")
    else:
        options.append("PREDICTOR=2")
    out_ds = gtiff_driver.Create(
        output_path,
        int(array.shape[1]),
        int(array.shape[0]),
        1,
        gdal_type,
        options=options
    )
    if out_ds is None:
        raise RuntimeError(f"无法创建输出 tif：{output_path}")
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    out_band = None
    out_ds = None

########################################################
def _write_envi_byte(output_path: str, array: np.ndarray, projection, geotransform):
    """将 byte 数组写入 ENVI .img 文件，并自动生成 .hdr，不标记任何 nodata"""
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # 使用 affine.Affine.from_gdal 转换几何变换
    transform = Affine.from_gdal(*geotransform)
    profile = {
        'driver': 'ENVI',
        'width': int(array.shape[1]),
        'height': int(array.shape[0]),
        'count': 1,
        'dtype': 'uint8',
        'crs': projection,
        'transform': transform,
        # 不设置 nodata，0 值将作为普通有效像素写入
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array, 1)

########################################################
def _get_band_names_from_src(src):
    """从 rasterio 数据集获取波段名称"""
    band_descriptions = list(src.descriptions)
    if not any(band_descriptions):
        envi_tags = src.tags(ns="ENVI")
        band_names_text = envi_tags.get("band names")
        if band_names_text:
            band_descriptions = [
                x.strip()
                for x in band_names_text.replace("{", "").replace("}", "").split(",")
            ]
    return band_descriptions

########################################################
def _get_band_indices(band_names: list, target_names: list) -> list:
    """获取目标波段的索引（1-based）"""
    indices = []
    for name in target_names:
        if name not in band_names:
            raise ValueError(f"找不到波段：{name}\n当前波段名为：{band_names}")
        indices.append(band_names.index(name) + 1)
    return indices

########################################################
def _stretch_to_rgb_uint8(image: np.ndarray, percentile=(2, 98), background_color=(1, 1, 1)) -> np.ndarray:
    """将多波段影像拉伸为 RGB uint8"""
    h, w = image.shape[1], image.shape[2]
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    bg = np.array(
        [int(max(0.0, min(1.0, float(c))) * 255) for c in background_color],
        dtype=np.uint8,
    )
    valid_mask = np.any(image > 0, axis=0)
    if not valid_mask.any():
        rgb[:, :, :] = bg
        del valid_mask
        return rgb
    invalid_mask = ~valid_mask
    for i in range(3):
        band = image[i]
        vals = band[valid_mask]
        p_low, p_high = np.percentile(vals, percentile)
        del vals
        if p_high == p_low:
            out_band = np.zeros((h, w), dtype=np.uint8)
        else:
            tmp = np.empty((h, w), dtype=np.float32)
            np.subtract(band, p_low, out=tmp)
            tmp *= np.float32(255.0 / (p_high - p_low))
            np.clip(tmp, 0, 255, out=tmp)
            out_band = tmp.astype(np.uint8)
            del tmp
        out_band[invalid_mask] = bg[i]
        rgb[:, :, i] = out_band
        del out_band
    del valid_mask, invalid_mask
    return rgb

########################################################
def _trim_memory():
    """尽量归还 Python / GDAL / libc 持有的临时内存。"""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        # 限制 GDAL 全局缓存，避免长循环中缓存持续膨胀
        gdal.SetCacheMax(256 * 1024 * 1024)
    except Exception:
        pass
    try:
        # 释放 GDAL VSI / curl 缓存；某些版本没有该函数，忽略即可
        gdal.VSICurlClearCache()
    except Exception:
        pass
    try:
        # Linux 下把 glibc 分配器中可释放的堆内存交还给系统
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

########################################################
def _scale_to_uint8_inplace(arr, percentile=(0, 99), divisor=1.0):
    arr = arr.astype(np.float32, copy=False)
    if divisor not in (1, 1.0):
        arr /= np.float32(divisor)
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        arr[~finite_mask] = 0
    del finite_mask
    p_low, p_high = np.percentile(arr, percentile)
    if p_high == p_low:
        out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        np.subtract(arr, p_low, out=arr)
        arr *= np.float32(255.0 / (p_high - p_low))
        np.clip(arr, 0, 255, out=arr)
        out = arr.astype(np.uint8)
    del arr
    return out

########################################################
def _process_single_component(geom, lake_seed, prepared_lake_seed, min_intersection_area, min_intersection_ratio):
    if not prepared_lake_seed.intersects(geom):
        return False, 0.0, 0.0
    inter = geom.intersection(lake_seed)
    inter_area = float(inter.area) if inter is not None and not inter.is_empty else 0.0
    geom_area = float(geom.area) if geom.area else 0.0
    inter_ratio = inter_area / geom_area if geom_area > 0 else 0.0
    keep = (
        inter_area >= float(min_intersection_area)
        and inter_ratio >= float(min_intersection_ratio)
    )
    return keep, inter_area, inter_ratio

########################################################
def _copy_shp_dataset(src_shp: PathLike, dst_shp: PathLike) -> None:
    src_shp = Path(src_shp)
    dst_shp = Path(dst_shp)
    dst_shp.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [
        ".shp", ".shx", ".dbf", ".prj", ".cpg",
        ".qpj", ".sbn", ".sbx", ".shp.xml"
    ]
    copied = 0
    for suffix in suffixes:
        src_file = src_shp.with_suffix(suffix)
        dst_file = dst_shp.with_suffix(suffix)
        if src_file.exists():
            try:
                if src_file.resolve() == dst_file.resolve():
                    copied += 1
                    continue
            except Exception:
                pass
            shutil.copy2(src_file, dst_file)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"没有找到可复制的 shapefile 文件：{src_shp}")

########################################################
def _repair_geometry(geom: BaseGeometry) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    if make_valid is not None:
        try:
            fixed = make_valid(geom)
            if fixed is not None and not fixed.is_empty:
                return fixed
        except Exception:
            pass
    try:
        fixed = geom.buffer(0)
        if fixed is not None and not fixed.is_empty:
            return fixed
    except Exception:
        pass
    return None

########################################################
def _clean_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(_repair_geometry)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf

########################################################
def _union_all(geoms: Iterable[BaseGeometry]) -> BaseGeometry:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("没有可用于合并的 geometry。")
    return unary_union(geoms)

########################################################
def _polygon_parts(geom: BaseGeometry) -> List[BaseGeometry]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts: List[BaseGeometry] = []
        for sub_geom in geom.geoms:
            parts.extend(_polygon_parts(sub_geom))
        return parts
    return []

########################################################
def _copy_or_link_file(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.resolve() == dst_path.resolve():
            return
    except Exception:
        pass
    try:
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        os.link(src_path, dst_path)
    except Exception:
        shutil.copy2(src_path, dst_path)

########################################################
def _remove_shapefile_dataset(shp_path):
    shp_path = Path(shp_path)
    for suffix in (
        ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj",
        ".sbn", ".sbx", ".shp.xml", ".fix"
    ):
        p = shp_path.with_suffix(suffix)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

########################################################
def _find_band_index_general(band_names, candidates):
    norm_names = []
    for name in band_names:
        if name is None:
            norm_names.append("")
        else:
            norm_names.append(str(name).lower())
    for candidate in candidates:
        candidate = str(candidate).lower()
        for i, name in enumerate(norm_names):
            if candidate == name or candidate in name:
                return i
    return None

########################################################
def _scale_reflectance(data):
    data = data.astype("float32", copy=False)
    valid_values = data[np.isfinite(data)]
    if valid_values.size > 0:
        if np.nanmax(valid_values) > 2.0:
            data = data / 10000.0
    return data

########################################################
def _simple_s2_cloud_mask(data, band_names):
    data = _scale_reflectance(data)
    i_blue = _find_band_index_general(band_names, ["rhot_492", "492", "b02", "blue"])
    i_green = _find_band_index_general(band_names, ["rhot_560", "560", "b03", "green"])
    i_red = _find_band_index_general(band_names, ["rhot_665", "665", "b04", "red"])
    i_nir = _find_band_index_general(band_names, ["rhot_833", "833", "rhot_842", "842", "b08", "nir"])
    i_swir1 = _find_band_index_general(band_names, ["rhot_1614", "1614", "rhot_1600", "b11", "swir1"])
    if any(i is None for i in [i_blue, i_green, i_red, i_nir, i_swir1]):
        raise ValueError("无法自动找到云检测所需波段：Blue / Green / Red / NIR / SWIR1。")
    blue = data[i_blue]
    green = data[i_green]
    red = data[i_red]
    nir = data[i_nir]
    swir1 = data[i_swir1]
    vis_mean = (blue + green + red) / 3.0
    ndsi = (green - swir1) / (green + swir1 + 1e-6)
    cloud_mask = (
        (vis_mean > 0.20) &
        (blue > 0.18) &
        (green > 0.18) &
        (red > 0.16) &
        (nir > 0.18) &
        (swir1 > 0.10) &
        (ndsi < 0.80)
    )
    return cloud_mask

########################################################
def _read_png_and_valid_mask(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    if len(img.shape) == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        valid_mask = gray > 0
    elif img.shape[2] == 4:
        bgr = img[:, :, 0:3]
        alpha = img[:, :, 3]
        valid_mask = alpha > 10
    else:
        bgr = img[:, :, 0:3]
        valid_mask = np.any(bgr > 0, axis=2)
    valid_mask_uint8 = valid_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    valid_mask_clean = cv2.morphologyEx(valid_mask_uint8, cv2.MORPH_OPEN, kernel).astype(bool)
    if np.count_nonzero(valid_mask_clean) > 0:
        valid_mask = valid_mask_clean
    return bgr, valid_mask

########################################################
def Export_s2_l1c_toa10000_int16_10m_envi_projected(
    safe_dir,
    bands,
    out_dir,
    target_epsg=None,
    resampling="bilinear",
    raw_nodata=0,
    output_nodata=0,
    scale_factor=10000,
    envi_suffix=".img",
    clip_reflectance=False,
    overwrite=True,
    verbose=True,
    clip_vector=None,
    clip_all_touched=False
):
    """
    将 Sentinel-2 L1C SAFE 数据导出为 10 m、统一投影的 ENVI .img 文件。
    """
    safe_dir = Path(safe_dir)
    out_dir = Path(out_dir)
    if not safe_dir.exists():
        raise FileNotFoundError(f"SAFE 文件夹不存在：{safe_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not envi_suffix.startswith("."):
        envi_suffix = "." + envi_suffix
    int16_info = np.iinfo(np.int16)
    if not (int16_info.min <= int(output_nodata) <= int16_info.max):
        raise ValueError(f"output_nodata 必须在 int16 范围内：{int16_info.min} ~ {int16_info.max}")
    if scale_factor <= 0:
        raise ValueError("scale_factor 必须大于 0")
    native_resolution = {
        "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20,
        "B06": 20, "B07": 20, "B08": 10, "B8A": 20, "B09": 60,
        "B10": 60, "B11": 20, "B12": 20,
    }
    band_id_to_name = {
        "0": "B01", "1": "B02", "2": "B03", "3": "B04", "4": "B05",
        "5": "B06", "6": "B07", "7": "B08", "8": "B8A", "9": "B09",
        "10": "B10", "11": "B11", "12": "B12",
    }
    try:
        default_resampling = getattr(Resampling, resampling)
    except AttributeError:
        raise ValueError(f"不支持的重采样方法：{resampling}")
    bands = [_normalize_band_name(b) for b in bands]
    for band in bands:
        if band not in native_resolution:
            raise ValueError(f"{band} 不是 Sentinel-2 L1C 光谱波段，不能转换为 TOA 反射率")
    quantification_value, radio_offsets, metadata_file = _read_l1c_reflectance_metadata(safe_dir, band_id_to_name)
    image_exts = {".jp2", ".tif", ".tiff"}
    image_files = [
        p for p in safe_dir.rglob("*")
        if p.suffix.lower() in image_exts and any(part.upper() == "IMG_DATA" for part in p.parts)
    ]
    if len(image_files) == 0:
        raise FileNotFoundError(f"在 SAFE 文件夹的 IMG_DATA 中没有找到 JP2/TIF 文件：{safe_dir}")
    band_files = {}
    for band in bands:
        band_files[band] = _choose_native_band_file(image_files, band, native_resolution)
    ref_file = _find_10m_reference_file(image_files)
    with rasterio.open(ref_file) as ref:
        ref_crs = ref.crs
    target_crs = _make_target_crs(target_epsg, ref_crs)
    out_crs, full_out_transform, full_out_width, full_out_height, did_reproject_crs = _build_output_grid(ref_file, target_crs)
    out_transform = full_out_transform
    out_width = full_out_width
    out_height = full_out_height
    output_window = None
    clip_mask = None
    clip_geometry_count = 0
    if clip_vector is not None:
        clip_geometries, clip_bounds = _read_clip_vector_geometries(clip_vector, out_crs, clip_all_touched)
        clip_geometry_count = len(clip_geometries)
        output_window, out_transform, out_width, out_height = _crop_grid_to_bounds(
            full_out_transform, full_out_width, full_out_height, clip_bounds
        )
        clip_mask = geometry_mask(
            clip_geometries,
            out_shape=(out_height, out_width),
            transform=out_transform,
            invert=True,
            all_touched=bool(clip_all_touched)
        )
        if not bool(clip_mask.any()):
            raise ValueError("clip_vector 与输出 10 m 网格没有重叠像元；如果矢量很窄，可以尝试 clip_all_touched=True。")
    output_paths = {}
    min_allowed = int16_info.min
    max_allowed = int16_info.max
    for band in bands:
        src_path = band_files[band]
        out_path = out_dir / (_output_band_name(band) + envi_suffix)
        hdr_path = out_path.with_suffix(".hdr")
        aux_path = Path(str(out_path) + ".aux.xml")
        if not overwrite:
            if out_path.exists():
                raise FileExistsError(f"输出文件已存在：{out_path}")
            if hdr_path.exists():
                raise FileExistsError(f"ENVI 头文件已存在：{hdr_path}")
        if overwrite:
            for p in [out_path, hdr_path, aux_path]:
                if p.exists():
                    p.unlink()

        offset = radio_offsets.get(band, 0.0)
        with rasterio.open(src_path) as src:
            profile = {
                "driver": "ENVI",
                "width": out_width,
                "height": out_height,
                "count": 1,
                "crs": out_crs,
                "transform": out_transform,
                "dtype": "int16",
                "nodata": int(output_nodata),
                "interleave": "bsq",
            }
            same_full_grid = (
                src.crs == out_crs
                and src.width == full_out_width
                and src.height == full_out_height
                and src.transform.almost_equals(full_out_transform)
            )
            if same_full_grid:
                if output_window is None:
                    dn_data = src.read(1, out_dtype="float32")
                else:
                    dn_data = src.read(1, window=output_window, out_dtype="float32")
            else:
                fill_value = raw_nodata if raw_nodata is not None else 0
                dn_data = np.full(shape=(out_height, out_width), fill_value=fill_value, dtype="float32")
                src_nodata = src.nodata
                if src_nodata is None:
                    src_nodata = raw_nodata
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dn_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=out_transform,
                    dst_crs=out_crs,
                    dst_nodata=raw_nodata,
                    resampling=default_resampling
                )
            if raw_nodata is None:
                valid_mask = np.ones(dn_data.shape, dtype=bool)
            else:
                valid_mask = dn_data != raw_nodata
            if clip_mask is not None:
                valid_mask &= clip_mask
            out_data = np.full(dn_data.shape, int(output_nodata), dtype="int16")
            if bool(valid_mask.any()):
                if bool(valid_mask.all()):
                    dn_data += float(offset)
                    dn_data /= float(quantification_value)
                    if clip_reflectance:
                        np.clip(dn_data, 0.0, 1.0, out=dn_data)
                    dn_data *= float(scale_factor)
                    np.rint(dn_data, out=dn_data)
                    np.clip(dn_data, min_allowed, max_allowed, out=dn_data)
                    out_data = dn_data.astype("int16")
                else:
                    valid_values = dn_data[valid_mask].astype("float32", copy=True)
                    valid_values += float(offset)
                    valid_values /= float(quantification_value)
                    if clip_reflectance:
                        np.clip(valid_values, 0.0, 1.0, out=valid_values)
                    valid_values *= float(scale_factor)
                    np.rint(valid_values, out=valid_values)
                    np.clip(valid_values, min_allowed, max_allowed, out=valid_values)
                    out_data[valid_mask] = valid_values.astype("int16")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_data, 1)
                dst.set_band_description(1, band)
                dst.update_tags(
                    1,
                    band=band,
                    unit="TOA_reflectance_scaled",
                    stored_value=f"round(TOA_reflectance * {scale_factor})",
                    reflectance_formula=f"TOA_reflectance = stored_value / {scale_factor}",
                    reflectance_scale_factor=str(1.0 / float(scale_factor)),
                    source_file=src_path.name,
                    quantification_value=str(quantification_value),
                    radio_add_offset=str(offset),
                    native_resolution=str(native_resolution.get(band, "unknown")),
                    output_resolution="10m",
                    output_dtype="int16",
                    output_nodata=str(output_nodata),
                    target_epsg=str(out_crs.to_epsg()),
                    clip_enabled=str(clip_vector is not None),
                    clip_all_touched=str(bool(clip_all_touched)),
                    clip_geometry_count=str(clip_geometry_count),
                    format="ENVI")
        _append_scale_info_to_hdr(
            hdr_path=hdr_path,
            band=band,
            offset=offset,
            quantification_value=quantification_value,
            out_crs=out_crs,
            scale_factor=scale_factor,
            clip_vector=clip_vector,
            clip_all_touched=clip_all_touched,
            clip_geometry_count=clip_geometry_count
        )
        output_paths[band] = out_path
    return output_paths

########################################################
def select_target_lake_by_original_shp(
    water_shp: PathLike,
    lake_shp: PathLike,
    out_shp: PathLike,
    *,
    min_intersection_area: float = 0.0,
    min_intersection_ratio: float = 0.0,
    dissolve_output: bool = True,
) -> gpd.GeoDataFrame:
    water_shp = Path(water_shp)
    lake_shp = Path(lake_shp)
    out_shp = Path(out_shp)
    water = gpd.read_file(water_shp)
    water_raw = water.copy()
    lake = gpd.read_file(lake_shp)
    if water.empty or lake.empty:
        raise ValueError("输入矢量文件不能为空。")
    water_crs = water.crs
    if water_crs is None and lake.crs is not None:
        water = water.set_crs(lake.crs, allow_override=True)
        water_raw = water_raw.set_crs(lake.crs, allow_override=True)
        water_crs = lake.crs
    if lake.crs is None and water_crs is not None:
        lake = lake.set_crs(water_crs, allow_override=True)
    elif water_crs is not None and lake.crs != water_crs:
        lake = lake.to_crs(water_crs)
    water = _clean_gdf(water)
    lake = _clean_gdf(lake)
    if water.empty or lake.empty:
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        fallback = water_raw.copy()
        if water_crs is not None:
            fallback = fallback.set_crs(water_crs, allow_override=True)
        fallback.to_file(out_shp, encoding="utf-8")
        return fallback
    lake_seed = _union_all(lake.geometry)
    water_union = _union_all(water.geometry)
    water_components = _polygon_parts(water_union)
    if len(water_components) == 0:
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        fallback = water_raw.copy()
        if water_crs is not None:
            fallback = fallback.set_crs(water_crs, allow_override=True)
        fallback.to_file(out_shp, encoding="utf-8")
        return fallback
    comp_gdf = gpd.GeoDataFrame(
        {"cmp_id": np.arange(1, len(water_components) + 1, dtype=np.int32)},
        geometry=water_components,
        crs=water_crs,
    )
    try:
        candidate_idx = comp_gdf.sindex.query(lake_seed, predicate="intersects")
    except TypeError:
        candidate_idx = list(comp_gdf.sindex.intersection(lake_seed.bounds))
    except Exception:
        prepared_lake_seed = prep(lake_seed)
        candidate_idx = [i for i, geom in enumerate(comp_gdf.geometry) if prepared_lake_seed.intersects(geom)]
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        selected = comp_gdf.iloc[[]].copy()
    else:
        candidate_idx = np.unique(candidate_idx)
        candidates = comp_gdf.iloc[candidate_idx].copy()
        try:
            intersects_mask = candidates.geometry.intersects(lake_seed).to_numpy()
            candidates = candidates.iloc[intersects_mask].copy()
            candidate_idx = candidate_idx[intersects_mask]
        except Exception:
            pass
        if candidates.empty:
            selected = comp_gdf.iloc[[]].copy()
        else:
            inter_geoms = candidates.geometry.intersection(lake_seed)
            overlap_areas = inter_geoms.area.to_numpy(dtype="float64", copy=False)
            geom_areas = candidates.geometry.area.to_numpy(dtype="float64", copy=False)
            overlap_ratios = np.divide(
                overlap_areas,
                geom_areas,
                out=np.zeros_like(overlap_areas, dtype="float64"),
                where=geom_areas > 0,
            )
            keep_flags = (
                (overlap_areas >= float(min_intersection_area))
                & (overlap_ratios >= float(min_intersection_ratio))
            )
            comp_gdf["ov_area"] = 0.0
            comp_gdf["ov_ratio"] = 0.0
            comp_gdf.loc[candidates.index, "ov_area"] = overlap_areas
            comp_gdf.loc[candidates.index, "ov_ratio"] = overlap_ratios
            comp_gdf["area"] = comp_gdf.geometry.area
            selected = candidates.iloc[keep_flags].copy()
            selected["ov_area"] = overlap_areas[keep_flags]
            selected["ov_ratio"] = overlap_ratios[keep_flags]
            selected["area"] = geom_areas[keep_flags]
    if selected.empty:
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        fallback = water_raw.copy()
        if water_crs is not None:
            fallback = fallback.set_crs(water_crs, allow_override=True)
        fallback.to_file(out_shp, encoding="utf-8")
        return fallback
    if dissolve_output:
        selected_union = _union_all(selected.geometry)
        selected_parts = _polygon_parts(selected_union)
        selected = gpd.GeoDataFrame(
            {
                "lake_id": list(range(1, len(selected_parts) + 1)),
                "area": [float(g.area) for g in selected_parts],
            },
            geometry=selected_parts,
            crs=water_crs,
        )
    if water_crs is not None:
        selected = selected.set_crs(water_crs, allow_override=True)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    selected.to_file(out_shp, encoding="utf-8")
    return selected

########################################################
def delete_folder_contents(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return
    if not folder.is_dir():
        return
    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, onerror=handle_remove_readonly)
        except Exception:
            pass

########################################################
def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

########################################################
def Make_three_panel_overlay(
    raster_path,
    vector1_path,
    vector2_path,
    output_path,
    rgb_band_names=("rhot_833", "rhot_665", "rhot_560"),
    vector1_color="yellow",
    vector2_color="yellow",
    boundary_width=2,
    crop_to_vectors=True,
    buffer=1500,
    percentile=(2, 98),
    dpi=500,
    figsize=(15, 5),
    background_color=(1, 1, 1)
):
    with rasterio.open(raster_path) as src:
        if src.count <= 3 or src.width <= 3 or src.height <= 3:
            raise ValueError("影像的波段数、行数或列数小于3，无法进行处理。")
        band_names = _get_band_names_from_src(src)
        band_indices = _get_band_indices(band_names, rgb_band_names)
        fig = None
        try:
            gdf1 = gpd.read_file(vector1_path)
            gdf2 = gpd.read_file(vector2_path)
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                if raster_crs is None:
                    raise ValueError("影像没有坐标系信息。")
                if gdf1.crs is None:
                    raise ValueError("矢量1没有坐标系信息。")
                if gdf2.crs is None:
                    raise ValueError("矢量2没有坐标系信息。")
                if gdf1.crs != raster_crs:
                    gdf1 = gdf1.to_crs(raster_crs)
                if gdf2.crs != raster_crs:
                    gdf2 = gdf2.to_crs(raster_crs)
                band_names = _get_band_names_from_src(src)
                band_indices = _get_band_indices(band_names, rgb_band_names)
                if crop_to_vectors:
                    minx1, miny1, maxx1, maxy1 = gdf1.total_bounds
                    minx2, miny2, maxx2, maxy2 = gdf2.total_bounds
                    left = max(src.bounds.left, min(minx1, minx2) - buffer)
                    bottom = max(src.bounds.bottom, min(miny1, miny2) - buffer)
                    right = min(src.bounds.right, max(maxx1, maxx2) + buffer)
                    top = min(src.bounds.top, max(maxy1, maxy2) + buffer)
                    window = from_bounds(left, bottom, right, top, src.transform)
                    window = window.round_offsets().round_lengths()
                    image = src.read(band_indices, window=window, out_dtype="float32")
                    transform = src.window_transform(window)
                else:
                    image = src.read(band_indices, out_dtype="float32")
                    transform = src.transform
            rgb = _stretch_to_rgb_uint8(image, percentile=percentile, background_color=background_color)
            del image
            _trim_memory()
            img_left = transform.c
            img_top = transform.f
            img_right = img_left + transform.a * rgb.shape[1]
            img_bottom = img_top + transform.e * rgb.shape[0]
            extent = [img_left, img_right, img_bottom, img_top]
            fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(background_color)
            for ax in axes:
                ax.set_axis_off()
                ax.set_facecolor(background_color)
            axes[0].imshow(rgb, extent=extent, origin="upper")
            axes[1].imshow(rgb, extent=extent, origin="upper")
            gdf1.boundary.plot(ax=axes[1], edgecolor=vector1_color, linewidth=boundary_width)
            axes[2].imshow(rgb, extent=extent, origin="upper")
            gdf2.boundary.plot(ax=axes[2], edgecolor=vector2_color, linewidth=boundary_width)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
        finally:
            if fig is not None:
                plt.close(fig)
            try:
                del rgb
            except Exception:
                pass
            try:
                del gdf1
            except Exception:
                pass
            try:
                del gdf2
            except Exception:
                pass
            try:
                del axes
            except Exception:
                pass
            _trim_memory()

########################################################
def Water_create_PNG_file_of_Sentinel2(Image_path, Water_img_dir, Out_water_PNG_dir):
    Image_dataset = None
    Water_dataset = None
    try:
        Image_dataset = gdal.Open(Image_path, gdal.GA_ReadOnly)
        if Image_dataset is None:
            raise FileNotFoundError(f"无法打开影像：{Image_path}")
        R_raw = Image_dataset.GetRasterBand(8).ReadAsArray().astype(np.float32, copy=False)
        if np.nanmax(R_raw) <= 0:
            del R_raw
            return
        R = _scale_to_uint8_inplace(R_raw, percentile=(1, 99))
        G = _scale_to_uint8_inplace(
            Image_dataset.GetRasterBand(4).ReadAsArray().astype(np.float32, copy=False),
            percentile=(1, 99),
        )
        B = _scale_to_uint8_inplace(
            Image_dataset.GetRasterBand(3).ReadAsArray().astype(np.float32, copy=False),
            percentile=(1, 99),
        )
        h, w = R.shape
        rgb_array = np.empty((h, w * 2, 3), dtype=np.uint8)
        rgb_array[:, :w, 0] = R
        rgb_array[:, :w, 1] = G
        rgb_array[:, :w, 2] = B
        rgb_array[:, w:, 0] = R
        rgb_array[:, w:, 1] = G
        rgb_array[:, w:, 2] = B
        del R, G, B
        _trim_memory()
        Water_dataset = gdal.Open(Water_img_dir, gdal.GA_ReadOnly)
        if Water_dataset is None:
            raise FileNotFoundError(f"无法打开水体掩膜：{Water_img_dir}")
        Water_array = Water_dataset.GetRasterBand(1).ReadAsArray()
        Water_mask = Water_array == 1
        if Water_mask.any():
            right_panel = rgb_array[:, w:, :]
            right_panel[Water_mask] = (255, 215, 0)
            del right_panel
        del Water_array, Water_mask
        RGB_image = Image.fromarray(rgb_array, mode="RGB")
        RGB_image.save(Out_water_PNG_dir, dpi=(500, 500))
        RGB_image.close()
        del RGB_image, rgb_array
    finally:
        Water_dataset = None
        Image_dataset = None
        _trim_memory()

########################################################
def shp_to_TEM_tiff(shp_file, refore_tif, output_tiff):
    data_source = None
    img = None
    out_ds = None
    out_band = None
    try:
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.Open(shp_file, 0)
        if data_source is None:
            raise FileNotFoundError(f"无法打开 shp：{shp_file}")
        shp_layer = data_source.GetLayer()
        img = gdal.Open(refore_tif, gdal.GA_ReadOnly)
        if img is None:
            raise FileNotFoundError(f"无法打开参考影像：{refore_tif}")
        projection = img.GetProjection()
        transform = img.GetGeoTransform()
        cols = img.RasterXSize
        rows = img.RasterYSize
        gtiff_driver = gdal.GetDriverByName("GTiff")
        out_ds = gtiff_driver.Create(
            output_tiff,
            cols,
            rows,
            1,
            gdal.GDT_Byte,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        out_ds.SetGeoTransform(transform)
        out_ds.SetProjection(projection)
        out_band = out_ds.GetRasterBand(1)
        out_band.Fill(0)
        gdal.RasterizeLayer(out_ds, [1], shp_layer, burn_values=[1])
        out_band.FlushCache()
        mask_array = out_band.ReadAsArray().astype(np.uint8, copy=False)
        ref_array = img.GetRasterBand(10).ReadAsArray()
        mask_array[(ref_array <= 0) | (ref_array >= 10000)] = 0
        out_band.WriteArray(mask_array)
        out_band.FlushCache()
        del mask_array, ref_array
    finally:
        out_band = None
        out_ds = None
        img = None
        data_source = None
        _trim_memory()

########################################################
def prepare_raster_for_arcmap(
    raster_path,
    resampling="average",
    levels=(2, 4, 8, 16, 32, 64),
    histogram_bins=256,
    approx_ok=True,
    use_rrd=False
):
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"文件不存在：{raster_path}")
    if not raster_path.is_file():
        raise ValueError(f"输入的不是文件路径：{raster_path}")
    old_config = {
        "GDAL_PAM_ENABLED": gdal.GetConfigOption("GDAL_PAM_ENABLED"),
        "GDAL_NUM_THREADS": gdal.GetConfigOption("GDAL_NUM_THREADS"),
        "USE_RRD": gdal.GetConfigOption("USE_RRD"),
        "COMPRESS_OVERVIEW": gdal.GetConfigOption("COMPRESS_OVERVIEW"),
        "BIGTIFF_OVERVIEW": gdal.GetConfigOption("BIGTIFF_OVERVIEW"),
    }
    result = {
        "file": str(raster_path),
        "driver": None,
        "bands": 0,
        "pyramid": False,
        "statistics_histogram": False,
        "success": False,
        "message": ""
    }
    ds = None
    try:
        gdal.SetConfigOption("GDAL_PAM_ENABLED", "YES")
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
        if use_rrd:
            gdal.SetConfigOption("USE_RRD", "YES")
        else:
            gdal.SetConfigOption("USE_RRD", "NO")
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            gdal.SetConfigOption("BIGTIFF_OVERVIEW", "IF_SAFER")
        ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL 无法打开该影像：{raster_path}")
        result["driver"] = ds.GetDriver().ShortName
        result["bands"] = ds.RasterCount
        if ds.RasterCount < 1:
            raise RuntimeError("该影像没有波段。")
        err = ds.BuildOverviews(resampling.upper(), list(levels))
        if err not in (0, None):
            raise RuntimeError(f"生成金字塔失败，错误码：{err}")
        result["pyramid"] = True
        for band_index in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_index)
            stats = band.ComputeStatistics(bool(approx_ok))
            if stats is None or len(stats) < 4:
                raise RuntimeError(f"Band {band_index} 统计值计算失败。")
            min_value = float(stats[0])
            max_value = float(stats[1])
            mean_value = float(stats[2])
            std_value = float(stats[3])
            band.SetStatistics(min_value, max_value, mean_value, std_value)
            hist_min = min_value
            hist_max = max_value
            if hist_min == hist_max:
                hist_min = hist_min - 0.5
                hist_max = hist_max + 0.5
            hist = band.GetHistogram(hist_min, hist_max, int(histogram_bins), 1, int(bool(approx_ok)))
            band.SetDefaultHistogram(hist_min, hist_max, hist)
        ds.FlushCache()
        ds = None
        result["statistics_histogram"] = True
        result["success"] = True
        result["message"] = "完成"
        return result
    except Exception as e:
        result["success"] = False
        result["message"] = str(e)
        return result
    finally:
        ds = None
        for key, value in old_config.items():
            gdal.SetConfigOption(key, value)

########################################################
def calculate_image_completeness(image_path):
    bgr = None
    valid_mask = None
    try:
        bgr, valid_mask = _read_png_and_valid_mask(image_path)
        total_pixels = valid_mask.shape[0] * valid_mask.shape[1]
        valid_pixels = np.count_nonzero(valid_mask)
        if total_pixels == 0:
            return 0.0
        return float(valid_pixels / total_pixels)
    finally:
        try:
            del bgr, valid_mask
        except Exception:
            pass
        _trim_memory()

########################################################
def calculate_sharpness(image_path):
    bgr, valid_mask = _read_png_and_valid_mask(image_path)
    valid_pixels = np.count_nonzero(valid_mask)
    if valid_pixels < 100:
        return 0.0
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_for_calc = img.copy()
    fill_value = np.median(img[valid_mask])
    img_for_calc[~valid_mask] = fill_value
    img_blurred = cv2.GaussianBlur(img_for_calc, (5, 5), 0)
    laplacian = cv2.Laplacian(img_blurred, cv2.CV_64F)
    kernel = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if np.count_nonzero(inner_mask) < 100:
        inner_mask = valid_mask
    variance = laplacian[inner_mask].var()
    return float(variance)

########################################################
def calculate_contrast(image_path):
    bgr, valid_mask = _read_png_and_valid_mask(image_path)
    valid_pixels = np.count_nonzero(valid_mask)
    if valid_pixels < 100:
        return 0.0
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrast = img[valid_mask].std()
    return float(contrast)

########################################################
def calculate_snow_coverage(image_path):
    bgr, valid_mask = _read_png_and_valid_mask(image_path)
    valid_pixels = np.count_nonzero(valid_mask)
    if valid_pixels < 100:
        return 1.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    snow_mask = cv2.inRange(hsv, lower_white, upper_white) > 0
    snow_pixels = np.count_nonzero(snow_mask & valid_mask)
    return float(snow_pixels / valid_pixels)

########################################################
def calculate_image_quality_metrics(image_path):
    """一次性计算 PNG 的完整度、清晰度、对比度和雪覆盖率，并在结束时释放数组。"""
    bgr = valid_mask = gray = img_for_calc = img_blurred = laplacian = None
    hsv = snow_mask = inner_mask = valid_mask_u8 = None
    try:
        bgr, valid_mask = _read_png_and_valid_mask(image_path)
        total_pixels = valid_mask.shape[0] * valid_mask.shape[1]
        valid_pixels = np.count_nonzero(valid_mask)
        completeness = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0
        if valid_pixels < 100:
            return completeness, 0.0, 0.0, 1.0

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        valid_gray = gray[valid_mask]
        contrast = float(valid_gray.std())
        fill_value = np.median(valid_gray)
        del valid_gray

        img_for_calc = gray.copy()
        img_for_calc[~valid_mask] = fill_value
        img_blurred = cv2.GaussianBlur(img_for_calc, (5, 5), 0)
        laplacian = cv2.Laplacian(img_blurred, cv2.CV_64F)
        kernel = np.ones((5, 5), np.uint8)
        valid_mask_u8 = valid_mask.astype(np.uint8)
        inner_mask = cv2.erode(valid_mask_u8, kernel, iterations=1).astype(bool)
        if np.count_nonzero(inner_mask) < 100:
            inner_mask = valid_mask
        sharpness = float(laplacian[inner_mask].var())

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 40, 255], dtype=np.uint8)
        snow_mask = cv2.inRange(hsv, lower_white, upper_white) > 0
        snow_pixels = np.count_nonzero(snow_mask & valid_mask)
        snow_coverage = float(snow_pixels / valid_pixels)
        return completeness, sharpness, contrast, snow_coverage
    finally:
        try:
            del bgr, valid_mask, gray, img_for_calc, img_blurred, laplacian
            del hsv, snow_mask, inner_mask, valid_mask_u8
        except Exception:
            pass
        _trim_memory()

########################################################
def find_best_image(
    image_list,
    min_absolute_completeness=0.20,
    min_relative_completeness=0.80,
    verbose=True,
    first_date_str_pixel=None,
    last_date_str_pixel=None,
):
    """
    从 PNG 列表中选择最佳 PNG。

    重要兜底逻辑：
    1. 只要 image_list 里存在 PNG 路径，就尽量返回一个 PNG；
    2. 如果传入 first_date_str_pixel / last_date_str_pixel，则只允许选择该时间段内的 PNG；
    3. 阈值筛选为空时，不再返回空字符串，而是退回到所有可评分 PNG；
    4. 全部评分失败时，返回第一个存在的 PNG；
    5. 即使路径暂时不可读，也返回第一个 PNG 路径，避免主循环因为空字符串直接跳过。
    """
    all_png_paths = []
    seen = set()
    for file_path in image_list:
        if file_path is None:
            continue
        file_path = str(file_path)
        if not file_path.lower().endswith(".png"):
            continue
        if first_date_str_pixel is not None and last_date_str_pixel is not None:
            try:
                if not _path_has_date_in_period(file_path, first_date_str_pixel, last_date_str_pixel):
                    continue
            except Exception as e:
                if verbose:
                    print(f"PNG日期解析失败，已跳过：{file_path}，原因：{e}")
                continue
        key = os.path.abspath(file_path)
        if key in seen:
            continue
        seen.add(key)
        all_png_paths.append(file_path)

    if len(all_png_paths) == 0:
        return ""

    # 优先评分真实存在的 PNG；若都不存在，仍返回第一个 PNG 字符串，便于日志定位。
    existing_png_paths = [p for p in all_png_paths if os.path.exists(p)]
    if len(existing_png_paths) == 0:
        if verbose:
            print(f"没有找到实际存在的 PNG 文件，返回第一个 PNG 路径用于兜底：{all_png_paths[0]}")
        return all_png_paths[0]

    try:
        png_paths = natsorted(existing_png_paths)
    except Exception:
        png_paths = sorted(existing_png_paths)

    scored_records = []
    failed_paths = []
    for order_id, file_path in enumerate(png_paths):
        try:
            completeness, sharpness, contrast, snow_coverage = calculate_image_quality_metrics(file_path)
            score = (
                sharpness * 0.5
                + contrast * 0.3
                - snow_coverage * 100.0 * 0.2
                + completeness * 100.0 * 0.1
            )
            scored_records.append({
                "path": file_path,
                "score": float(score),
                "completeness": float(completeness),
                "sharpness": float(sharpness),
                "contrast": float(contrast),
                "snow_coverage": float(snow_coverage),
                "order_id": int(order_id),
            })
        except Exception as e:
            failed_paths.append(file_path)
            if verbose:
                print(f"PNG质量计算失败：{file_path}，原因：{e}")
        finally:
            if order_id % 3 == 0:
                _trim_memory()

    if len(scored_records) == 0:
        fallback_png = png_paths[0]
        if verbose:
            print(f"所有 PNG 评分失败，兜底选择第一个存在的 PNG：{fallback_png}")
        return fallback_png

    max_completeness = max(item["completeness"] for item in scored_records)
    candidate_records = []
    for item in scored_records:
        completeness = item["completeness"]
        absolute_ok = completeness >= float(min_absolute_completeness)
        relative_ok = completeness >= max_completeness * float(min_relative_completeness)
        if absolute_ok and relative_ok:
            candidate_records.append(item)

    # 旧逻辑这里会返回空字符串，导致 L4 漏掉；现在改为退回所有可评分 PNG。
    if len(candidate_records) == 0:
        candidate_records = scored_records
        if verbose:
            print("没有 PNG 同时满足完整度阈值，已启用兜底：在所有可评分 PNG 中选择最高分。")

    best_record = max(
        candidate_records,
        key=lambda item: (
            item["score"],
            item["completeness"],
            -item["snow_coverage"],
            -item["order_id"],
        )
    )
    _trim_memory()
    return best_record["path"]

########################################################
def Obtain_shp_files_from_onezero_tif_file(Tiff_file, Out_shp):
    src_ds = None
    out_ds = None
    feature_count = 0
    Out_shp = Path(Out_shp)
    try:
        Tiff_file = str(Tiff_file)
        Out_shp.parent.mkdir(parents=True, exist_ok=True)
        _remove_shapefile_dataset(Out_shp)
        src_ds = gdal.Open(Tiff_file, gdal.GA_ReadOnly)
        if src_ds is None:
            raise FileNotFoundError(f"无法打开 tif：{Tiff_file}")
        band = src_ds.GetRasterBand(1)
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        out_ds = shp_driver.CreateDataSource(str(Out_shp))
        if out_ds is None:
            raise RuntimeError(f"无法创建 shp：{Out_shp}")
        srs = None
        projection = src_ds.GetProjection()
        if projection:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)
        layer = out_ds.CreateLayer(Out_shp.stem, srs, ogr.wkbPolygon)
        field_defn = ogr.FieldDefn("value", ogr.OFTInteger)
        layer.CreateField(field_defn)
        gdal.Polygonize(band, band, layer, 0, [], callback=None)
        layer.SyncToDisk()
        feature_count = int(layer.GetFeatureCount())
    finally:
        out_ds = None
        src_ds = None
        if feature_count == 0:
            _remove_shapefile_dataset(Out_shp)
        _trim_memory()

########################################################
def _otsu_threshold_1d(values, nbins=512, value_range=(-1.0, 1.0)):
    """
    纯 NumPy 实现的一维 Otsu 阈值。
    用于 RWI 场景自适应水体分割，不依赖 skimage。
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("没有有效 RWI 像元，无法计算 Otsu 阈值。")

    if value_range is None:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = map(float, value_range)
        values = np.clip(values, vmin, vmax)

    if vmin == vmax:
        return vmin

    hist, edges = np.histogram(values, bins=int(nbins), range=(vmin, vmax))
    hist = hist.astype(np.float64, copy=False)

    if hist.sum() == 0:
        raise ValueError("RWI 直方图为空，无法计算 Otsu 阈值。")

    centers = (edges[:-1] + edges[1:]) / 2.0

    weight0 = np.cumsum(hist)
    weight1 = hist.sum() - weight0

    mean_cumsum = np.cumsum(hist * centers)
    mean_total = mean_cumsum[-1]

    mean0 = mean_cumsum / np.maximum(weight0, 1e-12)
    mean1 = (mean_total - mean_cumsum) / np.maximum(weight1, 1e-12)

    valid = (weight0 > 0) & (weight1 > 0)

    between_var = np.full_like(centers, -np.inf, dtype=np.float64)
    between_var[valid] = (
        weight0[valid]
        * weight1[valid]
        * (mean0[valid] - mean1[valid]) ** 2
    )

    return float(centers[int(np.argmax(between_var))])

########################################################
def _calculate_rwi(green, red, nir, swir1, eps=1e-6):
    """
    计算 Robust Water Index, RWI。

    输入必须是 0~1 反射率。

    RWI = [G - NIR*NIR - (1 - NIR)*SWIR1 - R*SWIR1]
          ------------------------------------------------
          [G + NIR*NIR + (1 - NIR)*SWIR1 + R*SWIR1]
    """
    green = green.astype(np.float32, copy=False)
    red = red.astype(np.float32, copy=False)
    nir = nir.astype(np.float32, copy=False)
    swir1 = swir1.astype(np.float32, copy=False)

    penalty = nir * nir + (1.0 - nir) * swir1 + red * swir1
    numerator = green - penalty
    denominator = green + penalty

    rwi = np.full(green.shape, np.nan, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=rwi,
        where=np.isfinite(denominator) & (np.abs(denominator) > eps),
    )
    np.clip(rwi, -1.0, 1.0, out=rwi)
    return rwi

########################################################
def Obtain_water_mask_from_image_by_index(
    Image_name_dir,
    Temp_path,
    Single_shape_name_dir,
    water_mndwi_threshold=0.18,
    ice_ndsi_threshold=0.35,
    ice_green_min=0.12,
    ice_swir1_max=0.22,
    cloud_visible_min=0.18,
    cloud_swir1_min=0.16,
    cirrus_threshold=0.012,
    save_diagnostic_rasters=False,
    build_arcmap_overviews=False,
    keep_water_raster=False,
    water_rwi_threshold="otsu",
    rwi_otsu_bins=512,
    save_rwi_raster=False,
    fallback_mndwi_threshold=0.1
):
    """
    使用 RWI + Otsu/手动阈值从 Sentinel-2 反射率影像提取水体。

    这个版本用于直接替换原 Process_Sentinel2_images_for_waters_02.py 中的
    Obtain_water_mask_from_image_by_index 函数。函数名称、前三个输入参数、
    默认输出文件名和原来的处理流程保持一致，只把原来的 MNDWI + 冰雪补充
    规则替换为 RWI 水体指数。

    默认输出保持原函数习惯：
    1. <Image_basename>.Swater.shp
    2. keep_water_raster=True 时额外保存 <Image_basename>.Swater.tif
    3. save_diagnostic_rasters=True 时额外保存 <Image_basename>.Scloud.tif
    4. save_rwi_raster=True 时额外保存 <Image_basename>.Srwi.tif

    参数说明
    --------
    water_rwi_threshold : "otsu" or float, default "otsu"
        "otsu" 表示用 Otsu 自动阈值；float 表示使用手动 RWI 阈值。
    rwi_otsu_bins : int, default 512
        Otsu 直方图分箱数。
    save_rwi_raster : bool, default False
        是否保存 RWI 指数栅格。
    fallback_mndwi_threshold : float, default 0.1
        当 RWI/Otsu 因没有有效像元或直方图为空而无法计算时，
        自动改用 MNDWI > fallback_mndwi_threshold 生成水体掩膜，
        默认即 MNDWI > 0.1，避免整批程序因为单景异常中断。

    兼容性说明
    ----------
    water_mndwi_threshold / ice_ndsi_threshold / ice_green_min / ice_swir1_max
    保留在函数签名中，是为了不影响脚本中已有调用；RWI 模式下默认不再使用
    这些 MNDWI/冰雪阈值参数。
    """
    # 保留旧参数，避免外部调用报错；RWI 模式下不使用这些值。
    _ = water_mndwi_threshold, ice_ndsi_threshold, ice_green_min, ice_swir1_max, cirrus_threshold

    Image_dir_name = os.path.dirname(Image_name_dir)
    Image_basename = os.path.basename(Image_name_dir)
    Image_basename_label = Image_basename[0:len(Image_basename) - 4]

    # 输出路径设置：保持原函数命名规则不变
    Out_water_shp_name = Image_basename_label + '.Swater.shp'
    Out_water_shp_name_dir = os.path.join(Image_dir_name, Out_water_shp_name)
    Out_temp_water_shp_name_dir = os.path.join(Temp_path, Out_water_shp_name)
    Out_water_tif_name_dir = os.path.join(Image_dir_name, Image_basename_label + '.Swater.tif')
    Out_temp_tif_dir = os.path.join(Temp_path, Image_basename_label + '.Swater_temp.tif')

    Out_cloud_name_dir = None
    if save_diagnostic_rasters:
        Out_cloud_name_dir = os.path.join(Image_dir_name, Image_basename_label + '.Scloud.tif')

    Out_rwi_name_dir = None
    if save_rwi_raster:
        Out_rwi_name_dir = os.path.join(Image_dir_name, Image_basename_label + '.Srwi.tif')

    os.makedirs(Temp_path, exist_ok=True)

    ds = gdal.Open(Image_name_dir, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开影像：{Image_name_dir}")

    try:
        band_count = ds.RasterCount
        rows, cols = ds.RasterYSize, ds.RasterXSize
        if band_count < 5 or rows <= 3 or cols <= 3:
            raise ValueError(f"影像至少需要5个波段，且尺寸>3。实际波段数={band_count}, 行={rows}, 列={cols}")

        proj = ds.GetProjection()
        geotrans = ds.GetGeoTransform()
        band_names = _get_gdal_band_names(ds, Image_name_dir)

        # 自动识别波段；前面导出的 ENVI 5 波段通常为：
        # 1 Blue, 2 Green, 3 Red, 4 NIR, 5 SWIR1。
        blue_idx = _find_band_index(ds, band_names,
                                    ['rhot_492', 'rhos_492', 'B02', 'B2', 'blue'],
                                    fallback_index=1)
        green_idx = _find_band_index(ds, band_names,
                                     ['rhot_560', 'rhos_560', 'B03', 'B3', 'green'],
                                     fallback_index=2)
        red_idx = _find_band_index(ds, band_names,
                                   ['rhot_665', 'rhos_665', 'B04', 'B4', 'red'],
                                   fallback_index=3)
        nir_idx = _find_band_index(ds, band_names,
                                   ['rhot_833', 'rhot_842', 'rhot_865',
                                    'rhos_833', 'rhos_842', 'rhos_865',
                                    'B08', 'B8', 'nir'],
                                   fallback_index=4)
        swir1_idx = _find_band_index(ds, band_names,
                                     ['rhot_1614', 'rhot_1610', 'rhot_1612',
                                      'rhos_1614', 'rhos_1610', 'rhos_1612',
                                      'B11', 'swir1'],
                                     fallback_index=5)

        # 读取反射率；_read_reflectance 会把 0~10000 缩放到 0~1。
        blue = _read_reflectance(ds, blue_idx)
        green = _read_reflectance(ds, green_idx)
        red = _read_reflectance(ds, red_idx)
        nir = _read_reflectance(ds, nir_idx)
        swir1 = _read_reflectance(ds, swir1_idx)

        # 有效像元：保留原函数“0 作为背景/无效值”的习惯，
        # 同时要求 RWI 所需波段处于合理反射率范围。
        valid = (
            (green > 0.0) & (green < 1.0) &
            (red >= 0.0) & (red < 1.0) &
            (nir >= 0.0) & (nir < 1.0) &
            (swir1 >= 0.0) & (swir1 < 1.0)
        )

        # 沿用原函数的简单云掩膜，用于排除明显云像元，保持输出流程一致。
        cloud = (blue > cloud_visible_min) & (green > cloud_visible_min) & \
                (red > cloud_visible_min) & (swir1 > cloud_swir1_min) & valid
        not_cloud_valid = valid & (~cloud)

        # 核心替换：优先使用 RWI + Otsu/手动阈值。
        # 如果 RWI/Otsu 没有有效像元或直方图为空，不再抛出异常，
        # 直接兜底改用 MNDWI > 0.1，保证批处理不中断。
        rwi = _calculate_rwi(green=green, red=red, nir=nir, swir1=swir1)

        valid_rwi = not_cloud_valid & np.isfinite(rwi)
        use_mndwi_fallback = False
        fallback_reason = ""
        used_threshold = None

        if isinstance(water_rwi_threshold, str) and water_rwi_threshold.lower() == "otsu":
            if int(np.count_nonzero(valid_rwi)) == 0:
                use_mndwi_fallback = True
                fallback_reason = "没有有效 RWI 像元"
            else:
                try:
                    used_threshold = _otsu_threshold_1d(
                        rwi[valid_rwi],
                        nbins=rwi_otsu_bins,
                        value_range=(-1.0, 1.0)
                    )
                except Exception as e:
                    use_mndwi_fallback = True
                    fallback_reason = str(e)
        else:
            try:
                used_threshold = float(water_rwi_threshold)
            except Exception as e:
                use_mndwi_fallback = True
                fallback_reason = f"water_rwi_threshold 无法转为 float：{e}"

        if use_mndwi_fallback:
            # 兜底方法：使用原函数中 MNDWI 的核心判断方式，但阈值固定为 0.1。
            # 注意这里的 valid_mndwi 不依赖 NIR，因此即使 NIR 异常导致 RWI 无效，
            # 也可以继续产生水体掩膜。
            valid_mndwi = (
                (green > 0.0) & (green < 1.0) &
                (swir1 >= 0.0) & (swir1 < 1.0)
            )
            cloud_mndwi = (blue > cloud_visible_min) & (green > cloud_visible_min) & \
                           (red > cloud_visible_min) & (swir1 > cloud_swir1_min) & valid_mndwi
            not_cloud_valid_mndwi = valid_mndwi & (~cloud_mndwi)

            eps = np.float32(1e-6)
            mndwi = (green - swir1) / (green + swir1 + eps)
            np.nan_to_num(mndwi, copy=False, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)

            used_threshold = float(fallback_mndwi_threshold)
            water_mask = ((mndwi > used_threshold) & not_cloud_valid_mndwi).astype(np.uint8)

            # 诊断云图也切换为 MNDWI 兜底所使用的云掩膜，避免 RWI valid 为空时云图全为 0。
            cloud = cloud_mndwi
            print(
                f"[Warning] {Image_basename_label}: RWI/Otsu 无法计算，"
                f"已自动改用 MNDWI > {used_threshold}。原因：{fallback_reason}"
            )
        else:
            water_mask = ((rwi > used_threshold) & valid_rwi).astype(np.uint8)

        # 形态学开运算去噪：保持原函数输出风格。
        kernel = np.ones((3, 3), dtype=np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        # 2. 输出诊断栅格（可选）
        if save_diagnostic_rasters and Out_cloud_name_dir is not None:
            cloud_mask_byte = cloud.astype(np.uint8)
            _write_single_band_tif(Out_cloud_name_dir, cloud_mask_byte, proj, geotrans, gdal.GDT_Byte)

        if save_rwi_raster and Out_rwi_name_dir is not None:
            rwi_out = np.where(valid, rwi, -9999.0).astype(np.float32)
            _write_single_band_tif(Out_rwi_name_dir, rwi_out, proj, geotrans, gdal.GDT_Float32)

        # 3. 水体栅格 -> Shapefile；保持原函数输出一致。
        driver = gdal.GetDriverByName('GTiff')
        temp_ds = driver.Create(Out_temp_tif_dir, cols, rows, 1, gdal.GDT_Byte)
        if temp_ds is None:
            raise RuntimeError(f"无法创建临时水体栅格：{Out_temp_tif_dir}")
        temp_ds.SetProjection(proj)
        temp_ds.SetGeoTransform(geotrans)
        temp_ds.GetRasterBand(1).WriteArray(water_mask)
        temp_ds = None

        Obtain_shp_files_from_onezero_tif_file(Out_temp_tif_dir, Out_temp_water_shp_name_dir)

        # 4. 裁剪到目标湖泊
        if os.path.exists(Out_temp_water_shp_name_dir):
            try:
                select_target_lake_by_original_shp(
                    water_shp=Out_temp_water_shp_name_dir,
                    lake_shp=Single_shape_name_dir,
                    out_shp=Out_water_shp_name_dir
                )
            except ValueError as e:
                if "没有筛选到目标湖泊" in str(e):
                    _copy_shp_dataset(Out_temp_water_shp_name_dir, Out_water_shp_name_dir)
                else:
                    raise

        # 5. 清理临时文件
        if os.path.exists(Out_temp_tif_dir):
            os.remove(Out_temp_tif_dir)

        # 根据 keep_water_raster 决定是否保留永久水体栅格
        if keep_water_raster:
            out_ds = driver.Create(Out_water_tif_name_dir, cols, rows, 1, gdal.GDT_Byte)
            if out_ds is None:
                raise RuntimeError(f"无法创建水体栅格：{Out_water_tif_name_dir}")
            out_ds.SetProjection(proj)
            out_ds.SetGeoTransform(geotrans)
            out_ds.GetRasterBand(1).WriteArray(water_mask)
            out_ds = None

        # 删除 Temp_path 下的 Shapefile 组分
        for suffix in ('.shp', '.shx', '.prj', '.dbf', '.cpg', '.qpj'):
            temp_file = os.path.join(Temp_path, Image_basename_label + '.Swater' + suffix)
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # 6. 构建 ArcMap 金字塔（可选）
        if build_arcmap_overviews:
            rasters = []
            if keep_water_raster and os.path.exists(Out_water_tif_name_dir):
                rasters.append(Out_water_tif_name_dir)
            if save_diagnostic_rasters and Out_cloud_name_dir and os.path.exists(Out_cloud_name_dir):
                rasters.append(Out_cloud_name_dir)
            if save_rwi_raster and Out_rwi_name_dir and os.path.exists(Out_rwi_name_dir):
                rasters.append(Out_rwi_name_dir)
            for rast in rasters:
                prepare_raster_for_arcmap(rast, resampling="nearest")

    finally:
        ds = None
        # 这些变量是大数组，设为 None 可以立即降低长循环中的峰值和驻留内存。
        try:
            blue = green = red = nir = swir1 = None
            valid = valid_rwi = cloud = not_cloud_valid = None
            rwi = rwi_out = water_mask = None
            cloud_mask_byte = temp_ds = out_ds = None
        except Exception:
            pass
        _trim_memory()

########################################################
def Image_PNG_with_small_size(PNG_file):
    Image.MAX_IMAGE_PIXELS = None
    Out_small_PNG_name = PNG_file[0:len(PNG_file)-4] + '.small.PNG'
    with Image.open(PNG_file) as image:
        base_width = int(image.size[0] / 10)
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * float(w_percent))
        if h_size > 0 and base_width > 0:
            image2 = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            image2.save(Out_small_PNG_name)
            image2.close()
            del image2
    _trim_memory()

########################################################
def project_xy(tif_path):
    dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"无法打开影像：{tif_path}")
    try:
        geo_information = dataset.GetGeoTransform()
        col = dataset.RasterXSize
        row = dataset.RasterYSize
        top_left_corner_lon = geo_information[0]
        top_left_corner_lat = geo_information[3]
        bottom_left_corner_lon = geo_information[0] + row * geo_information[2]
        bottom_left_corner_lat = geo_information[3] + row * geo_information[5]
        top_right_corner_lon = geo_information[0] + col * geo_information[1]
        top_right_corner_lat = geo_information[3] + col * geo_information[4]
        bottom_right_corner_lon = geo_information[0] + col * geo_information[1] + row * geo_information[2]
        bottom_right_corner_lat = geo_information[3] + col * geo_information[4] + row * geo_information[5]
        return (top_left_corner_lon, top_left_corner_lat, bottom_left_corner_lon,
                bottom_left_corner_lat, top_right_corner_lon, top_right_corner_lat,
                bottom_right_corner_lon, bottom_right_corner_lat)
    finally:
        dataset = None
        _trim_memory()

########################################################
def Convert_png_transparent(src_file, dst_file, bg_color=(0, 0, 0)):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src_file) as image:
        array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    bg = np.asarray(bg_color, dtype=np.uint8)
    mask = np.all(array[:, :, :3] == bg, axis=2)
    array[:, :, 3] = 255
    array[:, :, 3][mask] = 0
    out_img = Image.fromarray(array, mode="RGBA")
    out_img.save(dst_file, "PNG", dpi=(1000, 1000))
    out_img.close()
    del out_img, array, mask, bg
    _trim_memory()

########################################################
def Create_PNG_file_of_Sentinel2(Image_path):
    Path_dir = os.path.dirname(Image_path)
    Basename = os.path.basename(Image_path)
    Basename_label = Basename[0:len(Basename)-4]
    Image_dataset = None
    try:
        Image_dataset = gdal.Open(Image_path, gdal.GA_ReadOnly)
        if Image_dataset is None:
            raise FileNotFoundError(f"无法打开影像：{Image_path}")
        Band_count = Image_dataset.RasterCount
        if Band_count <= 2:
            return
        R_raw = Image_dataset.GetRasterBand(4).ReadAsArray().astype(np.float32, copy=False)
        if np.nanmax(R_raw) <= 0:
            del R_raw
            return
        R = _scale_to_uint8_inplace(R_raw, percentile=(0, 99), divisor=10000.0)
        G = _scale_to_uint8_inplace(
            Image_dataset.GetRasterBand(3).ReadAsArray().astype(np.float32, copy=False),
            percentile=(0, 99),
            divisor=10000.0,
        )
        B = _scale_to_uint8_inplace(
            Image_dataset.GetRasterBand(2).ReadAsArray().astype(np.float32, copy=False),
            percentile=(0, 99),
            divisor=10000.0,
        )
        rgb_array = np.empty((R.shape[0], R.shape[1], 3), dtype=np.uint8)
        rgb_array[:, :, 0] = R
        rgb_array[:, :, 1] = G
        rgb_array[:, :, 2] = B
        del R, G, B
        RGB_image = Image.fromarray(rgb_array, mode="RGB")
        RGB_image.save(Path_dir + '/' + Basename_label + '.temp.PNG', dpi=(1000, 1000))
        RGB_image.close()
        del RGB_image, rgb_array
    finally:
        Image_dataset = None
        _trim_memory()

########################################################
def Merge_Sentinel2_image_data(Rrs_for_merge_list):
    """
    将多个 Sentinel-2 处理后的 ENVI 波段文件（.img）合并为单个文件。
    使用 gdal.Warp 替代外部命令行工具 gdal_merge.py，并在每个波段结束后关闭 GDAL 数据集。
    """
    S2_Rrs_list = []
    Merge_path = ''

    for Sub_i in Rrs_for_merge_list:
        if 'SAFE' in Sub_i:
            Merge_path = os.path.dirname(Sub_i) + '/' + (os.path.basename(Sub_i))[0:19]
            os.makedirs(Merge_path, exist_ok=True)
            if os.path.exists(os.path.join(Sub_i, 'B2.img')):
                S2_Rrs_list.append(Sub_i)

    if not S2_Rrs_list:
        return ''

    first_dir = S2_Rrs_list[0]
    band_files = [f for f in os.listdir(first_dir) if f.endswith('.img')]

    for band_file in band_files:
        src_paths = []
        warp_ds = None
        try:
            for sub_dir in S2_Rrs_list:
                src_path = os.path.join(sub_dir, band_file)
                if os.path.exists(src_path):
                    src_paths.append(src_path)
            if not src_paths:
                continue

            out_path = os.path.join(Merge_path, band_file)
            warp_options = gdal.WarpOptions(
                format='ENVI',
                srcNodata=0,
                dstNodata=0,
                resampleAlg='near',
                multithread=True,
                warpOptions=['NUM_THREADS=ALL_CPUS'],
            )
            warp_ds = gdal.Warp(out_path, src_paths, options=warp_options)
            if warp_ds is not None:
                warp_ds.FlushCache()
        finally:
            warp_ds = None
            try:
                del src_paths
            except Exception:
                pass
            _trim_memory()

    if any(os.path.exists(os.path.join(Merge_path, f)) for f in band_files):
        return Merge_path
    return ''

########################################################
def Replace_nan_to_zero_of_tif(input_raster, output_raster):
    upper_threshold = 9999
    lower_threshold = -9999
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.int16, nodata=0)
        with rasterio.open(output_raster, 'w', **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(1, window=window, out_dtype='float32')
                invalid = (~np.isfinite(data)) | (data > upper_threshold) | (data < lower_threshold)
                data[invalid] = 0
                data *= np.float32(10000.0)
                np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max, out=data)
                data_scaled = data.astype(np.int16)
                dst.write(data_scaled, 1, window=window)
                del data, data_scaled, invalid
    _trim_memory()

########################################################
my_queue = queue.Queue()
def storeInQueue(f):
    def wrapper(*args):
        result = ''
        try:
            result = f(*args)
        except Exception as e:
            result = ''
        finally:
            my_queue.put(result)
            _trim_memory()
    return wrapper

########################################################
@storeInQueue
def Process_Sentinel2_get_rhos_IGA(sentinel2_1C_specific, Shape_file,L2_path):
    Shape_name = shapefile.Reader(Shape_file, encoding="gb2312")
    Lon1 = round(float((Shape_name.bbox)[0]), 4)
    Lon2 = round(float((Shape_name.bbox)[2]), 4)
    Lat1 = round(float((Shape_name.bbox)[1]), 4)
    Lat2 = round(float((Shape_name.bbox)[3]), 4)
    Lat_mean = (Lat1 + Lat2) / 2
    Lon_mean = (Lon1 + Lon2) / 2
    Zone = str(int(31.0 + Lon_mean / 6.0)).zfill(2)
    if Lat_mean < 0:
        Zone = Zone + ' ' + '+south'
    srs = osr.SpatialReference()
    crs = CRS.from_string('+proj=utm +zone=' + Zone)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int((crs.to_authority())[1]))
    Sentinel_basename = os.path.basename(sentinel2_1C_specific)
    L2_out_path = L2_path + '/' + Sentinel_basename
    if not os.path.exists(L2_path):
        os.mkdir(L2_path)
    if not os.path.exists(L2_out_path):
        os.mkdir(L2_out_path)
    Export_s2_l1c_toa10000_int16_10m_envi_projected(
        safe_dir=sentinel2_1C_specific,
        bands=["B02", "B03", "B04", "B08", "B11"],
        out_dir=L2_out_path,
        target_epsg=int((crs.to_authority())[1]),
        clip_vector=Shape_file,
        clip_all_touched=False
    )
    if os.path.exists(L2_out_path):
        Sen_test_list = os.listdir(L2_out_path)
        if len(Sen_test_list) > 3:
            return L2_out_path
        else:
            return ''
    else:
        return ''

########################################################
def extract_source_files(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if 'Source_files_begin' in line:
            start = i + 1
        elif 'Source_files_end' in line:
            end = i
            break
    if start is not None and end is not None:
        source_files = [line.strip() for line in lines[start:end]]
        return source_files
    else:
        return []

########################################################
def _normalize_date_label(date_value):
    """把 YYYY-MM-DD / YYYYMMDD / datetime 统一转成 YYYYMMDD。"""
    if isinstance(date_value, datetime):
        return date_value.strftime("%Y%m%d")
    if date_value is None:
        raise ValueError("日期不能为空。")
    text = str(date_value).strip()
    if re.fullmatch(r"\d{8}", text):
        datetime.strptime(text, "%Y%m%d")
        return text
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y%m%d")
    compact = re.sub(r"[^0-9]", "", text)
    if re.fullmatch(r"\d{8}", compact):
        datetime.strptime(compact, "%Y%m%d")
        return compact
    raise ValueError(f"无法解析日期：{date_value}")

########################################################
def _extract_date_labels_from_name(file_name):
    """从文件名中提取所有有效 8 位日期标签，返回 YYYYMMDD 列表。"""
    base_name = os.path.basename(str(file_name))
    labels = []

    def add_label(text):
        try:
            label = _normalize_date_label(text)
        except Exception:
            return
        if label not in labels:
            labels.append(label)

    # Sentinel 文件名常见形式：..._YYYYMMDDT103021_...
    for match in re.finditer(r"(?:^|[_-])(\d{8})T\d{6}", base_name):
        add_label(match.group(1))
    # 普通输出文件名常见形式：..._YYYYMMDD_... / ..._YYYYMMDD.PNG
    for match in re.finditer(r"(?:^|[_-])(\d{8})(?=[_.-]|$)", base_name):
        add_label(match.group(1))
    # 兜底：文件名中任意有效 8 位日期。
    for match in re.finditer(r"(\d{8})", base_name):
        add_label(match.group(1))
    return labels

########################################################
def _extract_date_label_from_name(file_name):
    """从文件名中提取第一个有效 8 位日期标签；找不到则返回 None。"""
    labels = _extract_date_labels_from_name(file_name)
    return labels[0] if labels else None

########################################################
def _path_has_date_in_period(file_path, first_date_str_pixel, last_date_str_pixel):
    """判断文件名中的任一有效日期是否落在指定时间段内。"""
    first_label = _normalize_date_label(first_date_str_pixel)
    last_label = _normalize_date_label(last_date_str_pixel)
    first_int = int(first_label)
    last_int = int(last_label)
    if first_int > last_int:
        raise ValueError(f"起始日期晚于结束日期：{first_date_str_pixel} > {last_date_str_pixel}")
    labels = _extract_date_labels_from_name(file_path)
    for label in labels:
        date_int = int(label)
        if first_int <= date_int <= last_int:
            return True
    return False

########################################################
def _collect_png_candidates_for_period(l3_path, first_date_str_pixel, last_date_str_pixel):
    """只从 L3 收集指定时间段内的 PNG，大小写不敏感，并排除 small PNG。"""
    candidates = []
    seen = set()
    first_label = _normalize_date_label(first_date_str_pixel)
    last_label = _normalize_date_label(last_date_str_pixel)
    first_int = int(first_label)
    last_int = int(last_label)
    if first_int > last_int:
        raise ValueError(f"起始日期晚于结束日期：{first_date_str_pixel} > {last_date_str_pixel}")

    directory = Path(l3_path)
    if not directory.exists() or not directory.is_dir():
        return candidates
    try:
        names = os.listdir(directory)
    except Exception:
        return candidates

    for name in names:
        name_lower = name.lower()
        if not name_lower.endswith('.png'):
            continue
        if '.small.' in name_lower:
            continue
        if not _path_has_date_in_period(name, first_label, last_label):
            continue
        path = str(directory / name)
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    try:
        return natsorted(candidates)
    except Exception:
        return sorted(candidates)

########################################################
def _move_selected_product_to_l4(best_png_path, l4_path, overwrite=False):
    """把最佳 PNG 及同名 img/hdr/txt/JPEG/水体 shp 组分移动到 L4。"""
    if not best_png_path:
        return []
    src_png = Path(best_png_path)
    l4_path = Path(l4_path)
    l4_path.mkdir(parents=True, exist_ok=True)

    # 如果最佳结果已经在 L4 中，直接认为成功，不重复移动。
    try:
        if src_png.parent.resolve() == l4_path.resolve():
            return [str(src_png)] if src_png.exists() else []
    except Exception:
        pass

    stem = src_png.stem
    suffixes = [
        '.PNG', '.png', '.JPEG', '.JPG', '.jpeg', '.jpg', '.txt',
        '.img', '.hdr', '.img.aux.xml', '.img.ovr',
        '.Swater.shp', '.Swater.shx', '.Swater.prj', '.Swater.dbf', '.Swater.cpg',
        '.Swater.qpj', '.Swater.sbn', '.Swater.sbx', '.Swater.shp.xml',
    ]
    moved_files = []
    for suffix in suffixes:
        src_file = src_png.parent / f'{stem}{suffix}'
        if not src_file.exists():
            continue
        dst_file = l4_path / src_file.name
        try:
            if dst_file.exists():
                if overwrite:
                    if dst_file.is_dir():
                        shutil.rmtree(dst_file, onerror=handle_remove_readonly)
                    else:
                        dst_file.unlink()
                else:
                    continue
            shutil.move(str(src_file), str(dst_file))
            moved_files.append(str(dst_file))
        except Exception as e:
            print(f"移动文件失败：{src_file} -> {dst_file}，原因：{e}")
    _trim_memory()
    return moved_files

########################################################
def select_and_move_best_png_for_period(
    l3_path,
    l4_path,
    first_date_str_pixel,
    last_date_str_pixel,
    min_absolute_completeness=0.20,
    min_relative_completeness=0.80,
):
    """
    只从 L3 中找指定时间段 PNG，选出最佳 PNG 后移动整套产品到 L4。
    返回最终选中的 PNG 路径；L3 在该时间段没有任何 PNG 时返回空字符串。
    """
    first_label = _normalize_date_label(first_date_str_pixel)
    last_label = _normalize_date_label(last_date_str_pixel)
    png_candidates = _collect_png_candidates_for_period(
        l3_path=l3_path,
        first_date_str_pixel=first_label,
        last_date_str_pixel=last_label,
    )
    if len(png_candidates) == 0:
        #print(f"时间段 {first_label}-{last_label} 在 L3 没有找到 PNG 候选，L4 无法新增该时间段数据。")
        return ""

    best_png = find_best_image(
        png_candidates,
        min_absolute_completeness=min_absolute_completeness,
        min_relative_completeness=min_relative_completeness,
        verbose=True,
        first_date_str_pixel=first_label,
        last_date_str_pixel=last_label,
    )
    if not best_png:
        # 理论上只要 png_candidates 非空，find_best_image 不会再返回空；这里再保一层底。
        best_png = png_candidates[0]

    if not _path_has_date_in_period(best_png, first_label, last_label):
        #print(f"最佳 PNG 不在时间段 {first_label}-{last_label} 内，已跳过：{best_png}")
        return ""

    moved_files = _move_selected_product_to_l4(best_png, l4_path)
    if len(moved_files) > 0:
        final_png = str(Path(l4_path) / Path(best_png).name)
    else:
        final_png = best_png
    #print(f"时间段 {first_label}-{last_label} 选择 PNG：{final_png}")
    _trim_memory()
    return final_png

########################################################
if __name__ == "__main__":
    MAX_PARALLEL_SENTINEL_PROCESS = 2
    Copernicus_username, Copernicus_password = get_copernicus_credentials()
    Shape_base_path = r'/media/a/Data1/South_China/Shape/Buffer_province/青海省/Split/part_03' 
    Single_base_path = r'/media/a/Data1/South_China/Shape/Single' 
    First_days_list=['2015-12-01','2016-03-01','2016-06-01',
                        '2016-12-01','2017-03-01','2017-06-01',
                        '2017-12-01','2018-03-01','2018-06-01',
                        '2018-12-01','2019-03-01','2019-06-01',
                        '2019-12-01','2020-03-01','2020-06-01',
                        '2020-12-01','2021-03-01','2021-06-01',
                        '2021-12-01','2022-03-01','2022-06-01',
                        '2022-12-01','2023-03-01','2023-06-01',
                        '2023-12-01','2024-03-01','2024-06-01',
                        '2024-12-01','2025-03-01','2025-06-01']
    Last_days_list= ['2016-02-29','2016-05-31','2016-08-31',
                        '2017-02-28','2017-05-31','2017-08-31',
                        '2018-02-28','2018-05-31','2018-08-31',
                        '2019-02-28','2019-05-31','2019-08-31',
                        '2020-02-29','2020-05-31','2020-08-31',
                        '2021-02-28','2021-05-31','2021-08-31',
                        '2022-02-28','2022-05-31','2022-08-31',
                        '2023-02-28','2023-05-31','2023-08-31',
                        '2024-02-29','2024-05-31','2024-08-31',
                        '2025-02-28','2025-05-31','2025-08-31']
    Cloud_value = '10'
    L1_path = '/media/a/Data1/South_China/L1'
    L2_path = '/home/a/Sentinel2/L2'
    L3_basename_path = '/media/a/Data1/South_China/L3'
    L4_basename_path = '/media/a/Data1/South_China/L4'
    Txt_log_path = r'/media/a/Data1/South_China/Process_order.txt'
    Path_row_file = r'/media/a/Data1/South_China/Shape/Base/sentinel_2_index_shapefile.shp'
    Temp_path = r'/media/a/Data1/South_China/Temp'
    Out_overlay_file = Temp_path + '/' + 'study_shapefile'
    if os.path.exists(Temp_path + '/' + 'study_shapefile'):
        shutil.rmtree(Temp_path + '/' + 'study_shapefile')
    Shape_dir_names_first = os.listdir(Shape_base_path)
    Shape_dir_names = natsorted(Shape_dir_names_first)
    for shp_i in range(3366, len(Shape_dir_names)):
        Shape_name = Shape_dir_names[shp_i]
        if '.shp' in Shape_name:
            print(shp_i, Shape_name)
            with open(Txt_log_path, 'w+', encoding='utf-8') as Order_shp_out_txt:
                Order_shp_out_txt.write(str(shp_i) + '\n')
                Order_shp_out_txt.write(str(Shape_name))
            Shape_file = Shape_base_path + '/' + Shape_name
            Single_shape_name_dir = Single_base_path + '/' + Shape_name
            Path_low_list = Get_path_row_of_sentine2_by_shapefile(Path_row_file, Shape_file, Out_overlay_file)
            if os.path.exists(Temp_path + '/' + 'study_shapefile'):
                shutil.rmtree(Temp_path + '/' + 'study_shapefile')
            L3_path = L3_basename_path + '/' + Shape_name
            L4_path = L4_basename_path + '/' + Shape_name
            if not os.path.exists(L2_path):
                os.mkdir(L2_path)
            if not os.path.exists(L3_path):
                os.mkdir(L3_path)
            if not os.path.exists(L4_basename_path):
                os.mkdir(L4_basename_path)
            if not os.path.exists(L4_path):
                os.mkdir(L4_path)
            for date_i in range(0, len(First_days_list)):
                First_date_str_pixel = First_days_list[date_i]
                Last_date_str_pixel = Last_days_list[date_i]
                First_date_label = (First_date_str_pixel.split("-")[0] +
                                    First_date_str_pixel.split("-")[1] +
                                    First_date_str_pixel.split("-")[2])
                Last_date_label = (Last_date_str_pixel.split("-")[0] +
                                   Last_date_str_pixel.split("-")[1] +
                                   Last_date_str_pixel.split("-")[2])
                print([First_date_str_pixel, Last_date_str_pixel])
                Date_Begin_Year = First_date_str_pixel.split("-")[0]
                Date_Begin_Month = First_date_str_pixel.split("-")[1]
                Date_Begin_Day = First_date_str_pixel.split("-")[2]
                Date_End_Year = Last_date_str_pixel.split("-")[0]
                Date_End_Month = Last_date_str_pixel.split("-")[1]
                Date_End_Day = Last_date_str_pixel.split("-")[2]
                Month_Sentinel2_1C_files = Find_Sentinel2_using_IGA(
                    Date_Begin_Year, Date_Begin_Month, Date_Begin_Day,
                    Date_End_Year, Date_End_Month, Date_End_Day,
                    Cloud_value, L1_path, Shape_file, Path_low_list)
                if len(Month_Sentinel2_1C_files) > 0:
                    Date_array = extract_unique_sorted_dates(Month_Sentinel2_1C_files)
                    if len(Date_array) > 0:
                        Image_coverage_ratio_total = 0
                        for date_j in range(0, len(Date_array)):
                            Date_pixel = Date_array[date_j]
                            print(Date_pixel)
                            Date_year_pixel = Date_pixel.split("-")[0]
                            Date_month_pixel = Date_pixel.split("-")[1]
                            Date_day_pixel = Date_pixel.split("-")[2]
                            Day_Sentinel2_1C_files = Download_Sentinel2_using_IGA(
                                Date_year_pixel, Date_month_pixel, Date_day_pixel,
                                Date_year_pixel, Date_month_pixel, Date_day_pixel,
                                Cloud_value, L1_path, Shape_file, Path_low_list,
                                Copernicus_username, Copernicus_password)
                            #for day_test_i in range(0, len(Day_Sentinel2_1C_files)):
                                #print(Day_Sentinel2_1C_files[day_test_i])
                            if len(Day_Sentinel2_1C_files) > 0:
                                S2A_downloaded_number = 0
                                S2B_downloaded_number = 0
                                S2C_downloaded_number = 0
                                S2A_download_list = list()
                                S2B_download_list = list()
                                S2C_download_list = list()
                                for download_s2_test_i in range(0, len(Day_Sentinel2_1C_files)):
                                    Download_s2_name = os.path.basename(Day_Sentinel2_1C_files[download_s2_test_i])
                                    if 'S2A' in Download_s2_name:
                                        S2A_download_list.append(Download_s2_name)
                                    if 'S2B' in Download_s2_name:
                                        S2B_download_list.append(Download_s2_name)
                                    if 'S2C' in Download_s2_name:
                                        S2C_download_list.append(Download_s2_name)
                                Sentinel2_rhos_files_array = list()
                                Sentinel_1C_threading = []
                                for sentinel2_1C_specific in Day_Sentinel2_1C_files:
                                    sentinel2_1C_name = os.path.basename(sentinel2_1C_specific)
                                    sentinel2_1C_type = sentinel2_1C_name[0:3]
                                    if ((sentinel2_1C_type == 'S2A' and S2A_downloaded_number == 0) or
                                        (sentinel2_1C_type == 'S2B' and S2B_downloaded_number == 0) or
                                        (sentinel2_1C_type == 'S2C' and S2C_downloaded_number == 0)):
                                        T_p_single = threading.Thread(target=Process_Sentinel2_get_rhos_IGA,args=(sentinel2_1C_specific, Shape_file, L2_path))
                                        Sentinel_1C_threading.append(T_p_single)
                                for batch_start in range(0,len(Sentinel_1C_threading), MAX_PARALLEL_SENTINEL_PROCESS):
                                    batch_threads = Sentinel_1C_threading[batch_start:batch_start + MAX_PARALLEL_SENTINEL_PROCESS]
                                    for thread_1c_single_i in batch_threads:
                                        thread_1c_single_i.start()
                                        time.sleep(1)
                                    for thread_1c_single_j in batch_threads:
                                        thread_1c_single_j.join()
                                    for _ in batch_threads:
                                        Sentinel2_rhos_files_array.append(my_queue.get())
                                    del batch_threads
                                    _trim_memory()
                                if len(Sentinel2_rhos_files_array)>0:
                                    test_rhos_total = np.zeros(len(Sentinel2_rhos_files_array))
                                    for test_rhos_i in range(0,len(Sentinel2_rhos_files_array)):
                                        test_rhos_total[test_rhos_i] = len(Sentinel2_rhos_files_array[test_rhos_i])
                                    if np.max(test_rhos_total)>0:
                                        Sentinel2A_1C_rhos_files_array = list()
                                        Sentinel2B_1C_rhos_files_array = list()
                                        Sentinel2C_1C_rhos_files_array = list()
                                        if len(Sentinel2_rhos_files_array)>0:
                                            for img_class_i in range(0,len(Sentinel2_rhos_files_array)):
                                                Image_class_name_dir_i = Sentinel2_rhos_files_array[img_class_i]
                                                Image_class_name_i = os.path.basename(Image_class_name_dir_i)
                                                if 'S2A_M' in Image_class_name_i:
                                                    Sentinel2A_1C_rhos_files_array.append(Image_class_name_dir_i)
                                                if 'S2B_M' in Image_class_name_i:
                                                    Sentinel2B_1C_rhos_files_array.append(Image_class_name_dir_i)
                                                if 'S2C_M' in Image_class_name_i:
                                                    Sentinel2C_1C_rhos_files_array.append(Image_class_name_dir_i)
                                        if len(Sentinel2A_1C_rhos_files_array)>0:
                                            Sentinel2A_rhos_merge_file=Merge_Sentinel2_image_data(Sentinel2A_1C_rhos_files_array)
                                        if len(Sentinel2B_1C_rhos_files_array)>0:
                                            Sentinel2B_rhos_merge_file=Merge_Sentinel2_image_data(Sentinel2B_1C_rhos_files_array)
                                        if len(Sentinel2C_1C_rhos_files_array)>0:
                                            Sentinel2C_rhos_merge_file=Merge_Sentinel2_image_data(Sentinel2C_1C_rhos_files_array)
                                        ##########################################################################################    
                                        if len(Sentinel2A_1C_rhos_files_array)>0:
                                            List_for_S2A_stack = list()
                                            Band_names_S2A_for_stack = list()
                                            Band_label_in_array = ["B2", "B3", "B4", "B8", "B11"]
                                            Band_label_out_array = ["492", "560", "665", "833", "1614"]
                                            for List_S3A_i in range(0,len(Band_label_in_array)):
                                                List_band_label = Band_label_in_array[List_S3A_i]
                                                List_out_band_label = Band_label_out_array[List_S3A_i]
                                                if os.path.exists(Sentinel2A_rhos_merge_file + '/' + List_band_label + '.img'):
                                                    List_for_S2A_stack.append(Sentinel2A_rhos_merge_file + '/' + List_band_label + '.img')
                                                    Band_names_S2A_for_stack.append('rhot_' + List_out_band_label)
                                            if len(List_for_S2A_stack)>0:
                                                with rasterio.open(List_for_S2A_stack[0]) as raster_info:
                                                    raster_meta = raster_info.meta.copy()
                                                raster_meta.update(count=len(List_for_S2A_stack))
                                                raster_meta.update(driver="ENVI")
                                                with rasterio.open(Sentinel2A_rhos_merge_file + '.img', "w", **raster_meta) as dst:
                                                    for id, layer in enumerate(List_for_S2A_stack, start=1):
                                                        with rasterio.open(layer) as src:
                                                            for _, window in src.block_windows(1):
                                                                block = src.read(1, window=window)
                                                                dst.write(block, id, window=window)
                                                                del block
                                                        _trim_memory()
                                                hdr_out_txt = open(Sentinel2A_rhos_merge_file + '.hdr', 'r')
                                                hdr_lines = hdr_out_txt.readlines()
                                                hdr_band_pos=0
                                                for hdr_i in range(0,len(hdr_lines)):
                                                    if 'band names =' in hdr_lines[hdr_i]:
                                                        hdr_band_pos = hdr_i
                                                with open(Sentinel2A_rhos_merge_file + '.txt', 'w') as g:
                                                    for hdr_j in range(0, hdr_band_pos):
                                                        g.write(hdr_lines[hdr_j])
                                                    g.write('band names = {' + '\n')
                                                    for bname_i in range(0, len(Band_names_S2A_for_stack)):
                                                        bname = Band_names_S2A_for_stack[bname_i]
                                                        if bname_i < len(Band_names_S2A_for_stack) - 1:
                                                            g.write(bname + ',' + '\n')
                                                        else:
                                                            g.write(bname + '}')
                                                g.close()
                                                L4_sentinel_2A_merge_envi_file=(L3_path+'/'+os.path.basename(Sentinel2A_rhos_merge_file))
                                                if os.path.exists(Sentinel2A_rhos_merge_file+'.hdr'):
                                                    os.remove(Sentinel2A_rhos_merge_file+'.hdr')
                                                if os.path.exists(Sentinel2A_rhos_merge_file+'.img.aux.xml'):
                                                    os.remove(Sentinel2A_rhos_merge_file+'.img.aux.xml')
                                                if os.path.exists(Sentinel2A_rhos_merge_file+'.txt'):
                                                    shutil.move(Sentinel2A_rhos_merge_file+'.txt',Sentinel2A_rhos_merge_file + '.hdr')
                                                if os.path.exists(Sentinel2A_rhos_merge_file+'.img'):
                                                    shutil.move(Sentinel2A_rhos_merge_file+'.img',L4_sentinel_2A_merge_envi_file + '.img')
                                                if os.path.exists(Sentinel2A_rhos_merge_file+'.hdr'):
                                                    shutil.move(Sentinel2A_rhos_merge_file+'.hdr',L4_sentinel_2A_merge_envi_file + '.hdr')
                                                raster_info = None
                                                raster_meta = None
                                                dst = None
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.img'):
                                                    Obtain_water_mask_from_image_by_index(L4_sentinel_2A_merge_envi_file+'.img',Temp_path,Single_shape_name_dir,water_rwi_threshold='otsu')
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.img'):
                                                    prepare_raster_for_arcmap(L4_sentinel_2A_merge_envi_file+'.img',resampling="nearest")
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.img'):
                                                    Create_PNG_file_of_Sentinel2(L4_sentinel_2A_merge_envi_file+'.img')
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.temp.PNG'): 
                                                    Convert_png_transparent(L4_sentinel_2A_merge_envi_file+'.temp.PNG',L4_sentinel_2A_merge_envi_file+'.PNG',bg_color=(0,0,0)) 
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.img') and os.path.exists(Single_shape_name_dir) and os.path.exists(L4_sentinel_2A_merge_envi_file+'.Swater.shp'):
                                                    Make_three_panel_overlay(raster_path=L4_sentinel_2A_merge_envi_file+'.img',vector1_path=Single_shape_name_dir,vector2_path=L4_sentinel_2A_merge_envi_file+'.Swater.shp',output_path=L4_sentinel_2A_merge_envi_file+'.JPEG',background_color=(0, 0, 0))   
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.temp.PNG'):
                                                    os.remove(L4_sentinel_2A_merge_envi_file+'.temp.PNG')  
                                                if os.path.exists(Sentinel2A_rhos_merge_file):
                                                    shutil.rmtree(Sentinel2A_rhos_merge_file)    
                                                S2A_all_tuple = project_xy(L4_sentinel_2A_merge_envi_file+'.img') 
                                                S2A_L3_basename=os.path.basename(L4_sentinel_2A_merge_envi_file+'.img')
                                                S2A_L3_Dirname=os.path.dirname(L4_sentinel_2A_merge_envi_file+'.img')
                                                S2A_Txt_path=S2A_L3_Dirname+'/'+S2A_L3_basename[0:len(S2A_L3_basename)-4]+'.txt'
                                                Sensor_type=S2A_L3_basename[0:3]
                                                with open(S2A_Txt_path, 'w') as g:
                                                      g.write('Basename:'+S2A_L3_basename+'\n')
                                                      g.write('Sensortype:'+Sensor_type+'\n')
                                                      g.write('left_top_corner:'+str(S2A_all_tuple[0])+','+str(S2A_all_tuple[1])+'\n')
                                                      g.write('left_bottom_corner:'+str(S2A_all_tuple[2])+','+str(S2A_all_tuple[3])+'\n')
                                                      g.write('right_top_corner:'+str(S2A_all_tuple[4])+','+str(S2A_all_tuple[5])+'\n')
                                                      g.write('right_bottom_corner:'+str(S2A_all_tuple[6])+','+str(S2A_all_tuple[7])+'\n')
                                                      g.write('Source_files_begin'+'\n')
                                                      for S2A_down_sen_file_i in Sentinel2A_1C_rhos_files_array:
                                                          Out_sen_file_name=os.path.basename(S2A_down_sen_file_i)
                                                          g.write(Out_sen_file_name+'\n')   
                                                      g.write('Source_files_end'+'\n')
                                                g.close()  
                                                S2A_all_tuple=None
                                                Band_label_out_array=None 
                                                for L2_S2A_rest_i in range(0,len(S2A_download_list)):
                                                    S2A_rest_image_dir=L2_path+'/'+S2A_download_list[L2_S2A_rest_i]
                                                    if os.path.exists(S2A_rest_image_dir):
                                                       shutil.rmtree(S2A_rest_image_dir) 
                                                if os.path.exists(L4_sentinel_2A_merge_envi_file+'.img'):        
                                                    Image_coverage_ratio=Calculate_coverage_ratio(image_path=L4_sentinel_2A_merge_envi_file+'.img',vector_path=Single_shape_name_dir) 
                                                    print(str(L4_sentinel_2A_merge_envi_file+'.img'),str(Image_coverage_ratio))
                                                    if Image_coverage_ratio>=0.90:  
                                                        Image_coverage_ratio_total=max(Image_coverage_ratio_total,Image_coverage_ratio)   
                                        ##########################################################################################    
                                        if len(Sentinel2B_1C_rhos_files_array)>0:
                                            List_for_S2B_stack = list()
                                            Band_names_S2B_for_stack = list()
                                            Band_label_in_array = ["B2", "B3", "B4", "B8", "B11"]
                                            Band_label_out_array = ["492", "560", "665", "833", "1614"]
                                            for List_S3B_i in range(0,len(Band_label_in_array)):
                                                List_band_label = Band_label_in_array[List_S3B_i]
                                                List_out_band_label = Band_label_out_array[List_S3B_i]
                                                if os.path.exists(Sentinel2B_rhos_merge_file + '/' + List_band_label + '.img'):
                                                    List_for_S2B_stack.append(Sentinel2B_rhos_merge_file + '/' + List_band_label + '.img')
                                                    Band_names_S2B_for_stack.append('rhot_' + List_out_band_label)
                                            if len(List_for_S2B_stack)>0:
                                                with rasterio.open(List_for_S2B_stack[0]) as raster_info:
                                                    raster_meta = raster_info.meta.copy()
                                                raster_meta.update(count=len(List_for_S2B_stack))
                                                raster_meta.update(driver="ENVI")
                                                with rasterio.open(Sentinel2B_rhos_merge_file + '.img', "w", **raster_meta) as dst:
                                                    for id, layer in enumerate(List_for_S2B_stack, start=1):
                                                        with rasterio.open(layer) as src:
                                                            for _, window in src.block_windows(1):
                                                                block = src.read(1, window=window)
                                                                dst.write(block, id, window=window)
                                                                del block
                                                        _trim_memory()
                                                hdr_out_txt = open(Sentinel2B_rhos_merge_file + '.hdr', 'r')
                                                hdr_lines = hdr_out_txt.readlines()
                                                hdr_band_pos=0
                                                for hdr_i in range(0,len(hdr_lines)):
                                                    if 'band names =' in hdr_lines[hdr_i]:
                                                        hdr_band_pos = hdr_i
                                                with open(Sentinel2B_rhos_merge_file + '.txt', 'w') as g:
                                                    for hdr_j in range(0, hdr_band_pos):
                                                        g.write(hdr_lines[hdr_j])
                                                    g.write('band names = {' + '\n')
                                                    for bname_i in range(0, len(Band_names_S2B_for_stack)):
                                                        bname = Band_names_S2B_for_stack[bname_i]
                                                        if bname_i < len(Band_names_S2B_for_stack) - 1:
                                                            g.write(bname + ',' + '\n')
                                                        else:
                                                            g.write(bname + '}')
                                                g.close()
                                                L4_sentinel_2B_merge_envi_file=(L3_path+'/'+os.path.basename(Sentinel2B_rhos_merge_file))
                                                if os.path.exists(Sentinel2B_rhos_merge_file+'.hdr'):
                                                    os.remove(Sentinel2B_rhos_merge_file+'.hdr')
                                                if os.path.exists(Sentinel2B_rhos_merge_file+'.img.aux.xml'):
                                                    os.remove(Sentinel2B_rhos_merge_file+'.img.aux.xml')
                                                if os.path.exists(Sentinel2B_rhos_merge_file+'.txt'):
                                                    shutil.move(Sentinel2B_rhos_merge_file+'.txt',Sentinel2B_rhos_merge_file + '.hdr')
                                                if os.path.exists(Sentinel2B_rhos_merge_file+'.img'):
                                                    shutil.move(Sentinel2B_rhos_merge_file+'.img',L4_sentinel_2B_merge_envi_file + '.img')
                                                if os.path.exists(Sentinel2B_rhos_merge_file+'.hdr'):
                                                    shutil.move(Sentinel2B_rhos_merge_file+'.hdr',L4_sentinel_2B_merge_envi_file + '.hdr')
                                                raster_info = None
                                                raster_meta = None
                                                dst = None
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.img'):
                                                    Obtain_water_mask_from_image_by_index(L4_sentinel_2B_merge_envi_file+'.img',Temp_path,Single_shape_name_dir,water_rwi_threshold='otsu')
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.img'):
                                                    prepare_raster_for_arcmap(L4_sentinel_2B_merge_envi_file+'.img',resampling="nearest")
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.img'):
                                                    Create_PNG_file_of_Sentinel2(L4_sentinel_2B_merge_envi_file+'.img')
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.temp.PNG'): 
                                                    Convert_png_transparent(L4_sentinel_2B_merge_envi_file+'.temp.PNG',L4_sentinel_2B_merge_envi_file+'.PNG',bg_color=(0,0,0)) 
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.img') and os.path.exists(Single_shape_name_dir) and os.path.exists(L4_sentinel_2B_merge_envi_file+'.Swater.shp'):
                                                    Make_three_panel_overlay(raster_path=L4_sentinel_2B_merge_envi_file+'.img',vector1_path=Single_shape_name_dir,vector2_path=L4_sentinel_2B_merge_envi_file+'.Swater.shp',output_path=L4_sentinel_2B_merge_envi_file+'.JPEG',background_color=(0, 0, 0))   
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.temp.PNG'):
                                                    os.remove(L4_sentinel_2B_merge_envi_file+'.temp.PNG')  
                                                if os.path.exists(Sentinel2B_rhos_merge_file):
                                                    shutil.rmtree(Sentinel2B_rhos_merge_file)    
                                                S2B_all_tuple = project_xy(L4_sentinel_2B_merge_envi_file+'.img') 
                                                S2B_L3_basename=os.path.basename(L4_sentinel_2B_merge_envi_file+'.img')
                                                S2B_L3_Dirname=os.path.dirname(L4_sentinel_2B_merge_envi_file+'.img')
                                                S2B_Txt_path=S2B_L3_Dirname+'/'+S2B_L3_basename[0:len(S2B_L3_basename)-4]+'.txt'
                                                Sensor_type=S2B_L3_basename[0:3]
                                                with open(S2B_Txt_path, 'w') as g:
                                                      g.write('Basename:'+S2B_L3_basename+'\n')
                                                      g.write('Sensortype:'+Sensor_type+'\n')
                                                      g.write('left_top_corner:'+str(S2B_all_tuple[0])+','+str(S2B_all_tuple[1])+'\n')
                                                      g.write('left_bottom_corner:'+str(S2B_all_tuple[2])+','+str(S2B_all_tuple[3])+'\n')
                                                      g.write('right_top_corner:'+str(S2B_all_tuple[4])+','+str(S2B_all_tuple[5])+'\n')
                                                      g.write('right_bottom_corner:'+str(S2B_all_tuple[6])+','+str(S2B_all_tuple[7])+'\n')
                                                      g.write('Source_files_begin'+'\n')
                                                      for S2B_down_sen_file_i in Sentinel2B_1C_rhos_files_array:
                                                          Out_sen_file_name=os.path.basename(S2B_down_sen_file_i)
                                                          g.write(Out_sen_file_name+'\n')   
                                                      g.write('Source_files_end'+'\n')
                                                g.close()  
                                                S2B_all_tuple=None
                                                Band_label_out_array=None 
                                                for L2_S2B_rest_i in range(0,len(S2B_download_list)):
                                                    S2B_rest_image_dir=L2_path+'/'+S2B_download_list[L2_S2B_rest_i]
                                                    if os.path.exists(S2B_rest_image_dir):
                                                       shutil.rmtree(S2B_rest_image_dir) 
                                                if os.path.exists(L4_sentinel_2B_merge_envi_file+'.img'):        
                                                    Image_coverage_ratio=Calculate_coverage_ratio(image_path=L4_sentinel_2B_merge_envi_file+'.img',vector_path=Single_shape_name_dir) 
                                                    print(str(L4_sentinel_2B_merge_envi_file+'.img'),str(Image_coverage_ratio))
                                                    if Image_coverage_ratio>=0.90:  
                                                        Image_coverage_ratio_total=max(Image_coverage_ratio_total,Image_coverage_ratio)           
                                        ##########################################################################################    
                                        if len(Sentinel2C_1C_rhos_files_array)>0:
                                            List_for_S2C_stack = list()
                                            Band_names_S2C_for_stack = list()
                                            Band_label_in_array = ["B2", "B3", "B4", "B8", "B11"]
                                            Band_label_out_array = ["492", "560", "665", "833", "1614"]
                                            for List_S3C_i in range(0,len(Band_label_in_array)):
                                                List_band_label = Band_label_in_array[List_S3C_i]
                                                List_out_band_label = Band_label_out_array[List_S3C_i]
                                                if os.path.exists(Sentinel2C_rhos_merge_file + '/' + List_band_label + '.img'):
                                                    List_for_S2C_stack.append(Sentinel2C_rhos_merge_file + '/' + List_band_label + '.img')
                                                    Band_names_S2C_for_stack.append('rhot_' + List_out_band_label)
                                            if len(List_for_S2C_stack)>0:
                                                with rasterio.open(List_for_S2C_stack[0]) as raster_info:
                                                    raster_meta = raster_info.meta.copy()
                                                raster_meta.update(count=len(List_for_S2C_stack))
                                                raster_meta.update(driver="ENVI")
                                                with rasterio.open(Sentinel2C_rhos_merge_file + '.img', "w", **raster_meta) as dst:
                                                    for id, layer in enumerate(List_for_S2C_stack, start=1):
                                                        with rasterio.open(layer) as src:
                                                            for _, window in src.block_windows(1):
                                                                block = src.read(1, window=window)
                                                                dst.write(block, id, window=window)
                                                                del block
                                                        _trim_memory()
                                                hdr_out_txt = open(Sentinel2C_rhos_merge_file + '.hdr', 'r')
                                                hdr_lines = hdr_out_txt.readlines()
                                                hdr_band_pos=0
                                                for hdr_i in range(0,len(hdr_lines)):
                                                    if 'band names =' in hdr_lines[hdr_i]:
                                                        hdr_band_pos = hdr_i
                                                with open(Sentinel2C_rhos_merge_file + '.txt', 'w') as g:
                                                    for hdr_j in range(0, hdr_band_pos):
                                                        g.write(hdr_lines[hdr_j])
                                                    g.write('band names = {' + '\n')
                                                    for bname_i in range(0, len(Band_names_S2C_for_stack)):
                                                        bname = Band_names_S2C_for_stack[bname_i]
                                                        if bname_i < len(Band_names_S2C_for_stack) - 1:
                                                            g.write(bname + ',' + '\n')
                                                        else:
                                                            g.write(bname + '}')
                                                g.close()
                                                L4_sentinel_2C_merge_envi_file=(L3_path+'/'+os.path.basename(Sentinel2C_rhos_merge_file))
                                                if os.path.exists(Sentinel2C_rhos_merge_file+'.hdr'):
                                                    os.remove(Sentinel2C_rhos_merge_file+'.hdr')
                                                if os.path.exists(Sentinel2C_rhos_merge_file+'.img.aux.xml'):
                                                    os.remove(Sentinel2C_rhos_merge_file+'.img.aux.xml')
                                                if os.path.exists(Sentinel2C_rhos_merge_file+'.txt'):
                                                    shutil.move(Sentinel2C_rhos_merge_file+'.txt',Sentinel2C_rhos_merge_file + '.hdr')
                                                if os.path.exists(Sentinel2C_rhos_merge_file+'.img'):
                                                    shutil.move(Sentinel2C_rhos_merge_file+'.img',L4_sentinel_2C_merge_envi_file + '.img')
                                                if os.path.exists(Sentinel2C_rhos_merge_file+'.hdr'):
                                                    shutil.move(Sentinel2C_rhos_merge_file+'.hdr',L4_sentinel_2C_merge_envi_file + '.hdr')
                                                raster_info = None
                                                raster_meta = None
                                                dst = None
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.img'):
                                                    Obtain_water_mask_from_image_by_index(L4_sentinel_2C_merge_envi_file+'.img',Temp_path,Single_shape_name_dir,water_rwi_threshold='otsu')
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.img'):
                                                    prepare_raster_for_arcmap(L4_sentinel_2C_merge_envi_file+'.img',resampling="nearest")
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.img'):
                                                    Create_PNG_file_of_Sentinel2(L4_sentinel_2C_merge_envi_file+'.img')
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.temp.PNG'): 
                                                    Convert_png_transparent(L4_sentinel_2C_merge_envi_file+'.temp.PNG',L4_sentinel_2C_merge_envi_file+'.PNG',bg_color=(0,0,0)) 
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.img') and os.path.exists(Single_shape_name_dir) and os.path.exists(L4_sentinel_2C_merge_envi_file+'.Swater.shp'):
                                                    Make_three_panel_overlay(raster_path=L4_sentinel_2C_merge_envi_file+'.img',vector1_path=Single_shape_name_dir,vector2_path=L4_sentinel_2C_merge_envi_file+'.Swater.shp',output_path=L4_sentinel_2C_merge_envi_file+'.JPEG',background_color=(0, 0, 0))   
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.temp.PNG'):
                                                    os.remove(L4_sentinel_2C_merge_envi_file+'.temp.PNG')  
                                                if os.path.exists(Sentinel2C_rhos_merge_file):
                                                    shutil.rmtree(Sentinel2C_rhos_merge_file)    
                                                S2C_all_tuple = project_xy(L4_sentinel_2C_merge_envi_file+'.img') 
                                                S2C_L3_basename=os.path.basename(L4_sentinel_2C_merge_envi_file+'.img')
                                                S2C_L3_Dirname=os.path.dirname(L4_sentinel_2C_merge_envi_file+'.img')
                                                S2C_Txt_path=S2C_L3_Dirname+'/'+S2C_L3_basename[0:len(S2C_L3_basename)-4]+'.txt'
                                                Sensor_type=S2C_L3_basename[0:3]
                                                with open(S2C_Txt_path, 'w') as g:
                                                      g.write('Basename:'+S2C_L3_basename+'\n')
                                                      g.write('Sensortype:'+Sensor_type+'\n')
                                                      g.write('left_top_corner:'+str(S2C_all_tuple[0])+','+str(S2C_all_tuple[1])+'\n')
                                                      g.write('left_bottom_corner:'+str(S2C_all_tuple[2])+','+str(S2C_all_tuple[3])+'\n')
                                                      g.write('right_top_corner:'+str(S2C_all_tuple[4])+','+str(S2C_all_tuple[5])+'\n')
                                                      g.write('right_bottom_corner:'+str(S2C_all_tuple[6])+','+str(S2C_all_tuple[7])+'\n')
                                                      g.write('Source_files_begin'+'\n')
                                                      for S2C_down_sen_file_i in Sentinel2C_1C_rhos_files_array:
                                                          Out_sen_file_name=os.path.basename(S2C_down_sen_file_i)
                                                          g.write(Out_sen_file_name+'\n')   
                                                      g.write('Source_files_end'+'\n')
                                                g.close()  
                                                S2C_all_tuple=None
                                                Band_label_out_array=None 
                                                for L2_S2C_rest_i in range(0,len(S2C_download_list)):
                                                    S2C_rest_image_dir=L2_path+'/'+S2C_download_list[L2_S2C_rest_i]
                                                    if os.path.exists(S2C_rest_image_dir):
                                                       shutil.rmtree(S2C_rest_image_dir)    
                                                if os.path.exists(L4_sentinel_2C_merge_envi_file+'.img'):        
                                                    Image_coverage_ratio=Calculate_coverage_ratio(image_path=L4_sentinel_2C_merge_envi_file+'.img',vector_path=Single_shape_name_dir) 
                                                    print(str(L4_sentinel_2C_merge_envi_file+'.img'),str(Image_coverage_ratio))
                                                    if Image_coverage_ratio>=0.90:  
                                                        Image_coverage_ratio_total=max(Image_coverage_ratio_total,Image_coverage_ratio)      
                            if os.path.exists(L2_path):
                                   delete_folder_contents(L2_path)     
                            if Image_coverage_ratio_total>=0.90:
                                   break        
                Best_PNG_out_name_dir = select_and_move_best_png_for_period(
                    l3_path=L3_path,
                    l4_path=L4_path,
                    first_date_str_pixel=First_date_str_pixel,
                    last_date_str_pixel=Last_date_str_pixel,
                    min_absolute_completeness=0.20,
                    min_relative_completeness=0.80,
                )
                _trim_memory()
            if os.path.exists(Temp_path+'/'+'study_shapefile'):
                    shutil.rmtree(Temp_path+'/'+'study_shapefile')
