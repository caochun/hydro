#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-2 L1C query and download helpers for Copernicus Data Space."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests
try:
    import geopandas as gpd
except ImportError:
    gpd = None
try:
    import shapefile
except ImportError:
    shapefile = None


CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PRESET_BBOXES = {
    # Nanjing urban area / main city, WGS84 lon/lat: west, south, east, north.
    "nanjing_urban": (118.55, 31.85, 119.15, 32.25),
}
ORIGINAL_PERIODS = [
    ("2015-12-01", "2016-02-29"),
    ("2016-03-01", "2016-05-31"),
    ("2016-06-01", "2016-08-31"),
    ("2016-12-01", "2017-02-28"),
    ("2017-03-01", "2017-05-31"),
    ("2017-06-01", "2017-08-31"),
    ("2017-12-01", "2018-02-28"),
    ("2018-03-01", "2018-05-31"),
    ("2018-06-01", "2018-08-31"),
    ("2018-12-01", "2019-02-28"),
    ("2019-03-01", "2019-05-31"),
    ("2019-06-01", "2019-08-31"),
    ("2019-12-01", "2020-02-29"),
    ("2020-03-01", "2020-05-31"),
    ("2020-06-01", "2020-08-31"),
    ("2020-12-01", "2021-02-28"),
    ("2021-03-01", "2021-05-31"),
    ("2021-06-01", "2021-08-31"),
    ("2021-12-01", "2022-02-28"),
    ("2022-03-01", "2022-05-31"),
    ("2022-06-01", "2022-08-31"),
    ("2022-12-01", "2023-02-28"),
    ("2023-03-01", "2023-05-31"),
    ("2023-06-01", "2023-08-31"),
    ("2023-12-01", "2024-02-29"),
    ("2024-03-01", "2024-05-31"),
    ("2024-06-01", "2024-08-31"),
    ("2024-12-01", "2025-02-28"),
    ("2025-03-01", "2025-05-31"),
    ("2025-06-01", "2025-08-31"),
]


def load_env_file(env_path=None, override=False):
    """Load simple KEY=VALUE lines from .env without adding a dependency."""
    candidate_paths = []
    if env_path is not None:
        candidate_paths.append(Path(env_path))
    else:
        module_dir = Path(__file__).resolve().parent
        candidate_paths.extend([
            module_dir / ".env",
            module_dir.parent / ".env",
        ])
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
        return str(candidate)
    return ""


def get_copernicus_credentials(env_path=None):
    load_env_file(env_path=env_path)
    username = os.environ.get("COPERNICUS_USERNAME")
    password = os.environ.get("COPERNICUS_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "请在 .env 中填写 COPERNICUS_USERNAME 和 COPERNICUS_PASSWORD，"
            "或设置同名环境变量。"
        )
    return username, password


def get_images_by_date(path_list, date):
    target_date = date.replace("-", "")
    image_list = []
    for path in path_list:
        filename = Path(path).name
        match = re.search(r"MSIL1C_(\d{8})T\d{6}", filename)
        if match:
            image_date = match.group(1)
            if image_date == target_date:
                image_list.append(path)
    return sorted(image_list)


def extract_unique_sorted_dates(file_paths):
    date_objects = set()
    pattern = re.compile(r"_MSIL1C_(\d{8})T\d{6}")
    for path in file_paths:
        filename = Path(path).name
        match = pattern.search(filename)
        if not match:
            continue
        date_str = match.group(1)
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            date_objects.add(date_obj)
        except ValueError:
            continue
    if len(date_objects) == 0:
        return []
    months_in_dates = {date_obj.month for date_obj in date_objects}
    month_order_groups = [
        ({12, 1, 2}, [1, 12, 2]),
        ({3, 4, 5}, [4, 3, 5]),
        ({6, 7, 8}, [7, 8, 6]),
    ]
    selected_month_order = None
    for month_group, month_order in month_order_groups:
        if months_in_dates.issubset(month_group):
            selected_month_order = month_order
            break
    if selected_month_order is None:
        sorted_dates = sorted(date_objects)
    else:
        month_rank = {month: rank for rank, month in enumerate(selected_month_order)}
        sorted_dates = sorted(
            date_objects,
            key=lambda date_obj: (month_rank.get(date_obj.month, 999), date_obj),
        )
    return [date_obj.strftime("%Y-%m-%d") for date_obj in sorted_dates]


def Day_path_row_files_can_be_full_test(Path_low_list, Name_array_list):
    Day_path_row_full_label = 0
    if len(Name_array_list) > 0:
        Name_path_row_day_total = []
        for name_i in range(0, len(Name_array_list)):
            Name_pixel = Name_array_list[name_i]
            Name_path_row_pixel = Name_pixel[39:44]
            Name_path_row_day_total.append(Name_path_row_pixel)
        Name_path_row_day_total = list(dict.fromkeys(Name_path_row_day_total))
        if Counter(Name_path_row_day_total) == Counter(Path_low_list):
            Day_path_row_full_label = 1
        else:
            Day_path_row_full_label = 0
    return Day_path_row_full_label


def _shape_bbox_with_padding(shape_file, padding=0.1):
    if shapefile is None:
        raise ImportError("读取 shapefile 需要安装 pyshp：pip install pyshp")
    shape_name = shapefile.Reader(shape_file, encoding="gb2312")
    lon1 = float(shape_name.bbox[0])
    lon2 = float(shape_name.bbox[2])
    lat1 = float(shape_name.bbox[1])
    lat2 = float(shape_name.bbox[3])
    return [
        str(max([lat1 - padding, -90])),
        str(min([lat2 + padding, 90])),
        str(max([lon1 - padding, -180])),
        str(min([lon2 + padding, 180])),
    ]


def _geo_array_from_bbox(bbox, padding=0.0):
    lon1, lat1, lon2, lat2 = [float(v) for v in bbox]
    if lon2 <= lon1 or lat2 <= lat1:
        raise ValueError("bbox 格式应为 west,south,east,north，且 east>west、north>south。")
    return [
        str(max([lat1 - padding, -90])),
        str(min([lat2 + padding, 90])),
        str(max([lon1 - padding, -180])),
        str(min([lon2 + padding, 180])),
    ]


def _date_text(year, month, day):
    return (
        str(int(year)).zfill(4)
        + "-"
        + str(int(month)).zfill(2)
        + "-"
        + str(int(day)).zfill(2)
    )


def _build_catalogue_filter(date_begin, date_end, cloud_value, geo_array):
    footprint_string = (
        "OData.CSC.Intersects(area=geography'SRID=4326;"
        + "POLYGON(("
        + geo_array[2]
        + " "
        + geo_array[0]
        + ","
        + geo_array[2]
        + " "
        + geo_array[1]
        + ","
        + geo_array[3]
        + " "
        + geo_array[1]
        + ","
        + geo_array[3]
        + " "
        + geo_array[0]
        + ","
        + geo_array[2]
        + " "
        + geo_array[0]
        + "))') "
    )
    cloud_string = (
        "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        "and att/OData.CSC.DoubleAttribute/Value le " + str(cloud_value) + " )"
    )
    return (
        "contains(Name,'MSIL1C')"
        + " and "
        + cloud_string
        + " and "
        + footprint_string
        + " and ContentDate/Start gt "
        + date_begin
        + "T00:00:00.000Z"
        + " and ContentDate/Start lt "
        + date_end
        + "T23:59:59.000Z"
    )


def _query_sentinel2_products(date_begin, date_end, cloud_value, shape_file):
    geo_array = _shape_bbox_with_padding(shape_file)
    return _query_sentinel2_products_by_geo_array(date_begin, date_end, cloud_value, geo_array)


def _query_sentinel2_products_by_geo_array(
    date_begin,
    date_end,
    cloud_value,
    geo_array,
    timeout=60,
):
    query_filter = _build_catalogue_filter(date_begin, date_end, cloud_value, geo_array)
    response = requests.get(
        CATALOGUE_URL,
        params={"$filter": query_filter, "$expand": "Attributes", "$top": "1000"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("value", [])


def query_sentinel2_l1c_by_bbox(
    date_begin,
    date_end,
    cloud_value,
    bbox,
    padding=0.0,
    timeout=60,
):
    geo_array = _geo_array_from_bbox(bbox, padding=padding)
    return _query_sentinel2_products_by_geo_array(
        date_begin=date_begin,
        date_end=date_end,
        cloud_value=cloud_value,
        geo_array=geo_array,
        timeout=timeout,
    )


def query_sentinel2_l1c_by_shape_file(
    date_begin,
    date_end,
    cloud_value,
    shape_file,
    padding=0.1,
    timeout=60,
):
    geo_array = _shape_bbox_with_padding(shape_file, padding=padding)
    return _query_sentinel2_products_by_geo_array(
        date_begin=date_begin,
        date_end=date_end,
        cloud_value=cloud_value,
        geo_array=geo_array,
        timeout=timeout,
    )


def query_sentinel2_l1c_by_bbox_periods(
    periods,
    cloud_value,
    bbox,
    padding=0.0,
    timeout=60,
    verbose=False,
):
    products_by_id = {}
    for index, (date_begin, date_end) in enumerate(periods, start=1):
        if verbose:
            print(f"[{index}/{len(periods)}] 查询 {date_begin} 至 {date_end} ...", flush=True)
        products = query_sentinel2_l1c_by_bbox(
            date_begin=date_begin,
            date_end=date_end,
            cloud_value=cloud_value,
            bbox=bbox,
            padding=padding,
            timeout=timeout,
        )
        if verbose:
            print(f"  返回 {len(products)} 个产品。", flush=True)
        for product in products:
            product_id = product.get("Id")
            if product_id:
                products_by_id[str(product_id)] = product
    products = list(products_by_id.values())
    products.sort(key=lambda item: str(item.get("Name", "")))
    return products


def query_sentinel2_l1c_by_shape_file_periods(
    periods,
    cloud_value,
    shape_file,
    padding=0.1,
    timeout=60,
    verbose=False,
):
    products_by_id = {}
    for index, (date_begin, date_end) in enumerate(periods, start=1):
        if verbose:
            print(f"[{index}/{len(periods)}] 查询 {date_begin} 至 {date_end} ...", flush=True)
        products = query_sentinel2_l1c_by_shape_file(
            date_begin=date_begin,
            date_end=date_end,
            cloud_value=cloud_value,
            shape_file=shape_file,
            padding=padding,
            timeout=timeout,
        )
        if verbose:
            print(f"  返回 {len(products)} 个产品。", flush=True)
        for product in products:
            product_id = product.get("Id")
            if product_id:
                products_by_id[str(product_id)] = product
    products = list(products_by_id.values())
    products.sort(key=lambda item: str(item.get("Name", "")))
    return products


def _product_tile_name(product_name):
    return product_name[39:44]


def _filter_products_by_tiles(products, path_low_list):
    ids = []
    names = []
    for product in products:
        product_id = str(product.get("Id", "")).strip()
        product_name = str(product.get("Name", "")).strip()
        if not product_id or not product_name:
            continue
        tile_pixel = _product_tile_name(product_name)
        if tile_pixel in path_low_list:
            ids.append(product_id)
            names.append(product_name)
    return ids, names


def request_download_sentinel_file(
    Sentinel_access_token,
    Sentinel_url,
    Out_zip_name_dir,
    retries=3,
    retry_wait=10,
):
    out_zip_name_dir = str(Out_zip_name_dir)
    Path(out_zip_name_dir).parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {Sentinel_access_token}"}
    url = str(Sentinel_url).strip().strip("'\"")
    part_zip_name_dir = out_zip_name_dir + ".part"
    for attempt in range(1, int(retries) + 1):
        try:
            with requests.Session() as session:
                request_headers = dict(headers)
                resume_size = 0
                if os.path.exists(part_zip_name_dir):
                    resume_size = os.path.getsize(part_zip_name_dir)
                    if resume_size > 0:
                        request_headers["Range"] = f"bytes={resume_size}-"
                session.headers.update(request_headers)
                response = session.get(url, stream=True, timeout=(60, 300))
                if response.status_code not in (200, 206):
                    print(
                        f"Failed to download file. Status code: {response.status_code}",
                        flush=True,
                    )
                    return ""
                if response.status_code == 200 and resume_size > 0:
                    os.remove(part_zip_name_dir)
                    resume_size = 0
                file_mode = "ab" if response.status_code == 206 and resume_size > 0 else "wb"
                with open(part_zip_name_dir, file_mode) as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            shutil.move(part_zip_name_dir, out_zip_name_dir)
            break
        except Exception as exc:
            if attempt >= int(retries):
                raise
            print(
                f"下载中断，{retry_wait} 秒后重试 "
                f"({attempt}/{retries})：{exc}",
                flush=True,
            )
            time.sleep(float(retry_wait))
    if os.path.exists(out_zip_name_dir):
        return out_zip_name_dir
    return ""


def _get_copernicus_access_token(Copernicus_username, Copernicus_password):
    data = {
        "grant_type": "password",
        "username": Copernicus_username,
        "password": Copernicus_password,
        "client_id": "cdse-public",
    }
    response = requests.post(TOKEN_URL, data=data, timeout=120)
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("Copernicus token response does not contain access_token.")
    return access_token, payload


def write_copernicus_key_file(Copernicus_username, Copernicus_password, Key_file):
    _, payload = _get_copernicus_access_token(
        Copernicus_username,
        Copernicus_password,
    )
    with open(Key_file, "w+", encoding="utf-8") as key_txt:
        key_txt.write(json.dumps(payload, ensure_ascii=False))
    return Key_file


def _unzip_product(zip_path, out_dir):
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    if not zipfile.is_zipfile(zip_path):
        return False
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    return True


def _download_one_product(product_id, product_name, l1_path, username, password):
    safe_dir = Path(l1_path) / product_name
    metadata_path = safe_dir / "MTD_MSIL1C.xml"
    if metadata_path.exists():
        return str(safe_dir)
    token, _ = _get_copernicus_access_token(username, password)
    download_url = f"{DOWNLOAD_URL}({product_id})/$value"
    zip_path = Path(l1_path) / f"{product_name}.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
    request_download_sentinel_file(token, download_url, zip_path)
    if zip_path.exists() and _unzip_product(zip_path, l1_path):
        if metadata_path.exists():
            return str(safe_dir)
    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception:
            pass
    return ""


def Download_Sentinel2_using_IGA(
    Date_Begin_Year,
    Date_Begin_Month,
    Date_Begin_Day,
    Date_End_Year,
    Date_End_Month,
    Date_End_Day,
    Cloud_value,
    L1_path,
    Shape_file,
    Path_low_list,
    Copernicus_username,
    Copernicus_password,
):
    image_list_out = []
    Path(L1_path).mkdir(parents=True, exist_ok=True)
    date_begin = _date_text(Date_Begin_Year, Date_Begin_Month, Date_Begin_Day)
    date_end = _date_text(Date_End_Year, Date_End_Month, Date_End_Day)
    products = _query_sentinel2_products(date_begin, date_end, Cloud_value, Shape_file)
    id_array_list, name_array_list = _filter_products_by_tiles(products, Path_low_list)
    day_path_row_test_label = Day_path_row_files_can_be_full_test(
        Path_low_list,
        name_array_list,
    )
    if day_path_row_test_label != 1:
        return image_list_out
    for esa_image_id, esa_image_name in zip(id_array_list, name_array_list):
        try:
            downloaded = _download_one_product(
                esa_image_id,
                esa_image_name,
                L1_path,
                Copernicus_username,
                Copernicus_password,
            )
        except Exception:
            time.sleep(60)
            downloaded = _download_one_product(
                esa_image_id,
                esa_image_name,
                L1_path,
                Copernicus_username,
                Copernicus_password,
            )
        if downloaded:
            image_list_out.append(downloaded)
    return image_list_out


def download_sentinel2_l1c_by_bbox(
    date_begin,
    date_end,
    cloud_value,
    bbox,
    out_dir,
    username=None,
    password=None,
    padding=0.0,
    max_products=None,
    dry_run=False,
):
    products = query_sentinel2_l1c_by_bbox(
        date_begin=date_begin,
        date_end=date_end,
        cloud_value=cloud_value,
        bbox=bbox,
        padding=padding,
    )
    products = [
        product
        for product in products
        if product.get("Id") and product.get("Name")
    ]
    products.sort(key=lambda item: str(item.get("Name", "")))
    if max_products is not None:
        products = products[: int(max_products)]
    if dry_run:
        return []
    if username is None or password is None:
        username, password = get_copernicus_credentials()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    downloaded = []
    for product in products:
        product_path = _download_one_product(
            product_id=str(product["Id"]),
            product_name=str(product["Name"]),
            l1_path=out_dir,
            username=username,
            password=password,
        )
        if product_path:
            downloaded.append(product_path)
    return downloaded


def download_sentinel2_l1c_by_bbox_periods(
    periods,
    cloud_value,
    bbox,
    out_dir,
    username=None,
    password=None,
    padding=0.0,
    max_products=None,
    dry_run=False,
):
    products = query_sentinel2_l1c_by_bbox_periods(
        periods=periods,
        cloud_value=cloud_value,
        bbox=bbox,
        padding=padding,
    )
    products = [
        product
        for product in products
        if product.get("Id") and product.get("Name")
    ]
    if max_products is not None:
        products = products[: int(max_products)]
    if dry_run:
        return []
    if username is None or password is None:
        username, password = get_copernicus_credentials()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    downloaded = []
    for product in products:
        product_path = _download_one_product(
            product_id=str(product["Id"]),
            product_name=str(product["Name"]),
            l1_path=out_dir,
            username=username,
            password=password,
        )
        if product_path:
            downloaded.append(product_path)
    return downloaded


def download_sentinel2_l1c_by_shape_file(
    date_begin,
    date_end,
    cloud_value,
    shape_file,
    out_dir,
    username=None,
    password=None,
    padding=0.1,
    max_products=None,
    dry_run=False,
):
    products = query_sentinel2_l1c_by_shape_file(
        date_begin=date_begin,
        date_end=date_end,
        cloud_value=cloud_value,
        shape_file=shape_file,
        padding=padding,
    )
    products = [
        product
        for product in products
        if product.get("Id") and product.get("Name")
    ]
    products.sort(key=lambda item: str(item.get("Name", "")))
    if max_products is not None:
        products = products[: int(max_products)]
    if dry_run:
        return []
    if username is None or password is None:
        username, password = get_copernicus_credentials()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    downloaded = []
    for product in products:
        product_path = _download_one_product(
            product_id=str(product["Id"]),
            product_name=str(product["Name"]),
            l1_path=out_dir,
            username=username,
            password=password,
        )
        if product_path:
            downloaded.append(product_path)
    return downloaded


def download_sentinel2_l1c_by_shape_file_periods(
    periods,
    cloud_value,
    shape_file,
    out_dir,
    username=None,
    password=None,
    padding=0.1,
    max_products=None,
    dry_run=False,
):
    products = query_sentinel2_l1c_by_shape_file_periods(
        periods=periods,
        cloud_value=cloud_value,
        shape_file=shape_file,
        padding=padding,
    )
    products = [
        product
        for product in products
        if product.get("Id") and product.get("Name")
    ]
    if max_products is not None:
        products = products[: int(max_products)]
    if dry_run:
        return []
    if username is None or password is None:
        username, password = get_copernicus_credentials()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    downloaded = []
    for product in products:
        product_path = _download_one_product(
            product_id=str(product["Id"]),
            product_name=str(product["Name"]),
            l1_path=out_dir,
            username=username,
            password=password,
        )
        if product_path:
            downloaded.append(product_path)
    return downloaded


def Find_Sentinel2_using_IGA(
    Date_Begin_Year,
    Date_Begin_Month,
    Date_Begin_Day,
    Date_End_Year,
    Date_End_Month,
    Date_End_Day,
    Cloud_value,
    L1_path,
    Shape_file,
    Path_low_list,
):
    Path(L1_path).mkdir(parents=True, exist_ok=True)
    date_begin = _date_text(Date_Begin_Year, Date_Begin_Month, Date_Begin_Day)
    date_end = _date_text(Date_End_Year, Date_End_Month, Date_End_Day)
    products = _query_sentinel2_products(date_begin, date_end, Cloud_value, Shape_file)
    _, name_array_list = _filter_products_by_tiles(products, Path_low_list)
    if Day_path_row_files_can_be_full_test(Path_low_list, name_array_list) >= 0:
        return list(name_array_list)
    return []


def get_date_list(begin_date, end_date):
    dates = []
    dt = datetime.strptime(begin_date, "%Y-%m-%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt += timedelta(days=1)
        date = dt.strftime("%Y-%m-%d")
    return dates


def Get_path_row_of_sentine2_by_shapefile(Path_row_file, Shape_file, Out_overlay_file):
    if gpd is None:
        raise ImportError("按 shapefile 叠加筛选 tile 需要安装 geopandas。")
    path_row_list = []
    gdf1 = None
    gdf2 = None
    gdf_inter = None
    try:
        gdf1 = gpd.read_file(Path_row_file, encoding="gb2312")
        gdf2 = gpd.read_file(Shape_file, encoding="gb2312")
        if gdf1.crs != gdf2.crs:
            gdf2 = gdf2.to_crs(gdf1.crs)
        gdf_inter = gpd.overlay(gdf1, gdf2, how="intersection")
        gdf_inter.to_file(
            filename=Out_overlay_file,
            driver="ESRI Shapefile",
            encoding="gb2312",
        )
        if "Name" in gdf_inter.columns:
            path_row_list = gdf_inter["Name"].dropna().astype(str).tolist()
        else:
            print("字段 Name 不存在，当前字段有：", list(gdf_inter.columns))
    except Exception as e:
        print("error:", e)
    finally:
        del gdf1, gdf2, gdf_inter
    return path_row_list


def _parse_bbox(text):
    if text in PRESET_BBOXES:
        return PRESET_BBOXES[text]
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox 应为 west,south,east,north，或使用预设 nanjing_urban。"
        )
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox 四个值必须是数字。") from exc


def _cloud_cover(product):
    for attribute in product.get("Attributes", []):
        if attribute.get("Name") == "cloudCover":
            return attribute.get("Value")
    return None


def _product_date(product_name):
    match = re.search(r"_MSIL1C_(\d{8})T\d{6}", str(product_name))
    if not match:
        return ""
    return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")


def _parse_csv_values(text):
    values = []
    for part in str(text).split(","):
        value = part.strip()
        if value:
            values.append(value)
    return values


def _filter_products(products, tiles=None, product_date=None):
    if tiles:
        wanted_tiles = {str(tile).strip().upper().removeprefix("T") for tile in tiles}
        products = [
            product
            for product in products
            if _product_tile_name(str(product.get("Name", ""))).upper() in wanted_tiles
        ]
    if product_date:
        wanted_date = str(product_date).strip()
        products = [
            product
            for product in products
            if _product_date(str(product.get("Name", ""))) == wanted_date
        ]
    return products


def _print_products(products):
    if not products:
        print("没有查询到符合条件的 Sentinel-2 L1C 产品。")
        return
    for index, product in enumerate(products, start=1):
        name = product.get("Name", "")
        cloud = _cloud_cover(product)
        tile = _product_tile_name(str(name)) if name else ""
        print(f"{index:03d} tile={tile} cloud={cloud} name={name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L1C SAFE products from Copernicus Data Space."
    )
    parser.add_argument(
        "--bbox",
        type=_parse_bbox,
        default=PRESET_BBOXES["nanjing_urban"],
        help=(
            "WGS84 bbox as west,south,east,north. "
            "Default: nanjing_urban = 118.55,31.85,119.15,32.25"
        ),
    )
    parser.add_argument(
        "--shape-file",
        help=(
            "Use a shapefile extent as the query area. "
            "When set, this overrides --bbox."
        ),
    )
    parser.add_argument(
        "--shape-padding",
        type=float,
        default=0.1,
        help="Padding in degrees around --shape-file bbox, default 0.1.",
    )
    parser.add_argument("--start", help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--use-original-periods",
        action="store_true",
        help=(
            "Use the original script periods: 2015-12-01 to 2025-08-31, "
            "winter/spring/summer only."
        ),
    )
    parser.add_argument("--cloud", default="10", help="Max cloud cover, default 10.")
    parser.add_argument(
        "--out-dir",
        default="data_download/downloads/nanjing_urban",
        help="Output directory for SAFE products.",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Limit number of products to download/query output.",
    )
    parser.add_argument(
        "--tiles",
        type=_parse_csv_values,
        help="Comma-separated Sentinel-2 tiles to keep, e.g. 47SNA,47SNB,47SPA,47SPB.",
    )
    parser.add_argument(
        "--product-date",
        help="Keep only products acquired on this date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only query and print products; do not download.",
    )
    args = parser.parse_args(argv)
    if args.use_original_periods:
        if args.shape_file:
            products = query_sentinel2_l1c_by_shape_file_periods(
                periods=ORIGINAL_PERIODS,
                cloud_value=args.cloud,
                shape_file=args.shape_file,
                padding=args.shape_padding,
                verbose=True,
            )
        else:
            products = query_sentinel2_l1c_by_bbox_periods(
                periods=ORIGINAL_PERIODS,
                cloud_value=args.cloud,
                bbox=args.bbox,
                verbose=True,
            )
    else:
        if not args.start or not args.end:
            parser.error("--start and --end are required unless --use-original-periods is set.")
        if args.shape_file:
            products = query_sentinel2_l1c_by_shape_file(
                date_begin=args.start,
                date_end=args.end,
                cloud_value=args.cloud,
                shape_file=args.shape_file,
                padding=args.shape_padding,
            )
        else:
            products = query_sentinel2_l1c_by_bbox(
                date_begin=args.start,
                date_end=args.end,
                cloud_value=args.cloud,
                bbox=args.bbox,
            )
    products = [
        product
        for product in products
        if product.get("Id") and product.get("Name")
    ]
    products = _filter_products(
        products,
        tiles=args.tiles,
        product_date=args.product_date,
    )
    products.sort(key=lambda item: str(item.get("Name", "")))
    if args.max_products is not None:
        products = products[: args.max_products]
    print(f"查询到 {len(products)} 个产品。")
    _print_products(products)
    if args.dry_run:
        return 0
    username, password = get_copernicus_credentials()
    downloaded = []
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for product in products:
        product_path = _download_one_product(
            product_id=str(product["Id"]),
            product_name=str(product["Name"]),
            l1_path=args.out_dir,
            username=username,
            password=password,
        )
        if product_path:
            downloaded.append(product_path)
            print(f"已下载：{product_path}")
    print(f"完成，成功下载 {len(downloaded)} 个产品。")
    return 0


__all__ = [
    "Day_path_row_files_can_be_full_test",
    "Download_Sentinel2_using_IGA",
    "Find_Sentinel2_using_IGA",
    "Get_path_row_of_sentine2_by_shapefile",
    "ORIGINAL_PERIODS",
    "download_sentinel2_l1c_by_bbox",
    "download_sentinel2_l1c_by_bbox_periods",
    "download_sentinel2_l1c_by_shape_file",
    "download_sentinel2_l1c_by_shape_file_periods",
    "extract_unique_sorted_dates",
    "get_copernicus_credentials",
    "get_date_list",
    "get_images_by_date",
    "load_env_file",
    "query_sentinel2_l1c_by_bbox",
    "query_sentinel2_l1c_by_bbox_periods",
    "query_sentinel2_l1c_by_shape_file",
    "query_sentinel2_l1c_by_shape_file_periods",
    "request_download_sentinel_file",
    "write_copernicus_key_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
