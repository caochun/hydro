#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Sentinel-2 query/download manifests from Buffer_province shapefiles.

This script does not download SAFE data. It prepares two planning layers:

1. Shape index: one row per buffer shapefile, including bbox, area, and
   intersecting Sentinel-2 MGRS tiles.
2. Product manifest: one row per shape/product relation plus a product-level
   deduplicated table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import time
from numbers import Integral
from collections import defaultdict
from pathlib import Path

import requests
import shapefile
from pyproj import Transformer
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform
from shapely.strtree import STRtree

from sentinel2_download import CATALOGUE_URL, _cloud_cover, _product_date, _product_tile_name


DEFAULT_ROOT = Path("data_download/Buffer_province")
DEFAULT_SINGLE_ROOT = Path("data_download/single")
DEFAULT_TILE_INDEX = Path("data_download/Base/sentinel_2_index_shapefile.shp")
DEFAULT_OUT_DIR = Path("data_download/downloads/sentinel2_plan")
DEFAULT_SINGLE_OUT_DIR = Path("data_download/downloads/sentinel2_single_plan")
DEFAULT_INDEX_CSV = DEFAULT_OUT_DIR / "buffer_shape_index.csv"
DEFAULT_TILE_CSV = DEFAULT_OUT_DIR / "buffer_shape_tiles.csv"
DEFAULT_PRODUCT_CSV = DEFAULT_OUT_DIR / "sentinel2_products_unique.csv"
DEFAULT_SHAPE_PRODUCT_CSV = DEFAULT_OUT_DIR / "shape_product_manifest.csv"
DEFAULT_QUERY_DB = DEFAULT_OUT_DIR / "sentinel2_query_cache.sqlite"


def _iter_shape_paths(root: Path, group: str, layout: str = "buffer", limit: int | None = None):
    paths = sorted(root.rglob("*.shp"))
    count = 0
    for path in paths:
        parts = path.relative_to(root).parts
        if layout == "single":
            if len(parts) != 1:
                continue
        else:
            if len(parts) < 2:
                continue
            if group != "all" and parts[1] != group:
                continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            break


def _reader(path: Path):
    return shapefile.Reader(str(path), encoding="gb2312")


def _close_reader(reader):
    try:
        reader.close()
    except Exception:
        pass


def _read_union_geometry(path: Path):
    reader = _reader(path)
    try:
        geoms = [shapely_shape(item.__geo_interface__) for item in reader.iterShapes()]
    finally:
        _close_reader(reader)
    if not geoms:
        return None
    geom = geoms[0]
    for other in geoms[1:]:
        geom = geom.union(other)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def _load_tile_index(tile_index: Path):
    names = []
    geoms = []
    reader = _reader(tile_index)
    try:
        name_idx = None
        for idx, field in enumerate(reader.fields[1:]):
            if field[0] == "Name":
                name_idx = idx
                break
        if name_idx is None:
            raise ValueError(f"{tile_index} does not contain field Name")
        for sr in reader.iterShapeRecords():
            geom = shapely_shape(sr.shape.__geo_interface__)
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            names.append(str(sr.record[name_idx]).upper())
            geoms.append(geom)
    finally:
        _close_reader(reader)
    tree = STRtree(geoms)
    geom_to_name = {id(geom): name for geom, name in zip(geoms, names)}
    return tree, geom_to_name


def _tree_query(tree: STRtree, geom):
    candidates = tree.query(geom)
    if len(candidates) == 0:
        return []
    first = candidates[0]
    if isinstance(first, Integral):
        return [tree.geometries[int(index)] for index in candidates]
    return list(candidates)


def _bbox(geom):
    xmin, ymin, xmax, ymax = geom.bounds
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width_deg": xmax - xmin,
        "height_deg": ymax - ymin,
    }


def _geo_array_from_bbox(xmin, ymin, xmax, ymax, padding):
    return [
        str(max(float(ymin) - padding, -90)),
        str(min(float(ymax) + padding, 90)),
        str(max(float(xmin) - padding, -180)),
        str(min(float(xmax) + padding, 180)),
    ]


def _build_tile_catalogue_filter(tile, start, end, cloud, product_type):
    tile = str(tile).upper().removeprefix("T")
    product_text = f"MSI{product_type.upper()}"
    cloud_string = (
        "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {cloud} )"
    )
    return (
        f"contains(Name,'{product_text}')"
        f" and contains(Name,'_T{tile}_')"
        f" and {cloud_string}"
        f" and ContentDate/Start gt {start}T00:00:00.000Z"
        f" and ContentDate/Start lt {end}T23:59:59.000Z"
    )


def _query_tile_products(tile, start, end, cloud, product_type, timeout=120):
    query_filter = _build_tile_catalogue_filter(tile, start, end, cloud, product_type)
    response = requests.get(
        CATALOGUE_URL,
        params={"$filter": query_filter, "$top": "1000"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("value", [])


def _init_cache(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tile_queries (
            tile TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            cloud TEXT NOT NULL,
            product_type TEXT NOT NULL,
            status TEXT NOT NULL,
            product_json TEXT,
            error TEXT,
            queried_at TEXT NOT NULL,
            PRIMARY KEY (tile, start_date, end_date, cloud, product_type)
        )
        """
    )
    conn.commit()
    return conn


def _cache_key(args, tile):
    return (
        str(tile).upper().removeprefix("T"),
        args.start,
        args.end,
        str(args.cloud),
        args.product_type.upper(),
    )


def _get_cached(conn, args, tile):
    row = conn.execute(
        """
        SELECT status, product_json, error FROM tile_queries
        WHERE tile=? AND start_date=? AND end_date=? AND cloud=? AND product_type=?
        """,
        _cache_key(args, tile),
    ).fetchone()
    if not row:
        return None
    status, product_json, error = row
    if status != "ok":
        return {"status": status, "products": [], "error": error}
    return {"status": status, "products": json.loads(product_json or "[]"), "error": error}


def _set_cached(conn, args, tile, status, products=None, error=""):
    conn.execute(
        """
        INSERT OR REPLACE INTO tile_queries
        (tile, start_date, end_date, cloud, product_type, status, product_json, error, queried_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (*_cache_key(args, tile), status, json.dumps(products or []), error),
    )
    conn.commit()


def _product_record(product):
    name = str(product.get("Name", ""))
    return {
        "product_id": str(product.get("Id", "")),
        "product_name": name,
        "tile": _product_tile_name(name),
        "product_date": _product_date(name),
        "cloud_cover": _cloud_cover(product),
        "content_start": product.get("ContentDate", {}).get("Start", ""),
        "content_end": product.get("ContentDate", {}).get("End", ""),
        "online": product.get("Online", ""),
        "size": product.get("ContentLength", ""),
    }


def build_index(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv = Path(args.index_csv)
    tile_csv = Path(args.tile_csv)

    tree, geom_to_name = _load_tile_index(Path(args.tile_index))
    area_transform = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform

    index_fields = [
        "shape_key",
        "province",
        "group",
        "subpart",
        "shape_id",
        "shape_file",
        "feature_count",
        "area_km2",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width_deg",
        "height_deg",
        "tile_count",
        "tiles",
    ]
    tile_fields = ["shape_key", "province", "group", "shape_id", "tile"]

    total = 0
    with index_csv.open("w", newline="", encoding="utf-8") as index_f, tile_csv.open(
        "w", newline="", encoding="utf-8"
    ) as tile_f:
        index_writer = csv.DictWriter(index_f, fieldnames=index_fields)
        tile_writer = csv.DictWriter(tile_f, fieldnames=tile_fields)
        index_writer.writeheader()
        tile_writer.writeheader()

        for path in _iter_shape_paths(Path(args.root), args.group, args.layout, args.limit):
            rel = path.relative_to(Path(args.root))
            parts = rel.parts
            shape_id = path.stem
            if args.layout == "single":
                province = ""
                group = "single"
                subpart = ""
                shape_key = shape_id
            else:
                province = parts[0] if len(parts) > 0 else ""
                group = parts[1] if len(parts) > 1 else ""
                subpart = parts[2] if len(parts) > 3 else ""
                shape_key = str(rel.with_suffix(""))

            geom = _read_union_geometry(path)
            if geom is None or geom.is_empty:
                continue
            candidates = _tree_query(tree, geom)
            tiles = sorted(
                {
                    geom_to_name[id(tile_geom)]
                    for tile_geom in candidates
                    if tile_geom.intersects(geom)
                }
            )
            bounds = _bbox(geom)
            area_km2 = transform(area_transform, geom).area / 1_000_000
            reader = _reader(path)
            try:
                feature_count = len(reader)
            finally:
                _close_reader(reader)

            row = {
                "shape_key": shape_key,
                "province": province,
                "group": group,
                "subpart": subpart,
                "shape_id": shape_id,
                "shape_file": str(path),
                "feature_count": feature_count,
                "area_km2": f"{area_km2:.6f}",
                **{key: f"{value:.10f}" for key, value in bounds.items()},
                "tile_count": len(tiles),
                "tiles": ";".join(tiles),
            }
            index_writer.writerow(row)
            for tile in tiles:
                tile_writer.writerow(
                    {
                        "shape_key": shape_key,
                        "province": province,
                        "group": group,
                        "shape_id": shape_id,
                        "tile": tile,
                    }
                )
            total += 1
            if total % 5000 == 0:
                print(f"indexed {total} shapes", flush=True)

    print(f"完成 shape 索引：{total} 个 shape")
    print(f"索引文件：{index_csv}")
    print(f"shape-tile 明细：{tile_csv}")


def _load_shape_index(index_csv: Path, limit: int | None = None):
    rows = []
    with index_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def query_products(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_rows = _load_shape_index(Path(args.index_csv), args.limit_shapes)
    tile_to_shapes = defaultdict(list)
    for row in shape_rows:
        for tile in row.get("tiles", "").split(";"):
            tile = tile.strip().upper()
            if tile:
                tile_to_shapes[tile].append(row)

    tiles = sorted(tile_to_shapes)
    if args.limit_tiles is not None:
        tiles = tiles[: args.limit_tiles]

    conn = _init_cache(Path(args.cache_db))
    products_by_id = {}
    tile_products = defaultdict(dict)
    query_count = 0

    for index, tile in enumerate(tiles, start=1):
        cached = _get_cached(conn, args, tile)
        if cached is not None and not args.refresh:
            products = cached["products"]
            print(f"[{index}/{len(tiles)}] cache tile={tile} products={len(products)}")
        else:
            try:
                print(f"[{index}/{len(tiles)}] query tile={tile}", flush=True)
                products = _query_tile_products(
                    tile=tile,
                    start=args.start,
                    end=args.end,
                    cloud=args.cloud,
                    product_type=args.product_type,
                    timeout=args.timeout,
                )
                products = [p for p in products if p.get("Id") and p.get("Name")]
                _set_cached(conn, args, tile, "ok", products=products)
                query_count += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:
                _set_cached(conn, args, tile, "error", products=[], error=repr(exc))
                print(f"  查询失败 tile={tile}: {exc}", flush=True)
                products = []
        for product in products:
            product_record = _product_record(product)
            product_tile = product_record["tile"].upper()
            if product_tile != tile:
                continue
            products_by_id[product_record["product_id"]] = product_record
            tile_products[tile][product_record["product_id"]] = product_record

    product_fields = [
        "product_id",
        "product_name",
        "tile",
        "product_date",
        "cloud_cover",
        "content_start",
        "content_end",
        "online",
        "size",
        "shape_count",
    ]
    with Path(args.product_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=product_fields)
        writer.writeheader()
        for product in sorted(products_by_id.values(), key=lambda r: (r["product_date"], r["tile"], r["product_name"])):
            shape_count = len(tile_to_shapes.get(product["tile"].upper(), []))
            writer.writerow({**product, "shape_count": shape_count})

    manifest_fields = [
        "shape_key",
        "province",
        "group",
        "shape_id",
        "tile",
        "product_id",
        "product_name",
        "product_date",
        "cloud_cover",
    ]
    manifest_rows = 0
    with Path(args.shape_product_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for tile in sorted(tile_products):
            for shape in tile_to_shapes[tile]:
                for product in sorted(tile_products[tile].values(), key=lambda r: (r["product_date"], r["product_name"])):
                    writer.writerow(
                        {
                            "shape_key": shape["shape_key"],
                            "province": shape["province"],
                            "group": shape["group"],
                            "shape_id": shape["shape_id"],
                            "tile": tile,
                            "product_id": product["product_id"],
                            "product_name": product["product_name"],
                            "product_date": product["product_date"],
                            "cloud_cover": product["cloud_cover"],
                        }
                    )
                    manifest_rows += 1

    summary = {
        "shape_rows": len(shape_rows),
        "tiles": len(tiles),
        "network_queries": query_count,
        "unique_products": len(products_by_id),
        "shape_product_rows": manifest_rows,
        "start": args.start,
        "end": args.end,
        "cloud": args.cloud,
        "product_type": args.product_type,
        "outputs": {
            "products": str(args.product_csv),
            "shape_product_manifest": str(args.shape_product_csv),
            "cache_db": str(args.cache_db),
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create Sentinel-2 planning manifests from Buffer_province shapefiles."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Build local shapefile-to-Sentinel-tile index.")
    index.add_argument("--root", default=str(DEFAULT_ROOT))
    index.add_argument("--tile-index", default=str(DEFAULT_TILE_INDEX))
    index.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    index.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    index.add_argument("--tile-csv", default=str(DEFAULT_TILE_CSV))
    index.add_argument(
        "--layout",
        choices=["buffer", "single"],
        default="buffer",
        help="Input layout: Buffer_province province/group folders, or flat data_download/single.",
    )
    index.add_argument("--group", choices=["Total", "Split", "all"], default="Total")
    index.add_argument("--limit", type=int, default=None)
    index.set_defaults(func=build_index)

    query = sub.add_parser("query", help="Query Copernicus by unique tile and write deduped product manifests.")
    query.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    query.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    query.add_argument("--product-csv", default=str(DEFAULT_PRODUCT_CSV))
    query.add_argument("--shape-product-csv", default=str(DEFAULT_SHAPE_PRODUCT_CSV))
    query.add_argument("--summary-json", default=str(DEFAULT_OUT_DIR / "sentinel2_plan_summary.json"))
    query.add_argument("--cache-db", default=str(DEFAULT_QUERY_DB))
    query.add_argument("--start", required=True)
    query.add_argument("--end", required=True)
    query.add_argument("--cloud", default="10")
    query.add_argument("--product-type", choices=["L1C", "L2A"], default="L1C")
    query.add_argument("--limit-shapes", type=int, default=None)
    query.add_argument("--limit-tiles", type=int, default=None)
    query.add_argument("--sleep", type=float, default=0.2)
    query.add_argument("--timeout", type=float, default=120)
    query.add_argument("--refresh", action="store_true")
    query.set_defaults(func=query_products)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
