# S2 水体提取工具

基于 NDWI / MNDWI 从 Sentinel-2 ENVI 格式影像提取水体，输出矢量和预览图。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install rasterio fiona shapely numpy Pillow
```

> 依赖系统级 GDAL。macOS 用 `brew install gdal` 安装。

## 运行

```bash
source .venv/bin/activate
python extract_water_ndwi.py
```

修改脚本顶部配置区后直接运行，无命令行参数。

## 配置参数

| 参数 | 说明 |
|---|---|
| `IMAGE_PATH` | 输入 `.img` 文件路径 |
| `OUT_DIR` | 输出目录 |
| `BAND_B02/03/04/08/11` | 波段索引，留 `None` 从 `.hdr` 自动检测 |
| `IMAGE_SCALE` | 反射率缩放因子，默认 10000 |
| `NDWI_THRESH` / `MNDWI_THRESH` | 水体判定阈值，默认 0.0，调大=更保守 |
| `CLOUD_B02_MAX` | 云过滤阈值（B02 反射率），默认 0.20，调小=更严格 |
| `MIN_AREA_M2` | 最小水体面积（m²），过滤噪声小斑块 |
| `USE_MNDWI` | 是否启用 MNDWI（需要有 B11/SWIR 波段） |

## 波段自动检测

脚本从 `.hdr` 的 `band names` 字段匹配以下名称：

| 波段 | 匹配名 | 用途 |
|---|---|---|
| B02 | `rhos_492` | Blue，云过滤 / RGB 底图 |
| B03 | `rhos_560` | Green，NDWI / MNDWI 分子 |
| B04 | `rhos_665` | Red，RGB 底图 |
| B08 | `rhos_833` | NIR，NDWI 分母 |
| B11 | `rhos_1614` | SWIR1，MNDWI 分母（可选） |

没有 B11 时自动跳过 MNDWI，仅用 NDWI。

## 输出文件

| 文件 | 说明 |
|---|---|
| `*_water.shp/shx/dbf/prj` | 水体矢量面，含 `area_m2` 属性 |
| `*_visual.png` | 真彩色底图 + 蓝色水体叠加预览 |
| `*_ndwi.img/.hdr` | NDWI 连续值栅格 |
| `*_mndwi.img/.hdr` | MNDWI 连续值栅格（有 B11 时输出） |
| `*_water.img/.hdr` | 二值水体掩膜栅格（0/1） |

交付给客户只需 `*_water.shp` 系列（4 个文件）+ `*_visual.png`。

## 已处理数据

| 影像 | 时间 | 波段数 | 方法 | 最小面积 | 面要素数 |
|---|---|---|---|---|---|
| S2A_MSIL1C_20250701 | 2025-07-01 | 16（前5为角度） | NDWI\|MNDWI | 1000 m² | 5464 |
| S2B_MSIL1C_20260202 | 2026-02-02 | 4 | NDWI | 100000 m²（0.1 km²） | 1015 |
