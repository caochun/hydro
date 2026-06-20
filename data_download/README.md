# 源数据下载管理

这个目录目前放 Sentinel-2 源数据下载相关代码。

| 文件 | 用途 |
|---|---|
| `sentinel2_download.py` | Copernicus Data Space 查询、认证、下载、解压 Sentinel-2 L1C SAFE 数据 |
| `sentinel2_plan.py` | 从 `Buffer_province` 批量生成 shapefile 索引、Sentinel tile 索引、产品去重 manifest；不下载 SAFE |
| `experiment_external_priors.py` | 为 `data/实验数据` 匹配 JRC GSW、ESA WorldCover、Dynamic World 外部水体先验 |
| `Process_Sentinel2_images_for_waters_02_RWI_MNDWI_fallback.py` | 原完整水体处理流程，下载函数已改为从 `sentinel2_download.py` 导入 |

运行完整流程前，把 Copernicus 账号密码写入项目根目录 `.env`，也可以写入 `data_download/.env`：

```dotenv
COPERNICUS_USERNAME=你的账号
COPERNICUS_PASSWORD=你的密码
```

`.env` 已加入 git 忽略规则，不应提交真实账号密码。

## 下载南京市区 Sentinel-2 L1C

`sentinel2_download.py` 默认使用南京市区 bbox：

```text
118.55,31.85,119.15,32.25
```

先只查询、不下载：

```bash
python3 data_download/sentinel2_download.py \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --cloud 10 \
  --dry-run
```

正式下载到默认目录 `data_download/downloads/nanjing_urban`：

```bash
python3 data_download/sentinel2_download.py \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --cloud 10
```

限制只下载前 1 个产品：

```bash
python3 data_download/sentinel2_download.py \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --cloud 10 \
  --max-products 1
```

如需自定义范围：

```bash
python3 data_download/sentinel2_download.py \
  --bbox 118.6,31.9,119.0,32.2 \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --dry-run
```

也可以直接用 shapefile 的外包矩形作为查询范围。下面用青海湖缓冲区示例，默认会在 shapefile bbox 外再加 `0.1` 度 padding：

```bash
./.venv/bin/python data_download/sentinel2_download.py \
  --shape-file data_download/Buffer_province/青海省/Total/19414.shp \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --cloud 10 \
  --out-dir data_download/downloads/qinghai_lake_19414 \
  --dry-run
```

确认候选产品后正式下载：

```bash
./.venv/bin/python data_download/sentinel2_download.py \
  --shape-file data_download/Buffer_province/青海省/Total/19414.shp \
  --start 2025-06-01 \
  --end 2025-06-30 \
  --cloud 10 \
  --out-dir data_download/downloads/qinghai_lake_19414
```

沿用原完整脚本的时间范围，也就是 2015-12-01 到 2025-08-31 的冬/春/夏季窗口：

```bash
./.venv/bin/python data_download/sentinel2_download.py \
  --use-original-periods \
  --cloud 10 \
  --dry-run
```

确认候选产品后正式下载：

```bash
./.venv/bin/python data_download/sentinel2_download.py \
  --use-original-periods \
  --cloud 10
```

## 批量下载前生成计划表

不要按每个 shapefile 分别下载 Sentinel 数据。推荐先生成索引和 manifest：

```text
Buffer_province shapefile -> Sentinel tile -> Copernicus product -> product_id 去重
```

第 1 步：离线生成 `Total` shapefile 到 Sentinel-2 tile 的索引，不访问 Copernicus：

```bash
./.venv/bin/python data_download/sentinel2_plan.py index --group Total
```

输出：

```text
data_download/downloads/sentinel2_plan/buffer_shape_index.csv
data_download/downloads/sentinel2_plan/buffer_shape_tiles.csv
```

第 2 步：按唯一 Sentinel tile 查询 Copernicus 元数据，生成 shape-product 明细和 product 去重表。访问 Copernicus 时建议显式 no-proxy：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='*' no_proxy='*' \
./.venv/bin/python data_download/sentinel2_plan.py query \
  --start 2025-06-01 \
  --end 2025-08-31 \
  --cloud 10 \
  --product-type L1C \
  --sleep 0.2
```

输出：

```text
data_download/downloads/sentinel2_plan/sentinel2_products_unique.csv
data_download/downloads/sentinel2_plan/shape_product_manifest.csv
data_download/downloads/sentinel2_plan/sentinel2_query_cache.sqlite
data_download/downloads/sentinel2_plan/sentinel2_plan_summary.json
```

`sentinel2_products_unique.csv` 是后续下载队列的基础；同一个 SAFE 产品只需要下载一次，多个 shapefile 可以共用本地缓存。

## 实验数据外部水体先验

对 `data/实验数据` 下的 `*_Swater.shp`，可用 JRC GSW occurrence/seasonality 与 ESA WorldCover 采样生成统计表：

```bash
./.venv/bin/python data_download/experiment_external_priors.py stats
```

输出：

```text
data_download/downloads/experiment_external_water/experiment_external_prior_stats.csv
data_download/downloads/experiment_external_water/experiment_external_prior_summary_by_sample.csv
```

Dynamic World 是 Google Earth Engine ImageCollection，不是普通静态 GeoTIFF tile。首次使用前需要安装并认证 Earth Engine：

```bash
./.venv/bin/python -m pip install earthengine-api
./.venv/bin/earthengine authenticate
```

认证后下载实验区域 Dynamic World water/label GeoTIFF。访问 Earth Engine 时建议显式 no-proxy：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='*' no_proxy='*' \
./.venv/bin/python data_download/experiment_external_priors.py dw-download
```

如果只想先生成待下载区域清单：

```bash
./.venv/bin/python data_download/experiment_external_priors.py dw-manifest
```
