#!/usr/bin/env python3
"""
批量提取太湖区域水体（algal bloom 目录，3 景 UTM51N）。
对每景分别跑 UNetSmall 和 SMP 模型，再融合输出最终矢量。
"""
import sys
from pathlib import Path

DATA_DIR = Path("/Users/chun/Develop/hydro/data/algal bloom")
RUNS_DIR = Path("/Users/chun/Develop/hydro/runs")
OUT_DIR  = Path("/Users/chun/Develop/hydro/water_out_taihu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAIHU_SCENES = [
    "S2B_MSIL1C_20250701",
    "S2C_MSIL1C_20250616",
    "S2C_MSIL1C_20250706",
]

sys.path.insert(0, str(Path(__file__).parent))
import extract_water_unet as ewu
import ensemble as ens


def extract(stem: str, model_file: str):
    ewu.IMAGE_PATH = DATA_DIR / f"{stem}.img"
    ewu.MODEL_PATH = RUNS_DIR / model_file
    ewu.OUT_DIR    = OUT_DIR
    print(f"\n{'='*55}\n[提取] {stem} / {model_file}\n{'='*55}")
    ewu.main()


def fuse(stem: str):
    ens.SCENE_STEM = stem
    ens.OUT_DIR    = OUT_DIR
    ens.IMAGE_PATH = DATA_DIR / f"{stem}.img"
    ens.FUSION     = "max"
    print(f"\n{'='*55}\n[融合] {stem}\n{'='*55}")
    ens.main()


for stem in TAIHU_SCENES:
    if not (DATA_DIR / f"{stem}.img").exists():
        print(f"[WARN] 不存在: {stem}，跳过")
        continue
    extract(stem, "best_model.pt")
    extract(stem, "best_model_smp.pt")
    fuse(stem)

print("\n[完成] 结果保存至", OUT_DIR)
