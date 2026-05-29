import rasterio
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# ========== 配置 ==========
DATA_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data/WuHan")          # 所有 tif 所在文件夹
MASK_PATH = DATA_DIR / "LUCL.tif"
OUT_HDF5 = "/home/orange/file/pc/pku_sgis/work/Data/wuhan.h5"
OUT_CSV  = "/home/orange/file/pc/pku_sgis/work/Data/wuhan.csv"   # 修改为带坐标的最终文件

# 因子文件命名规则（请根据实际情况修改）
# 格式: ai1.tif, ed1.tif, frac1.tif, pland1.tif ... 共8类
def factor_path(name, class_id):
    return DATA_DIR / f"{name}{class_id}.tif"

# 控制因子
CTRL_FILES = {
    "DEM": "DEM.tif",
    "SVF": "SVF.tif",
    "DTG": "DTG.tif",
    "DTW": "DTW.tif"
}

# ========== 1. 生成有效像素掩膜、索引及地理坐标 ==========
print("1. 生成掩膜、有效像素索引及坐标...")
with rasterio.open(MASK_PATH) as src:
    lucl = src.read(1)
    transform = src.transform                # 获取仿射变换参数
    mask = (lucl >= 1) & (lucl <= 8)        # 研究区内
    valid_count = mask.sum()
    print(f"   有效像素数: {valid_count}")

    # 展平后的索引（用于快速提取，行优先顺序）
    flat_indices = np.flatnonzero(mask)
    # 同时保存 LUCL 本身的有效数据
    lucl_valid = lucl[mask]

    # 提取所有有效像素的行列号（与 flat_indices 顺序一致）
    rows, cols = np.where(mask)
    # 行列号转 UTM 坐标（WGS_1984_UTM_Zone_50N）
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    # xs, ys 是与有效像素一一对应的坐标数组

# ========== 2. 创建 HDF5 文件，逐步写入各列 ==========
print("2. 创建 HDF5 数据集...")
with h5py.File(OUT_HDF5, 'w') as f:
    # 先写入掩膜对应的 LUCL 类别（可用于分组分析）
    f.create_dataset("LU_class", data=lucl_valid, dtype=np.int8)

    # 定义一个辅助函数：读取栅格并提取有效像素
    def add_raster_layer(f, layer_name, tif_path, dtype=np.float32):
        print(f"   处理: {layer_name} <- {tif_path.name}")
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            # 处理文件自身的 NoData
            nd = src.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            # 提取有效像素
            valid_data = arr.flat[flat_indices]  # 使用展平索引
        f.create_dataset(layer_name, data=valid_data, dtype=dtype)

    # 2.1 写入因变量 LST
    add_raster_layer(f, "LST", DATA_DIR / "LST.tif")

    # 2.2 写入32个景观因子
    metric_names = ["ai", "ed", "frac", "pland"]
    for class_id in range(1, 9):
        for metric in metric_names:
            layer = f"{metric}_{class_id}"
            path = factor_path(metric, class_id)
            if not path.exists():
                print(f"警告: 文件不存在 {path}，该列将用 NaN 填充")
                f.create_dataset(layer, data=np.full(valid_count, np.nan))
            else:
                add_raster_layer(f, layer, path)

    # 2.3 写入4个控制因子
    for key, filename in CTRL_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"警告: 文件不存在 {path}，该列将用 NaN 填充")
            f.create_dataset(key, data=np.full(valid_count, np.nan))
        else:
            add_raster_layer(f, key, path)

print("   HDF5 文件写入完成。")

# ========== 3. 从 HDF5 生成带坐标的 CSV ==========
print("3. 从 HDF5 读取并导出带坐标的 CSV ...")
with h5py.File(OUT_HDF5, 'r') as f:
    data_dict = {col: f[col][:] for col in f.keys()}
    df = pd.DataFrame(data_dict)

# 添加地理坐标列（与数据顺序一致，均基于同一掩膜提取）
df["x"] = xs
df["y"] = ys

# 删除存在任何 NaN 的行（如果有些因子有缺失）
initial_rows = len(df)
df.dropna(inplace=True)
print(f"   删除 NaN 前: {initial_rows} 行, 后: {len(df)} 行")

# 导出为 CSV
df.to_csv(OUT_CSV, index=False)
print(f"   数据集已保存为: {OUT_CSV}")
print("全部完成。")