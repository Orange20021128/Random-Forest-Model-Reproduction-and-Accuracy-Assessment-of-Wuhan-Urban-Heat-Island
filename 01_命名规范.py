import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data")   # 根据实际情况修改
INPUT_CSV  = BASE_DIR / "wuhan.csv"      # 上一步生成的带坐标数据集
OUTPUT_CSV = BASE_DIR / "wuhan_named.csv"       # 列名规范化后的输出文件

# ========== 类别和指数映射 ==========
CLASS_MAP = {
    1: "WT",  # 水体
    2: "GS",  # 绿地
    3: "AL",  # 农业用地
    4: "PF",  # 公共设施用地
    5: "CL",  # 商业用地
    6: "RL",  # 居住用地
    7: "IM",  # 工业用地
    8: "TL"   # 交通用地
}

METRIC_MAP = {
    "ai": "AI",      # 聚集度指数
    "ed": "ED",      # 边缘密度
    "frac": "FRAC",  # 分形维数（复杂度）
    "pland": "PLAND" # 景观百分比（覆盖率）
}

# ========== 读取数据 ==========
print(f"读取数据：{INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# ========== 重命名特征列 ==========
def normalize_col(col_name):
    """
    将 'metric_classid' 格式的列名转换为 'ClassName_Index'
    例如 'pland_2' -> 'GS_PLAND'
    """
    parts = col_name.split("_")
    if len(parts) == 2 and parts[0] in METRIC_MAP:
        metric = parts[0]
        class_id = int(parts[1])
        class_abbr = CLASS_MAP.get(class_id, f"L{class_id}")  # 若类别未知则保留数字
        metric_abbr = METRIC_MAP[metric]
        return f"{class_abbr}_{metric_abbr}"
    return col_name  # 非景观因子列保持不变

# 应用重命名
new_columns = [normalize_col(col) for col in df.columns]
df.columns = new_columns

# ========== 打印重命名示例 ==========
old_landscape_cols = [col for col in df.columns if any(k in col for k in METRIC_MAP.values())]
print(f"规范化后的景观指数列（示例，共{len(old_landscape_cols)}列）:")
print(old_landscape_cols[:10], "..." if len(old_landscape_cols) > 10 else "")

# ========== 保存结果 ==========
df.to_csv(OUTPUT_CSV, index=False)
print(f"列名规范化完成，已保存至：{OUTPUT_CSV}")