import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import time

# ========== 配置（保持你原有路径不变） ==========
DATA_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/wuhan_named.csv")
GROUPS_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/spatial_groups.npy")
OUTPUT_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data/Models")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== 内存与速度优化参数 ==========
N_ESTIMATORS = 300
MAX_DEPTH = 25
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
N_JOBS = 12
N_FOLDS = 5
RANDOM_STATE = 42

# ========== 加载数据 ==========
print("=" * 60)
print("正在加载数据集...")
start_time = time.time()
df = pd.read_csv(DATA_PATH)
groups = np.load(GROUPS_PATH)

# 保留坐标、特征、标签
xy = df[["x", "y"]].copy()  # 完整坐标
X = df.drop(["x", "y", "LST", "LU_class"], axis=1)
y = df["LST"]

print(f"数据集大小: {X.shape[0]} 样本，{X.shape[1]} 特征")
print(f"空间分组数: {len(np.unique(groups))} 个地理簇")
print(f"数据加载耗时: {time.time() - start_time:.2f} 秒")

# ========== 初始化模型 ==========
print("\n" + "=" * 60)
print("初始化Random Forest模型...")
print(f"模型参数: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_split={MIN_SAMPLES_SPLIT}")

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=1
)

# ========== 空间块交叉验证 ==========
print("\n" + "=" * 60)
print("开始5折空间块交叉验证...")
gkf = GroupKFold(n_splits=N_FOLDS)

fold_results = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    fold_start = time.time()
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    fold_results.append({
        "fold": fold + 1,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    })
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold + 1} 结果: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Fold {fold + 1} 耗时: {time.time() - fold_start:.2f} 秒")

# ========== 计算整体结果 ==========
print("\n" + "=" * 60)
print("交叉验证完成，计算整体结果...")
results_df = pd.DataFrame(fold_results)
overall_results = {
    "R²_Mean": results_df["r2"].mean(),
    "R²_Std": results_df["r2"].std(),
    "RMSE_Mean": results_df["rmse"].mean(),
    "RMSE_Std": results_df["rmse"].std(),
    "MAE_Mean": results_df["mae"].mean(),
    "MAE_Std": results_df["mae"].std()
}

# ========== 训练最终全量模型 ==========
print("\n" + "=" * 60)
print("训练最终全量模型（用于后续GeoShapley）...")
final_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_LEAF,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features="sqrt",
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=1
)
final_model.fit(X, y)

# ========== 关键修改：打包模型 + 坐标 + 特征信息一起保存 ==========
print("\n" + "=" * 60)
print("正在保存模型（内含x/y坐标、特征名、全量数据）...")

# 把所有需要GeoShapley的信息全部打包
model_package = {
    "model": final_model,  # 随机森林模型
    "xy_coords": xy.values,  # 全部样本 x,y 坐标
    "feature_names": X.columns.tolist(),  # 特征名
    "X_data": X.values,  # 全量特征
    "y_true": y.values  # 真实LST
}

# 保存打包后的模型
joblib.dump(model_package, OUTPUT_DIR / "RandomForest_with_coords.pkl", compress=3)

# 保留你原来的其他保存不变
joblib.dump(fold_results, OUTPUT_DIR / "cv_fold_results.pkl")
joblib.dump(overall_results, OUTPUT_DIR / "cv_overall_results.pkl")
np.save(OUTPUT_DIR / "all_y_true.npy", np.array(all_y_true))
np.save(OUTPUT_DIR / "all_y_pred.npy", np.array(all_y_pred))

# ========== 输出最终结果 ==========
print("\n" + "=" * 80)
print("✅ 模型训练完成（已内置坐标信息）")
print("=" * 80)
print(f"{'指标':<10} {'均值':<12} {'标准差':<12}")
print("-" * 80)
print(f"R²:       {overall_results['R²_Mean']:.4f} ± {overall_results['R²_Std']:.4f}")
print(f"RMSE:     {overall_results['RMSE_Mean']:.4f} ± {overall_results['RMSE_Std']:.4f}")
print(f"MAE:      {overall_results['MAE_Mean']:.4f} ± {overall_results['MAE_Std']:.4f}")
print("=" * 80)

print("\n📦 新模型文件：RandomForest_with_coords.pkl")
print("内含：随机森林 + 全部x/y坐标 + 特征名 + 全量特征数据，直接可用GeoShapley")
print(f"\n⏱️ 总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
