import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

# 检查 GPU 是否可用于 XGBoost
try:
    import xgboost as xgb
    # 测试 GPU 是否可用
    test_model = xgb.XGBRegressor(device='cuda', n_estimators=1)
    # 实际训练一个微小数据集来确认 GPU 可用（避免参数错误）
    test_model.fit(np.random.rand(10, 2), np.random.rand(10))
    GPU_AVAILABLE = True
    print("检测到 GPU，将启用 CUDA 加速 XGBoost。")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU 不可用或未安装 GPU 版 XGBoost，将使用 CPU 训练。错误: {e}")

# 配置
DATA_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/wuhan_named.csv")
GROUPS_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/spatial_groups.npy")
OUTPUT_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data/Models")
OUTPUT_DIR.mkdir(exist_ok=True)

# 加载数据
df = pd.read_csv(DATA_PATH)
X = df.drop(["x", "y", "LST", "LU_class"], axis=1)
y = df["LST"]
groups = np.load(GROUPS_PATH)
gkf = GroupKFold(n_splits=5)

# 定义超参数搜索空间（与之前相同，但 XGBoost 的 device 不在搜索范围内）
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 1],
        "kernel": ["rbf"]
    }
}

# 初始化模型，XGBoost 基础参数根据 GPU 情况设置
if GPU_AVAILABLE:
    xgb_base = XGBRegressor(random_state=42, device='cuda', verbosity=0)  # 新版推荐使用 device='cuda'
    # 如果需要兼容旧版，也可以用 tree_method='gpu_hist', predictor='gpu_predictor'
    # xgb_base = XGBRegressor(random_state=42, tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0)
else:
    xgb_base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

models = {
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost": xgb_base,
    "SVM": SVR()
}

# 存储每个模型的最佳参数和性能
best_models = {}
cv_results = {}

print("开始模型训练与超参数调优...")
for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"正在训练 {name} 模型...")

    # SVM需要特征标准化
    if name == "SVM":
        print("  对SVM进行特征标准化...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train = X_scaled
        # 保存标准化器
        joblib.dump(scaler, OUTPUT_DIR / f"{name}_scaler.pkl")
    else:
        X_train = X

    # 随机搜索超参数（使用空间交叉验证）
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],
        n_iter=20,
        cv=gkf,
        scoring="r2",
        n_jobs=1 if GPU_AVAILABLE and name == "XGBoost" else -1,  # GPU 训练时 n_jobs 建议设为 1，避免冲突
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y, groups=groups)

    # 保存最佳模型
    best_model = search.best_estimator_
    best_models[name] = best_model
    joblib.dump(best_model, OUTPUT_DIR / f"{name}_best_model.pkl")

    # 记录交叉验证结果
    cv_results[name] = {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_scores": search.cv_results_["mean_test_score"]
    }

    print(f"  最佳R²: {search.best_score_:.4f}")
    print(f"  最佳参数: {search.best_params_}")

# 保存所有结果
joblib.dump(cv_results, OUTPUT_DIR / "cv_results.pkl")
print("\n所有模型训练完成，结果已保存至 Models 目录")