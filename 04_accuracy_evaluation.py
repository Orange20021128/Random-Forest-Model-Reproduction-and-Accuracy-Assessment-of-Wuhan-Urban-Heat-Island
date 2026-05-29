import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# ========== 配置（与简化版03完全一致，无需修改） ==========
DATA_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/wuhan_named.csv")
MODEL_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data/Models")
OUTPUT_DIR = Path("/home/orange/file/pc/pku_sgis/work/Data/Results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== 加载简化版03已生成的结果 ==========
print("="*60)
print("正在加载简化版03的计算结果...")

# 加载交叉验证结果
cv_overall = joblib.load(MODEL_DIR / "cv_overall_results.pkl")
cv_folds = joblib.load(MODEL_DIR / "cv_fold_results.pkl")
# 加载所有样本的真实值和预测值（已完成预测，无需重新计算）
all_y_true = np.load(MODEL_DIR / "all_y_true.npy")
all_y_pred = np.load(MODEL_DIR / "all_y_pred.npy")

print("✅ 结果加载完成，无需重新预测")

# ========== 计算整体精度指标（验证用） ==========
print("\n" + "="*60)
print("计算全样本整体精度...")
overall_r2 = r2_score(all_y_true, all_y_pred)
overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_mae = mean_absolute_error(all_y_true, all_y_pred)

# ========== 生成结果表格 ==========
# 1. 每折详细结果
fold_df = pd.DataFrame(cv_folds)
fold_df.columns = ["折数", "R²", "RMSE", "MAE"]
fold_df.to_csv(OUTPUT_DIR / "rf_cv_fold_results.csv", index=False)

# 2. 整体对比结果
results_df = pd.DataFrame([{
    "模型": "Random Forest (简化版)",
    "R²_均值": cv_overall["R²_Mean"],
    "R²_标准差": cv_overall["R²_Std"],
    "RMSE_均值": cv_overall["RMSE_Mean"],
    "RMSE_标准差": cv_overall["RMSE_Std"],
    "MAE_均值": cv_overall["MAE_Mean"],
    "MAE_标准差": cv_overall["MAE_Std"],
    "全样本R²": overall_r2,
    "全样本RMSE": overall_rmse,
    "全样本MAE": overall_mae
}])
results_df.to_csv(OUTPUT_DIR / "model_accuracy_comparison.csv", index=False)

# ========== 打印最终结果（对齐论文格式） ==========
print("\n" + "="*80)
print("✅ Random Forest 精度评价结果")
print("="*80)
print("\n📊 3折空间交叉验证结果（均值±标准差）:")
print(f"  R²:   {cv_overall['R²_Mean']:.4f} ± {cv_overall['R²_Std']:.4f}")
print(f"  RMSE: {cv_overall['RMSE_Mean']:.4f} ± {cv_overall['RMSE_Std']:.4f}")
print(f"  MAE:  {cv_overall['MAE_Mean']:.4f} ± {cv_overall['MAE_Std']:.4f}")

print("\n🌍 全样本整体精度:")
print(f"  R²:   {overall_r2:.4f}")
print(f"  RMSE: {overall_rmse:.4f}")
print(f"  MAE:  {overall_mae:.4f}")

print("\n📄 论文徐州研究区750m尺度参考结果:")
print("  RandomForest: R²=0.641, RMSE=1.171, MAE≈0.900")

print("\n💾 结果文件已保存:")
print("  - Results/rf_cv_fold_results.csv: 每折详细结果")
print("  - Results/model_accuracy_comparison.csv: 整体精度对比")
print("="*80)
