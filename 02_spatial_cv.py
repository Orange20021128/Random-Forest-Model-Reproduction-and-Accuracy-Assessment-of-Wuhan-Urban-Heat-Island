import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
# import matplotlib.pyplot as plt
from pathlib import Path

# 配置
DATA_PATH = Path("/home/orange/file/pc/pku_sgis/work/Data/wuhan_named.csv")
N_CLUSTERS = 100  # 论文中使用100个地理簇
N_FOLDS = 5       # 5折交叉验证

# 加载数据
df = pd.read_csv(DATA_PATH)
X = df.drop(["x", "y", "LST", "LU_class"], axis=1)  # 特征
y = df["LST"]                                      # 因变量
coords = df[["x", "y"]]                            # 坐标
# groups = None                                      # 后续存储簇标签

print("正在进行空间聚类生成地理块...")
# 使用K-means对坐标进行聚类，生成空间块
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
groups = kmeans.fit_predict(coords)

print(f"生成了 {N_CLUSTERS} 个地理簇，每个簇平均样本数: {len(df)/N_CLUSTERS:.1f}")

# 初始化空间块交叉验证
gkf = GroupKFold(n_splits=N_FOLDS)

# 验证交叉验证的分组正确性
print("\n交叉验证分组验证:")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    # 确保训练集和测试集没有重叠的簇
    assert len(np.intersect1d(train_groups, test_groups)) == 0, "空间块交叉验证分组错误！"
    print(f"  Fold {fold+1}: 训练集 {len(train_idx)} 样本，测试集 {len(test_idx)} 样本")

# 保存分组结果（后续模型训练使用）
np.save(Path("/home/orange/file/pc/pku_sgis/work/Data/spatial_groups.npy"), groups)
print("\n空间分组结果已保存至 spatial_groups.npy")
