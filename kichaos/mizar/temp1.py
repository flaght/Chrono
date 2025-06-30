import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 假设 df_full 是您加载的全量数据
df_full = pd.read_feather("your_full_data.feather")
df_full['trade_time'] = pd.to_datetime(df_full['trade_time'])

# --- 1. 定义预测目标 (Label) ---
# 预测目标：未来30分钟的收益率
prediction_horizon = 30
df_full['future_return'] = df_full['close'].shift(-prediction_horizon) / df_full['close'] - 1

# --- 2. 创建分类标签 (Classification Label) ---
# 定义上涨和下跌的阈值，例如0.1%
threshold = 0.001

def create_label(ret):
    if ret > threshold:
        return 1  # 上涨
    elif ret < -threshold:
        return -1 # 下跌
    else:
        return 0  # 震荡

df_full['label'] = df_full['future_return'].apply(create_label)

# --- 3. 清理数据 ---
# 删除最后N行，因为它们的 'future_return' 和 'label' 是 NaN
df_full.dropna(subset=['future_return', 'label'], inplace=True)
df_full['label'] = df_full['label'].astype(int)

# 识别所有88个特征列
all_feature_columns = [col for col in df_full.columns if col not in ['trade_time', 'code', 'close', 'future_return', 'label']]
print(f"找到 {len(all_feature_columns)} 个特征。")


#### 找出那些与未来收益率关系最稳定的因子。
# --- 1. 按天计算IC值 ---
daily_ics = []
# 使用 Spearman 秩相关，对异常值不敏感
def calculate_daily_ic(daily_data):
    daily_ic_row = {'date': daily_data['trade_time'].iloc[0].date()}
    for factor in all_feature_columns:
        # 确保数据量足够且不全为常数
        if len(daily_data[factor].dropna()) > 5 and daily_data[factor].nunique() > 1:
            corr, _ = spearmanr(daily_data[factor], daily_data['future_return'], nan_policy='omit')
            daily_ic_row[factor] = corr if not np.isnan(corr) else 0
        else:
            daily_ic_row[factor] = 0
    return daily_ic_row

# 使用 groupby().apply() 高效计算
daily_ic_list = df_full.groupby(df_full['trade_time'].dt.date).apply(calculate_daily_ic)
df_daily_ic = pd.DataFrame(list(daily_ic_list))

print("每日IC计算完成:")
print(df_daily_ic.head())


# --- 2. 计算ICIR并进行筛选 ---
ic_summary = []
for factor in all_feature_columns:
    ic_series = df_daily_ic[factor]
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()

    # 计算 ICIR，处理分母为0的情况
    icir = mean_ic / (std_ic + 1e-8)

    ic_summary.append({
        'factor': factor,
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir
    })

df_ic_summary = pd.DataFrame(ic_summary)
df_ic_summary['abs_icir'] = df_ic_summary['icir'].abs()

# 按照ICIR绝对值降序排列
df_ic_summary = df_ic_summary.sort_values(by='abs_icir', ascending=False).reset_index(drop=True)

print("\n因子ICIR分析报告:")
print(df_ic_summary)

# --- 3. 【筛选】选择ICIR最高的20个因子作为候选 ---
# 这里的20是一个超参数，可以根据实际情况调整 (例如15-30)
num_icir_selected = 20
stable_factors = df_ic_summary.head(num_icir_selected)['factor'].tolist()

print(f"\n经过ICIR筛选后，选出 {len(stable_factors)} 个稳定因子:")
print(stable_factors)


## 寻找组合预测能力最强的。
# --- 1. 准备LGBM的训练和验证数据 ---
# 使用所有可用的数据来训练LGBM，因为它只是一个特征筛选工具
# 如果数据量巨大，可以只用RL的训练集部分
X = df_full[stable_factors]
y = df_full['label']

# 将标签从[-1, 0, 1]映射到[0, 1, 2]，以符合LGBM的要求
y_mapped = y.map({-1: 0, 0: 1, 1: 2})

# 从数据中划分出一部分用于早停验证
X_train, X_val, y_train, y_val = train_test_split(
    X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)

# --- 2. 定义LGBM参数并训练 ---
lgbm_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1
}

model = lgb.LGBMClassifier(**lgbm_params)

print("\n开始训练LightGBM模型进行特征筛选...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
print("LightGBM模型训练完成。")

# --- 3. 获取并展示特征重要性 ---
df_feature_importance = pd.DataFrame({
    'feature': model.booster_.feature_name(),
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nLightGBM特征重要性排序:")
print(df_feature_importance)

# 绘制特征重要性图
plt.figure(figsize=(10, 8))
lgb.plot_importance(model, max_num_features=num_icir_selected, height=0.8)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.show()

# --- 4. 【最终筛选】选择重要性最高的 8 个因子 ---
# 这里的8是一个超参数，可以根据重要性断崖来决定 (例如5-10)
num_final_factors = 8
final_selected_factors = df_feature_importance.head(num_final_factors)['feature'].tolist()

print(f"\n最终筛选出的 {num_final_factors} 个核心特征为:")
print(final_selected_factors)

# 如果树的深度（max_depth）很浅，模型可能更偏好那些有直接线性关系的强特征，而忽略了需要更深层次组合才能发挥作用的弱特征。
# 如果学习率（learning_rate）过高，模型可能会过早地在几个强特征上收敛，导致其他特征的重要性被低估。
# 正则化参数（reg_alpha, reg_lambda）的强度也会影响模型的特征选择偏好。