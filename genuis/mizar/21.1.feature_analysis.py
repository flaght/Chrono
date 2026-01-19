import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.lsx001 import fetch_times
from kdutils.macro2 import *


from kdutils.tactix import Tactix


def target1(final_data,target_col = 'nxt1_ret_15h'):
    print("\n[2.3] 目标变量分析")
    print("-" * 40)
    pdb.set_trace()
    print(f"  变量名: {target_col}")
    print(f"  非空样本数: {final_data[target_col].notna().sum():,} / {len(final_data):,}")
    print(f"  缺失率: {final_data[target_col].isna().mean()*100:.2f}%")

    target_data = final_data[target_col].dropna()
    print(f"\n  统计量:")
    print(f"    均值: {target_data.mean():.6f}")
    print(f"    标准差: {target_data.std():.6f}")
    print(f"    最小值: {target_data.min():.6f}")
    print(f"    25分位: {target_data.quantile(0.25):.6f}")
    print(f"    中位数: {target_data.median():.6f}")
    print(f"    75分位: {target_data.quantile(0.75):.6f}")
    print(f"    最大值: {target_data.max():.6f}")

    print(f"\n  方向分布:")
    positive_pct = (target_data > 0).mean()
    negative_pct = (target_data < 0).mean()
    zero_pct = (target_data == 0).mean()
    print(f"    正收益: {positive_pct*100:.2f}%")
    print(f"    负收益: {negative_pct*100:.2f}%")
    print(f"    零收益: {zero_pct*100:.2f}%")

def clear1(final_data, target_col = 'nxt1_ret_15h'):
    feature_cols_all = [col for col in final_data.columns if col not in ['trade_time', 'code', target_col]]
    nan_stats = pd.DataFrame({
    'feature': feature_cols_all,
    'nan_count': [final_data[col].isna().sum() for col in feature_cols_all],
    'nan_ratio': [final_data[col].isna().mean() for col in feature_cols_all]
    }).sort_values('nan_ratio', ascending=False)
    
    #final_data = final_data.set_index(['trade_time','code'])
    print("\n" + "=" * 80)
    print("第3步：数据清洗")
    print("=" * 80)

    print("\n【目的】去除无效数据，确保数据质量")
    original_shape = final_data.shape
    print(f"清洗前数据: {original_shape}")

    print("\n[3.1] 处理目标变量缺失值")
    print("-" * 40)
    before_len = len(final_data)
    final_data = final_data.dropna(subset=[target_col])
    after_len = len(final_data)
    print(f"  删除目标变量NaN: {before_len:,} → {after_len:,} (删除{before_len - after_len:,}行)")

    print("\n[3.2] 删除高缺失率特征")
    print("-" * 40)
    nan_threshold = 0.5
    print(f"  阈值: NaN比例 > {nan_threshold*100}%")

    high_nan_cols = nan_stats[nan_stats['nan_ratio'] > nan_threshold]['feature'].tolist()
    print(f"  发现 {len(high_nan_cols)} 个高缺失率特征")

    if len(high_nan_cols) > 0:
        print(f"  删除特征:")
        for i, col in enumerate(high_nan_cols[:5], 1):
            print(f"    {i}. {col} (NaN比例: {nan_stats[nan_stats['feature']==col]['nan_ratio'].values[0]*100:.1f}%)")
        if len(high_nan_cols) > 5:
            print(f"    ... (共{len(high_nan_cols)}个)")
    
        final_data = final_data.drop(columns=high_nan_cols)
        print(f"  ✓ 删除完成")
    
    print("\n[3.3] 删除剩余NaN行")
    print("-" * 40)
    before_len = len(final_data)
    final_data = final_data.dropna()
    after_len = len(final_data)
    print(f"  删除包含NaN的行: {before_len:,} → {after_len:,} (删除{before_len - after_len:,}行)")

    print(f"\n  【说明】因子数据通常不适合填充NaN，因为：")
    print(f"    1. 填充会引入虚假信号")
    print(f"    2. 技术指标的NaN往往表示无法计算（如窗口期初）")
    print(f"    3. 删除NaN更保守但更可靠")

    print("\n[3.5] 时间排序")
    print("-" * 40)
    print(f"  【重要】时间序列预测必须按时间排序！")
    final_data = final_data.sort_values('trade_time').reset_index(drop=True)
    print(f"  ✓ 已按时间升序排序")
    print(f"  时间范围: {final_data['trade_time'].min()} 至 {final_data['trade_time'].max()}")

    print("\n[3.6] 清洗总结")
    print("-" * 40)
    cleaned_shape = final_data.shape
    print(f"  清洗前: {original_shape}")
    print(f"  清洗后: {cleaned_shape}")
    print(f"  样本保留率: {cleaned_shape[0]/original_shape[0]*100:.1f}%")
    print(f"  特征保留率: {cleaned_shape[1]/original_shape[1]*100:.1f}%")
    return final_data

def features1(final_data, feature_cols, exclude_cols, var_threshold=1e-10):
    print("\n" + "=" * 80)
    print("第4步：特征工程")
    print("=" * 80)

    print("\n【目的】筛选有效特征，提升模型性能")
    feature_vars = final_data[feature_cols].var()
    zero_var_features = feature_vars[feature_vars < var_threshold].index.tolist()
    print(f"  方差阈值: {var_threshold}")
    print(f"  零方差特征数: {len(zero_var_features)}")

    if len(zero_var_features) > 0:
        print(f"  删除特征示例:")
        for i, feat in enumerate(zero_var_features[:5], 1):
            print(f"    {i}. {feat}")
        if len(zero_var_features) > 5:
            print(f"    ... (共{len(zero_var_features)}个)")
    
        feature_cols = [f for f in feature_cols if f not in zero_var_features]
        print(f"  ✓ 删除完成")

    print(f"  剩余特征数: {len(feature_cols)}")

    print("\n[4.3] 计算因子IC（Information Coefficient）")
    print("-" * 40)
    print(f"  【说明】IC是因子与目标变量的相关系数，衡量因子的预测能力")
    print(f"  IC越高，因子预测能力越强")

    ic_dict = {}
    print(f"  计算中...")

    for i, col in enumerate(feature_cols, 1):
        if i % 50 == 0:
            print(f"    进度: {i}/{len(feature_cols)}")
    
        try:
            ic = final_data[col].corr(final_data[target_col])
            ic_dict[col] = abs(ic) if not np.isnan(ic) else 0
        except:
            ic_dict[col] = 0

    ic_series = pd.Series(ic_dict).sort_values(ascending=False)

    print(f"\n  IC统计:")
    print(f"    平均IC: {ic_series.mean():.4f}")
    print(f"    中位IC: {ic_series.median():.4f}")
    print(f"    最大IC: {ic_series.max():.4f}")
    print(f"    IC>0.01的因子数: {(ic_series > 0.01).sum()}")
    print(f"    IC>0.03的因子数: {(ic_series > 0.03).sum()}")

    print(f"\n  Top 20 高IC因子:")
    print(ic_series.head(20).to_frame('IC'))
    return ic_dict


def smart1(df, feature_cols, target_col, ic_dict, 
                           corr_threshold=0.98, ic_threshold=0.001):
    print(f"\n  参数设置:")
    print(f"    相关性阈值: {corr_threshold}")
    print(f"    IC阈值: {ic_threshold}")

    X = df[feature_cols]
    print(f"\n  步骤1: 计算特征相关性矩阵...")
    print(f"    矩阵大小: {len(feature_cols)} × {len(feature_cols)}")
    corr_matrix = X.corr().abs()

    print(f"\n  步骤2: 识别高相关特征对...")
    upper_tri = np.triu(corr_matrix.values, k=1)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if upper_tri[i, j] > corr_threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    upper_tri[i, j]
                ))
    
    print(f"    发现 {len(high_corr_pairs)} 个高相关特征对（相关性>{corr_threshold}）")

     # 基于IC筛选
    print(f"\n  步骤3: 基于IC筛选高相关特征...")
    to_drop = set()
    
    for col1, col2, corr_val in high_corr_pairs:
        # 保留IC高的因子
        if ic_dict[col1] < ic_dict[col2]:
            to_drop.add(col1)
        else:
            to_drop.add(col2)
    
    print(f"    删除 {len(to_drop)} 个低IC特征")
    
    remaining_features = [f for f in feature_cols if f not in to_drop]
    print(f"\n  步骤4: 删除低IC因子...")
    low_ic_features = [f for f in remaining_features if ic_dict[f] < ic_threshold]
    print(f"    删除 {len(low_ic_features)} 个IC<{ic_threshold}的因子")
    
    remaining_features = [f for f in remaining_features if f not in low_ic_features]
    
    return remaining_features



def read_data(method, task_id, instruments, period, name):
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "{0}_data.feather".format(name))
    final_data = pd.read_feather(filename)
    return final_data

def fetch_matrix(df, selected_features):
    print("\n[5.1] 提取特征矩阵和目标变量")
    print("-" * 40)

    X = df[selected_features].values
    y = df[target_col].values
    dates = df['trade_time'].values

    print(f"  特征矩阵 X: {X.shape} (样本数 × 特征数)")
    print(f"  目标变量 y: {y.shape}")
    print(f"  时间序列: {len(dates)}")

    print(f"\n  数据类型:")
    print(f"    X: {X.dtype}")
    print(f"    y: {y.dtype}")

    print(f"\n  数据范围:")
    print(f"    X最小值: {X.min():.6f}")
    print(f"    X最大值: {X.max():.6f}")
    print(f"    y最小值: {y.min():.6f}")
    print(f"    y最大值: {y.max():.6f}")


    # 5.2 时间序列划分
    print("\n[5.2] 时间序列划分")
    print("-" * 40)

    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]


    print(f"  训练/测试比例: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print(f"  划分点索引: {split_idx}")

    print(f"\n  [训练集]")
    print(f"    样本数: {len(X_train):,}")
    print(f"    时间范围: {dates_train[0]} 至 {dates_train[-1]}")
    print(f"    时间跨度: {(pd.Timestamp(dates_train[-1]) - pd.Timestamp(dates_train[0])).days} 天")
    print(f"    目标变量统计:")
    print(f"      均值: {y_train.mean():.6f}")
    print(f"      标准差: {y_train.std():.6f}")
    print(f"      正收益比例: {(y_train > 0).mean()*100:.2f}%")

    print(f"\n  [测试集]")
    print(f"    样本数: {len(X_test):,}")
    print(f"    时间范围: {dates_test[0]} 至 {dates_test[-1]}")
    print(f"    时间跨度: {(pd.Timestamp(dates_test[-1]) - pd.Timestamp(dates_test[0])).days} 天")
    print(f"    目标变量统计:")
    print(f"      均值: {y_test.mean():.6f}")
    print(f"      标准差: {y_test.std():.6f}")
    print(f"      正收益比例: {(y_test > 0).mean()*100:.2f}%")

    # 检查训练集和测试集分布差异
    print(f"\n  [分布一致性检查]")
    mean_diff = abs(y_train.mean() - y_test.mean())
    std_ratio = y_test.std() / y_train.std()
    print(f"    均值差异: {mean_diff:.6f}")
    print(f"    标准差比: {std_ratio:.2f}")
    if std_ratio > 1.5 or std_ratio < 0.67:
        print(f"    ⚠️  警告: 测试集波动性与训练集差异较大")
    else:
        print(f"    ✓ 训练集和测试集分布相对一致")

if __name__ == '__main__':
    variant = Tactix().start()
    final_data = read_data(method=variant.method,
                           task_id=variant.task_id, 
                           instruments=variant.instruments, 
                           period=variant.period, 
                           name='final')
    pdb.set_trace()
    target_col = 'nxt1_ret_15h'
    exclude_cols = ['trade_time', 'code', target_col]
    feature_cols = [col for col in final_data.columns if col not in exclude_cols]
    final_data = clear1(final_data=final_data)

    ic_dict = features1(final_data=final_data, feature_cols=feature_cols, 
              exclude_cols=exclude_cols, var_threshold=1e-10)

    remaining_features = smart1(df=final_data, feature_cols=feature_cols, target_col=target_col, ic_dict=ic_dict)

    print(f"\n测试不同相关性阈值的效果:")
    for threshold in [0.99, 0.98, 0.95]:
        selected = smart1(df=final_data, feature_cols=feature_cols, target_col=target_col, ic_dict=ic_dict,
                          corr_threshold=threshold, ic_threshold=0.02)
        reduction_rate = (1 - len(selected)/len(feature_cols)) * 100
        print(f"  阈值{threshold}: {len(feature_cols)} → {len(selected)} ({reduction_rate:.1f}%减少)")


    print(f"\n执行最终筛选（阈值=0.98）:")
    selected_features = smart1(df=final_data, feature_cols=feature_cols, target_col=target_col, ic_dict=ic_dict,
                          corr_threshold=0.98, ic_threshold=0.001)

    print(f"\n[特征筛选总结]")
    print(f"  原始特征数: {len(feature_cols)}")
    print(f"  筛选后特征数: {len(selected_features)}")
    print(f"  保留率: {len(selected_features)/len(feature_cols)*100:.1f}%")


    print("\n" + "=" * 80)
    print("第5步：准备训练数据")
    print("=" * 80)

    print("\n【关键】时间序列预测必须按时间顺序划分数据！")
    print("  ✓ 正确: 前70%训练，后30%测试")
    print("  ✗ 错误: 随机划分（会导致用未来预测过去）")


    fetch_matrix(final_data, selected_features)
