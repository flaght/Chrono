### original_factors_train_
import os, pdb, copy, time, math
from functools import partial
import pdb, argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import *


def load_datasets(variant):
    dirs = os.path.join(
        base_path, variant['method'], 'normal', variant['g_instruments'],
        'rollings', 'normal_factors3',
        "{0}_{1}".format(variant['categories'], variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['swindow'])))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_filename = os.path.join(
            dirs, "original_factors_train_{0}.feather".format(i))
        val_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        test_filename = os.path.join(
            dirs, "original_factors_test_{0}.feather".format(i))

        train_data = pd.read_feather(train_filename)
        val_data = pd.read_feather(val_filename)
        test_data = pd.read_feather(test_filename)

        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data, test_data)
    return data_mapping


def create_data(variant):
    data_mapping = load_datasets(variant)
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_data, val_data, _ = data_mapping[i]
        total_data = pd.concat([train_data, val_data], axis=0)
        pdb.set_trace()
        train1 = pd.read_feather("temp/normal_train_24.feather")
        val1 = pd.read_feather("temp/normal_val_24.feather")
        total_data = pd.concat([train1, val1], axis=0)
        #test1 = pd.read_feather("temp/normal_test_24.feather")
        ## 定义上涨 下跌的阈值 例如:
        threshold = COST_MAPPING[variant['code']]['buy'] * 1.3
        pdb.set_trace()

        ## lgbm 从0开始的非负整数
        def create_label(ret):
            if ret > threshold:
                return 2  # 上涨
            elif ret < -threshold:
                return 1  # 下跌
            else:
                return 0  # 震荡

        total_data['label'] = total_data['nxt1_ret']

    total_data['label'] = total_data['nxt1_ret'].apply(create_label)
    # --- 3. 清理数据 ---
    # 删除最后N行，因为它们的 'future_return' 和 'label' 是 NaN
    total_data.dropna(subset=['nxt1_ret', 'label'], inplace=True)
    total_data['label'] = total_data['label'].astype(int)
    return total_data


# 使用 Spearman 秩相关，对异常值不敏感
def calculate_ic(total_data, features):
    ic_row = []
    for feature in features:
        print(feature)
        if len(total_data[feature].dropna()
               ) > 5 and total_data[feature].nunique() > 1:
            corr, _ = spearmanr(total_data[feature],
                                total_data['nxt1_ret'],
                                nan_policy='omit')
            corr = corr if not np.isnan(corr) else 0
            ic_row.append({'name': feature, 'corr': corr})
    return pd.DataFrame(ic_row)


def icir_features(total_data, icir_threshold=0.5):
    rolling_window = 60
    features = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code'] +
        ['price', 'close', 'nxt1_ret', 'label']
    ]

    # --- 1. 计算滚动窗口的时序IC ---
    # 定义滚动窗口的大小
    # 传入的是原始的收益率，spearmanr 函数也帮你“在幕后”完成了排名的工作。
    # Pandas的 rolling() 功能非常强大且经过高度优化，但它没有内置的 .spearman() 方法。
    # 内置的、速度极快的 .corr() 方法，但这个方法计算的是皮尔逊相关系数，它处理的是原始值，对异常值很敏感。
    # 极其慢：df[factor].rolling(60).apply(lambda x: spearmanr(x, corresponding_returns).correlation)
    ranked_features = total_data[features].rank(method='first')
    ranked_return = total_data['nxt1_ret'].rank(method='first')

    rolling_ic = pd.DataFrame(index=total_data.index)
    for factor in features:
        rolling_correlation = ranked_features[factor].rolling(
            window=rolling_window,
            min_periods=int(rolling_window * 0.5)  # 至少需要30个点才能开始计算
        ).corr(ranked_return)
        rolling_ic[factor] = rolling_correlation

    rolling_ic.dropna(inplace=True)
    print("\n滚动IC计算完成。查看滚动IC序列的概况:")
    # 显示每个因子IC序列的统计信息
    print(rolling_ic.describe())

    # --- 2. 基于滚动IC序列计算ICIR ---
    # 现在的逻辑是：对每个因子产生的整个IC时间序列，计算其均值和标准差
    ic_summary = []
    for factor in features:
        print(factor)
        ic_series = rolling_ic[factor]
        # 确保有足够的IC值来计算
        if len(ic_series) > 0:
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            # 计算 ICIR，处理分母为0的情况
            icir = mean_ic / (std_ic + 1e-8)
        else:
            mean_ic = 0
            std_ic = 0
            icir = 0

        ic_summary.append({
            'factor':
            factor,
            'mean_ic':
            mean_ic,
            'std_ic':
            std_ic,
            'icir':
            icir,
            # 这里的 positive_ic_rate 衡量的是在所有滚动窗口中，IC为正的窗口占比
            'positive_ic_rate':
            (ic_series > 0).mean() if len(ic_series) > 0 else 0,
        })
    ic_summary = pd.DataFrame(ic_summary)
    ic_summary['abs_icir'] = ic_summary['icir'].abs()

    # 按照ICIR绝对值降序排列
    ic_summary = ic_summary.sort_values(by='abs_icir',
                                        ascending=False).reset_index(drop=True)

    print("\n单品种因子ICIR分析报告 (基于滚动IC, Top 20):")
    print(ic_summary.head(20))

    # --- 3. 【筛选】选择ICIR最高的20个因子作为候选 ---
    '''
    |ICIR| < 0.3: 较弱/不稳定。因子信号的稳定性存疑，可能含有大量噪声，需要进一步改进或谨慎使用。
    0.3 <= |ICIR| < 0.5: 可用/良好。这通常被认为是有效因子的入门门槛。因子在大部分时间里提供了有价值的预测信息，但仍有不小的波动。
    0.5 <= |ICIR| < 0.7: 优秀。表明因子具有相当高的稳定性，其预测能力在不同市场环境下都比较可靠。
    |ICIR| >= 0.7: 极好。这种因子非常少见，通常是策略的核心驱动力。
    |ICIR| > 1.0 - 1.5: 好得难以置信 (Too Good to Be True)。当看到如此高的ICIR时，应高度警惕，首先要怀疑的不是找到了“圣杯”，而是数据中可能存在前视偏差 (Lookahead Bias) 或其他类型的过拟合。
    '''
    #num_icir_selected = 20
    #stable_factors = ic_summary.head(num_icir_selected)['factor'].tolist()
    stable_factors = ic_summary[ic_summary['abs_icir']
                                > icir_threshold]  #['factor'].tolist()

    print(f"\n经过滚动ICIR筛选后，选出 {len(stable_factors)} 个稳定因子:")
    print(stable_factors)
    return stable_factors


## 1. 保留重要性高  2.保留ICIR高的
def filter_correlation(total_data, stable_factors, sort_key, threshold=0.7):
    corr_matrix = total_data[stable_factors['factor'].tolist()].corr().abs()
    selected_factors = []
    dropped_factors = []
    for _, row in stable_factors.iterrows():
        factor = row['factor']
        if not selected_factors:
            selected_factors.append(factor)
            print(f"✅ 添加第一个因子: {factor} ({sort_key}: {row[sort_key]:.4f})")
            continue

        # 检查当前因子与已选因子的相关性
        is_highly_correlated = False
        for selected_factor in selected_factors:
            correlation = corr_matrix.loc[factor, selected_factor]
            if correlation > threshold:
                is_highly_correlated = True
                print(f"❌ 丢弃因子: {factor} ({sort_key}: {row[sort_key]:.4f}).")
                print(
                    f"   (原因: 与已选因子 '{selected_factor}' 相关性为 {correlation:.2f} > {threshold})"
                )
                dropped_factors.append(factor)
                break  # 只要和一个高相关，就跳出内层循环

        if not is_highly_correlated:
            selected_factors.append(factor)
            print(f"✅ 添加因子: {factor} (abs_icir: {row['abs_icir']:.4f})")

    return selected_factors, dropped_factors


def lgbm_features(total_data, stable_factors):
    X = total_data[stable_factors]
    y = total_data['label']
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y)

    # --- 2. 定义LGBM参数并训练 ---
    lgbm_params = {
        'objective': 'multiclass',  # 或 'binary'，取决于你的标签是多分类还是二分类
        'num_class': 3,  # 如果是三分类（涨/跌/平），需要指定类别数
        'metric': 'multi_logloss',  # 或 'logloss'/'auc' for binary
        'boosting_type': 'gbdt',

        # --- 核心控制参数 ---
        'n_estimators': 1000,  # 树的数量，设置一个较大的值
        'learning_rate': 0.05,  # 学习率，不宜过大
        'num_leaves': 31,  # 每棵树的叶子节点数，31是经典值
        'max_depth': -1,  # 不限制树的深度

        # --- 正则化与防过拟合 ---
        'reg_alpha': 0.1,  # L1 正则化
        'reg_lambda': 0.1,  # L2 正则化
        'colsample_bytree': 0.8,  # 训练每棵树时随机选择80%的特征
        'subsample': 0.8,  # 训练每棵树时随机选择80%的数据

        # --- 其他 ---
        'n_jobs': -1,  # 使用所有CPU核心
        'seed': 42,  # 随机种子，保证结果可复现
        'verbose': 1  # 关闭冗长的输出
    }
    model = lgb.LGBMClassifier(**lgbm_params)
    print("开始训练LightGBM模型进行特征筛选...")
    model.fit(X_train,
              y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    print("LightGBM模型训练完成。")
    # --- 3. 获取并展示特征重要性 ---
    feature_importance = pd.DataFrame({
        'factor': model.booster_.feature_name(),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nLightGBM特征重要性排序:")
    print(feature_importance)
    return feature_importance


'''
象限	ICIR	LGBM Importance	       特征	                     策略
一	     高	    高	               黄金因子 (Gold Factors)	      【首选】 这些是既稳定又强大的因子，是模型的核心基石。
二	     低	    高	               组合因子 (Ensemble Gems)	      【次选】 单独看可能不稳定，但在模型中与其他因子组合时能发挥巨大作用。它们可能捕捉了重要的非线性关系。
三	     低	    低	               噪声因子 (Noise)	              【剔除】 既不稳定，在复杂模型中也没什么用。应优先剔除。
四	     高	    低	               线性稳定因子 (Linear Anchors)   【备选/观察】 自身与收益率有很强的线性关系，但可能被其他因子包含了信息（共线性），或其价值在复杂的决策树中难以体现。可以保留一部分作为模型的“稳定锚”。
'''


def quadrants(stable_factors):
    pdb.set_trace()
    stable_factors = stable_factors[stable_factors['keep'] == 1].copy()
    stable_factors['icir_rank'] = stable_factors['abs_icir'].rank(
        ascending=False)
    stable_factors['importance_rank'] = stable_factors['importance'].rank(
        ascending=False, method='first')
    # 2. 创建一个综合得分，例如平均排名
    stable_factors['avg_rank'] = (stable_factors['icir_rank'] +
                                  stable_factors['importance_rank']) / 2
    stable_factors = stable_factors.sort_values(by='avg_rank').reset_index(
        drop=True)
    pdb.set_trace()
    # 使用中位数作为划分“高”和“低”的界限，这比设定固定阈值更具适应性
    icir_median = stable_factors['abs_icir'].median()
    importance_median = stable_factors['importance'].median()

    def assign_quadrant(row, icir_median, importance_median):
        is_icir_high = row['abs_icir'] >= icir_median
        is_importance_high = row['importance'] >= importance_median

        if is_icir_high and is_importance_high:
            return 1  #"Gold Factors (高ICIR, 高Importance)"
        elif not is_icir_high and is_importance_high:
            return 2  #"Ensemble Gems (低ICIR, 高Importance)"
        elif not is_icir_high and not is_importance_high:
            return 3  #"Noise (低ICIR, 低Importance)"
        elif is_icir_high and not is_importance_high:
            return 4  #"Linear Anchors (高ICIR, 低Importance)"

    assign_quadrant_with_medians = partial(assign_quadrant,
                                           icir_median=icir_median,
                                           importance_median=importance_median)
    print("--- 综合排名最高的因子 ---")
    print(stable_factors)
    stable_factors['quadrant'] = stable_factors.apply(
        assign_quadrant_with_medians, axis=1)
    stable_factors = stable_factors[[
        'factor', 'quadrant', 'importance', 'icir_rank', 'importance_rank',
        'avg_rank'
    ]]
    ## 当前只选1,2象限
    pdb.set_trace()
    stable_factors = stable_factors[stable_factors['quadrant'] <= 2]
    pdb.set_trace()
    return stable_factors['factor'].tolist()


def run(variant):
    '''
    LGBM看到所有稳定的因子，让它去判断组合的重要性。然后再对LGBM选出的Top N因子进行相关性分析，如果两个高度相关的因子都被选中，我们保留那个LGBM重要性更高的。
    '''
    total_data = create_data(variant=variant)


    sort_key = 'importance'
    ### 选择稳定性
    stable_factors = icir_features(total_data=total_data, icir_threshold=0.4)
    ### 寻找重要性
    import_factors = lgbm_features(
        total_data=total_data,
        stable_factors=stable_factors['factor'].tolist())

    merge_factors = stable_factors.merge(import_factors,
                                         on=['factor'],
                                         how='left').fillna(0)
    pdb.set_trace()

    merge_factors = merge_factors.sort_values(by=[sort_key], ascending=False)
    #### 剔除相关性
    selected_factors, dropped_factors = filter_correlation(
        total_data=total_data, sort_key=sort_key, stable_factors=merge_factors)
    merge_factors['keep'] = merge_factors['factor'].isin(
        selected_factors).astype(int)
    pdb.set_trace()
    selected_factors = quadrants(stable_factors=merge_factors)
    print(selected_factors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_days',
                        type=int,
                        default=60,
                        help='Training days')  ## 训练天数
    parser.add_argument('--val_days',
                        type=int,
                        default=10,
                        help='Validation days')  ## 验证天数

    parser.add_argument('--freq',
                        type=int,
                        default=10,
                        help='Frequency of training')  ## 多少个周期训练一次

    parser.add_argument('--method',
                        type=str,
                        default='aicso4',
                        help='Method name')  ## 方法

    parser.add_argument('--code', type=str, default='RB', help='Code')  ## 代码

    parser.add_argument('--g_instruments',
                        type=str,
                        default='rbb',
                        help='Instruments')  ## 标的

    parser.add_argument('--categories',
                        type=str,
                        default='o2o',
                        help='Categories')  ## 类别

    parser.add_argument('--horizon',
                        type=int,
                        default=1,
                        help='Prediction horizon')  ## 预测周期

    parser.add_argument('--nc', type=int, default=2,
                        help='Standard method')  ## 标准方式

    parser.add_argument('--swindow',
                        type=int,
                        default=60,
                        help='Rolling window')  ## 滚动窗口

    parser.add_argument('--g_start_pos',
                        type=int,
                        default=24,
                        help='Start position')  ## 开始位置

    parser.add_argument('--g_max_pos',
                        type=int,
                        default=1,
                        help='Max position')  ## 最大位置

    args = parser.parse_args()
    run(variant=vars(args))
