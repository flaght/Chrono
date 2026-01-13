import os, pdb
import pandas as pd
import numpy as np

from lib.lsx001 import fetch_times
from kdutils.macro2 import *


def analyze_feature(feature_importance):
    # 累积重要性
    cumulative_importance = feature_importance.cumsum() / feature_importance.sum()
    analysis = {
        'total_features': len(feature_importance),
        'max_importance': float(feature_importance.iloc[0]),
        'min_importance': float(feature_importance.iloc[-1]),
        'mean_importance': float(feature_importance.mean()),
        'median_importance': float(feature_importance.median()),
        'cumulative_importance': cumulative_importance.to_dict()
    }
    # 累积覆盖率分析
    coverage_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    coverage_analysis = {}
    for threshold in coverage_thresholds:
        n_features = (cumulative_importance <= threshold).sum() + 1
        coverage_analysis[f'top_{int(threshold*100)}pct'] = {
            'n_features': int(n_features),
            'coverage': float(cumulative_importance.iloc[min(n_features-1, len(cumulative_importance)-1)])
        }
    analysis['coverage_analysis'] = coverage_analysis

    # 找到肘部点（重要性急剧下降的位置）
    importance_diff = feature_importance.diff().abs()
    elbow_idx = importance_diff.idxmax()
    elbow_position = feature_importance.index.get_loc(elbow_idx)
    analysis['elbow_point'] = {
        'position': elbow_position + 1,
        'feature': elbow_idx,
        'importance': float(feature_importance.iloc[elbow_position]),
        'coverage': float(cumulative_importance.iloc[elbow_position])
    }

    return analysis

def show_results(analysis_results, name):
    print(f"\n{'='*60}")
    print(f"特征选择分析报告 - {name}")
    print(f"{'='*60}")
    print(f"总特征数量: {analysis_results['total_features']}")
    print(f"最大重要性: {analysis_results['max_importance']:.6f}")
    print(f"最小重要性: {analysis_results['min_importance']:.2e}")
    print(f"平均重要性: {analysis_results['mean_importance']:.6f}")

    # 显示累积重要性分析
    print(f"\n--- 累积重要性分析 ---")
    coverage_analysis = analysis_results['coverage_analysis']
    for key, info in coverage_analysis.items():
        pct = key.split('_')[1]
        print(f"前{info['n_features']:3d}个特征覆盖{pct}%的重要性")

    # 肘部点分析
    elbow_info = analysis_results['elbow_point']
    print(f"\n--- 肘部点分析 ---")
    print(f"肘部位置: 第{elbow_info['position']}个特征")
    print(f"肘部特征: {elbow_info['feature']}")
    print(f"肘部覆盖率: {elbow_info['coverage']:.1%}")

def selectio_feature(feature_importance, method: str = 'top_k',  **kwargs):
    pdb.set_trace()
    # 强制按重要性降序排列
    feature_importance = feature_importance.abs().sort_values(ascending=False)

    # 防止全 0 导致的除零错误
    total_importance = feature_importance.sum()
    if total_importance == 0:
        cumulative_importance = pd.Series(0, index=feature_importance.index)
    else:
        cumulative_importance = feature_importance.cumsum() / total_importance

    selected_features = []
    coverage = 0.0

    if method == 'top_k':
        k = kwargs.get('k', 50)
        k = min(k, len(feature_importance))
        if k > 0:
            selected_features = feature_importance.head(k).index.tolist()
            coverage = float(cumulative_importance.iloc[k-1])

    elif method == 'threshold':
        threshold = kwargs.get('threshold', 0.001)

        mask = feature_importance >= threshold
        selected_features = feature_importance[mask].index.tolist()
        
        # 严谨计算覆盖率：被选特征总分 / 总分
        if total_importance > 0:
            coverage = feature_importance[mask].sum() / total_importance
        else:
            coverage = 0.0

    elif method == 'cumulative':
        target_coverage = kwargs.get('target_coverage', 0.9)
        mask = cumulative_importance <= target_coverage
        # 只要还有 True，说明还没达标。sum() 计算有多少个 True，再 +1 包括达标的那个临界点
        n_features = mask.sum() 
        
        # 边界处理：如果第一个就超过了target (n_features=0)，至少选1个；如果没满，取n+1
        if n_features < len(feature_importance):
            n_features += 1
            
        selected_features = feature_importance.head(n_features).index.tolist()
        coverage = float(cumulative_importance.iloc[n_features-1])

    elif method == 'elbow':
        # 使用肘部检测 简单的最大一阶差分法
        imp_diff = feature_importance.diff(periods=-1).abs() # 计算当前值 - 下一个值的幅度
        if imp_diff.dropna().empty:
             # 特征极少的情况
             selected_features = feature_importance.index.tolist()
             coverage = 1.0
        else:
            # 找到下降最剧烈的点，在这个点截断（保留这个点之前的）
            elbow_idx = imp_diff.idxmax()
            # 获取该索引的位置整数
            loc = feature_importance.index.get_loc(elbow_idx)
            # 保留到该位置（包含该位置）
            selected_features = feature_importance.head(loc + 1).index.tolist()
            coverage = float(cumulative_importance.iloc[loc])

    else:
        raise ValueError(f"Unknown selection method: {method}")

    
    selection_info = {
        'method': method,
        'params': kwargs,
        'selected_count': len(selected_features),
        'total_features': len(feature_importance),
        'coverage': coverage,
        'selection_ratio': len(selected_features) / len(feature_importance)
    }

    return selected_features, selection_info

def train_model_coefs(method, task_id, instruments, period, name, 
        model_class, model_params):
    random_state = 42
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_{0}_data.feather".format(name))

    ### 数据已经进行过时序标准化处理
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])

    ### 使用训练集+验证集作为fit数据 (无交叉验证)
    fit_data = final_data.loc[
        time_array['train_time'][0]:time_array['val_time'][1]]

    ### 进行模型训练
    fit_data = fit_data.dropna()
    features = [
        col for col in final_data.columns
        if col not in ['nxt1_ret_{0}h'.format(period)]
    ]
    new_columns = ["f{0}".format(i) for i in range(0, len(features))]
    X = fit_data[features]
    X.columns = new_columns
    y = fit_data['nxt1_ret_{0}h'.format(period)]

    # 训练模型
    model = model_class(random_state=random_state, **model_params)
    model.fit(X, y)

    coefficients = pd.Series(model.coef_, index=features)

    ## 若存在负相关
    coefficients = coefficients.abs()
    ## 过滤极小的浮动误差
    valid_coefficients = coefficients[coefficients > 1e-6].sort_values(
        ascending=False)
    
    return valid_coefficients
    