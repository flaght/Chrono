from typing import List
import pandas as pd
import numpy as np
import pdb
from lib import logger
## 全量因子值相关性
def calculate_factor_values_correlation(df: pd.DataFrame,
                                       feature_cols: List[str]) -> pd.DataFrame:
    """
    计算全量因子值相关性矩阵

    Args:
        df: 数据框
        feature_cols: 特征列名列表

    Returns:
        因子值相关性矩阵（绝对值）
    calculate_factor_values_correlation(df, feature_cols)
    """
    corr_matrix = df[feature_cols].corr().abs()
    return corr_matrix

## 滚动因子值相关性
def calculate_rolling_factor_correlation(df: pd.DataFrame,
                                        feature_cols: List[str],
                                        roll_win: int = 20,
                                        resampling_win: int = 5) -> pd.DataFrame:
    """
    计算滚动因子值相关性矩阵

    Args:
        df: 数据框
        feature_cols: 特征列名列表
        roll_win: 滚动窗口大小
        resampling_win: 重采样间隔（分钟）

    Returns:
        滚动因子值相关性矩阵
    correlation_matrix = calculate_rolling_factor_correlation(df, feature_cols, roll_win=20, resampling_win=5)
    """
    if roll_win <= 0 or resampling_win <= 0:
        # 如果参数无效，回退到全量相关性
        return df[feature_cols].corr().abs()
    
    # 重采样数据
    df_resampled = df.copy()
    if 'trade_time' in df.columns:
        # 按时间重采样
        is_on_mark = df['trade_time'].dt.minute % int(resampling_win) == 0
        df_resampled = df[is_on_mark]

    # 计算滚动相关性
    rolling_corr_matrices = []

    for i in range(roll_win, len(df_resampled) + 1):
        window_data = df_resampled.iloc[i-roll_win:i][feature_cols]
        if len(window_data) >= max(5, roll_win // 2):  # 确保有足够的数据
            corr_matrix = window_data.corr().abs()
            rolling_corr_matrices.append(corr_matrix)

    if not rolling_corr_matrices:
        # 如果没有有效的滚动窗口，回退到全量相关性
        return df[feature_cols].corr().abs()

    # 计算平均相关性矩阵
    avg_corr_matrix = np.mean([corr.values for corr in rolling_corr_matrices], axis=0)
    result_matrix = pd.DataFrame(avg_corr_matrix,
                               index=feature_cols,
                               columns=feature_cols)

    return result_matrix

## 通用时序因子收益率相关性
def calculate_ic_correlation_matrix(df: pd.DataFrame,
                                   feature_cols: List[str],
                                   target_col: str) -> pd.DataFrame:
    """
    计算通用时序因子收益率相关性矩阵（按时间分组）

    Args:
        df: 包含特征和目标变量的数据框
        feature_cols: 特征列名列表
        target_col: 目标变量列名

    Returns:
        IC相关性矩阵

    
    # 使用示例
    correlation_matrix = calculate_ic_correlation_matrix(df, feature_cols, target_col='nxt1_ret_5h')
    """
    ic_series_dict = {}

    # 确定时间列
    time_col = 'trade_time' if 'trade_time' in df.columns else 'date'

    # 按时间分组计算IC
    grouped = df.groupby(time_col)

    for factor in feature_cols:
        ic_values = []

        for period, group in grouped:
            try:
                if len(group) >= 5:  # 确保有足够的数据
                    ic = group[factor].corr(group[target_col])
                    if not np.isnan(ic) and not np.isinf(ic):
                        ic_values.append(ic)
            except:
                continue

        if ic_values:
            ic_series_dict[factor] = ic_values

    if not ic_series_dict:
        # 如果IC计算失败，回退到因子值相关性
        return df[feature_cols].corr().abs()

    # 构建IC数据框
    max_len = max(len(ic_list) for ic_list in ic_series_dict.values())
    ic_df = pd.DataFrame(index=range(max_len))

    for factor, ic_values in ic_series_dict.items():
        # 对齐长度（用NaN填充）
        padded_ic = ic_values + [np.nan] * (max_len - len(ic_values))
        ic_df[factor] = padded_ic

    # 计算IC序列间的相关性
    ic_corr_matrix = ic_df.corr().abs().fillna(0)

    return ic_corr_matrix


#自定义因子收益率相关性
def calculate_custom_ic_correlation_matrix(df: pd.DataFrame,
                                          feature_cols: List[str],
                                          target_col: str,
                                          roll_win: int = 20,
                                          resampling_win: int = 5) -> pd.DataFrame:
    """
    计算自定义因子收益率相关性矩阵（使用滚动窗口+重采样）

    Args:
        df: 包含特征和目标变量的数据框
        feature_cols: 特征列名列表
        target_col: 目标变量列名
        roll_win: 滚动窗口大小
        resampling_win: 重采样间隔（分钟）

    Returns:
        自定义IC相关性矩阵

    correlation_matrix = calculate_custom_ic_correlation_matrix(df, feature_cols, target_col='nxt1_ret_5h', roll_win=20, resampling_win=5)
    """
    ic_series_dict = {}
    #for factor in feature_cols:
    for i, factor in enumerate(logger.progress(feature_cols, description="[green]计算滚动因子收益率相关性..[/green]"), 1):
        try:
            if roll_win > 0 and resampling_win > 0:
                # 复用现有的滚动窗口IC计算逻辑
                df1 = df[['trade_time','code',factor, target_col]]

                # 重采样
                is_on_mark = df1['trade_time'].dt.minute % int(resampling_win) == 0
                resample_data = df1[is_on_mark]

                # 计算滚动IC序列
                rolling_ic = resample_data[target_col].rolling(
                    window=roll_win,
                    min_periods=5
                ).corr(resample_data[factor])

                # 获取有效的IC值
                ic_values = rolling_ic.dropna().values

                if len(ic_values) > 0:
                    ic_series_dict[factor] = ic_values

        except Exception as e:
            print(f"计算因子 {factor} 的自定义IC序列失败: {e}")
            continue

    if not ic_series_dict:
        # 如果IC计算失败，回退到因子值相关性
        return df[feature_cols].corr().abs()
    # 构建IC数据框
    max_len = max(len(ic_list) for ic_list in ic_series_dict.values())
    ic_df = pd.DataFrame(index=range(max_len))

    for factor, ic_values in ic_series_dict.items():
        # 对齐长度（用NaN填充）
        padded_ic = list(ic_values) + [np.nan] * (max_len - len(ic_values))
        ic_df[factor] = padded_ic

    # 计算IC序列间的相关性
    ic_corr_matrix = ic_df.corr(method='spearman').abs().fillna(0)

    return ic_corr_matrix

def calculate_correlation_matrix(df: pd.DataFrame,
                                feature_cols: List[str],
                                method: str = 'custom_ic_correlation',
                                **kwargs) -> pd.DataFrame:
    """
    统一的因子相关性计算接口

    Args:
        df: 数据框
        feature_cols: 特征列名列表
        method: 计算方法 ('factor_values', 'rolling_factor_values',
                        'ic_correlation', 'custom_ic_correlation')
        target_col: 目标变量列名（IC相关性方法需要）
        roll_win: 滚动窗口大小
        resampling_win: 重采样间隔

    Returns:
        相关性矩阵
    """
    if method == 'factor_values':
        return calculate_factor_values_correlation(df, feature_cols)

    elif method == 'rolling_factor_values':
        return calculate_rolling_factor_correlation(df, feature_cols, kwargs['roll_win'], kwargs['resampling_win'])

    elif method == 'ic_correlation':
        if not kwargs['target_col']:
            raise ValueError("ic_correlation方法需要target_col参数")
        return calculate_ic_correlation_matrix(df, feature_cols, kwargs['target_col'])

    elif method == 'custom_ic_correlation':
        if not kwargs['target_col']:
            raise ValueError("custom_ic_correlation方法需要target_col参数")
        return calculate_custom_ic_correlation_matrix(
            df, feature_cols, kwargs['target_col'], kwargs['roll_win'], kwargs['resampling_win'])

    else:
        raise ValueError(f"不支持的相关性计算方法: {method}")