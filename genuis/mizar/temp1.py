import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lib.cux001 import FactorEvaluate1


# 将因子值处理成离散值
def rolling_rank_discretization(series,
                                window=24,
                                positive_quantile=0.9,
                                negative_quantile=0.1):
    """
    使用严格的历史滚动窗口进行排名离散化，避免任何未来数据。

    参数:
    series: 输入数据，Pandas Series。
    window: 滚动窗口大小（默认10）。
    positive_quantile: 正方向分位数阈值（默认0.9）。
    negative_quantile: 负方向分位数阈值（默认0.1）。

    返回:
    discrete_series: 离散化后的Series，值为-1, 0, 1。
    """
    # 初始化结果Series
    discrete_series = pd.Series(index=series.index, dtype=float)

    # 遍历每个窗口
    for i in range(len(series)):
        if i < window - 1:
            # 窗口不足时设置为NaN
            discrete_series.iloc[i] = np.nan
            continue

        # 获取当前窗口数据 (严格使用过去window个点)
        window_data = series.iloc[i - window + 1:i + 1]  # 包括当前点，但都是历史数据

        # 跳过包含NaN值的窗口
        if window_data.isna().any():
            discrete_series.iloc[i] = np.nan
            continue

        # 在窗口内计算排名标准化
        rank_data = window_data.rank()
        standardized = (rank_data / rank_data.sum()) - 0.5

        # 计算当前窗口的分位数阈值
        pos_threshold = standardized.quantile(positive_quantile)
        neg_threshold = standardized.quantile(negative_quantile)

        # 对窗口内的最后一个值（即当前点）进行离散化
        current_standardized = standardized.iloc[-1]
        if current_standardized >= pos_threshold:
            discrete_series.iloc[i] = 1
        elif current_standardized <= neg_threshold:
            discrete_series.iloc[i] = -1
        else:
            discrete_series.iloc[i] = 0

    return discrete_series


def run1(method, instruments, period, name, task_id):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        task_id, str(period))
    filename = os.path.join(dirs, "{0}_predict_data.feather".format(name))
    predict_data = pd.read_feather(filename)
    is_on_mark = predict_data['trade_time'].dt.minute % int(period) == 0
    predict_data = predict_data[is_on_mark]
    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)

    predict_data['pred_alpha_disc'] = predict_data.groupby('code').apply(
        lambda x: rolling_rank_discretization(x['predict'],
                                              window=24,
                                              positive_quantile=0.7,
                                              negative_quantile=0.3)).T
    pdb.set_trace()
    evaluate1 = FactorEvaluate1(factor_data=predict_data,
                                factor_name='pred_alpha_disc',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=240,
                                fee=0.000012,
                                scale_method='raw',
                                expression=name)
    stats_dt = evaluate1.run()
    print(stats_dt)


if __name__ == '__main__':
    method = 'cicso0'
    instruments = 'ims'
    period = 5
    name = 'lgbm'
    task_id = '200037'
    run1(method=method,
         instruments=instruments,
         period=period,
         name=name,
         task_id=task_id)
