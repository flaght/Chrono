import os
import pandas as pd
import numpy as np
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *

from kdutils.macro2 import *


def read_factors(method, instruments, task_id, period, name):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "{0}_predict_data.feather".format(name))
    predict_data = pd.read_feather(filename)
    is_on_mark = predict_data['trade_time'].dt.minute % int(period) == 0
    predict_data = predict_data[is_on_mark]
    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)
    return predict_data


def scale_factors(predict_data, method, win, factor_name):
    x = predict_data[factor_name]
    if method == 'roll_min_max':
        rmin = x.rolling(win).min()
        rmax = x.rolling(win).max()
        predict_data['transformed'] = 2 * \
                (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1
    elif method == 'roll_zscore':
        mu = x.rolling(win).mean()
        sg = x.rolling(win).std()
        predict_data['transformed'] = (
            (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'roll_quantile':
        q25 = x.rolling(win).quantile(0.25)
        q75 = x.rolling(win).quantile(0.75)
        predict_data['transformed'] = 2 * \
                (x - q25) / (q75 - q25).clip(lower=1e-8) - 1

    elif method == 'ew_zscore':
        ema = x.ewm(span=win, adjust=False).mean()
        evar = x.ewm(span=win, adjust=False).var()
        predict_data['transformed'] = (
            (x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'train_const':
        # 用前 roll 个样本做训练集
        mu = x.iloc[:win].mean()
        sg = x.iloc[:win].std()
        predict_data['transformed'] = (
            (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'raw':
        # 直接使用原始值，不进行任何缩放，假设为已经处理好的因子值，离散值为[-1,0,1], 连续值为[-1，1]
        predict_data['transformed'] = x
    else:
        raise ValueError('Unknown scale_method')


def signal_function(factors_data, signal_method, signal_params):
    pos_data = eval(signal_method)(factor_data=factors_data, **signal_params)
    return pos_data


def strategy_function(pos_data, total_data, strategy_method, strategy_params):
    pos_data1 = eval(strategy_method)(signal=pos_data,
                                      total_data=total_data,
                                      **strategy_params)
    return pos_data1


def create_position(predict_data,
                    signal_method,
                    signal_params,
                    dropna=False,
                    factor_name='transformed',
                    strategy_method=None,
                    strategy_params=None):
    predict_data = predict_data.dropna(
        subset=[factor_name]) if dropna else predict_data
    total_data1 = predict_data.copy().set_index(['trade_time'])
    total_data2 = predict_data.copy().set_index(['trade_time',
                                                 'code']).unstack()
    factors_data = predict_data.set_index(['trade_time',
                                           'code'])[[factor_name]]

    pos_data = eval(signal_method)(factor_data=factors_data, **signal_params)

    if isinstance(strategy_method, str) and isinstance(strategy_params, dict):
        pos_data = eval(strategy_method)(signal=pos_data,
                                         total_data=total_data1,
                                         **strategy_params)
    return pos_data, total_data2
