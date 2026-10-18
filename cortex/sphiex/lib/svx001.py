
import pandas as pd
import numpy as np

def scale_factors(factor_data, method, win, factor_name):
    x = factor_data[factor_name]
    if method == 'roll_min_max':
        rmin = x.rolling(win).min()
        rmax = x.rolling(win).max()
        factor_data['transformed'] = 2 * \
                (x - rmin) / (rmax - rmin).clip(lower=1e-8) - 1
    elif method == 'roll_zscore':
        mu = x.rolling(win).mean()
        sg = x.rolling(win).std()
        factor_data['transformed'] = (
            (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'roll_quantile':
        q25 = x.rolling(win).quantile(0.25)
        q75 = x.rolling(win).quantile(0.75)
        factor_data['transformed'] = 2 * \
                (x - q25) / (q75 - q25).clip(lower=1e-8) - 1

    elif method == 'ew_zscore':
        ema = x.ewm(span=win, adjust=False).mean()
        evar = x.ewm(span=win, adjust=False).var()
        factor_data['transformed'] = (
            (x - ema) / np.sqrt(evar).clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'train_const':
        # 用前 roll 个样本做训练集
        mu = x.iloc[:win].mean()
        sg = x.iloc[:win].std()
        factor_data['transformed'] = (
            (x - mu) / sg.clip(lower=1e-8)).clip(-3, 3) / 3

    elif method == 'raw':
        # 直接使用原始值，不进行任何缩放，假设为已经处理好的因子值，离散值为[-1,0,1], 连续值为[-1，1]
        factor_data['transformed'] = x
    else:
        raise ValueError('Unknown scale_method')