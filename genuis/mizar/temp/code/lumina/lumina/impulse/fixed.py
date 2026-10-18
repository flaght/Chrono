import pdb
import numpy as np
import pandas as pd
from .helper import *
#from ultron.ump.core.helper import *



def safe_log(value, drift=1):
    pvalue = value.mask(value.shift(drift) != 0,
                        value / value.shift(drift)).mask(
                            value.shift(drift) == 0, np.nan)
    pvalue = pvalue.replace(0.0, np.nan)
    return pvalue.mask(~np.isnan(pvalue), np.log(pvalue))


def safe_logx(value, drift1, drift2):
    pvalue = value.mask(
        value.shift(drift2) != 0,
        value.shift(drift1) / value.shift(drift2)).mask(
            value.shift(drift2) == 0, np.nan)
    pvalue = pvalue.replace(0.0, np.nan)
    return pvalue.mask(~np.isnan(pvalue), np.log(pvalue))


def safe_div(value1, value2):
    pvalue = value1.mask(value2 != 0,
                         value1 / value2).mask(value2 == 0, np.nan)
    pvalue = pvalue.replace(0.0, np.nan)
    return pvalue


def rolling_window(a, window):
    """
    返回2D array的滑窗array的array
    """
    nanarr = np.empty((window - 1, a.shape[1]))
    nanarr[:] = np.nan
    a = np.vstack((nanarr, a))
    shape = (a.shape[0] - window + 1, window, a.shape[-1])
    strides = (a.strides[0], ) + a.strides
    a_rolling = np.lib.stride_tricks.as_strided(a,
                                                shape=shape,
                                                strides=strides)
    return a_rolling


def decay_array(length, reverse=True):
    decay_rate = -np.log(0.5)  # 使得第一个值为1，最后一个值为0.5
    indices = np.arange(length)
    decayed_values = np.exp(-decay_rate * indices)
    decayed_values = decayed_values[::-1] if reverse else decayed_values
    return decayed_values


## 兼容滚动
def roller_sum(data, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_sum(data, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_sum(data, window=wspan, min_periods=min_periods)


def roller_mean(data, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_mean(data, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_mean(data, window=wspan, min_periods=min_periods)


def roller_std(data, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_std(data, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_std(data, window=wspan, min_periods=min_periods)


def roller_corr(data1, data2, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_corr(data1, data2, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_corr(data1,
                               data2,
                               window=wspan,
                               min_periods=min_periods)


def roller_cov(data1, data2, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_cov(data1, data2, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_cov(data1,
                              data2,
                              window=wspan,
                              min_periods=min_periods)


def roller_skew(data, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support rank operation. Please use pd_rolling_rank directly.")
    else:
        return pd_rolling_skew(data, window=wspan, min_periods=min_periods)


def roller_var(data, wspan, min_periods, method):
    if method == 'ewm':
        return pd_ewm_var(data, span=wspan, min_periods=min_periods)
    else:
        return pd_rolling_var(data, window=wspan, min_periods=min_periods)


def roller_rank(data, wspan, min_periods, method, pct=False):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support rank operation. Please use pd_rolling_rank directly.")
    else:
        return pd_rolling_rank(data, window=wspan, min_periods=min_periods, pct=pct)


def roller_median(data, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support median operation. Please use pd_rolling_median directly.")
    else:
        return pd_rolling_median(data, window=wspan, min_periods=min_periods)


def roller_max(data, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support max operation. Please use pd_rolling_max directly.")
    else:
        return pd_rolling_max(data, window=wspan, min_periods=min_periods)


def roller_min(data, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support min operation. Please use pd_rolling_min directly.")
    else:
        return pd_rolling_min(data, window=wspan, min_periods=min_periods)


def roller_kurt(data, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support kurt operation. Please use pd_rolling_kurt directly.")
    else:
        return pd_rolling_kurt(data, window=wspan, min_periods=min_periods)


def roller_quantile(data, quantile, wspan, min_periods, method):
    if method == 'ewm':
        raise NotImplementedError("EWM does not support quantile operation. Please use pd_rolling_quantile directly.")
    else:
        return pd_rolling_quantile(data, quantile=quantile, window=wspan, min_periods=min_periods)