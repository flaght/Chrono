import numpy as np
import pandas as pd


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


def pos_sum(ret, n):
    decay_weight = decay_array(n)
    return np.sum(ret * decay_weight[:, np.newaxis], axis=0)


def calc_umr(values, window):
    factor = pd.DataFrame(map(lambda ret: pos_sum(ret, window),
                              rolling_window(values.values, window)),
                          index=values.index,
                          columns=values.columns)
    return factor
