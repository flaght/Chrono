import pdb
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


def rolling_1p(value, window, func1, name):
    pname = value.name
    value = value.to_frame()
    x1_rolling = rolling_window(value, window)
    values = pd.DataFrame(map(lambda x1: func1(x1), x1_rolling),
                          index=value.index,
                          columns=value.columns)
    values = values[pname]
    values.name = name
    return values


def rolling_2p(value1, value2, window, func1, name):
    value1 = value1.to_frame()
    value2 = value2.to_frame()
    x1_rolling = rolling_window(value1, window)
    x2_rolling = rolling_window(value2, window)
    values = pd.DataFrame(map(lambda x1, x2: func1(x1, x2), x1_rolling,
                              x2_rolling),
                          index=value1.index,
                          columns=value1.columns)
    values = values[value1.columns[0]]
    values.name = name
    return values
