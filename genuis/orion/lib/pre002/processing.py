import pandas as pd
import numpy as np

def standardize(raw_data):
    returndata = raw_data.sub(raw_data.mean(axis=1),
                              axis='rows').div(raw_data.std(axis=1),
                                               axis='rows')
    returndata.replace(np.inf, np.nan)
    returndata.replace(-np.inf, np.nan)
    return returndata

def normalize(raw_data):  # 归一化到[-1,1]
    dmax = raw_data.max(skipna=True, axis=1)
    dmin = raw_data.min(skipna=True, axis=1)
    returndata = raw_data.sub((dmax + dmin) / 2, axis='rows').div(
        (dmax - dmin) / 2, axis='rows')
    returndata[np.isinf(returndata)] = 0
    return returndata

def winsorize(indata, win_type='N', n_draw=5, pvalue=0.05):
    '''
    极值处理函数

    :param raw_data: 输入待处理因子，data_array
    :param win_type: [String] 去极值处理的类型选择, 包括正态分布去极值和分位数去极值，分别为'N'/'Q', 默认为前者
    :param n_draw: [int] 正态分布去极值的迭代次数，只有当win_type='NormDistDraw'，更改该参数才有意义；合法输入为正整数，默认值为5
    :param pvalue: [float] 分位数去极值的分位数指定，只有当win_type='QuantileDraw'，更改该参数才有意义；合法输入为(0,1)区间内的浮点数，默认值为0.05
    :return: 经过去极值处理之后的因子值 data_array
    '''

    # Local Process
    raw_data = indata.values
    data = raw_data.copy()  # do not modify input data
    l = data.shape[1]
    if win_type == 'Q':
        bott = np.nanquantile(data, pvalue / 2, axis=1, keepdims=True)
        upper = np.nanquantile(data, 1 - pvalue / 2., axis=1, keepdims=True)
        tbott = np.repeat(bott, l, axis=1)
        tupper = np.repeat(upper, l, axis=1)
        data[data < bott] = tbott[data < bott]
        data[data > upper] = tupper[data > upper]
    else:
        for i in range(n_draw):
            std = data.std(axis=1, keepdims=True)
            mean = data.mean(axis=1, keepdims=True)
            bott = mean - 3 * std
            upper = mean + 3 * std
            tbott = np.repeat(bott, l, axis=1)
            tupper = np.repeat(upper, l, axis=1)
            data[data < bott] = tbott[data < bott]
            data[data > upper] = tupper[data > upper]
    return pd.DataFrame(data, index=indata.index, columns=indata.columns)

