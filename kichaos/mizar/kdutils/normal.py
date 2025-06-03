import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler


def sklearn_fit(data):
    scaler = StandardScaler()
    scaler.fit(data.values)
    return scaler


def sklearn_normal(data, features, scaler=None):
    data1 = data.set_index(['trade_time', 'code'])
    data_price = data1[['price']]
    data1 = data1[features]
    ## sklearn 标准化
    factors_data1 = scaler.transform(data1.values)
    factors_data1 = pd.DataFrame(factors_data1,
                                 columns=features,
                                 index=data1.index)
    factors_data1 = pd.concat([factors_data1, data_price], axis=1)
    return factors_data1


def maxmin_normal(data, horizon, min_value=-1, max_value=1):

    def normalize_array1(arr, min_value, max_value):
        # 使用np.where来实现条件替换
        normalized = np.zeros_like(arr)
        max_value = np.max(arr)
        min_value = np.min(arr)
        nonzero_indices = arr != 0
        normalized[nonzero_indices] = np.where(
            arr[nonzero_indices] == max_value, max_value,
            np.where(
                arr[nonzero_indices] == min_value, min_value,
                arr[nonzero_indices] / np.abs(arr[nonzero_indices]) *
                np.where(arr[nonzero_indices] > 0, arr[nonzero_indices] /
                         max_value, arr[nonzero_indices] / min_value)))
        return normalized

    data1 = data.set_index(['trade_time', 'code'])
    horizon_data = normalize_array1(arr=data1.values,
                                    max_value=max_value,
                                    min_value=min_value)
    horizon_data = pd.DataFrame(horizon_data,
                                columns=["nxt1_ret_{0}h".format(horizon)],
                                index=data1.index)
    return horizon_data


def rolling_normal(data, window):
    current_data = data.set_index(['trade_time', 'code']).unstack()
    rolling_data = current_data.rolling(window=window)
    condition1 = (rolling_data.std() == 0)
    normalize_data = ((current_data - rolling_data.mean()) /
                      rolling_data.std()).where(~condition1, 0)
    normalize_data = normalize_data.stack()
    return normalize_data
