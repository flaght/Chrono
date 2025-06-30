import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer


## 分位数变换
def sklean_fit1(data):
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaler.fit(data.values)
    return scaler


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


def rolling_train(data, features, window_size):
    current_data = data.set_index(['trade_time', 'code'])[features].unstack()
    price_data = data.set_index(['trade_time', 'code'])[['price']].unstack()
    returns_data = data.set_index(['trade_time',
                                   'code'])[['nxt1_ret']].unstack()
    train_rolling_mean = current_data.rolling(window=window_size,
                                              min_periods=1).mean()
    train_rolling_std = current_data.rolling(window=window_size,
                                             min_periods=1).std().replace(
                                                 0, 1e-6)
    normal_train = (current_data - train_rolling_mean) / train_rolling_std

    normal_train = pd.concat([normal_train, price_data, returns_data], axis=1)
    #normal_train = pd.concat([normal_train, price_data], axis=1)
    normal_train = normal_train.stack()
    return normal_train.dropna().reset_index().loc[window_size:]


def rolling_val(train_data, val_data, features, window_size):
    history_data = train_data.set_index(['trade_time',
                                         'code'])[features].unstack()
    current_data = val_data.set_index(['trade_time',
                                       'code'])[features].unstack()
    price_data = val_data.set_index(['trade_time',
                                     'code'])[['price']].unstack()

    returns_data = val_data.set_index(['trade_time',
                                       'code'])[['nxt1_ret']].unstack()

    train_suffix = history_data.tail(window_size - 1)
    val_with_history = pd.concat([train_suffix, current_data])

    # 现在对这个拼接好的数据框进行滚动计算
    val_rolling_mean = val_with_history[features].rolling(
        window=window_size, min_periods=1).mean()
    val_rolling_std = val_with_history[features].rolling(
        window=window_size, min_periods=1).std().replace(0, 1e-6)

    # 重要：只取原始 val2 对应的那部分结果
    val_rolling_mean_final = val_rolling_mean.loc[current_data.index]
    val_rolling_std_final = val_rolling_std.loc[current_data.index]

    normal_val = (current_data[features] -
                  val_rolling_mean_final) / val_rolling_std_final
    normal_val = pd.concat([normal_val, price_data, returns_data], axis=1)
    #normal_val = pd.concat([normal_val, price_data], axis=1)
    normal_val = normal_val.stack()
    return normal_val.dropna().reset_index()


def rolling_test(val_data, test_data, features, window_size):
    history_data = val_data.set_index(['trade_time',
                                       'code'])[features].unstack()
    current_data = test_data.set_index(['trade_time',
                                        'code'])[features].unstack()
    price_data = test_data.set_index(['trade_time',
                                      'code'])[['price']].unstack()
    returns_data = test_data.set_index(['trade_time',
                                       'code'])[['nxt1_ret']].unstack()

    val_suffix = history_data.tail(window_size - 1)
    test_with_history = pd.concat([val_suffix, current_data])

    # 现在对这个拼接好的数据框进行滚动计算
    test_rolling_mean = test_with_history[features].rolling(
        window=window_size, min_periods=1).mean()
    test_rolling_std = test_with_history[features].rolling(
        window=window_size, min_periods=1).std().replace(0, 1e-6)

    # 重要：只取原始 test2 对应的那部分结果
    testl_rolling_mean_final = test_rolling_mean.loc[current_data.index]
    test_rolling_std_final = test_rolling_std.loc[current_data.index]

    normal_test = (current_data[features] -
                   testl_rolling_mean_final) / test_rolling_std_final
    normal_test = pd.concat([normal_test, price_data, returns_data], axis=1)
    #normal_test = pd.concat([normal_test, price_data], axis=1)
    normal_test = normal_test.stack()
    return normal_test.dropna().reset_index()
