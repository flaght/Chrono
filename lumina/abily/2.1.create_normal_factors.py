import datetime, pdb, os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.repairor import process_features_for_window


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


def run(method, g_instruments):
    pdb.set_trace()
    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "train_data.feather")
    train_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])

    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "val_data.feather")
    val_data = pd.read_feather(filename).sort_values(by=['trade_time', 'code'])

    filename = os.path.join(base_path, method, g_instruments, 'merge',
                            "test_data.feather")
    test_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])

    base_columns = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'vwap'
    ]
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + base_columns
    ]
    ## 数据修复
    pdb.set_trace()

    train2, train_report = process_features_for_window(train_data, features)
    pdb.set_trace()
    train_report = pd.DataFrame(train_report.values())
    train_columns = train_report[train_report['status_value'] ==
                                 1]['name'].tolist()

    val2, val_report = process_features_for_window(val_data, features)
    val_report = pd.DataFrame(val_report.values())
    val_columns = val_report[val_report['status_value'] == 1]['name'].tolist()

    test2, test_report = process_features_for_window(test_data, features)
    test_report = pd.DataFrame(test_report.values())
    test_columns = val_report[val_report['status_value'] == 1]['name'].tolist()

    inter_columns = list(
        set(train_columns) & set(val_columns) & set(test_columns))

    train2 = train2[['trade_time', 'code'] + base_columns + inter_columns]
    val2 = val2[['trade_time', 'code'] + base_columns + inter_columns]
    test2 = test2[['trade_time', 'code'] + base_columns + inter_columns]
    pdb.set_trace()
    ## 保存路径
    to_path = os.path.join(base_path, method, g_instruments, 'repaired')
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    pdb.set_trace()
    train2.to_feather(os.path.join(to_path, "repaire_train_data.feather"))

    val2.to_feather(os.path.join(to_path, "repaire_val_data.feather"))

    test2.to_feather(os.path.join(to_path, "repaire_test_data.feather"))


if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    run(method=method, g_instruments=g_instruments)
