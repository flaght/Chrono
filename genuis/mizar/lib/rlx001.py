import os, pdb, math, itertools
import pandas as pd
import numpy as np
from kdutils.macro2 import *
from kdutils.common import fetch_temp_returns
from lib.lsx001 import fetch_times


### 不能使用时间方式，特征中会涉及多维，如当前时间点需要上一分钟或者上N分钟的特征维度，故使用
def normal1_factors(method, instruments, task_id, period, window):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))

    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename)
    pdb.set_trace()
    returns_data = fetch_temp_returns(method=method,
                                      instruments=instruments,
                                      datasets=['train', 'val', 'test'],
                                      category='returns')

    returns_data = returns_data.set_index(['trade_time',
                                           'code'])[['nxt1_ret_1h']].unstack()
    ## 先滚动标准化
    features = [
        col for col in final_data.columns if col not in
        ['trade_time', 'code', 'price', 'nxt1_ret_{0}h'.format(period)]
    ]
    current_data = final_data.set_index(['trade_time',
                                         'code'])[features].unstack()
    #returns_data = final_data.set_index(['trade_time', 'code'
    #                                     ])[['nxt1_ret_{0}h'.format(period)
    #                                         ]].unstack()

    data_rolling_mean = current_data.rolling(window=window,
                                             min_periods=1).mean()
    data_rolling_std = current_data.rolling(window=window,
                                            min_periods=1).std().replace(
                                                0, 1e-8)
    normal_data = (current_data - data_rolling_mean) / data_rolling_std
    pdb.set_trace()
    normal_data = normal_data.clip(-3, 3) / 3
    normal_data = pd.concat([normal_data, returns_data], axis=1)
    normal_data = normal_data.stack()
    normal_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    normal_data = normal_data = normal_data.dropna().reset_index().loc[window:]

    normal_data = normal_data.rename(columns={
        "nxt1_ret_1h": "nxt1_ret"
    }).set_index('trade_time')

    dirs = os.path.join(base_path, method, instruments, 'temp', "rl", "nos",
                        str(task_id), str(period), str(window))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    new_columns = ['code'
                   ] + ["f{0}".format(i)
                        for i in range(0, len(features))] + ['nxt1_ret']
    normal_data.columns = new_columns


    train_data = normal_data.loc[
        time_array['train_time'][0]:time_array['train_time'][1]].reset_index()
    val_data = normal_data.loc[
        time_array['val_time'][0]:time_array['val_time'][1]].reset_index()
    test_data = normal_data.loc[
        time_array['test_time'][0]:time_array['test_time'][1]].reset_index()
    pdb.set_trace()
    train_data.to_feather(
        os.path.join(dirs, "normal_train_data.feather"))
    val_data.to_feather(
        os.path.join(dirs, "normal_val_data.feather"))
    test_data.to_feather(
        os.path.join(dirs, "normal_test_data.feather"))
