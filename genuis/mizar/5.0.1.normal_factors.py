from dotenv import load_dotenv
import pandas as pd
import numpy as np
import datetime

load_dotenv()
from alphacopilot.calendars.api import *
from kdutils.macro2 import *

from kdutils.common import fetch_temp_returns

from lib.svx001 import scale_factors


def load_data(method, instruments, task_id, period):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename)
    ## 加载收益率 先用持仓一天收益率
    returns_data = fetch_temp_returns(method=method,
                                      instruments=instruments,
                                      datasets=['train', 'val', 'test'],
                                      category='returns')
    final_data = final_data.drop(['nxt1_ret_5h'], axis=1).merge(
        returns_data[['trade_time', 'code', 'nxt1_ret_1h']],
        on=['trade_time', 'code'])
    return final_data


def prepare_data(method, instruments, task_id, period, train_params):
    total_data = load_data(method=method,
                           instruments=instruments,
                           task_id=task_id,
                           period=period)
    
    min_date = total_data['trade_time'].min()
    start_time = advanceDateByCalendar('china.sse', min_date,
                                       '{0}b'.format(2)).strftime('%Y-%m-%d')
    total_data = total_data[total_data['trade_time'] > start_time]

    ### 使用多少天前的数据
    trade_time = total_data['trade_time'].max()

    start_time = advanceDateByCalendar(
        'china.sse', trade_time,
        '-{0}b'.format(train_params['past_days'] + train_params['train_days'] +
                       train_params['val_days'] + 5)).strftime('%Y-%m-%d')
    prepare_pd = total_data[total_data['trade_time'] >= start_time]

    horzion = 1
    features = [
        f for f in prepare_pd.columns
        if f not in ['trade_time', 'code', 'nxt1_ret_{0}h'.format(horzion)]
    ]
    new_features = ["feature_{0}".format(f) for f in range(0, len(features))]
    prepare_pd.columns = ['trade_time', 'code'
                          ] + new_features + ['nxt1_ret_{0}h'.format(horzion)]
    pdb.set_trace()
    return prepare_pd


def standard_features(prepare_features, method, win):
    features = prepare_features.columns
    predict_data = prepare_features.copy()
    for f in features:
        scale_factors(predict_data=predict_data,
                      method='roll_zscore',
                      win=240,
                      factor_name=f)
        prepare_features[f] = predict_data['transformed']
    return prepare_features


def split_factors1(prepare_pd,
                   period,
                   horizon,
                   start_time,
                   train_params,
                   instruments,
                   window,
                   time_format='%Y-%m-%d %H:%M:%S'):
    end_time = pd.to_datetime(
        prepare_pd['trade_time'].max()).strftime('%Y-%m-%d')
    start_time = advanceDateByCalendar(
        'china.sse', start_time, '-{}b'.format(train_params['past_days']))

    dates = makeSchedule(start_time,
                         end_time,
                         '{}b'.format(train_params['freq']),
                         calendar='china.sse',
                         dateRule=BizDayConventions.Following,
                         dateGenerationRule=DateGeneration.Backward)

    features = [
        col for col in prepare_pd.columns if col not in
        ['trade_time', 'code', 'price', 'nxt1_ret_{}h'.format(period)]
    ]
    normal_res = []
    for i in range(len(dates) - 1):
        print(dates[i], dates[i + 1], i)
        ### 测试集
        test_start_time = (dates[i] +
                           datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        test_end_time = (dates[i + 1] +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        test1 = prepare_pd[(prepare_pd['trade_time'] >= test_start_time)
                           & (prepare_pd['trade_time'] <= test_end_time)]
        if test1.empty:
            continue

        ### 验证集
        val_start_time = advanceDateByCalendar(
            'china.sse', test_start_time,
            '-{}b'.format(train_params['val_days']))
        val_end_time = advanceDateByCalendar('china.sse', test_start_time,
                                             '-0b')
        val1 = prepare_pd[(prepare_pd['trade_time'] >= val_start_time)
                          & (prepare_pd['trade_time'] <= val_end_time)]

        ### 训练集
        train_start_time = advanceDateByCalendar(
            'china.sse', test_start_time,
            '-{}b'.format(train_params['val_days'] +
                          train_params['train_days']))
        train_end_time = advanceDateByCalendar(
            'china.sse', test_start_time,
            '-{}b'.format(train_params['val_days']))

        train1 = prepare_pd[(prepare_pd['trade_time'] >= train_start_time)
                            & (prepare_pd['trade_time'] <= train_end_time)]

        train1[features] = train1[features].replace([np.inf, -np.inf], np.nan)
        val1[features] = val1[features].replace([np.inf, -np.inf], np.nan)
        test1[features] = test1[features].replace([np.inf, -np.inf], np.nan)
        print("train:{0}~{1},val:{2}~{3},test:{4}~{5}".format(
            train1['trade_time'].min(),
            train1['trade_time'].max(),
            val1['trade_time'].min(),
            val1['trade_time'].max(),
            test1['trade_time'].min(),
            test1['trade_time'].max(),
        ))

        normal_train = train1
        normal_val = val1
        normal_test = test1

        normal_res.append((normal_train, normal_val, normal_test))

    dirs = os.path.join(
        base_path, method, instruments, 'normal', 'rollings',
        'normal_factors3', "{0}".format(horizon),
        "{0}_{1}_{2}_{3}_{4}".format(str(train_params['freq']),
                                     str(train_params['train_days']),
                                     str(train_params['val_days']),
                                     str(train_params['nc']), str(window)))

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for i, (normal_train, normal_val, normal_test) in enumerate(normal_res):
        filename = os.path.join(dirs,
                                'normal_factors_train_{0}.feather'.format(i))
        normal_train.reset_index(drop=True).to_feather(filename)

        filename = os.path.join(dirs,
                                'normal_factors_val_{0}.feather'.format(i))
        normal_val.reset_index(drop=True).to_feather(filename)

        filename = os.path.join(dirs,
                                'normal_factors_test_{0}.feather'.format(i))
        normal_test.reset_index(drop=True).to_feather(filename)
        print(
            "normal_train.shape:{0}\n normal_val.shape:{1}\n normal_test.shape:{2}\n\n"
            .format(normal_train.shape, normal_val.shape, normal_test.shape))

    print("dirs:{0}".format(dirs))


def run1(method, instruments, task_id, period, stand_method):
    window = 240  ## 滚动标准化
    nc = 2
    freq = 10
    train_days = 60
    val_days = freq
    test_days = freq
    past_days = 500

    horizon = 1 ## 固定收益率
    train_params = {
        'window': window,
        'nc': nc,
        'freq': freq,  ## 每隔10天重新训练
        'train_days': train_days,  ## 训练数据
        'val_days': val_days,  ## 校验数据集
        'test_days': test_days,  ## 测试数据
        'past_days': past_days
    }

    prepare_pd = prepare_data(method=method,
                              instruments=instruments,
                              task_id=task_id,
                              period=period,
                              train_params=train_params)
    prepare_pd = prepare_pd.set_index(['trade_time', 'code'])
    prepare_returns = prepare_pd[['nxt1_ret_{0}h'.format(horizon)]]
    features = [
        f for f in prepare_pd.columns
        if f not in ['nxt1_ret_{0}h'.format(horizon)]
    ]
    prepare_features = prepare_pd[features]
    standard_features(prepare_features=prepare_features,
                      method=stand_method,
                      win=window)
    prepare_pd = pd.concat([prepare_features, prepare_returns], axis=1)
    prepare_pd = prepare_pd.reset_index()
    ### 使用多少天前的数据
    trade_time = prepare_pd['trade_time'].max()
    
    split_factors1(prepare_pd=prepare_pd,
                   period=period,
                   horizon=horizon,
                   start_time=trade_time,
                   train_params=train_params,
                   instruments=instruments,
                   window=window,
                   time_format='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    method = 'bicso0'
    instruments = 'rbb'
    period = 5
    task_id = '113001'
    run1(method=method,
         instruments=instruments,
         task_id=task_id,
         period=period,
         stand_method='roll_zscore')
