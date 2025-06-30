#### 1. 使用原始因子和收益率构建数据
#### 2. 使用原始因子和价格构建数据
#### 3. 使用挖掘因子和收益率构建数据
#### 4. 使用挖掘因子和价格构建数据
import pdb, os, datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ['INSTRUMENTS'] = 'rbb'
g_instruments = os.environ['INSTRUMENTS']

from kdutils.macro import base_path, codes
from kdutils.normal import sklearn_normal, sklearn_fit
from alphacopilot.api.calendars import advanceDateByCalendar, makeSchedule, BizDayConventions, DateGeneration
from kdutils.data import fetch_main_market


### 加载挖掘因子
def fetch_evolutions(method, categories, horizon):
    dirs = os.path.join(base_path, method, 'evolution', g_instruments,
                        str(horizon))
    filename = os.path.join(dirs, '{0}_factors.feather'.format(categories))
    factors = pd.read_feather(filename)
    return factors


def prepar_data(method, categories, horizon, type1, type2):
    if type1 == 'evolution':
        total_data = fetch_evolutions(method=method,
                                      categories=categories,
                                      horizon=horizon)
        total_data = total_data.sort_values(['trade_time', 'code'])
        ### 空值填充
        total_data = total_data.set_index(
            ['trade_time', 'code']).unstack().fillna(method='ffill').stack()
        total_data = total_data.reset_index()
    ## 价格直接提取
    if type2 == 'price':
        market_data = fetch_main_market(
            begin_date=total_data['trade_time'].min(),
            end_date=total_data['trade_time'].max(),
            codes=codes)
        total_data = total_data.merge(
            market_data[['trade_time', 'code', 'close']],
            on=['trade_time', 'code'])
        total_data.rename(columns={'close': 'price'}, inplace=True)

    min_date = total_data['trade_time'].min()
    start_time = advanceDateByCalendar('china.sse', min_date,
                                       '{0}b'.format(2)).strftime('%Y-%m-%d')
    total_data = total_data[total_data['trade_time'] > start_time]
    return total_data


def split_factors2(prepare_pd,
                   start_time,
                   method,
                   categories,
                   horizon,
                   train_params,
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
        col for col in prepare_pd.columns
        if col not in ['trade_time', 'code', 'price']
    ]
    res = []
    for i in range(len(dates) - 1):
        print(dates[i], dates[i + 1], i)

        ### 验证集
        val_start_time = dates[i].strftime('%Y-%m-%d')
        val_end_time = (dates[i + 1] +
                        datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        val1 = prepare_pd[(prepare_pd['trade_time'] >= val_start_time)
                          & (prepare_pd['trade_time'] <= val_end_time)]

        if val1.empty:
            continue

        ### 训练集
        train_start_time = advanceDateByCalendar(
            'china.sse', val_start_time,
            '-{}b'.format(train_params['train_days']))
        train_end_time = advanceDateByCalendar('china.sse', val_start_time,
                                               '-0b')
        train1 = prepare_pd[(prepare_pd['trade_time'] >= train_start_time)
                            & (prepare_pd['trade_time'] <= train_end_time)]

        if train1.empty:
            continue

        train1[features] = train1[features].replace([np.inf, -np.inf], np.nan)
        val1[features] = val1[features].replace([np.inf, -np.inf], np.nan)
        train1 = train1.fillna(method='ffill')
        val1 = val1.fillna(method='ffill')

        if train_params['nc'] == 1:
            window = train_params['window']
            window = 0
            scaler = sklearn_fit(
                train1.set_index(['trade_time', 'code'])[features])
            normal_train = sklearn_normal(train1, features, scaler)
            normal_val = sklearn_normal(val1, features, scaler)

        res.append((normal_train, normal_val, normal_val))

    print('done')
    dirs = os.path.join(
        base_path, method, 'normal', g_instruments, 'rolling',
        'normal_factors3', "{0}_{1}".format(categories, horizon),
        "{0}_{1}_{2}_{3}_{4}".format(str(train_params['freq']),
                                     str(train_params['train_days']),
                                     str(train_params['val_days']),
                                     str(train_params['nc']), str(window)))

    if not os.path.exists(dirs):
        os.makedirs(dirs)
    for i, (normal_train, normal_val, normal_test) in enumerate(res):
        filename = os.path.join(dirs,
                                'normal_factors_train_{0}.feather'.format(i))
        normal_train.reset_index().to_feather(filename)
        filename = os.path.join(dirs,
                                'normal_factors_val_{0}.feather'.format(i))
        normal_val.reset_index().to_feather(filename)

        filename = os.path.join(dirs,
                                'normal_factors_test_{0}.feather'.format(i))
        normal_test.reset_index().to_feather(filename)
        print(
            "normal_train.shape:{0}\n normal_val.shape:{1}\n normal_test.shape:{2}\n\n"
            .format(normal_train.shape, normal_val.shape, normal_test.shape))


def split_factors1(prepare_pd,
                   start_time,
                   method,
                   categories,
                   horizon,
                   train_params,
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
        col for col in prepare_pd.columns
        if col not in ['trade_time', 'code', 'price']
    ]
    res = []
    for i in range(len(dates) - 1):
        print(dates[i], dates[i + 1], i)
        ### 测试集
        test_start_time = dates[i].strftime('%Y-%m-%d')
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

        if val1.empty:
            continue
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
        train1 = train1.fillna(method='ffill')
        val1 = val1.fillna(method='ffill')
        test1 = test1.fillna(method='ffill')

        if train1.empty:
            continue
        if train_params['nc'] == 1:
            window = train_params['window']
            window = 0
            scaler = sklearn_fit(
                train1.set_index(['trade_time', 'code'])[features])
            normal_train = sklearn_normal(train1, features, scaler)
            normal_val = sklearn_normal(val1, features, scaler)
            normal_test = sklearn_normal(test1, features, scaler)
        res.append((normal_train, normal_val, normal_test))

    print('done')
    dirs = os.path.join(
        base_path, method, 'normal', g_instruments, 'rolling',
        'normal_factors3', "{0}_{1}".format(categories, horizon),
        "{0}_{1}_{2}_{3}_{4}".format(str(train_params['freq']),
                                     str(train_params['train_days']),
                                     str(train_params['val_days']),
                                     str(train_params['nc']), str(window)))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for i, (normal_train, normal_val, normal_test) in enumerate(res):
        filename = os.path.join(dirs,
                                'normal_factors_train_{0}.feather'.format(i))
        normal_train.reset_index().to_feather(filename)
        filename = os.path.join(dirs,
                                'normal_factors_val_{0}.feather'.format(i))
        normal_val.reset_index().to_feather(filename)

        filename = os.path.join(dirs,
                                'normal_factors_test_{0}.feather'.format(i))
        normal_test.reset_index().to_feather(filename)
        print(
            "normal_train.shape:{0}\n normal_val.shape:{1}\n normal_test.shape:{2}\nnormal_test path:{3}\n\n"
            .format(normal_train.shape, normal_val.shape, normal_test.shape,
                    filename))


def normal_factors(method,
                   categories,
                   horizon,
                   train_params,
                   type1='evolution',
                   type2='price'):
    total_data = prepar_data(method=method,
                             categories=categories,
                             horizon=horizon,
                             type1=type1,
                             type2=type2)

    ### 使用多少天前的数据
    trade_time = total_data['trade_time'].max()

    start_time = advanceDateByCalendar(
        'china.sse', trade_time,
        '-{0}b'.format(train_params['past_days'] + train_params['train_days'] +
                       train_params['val_days'])).strftime('%Y-%m-%d')
    prepare_pd = total_data[total_data['trade_time'] >= start_time]

    if 'kimto' in method:
        split_factors2(prepare_pd=prepare_pd,
                       start_time=trade_time,
                       train_params=train_params,
                       method=method,
                       categories=categories,
                       horizon=horizon)
    else:
        split_factors1(prepare_pd=prepare_pd,
                       start_time=trade_time,
                       train_params=train_params,
                       method=method,
                       categories=categories,
                       horizon=horizon)


if __name__ == '__main__':
    window = 3
    nc = 1

    train_days = 180
    val_days = 30
    test_days = 30

    freq = test_days
    past_days = train_days + val_days + test_days

    train_params = {
        'window': window,
        'nc': nc,
        'freq': freq,
        'train_days': train_days,
        'val_days': val_days,
        'test_days': test_days,
        'past_days': past_days
    }
    normal_factors(method='aicso3',
                   categories='o2o',
                   horizon=1,
                   type1='evolution',
                   type2='price',
                   train_params=train_params)
