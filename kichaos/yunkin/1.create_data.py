import datetime
import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path
from kdutils.data import fetch_market
from kdutils.ttime import get_dates


def split_three_parts(arr, ratios=[0.7, 0.2, 0.1]):
    n = len(arr)
    # 计算每个部分的大小
    part1 = int(np.ceil(n * ratios[0]))
    remaining = n - part1

    part2 = int(np.ceil(remaining * (ratios[1] / (ratios[1] + ratios[2]))))
    part3 = remaining - part2

    # 分割数组
    split1 = part1
    split2 = part1 + part2

    return arr[:split1], arr[split1:split2], arr[split2:]


def create_yields(data, horizon, offset=0):
    df = data.copy()
    df.set_index("trade_time", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    return df


## 读取指定时间主力合约行情
def load_market(begin_date, end_date, base_path, method):
    market_data = fetch_market(begin_date, end_date)
    dirs = os.path.join(base_path, method, 'basic')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, 'market_data.feather')
    print('save market data to {0}'.format(filename))
    market_data = market_data.drop_duplicates(subset=['trade_time', 'code'])
    market_data.sort_index().reset_index(drop=True).to_feather(filename)


### 收益率计算
def build_yields(method, horizon_sets, categories):
    #### 收益率 o2o T+1期开盘价和T+2期开盘价比. T期算因子, T+1期的开盘价交易，T+2期开盘价为一次收益计算
    dirs = os.path.join(base_path, method, 'basic')
    filename = os.path.join(dirs, 'market_data.feather')
    market_data = pd.read_feather(filename)
    if categories == 'o2o':
        pdb.set_trace()
        openp = market_data.set_index(['trade_time', 'code'])['open'].unstack()
        pre_openp = openp.shift(1)
        ret_o2o = np.log((openp) / pre_openp)
        yields_data = ret_o2o.shift(-2)
        yields_data = yields_data.stack()
        yields_data.name = 'chg_pct'
        yields_data = yields_data.reset_index()

    ##持仓目标收益率
    dirs = os.path.join(base_path, method, 'basic')
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for horizon in horizon_sets:
        df = create_yields(data=yields_data, horizon=horizon)
        filename = os.path.join(dirs,
                                '{0}_{1}h.feather'.format(categories, horizon))
        df.reset_index().to_feather(filename)
        print('save yields data to {0}'.format(filename))


## 合并数据并进行标准化
def normal_data(method, horizon, categories):
    dirs = os.path.join(base_path, method, 'basic')
    filename = os.path.join(dirs, 'market_data.feather')
    market_data = pd.read_feather(filename)

    yields_data = pd.read_feather(
        os.path.join(dirs, '{0}_{1}h.feather'.format(categories, horizon)))
    total_data = market_data.merge(yields_data,
                                   how='left',
                                   on=['trade_time', 'code'])
    total_data = total_data.dropna()
    total_data = total_data.set_index(['trade_time', 'code'])
    columns = total_data.columns
    grouped = total_data.groupby(level='trade_time')
    res = []
    for col in columns:
        print(col)
        dt1 = grouped[col].rank(pct=True)
        res.append(dt1)
    data = pd.concat(res, axis=1)
    data = data.sort_index()
    dirs = os.path.join(base_path, method, 'normal')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    count = data.groupby(level='trade_time').count()['close'].reset_index()
    count = count.rename(columns={'close': 'count'})
    data = data.reset_index().merge(count, on=['trade_time'], how='left')
    data = data[data['count'] > 20]
    data = data.drop(columns=['count'])
    pdb.set_trace()

    ### 过滤截面较小的时间段
    ### 71, 79,  51,  8,  4,  12,  3, 20, 11, 10,  2
    ### 10:16:00~10:30:00 13:01~13:30 8个
    ### 15:01:00~15:15:00 4个
    ### 23:01:00~01:00:00： 8个
    ## 01:01:00~02:30:00 3个

    times = pd.to_datetime(
        data['trade_time']).dt.strftime('%Y-%m-%d %H:%M:%S').unique()

    train_time, val_time, test_time = split_three_parts(times,
                                                        ratios=[0.7, 0.2, 0.1])
    train_data = data[data.trade_time.isin(train_time)].sort_values(
        by=['trade_time', 'code'])
    val_data = data[data.trade_time.isin(val_time)].sort_values(
        by=['trade_time', 'code'])
    test_data = data[data.trade_time.isin(test_time)].sort_values(
        by=['trade_time', 'code'])

    train_file = os.path.join(dirs, 'train_normal_{0}_{1}h.feather').format(
        categories, horizon)
    val_file = os.path.join(dirs, 'val_normal_{0}_{1}h.feather').format(
        categories, horizon)
    test_file = os.path.join(dirs, 'test_normal_{0}_{1}h.feather').format(
        categories, horizon)
    print('save train data to {0}'.format(train_file))
    print('save val data to {0}'.format(val_file))
    print('save test data to {0}'.format(test_file))
    train_data.reset_index(drop=True).to_feather(train_file)
    val_data.reset_index(drop=True).to_feather(val_file)
    test_data.reset_index(drop=True).to_feather(test_file)


def main(method):
    horizon_sets = [1, 2, 3, 5]
    begin_date, end_date = get_dates(method)
    
    load_market(begin_date=begin_date,
                end_date=end_date,
                base_path=base_path,
                method=method)
    

    build_yields(method=method, horizon_sets=horizon_sets, categories='o2o')
    normal_data(method=method, horizon=1, categories='o2o')


main('aicso3')
