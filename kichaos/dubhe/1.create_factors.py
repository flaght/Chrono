import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from ultron.tradingday import *
from ultron.strategy.models.processing import standardize as alk_standardize
from ultron.strategy.models.processing import winsorize as alk_winsorize

from kdutils.ttimes import get_dates
from kdutils.data import *
from kdutils.macro import base_path


## 提取交易因子
def fetch_data(method, universe='hs300', horzion=1):
    begin_date, end_date = get_dates(method=method)
    ## 加载hemres数据
    pdb.set_trace()
    hfactors_data = fetch_hfactors(begin_date,
                                   end_date,
                                   universe=universe,
                                   horzion=horzion,
                                   is_scale=False)
    ## 加载行情数据
    #market_data = fetch_market(begin_date, end_date, universe=universe)
    ## 加载涨跌幅 o2o
    chg_data = fetch_chgpct(begin_date=begin_date, end_date=end_date)
    pdb.set_trace()
    chg_data = chg_data.reset_index()
    hfactors_data = hfactors_data.reset_index()
    total_data = hfactors_data.merge(chg_data, on=['trade_date', 'code'])

    dirs = os.path.join(base_path, universe, str(horzion))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    total_data.to_feather(
        os.path.join(dirs, '{0}_factors.feather'.format(method)))


def create_yields(data, horizon, method, universe, offset=2):
    pdb.set_trace()
    df = data.copy()
    df.set_index("trade_date", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)

    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    filename = os.path.join(base_path, universe,
                            "{0}_{1}h_yields.feather".format(method, horizon))
    print(filename)
    df.reset_index().to_feather(filename)


### 创建收益率 1,2,3,4,5,10,15,30,60
def build_yields(method, universe='hs300'):
    filename = os.path.join(base_path, universe,
                            '{0}_factors.feather'.format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])
    total_data["trade_date"] = pd.to_datetime(total_data["trade_date"])
    horizon_sets = [1, 2, 3, 4, 5, 10, 15, 20, 30, 45, 60, 90]
    for horizon in horizon_sets:
        print(horizon)
        create_yields(data=total_data,
                      horizon=horizon,
                      method=method,
                      universe=universe)


### 因子标准化
def normal_factors(method, universe='hs300', horzion=1):
    pdb.set_trace()
    dirs = os.path.join(base_path, universe, str(horzion))
    filename = os.path.join(dirs, '{0}_factors.feather'.format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])

    total_data = total_data.set_index(['trade_date', 'code'])
    factors_name = total_data.columns
    factors_name = [
        f for f in factors_name
        if f not in ['trade_date', 'code', 'dummy', 'chg_pct']
    ]
    factors_data = total_data.unstack()
    res = []
    pdb.set_trace()
    for ff in factors_name:
        print(ff)
        f = alk_standardize(alk_winsorize(factors_data[ff])).unstack()
        f[np.isnan(f)] = 0
        f.name = ff
        res.append(f)
    dimension_data = pd.concat(res, axis=1)

    filename = os.path.join(dirs, "{0}_factors_normal.feather".format(method))
    dimension_data.swaplevel(
        'trade_date', 'code').sort_index().reset_index().to_feather(filename)


#### 收益率标准化
def normal_yields(method, universe='hs300'):

    res = []
    horizon_sets = [1, 2, 3, 4, 5, 10, 15, 20, 30, 45, 60, 90]
    for horizon in horizon_sets:
        print(horizon)
        horizon_data = pd.read_feather(
            os.path.join(base_path, universe,
                         "{0}_{1}h_yields.feather".format(method, horizon)))
        horizon_data = horizon_data.set_index(['trade_date',
                                               'code']).sort_index()
        horizon_data = horizon_data['nxt1_ret'].unstack()
        f = alk_standardize(alk_winsorize(horizon_data)).unstack()
        f.name = "nxt1_ret_{0}h".format(horizon)
        res.append(f)
    dimension_data = pd.concat(res, axis=1)
    filename = os.path.join(base_path, universe,
                            "{0}_yields_normal.feather".format(method))
    dimension_data.swaplevel(
        'trade_date', 'code').sort_index().reset_index().to_feather(filename)


#### 合并数据
def build_data(method, universe='hs300', horzion=1):
    dirs = os.path.join(base_path, universe, str(horzion))
    factors_filename = os.path.join(dirs, "{0}_factors_normal.feather".format(method))
    factors_data = pd.read_feather(factors_filename)

    horizon_filename = os.path.join(base_path, universe,
                                    "{0}_yields_normal.feather".format(method))
    horizon_data = pd.read_feather(horizon_filename)

    total_data = factors_data.merge(horizon_data, on=['trade_date', 'code'])

    filename = os.path.join(base_path, universe, str(horzion),
                            "{0}_model_normal.feather".format(method))

    total_data.dropna().set_index([
        'trade_date', 'code'
    ]).unstack().fillna(0).stack().reset_index().to_feather(filename)


def main(method, universe, horzion):
    fetch_data(method, universe, horzion=horzion)
    #build_yields(method)
    #normal_factors(method, universe, horzion)
    #normal_yields(method, universe)
    #build_data(method, universe, horzion)


main(method='sicro', universe='hs300', horzion=1)
