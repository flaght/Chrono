import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from ultron.strategy.models.processing import winsorize as alk_winsorize
from ultron.strategy.models.processing import standardize as alk_standardize
from kdutils.data import fetch_dfactors, fetch_chgpct


## 提取因子 标准化
def normal_data(method):
    dfactors_data = fetch_dfactors(base_path=os.environ['FACTOR_PATH'])
    dirs = os.path.join(os.environ['BASE_PATH'], method)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "normal_factors.feather")
    dfactors_data.reset_index().to_feather(filename)

## 返回第二天涨跌幅
def create_yields(data, horizon, method, offset=1):
    df = data.copy()
    df.set_index("trade_date", inplace=True)
    ## chg为log收益
    df['nxt1_ret'] = df['chg_pct']   ## ret_f1r_cc T -> T+1

    df = df.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)

    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    df.name = 'nxt1_ret'
    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "{0}h_yields.feather".format(horizon))
    pdb.set_trace()
    df.reset_index().to_feather(filename)


## 提取收益率
def build_yields(method):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], method)
    factors_data = pd.read_feather(os.path.join(dirs, "normal_factors.feather"))
    begin_date = factors_data['trade_date'].min()
    end_date = factors_data['trade_date'].max()
    chg_pct = fetch_chgpct(begin_date=begin_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    horizon_sets = [1, 2, 3, 4, 5]
    for horzion in horizon_sets:
        create_yields(data=chg_pct.copy(), horizon=horzion, method=method, offset=0)

##  标准化收益率
def normal_yields(method):
    res = []
    horizon_sets = [1, 2, 3, 4, 5]
    for horizon in horizon_sets:
        filename = os.path.join(os.environ['BASE_PATH'], method,
                            "{0}h_yields.feather".format(horizon))
        horizon_data = pd.read_feather(filename)
        horizon_data = horizon_data.set_index(['trade_date',
                                               'code']).sort_index()
        horizon_data = horizon_data['nxt1_ret'].unstack()
        f = alk_standardize(alk_winsorize(horizon_data)).unstack()
        f[np.isnan(f)] = 0
        f.name = "nxt1_ret_{0}h".format(horizon)
        res.append(f)

    dimension_data = pd.concat(res, axis=1)
    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "normal_yields.feather")
    dimension_data = dimension_data.swaplevel(
        'trade_date', 'code').sort_index().reset_index()
    print(filename)
    dimension_data.to_feather(filename)


def build_data(method):
    pdb.set_trace()
    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "normal_yields.feather")
    normal_yields_data = pd.read_feather(filename)

    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "normal_factors.feather")
    
    normal_factors_data = pd.read_feather(filename)

    total_data = normal_factors_data.merge(normal_yields_data, on=['trade_date', 'code'])
    total_data = total_data[(total_data['trade_date'] > '2020-02-04')&(total_data['trade_date'] < '2024-09-10')]
    total_data = total_data.sort_values(by=['trade_date','code'])
    dates = total_data['trade_date'].dt.strftime('%Y-%m-%d').unique().tolist()

    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]
    pdb.set_trace()
    ##切割样本
    train_filename = os.path.join(
        os.environ['BASE_PATH'], method,
        "train_model_normal.feather")
    train_data.reset_index(drop=True).to_feather(train_filename)

    val_filename = os.path.join(os.environ['BASE_PATH'], method,
                                "val_model_normal.feather")
    val_data.reset_index(drop=True).to_feather(val_filename)

def main(method):
    normal_data(method)
    #build_yields(method)
    #normal_yields(method)
    #build_data(method)

main(method=os.environ['DUMMY_NAME'])