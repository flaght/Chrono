## Crypto BN 收益率计算
import pdb, os, datetime
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.ttimes import get_dates
from kdutils.macro2 import TASK_MAPPING, base_path
from kdutils.tactix import Tactix


def create_basis_return(data):
    """
    计算基差对数收益 (v2v: open-to-open)

    y_t = log(S_{t+1}/S_t) - log(F_{t+1}/F_t)
      = [log(S_{t+1}) - log(S_t)] - [log(F_{t+1}) - log(F_t)]
    """
    wide = data.set_index(['trade_time', 'code'])

    s_open = wide['s_open'].unstack()
    f_open = wide['f_open'].unstack()
    # 单步 log-return
    s_log_ret = np.log(s_open / s_open.shift(1))
    f_log_ret = np.log(f_open / f_open.shift(1))

    # 基差 log-return = 做多现货 - 做空期货
    basis_log_ret = s_log_ret - f_log_ret

    # 让 T 行代表 [T+1 open -> T+2 open)
    basis_log_ret = basis_log_ret.shift(-2)

    return basis_log_ret


def create_yields(data, horizon, offset=0):
    """
    给定单步 chg_pct，滚动求 horizon 期累计 log-return

    逻辑: log-return 可直接累加
        nxt1_ret = sum(chg_pct[t:t+horizon])
        shift 使其对齐到 t 时刻作为"未来 horizon 期收益"
    """
    df = data.copy()
    df.set_index('trade_time', inplace=True)

    # log收益直接累加
    df['nxt1_ret'] = df['chg_pct']
    df = df.groupby('code').rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    #df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
    #    dropna=False)
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        future_stack=True)
    df.name = 'nxt1_ret'
    return df


# 核心：合成套利总收益 (basis + funding - cost)
def create_chg(data, cost_rate):
    wide = data.set_index(['trade_time', 'code'])
    f_open = wide['f_open'].unstack()
    f_log_ret = np.log(f_open / f_open.shift(1))

    target_f_log_ret = f_log_ret.shift(-2)

    funding = wide['f_funding_rate'].unstack().fillna(0.0)

    funding_rate = funding.shift(-1)

    r_future = np.exp(target_f_log_ret) - 1

    r_total = r_future - funding_rate - cost_rate

    y_total = np.log(1 + r_total)

    y_total = y_total.stack(future_stack=True)

    y_total.name = 'chg_pct'

    return y_total.reset_index()


def create_return(data, cost_rate=0.0):
    horizon_sets = [1, 2, 3, 5, 8, 10, 15]
    ## 计算 单步套利收益率 等同于股票期货当日涨跌幅
    chg_data = create_chg(data, cost_rate=cost_rate)
    # Step 2: 多 horizon 滚动
    res = []
    for horizon in horizon_sets:
        df = create_yields(data=chg_data.copy(), horizon=horizon)
        df.name = "nxt1_ret_{0}h".format(horizon)
        res.append(df)

    data1 = pd.concat(res, axis=1)

    # Step 3: 加权混合
    weights_raw = {
        'nxt1_ret_1h': 3,  # T+1 权重最大
        'nxt1_ret_2h': 2,  # T+2 其次
        'nxt1_ret_3h': 1  # T+3 最小
    }
    total_raw_weight = sum(weights_raw.values())
    weights = {col: w / total_raw_weight for col, w in weights_raw.items()}

    data1['time_weight'] = (data1['nxt1_ret_1h'] * weights['nxt1_ret_1h'] +
                            data1['nxt1_ret_2h'] * weights['nxt1_ret_2h'] +
                            data1['nxt1_ret_3h'] * weights['nxt1_ret_3h'])

    data1['equal_weight'] = data1[list(weights_raw.keys())].mean(axis=1)

    return data1


def returns_save(return_data, method, task_id):
    pdb.set_trace()
    start_date, end_date = get_dates(method)
    cond1 = (return_data.index.get_level_values(level=0)
             >= (datetime.datetime.strptime(start_date, '%Y-%m-%d') +
                 datetime.timedelta(days=1)).strftime('%Y-%m-%d')) & (
                     return_data.index.get_level_values(level=0)
                     <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    return_data = return_data[cond1]
    ff = return_data.sort_index().reset_index()
    dirs = os.path.join(base_path, method, 'derivative', task_id)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pdb.set_trace()
    filename = os.path.join(dirs, 'returns_data.feather')
    print(filename)
    ff.sort_index().reset_index(drop=True).to_feather(filename)


def run(method, task_id):
    dirs = os.path.join(base_path, method, 'basic', task_id)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    pdb.set_trace()
    cols = ['trade_time', 'code', 'open', 'funding_rate']
    data = raw_basic_data[cols].copy()
    data = data.rename(columns={'open': 'f_open','funding_rate':'f_funding_rate'})
    data['f_funding_rate'] = data['f_funding_rate'].fillna(0.0)
    return_data = create_return(data=data, cost_rate=0.0)
    return_data = return_data.dropna()
    returns_save(return_data=return_data, task_id=task_id, method=method)


if __name__ == '__main__':
    pdb.set_trace()
    variant = Tactix().start()
    run(method=variant.method, task_id=variant.task_id)
