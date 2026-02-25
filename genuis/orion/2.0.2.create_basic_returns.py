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
    df = df.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(future_stack=True)
    df.name = 'nxt1_ret'
    return df


# 核心：合成套利总收益 (basis + funding - cost)
def create_chg(data, cost_rate):
    """
    计算单步套利总收益 (log-return)

    严格流程:
        y_basis = log(S_{t+2}/S_{t+1}) - log(F_{t+2}/F_{t+1})
        R_basis = exp(y_basis) - 1
        R_total = R_basis + f_{t+1} - cost
        y_total = log(1 + R_total)

    Parameters
    ----------
    data : DataFrame
        原始数据，含 ['trade_time', 'code', 's_open', 'f_open', 'f_funding_rate']
    cost_rate : float
        单步成本率 (手续费 + 滑点等，占名义比例)

    Returns
    -------
    Series : chg_pct (long format, with trade_time + code index)
    """
    basis_log_ret = create_basis_return(data)
    # 取 funding rate (wide format)
    wide = data.set_index(['trade_time', 'code'])
    funding = wide['f_funding_rate'].unstack()

    # f_funding_rate 已经摊平到每小时
    # funding_rate > 0：多头付给空头 → 做空 → 收钱 → +正值
    # funding_rate < 0：空头付给多头 → 做空 → 付钱 → +负值（自然减少收益）
    # 让 T 行用到 f_{T+1}（对应 [T+1, T+2)）
    funding_income = funding.shift(-1)

    # Step 2: log → simple
    r_basis = np.exp(basis_log_ret) - 1

    # Step 3: 合成 (simple 口径)
    r_total = r_basis + funding_income - cost_rate

    # Step 4: simple → log
    y_total = np.log(1 + r_total)

    # 转 long format
    y_total = y_total.stack()
    y_total.name = 'chg_pct'

    return y_total.reset_index()


def create_return(data, cost_rate=0.0):
    horizon_sets = [1, 2, 3, 5, 10, 15]
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
    filename = os.path.join(dirs, 'returns_data.feather')
    print(filename)
    ff.sort_index().reset_index(drop=True).to_feather(filename)


def run(method, task_id):
    dirs = os.path.join(base_path, method, 'basic', task_id)
    file_name = os.path.join(dirs, "raw_basic.feather")
    raw_basic_data = pd.read_feather(file_name)
    pdb.set_trace()
    cols = ['trade_time', 'code', 's_open', 'f_open', 'f_funding_rate']
    data = raw_basic_data[cols].copy()

    return_data = create_return(data=data, cost_rate=0.0)
    return_data = return_data.dropna()
    returns_save(return_data=return_data,
                task_id=task_id,
                 method=method)


if __name__ == '__main__':
    pdb.set_trace()
    variant = Tactix().start()
    run(method=variant.method, task_id=variant.task_id)
