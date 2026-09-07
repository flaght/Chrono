import pdb, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import base_path, INSTRUMENTS_CODES
from lib.cux001 import FactorEvaluate1
from lib.bck002 import rebuild_executed_signal_for_eval
from kdutils.data import fetch_local_market1


def create_chg(market_data, name='vwap'):
    pricep = market_data.set_index(['trade_time', 'code'])[name].unstack()
    pre_pricep = pricep.shift(1)
    ret_v2v = np.log((pricep) / pre_pricep)
    yields_data = ret_v2v.shift(-2)
    yields_data = yields_data.stack()
    yields_data.name = 'chg_pct'
    return yields_data.reset_index()


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


def attach_position_labels(trader_data, position_data):
    """
    trade_records 里的 signal_type 只有 regular/base_position/close_position，
    不能直接表达 position_data 的 open/close。

    用 position_data 作为元数据表给 regular trade row 补回 open/close/pair_id。
    同一时间、合约、方向、手数可能重复，所以加组内序号避免 merge 一对多放大。
    """
    trader_regular = trader_data[trader_data['signal_type'] ==
                                 'regular'].copy()

    meta_cols = [
        'trade_time', 'code', 'direction', 'numbers', 'signal_type', 'pair_id',
        'open_trade_time', 'close_trade_time'
    ]
    meta_cols = [col for col in meta_cols if col in position_data.columns]
    position_meta = position_data[meta_cols].copy()
    position_meta = position_meta.rename(
        columns={'signal_type': 'signal_type_position'})

    keys = ['trade_time', 'code', 'direction', 'numbers']
    trader_regular['_seq'] = trader_regular.groupby(keys).cumcount()
    position_meta['_seq'] = position_meta.groupby(keys).cumcount()

    trade_labeled = trader_regular.merge(position_meta,
                                         on=keys + ['_seq'],
                                         how='left',
                                         suffixes=('_trade', '_position'))

    return trade_labeled.drop(columns=['_seq'])


### 目标仓位和信号对比
def contrast_position_to_signal(position_data, signal_data):
    position_open = position_data[position_data['signal_type'] ==
                                  'open'].copy()
    position_check = position_open[[
        'trade_time', 'code', 'direction', 'numbers', 'pair_id'
    ]].merge(signal_data[['trade_time', 'code', 'signal', 'nxt1_ret_5h']],
             on=['trade_time', 'code'],
             how='left')
    print("position open vs raw signal diff:")
    print((position_check['direction'] != position_check['signal']).mean())
    print(pd.crosstab(position_check['direction'], position_check['signal']))
    return position_check


### 目标仓位和交易记录对吧
def contrast_position_to_trader(trader_data, position_data):
    trade_check = attach_position_labels(trader_data, position_data)
    print("trade records missing position rows:")
    print(trade_check['signal_type_position'].isna().mean())
    print(trade_check[trade_check['signal_type_position'].isna()].head(20))
    print("trade records by position signal_type:")
    print(trade_check['signal_type_position'].value_counts(dropna=False))
    return trade_check


### 信号和交易记录对比
def contrast_trader_to_signal1(signal_data, trade_check):
    open_trade = trade_check[trade_check['signal_type_position'] ==
                             'open'].copy()
    open_cmp = open_trade.merge(
        signal_data[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="left")
    bad_open = open_cmp[open_cmp["direction"] != open_cmp["signal"]]
    print("open diff ratio:")
    print(len(bad_open) / len(open_cmp))
    print(pd.crosstab(open_cmp["direction"], open_cmp["signal"]))
    return open_cmp


### 可交易信号和交易记录对比
def contrast_trader_to_signal2(signal_data, trade_check):
    open_trade = trade_check[trade_check['signal_type_position'] ==
                             'open'].copy()
    signal_data_execute = signal_data[signal_data['signal'] != 0]
    open_cmp_execute = open_trade.merge(
        signal_data_execute[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="right")
    bad_open = open_cmp_execute[open_cmp_execute["direction"] !=
                                open_cmp_execute["signal"]]
    print("open execute diff ratio:")
    print(len(bad_open) / len(open_cmp_execute))
    print(
        pd.crosstab(open_cmp_execute["direction"], open_cmp_execute["signal"]))

    return open_cmp_execute


### 可交易信号和交易记录对比 带开仓价 平仓价
def contrast_trader_to_signal3(signal_data, trade_check):
    open_leg_trade = trade_check[trade_check['signal_type_position'] ==
                                 'open'].copy()
    close_leg_trade = trade_check[trade_check['signal_type_position'] ==
                                  'close'].copy()

    trader_data1 = open_leg_trade[[
        'trade_time', 'pair_id', 'exec_price', 'code'
    ]].rename(columns={
        'exec_price': 'open_price'
    }).merge(close_leg_trade[['pair_id', 'exec_price', 'code',
                              'trade_time']].rename(columns={
                                  'exec_price': 'close_price',
                                  'trade_time': 'close_time'
                              }),
             on=['pair_id', 'code'])

    signal_data_execute = signal_data[signal_data['signal'] != 0]
    trader_execute = trader_data1.merge(
        signal_data_execute[["trade_time", "code", "signal", "nxt1_ret_5h"]],
        on=["trade_time", "code"],
        how="left")
    return trader_execute


def load_data():
    signal_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/signal/rl/1018806311332385/erband_signal/1002_test.feather"

    position_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/position_data.feather'

    trader_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/trade_records.feather'

    signal_data = pd.read_feather(signal_path)
    position_data = pd.read_feather(position_path)
    trader_data = pd.read_feather(trader_path)

    trader_data['trade_time'] = pd.to_datetime(
        trader_data['date'].astype(str) +
        trader_data['min_time'].astype(str).str.zfill(4),
        format='%Y-%m-%d%H%M')
    return signal_data, position_data, trader_data


def run1():
    signal_data, position_data, trader_data = load_data()

    position_check = contrast_position_to_signal(position_data=position_data,
                                                 signal_data=signal_data)

    trade_check = contrast_position_to_trader(trader_data=trader_data,
                                              position_data=position_data)

    open_cmp = contrast_trader_to_signal1(signal_data=signal_data,
                                          trade_check=trade_check)

    open_cmp_execute = contrast_trader_to_signal2(signal_data=signal_data,
                                                  trade_check=trade_check)


### 收益率对比
def run2():
    instruments = 'rbb'
    signal_data, position_data, trader_data = load_data()
    trade_check = contrast_position_to_trader(trader_data=trader_data,
                                              position_data=position_data)

    trader_execute = contrast_trader_to_signal3(signal_data=signal_data,
                                                trade_check=trade_check)

    ## 读取行情
    start_date = trader_execute['trade_time'].min()
    end_date = trader_execute['trade_time'].max()
    non_factor_data = fetch_local_market1(
        base_path=os.environ['BAR_FUT_DIRS'],
        begin_date=start_date,
        end_date=end_date,
        method=None,
        codes=[INSTRUMENTS_CODES[instruments]])
    pcr_factor_data = fetch_local_market1(
        base_path=os.environ['BAR_FUT_DIRS'],
        begin_date=start_date,
        end_date=end_date,
        method='pcr',
        codes=[INSTRUMENTS_CODES[instruments]])

    res = []
    non_chg_data = create_chg(non_factor_data)
    non_df = create_yields(data=non_chg_data.copy(), horizon=5)
    non_df.name = "non_nxt1_ret_{0}h".format(5)
    res.append(non_df)

    pcr_chg_data = create_chg(pcr_factor_data)
    pcr_df = create_yields(data=pcr_chg_data.copy(), horizon=5)
    pcr_df.name = "pcr_nxt1_ret_{0}h".format(5)
    res.append(pcr_df)

    yields_data = pd.concat(res, axis=1)
    trader_execute = trader_execute.merge(yields_data,
                                          on=['trade_time', 'code'])
    trader_execute['execute_ret'] = np.log(trader_execute['close_price'] /
                                           trader_execute['open_price'])


if __name__ == '__main__':
    run2()
