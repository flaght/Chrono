import pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from lib.cux003 import FactorEvaluate1
from lib.bck002 import rebuild_executed_signal_for_eval

period = 5
ret_col = f"nxt1_ret_{period}h"

signal_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/signal/rl/1018806311332385/erband_signal/1002_test.feather"

position_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/position_data.feather'

trader_path = '/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/temp/model/113001/5/rl/backtest/rl/1013836755991964/1018806311332385/erband_signal/1002_test/trade_records.feather'

signal_data = pd.read_feather(signal_path)
position_data = pd.read_feather(position_path)
trader_data = pd.read_feather(
    trader_path
)  #.drop_duplicates(subset=['date', 'min_time', 'code', 'direction', 'signal_type'])

trader_data['trade_time'] = pd.to_datetime(
    trader_data['date'].astype(str) +
    trader_data['min_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d%H%M')


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


### 目标仓位数据和原始信号数据是否一致
position_open = position_data[position_data['signal_type'] == 'open'].copy()
position_check = position_open[[
    'trade_time', 'code', 'direction', 'numbers', 'pair_id'
]].merge(signal_data[['trade_time', 'code', 'signal', 'nxt1_ret_5h']],
         on=['trade_time', 'code'],
         how='left')
print("position open vs raw signal diff:")
print((position_check['direction'] != position_check['signal']).mean())
print(pd.crosstab(position_check['direction'], position_check['signal']))

### 目标仓位数据和成交数据是否一致
trade_check = attach_position_labels(trader_data, position_data)
print("trade records missing position rows:")
print(trade_check['signal_type_position'].isna().mean())
print(trade_check[trade_check['signal_type_position'].isna()].head(20))
print("trade records by position signal_type:")
print(trade_check['signal_type_position'].value_counts(dropna=False))

### 原始信号和成交数据是否一致
signal_data_execute = signal_data[signal_data['signal'] != 0]
open_trade = trade_check[trade_check['signal_type_position'] == 'open'].copy()
pdb.set_trace()
open_cmp = open_trade.merge(
    signal_data[["trade_time", "code", "signal", "nxt1_ret_5h"]],
    on=["trade_time", "code"],
    how="left")
pdb.set_trace()
open_cmp_execute = open_trade.merge(
    signal_data_execute[["trade_time", "code", "signal", "nxt1_ret_5h"]],
    on=["trade_time", "code"],
    how="right")

bad_open = open_cmp[open_cmp["direction"] != open_cmp["signal"]]
print("open diff ratio:")
print(len(bad_open) / len(open_cmp))
"""
direction = -1 且 signal = -1 ：67997 笔
direction = -1 且 signal =  1 ：0 笔

direction =  1 且 signal = -1 ：0 笔
direction =  1 且 signal =  1 ：97456 笔

signal        -1      1
direction              
-1         67997      0
 1             0  97456
"""
print(pd.crosstab(open_cmp["direction"], open_cmp["signal"]))

evaluate1 = FactorEvaluate1(factor_data=position_check,
                            factor_name='direction',
                            ret_name='nxt1_ret_5h',
                            roll_win=15,
                            fee=0.0,
                            scale_method='raw',
                            expression="postions",
                            resampling_win=period,
                            name="postions")
stt1 = evaluate1.run()
#evaluate1.plot_results()

pdb.set_trace()

open_cmp1 = open_cmp[['trade_time', 'signal', 'nxt1_ret_5h']]
pdb.set_trace()
evaluate2 = FactorEvaluate1(factor_data=open_cmp1,
                            factor_name='signal',
                            ret_name='nxt1_ret_5h',
                            roll_win=15,
                            fee=0.0,
                            scale_method='raw',
                            expression="open_cmp",
                            resampling_win=period,
                            name="open_cmp")
sst2 = evaluate2.run()

signal_data2 = signal_data[['trade_time', 'signal', 'nxt1_ret_5h']]

evaluate3 = FactorEvaluate1(factor_data=signal_data2,
                            factor_name='signal',
                            ret_name='nxt1_ret_5h',
                            roll_win=15,
                            fee=0.0,
                            scale_method='raw',
                            expression="open_cmp",
                            resampling_win=period,
                            name="open_cmp")
sst3 = evaluate3.run()

pdb.set_trace()
#evaluate2.plot_results()

print("stt1:{0}\n".format(stt1))
print("sst2:{0}\n".format(sst2))
