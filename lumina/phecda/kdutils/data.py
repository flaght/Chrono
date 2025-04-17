import pdb
import numpy as np
import pandas as pd
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI
from ultron.ump.indicator.atr import atr14, atr21

def calc_atr(kline_df):
    kline_df['atr21'] = 0
    #pdb.set_trace()
    if kline_df.shape[0] > 21:
        # 大于21d计算atr21
        kline_df['atr21'] = atr21(kline_df['high'].values,
                                  kline_df['low'].values,
                                  kline_df['pre_close'].values)
        # 将前面的bfill
        kline_df['atr21'].fillna(method='bfill', inplace=True)
    kline_df['atr14'] = 0
    if kline_df.shape[0] > 14:
        # 大于14d计算atr14
        kline_df['atr14'] = atr14(kline_df['high'].values,
                                  kline_df['low'].values,
                                  kline_df['pre_close'].values)
        # 将前面的bfill
        kline_df['atr14'].fillna(method='bfill', inplace=True)


def fetch_data(begin_date, end_date, codes):
    data = RetrievalAPI.get_main_price(begin_date=begin_date,
                                       end_date=end_date,
                                       codes=codes,
                                       method='pcr',
                                       format_data=0)
    res = []
    for code in data.keys():
        dt = data[code]
        dt['trade_time'] = pd.to_datetime(dt['barTime'])
        dt.rename(columns={
            'closePrice': 'close',
            'lowPrice': 'low',
            'highPrice': 'high',
            'openPrice': 'open',
            'totalVolume': 'volume',
            'totalValue': 'value',
            'openInterest': 'openint',
            'logRet': 'chg'
        },
                  inplace=True)

        dt = dt.drop(columns=['barTime', 'symbol', 'mincount', 'trade_date'],
                     axis=1)
        dt['price'] = dt[['high', 'low', 'close', 'open']].mean(axis=1)
        dt['vwap'] = dt[['high', 'low', 'close', 'open']].mean(axis=1)  #dt['value'] / dt['volume'] / 合约乘数
        dt['pre_close'] = dt['close'].shift(1)
        dt['p_change'] = (dt['close'] - dt['pre_close']) / dt['pre_close']
        res.append(dt)
    data = pd.concat(res, axis=0)
    ## 临时 过滤重复数据
    data = data.drop_duplicates(subset=['trade_time', 'code']).sort_values(
        by=['trade_time', 'code'])
    return data
