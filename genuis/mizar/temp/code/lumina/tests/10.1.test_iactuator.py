import os, sys, pdb, re
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI

from lumina.formual.iactuator import Iactuator

kd_engine = DBAPI.FetchEngine.create_engine('kd')


def fetch_basic(begin_date, end_date, codes):
    name = 'fut_basic'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [names.contractObject.in_(codes), names.flag == 1]
    basic_info = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name,
        clause_list=clause_list,
        columns=['contractObject', 'contMultNum', 'listDate'])
    basic_info = basic_info.sort_values(by='listDate',
                                        ascending=False).drop_duplicates(
                                            subset='contractObject',
                                            keep='first')
    return basic_info.rename(columns={'contractObject': 'code'})


def fetch_main_market(begin_date, end_date, codes):
    basic_info = fetch_basic(begin_date, end_date, codes)
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
        res.append(dt)
    data = pd.concat(res, axis=0)
    ## 临时 过滤重复数据
    data = data.merge(basic_info, on='code', how='left')
    data['vwap'] = data['value'] / data['volume'] / data['contMultNum']
    data = data.dropna(subset=['vwap'])
    data = data.drop_duplicates(subset=['trade_time', 'code']).sort_values(
        by=['trade_time', 'code'])
    return data


total_data = fetch_main_market(begin_date='2023-09-01',
                               end_date='2023-10-01',
                               codes=['IF'])
total_data = total_data.set_index(['trade_time', 'code']).unstack()
pdb.set_trace()
actor = Iactuator(k_split=16)
actor.calculate(total_data=total_data)