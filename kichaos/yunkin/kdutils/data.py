import pdb
import pandas as pd
from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI

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


def fetch_main_contract(begin_date, end_date, codes):
    name = 'fut_algin_factors'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [
        names.trade_date >= begin_date, names.trade_date <= end_date,
        names.flag == 1
    ]
    if codes is not None:
        clause_list.append(names.code.in_(codes))
    main_factors = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name, clause_list=clause_list, columns=['trade_date', 'code'])

    return main_factors


def fetch_market(begin_date, end_date, codes=None):
    pdb.set_trace()
    main_factors = fetch_main_contract(begin_date, end_date, codes)
    codes = main_factors['code'].unique().tolist()
    basic_info = fetch_basic(begin_date, end_date, codes)
    data = RetrievalAPI.get_main_price(begin_date=begin_date,
                                       end_date=end_date,
                                       codes=codes,
                                       method='pcr',
                                       format_data=0)
    market_data = [data[code] for code in codes if code in data]
    market_data = pd.concat(market_data,
                            axis=0).sort_values(by=['trade_date', 'code'])
    market_data['trade_time'] = pd.to_datetime(market_data['barTime'])
    market_data.rename(columns={
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
    market_data = market_data.sort_values(by=['trade_time', 'code'])
    market_data = market_data[(market_data.trade_date >= begin_date)
                              & (market_data.trade_date <= end_date)]
    market_data = market_data.drop(
        columns=['barTime', 'symbol', 'mincount', 'trade_date', 'chg'], axis=1)
    market_data = market_data.sort_values(by=['trade_time', 'code'])
    pdb.set_trace()
    return market_data
