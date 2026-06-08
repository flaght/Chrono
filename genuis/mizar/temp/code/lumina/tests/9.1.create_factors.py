import os, sys, pdb, re
import pandas as pd
import datetime

from dotenv import load_dotenv

load_dotenv()

from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI

sys.path.insert(0, os.path.abspath('../'))

import lumina.impulse.i001 as i001
import lumina.impulse.i002 as i002
import lumina.impulse.i003 as i003
import lumina.impulse.i004 as i004
import lumina.impulse.i005 as i005
import lumina.impulse.i006 as i006
import lumina.impulse.i007 as i007
import lumina.impulse.i008 as i008
import lumina.impulse.i009 as i009
import lumina.impulse.i010 as i010
import lumina.impulse.i011 as i011
import lumina.impulse.i012 as i012
import lumina.impulse.i013 as i013

kd_engine = DBAPI.FetchEngine.create_engine('kd')


def callback_save(factors_data, name, method, start_date, end_date):
    cond1 = (factors_data.index.get_level_values(
        level=0) >= start_date) & (factors_data.index.get_level_values(
            level=0) <= (datetime.datetime.strptime(end_date, '%Y-%m-%d') +
                         datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    factors_data = factors_data[cond1]
    ff = factors_data.sort_index().reset_index()
    ff1 = ff  #ff.set_index(['trade_time', 'code']).unstack()
    dirs = os.path.join('records', method, 'IF', 'factors')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs,
                            '{0}_factors.feather'.format(name.split('.')[-1]))
    ff1.sort_index().reset_index(drop=True).to_feather(filename)


def calculate_factors(data, callback, method, start_date, end_date):

    def run(data, i00, callback, method, start_date, end_date):
        res = []
        for f in i00.__all__:
            print(f)
            cls = getattr(i00, f)
            obj = cls()
            r1 = obj.calc_impulse(data.copy())
            values = list(r1.values())
            values1 = [v.sort_index() for v in values]
            dt = pd.concat(values1, axis=1).sort_index()
            res.append(dt)
        data = pd.concat(res, axis=1)
        callback(factors_data=data,
                 name=i00.__name__,
                 method=method,
                 start_date=start_date,
                 end_date=end_date)

    run(data, i001, callback, method, start_date, end_date)
    run(data, i002, callback, method, start_date, end_date)
    run(data, i003, callback, method, start_date, end_date)
    run(data, i004, callback, method, start_date, end_date)
    run(data, i005, callback, method, start_date, end_date)
    run(data, i006, callback, method, start_date, end_date)
    run(data, i007, callback, method, start_date, end_date)
    run(data, i008, callback, method, start_date, end_date)
    run(data, i009, callback, method, start_date, end_date)
    run(data, i010, callback, method, start_date, end_date)
    run(data, i011, callback, method, start_date, end_date)
    run(data, i012, callback, method, start_date, end_date)
    run(data, i013, callback, method, start_date, end_date)


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


def calc_factors(codes):
    start_date = '2021-01-01'
    end_date = '2023-10-01'
    data = fetch_main_market(begin_date=start_date,
                             end_date=end_date,
                             codes=codes)
    data = data.set_index(['trade_time', 'code']).unstack()
    calculate_factors(data,
                      callback=callback_save,
                      method='aa1',
                      start_date=start_date,
                      end_date=end_date)


def merge_factors(method):
    ## 加载因子
    factor_dirs = os.path.join('records', method, 'IF', 'factors')
    res = []
    for root, dirs, files in os.walk(factor_dirs):
        for file in files:
            if file.endswith('.feather'):
                factor_file = os.path.join(root, file)
                factor_data = pd.read_feather(factor_file)
                res.append(factor_data.set_index(['trade_time', 'code']))
    data = pd.concat(res[:5] + res[6:], axis=1).sort_index()
    data = data.unstack().fillna(method='ffill')
    factors_data = data.stack().dropna().reset_index()
    pdb.set_trace()
    start_date = factors_data['trade_time'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = factors_data['trade_time'].max().strftime('%Y-%m-%d %H:%M:%S')
    pdb.set_trace()
    data = fetch_main_market(begin_date=start_date,
                             end_date=end_date,
                             codes=codes)
    factors_data = factors_data.merge(data[[
        'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
        'volume', 'openint', 'vwap'
    ]],
                                      on=['trade_time', 'code'])
    dirs = os.path.join('records', method, 'IF', 'factors')
    filename = os.path.join(dirs, 'factors_data.feather')
    factors_data.to_feather(filename)


codes = ['IF']
method = 'aa1'
#calc_factors(codes)
merge_factors(method)
