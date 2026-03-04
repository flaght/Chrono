import pdb, os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from alphacopilot.api.calendars import advanceDateByCalendar
from alphacopilot.api.data import RetrievalAPI
from jdw import DBAPI
from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates


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
    #basic_info = fetch_basic(begin_date, end_date, codes)
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
        #dt['price'] = dt['value'] / dt[
        #    'volume']  #此处用于成交价，但会出现value volume为0情况，导致price为inf，此情况使用 olch均值代替
        #dt['price'] = dt['price'].where(
        #    dt['price'].notna(), dt[['high', 'low', 'close',
        #                             'open']].mean(axis=1))
        dt['price'] = dt[['high', 'low', 'close', 'open']].mean(axis=1)
        #dt['vwap'] = dt['value'] / dt['volume']  ## 除以最小单位
        res.append(dt)
    data = pd.concat(res, axis=0)
    ## 临时 过滤重复数据
    #data = data.merge(basic_info, on='code', how='left')
    #data['vwap'] = data['value'] / data['volume'] / data['contMultNum']
    #data = data.dropna(subset=['vwap'])
    data = data.drop_duplicates(subset=['trade_time', 'code']).sort_values(
        by=['trade_time', 'code'])
    return data


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

def fetch_returns(begin_date, end_date, codes):
    pdb.set_trace()
    res = []
    horizon_sets = [1, 2, 3, 5, 10, 15]
    market_data = fetch_main_market(begin_date=begin_date,
                                    end_date=end_date,
                                    codes=codes)
    chg_data = create_chg(market_data, name='close')
    for horizon in horizon_sets:
        df = create_yields(data=chg_data.copy(), horizon=horizon)
        df.name = "nxt1_ret_{0}h".format(horizon)
        res.append(df)

    data1 = pd.concat(res, axis=1)
    weights_raw = {
        'nxt1_ret_1h': 3,  # T+1 最大
        'nxt1_ret_2h': 2,  # T+2 其次
        'nxt1_ret_3h': 1  # T+3 最小
    }
    pdb.set_trace()
    total_raw_weight = sum(weights_raw.values())
    weights = {col: w / total_raw_weight for col, w in weights_raw.items()}

    data1['time_weight'] = (data1['nxt1_ret_1h'] * weights['nxt1_ret_1h'] +
                            data1['nxt1_ret_2h'] * weights['nxt1_ret_2h'] +
                            data1['nxt1_ret_3h'] * weights['nxt1_ret_3h'])

    data1['equal_weight'] = data1[weights_raw.keys()].mean(axis=1)
    return data1


def start(method, task_id):
    start_date, end_date = get_dates(method)
    begin_date1 = advanceDateByCalendar("china.sse", start_date,
                                        '-5b').strftime('%Y-%m-%d')
    end_date1 = advanceDateByCalendar("china.sse", end_date,
                                      '5b').strftime('%Y-%m-%d')
    
    returns_data = fetch_returns(begin_date=begin_date1,
                                 end_date=end_date1,
                                 codes=['IM','IC','IH','IF'])
    output_dir = os.path.join(base_path, method, TASK_MAPPING[task_id]['source'], 'basic', task_id)
    os.makedirs(output_dir, exist_ok=True)
    pdb.set_trace()
    returns_data.reset_index().to_feather(os.path.join(output_dir, "original_returns.feather"))
    

if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          task_id=variant.task_id)