import os, pdb
import numpy as np
import pandas as pd
from ultron.datasets.generate import create_data
from ultron.ump.indicator.atr import atr14, atr21
from ultron.ump.market.symbol_pd import _benchmark


def create_factors(start_time, end_time, m=30, n=10, freq='T', res_name=None):
    date_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    date_index.name = 'trade_time'
    codes = ["code_" + str(i) for i in range(0, n)]

    factors_res = [
        create_data(date_index=date_index,
                    codes=codes,
                    name="factor_{}".format(str(i))) for i in range(0, m)
    ]
    if isinstance(res_name, str):
        factors_res.append(
            create_data(date_index=date_index, codes=codes, name=res_name))
    factors_data = pd.concat(factors_res, axis=1)

    factors_data = factors_data.reset_index().rename(
        columns={'level_1': 'code'})
    return factors_data


def load_random_data(ticker_dim,
                     factors_dim,
                     res_name=None,
                     start_time='2023-06-01 00:01:00',
                     end_time='2023-06-02 00:01:00'):
    data = create_factors(start_time,
                          end_time,
                          m=factors_dim,
                          n=ticker_dim,
                          res_name=res_name)
    data['price'] = np.abs((data['factor_0'] + data['factor_1']) / 2) * 10
    data.index = data['trade_time'].rank(method='dense').astype(int) - 1
    data.index.name = None
    return data


def creat_cnstock_boxdata():
    columns = [
        'close', 'low', 'high', 'open', 'volume', 'value', 'openint', 'chg',
        'price'
    ]
    codes = {
        'code_0': '000001',
        'code_1': '000002',
        'code_2': '000004',
        'code_3': '689009',
        'code_3': '300676'
    }
    data = load_random_data(ticker_dim=len(codes),
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data['code'] = data['code'].replace(codes)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    data = data.unstack()
    pre_close = data['close'].shift(1)
    p_change = (data['close'] -
                data['close'].shift(1)) / data['close'].shift(1)
    pre_close = pre_close.stack()
    pre_close.name = 'pre_close'

    p_change = p_change.stack()
    p_change.name = 'p_change'
    data = data.stack().reset_index().merge(pre_close.reset_index(),
                                            on=['trade_time', 'code']).merge(
                                                p_change.reset_index(),
                                                on=['trade_time', 'code'])
    data['date_week'] = data['trade_time'].dt.weekday
    data['trade_date'] = data['trade_time'].dt.date
    data['date'] = data['trade_time'].dt.strftime('%Y%m%d').astype(int)
    data = data.rename(columns={'trade_time': 'ttime'})
    return data


def creat_cnfut_boxdata():
    columns = [
        'close', 'low', 'high', 'open', 'volume', 'value', 'openint', 'chg',
        'price'
    ]
    codes = {
        'code_0': 'RB0',
        'code_1': 'CU0',
        'code_2': 'ZC0',
        'code_3': 'TA0',
        'code_3': 'CF0'
    }
    data = load_random_data(ticker_dim=len(codes),
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data['code'] = data['code'].replace(codes)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    data = data.unstack()
    pre_close = data['close'].shift(1)
    p_change = (data['close'] -
                data['close'].shift(1)) / data['close'].shift(1)
    pre_close = pre_close.stack()
    pre_close.name = 'pre_close'

    p_change = p_change.stack()
    p_change.name = 'p_change'
    data = data.stack().reset_index().merge(pre_close.reset_index(),
                                            on=['trade_time', 'code']).merge(
                                                p_change.reset_index(),
                                                on=['trade_time', 'code'])
    data['date_week'] = data['trade_time'].dt.weekday
    data['trade_date'] = data['trade_time'].dt.date
    data['date'] = data['trade_time'].dt.strftime('%Y%m%d').astype(int)
    data = data.rename(columns={'trade_time': 'ttime'})
    return data


######
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



def create_cnfut_box():
    data = creat_cnfut_boxdata()
    pdb.set_trace()
    benchmark = 'CU0'
    data['trade_date'] = data['ttime']
    benchmark_kl_pd = data[data['code'].isin([benchmark
                                              ])].set_index('trade_date')
    benchmark_kl_pd['key'] = list(range(0, len(benchmark_kl_pd)))
    benchmark_kl_pd.fillna(0, inplace=True)
    benchmark_kl_pd.name = benchmark
    calc_atr(benchmark_kl_pd)

    pick_kl_pd_dict = {}
    choice_code = []
    choice_symbols = [
        code for code in data['code'].unique().tolist() if code != benchmark
    ]
    for code in choice_symbols:
        kl_pd = data.set_index('code').loc[code].reset_index().set_index(
            'trade_date')
        kl_pd.name = str(code)
        kl_pd = _benchmark(kl_pd, benchmark_kl_pd)
        if kl_pd is None:
            continue
        #kl_pd.sort_index(inplace=True)
        calc_atr(kl_pd)
        kl_pd['key'] = list(range(0, len(kl_pd)))
        pick_kl_pd_dict[str(code)] = kl_pd
        choice_code.append(str(code))
    return benchmark_kl_pd, pick_kl_pd_dict, choice_code


def create_cnstock_box():
    data = creat_cnstock_boxdata()
    benchmark = '000001'
    data['trade_date'] = data['ttime']
    benchmark_kl_pd = data[data['code'].isin([benchmark
                                              ])].set_index('trade_date')
    benchmark_kl_pd['key'] = list(range(0, len(benchmark_kl_pd)))
    benchmark_kl_pd.fillna(0, inplace=True)
    benchmark_kl_pd.name = benchmark
    calc_atr(benchmark_kl_pd)

    pick_kl_pd_dict = {}
    choice_code = []
    choice_symbols = [
        code for code in data['code'].unique().tolist() if code != benchmark
    ]
    for code in choice_symbols:
        kl_pd = data.set_index('code').loc[code].reset_index().set_index(
            'trade_date')
        if code.startswith("600") or code.startswith("601") or code.startswith(
                "603") or code.startswith("688") or code.startswith(
                    "689") or code.startswith("605"):
            k1 = 'sh'
        elif code.startswith("000") or code.startswith(
                "002") or code.startswith("300") or code.startswith(
                    "00") or code.startswith("30"):
            k1 = 'sz'
        else:
            print('code:{} is not supported'.format(code))
        kl_pd.name = k1 + str(code)
        kl_pd = _benchmark(kl_pd, benchmark_kl_pd)
        if kl_pd is None:
            continue
        #kl_pd.sort_index(inplace=True)
        calc_atr(kl_pd)
        kl_pd['key'] = list(range(0, len(kl_pd)))
        pick_kl_pd_dict[k1 + str(code)] = kl_pd
        choice_code.append(k1 + str(code))
    return benchmark_kl_pd, pick_kl_pd_dict, choice_code


def create_agent_data(ticker_dim, columns):
    factors = create_factors(start_time='2023-06-01 00:01:00',
                             end_time='2023-06-01 00:01:00',
                             m=len(columns),
                             n=ticker_dim,
                             res_name=None)
    factors = factors.set_index(['trade_time','code'])
    factors.columns = columns
    return factors
