import pdb, os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from kdutils.macro import base_path, codes
from kdutils.ttimes import get_dates
from kdutils.data import fetch_data, calc_atr

from ultron.ump.market.symbol_pd import _benchmark
from ultron.kdutils.file import dump_pickle
from alphacopilot.api.calendars import advanceDateByCalendar


def transform_data(data, benchmark='IF'):
    benchmark = '{0}0'.format(benchmark)
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


def fetch_data1(method):
    benchmark = 'IF'  #'CU'
    start_date, end_date = get_dates(method)
    start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(2)).strftime('%Y-%m-%d')
    data = fetch_data(begin_date=start_time,
                      end_date=end_date,
                      codes=codes + [benchmark])
    ###格式转换 ## 暂时以IF为benchmark，但策略里不能使用benchmark
    ### 标准化标的
    data['code'] = data['code'] + '0'
    data.rename(columns={'trade_time': 'ttime'}, inplace=True)
    data['date_week'] = data['ttime'].dt.weekday
    data['trade_date'] = pd.to_datetime(data['ttime'])
    data['date'] = data['ttime'].dt.strftime('%Y%m%d').astype(int)
    data['ttime'] = pd.to_datetime(data['ttime'])
    return transform_data(data, benchmark=benchmark)


def main(method):
    benchmark_kl_pd, pick_kl_pd_dict, choice_code = fetch_data1(method)
    ### 保存数据
    dirs = os.path.join(base_path, method)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    dump_pickle(benchmark_kl_pd,
                os.path.join(
                    dirs,
                    'benckmark_{0}.pkl'.format(os.environ['INSTRUMENTS'])),
                how='high')

    dump_pickle(pick_kl_pd_dict,
                os.path.join(dirs,
                             'pick_{0}.pkl'.format(os.environ['INSTRUMENTS'])),
                how='high')

    dump_pickle(choice_code,
                os.path.join(
                    dirs, 'choice_{0}.pkl'.format(os.environ['INSTRUMENTS'])),
                how='high')


main('mini')
