### 测试指定因子转信号
import pdb
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import ultron.factor.empyrical as empyrical
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from kdutils.data import fetch_main_market

from kdutils.macro2 import *

def fetch_data():
    data = pd.read_parquet('./records/temp1/pred_alpha_cta.parquet')
    data['trade_time'] = pd.to_datetime(
        data['date'].astype(str) + data['minTime'].astype(str).str.zfill(6),
        format='%Y%m%d%H%M%S')
    data = data.rename(columns={'Code': 'code', 'pred_alpha': 'transformed'})
    data = data.sort_values(by=['trade_time', 'code'])
    data['code'] = 'IM'
    return data[['trade_time', 'code', 'transformed']]


def fetch_market(begin_date, end_date, codes):
    market_data = fetch_main_market(begin_date=begin_date,
                                    end_date=end_date,
                                    codes=codes)
    return market_data


factors_data = fetch_data()
factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
min_time = factors_data['trade_time'].min().strftime('%Y-%m-%d')
max_time = factors_data['trade_time'].max().strftime('%Y-%m-%d')

market_data = fetch_market(min_time, max_time, ['IM'])
market_data['trade_time'] = pd.to_datetime(market_data['trade_time'])
pdb.set_trace()


total_data = market_data.merge(factors_data, on=['trade_time', 'code'])

pdb.set_trace()
total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])
total_data1 = total_data.set_index(['trade_time'])
total_data2 = total_data.set_index(['trade_time', 'code']).unstack()

signal_method = 'quantile_signal'
strategy_method = 'trailing_atr_strategy'
signal_params = {'roll_num': 720, 'threshold': 0.8}
strategy_params = {
    'atr_period': 10,
    'atr_multiplier': 4,
    'max_volume': 1,
    'maN': 60
}
pdb.set_trace()

instruments = 'ims'

strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]] * 0.005,
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

factors_data1 = factors_data.set_index(['trade_time','code'])
pos_data = eval(signal_method)(factor_data=factors_data1,
                                       **signal_params)
pdb.set_trace()
#total_data1 = cycle_total_data.reset_index().set_index(
#            ['trade_time', 'code']).unstack()
pos_data1 = eval(strategy_method)(signal=pos_data,
                                          total_data=total_data2,
                                          **strategy_params)

df = calculate_ful_ts_ret(pos_data=pos_data1,
                                  total_data=total_data2,
                                  strategy_settings=strategy_settings)
returns = df['a_ret']
fitness = empyrical.sharpe_ratio(returns=returns,
                                         period=empyrical.DAILY)
pdb.set_trace()
print('-->')