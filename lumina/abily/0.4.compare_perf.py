import pandas as pd
import pdb
from lumina.genetic.signal.method import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
import ultron.factor.empyrical as empyrical
from dotenv import load_dotenv

load_dotenv()

from kdutils.common import *
from kdutils.macro2 import *
from kdutils.data import fetch_main_market



def calc(signal_data, total_data2, strategy_settings, name):
    df1 = calculate_ful_ts_ret(pos_data=signal_data,
                               total_data=total_data2.copy(0),
                               strategy_settings=strategy_settings)
    returns2 = df1['a_ret']
    returns2.name = name
    return returns2


def singal_func(factors_data, signal_method, **signal_params):
    pos_data = eval(signal_method)(factor_data=factors_data, **signal_params)
    pos_data = pos_data.reset_index().set_index('trade_time')[['IM']]
    pos_data.columns = pd.MultiIndex.from_tuples([('pos', 'IM')])
    return pos_data


def strategy_func(pos_data, total_data, strategy_method, **strategy_params):
    pos_data = eval(strategy_method)(signal=pos_data,
                                     total_data=total_data,
                                     **strategy_params)
    return pos_data


def test1():
    instruments = 'ims'
    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]] * 0.05,
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    pdb.set_trace()
    returns = pd.read_parquet('records/pred_alpha_cta.parquet')
    returns['trade_time'] = pd.to_datetime(returns['date'].astype(str) + ' ' +
                                           returns['minTime'].astype(str))
    returns['trade_time'] = pd.to_datetime(returns['trade_time'])
    returns['code'] = 'IM'

    returns = returns[['trade_time', 'ret_15']].rename(columns={
        'ret_15': 'IM'
    }).set_index('trade_time')

    data = pd.read_feather('_bak/20250905/test.feat')
    begin_date = '2023-12-10'  #data['trade_time'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = '2024-04-10'  #data['trade_time'].max().strftime('%Y-%m-%d %H:%M:%S')
    pdb.set_trace()
    market_data = fetch_main_market(begin_date=begin_date,
                                    end_date=end_date,
                                    codes=['IM'])
    market_data_indexed = market_data.set_index('trade_time')
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'value': 'sum'
    }
    market_data_15min = market_data_indexed.resample(
        '15T', label='right', closed='right').agg(aggregation_rules)
    market_data_15min.dropna(inplace=True)
    market_data_15min.reset_index(inplace=True)
    market_data_15min['code'] = 'IM'

    total_data = data.merge(market_data_15min, on=['trade_time'])
    data = total_data[['trade_time', 'f_scaled']]
    data = data.set_index(['trade_time'])
    new_columns = pd.MultiIndex.from_tuples([('pos', 'IM')])
    pos_data1 = data.copy()
    pos_data1.columns = new_columns
    market_data = total_data.drop(['f_scaled'], axis=1)
    market_data['vwap'] = market_data['value'] / market_data['volume'] / 200
    total_data2 = market_data.set_index(['trade_time', 'code']).unstack()

    pdb.set_trace()
    market_data1 = market_data.set_index('trade_time')
    temp_return2 = market_data1['vwap'].shift(
        -15) / market_data1['vwap'].shift(-1) - 1
    #data.reset_index().merge(return2, on=['trade_time'])
    temp_return2 = temp_return2.reset_index()
    temp_return2['trade_time'] = pd.to_datetime(temp_return2['trade_time'])
    temp_return2['code'] = 'IM'
    temp_return2 = temp_return2[['trade_time', 'vwap']].rename(columns={
        'vwap': 'IM'
    }).set_index('trade_time')
    temp_return2 = temp_return2.reindex(returns.index)
    pdb.set_trace()
    df = calculate_ful_ts_ret(pos_data=pos_data1,
                              total_data=total_data2,
                              strategy_settings=strategy_settings,
                              temp_returns=returns)

    df = calculate_ful_ts_ret(pos_data=pos_data1,
                              total_data=total_data2,
                              strategy_settings=strategy_settings,
                              temp_returns=temp_return2)
    returns = df['a_ret']
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    print('-->')


def test2():
    instruments = 'ims'
    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': 0,
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    begin_date = '2023-12-10'
    end_date = '2024-04-10'

    market_data = fetch_main_market(begin_date=begin_date,
                                    end_date=end_date,
                                    codes=['IM'])
    market_data_indexed = market_data.set_index('trade_time')
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'value': 'sum'
    }
    market_data_15min = market_data_indexed.resample(
        '15T', label='right', closed='right').agg(aggregation_rules)
    market_data['vwap'] = market_data['value'] / market_data['volume'] / 200
    market_data_15min.dropna(inplace=True)
    market_data_15min.reset_index(inplace=True)
    market_data_15min['code'] = 'IM'

    total_data1 = pd.read_feather('11.feat')
    factor_data1 = total_data1[['trade_time', 'code', 'pred_alpha']]
    return_data1 = total_data1[['trade_time', 'code', 'nxt1_ret_15h']]

    factors_data2 = factor_data1.rename(columns={
        'pred_alpha': 'transformed'
    }).set_index(['trade_time', 'code'])

    total_data2 = market_data_15min.set_index(['trade_time', 'code']).unstack()


    signal_method = 'rollrank_signal'
    signal_params = {'roll_num': 24, 'roll_num': 0.7}
    signal1 = singal_func(factors_data=factors_data2,
                          signal_method=signal_method,
                          **signal_params)
    #res[signal_method] = calc(signal1, signal_method)
    signal1.head()


test2()
