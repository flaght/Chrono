import os, pdb, itertools
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl
from lumina.genetic import Actuator
from lumina.genetic.fusion import Rotor
from kdutils.macro import *
from kdutils.file import fetch_file_data
import ultron.factor.empyrical as empyrical
from lumina.genetic.fusion.macro import EmpyricalTuple


## 找到对应的策略信息
def fetch_rotor(base_path, name, code):
    rotor = Rotor.from_pickle(path=os.path.join(base_path, 'kmeans',
                                                code.lower()),
                              name=name)
    return rotor


def create_postions(k_split, filter_strategies, total_data):
    actuator = Actuator(k_split=k_split)

    strategies_data = actuator.calculate(strategies_infos=filter_strategies,
                                         total_data=total_data)
    return strategies_data


def merge_signals(strategies_data, filter_strategies):
    actuator = Actuator(k_split=4)
    weights_data = actuator.fitness_weight(strategies_infos=filter_strategies)
    positions_data = actuator.fitness_signal(
        strategies_infos=filter_strategies,
        strategies_data=strategies_data,
        weights_data=weights_data)
    return positions_data, weights_data


def rank_positions(positions_data):
    positions_data.name = 'value'
    positions_data = positions_data.reset_index().set_index('trade_time')
    ## 分位数
    positions_data['rank'] = positions_data['value'].rank(method='first',
                                                          pct=True)
    return positions_data


def filter_threshold(method):
    base_path1 = os.path.join(base_path, method, g_instruments)
    total_data = fetch_file_data(base_path=base_path,
                                 method=method,
                                 g_instruments=g_instruments,
                                 datasets=['train_data', 'val_data'])
    pdb.set_trace()
    basic_rotor = fetch_rotor(base_path=base_path1,
                              name='1046921830',
                              code=instruments_codes[g_instruments][0])
    strategies_data = create_postions(k_split=16,
                                      filter_strategies=basic_rotor.strategies,
                                      total_data=total_data)
    positions_data, weights_data = merge_signals(strategies_data,
                                                 basic_rotor.strategies)

    positions_data = rank_positions(positions_data=positions_data)
    pdb.set_trace()
    ## 小于 -0.2385 空头  大于 0.1691  多头
    print(positions_data)


def create_metrics(method, min_threshold, max_threshold):
    strategy_settings = {
        'capital': 10000000,
        'commission': COST_MAPPING[instruments_codes[g_instruments][0]],
        'slippage': SLIPPAGE_MAPPING[instruments_codes[g_instruments][0]],
        'size': CONT_MULTNUM_MAPPING[instruments_codes[g_instruments][0]]
    }
    ## 小于 -0.2385 空头  大于 0.1691  多头
    base_path1 = os.path.join(base_path, method, g_instruments)
    total_data = fetch_file_data(base_path=base_path,
                                 method=method,
                                 g_instruments=g_instruments,
                                 datasets=['train_data','val_data','test_data'])
    
    market_data = total_data.set_index(['trade_time', 'code'])[[
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'vwap'
    ]]

    market_data = market_data.unstack()

    market_data['trade_vol', market_data['open'].columns[0]] = (
        strategy_settings['capital'] / market_data['open'] /
        strategy_settings['size'])

    basic_rotor = fetch_rotor(base_path=base_path1,
                              name='1046921830',
                              code=instruments_codes[g_instruments][0])
    strategies_data = create_postions(k_split=16,
                                      filter_strategies=basic_rotor.strategies,
                                      total_data=total_data)
    positions_data, weights_data = merge_signals(strategies_data,
                                                 basic_rotor.strategies)
    positions_data.name = 'value'
    positions_data = positions_data.reset_index().set_index('trade_time')
    positions_data['signal'] = np.where(
        positions_data['value'] < min_threshold, -1,
        np.where(positions_data['value'] > max_threshold, 1, 0))
    pdb.set_trace()
    signal_data = positions_data['signal']
    signal_data = signal_data.to_frame()
    signal_data.columns = pd.MultiIndex.from_tuples([('pos', 'IM')])
    df = calculate_ful_ts_pnl(pos_data=signal_data,
                              total_data=market_data,
                              strategy_settings=strategy_settings)

    returns = df['ret']
    calmar_ratio = empyrical.calmar_ratio(returns=returns,
                                          period=empyrical.DAILY)
    sharpe_ratio = empyrical.sharpe_ratio(returns=returns,
                                          period=empyrical.DAILY)
    sortino_ratio = empyrical.sortino_ratio(returns=returns,
                                            period=empyrical.DAILY)
    max_drawdown = empyrical.max_drawdown(returns=returns)
    annual_return = empyrical.annual_return(returns=returns,
                                            period=empyrical.DAILY)
    annual_volatility = empyrical.annual_volatility(returns=returns,
                                                    period=empyrical.DAILY)

    metrics = EmpyricalTuple(name='1046921830',
                             annual_return=annual_return,
                             annual_volatility=annual_volatility,
                             calmar=calmar_ratio,
                             sharpe=sharpe_ratio,
                             max_drawdown=max_drawdown,
                             sortino=sortino_ratio,
                             returns_series=returns)
    print(metrics)


if __name__ == '__main__':
    #filter_threshold(method='aicso2')
    create_metrics(method='aicso2',
                   min_threshold=-0.2385,
                   max_threshold=0.1691)
