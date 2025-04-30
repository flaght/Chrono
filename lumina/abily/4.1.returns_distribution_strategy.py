## 收益率法 将多个策略信号合成值转化为信号值

import os, pdb, itertools
import pandas as pd
import numpy as np


from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ims'
g_instruments = os.environ['INSTRUMENTS']

import ultron.factor.empyrical as empyrical
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl
from kdutils.file import fetch_file_data
from lumina.genetic import Actuator
from lumina.genetic.fusion import Rotor
from lumina.genetic.process import *
from kdutils.macro import *


def generate_intervals(n_parts):
    edges = np.linspace(0, 1, n_parts + 1)
    intervals = [
        (i, round(edges[i], 10), round(edges[i + 1], 10))
        #f"{round(edges[i], 10)}~{round(edges[i+1], 10)}"
        for i in range(n_parts)
    ]
    return intervals


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


def process_market(total_data):
    market_data = total_data[[
        'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
        'volume', 'openint', 'vwap'
    ]].set_index(['trade_time', 'code']).unstack()
    market_data['trade_vol', market_data['open'].columns[0]] = (
        strategy_setting['capital'] / market_data['open'] /
        strategy_setting['size'])
    return market_data


def create_metrics(column, positions_data, market_data, strategy_setting):
    positions_data = positions_data.copy()
    positions_data['new_value'] = positions_data['value'].where(
        (positions_data['rank'] > column[1]) &
        (positions_data['rank'] <= column[2]),  # 条件表达式 [4]
        other=0  # 不满足条件时设为 0
    )

    current_signals = positions_data.reset_index().set_index(
        ['trade_time', 'code'])['new_value'].unstack()
    current_signals.columns = pd.MultiIndex.from_tuples([('pos', 'IM')])

    df = calculate_ful_ts_pnl(pos_data=current_signals,
                              total_data=market_data,
                              strategy_settings=strategy_setting)
    returns = df['ret']
    calamr = empyrical.calmar_ratio(returns=returns, period=empyrical.DAILY)
    sharpe = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    returns_mean = returns.mean()
    return {
        'name': column[0],
        'calamr_ratio': calamr,
        'sharpe_ratio': sharpe,
        'returns_mean': returns_mean
    }


@add_process_env_sig
def run_metrics(target_column, positions_data, market_data, strategy_setting):
    metrics = run_process(target_column=target_column,
                          callback=create_metrics,
                          positions_data=positions_data,
                          market_data=market_data,
                          strategy_setting=strategy_setting)
    return metrics


if __name__ == '__main__':
    method = 'aicso2'
    k_split = 4
    n_parts = 10
    base_path1 = os.path.join(base_path, method, g_instruments)
    groups1 = generate_intervals(n_parts=n_parts)
    strategy_setting = {
        'capital': 10000000,
        'commission': 2.3e-05,
        'slippage': 0.0001,
        'size': 200
    }
    total_data = fetch_file_data(base_path=base_path,
                                 method=method,
                                 g_instruments=g_instruments,
                                 datasets=['train_data','val_data','test_data'])
    pdb.set_trace()
    basic_rotor = fetch_rotor(base_path=base_path1,
                              name='1046921830',
                              code=instruments_codes[g_instruments][0])
    strategies_data = create_postions(k_split=4,
                                      filter_strategies=basic_rotor.strategies,
                                      total_data=total_data)
    positions_data, weights_data = merge_signals(strategies_data,
                                                 basic_rotor.strategies)

    positions_data = rank_positions(positions_data=positions_data)

    market_data = process_market(total_data=total_data)

    process_list = split_k(4, groups1)

    res = create_parellel(process_list=process_list,
                          callback=run_metrics,
                          positions_data=positions_data,
                          market_data=market_data,
                          strategy_setting=strategy_setting)
    res = list(itertools.chain.from_iterable(res))
    res1 = pd.DataFrame(res)
    base_path2 = os.path.join(base_path, method, g_instruments, 'distri')
    if not os.path.exists(base_path2):
        os.makedirs(base_path2)
    res1.to_feather(os.path.join(base_path2, "{0}.feather".format(n_parts)))
