### optuna + girdsearch 寻优


import pdb, os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from lumina.genetic.geneticist.mutation import Generator
from lumina.genetic.geneticist.mutation import Actuator
from lumina.genetic.signal import signal_mapping
from lumina.genetic.strategy import strategy_mapping
from lumina.genetic.geneticist.mutation import Optimizer
import ultron.factor.empyrical as empyrical
from kdutils.macro2 import *


def callback_fitness(factor_data, pos_data, total_data, signal_method,
                     strategy_method, factor_sets, custom_params,
                     default_value):
    strategy_settings = custom_params['strategy_settings']
    df = calculate_ful_ts_ret(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['a_ret']
    #empyrical.cagr(returns=returns, period=empyrical.DAILY)
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    return fitness


if __name__ == '__main__':

    strategy_info = {
        'name': '1111111',
        'formual':
        "MIChimoku(2,RSI(16,'dv001_5_10_1'),MDPO(2,'cj010_5_10_0'))",
        'strategy_method': 'trailing_atr_strategy',
        'strategy_params': {
            'atr_multiplier': 7.0,
            'atr_period': 10,
            'maN': 60
        },
        'signal_method': 'mean_signal',
        'signal_params': {
            'roll_num': 60,
            'threshold': 0.4
        }
    }

    # --- 1. 定义通用搜索规则 ---
    SEARCH_RULES = {
        'default': {
            'range_pct': [0.7, 1.3],
            'fine_range_pct': [0.9, 1.1]
        },
        'RSI': {
            'range_pct': [0.5, 1.5],
            'step': 5,
            'fine_range_pct': [0.9, 1.1],
            'fine_step': 1
        },
        'roll_num': {
            'range_pct': [0.5, 1.5],
            'step': 10,
            'fine_range_pct': [0.8, 1.2],
            'fine_step': 5
        },
    }

    instruments = 'ims'
    method = 'aicso0'
    strategy_settings = {
        #'capital': 10000000,
        'mode': COST_MODE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

    filename = os.path.join(
        base_path, method, instruments,
        DATAKIND_MAPPING[str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])],
        'train_data.feather')

    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.set_index('trade_time')
    factor_columns = [
        col for col in factors_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]

    strategy_settings = {
        #'capital': 10000000,
        'mode': COST_MODE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }
    rootid = 200036
    configure = {
        'rootid': rootid,
        'backup_cycle': 1,
        'coverage_rate': 0.7,
        'custom_params': {
            'g_instruments': instruments,
            'dethod': method,
            'strategy_settings': strategy_settings,
            'task_id': rootid,
            'method': PERFORMANCE_MAPPING[str(rootid)],
        }
    }

    # b. 创建带参数的信号函数列表
    rolling_sets = [50]
    threshold_sets = [0.2]
    signal_functions = signal_mapping[strategy_info['signal_method']](
        rolling_sets=rolling_sets, threshold_sets=threshold_sets)

    # c. 创建带参数的策略函数列表
    atr_period_sets = [8]
    atr_multiplier_sets = [7]
    maN_sets = [60]
    pdb.set_trace()
    strategy_functions = strategy_mapping[strategy_info['strategy_method']](
        atr_multiplier_sets=atr_multiplier_sets,
        atr_period_sets=atr_period_sets,
        maN_sets=maN_sets)

    actuator = Actuator(k_split=32, callback_fitness=callback_fitness)

    optimizer = Optimizer(actuator=actuator,
                          total_data=factors_data,
                          configure=configure,
                          search_rules=SEARCH_RULES,
                          signals_sets=signal_functions,
                          strategies_sets=strategy_functions,
                          factor_columns=factor_columns,
                          callback_fitness=callback_fitness)

    ### 
    top_10 = optimizer.optimize1(strategy_info, n_trials=5, top_n_results=3)
