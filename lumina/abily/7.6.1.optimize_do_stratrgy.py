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
from kdutils.data import fetch_main_market
import ultron.factor.empyrical as empyrical
from kdutils.macro2 import *
from ultron.factor.genetic.geneticist.operators import *


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


def run():
    instruments = 'ims'
    method = 'aicso0'

    rootid = 200036

    ## 加载数据
    factors_data = fetch_data()
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    min_time = factors_data['trade_time'].min().strftime('%Y-%m-%d')
    max_time = factors_data['trade_time'].max().strftime('%Y-%m-%d')

    market_data = fetch_market(min_time, max_time, ['IM'])
    market_data['trade_time'] = pd.to_datetime(market_data['trade_time'])

    total_data = market_data.merge(factors_data, on=['trade_time', 'code'])
    total_data = total_data.set_index('trade_time')

    #exp1 = "MA(1,'transformed')"
    #dt1 = calc_factor(exp1, total_data=total_data, indexs=[], key='code')
    factor_columns = [
        col for col in total_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]
    strategy_info = {
        'name': '1111111',
        'formual': "MA(1,'transformed')",
        'strategy_method': 'trailing_atr_strategy',
        'strategy_params': {
            'atr_multiplier': 7.0,
            'atr_period': 10
        },
        'signal_method': 'quantile_signal',
        'signal_params': {
            'roll_num': 120,
            'threshold': 0.2
        }
    }

    SEARCH_RULES = {
        'default': {
            'range_pct': [0.7, 1.3],
            'range_pct': [1, 1.8],
            'step': 1,
            'fine_range_pct': [1, 1.1],
            'fine_step': 1
        },
        'atr_multiplier': {
            'range_pct': [1, 1.8],
            'step': 1.5,
            'fine_range_pct': [1, 1.1],
            'fine_step': 1
        },
        'atr_period': {
            'range_pct': [1, 1.8],
            'step': 10,
            'fine_range_pct': [1, 1.2],
            'fine_step': 5
        },
        'roll_num': {
            'range_pct': [0.8, 1.5],
            'step': 10,
            'fine_range_pct': [0.8, 1.2],
            'fine_step': 5
        },
        'threshold': {
            'range_pct': [0.8, 1.5],
            'step': 0.2,
            'fine_range_pct': [0.8, 1.5],
            'fine_step': 0.01
        }
    }

    strategy_settings = {
        #'capital': 10000000,
        'mode': COST_MODE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]] * 0.005,
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }

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
    rolling_sets = [100]
    threshold_sets = [0.4]
    signal_functions = signal_mapping[strategy_info['signal_method']](
        rolling_sets=rolling_sets, threshold_sets=threshold_sets)

    # c. 创建带参数的策略函数列表
    atr_period_sets = [8]  ## 使用一个参数即可
    atr_multiplier_sets = [10]
    maN_sets = [60]
    strategy_functions = strategy_mapping[strategy_info['strategy_method']](
        atr_multiplier_sets=atr_multiplier_sets,
        atr_period_sets=atr_period_sets,
        maN_sets=maN_sets)

    actuator = Actuator(k_split=32, callback_fitness=callback_fitness)

    optimizer = Optimizer(actuator=actuator,
                          total_data=total_data,
                          configure=configure,
                          search_rules=SEARCH_RULES,
                          signals_sets=signal_functions,
                          strategies_sets=strategy_functions,
                          factor_columns=factor_columns,
                          callback_fitness=callback_fitness,
                          k_split=4)

    top_10 = optimizer.optimize2(strategy_info,
                                 coarse_n_trials=500,
                                 fine_n_trials=30,
                                 top_n_results=10)
    results = [t.output() for t in top_10]
    results = pd.DataFrame(results)
    results = results[[
        'formual', 'final_fitness', 'strategy_method', 'strategy_params',
        'signal_method', 'signal_params'
    ]]
    print(results)


if __name__ == '__main__':
    run()
