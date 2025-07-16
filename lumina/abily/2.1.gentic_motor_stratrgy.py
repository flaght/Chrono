### 通过寻优算法 挖掘策略
import datetime, pdb, os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from kdutils.macro2 import *
from kdutils.operators_sets import operators_sets
#from kdutils.callback import callback_fitness, callback_models
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from lumina.genetic.geneticist.warehouse import sequential_gaind
import ultron.factor.empyrical as empyrical
from lumina.genetic import Motor
from ultron.factor.genetic.geneticist.operators import calc_factor


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score']
    dethod = custom_params['dethod']
    method = custom_params['method']

    candidate_positions = [program.position_data for program in best_programs]
    candidate_positions = pd.concat(candidate_positions, axis=1)

    best_programs = [p.output() for p in best_programs]
    best_programs = pd.DataFrame(best_programs)
    best_programs = best_programs.sort_values(by=['final_fitness'],
                                              ascending=False)

    #dirs = os.path.join('temp', dethod, method,
    #                    INSTRUMENTS_CODES[g_instruments],'evolution')
    dirs = os.path.join('temp', dethod, str(rootid), 'evolution')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    names = rootid
    programs_filename = os.path.join(dirs, f'programs_{names}.feather')
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)
        best_programs = best_programs.drop_duplicates(subset=['name'])

    positions_file = os.path.join(dirs, f'positions_{names}.feather')
    if os.path.exists(positions_file):
        old_positions = pd.read_feather(positions_file).set_index(
            ['trade_time', 'code'])
        candidate_positions = pd.concat([old_positions, candidate_positions],
                                        axis=1)
        #duplicate_columns = candidate_factors.columns[candidate_factors.columns.duplicated()]
        candidate_positions = candidate_positions.loc[:, ~candidate_positions.
                                                      columns.duplicated()]
        candidate_positions = candidate_positions.sort_values(
            by=['trade_time', 'code'])

    selected_positions = sequential_gaind(
        candidate_positions=candidate_positions,
        programs_data=best_programs,
        total_data=total_data,
        custom_params=custom_params,
        corr_threshold=custom_params['gain']['corr_threshold'],
        fitness_threshold=custom_params['gain']['fitness_threshold'],
        gain_threshold=custom_params['gain']['gain_threshold'])
    if selected_positions is None:
        print("no selected program")
        return
    print("candidate_factors 共:{0}, selected_positions 共:{1}, 减少:{2}".format(
        len(candidate_positions.columns), len(selected_positions.columns),
        len(candidate_positions.columns) - len(selected_positions.columns)))

    ## 筛选best_programs
    if selected_positions.empty:
        print(best_programs)
        return

    positions_columns = selected_positions.columns
    best_programs = best_programs[best_programs.name.isin(positions_columns)]
    best_programs = best_programs.drop_duplicates(subset=['name'])
    final_programs = best_programs[best_programs['final_fitness'] >
                                   standard_score]
    if final_programs.shape[0] < tournament_size:
        best_programs = best_programs.sort_values('final_fitness',
                                                  ascending=False)
        final_programs = best_programs.head(tournament_size)

    final_programs = final_programs.sort_values('final_fitness',
                                                ascending=False)

    print(final_programs[[
        'name', 'formual', 'final_fitness', 'raw_fitness', 'max_corr',
        'penalty', 'alpha'
    ]].head(10))
    print(programs_filename)

    ### 去重
    final_programs = final_programs.drop_duplicates(subset=['name'])
    final_programs.reset_index(drop=True).to_feather(programs_filename)
    ## 保留最后和final_programs一致的因子
    selected_positions = selected_positions.loc[:, ~selected_positions.columns.
                                                duplicated()]
    selected_positions[final_programs.name.tolist()].reset_index().to_feather(
        positions_file)


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


def train(method, instruments):
    rootid = INDEX_MAPPING[INSTRUMENTS_CODES[instruments]]

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

    population_size = 500#500  #500
    tournament_size = 100#100  #100
    standard_score = 0.5
    strategy_settings = {
        #'capital': 10000000,
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }
    configure = {
        'n_jobs': 16,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 4,
        'rootid': rootid,
        'generations': 3,
        'custom_params': {
            'g_instruments': instruments,
            'dethod': method,
            'tournament_size': tournament_size,
            'standard_score': standard_score,
            'strategy_settings': strategy_settings,
            'task_id': rootid,
            'method': PERFORMANCE_MAPPING[str(rootid)],
            'filter_custom': {
                #'returns':
                #FILTER_YEAR_MAPPING[INSTRUMENTS_CODES[g_instruments]]
            },
            'gain': {
                'corr_threshold': 0.5,
                'fitness_scale': 0.7,
                'gain_threshold': 0.1
            },
            'adaptive': {
                "initial_alpha": 0.01,
                "target_penalty_ratio": 0.02,
                "adjustment_speed": 0.05,
                "lookback_period": 5
            },
            'warehouse': {
                "n_benchmark_clusters": 200,
                "distill_trigger_size": 100
            },
            'threshold': {
                "initial_threshold": 0.08,
                "target_percentile": 0.75,
                "min_threshold": 0.08,
                "max_threshold": 4.0,
                "adjustment_speed": 0.1
            }
        }
    }
    motor = Motor(factor_columns=factor_columns,
                  callback_fitness=callback_fitness,
                  callback_save_model=callback_models)
    pdb.set_trace()
    motor.calculate(total_data=factors_data,
                    configure=configure,
                    operators_sets=operators_sets,
                    signals_sets=None,
                    strategies_sets=None)


def test1(method, g_instruments):
    filename = os.path.join(base_path, method, g_instruments, 'level2',
                            'train_data.feather')
    total_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    pdb.set_trace()
    rootid = INDEX_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])
    total_data = total_data.set_index('trade_time')
    factor_columns = [
        col for col in total_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]
    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }
    pdb.set_trace()
    expression = "MMIN(14,MACCBands(14,'price_imbalance_3','smart_money_out'))"
    signal_method = 'triple_barrier_signal'
    strategy_method = 'trailing_atr_strategy'
    signal_params = {'roll_num': 25, 'threshold': 0.6}
    strategy_params = {
        'atr_multiplier': 2.0,
        'atr_period': 10.0,
        'holding_period': None,
        'max_volume': 1,
        'trailing_percent': None
    }
    signal_params = {
        key: value
        for key, value in signal_params.items() if value is not None
    }
    strategy_params = {
        key: value
        for key, value in strategy_params.items() if value is not None
    }

    indexs = ['trade_date']
    key = 'code'
    backup_cycle = 1
    factor_data = calc_factor(expression, total_data, indexs, key)
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data['transformed'] = np.where(
        np.abs(factor_data.transformed.values) > 0.000001,
        factor_data.transformed.values, np.nan)
    factor_data = factor_data.loc[factor_data.index.unique()[backup_cycle:]]

    factor_data1 = factor_data.reset_index().set_index(['trade_time', 'code'])

    cycle_total_data = total_data.copy()
    cycle_total_data = cycle_total_data.loc[cycle_total_data.index.unique()
                                            [backup_cycle:]]

    total_data1 = cycle_total_data.reset_index().set_index(
        ['trade_time', 'code']).unstack()

    pos_data = eval(signal_method)(factor_data=factor_data1, **signal_params)

    pos_data1 = eval(strategy_method)(signal=pos_data,
                                      total_data=total_data1,
                                      **strategy_params)

    df = calculate_ful_ts_ret(pos_data=pos_data1,
                              total_data=total_data1,
                              strategy_settings=strategy_settings)

    returns = df['a_ret']
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)

if __name__ == '__main__':
    method = 'aicso0'
    instruments = 'rbb'
    train(method=method, instruments=instruments)
