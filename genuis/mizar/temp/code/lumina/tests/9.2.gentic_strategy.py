import os, sys, pdb, re
import pandas as pd
from ultron.factor.genetic.geneticist.operators import custom_transformer
import ultron.factor.empyrical as empyrical
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret
from lumina.genetic.geneticist.genetic import Gentic
from lumina.genetic.geneticist.engine import Engine
from lumina.genetic.geneticist.warehouse import sequential_gaind

# 'MRSquared',
two_operators_sets = [
    'MConVariance', 'MMASSI', 'MACCBands', 'MPWMA', 'MIChimoku', 'MRes',
    'MMeanRes', 'MCORR', 'MCoef', 'MSLMean', 'MSmart', 'MSharp', 'MSortino',
    'MINIMUM', 'MAXIMUM', 'ADDED', 'SUBBED', 'MUL', 'DIV', 'MOD'
]
#, 'MHMA', 'MDPO', 'MARETURNLog',
one_operators_sets = [
    'MA', 'MPERCENT', 'MMedian', 'MADiff', 'MADecay', 'MMAX', 'MMIN', 'MDPO',
    'MARGMAX', 'MARGMIN', 'MRANK', 'MQUANTILE', 'MCPS', 'MDIFF', 'MMaxDiff',
    'MMinDiff', 'MSUM', 'MPRO', 'MVARIANCE', 'MVHF', 'MDPO', 'MT3', 'MDEMA',
    'MIR', 'MSKEW', 'MKURT', 'MSTD', 'MNPOSITIVE', 'MAPOSITIVE', 'EMA', 'RSI',
    'WMA', 'MMaxDrawdown', 'MMDrawdown', 'SIGN', 'AVG', 'SQRT', 'DIFF', 'LOG2',
    'LOG10', 'LOG', 'EXP', 'FRAC', 'SIGLOG2ABS', 'SIGLOG10ABS', 'SIGLOGABS',
    'POW', 'ABS', 'ACOS', 'ASIN', 'NORMINV', 'CEIL', 'FLOOR', 'ROUND', 'TANH',
    'RELU', 'SHIFT', 'DELTA', 'SIGMOID', 'RETURNSimple', 'RETURNLog'
]

two_operators_sets = ['MCORR', 'MUL', 'MRes']
one_operators_sets = ['MRANK', 'ACOS']


def evolution(rootid, method):

    standard_score = 1.1
    custom_params = {
        'horizon': '1',
        rootid: rootid,
        'dethod': method,
        'strategy_settings': {
            'commission': 0.00023,
            'slippage': 1.7e-08,
            'size': 200
        },
        'gain': {
            'corr_threshold': 0.6,
            'fitness_scale': 0.7,
            'gain_threshold': 0.3
        },
        'adaptive': {
            "initial_alpha": 0.02,
            "target_penalty_ratio": 0.4,
            "adjustment_speed": 0.05,
            "lookback_period": 5
        },
        'warehouse': {
            "n_benchmark_clusters": 200,
            "distill_trigger_size": 20
        }
    }
    configure = {
        'n_jobs': 4,
        'population_size': 20,
        'tournament_size': 10,
        'init_depth': 3,
        'evaluate': 'both_evaluate',
        'method': 'fitness',
        'crossover': 0.3,
        'point_replace': 0.2,
        'hoist_mutation': 0.2,
        'subtree_mutation': 0.2,
        'point_mutation': 0.1,
        'generations': 128,
        'standard_score': standard_score,
        'rootid': rootid,
        'stopping_criteria': 100,
        'convergence': 0.00002,
        'custom_params': custom_params
    }
    filename = os.path.join('records', method, 'IF', 'factors',
                            "factors_data.feather")
    factors_data = pd.read_feather(filename)
    factors_data = factors_data.set_index('trade_time')
    factors_data = factors_data.loc['2023-04-01':]
    factor_columns = [
        col for col in factors_data.columns if col not in [
            'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
            'volume', 'openint', 'vwap'
        ]
    ]
    operators_sets = two_operators_sets + one_operators_sets
    operators_sets = custom_transformer(operators_sets)
    engine = Engine(population_size=configure['population_size'],
                    tournament_size=configure['tournament_size'],
                    init_depth=(1, configure['init_depth']),
                    generations=configure['generations'],
                    n_jobs=configure['n_jobs'],
                    stopping_criteria=configure['stopping_criteria'],
                    p_crossover=configure['crossover'],
                    p_point_mutation=configure['point_mutation'],
                    p_subtree_mutation=configure['subtree_mutation'],
                    p_hoist_mutation=configure['hoist_mutation'],
                    p_point_replace=configure['point_replace'],
                    rootid=configure['rootid'],
                    factor_sets=factor_columns,
                    standard_score=configure['standard_score'],
                    operators_set=operators_sets,
                    backup_cycle=1,
                    convergence=configure['convergence'],
                    fitness=callback_fitness,
                    save_model=callback_models,
                    custom_params=configure['custom_params'])
    engine.train(total_data=factors_data)


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    tournament_size = 20  #custom_params['tournament_size']
    standard_score = 1.0  #custom_params['standard_score']
    dethod = 'sac'  #custom_params['dethod']
    method = 'ascio'  #custom_params['method']

    candidate_positions = [program.position_data for program in best_programs]
    candidate_positions = pd.concat(candidate_positions, axis=1)

    best_programs = [p.output() for p in best_programs]
    best_programs = pd.DataFrame(best_programs)
    best_programs = best_programs.sort_values(by=['final_fitness'],
                                              ascending=False)

    dirs = os.path.join('temp', dethod, method, 'IF', 'evolution')
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


def adjust_returns(value):
    abs_value = abs(value)
    if abs_value > 1:
        return 0.9 if value > 0 else -0.9
    return value


def callback_fitness(factor_data, pos_data, total_data, signal_method,
                     strategy_method, factor_sets, custom_params,
                     default_value):
    strategy_settings = {}
    df = calculate_ful_ts_ret(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['a_ret'].apply(lambda x: 0.9 * (x / abs(x))
                                if abs(x) > 1 else x)
    #empyrical.cagr(returns=returns, period=empyrical.DAILY)
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    return fitness


evolution('33333', 'aa1')
