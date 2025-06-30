import pandas as pd
import pdb, argparse
import os, pdb, math, itertools

from lumina.evolution.genetic import merge_factors
from lumina.evolution.engine import Engine
from lumina.evolution.warehouse import sequential_gain
from ultron.factor.genetic.geneticist.operators import custom_transformer

from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import *


def load_datasets(variant):
    dirs = os.path.join(
        base_path, variant['method'], 'normal', variant['g_instruments'],
        'rollings', 'normal_factors3',
        "{0}_{1}".format(variant['categories'], variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['swindow'])))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_filename = os.path.join(
            dirs, "original_factors_train_{0}.feather".format(i))
        val_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        test_filename = os.path.join(
            dirs, "original_factors_test_{0}.feather".format(i))
        train_data = pd.read_feather(train_filename)
        val_data = pd.read_feather(val_filename)
        test_data = pd.read_feather(test_filename)

        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data, test_data)
    return data_mapping


def create_data(variant, merge=True):
    variant['g_start_pos'] = variant['g_start_pos'] - 1
    data_mapping = load_datasets(variant)
    res = []
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_data, val_data, test_data = data_mapping[i]
        res.append(train_data)
        res.append(val_data)
        res.append(test_data)
    total_data = pd.concat(res, axis=0).sort_values(by=['trade_time', 'code'])
    total_data = total_data[3100:].reset_index(drop=True)
    return total_data


def callback_fitness(factor_data, total_data, factor_sets, custom_params,
                     default_value):
    rolling_window = 60
    returns = total_data[['trade_time', 'code', 'nxt1_ret']]
    data = factor_data.reset_index().merge(returns, on=['trade_time', 'code'])
    data = data.set_index(
        ['trade_time',
         'code']).dropna(subset=['nxt1_ret', 'transformed']).fillna(0)

    ranked_features = data['transformed'].rank(method='first')
    ranked_return = data['nxt1_ret'].rank(method='first')
    rolling_ic = ranked_features.rolling(
        window=rolling_window,
        min_periods=int(rolling_window * 0.5)).corr(ranked_return)

    ic_mean = rolling_ic.mean()
    ic_std = rolling_ic.std()
    icir = ic_mean / (ic_std + 1e-6)
    fitness = ic_mean if math.fabs(icir) > 0.1 else 0
    #fitness = 0.5 * (icir) + 0.5 * math.fabs(r_ic_mean)
    return fitness


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    candidate_factors = merge_factors(best_programs=best_programs)
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score']
    dethod = custom_params['dethod']
    method = custom_params['method']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)
    dirs = os.path.join(base_path, dethod, method,
                        custom_params['g_instruments'], 'evolution')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    names = rootid
    programs_filename = os.path.join(dirs, f'programs_{names}.feather')
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)

    factors_file = os.path.join(dirs, f'factors_{names}.feather')
    if os.path.exists(factors_file):
        old_factors = pd.read_feather(factors_file).set_index(
            ['trade_time', 'code'])
        candidate_factors = pd.concat([old_factors, candidate_factors], axis=1)
        #duplicate_columns = candidate_factors.columns[candidate_factors.columns.duplicated()]
        candidate_factors = candidate_factors.loc[:, ~candidate_factors.
                                                  columns.duplicated()]
        candidate_factors = candidate_factors.sort_values(
            by=['trade_time', 'code'])

    ### 相关性过滤剔除
    returns_series = total_data.reset_index().set_index(['trade_time',
                                                         'code'])['nxt1_ret']

    selected_factors = sequential_gain(basic_factors=candidate_factors,
                                       returns_series=returns_series,
                                       ic_threshold=custom_params['gain']['fitness_threshold'],
                                       corr_threshold=custom_params['gain']['corr_threshold'],
                                       gain_threshold=custom_params['gain']['gain_threshold'])

    print("candidate_factors 共:{0}, selected_factors 共:{1}, 减少:{2}".format(
        len(candidate_factors.columns), len(selected_factors.columns),
        len(candidate_factors.columns) - len(selected_factors.columns)))
    ## 筛选best_programs
    if selected_factors.empty:
        print(best_programs)
        return
    factors_columns = selected_factors.columns
    best_programs = best_programs[best_programs.name.isin(factors_columns)]
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
    candidate_factors = candidate_factors.loc[:, ~candidate_factors.columns.
                                              duplicated()]
    candidate_factors[final_programs.name.tolist()].reset_index().to_feather(
        factors_file)


def evolution(variant):
    two_operators_sets = [
        'MConVariance', 'MMASSI', 'MACCBands', 'MPWMA', 'MIChimoku', 'MRes',
        'MMeanRes', 'MCORR', 'MCoef', 'MSLMean', 'MSmart', 'MSharp',
        'MSortino', 'MINIMUM', 'MAXIMUM', 'ADDED', 'SUBBED', 'MUL', 'DIV',
        'MOD'
    ]
    one_operators_sets = [
        'MA', 'MPERCENT', 'MMedian', 'MADiff', 'MADecay', 'MMAX', 'MMIN',
        'MDPO', 'MARGMAX', 'MARGMIN', 'MRANK', 'MQUANTILE', 'MCPS', 'MDIFF',
        'MMaxDiff', 'MMinDiff', 'MSUM', 'MPRO', 'MVARIANCE', 'MVHF', 'MDPO',
        'MT3', 'MDEMA', 'MIR', 'MSKEW', 'MKURT', 'MSTD', 'MNPOSITIVE',
        'MAPOSITIVE', 'EMA', 'RSI', 'WMA', 'MMaxDrawdown', 'MMDrawdown',
        'SIGN', 'AVG', 'SQRT', 'DIFF', 'LOG2', 'LOG10', 'LOG', 'EXP', 'FRAC',
        'SIGLOG2ABS', 'SIGLOG10ABS', 'SIGLOGABS', 'POW', 'ABS', 'ACOS', 'ASIN',
        'NORMINV', 'CEIL', 'FLOOR', 'ROUND', 'TANH', 'RELU', 'SHIFT', 'DELTA',
        'SIGMOID', 'RETURNSimple', 'RETURNLog'
    ]
    total_data = create_data(variant=variant)

    factor_columns = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret', 'price']
    ]

    population_size = 500
    tournament_size = 100
    standard_score = 0.5

    operators_sets = two_operators_sets + one_operators_sets
    operators_sets = custom_transformer(operators_sets)
    rootid = '200008'
    #tournament_size = 15
    standard_score = 0.1
    custom_params = {
        'horizon': '1',
        'rootid': rootid,
        'tournament_size': tournament_size,
        'standard_score': standard_score,
        'dethod': 'ic',
        'method': variant['method'],
        'g_instruments': variant['g_instruments'],
        'gain': {
            'corr_threshold': 0.75,  ## 相关性
            'fitness_scale': 0.5,  ## 标准分缩放
            'gain_threshold': 0.15  ## 增量阈值
        },
        'adaptive': {
            "initial_alpha": 0.02,
            "target_penalty_ratio": 0.4,
            "adjustment_speed": 0.05,
            "lookback_period": 5
        },
        'warehouse': {
            "n_benchmark_clusters": 300,  ##
            "distill_trigger_size": 100
        },
        'threshold': {
            "initial_threshold": standard_score * 0.6,  ## 初始分
            "target_percentile": 0.75,
            "min_threshold": standard_score * 0.7,
            "max_threshold": standard_score * 5,
            "adjustment_speed": 0.1
        }
    }

    configure = {
        'n_jobs': 8,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 6,
        'evaluate': 'both_evaluate',
        'method': 'fitness',
        'crossover': 0.3,
        'point_replace': 0.2,
        'hoist_mutation': 0.2,
        'subtree_mutation': 0.2,
        'point_mutation': 0.1,
        'generations': 10,
        'standard_score': 0.1,
        'stopping_criteria': 5,
        'convergence': 0.0002,
        'custom_params': custom_params,
        'rootid': rootid,
    }
    pdb.set_trace()
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
                    operators_sets=operators_sets,
                    backup_cycle=1,
                    convergence=configure['convergence'],
                    fitness=callback_fitness,
                    save_model=callback_models,
                    custom_params=configure['custom_params'])

    factors_data = total_data.reset_index().set_index('trade_time')
    engine.train(total_data=factors_data)

    print('-->')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_days',
                        type=int,
                        default=60,
                        help='Training days')  ## 训练天数
    parser.add_argument('--val_days',
                        type=int,
                        default=10,
                        help='Validation days')  ## 验证天数

    parser.add_argument('--freq',
                        type=int,
                        default=10,
                        help='Frequency of training')  ## 多少个周期训练一次

    parser.add_argument('--method',
                        type=str,
                        default='aicso4',
                        help='Method name')  ## 方法

    parser.add_argument('--code', type=str, default='RB', help='Code')  ## 代码

    parser.add_argument('--g_instruments',
                        type=str,
                        default='rbb',
                        help='Instruments')  ## 标的

    parser.add_argument('--categories',
                        type=str,
                        default='o2o',
                        help='Categories')  ## 类别

    parser.add_argument('--horizon',
                        type=int,
                        default=1,
                        help='Prediction horizon')  ## 预测周期

    parser.add_argument('--nc', type=int, default=2,
                        help='Standard method')  ## 标准方式

    parser.add_argument('--swindow',
                        type=int,
                        default=60,
                        help='Rolling window')  ## 滚动窗口

    parser.add_argument('--g_start_pos',
                        type=int,
                        default=24,
                        help='Start position')  ## 开始位置

    parser.add_argument('--g_max_pos',
                        type=int,
                        default=1,
                        help='Max position')  ## 最大位置

    args = parser.parse_args()
    evolution(variant=vars(args))
