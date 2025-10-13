import pandas as pd
import numpy as np
import pdb, argparse, random
import os, pdb, math, itertools
from scipy import stats  # 用于计算秩相关IC
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix

from ultron.factor.genetic.geneticist.operators import custom_transformer
from lumina.evolution.genetic import merge_factors
from lumina.evolution.engine import Engine
from lumina.evolution.warehouse import sequential_gain

from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

#from lib.aux001 import *
from lib.cux001 import *


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    candidate_factors = merge_factors(best_programs=best_programs)
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score'] * 0.01
    dethod = custom_params['dethod']
    method = custom_params['method']
    return_name = custom_params['return_name']
    session = custom_params['session']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)
    #dirs = os.path.join(base_path, "gentic", dethod, method,
    #                    custom_params['g_instruments'], return_name)
    dirs = os.path.join(base_path, method,
                        custom_params['g_instruments'], "gentic", dethod,
                        str(rootid), return_name, str(session))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    programs_filename = os.path.join(dirs,
                                     f'programs_{rootid}_{session}.feather')
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)

    factors_file = os.path.join(dirs, f'factors_{rootid}_{session}.feather')
    if os.path.exists(factors_file):
        old_factors = pd.read_feather(factors_file).set_index(
            ['trade_time', 'code'])
        candidate_factors = pd.concat([old_factors, candidate_factors], axis=1)
        candidate_factors = candidate_factors.loc[:, ~candidate_factors.
                                                  columns.duplicated()]
        candidate_factors = candidate_factors.sort_values(
            by=['trade_time', 'code'])

    ### 相关性过滤剔除
    returns_series = total_data.reset_index().set_index(['trade_time',
                                                         'code'])['nxt1_ret']

    if 'gain' in custom_params:
        selected_factors = sequential_gain(
            basic_factors=candidate_factors,
            returns_series=returns_series,
            ic_threshold=custom_params['gain']['fitness_threshold'],
            corr_threshold=custom_params['gain']['corr_threshold'],
            gain_threshold=custom_params['gain']['gain_threshold'])
    else:
        selected_factors = candidate_factors

    if selected_factors is None:
        print("no selected program")
        return

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

    final_programs = best_programs[
        (best_programs['final_fitness'] > standard_score)
        & (best_programs['final_fitness'] > 0)]

    final_programs = final_programs.drop_duplicates(subset=['features'])
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


def callback_fitness(factor_data, total_data, factor_sets, custom_params,
                     default_value):
    fee = 0.000003
    min_ic_threshold = 0.001  # 全周期IC至少要大于1%
    try:
        returns = total_data[['trade_time', 'code', 'nxt1_ret']]
        if 'trade_time' not in factor_data.columns:
            factor_data = factor_data.reset_index()
        is_on_mark = returns['trade_time'].dt.minute % int(
            custom_params['horizon']) == 0
        returns = returns[is_on_mark]

        is_on_mark = factor_data['trade_time'].dt.minute % int(
            custom_params['horizon']) == 0
        factor_data = factor_data[is_on_mark]

        data = pd.merge(factor_data,
                        returns,
                        on=['trade_time', 'code'],
                        how='inner')
        data['trade_time'] = pd.to_datetime(data['trade_time'])
        data.set_index('trade_time', inplace=True)

        data = data[['transformed', 'nxt1_ret']]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
    except Exception:
        return 0.0

    if data['transformed'].std() < 1e-8:
        return 0.0

    evaluate1 = FactorEvaluate1(factor_data=data.reset_index(),
                                factor_name='transformed',
                                ret_name='nxt1_ret',
                                roll_win=240,
                                fee=0.000,
                                scale_method='roll_zscore')
    stats_df = evaluate1.run()
    fitness = math.fabs(stats_df['ic_mean'])
    if not np.isfinite(fitness):
        return 0.0
    if abs(fitness) < min_ic_threshold:
        return 0.0  # 预测能力太弱，直接淘汰

    ## 不是有限数   ## 为nan
    if not np.isfinite(stats_df['calmar']) or np.isnan(
            stats_df['calmar']) or stats_df['calmar'] <= 0:
        return 0.0

    if not np.isfinite(stats_df['sharpe1']) or np.isnan(
            stats_df['sharpe1']) or stats_df['sharpe1'] <= 0:
        return 0.0

    if stats_df['calmar'] < 1.0 or stats_df['sharpe2'] < 1.0:
        return 0.0
    ## ic 绝对值大于1 不正常
    if fitness >= 1:
        return 0.0
    '''
    if data['transformed'].std() < 1e-8:
        return 0.0

    ic, _ = stats.spearmanr(data['transformed'], data['nxt1_ret'])
    if not np.isfinite(ic):
        return 0.0
    if abs(ic) < min_ic_threshold:
        return 0.0  # 预测能力太弱，直接淘汰
    fitness = math.fabs(ic)
    '''
    '''
    try:
        #   i. 因子缩放 (使用全样本z-score，速度快)
        factor_series = data['transformed']
        mu = factor_series.mean()
        sigma = factor_series.std()
        pos = ((factor_series - mu) / (sigma + 1e-8)).clip(-3, 3) / 3

        # 如果IC为负，反转头寸
        if ic < 0:
            pos *= -1

        #   ii. 计算净收益
        gross_ret = pos * data['nxt1_ret']
        turnover = np.abs(np.diff(pos, prepend=0))
        net_ret = (gross_ret - fee * turnover).fillna(0)

        #   iii. 计算绩效指标
        if net_ret.std() < 1e-8:
            sharpe = 0
            calmar = 0
        else:
            nav = (1 + net_ret).cumprod()
            total_ret = nav.iloc[-1] - 1

            sharpe = net_ret.mean() / net_ret.std() if net_ret.mean().std(
            ) != 0 else 0
            max_dd = (nav / nav.cummax() - 1).min()
            calmar = total_ret / abs(max_dd) if max_dd != 0 else 0
    except Exception:
        # PNL计算出错，返回0
        return 0.0

    factor_autocorr = data['transformed'].autocorr(lag=1)
    if abs(factor_autocorr) > max_factor_autocorr_threshold:
        return 0.0
    sharpe = max(0, sharpe)
    calmar = max(0, calmar)
    fitness = (0.5 * abs(ic) * 10) + (0.5 * (0.6 * sharpe + 0.4 * calmar))
    # 乘以10是为了让IC的量级与Sharpe/Calmar大致相当

    if not np.isfinite(fitness):
        return 0.0
    '''
    return fitness


def train(method, instruments, period, session, task_id, count=0):
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
        'SIGMOID', 'LAST'
    ]
    rootid = task_id  #INDEX_MAPPING[INSTRUMENTS_CODES[instruments]]
    ## 加载数据
    ## 加载因子+ 基础数据
    total_factors = fetch_temp_data(method=method,
                                    task_id=rootid,
                                    instruments=instruments,
                                    datasets=['train', 'val'])

    total_returns = fetch_temp_returns(method=method,
                                       instruments=instruments,
                                       datasets=['train', 'val'],
                                       category='returns')
    total_data = total_factors.merge(total_returns, on=['trade_time', 'code'])

    total_data.filter(regex="^nxt1").columns.to_list()
    nxt1_columns = total_data.filter(regex="^nxt1").columns.to_list()
    basic_columns = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint'
    ]

    regex_pattern = r'^[^_]+_(5|10|15)_.*'
    not_columns = total_data.columns[total_data.columns.str.contains(
        regex_pattern)]
    factor_columns = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code', 'symbol'] + nxt1_columns +
        basic_columns + ['time_weight', 'equal_weight'] + not_columns.tolist()
    ]

    ## 随机取个数

    ##
    #if feature_count > 0:
    #    pdb.set_trace()
    pdb.set_trace()
    factor_columns = factor_columns if count == 0 else random.sample(
        factor_columns, count)
    ''' 
    factor_columns = [
        'tc017_1_2_1', 'ixy001_1_2_1', 'cr028_1_2_0', 'oi011_1_2_0',
        'dv007_1_2_0', 'cr029_1_2_0', 'tf002_1_2_0', 'cj003_1_2_0',
        'oi026_1_2_0', 'tc010_1_2_0', 'tf019_1_2_0', 'tn009_1_2_1_1',
        'cr015_1_2_0', 'tc010_1_2_1', 'tn009_1_2_1_4', 'cr056_1_2_1',
        'cr057_1_2_1', 'oi027_1_2_0', 'tn006_1_2_0', 'cr018_1_2_0',
        'cr035_1_2_0', 'tn008_1_2_0_1', 'tn008_1_2_1_1', 'rv008_1_2_1_2',
        'ixy003_1_2_0', 'tc013_1_2_0', 'oi029_2_3_0', 'tn005_1_2_0',
        'cr009_1_2_0', 'cr009_1_2_1', 'oi001_1_2_0', 'oi013_1_2_0',
        'tf006_1_2_0', 'cr042_1_2_0', 'tn003_1_1_2_3_1', 'tc017_1_2_0',
        'dv003_1_2_0', 'cr056_1_2_0', 'cr057_1_2_0', 'cr027_1_2_1',
        'oi001_1_2_1', 'tn008_1_2_0_4', 'tn003_2_1_2_3_0', 'rv010_1_2_1_2',
        'iv011_1_2_0', 'tc005_1_1_2_0', 'tc012_1_1_2_0', 'tv017_1_2_0',
        'tc014_1_1_2_0', 'tc006_1_2_0', 'rv008_1_2_0_2', 'iv008_1_2_0',
        'tv006_1_2_1', 'cr006_1_2_0', 'oi041_2_3_1', 'cj012_1_2_0',
        'tc004_1_1_2_0', 'db006_1_2_0', 'tc008_1_2_0', 'tn008_1_2_1_4',
        'tf001_1_2_0', 'tv001_1_2', 'dv011_1_2_0', 'iv007_2_3_0',
        'xy005_1_2_1', 'xy004_1_2_1', 'tn001_1_2_1', 'rv006_1_2_1_2',
        'iv010_1_2_0', 'cr007_1_2_1', 'ixy013_1_2_0', 'dv009_1_2_0',
        'iv007_2_3_1', 'rv006_1_2_0_2', 'dv001_2_3_1', 'oi042_1_2_0',
        'tn001_1_2_0', 'cr019_1_2_0', 'cr008_1_2_0', 'cr008_1_2_1',
        'iv007_5_10_1', 'cr049_1_2_0', 'cj011_1_2_0', 'oi042_1_2_1',
        'oi008_1_2_1', 'tc016_1_1_2_1', 'cj011_1_2_1', 'cj006_1_2_1',
        'oi041_2_3_0', 'ixy007_1_2_1', 'tc001_2_3_0', 'tn009_1_2_0_1',
        'cr047_1_2_0', 'cr047_1_2_1', 'tv019_1_2_1', 'tn009_1_2_1_2',
        'tn009_1_2_1_3', 'dv003_1_2_1', 'rv002_1_2_1_2', 'ixy014_1_2_1',
        'iv011_1_2_1', 'tn008_1_2_0_3', 'oi004_1_2_1', 'rv010_2_3_0_2',
        'rv003_1_2_1_2', 'oi025_1_2_0', 'tn008_1_2_1_3', 'db006_1_2_1',
        'ixy008_1_2_1', 'oi008_1_2_0', 'db007_1_2_1', 'dv002_1_2_1',
        'iv007_1_2_0', 'tn009_1_2_0_4', 'cr042_1_2_1', 'rv005_1_2_1_2',
        'cr027_1_2_0', 'tn009_1_2_0_2', 'dv003_2_3_1', 'tn009_1_2_0_3',
        'tc007_1_2_1', 'tc007_1_2_0', 'cr061_1_2_1', 'iv007_10_15_1',
        'tc012_1_1_2_1', 'tv017_1_2_1', 'tc005_1_1_2_1', 'oi039_1_2_1',
        'tc006_1_2_1', 'tc014_1_1_2_1', 'tv019_2_3_1', 'tn003_2_1_2_3_1',
        'iv010_1_2_1', 'tv011_1_1_2_1', 'oi014_2_3_1', 'cr020_1_2_0',
        'tn008_1_2_1_2', 'rv004_2_3_1_2', 'rv005_1_2_1_1', 'tv018_1_2_0',
        'iv008_1_2_1', 'tn008_2_3_1_2', 'cr041_2_2_3_0', 'tf006_1_2_1',
        'rv005_1_2_0_2', 'rv002_2_3_0_2', 'oi032_2_2_3_1', 'rv004_2_3_0_2',
        'tv002_1_1_2_1', 'rv004_1_2_1_2', 'rv003_2_3_1_2', 'oi037_1_2_1',
        'oi001_2_3_1', 'rv002_2_3_1_2', 'tn008_2_3_1_4', 'iv012_1_2_1',
        'tn004_1_2_1', 'oi032_2_2_3_0', 'oi034_1_2_1', 'rv001_2_3_0_2',
        'cr017_1_2_0', 'tv015_1_2_0', 'xy005_1_2_0', 'xy004_1_2_0',
        'oi016_1_2_1', 'ixy001_1_2_0', 'tc015_1_2_1', 'tn008_2_3_1_3',
        'oi032_1_1_2_1', 'oi023_1_2_1', 'tv004_1_2_1', 'ixy014_1_2_0',
        'db005_1_2_1', 'tv002_2_2_3_1', 'oi045_1_2_1', 'ixy006_1_2_1',
        'tv008_1_2_0', 'oi016_1_2_0', 'rv003_2_3_0_2', 'tc003_1_2_1',
        'oi022_1_2_1', 'tc013_1_2_1', 'tc017_2_3_1', 'tn008_1_2_0_2',
        'cr012_1_2_0', 'tv014_1_2_1', 'tc010_2_3_1', 'tn006_1_2_1',
        'iv007_1_2_1', 'cr012_1_2_1', 'oi046_1_2_1', 'fz002_1_2_1',
        'oi040_2_3_0', 'tc016_2_2_3_0', 'dv005_1_2_1', 'ha004_2_3_1',
        'db007_1_2_0', 'dv006_1_2_1', 'tv011_2_2_3_1', 'tn008_2_3_0_2',
        'gd002_1_2_1', 'oi033_1_2_1', 'tv002_1_1_2_0', 'oi042_2_3_1',
        'oi041_1_2_1', 'rv004_1_2_1_1', 'oi031_1_2_1', 'tc016_1_1_2_0',
        'tv007_1_2_1', 'cj006_1_2_0', 'cr033_1_2_1', 'cr049_1_2_1',
        'dv001_2_3_0', 'cj010_1_2_1', 'ixy003_1_2_1', 'cr048_2_3_1',
        'tv018_1_2_1', 'oi025_1_2_1', 'cr007_2_3_1', 'oi003_1_2_1',
        'tf001_1_2_1', 'ixy007_1_2_0', 'rv006_2_3_1_2', 'oi036_1_2_1',
        'rv005_1_2_0_1'
    ][0:100]
    '''
    return_name = "nxt1_ret_{}h".format(period)
    ### 评估是才聚合
    '''
    ### 聚合处理 K线数据
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'value': 'sum',
        'volume': 'sum',  # 成交量是流量，用 'sum'
        'openint': 'last'  # 持仓量是存量，用 'last'
    }
    market_data = total_data[['trade_time', 'code'] + basic_columns]
    market_data_indexed = market_data.set_index('trade_time')
    agg_market_data = market_data_indexed.resample(
        '{0}T'.format(period), label='right',
        closed='right').agg(aggregation_rules)
    '''
    pdb.set_trace()
    if str(rootid) != '200037':
        agg_market_data = total_data[['trade_time', 'code'] + basic_columns]
        ###使用原始因子
        factors_data = total_data[['trade_time', 'code'] +
                                  factor_columns].merge(
                                      agg_market_data,
                                      on=['trade_time', 'code'
                                          ]).merge(total_returns[[
                                              'trade_time', 'code', return_name
                                          ]],
                                                   on=['trade_time', 'code'])
    else:
        factors_data = total_data[['trade_time', 'code'] +
                                  factor_columns].merge(
                                      total_returns[[
                                          'trade_time', 'code', return_name
                                      ]],
                                      on=['trade_time', 'code'])

    factors_data.rename(columns={return_name: 'nxt1_ret'}, inplace=True)
    operators_sets = two_operators_sets + one_operators_sets
    operators_sets = custom_transformer(operators_sets)
    #rootid = '200036'
    population_size = 80
    tournament_size = 30
    standard_score = 0.1
    custom_params = {
        'horizon': str(period),
        'rootid': rootid,
        'tournament_size': tournament_size,
        'standard_score': standard_score,
        'dethod': 'ic',
        'method': method,
        'g_instruments': instruments,
        'return_name': return_name,
        'session': session,
        #'gain': {
        #    'corr_threshold': 0.75,  ## 相关性
        #    'fitness_scale': 0.5,  ## 标准分缩放
        #    'gain_threshold': 0.15  ## 增量阈值
        #},
        #'adaptive': {
        #    "initial_alpha": 0.02,
        #    "target_penalty_ratio": 0.4,
        #    "adjustment_speed": 0.05,
        #    "lookback_period": 5
        #},
        #'warehouse': {
        #    "n_benchmark_clusters": 300,  ##
        #    "distill_trigger_size": 100
        #},
        'threshold': {
            "initial_threshold": standard_score * 0.4,  ## 初始分
            "target_percentile": 0.35,  ## ## 目标分位数
            "min_threshold": standard_score * 0.2,
            "max_threshold": standard_score * 5,
            "adjustment_speed": 0.01  # ##  调整速度 (EMA平滑系数)
        }
    }

    configure = {
        'n_jobs': 1,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 4,
        'evaluate': 'both_evaluate',
        'method': 'fitness',
        'crossover': 0.3,
        'point_replace': 0.3,
        'hoist_mutation': 0.1,
        'subtree_mutation': 0.1,
        'point_mutation': 0.3,
        'generations': 4,
        'standard_score': 0.1,
        'stopping_criteria': 5,
        'convergence': 0.0002,
        'custom_params': custom_params,
        'rootid': rootid,
    }
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

    factors_data = factors_data.set_index('trade_time')
    engine.train(total_data=factors_data)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='cicso0',
                        help='data method')

    parser.add_argument('--task_id',
                        type=str,
                        default='200037',
                        help='task id')

    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    parser.add_argument('--session',
                        type=str,
                        default=202509226,
                        help='session')
    parser.add_argument('--count', type=int, default=150, help='count')
    args = parser.parse_args()
    #method = 'aicso0'
    #instruments = 'rbb'
    '''
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
          count=variant.count)
