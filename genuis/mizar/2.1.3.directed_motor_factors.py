import pandas as pd
import numpy as np
import pdb, argparse, random
import os, pdb, math, itertools
from scipy import stats  # 用于计算秩相关IC
from dotenv import load_dotenv

load_dotenv()
from kdutils.tactix import Tactix

from ultron.factor.genetic.geneticist.operators import Operators
from lumina.evolution.genetic import merge_factors
from lumina.evolution.engine import Engine
from lumina.evolution.warehouse import sequential_gain

from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

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
                                roll_win=15,
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

    return fitness


def train(method, instruments, period, session, task_id, count=0):
    seed_feature_list = [
        'close',  # 收盘价，构建价格变化序列的核心特征
        'volume',  # 成交量，构建成交量变化序列的核心特征
        'open',  # 开盘价，价格的代表性指标
        'high',  # 最高价，价格波动范围
        'low',  # 最低价，价格波动范围
        'twap',  # 时间加权平均价，价格的代表性指标
        'money',  # 成交额，价量关系的另一个维度
        'pct_change',  # 涨跌幅，价格变化的基础特征
        'pct_change_close',  # 相对昨收涨跌幅，价格变化的代理变量
        'pct_change_set'  # 相对昨结涨跌幅，价格变化的代理变量
    ]

    # Step 2: Define the curated set of high-quality operators.
    two_operators_sets = [
        'MCORR',  # 滚动相关性，价量相关性的核心算子
        'MCoef',  # 滚动回归系数，价量关系的线性度量
        'MConVariance',  # 时序协方差，价量关系的统计度量
        'MRSquared',  # 滚动R方，相关性强度的度量
        'MRes'  # 滚动残差，价量关系的非线性度量
    ]
    one_operators_sets = [
        'DELTA',  # 周期差值，计算价格和成交量变化量
        'DIFF',  # 一阶差分，变化量的另一种计算
        'SHIFT',  # 向前取值，获取历史数据，用于隔夜价量关系
        'MA',  # 移动平均，平滑价格和成交量序列
        'MSTD',  # 移动标准差，波动性度量
        'MSUM',  # 滚动求和，累积变化
        'MMAX',  # 周期最大值，极值分析
        'MMIN',  # 周期最小值，极值分析
        'MRANK',  # 时序排序，相对位置
        'MPERCENT',  # 时序百分位，相对排名
        'MSKEW',  # 移动偏度，分布特征
        'MKURT'  # 移动峰度，分布特征
    ]

    operators_sets = two_operators_sets + one_operators_sets

    parameter_search_space = [  # 短周期（捕捉日内波动）
        # 短周期（捕捉日内波动）
        5,
        10,
        15,
        30,

        # 中周期（捕捉趋势）
        60,
        90,
        120,

        # 长周期（捕捉日间关系）
        240,
        480,
        720
    ]

    rootid = task_id
    ## 加载数据
    total_factors = fetch_temp_data(method=method,
                                    task_id=rootid,
                                    instruments=instruments,
                                    datasets=['train', 'val'])

    total_returns = fetch_temp_returns(method=method,
                                       instruments=instruments,
                                       datasets=['train', 'val'],
                                       category='returns')
    total_data = total_factors.merge(total_returns, on=['trade_time', 'code'])
    pdb.set_trace()
    missing_features = [
        f for f in seed_feature_list if f not in total_data.columns
    ]
    if missing_features:
        raise ValueError(
            f"Seed features not found in dataset: {missing_features}")
    factor_columns = seed_feature_list

    if count > 0 and count < len(factor_columns):
        factor_columns = random.sample(factor_columns, count)

    nxt1_columns = total_data.filter(regex="^nxt1").columns.to_list()
    basic_columns = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint'
    ]

    return_name = "nxt1_ret_{}h".format(period)

    if str(rootid) != '200037':
        agg_market_data = total_data[['trade_time', 'code'] + basic_columns]
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

    operators_sets = Operators(
        periods=parameter_search_space).custom_transformer(operators_sets)

    population_size = 100
    tournament_size = 50
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
        'threshold': {
            "initial_threshold": standard_score * 0.4,
            "target_percentile": 0.35,
            "min_threshold": standard_score * 0.2,
            "max_threshold": standard_score * 5,
            "adjustment_speed": 0.01
        }
    }

    configure = {
        'n_jobs': 2,
        'population_size': population_size,
        'tournament_size': tournament_size,
        'init_depth': 3,
        'evaluate': 'both_evaluate',
        'method': 'fitness',
        'crossover': 0.3,
        'point_replace': 0.3,
        'hoist_mutation': 0.1,
        'subtree_mutation': 0.1,
        'point_mutation': 0.2,
        'generations': 30,
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
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
          count=variant.count)
