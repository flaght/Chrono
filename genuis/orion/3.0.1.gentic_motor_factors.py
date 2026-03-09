import pdb, os, datetime
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import base_path
from lib.cms003 import Metrics
from kdutils.tactix import Tactix
from lumina.evolution.engine import Engine
from ultron.factor.genetic.geneticist.operators import Operators
from lib.cms003 import Metrics as Metrics003


def callback_fitness(factor_data, total_data, factor_sets, custom_params,
                     default_value):
    min_ic_threshold = 0.01
    if factor_data['transformed'].std() < 1e-8:
        return 0.0
    try:
        #wide_data = factor_data.set_index('code', append=True).unstack()
        wide_factor = factor_data.pivot(columns='code', values=['transformed'])
        returns = total_data[['trade_time', 'code', 'nxt1_ret']]
        wide_returns = returns.pivot(index='trade_time',
                                     columns='code',
                                     values='nxt1_ret')
        result = Metrics003.quick(factors=wide_factor,
                                  returns=wide_returns,
                                  dummy=None,
                                  hold=1,
                                  skip=0,
                                  show_log=False,
                                  category=0)
    except Exception:
        return 0.0

    if np.isnan(result['ic']) or np.isnan(result['icir']):
        return 0.0
            
    if abs(result['ic']) < min_ic_threshold:
        return 0.0

    if result['icir'] < 0.01:
        return 0.0

    if result['turnover'] > 1.0:
        return 0.0

    return abs(result['ic'])


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    tournament_size = custom_params['tournament_size']
    method = custom_params['method']
    return_name = custom_params['return_name']
    session = custom_params['session']
    standard_score = custom_params['standard_score'] * 0.1
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)
    dirs = os.path.join(base_path, method, "gentic", str(rootid), return_name,
                        str(session))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    programs_filename = os.path.join(dirs,
                                     f'programs_{rootid}_{session}.feather')
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)

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
    ]])

    final_programs = final_programs.drop_duplicates(subset=['name'])
    final_programs.reset_index(drop=True).to_feather(programs_filename)


def load_data(method, task_id, ret_name):
    base_dir1 = os.path.join(base_path, method, 'base', str(task_id))
    ## 加载基础特征数据 使用原始值
    train_factors = pd.read_feather(
        os.path.join(base_dir1, "train_data.feather"))
    val_factors = pd.read_feather(os.path.join(base_dir1, "val_data.feather"))

    train_return = pd.read_feather(
        os.path.join(base_dir1, "train_return.feather"))
    val_return = pd.read_feather(os.path.join(base_dir1, "val_return.feather"))

    train_factors['trade_time'] = pd.to_datetime(train_factors['trade_time'])
    val_factors['trade_time'] = pd.to_datetime(val_factors['trade_time'])

    train_return['trade_time'] = pd.to_datetime(train_return['trade_time'])
    val_return['trade_time'] = pd.to_datetime(val_return['trade_time'])

    total_factors = pd.concat([train_factors, val_factors], axis=0)
    total_return = pd.concat([train_return, val_return], axis=0)

    factors_cols = [
        col for col in total_factors.columns if col not in [
            'trade_time', 'code', 'time_weight', 'equal_weight',
            'f_funding_rate', 'f_funding_interval'
            'nxt1_ret_1h', 'nxt1_ret_2h', 'nxt1_ret_3h', 'nxt1_ret_5h',
            'nxt1_ret_10h', 'nxt1_ret_15h'
        ]
    ]

    total_data = total_factors.merge(total_return, on=['trade_time', 'code'])
    total_data = total_data[['trade_time', 'code', ret_name] + factors_cols]
    total_data.rename(columns={ret_name: 'nxt1_ret'}, inplace=True)
    return total_data, factors_cols


def train(method, task_id, session, ret_name):
    two_operators_sets = [
        'MConVariance', 'MRes', 'MMeanRes', 'MCORR', 'MCoef', 'MSharp',
        'MSortino', 'MINIMUM', 'MAXIMUM', 'ADDED', 'SUBBED', 'MUL', 'DIV',
        'MOD'
    ]
    one_operators_sets = [
        'MA', 'MPERCENT', 'MMedian', 'MADecay', 'MMAX', 'MMIN', 'MDPO',
        'MARGMAX', 'MARGMIN', 'MRANK', 'MQUANTILE', 'MDIFF', 'MSUM',
        'MVARIANCE', 'MIR', 'MSKEW', 'MKURT', 'MSTD', 'MNPOSITIVE',
        'MAPOSITIVE', 'EMA', 'RSI', 'WMA', 'SIGN', 'AVG', 'SQRT', 'DIFF',
        'LOG2', 'LOG10', 'LOG', 'EXP', 'FRAC', 'SIGLOG2ABS', 'SIGLOG10ABS',
        'SIGLOGABS', 'ABS', 'ACOS', 'ASIN', 'NORMINV', 'CEIL', 'FLOOR',
        'ROUND', 'TANH', 'RELU', 'SHIFT', 'DELTA', 'SIGMOID', 'LAST'
    ]

    total_data, factors_cols = load_data(method=method,
                                         task_id=task_id,
                                         ret_name=ret_name)
    factors_data = total_data
    use_factor_columns = factors_cols

    operators_sets = two_operators_sets + one_operators_sets

    operators_sets = Operators(
        periods=[2, 3, 4, 5, 8]).custom_transformer(operators_sets)

    population_size = 50  #600  #500#500  #500
    tournament_size = 10  #150  #100#100  #100
    standard_score = 0.001
    generations = 4
    custom_params = {
        'horizon': ret_name,
        'rootid': task_id,
        'tournament_size': tournament_size,
        'standard_score': standard_score,
        'dethod': 'ic',
        'method': method,
        'return_name': ret_name,
        'session': session
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
        'hoist_mutation': 0.05,
        'subtree_mutation': 0.15,
        'point_mutation': 0.2,
        'generations': generations,
        'standard_score': standard_score,
        'stopping_criteria': 5,
        'convergence': 0.0002,
        'custom_params': custom_params,
        'rootid': task_id,
        'method': 'grow'  ## grow:多样性 full 规则性
    }
    engine = Engine(
        population_size=configure['population_size'],
        tournament_size=configure['tournament_size'],
        init_depth=(1, configure['init_depth']),
        init_method=configure['method'],
        generations=configure['generations'],
        n_jobs=configure['n_jobs'],
        stopping_criteria=configure['stopping_criteria'],
        p_crossover=configure['crossover'],
        p_point_mutation=configure['point_mutation'],
        p_subtree_mutation=configure['subtree_mutation'],
        p_hoist_mutation=configure['hoist_mutation'],
        p_point_replace=configure['point_replace'],
        rootid=configure['rootid'],
        factor_sets=use_factor_columns,  #factor_columns, 用于使用的特征例
        standard_score=configure['standard_score'],
        operators_sets=operators_sets,
        backup_cycle=1,
        convergence=configure['convergence'],
        fitness=callback_fitness,
        save_model=callback_models,
        custom_params=configure['custom_params'])

    factors_data = factors_data.sort_values(
        by=['trade_time', 'code']).set_index('trade_time')
    #if corr_threshold > 0 and corr_threshold < 1: ## 使用精选因子库相关性过滤
    #    warehouse.calculate_evaluate(factors_data, period)
    engine.train(total_data=factors_data)


if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method,
          task_id=variant.task_id,
          session=variant.session,
          ret_name=variant.ret_name)
