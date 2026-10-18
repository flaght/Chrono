import os, sys, pdb, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))
from lumina.evolution.genetic import merge_factors
from lumina.evolution.engine import Engine
from ultron.factor.genetic.geneticist.operators import custom_transformer
from lumina.evolution.warehouse import sequential_gain


def create_test_data(start_date, symbols, n):
    np.random.seed(42)
    num_symbols = len(symbols)
    return_values_cycle = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 预设的收益率循环值

    trade_time_range = pd.date_range(start=start_date, periods=n, freq='B')
    index = pd.MultiIndex.from_product([trade_time_range, symbols],
                                       names=['trade_time', 'code'])

    total_rows = n * num_symbols  # DataFrame的总行数

    base_price_paths = np.cumsum(np.random.randn(n, num_symbols), axis=0) + 100

    base_price_flat = base_price_paths.flatten()

    open_price = base_price_flat + np.random.randn(total_rows) * 0.5
    close_price = base_price_flat + np.random.randn(total_rows) * 0.5

    max_oc = np.maximum(open_price, close_price)
    min_oc = np.minimum(open_price, close_price)

    high_add = np.random.rand(
        total_rows) * 0.3 + 0.01  # 加一个小的正数确保high > max_oc
    low_sub = np.random.rand(total_rows) * 0.3 + 0.01  # 减一个小的正数确保low < min_oc

    high_price = max_oc + high_add
    low_price = min_oc - low_sub

    current_high_gt_low = high_price > low_price
    low_price = np.where(current_high_gt_low, low_price, high_price - 0.01)

    volume = np.random.randint(1000, 10000, size=total_rows)
    amount = volume * close_price

    num_return_values = len(return_values_cycle)
    single_stock_returns = np.tile(return_values_cycle,
                                   n // num_return_values + 1)[:n]

    returns = np.repeat(single_stock_returns, num_symbols)

    data_dict = {
        'open': open_price.round(2),
        'high': high_price.round(2),
        'low': low_price.round(2),
        'close': close_price.round(2),
        'volume': volume,
        'amount': amount.round(2),
        'return': returns  # 收益率已经是正确的形状和值
    }
    total_data = pd.DataFrame(data_dict, index=index)
    # 调整列顺序以匹配您的期望
    total_data = total_data[[
        'open', 'high', 'low', 'close', 'volume', 'amount', 'return'
    ]]

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
    r_ic_mean = ic_mean if math.fabs(ic_mean) > 0.06 else 0
    fitness = 0.7 * (ic_mean / ic_std) + 0.3 * math.fabs(r_ic_mean)
    return fitness


def callback_models(gen, rootid, best_programs, custom_params, total_data):
    candidate_factors = merge_factors(best_programs=best_programs)
    tournament_size = 20  #custom_params['tournament_size']
    standard_score = 0.3  #ustom_params['standard_score']
    dethod = '444'  #custom_params['dethod']
    method = 'asci'  #custom_params['method']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)
    dirs = os.path.join('./', dethod, method, 'IF', 'evolution')
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
                                       ic_threshold=0.06,
                                       corr_threshold=0.7,
                                       gain_threshold=0.2)

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

start_date = '2020-01-01'
symbols = ['A', 'B', 'C', 'D']
n = 800

rootid = 100001

total_data = create_test_data(start_date=start_date, symbols=symbols,
                              n=n).reset_index()
total_data.rename(columns={'return': 'nxt1_ret'}, inplace=True)
pdb.set_trace()
factor_columns = [
    col for col in total_data.columns
    if col not in ['trade_time', 'code', 'nxt1_ret']
]

population_size = 500
tournament_size = 100
standard_score = 0.8

operators_sets = two_operators_sets + one_operators_sets
operators_sets = custom_transformer(operators_sets)
pdb.set_trace()
custom_params = {
    'horizon': '1',
    rootid: rootid,
    'g_instruments': 'rbb',
    'dethod': '11',
    'gain': {
        'corr_threshold': 0.6,
        'fitness_scale': 0.7,
        'gain_threshold': 0.1
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
    },
    'threshold': {
        "initial_threshold": 0.07,
        "target_percentile": 0.75,
        "min_threshold": 0.05,
        "max_threshold": 0.4,
        "adjustment_speed": 0.1
    }
}
configure = {
    'n_jobs': 1,
    'population_size': 50,
    'tournament_size': 10,
    'init_depth': 6,
    'evaluate': 'both_evaluate',
    'method': 'fitness',
    'crossover': 0.3,
    'point_replace': 0.2,
    'hoist_mutation': 0.2,
    'subtree_mutation': 0.2,
    'point_mutation': 0.1,
    'generations': 5,
    'standard_score': 0.0065,
    'rootid': rootid,
    'stopping_criteria': 5,
    'convergence': 0.002,
    'custom_params': custom_params
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
print(total_data)
factors_data = total_data.reset_index().set_index('trade_time')
engine.train(total_data=factors_data)
