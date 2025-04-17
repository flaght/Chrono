import os, pdb, math
import pandas as pd
import numpy as np
from dotenv import load_dotenv

pd.options.mode.copy_on_write = True
load_dotenv()

from jdw import EntropyAPI
from ultron.factor.dimension.corrcoef import FCorrType
from ultron.ump.similar.corrcoef import ECoreCorrType
from ultron.ump.similar.corrcoef import corr_xy, corr_matrix, ECoreCorrType

all_operators_sets = [
    'SIGN', 'EXP', 'LOG', 'SQRT', 'ABS', 'CEIL', 'FLOOR', 'ROUND',
    'RETURNSimple', 'RETURNLog', 'SIGMOID', 'LOG2', 'LOG10', 'FRAC',
    'SIGLOG2ABS', 'SIGLOG10ABS', 'SIGLOGABS', 'SIGSQRTABS', 'CSRank',
    'CSZScore', 'CSQuantiles', 'CSMean', 'CSMeanAdjusted', 'CSSoftmax',
    'CSDemean', 'CSSTD', 'CSSKEW', 'CSSUM', 'CSWinsorize', 'CSRes',
    'CSFloorDiv', 'CSNeut', 'CSProj', 'CSSignPower', 'ADDED', 'SUBBED', 'MUL',
    'DIV', 'MINIMUM', 'MAXIMUM', 'EMA', 'MA', 'MADecay', 'MMAX', 'MARGMAX',
    'MMIN', 'MARGMIN', 'MRANK', 'MSUM', 'MVARIANCE', 'MSTD', 'RSI', 'DELTA',
    'SHIFT', 'MMedian', 'MADiff', 'MCPS', 'MDIFF', 'MMaxDiff', 'MMinDiff',
    'MIR', 'MZScore', 'MKURT', 'MSKEW', 'MCORR', 'MRes', 'MMeanRes', 'MSharp',
    'MCoef', 'MRSquared', 'MSortino'
]

two_operators = [
    'CSRes', 'CSFloorDiv', 'CSSignPower', 'CSNeut', 'CSProj', 'MCORR', 'MRes',
    'MMeanRes', 'MCoef', 'MRSquared', 'MSharp', 'MSortino', 'ADDED', 'SUBBED',
    'MUL', 'DIV', 'MOD', 'MINIMUM', 'MINIMUM'
]

non_cs_operators = [
    'SIGN', 'EXP', 'LOG', 'SQRT', 'ABS', 'CEIL', 'FLOOR', 'ROUND',
    'RETURNSimple', 'RETURNLog', 'SIGMOID', 'LOG2', 'LOG10', 'FRAC',
    'SIGLOG2ABS', 'SIGLOG10ABS', 'SIGLOGABS', 'SIGSQRTABS', 'ADDED', 'SUBBED',
    'MUL', 'DIV', 'MINIMUM', 'MAXIMUM', 'EMA', 'MA', 'MADecay', 'MMAX',
    'MARGMAX', 'MMIN', 'MARGMIN', 'MRANK', 'MSUM', 'MVARIANCE', 'MSTD', 'RSI',
    'DELTA', 'SHIFT', 'MMedian', 'MADiff', 'MCPS', 'MDIFF', 'MMaxDiff',
    'MMinDiff', 'MIR', 'MZScore', 'MKURT', 'MSKEW', 'MCORR', 'MRes',
    'MMeanRes', 'MSharp', 'MCoef', 'MRSquared', 'MSortino'
]


## 自定义评估函数 时序IC
def metrics_fitness(factor_data, total_data, factor_sets, custom_params,
                    default_value):
    returns = total_data[['trade_time', 'code', 'nxt1_ret']]
    factor_data = factor_data.reset_index()
    data = factor_data.merge(returns, on=['trade_time', 'code'])

    corr_value = corr_xy(data['transformed'], data['nxt1_ret'],
                         ECoreCorrType.E_CORE_TYPE_SPERM)
    return math.fabs(corr_value)


def callback_save(gen, rootid, best_programs, custom_params):
    data = pd.DataFrame([b.output() for b in best_programs])
    data['task_id'] = rootid
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', 'evolition',
        "evolition_{}_{}.feather".format(os.environ['HORIZON'], rootid))

    print("new  factors: {0}".format(
        data.sort_values('fitness', ascending=False)[['formual',
                                                      'fitness']].head(10)))

    if os.path.exists(file_name):
        old_data = pd.read_feather(file_name)
        data = pd.concat([old_data, data], axis=0)

    data = data.sort_values('fitness', ascending=False)
    data = data.drop_duplicates(subset=['features'])
    #
    #data = data.reset_index(drop=True).loc[:custom_params['tournament_size'] *
    #                                       5]
    data = data[data['fitness'] > 0.02]
    print(f'Saving {file_name}')
    print("all factors: {0}".format(data[['formual', 'fitness']].head(10)))
    data.reset_index(drop=True).to_feather(file_name)


## 构建挖掘器
def create_genetic(factor_columns,
                   operators,
                   universe,
                   horizon,
                   yield_name,
                   industry_name,
                   industry_level,
                   callback_fitness,
                   callback_save,
                   offset=0):
    return EntropyAPI.FuturesGeneticist(
        offset,
        horizon,
        factor_columns,
        universe,
        industry_name,
        industry_level,
        dummy_name=None,
        is_loop=True,  #是否一致循环跑
        operators=operators,
        factors_normal=True,
        callback_save=callback_save,
        callback_fitness=callback_fitness,
        yield_name=yield_name)


## 参数配置
def create_parameter(operators_sets,
                     industry_name='financial',
                     industry_level=1,
                     method='ic',
                     evaluate='both_evaluate',
                     universe='financial ',
                     horizon=1,
                     yields_name='nxt1_ret'):
    parameter = {
        "operators_sets": operators_sets,
        "industry_name": industry_name,
        "industry_level": industry_level,
        "method": method,
        "evaluate": evaluate,
        "universe": universe,
        "horizon": horizon,
        "yields_name": yields_name
    }
    return parameter


## 参数配置
def genetic_configure(population_size=10,
                      tournament_size=5,
                      init_depth=4,
                      generations=10,
                      n_jobs=1,
                      stopping_criteria=100,
                      standard_score=10,
                      crossover=0.4,
                      point_mutation=0.3,
                      subtree_mutation=0.1,
                      hoist_mutation=0.1,
                      point_replace=0.1,
                      convergence=0.02):
    configure = {
        "population_size": population_size,
        "tournament_size": tournament_size,
        "init_depth": init_depth,
        "generations": generations,
        "n_jobs": n_jobs,
        "stopping_criteria": stopping_criteria,
        "standard_score": standard_score,
        "crossover": crossover,
        "point_mutation": point_mutation,
        "subtree_mutation": subtree_mutation,
        "hoist_mutation": hoist_mutation,
        "point_replace": point_replace,
        "convergence": convergence
    }
    return configure


## 收益率处理
def create_yields(data, horizon, offset=1):
    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    dt.name = 'nxt1_ret'
    data = dt.reset_index().merge(data, on=['trade_time',
                                            'code']).reset_index(drop=True)
    return data.dropna(subset=['nxt1_ret'])


## 构建数据
## 数据格式：trade_date, code, features, nxt1_ret
def create_data(method, horizon):
    ### 加载计算的因子
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', 'factors',
                             f'{method}_lumina_features.feather')
    data = pd.read_feather(file_path)

    data = create_yields(data, int(horizon))
    return data


def main():
    horizon = os.getenv('HORIZON')
    data = create_data(os.getenv('METHOD'), horizon)
    features = [
        col for col in data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]

    ## 持仓1天收益
    parameter = create_parameter(non_cs_operators)
    configure = genetic_configure(n_jobs=4,
                                  population_size=50,
                                  tournament_size=10,
                                  init_depth=8,
                                  crossover=0.5,
                                  generations=10)
    genetic = create_genetic(factor_columns=features,
                             operators=parameter['operators_sets'],
                             universe=parameter['universe'],
                             horizon=horizon,
                             yield_name=parameter['yields_name'],
                             industry_name=parameter['industry_name'],
                             industry_level=parameter['industry_level'],
                             callback_fitness=metrics_fitness,
                             callback_save=callback_save,
                             offset=0)
    configure['method'] = parameter['method']
    configure['evaluate'] = parameter['evaluate']
    configure['yields_name'] = parameter['yields_name']
    configure['rootid'] = '108990000'
    genetic.calculate_result(total_data=data,
                             configure=configure,
                             custom_params=None)


main()
