import os, math, random, hashlib
import pandas as pd
import numpy as np
from lumina.genetic.util import create_id

from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

from kdutils.tactix import Tactix
from lib.optim001.optimizer import FactorOptimizer
from lib.cux001 import FactorEvaluate1
from lib.aux001 import calc_expression
from lib.iux001 import aggregation_data


def create_name_id(expression):
    m = hashlib.md5()
    m.update(bytes(expression, encoding='UTF-8'))
    return create_id(original=m.hexdigest(), digit=16)


def fetch_resource():
    # 读取算子依赖关系
    operators_pd = pd.read_csv(
        os.path.join(base_path, "resource",
                     "expression_dependencies.csv")).rename(
                         columns={
                             'Category': 'category',
                             'Expression': 'expression',
                             'Name': 'name',
                             'Description': 'description',
                             'Operator': 'operator_name',
                         })

    # 读取字段依赖关系
    fields_pd = pd.read_csv(
        os.path.join(base_path, "resource",
                     "level2_fields_dependencies.csv")).rename(
                         columns={
                             'types': 'field_type',
                             'Field': 'field_name',
                             'Formula': 'formula',
                             'Description': 'description',
                             'Dependencies': 'dependencies'
                         })
    return operators_pd, fields_pd


def objective_func(expression: str,
                   period: int,
                   total_data: pd.DataFrame,
                   total_data1: pd.DataFrame,
                   optimize_rule=None):
    factor_data = calc_expression(expression=expression,
                                  total_data=total_data1)
    dt = aggregation_data(factor_data=factor_data,
                          returns_data=total_data,
                          period=period)

    evaluate1 = FactorEvaluate1(factor_data=dt,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression=expression)

    result = evaluate1.run()
    result['ic_mean'] = math.fabs(result['ic_mean'])

    values = [0.0 for v in optimize_rule.values()]
    min_ic_threshold = 0.001
    if not np.isfinite(result['ic_mean']):
        return values

    if abs(result['ic_mean']) < min_ic_threshold:
        return values

    if not np.isfinite(result['calmar']) or np.isnan(
            result['calmar']) or result['calmar'] <= 0:
        return values

    if not np.isfinite(result['sharpe1']) or np.isnan(
            result['sharpe1']) or result['sharpe1'] <= 0:
        return values

    values = [result['ic_mean'], result['sharpe2'], result['calmar']]

    return values


def train(method, instruments, period, session, task_id, expressions):
    dethod = 'ic'
    dirs = os.path.join(base_path, method, instruments, "gentic", dethod,
                        str(task_id), "nxt1_ret_{}h".format(period),
                        str(session))

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    optimize_rule = {
        'ic_mean': 'maximize',
        'sharpe2': 'maximize',
        'profit_ratio': 'maximize'
    }

    operators_pd, fields_pd = fetch_resource()
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
    total_data = total_factors.merge(
        total_returns, on=['trade_time',
                           'code']).sort_values(by=['trade_time', 'code'])
    total_data1 = total_data.set_index(['trade_time'])

    optimizer = FactorOptimizer(operators_pd=operators_pd, fields_pd=fields_pd)
    res = []
    for expression in expressions:
        objectives = objective_func(expression=expression,
                                    period=period,
                                    total_data=total_data,
                                    total_data1=total_data1,
                                    optimize_rule=optimize_rule)
        res.append({
            'name': "ultron_{0}".format(create_name_id(expression)),
            "formual": expression,
            "final_fitness": objectives[0]
        })
        results = optimizer.optimize_expression(
            expression=expression,
            objective_function=objective_func,
            n_trials=40,  # 多目标优化需要更多试验
            total_data=total_data,
            total_data1=total_data1,
            period=period,
            optimize_parameters=True,
            optimize_operators=True,
            optimize_fields=True,
            optimize_rule=optimize_rule,
            study_name=f"multi_objective_{expression}",
            multi_objective=True,  # 启用多目标优化
            top_n=10  # 返回前5个最佳结果
        )
        res += [{
            "name":
            "ultron_{0}".format(create_name_id(result['expression'])),
            'formual':
            result['expression'],
            'final_fitness':
            result['score'][0]
        } for result in results['top_n_results']]

    final_programs = pd.DataFrame(res)
    programs_filename = os.path.join(dirs,
                                     f'programs_{task_id}_{session}.feather')
    
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        final_programs = pd.concat([old_programs, final_programs], axis=0)

    final_programs = final_programs.drop_duplicates(subset=['name'])
    final_programs.to_feather(programs_filename)


if __name__ == '__main__':
    expressions = [
        "SUBBED(MCORR(20, 'close', 'volume'), AVG(MCORR(20, 'close', 'volume')))",
        "MA(20, 'close')", "DELTA(5, 'close')", "MA(10, 'volume')",
        "DIV(MA(5, 'close'), MA(5, 'volume'))"
    ]

    train(method='cicso0',
          instruments='ims',
          period=15,
          task_id='200037',
          session='202510532',
          expressions=expressions)
    '''
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
          count=variant.count)
    '''
