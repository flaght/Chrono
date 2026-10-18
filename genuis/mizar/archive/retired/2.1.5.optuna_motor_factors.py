import os, math, hashlib
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
                   optimize_rule=None,
                   verbose=True,
                   logger=None,
                   trial_num=None):
    """
    目标函数，带详细日志
    
    Args:
        verbose: 是否打印详细日志（默认True）
        logger: OptimizationLogger 实例（用于统计）
        trial_num: 当前 trial 编号
    """
    # 缩短表达式用于日志显示
    expr_short = expression[:80] + "..." if len(
        expression) > 80 else expression

    try:
        # 第1步：计算因子
        factor_data = calc_expression(expression=expression,
                                      total_data=total_data1)
        dt = aggregation_data(factor_data=factor_data,
                              returns_data=total_data,
                              period=period)

        # 第2步：评估因子
        evaluate1 = FactorEvaluate1(factor_data=dt,
                                    factor_name='transformed',
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.000,
                                    scale_method='roll_zscore',
                                    expression=expression)

        result = evaluate1.run()
        result['ic_mean'] = math.fabs(result['ic_mean'])

        # 初始化返回值
        values = [0.0 for v in optimize_rule.values()]
        min_ic_threshold = 0.001

        # 验证1: IC是否有效
        if not np.isfinite(result['ic_mean']):
            if verbose:
                print(f"❌ [FILTER-1] IC无效(NaN/Inf) | {expr_short}")
            if logger and trial_num is not None:
                logger.log_trial(trial_num, 'filter_1', values, expression)
            return values

        # 验证2: IC是否足够大
        if abs(result['ic_mean']) < min_ic_threshold:
            if verbose:
                print(
                    f"❌ [FILTER-2] IC太小({result['ic_mean']:.6f} < {min_ic_threshold}) | {expr_short}"
                )
            if logger and trial_num is not None:
                logger.log_trial(trial_num, 'filter_2', values, expression)
            return values

        # 验证3: Calmar是否有效
        calmar_val = result.get('calmar', np.nan)
        if not np.isfinite(calmar_val) or np.isnan(
                calmar_val) or calmar_val <= 0:
            if verbose:
                print(f"❌ [FILTER-3] Calmar无效({calmar_val}) | {expr_short}")
            if logger and trial_num is not None:
                logger.log_trial(trial_num, 'filter_3', values, expression)
            return values

        # 验证4: Sharpe1是否有效
        sharpe1_val = result.get('sharpe1', np.nan)
        if not np.isfinite(sharpe1_val) or np.isnan(
                sharpe1_val) or sharpe1_val <= 0:
            if verbose:
                print(f"❌ [FILTER-4] Sharpe1无效({sharpe1_val}) | {expr_short}")
            if logger and trial_num is not None:
                logger.log_trial(trial_num, 'filter_4', values, expression)
            return values

        # 所有验证通过
        values = [result['ic_mean'], result['sharpe2'], result['calmar']]

        if verbose:
            print(
                f"✅ [VALID] IC={result['ic_mean']:.4f}, Sharpe2={result['sharpe2']:.4f}, "
                f"Calmar={result['calmar']:.4f} | {expr_short}")
        if logger and trial_num is not None:
            logger.log_trial(trial_num, 'valid', values, expression)

        return values

    except Exception as e:
        if verbose:
            print(f"❌ [EXCEPTION] {str(e)[:100]} | {expr_short}")
        values = [0.0 for v in optimize_rule.values()]
        if logger and trial_num is not None:
            logger.log_trial(trial_num, 'exception', values, expression)
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
            n_trials=10,  # 多目标优化需要更多试验
            total_data=total_data,
            total_data1=total_data1,
            period=period,
            optimize_parameters=True,
            optimize_operators=True,
            optimize_fields=True,
            optimize_rule=optimize_rule,
            study_name=f"multi_objective_{expression}",
            multi_objective=True,  # 启用多目标优化
            top_n=30  # 返回前5个最佳结果
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
    final_programs.reset_index(drop=True).to_feather(programs_filename)


if __name__ == '__main__':
    variant = Tactix().start()
    expressions = [
        #"SUBBED(MCORR(20, 'close', 'volume'), AVG(MCORR(20, 'close', 'volume')))",
        "MA(20, 'close')",
        "DELTA(5, 'close')",
        "MA(10, 'volume')",
        "DIV(MA(5, 'close'), MA(5, 'volume'))"
    ]

    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
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
