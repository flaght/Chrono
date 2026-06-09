import os, math, hashlib
import pandas as pd
import numpy as np
from lumina.genetic.util import create_id

from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

from kdutils.tactix import Tactix
from lib.optim001.parallel1 import ParallelOptimizer
from lib.cux001 import FactorEvaluate1
from lib.aux001 import calc_expression
from lib.iux001 import aggregation_data, merging_data1


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
        #dt = aggregation_data(factor_data=factor_data,
        #                      returns_data=total_data,
        #                      period=period)
        dt = merging_data1(factor_data=factor_data,
                           returns_data=total_data,
                           period=period)
        # 第2步：评估因子
        evaluate1 = FactorEvaluate1(factor_data=dt,
                                    factor_name='transformed',
                                    ret_name='nxt1_ret_{0}h'.format(period),
                                    roll_win=15,
                                    fee=0.000,
                                    scale_method='roll_zscore',
                                    expression=expression,
                                    resampling_win=period)

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
                calmar_val) or calmar_val <= 0 or calmar_val >= 10:
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

        # 验证5: Sharpe2是否有效
        sharpe2_val = result.get('sharpe2', np.nan)
        if not np.isfinite(sharpe2_val) or np.isnan(
                sharpe2_val) or sharpe2_val <= 0 or sharpe2_val > 8:
            if verbose:
                print(
                    f"❌ [FILTER-4] sharpe2_val({sharpe2_val}) | {expr_short}")
            if logger and trial_num is not None:
                logger.log_trial(trial_num, 'filter_5', values, expression)
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
    standard_score = 0.02
    n_jobs = 4
    n_trials = 300
    top_n = 150
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

    optimizer = ParallelOptimizer(operators_pd=operators_pd,
                                  fields_pd=fields_pd,
                                  n_jobs=n_jobs)

    best_programs = optimizer.optimize(expressions=expressions,
                                       objective_function=objective_func,
                                       total_data=total_data,
                                       total_data1=total_data1,
                                       period=period,
                                       optimize_rule=optimize_rule,
                                       multi_objective=True,
                                       n_trials=n_trials,
                                       top_n=top_n,
                                       optimize_operators=False,
                                       optimize_fields=True)
    programs_filename = os.path.join(dirs,
                                     f'programs_{rootid}_{session}.feather')
    if os.path.exists(programs_filename):
        old_programs = pd.read_feather(programs_filename)
        best_programs = pd.concat([old_programs, best_programs], axis=0)
    best_programs = best_programs.drop_duplicates(subset=['name'])
    best_programs = best_programs[best_programs['final_fitness'] >
                                  standard_score]
    print(programs_filename)
    print(best_programs)
    best_programs.reset_index(drop=True).to_feather(programs_filename)


if __name__ == '__main__':

    expressions = [
        # 第一梯队 (10个)
        "MRANK(15, MCoef(25, EMA(10, 'close'), 'close'))",
        "MRANK(20, DIV(SUBBED('high', 'low'), MA(15, SUBBED('high', 'low'))))",
        "MRANK(25, SUBBED(DIV('money', 'volume'), EMA(20, DIV('money', 'volume'))))",
        "MRANK(20, DIV(SUBBED(EMA(5, 'close'), EMA(20, 'close')), MSTD(20, 'close')))",
        "MRANK(25, SUBBED(MRANK(20, DELTA(5, 'close')), MRANK(20, DELTA(5, 'volume'))))",
        "MRes(20, MA(5, 'close'), 'close')",
        "MRANK(20, MCORR(30, DELTA(5, 'close'), SHIFT(5, DELTA(5, 'close'))))",
        "MRANK(25, ADDED(DIV(SUBBED(EMA(5, 'close'), EMA(12, 'close')), MSTD(20, 'close')), DIV(SUBBED(EMA(12, 'close'), EMA(35, 'close')), MSTD(35, 'close'))))",
        "ADDED(ADDED(MRANK(35, SUBBED(MRANK(25, DELTA(5, 'close')), MRANK(25, DELTA(5, 'volume')))), MRANK(35, MCORR(30, 'money', DIV('money', 'volume')))), MRANK(35, EMA(12, ADDED(ADDED('depth_imbalance_0', 'depth_imbalance_1'), ADDED('depth_imbalance_2', 'depth_imbalance_3')))))",
        "MRANK(30, MCORR(35, EMA(5, 'net_money_in'), DELTA(10, 'close')))",

        # 第二梯队 (20个)
        "MRANK(15, ADDED(MSKEW(20, 'pct_change'), MSKEW(10, 'pct_change')))",
        "MRANK(25, MCORR(25, 'money', DIV('money', 'volume')))",
        "MRANK(20, EMA(10, ADDED(ADDED('depth_imbalance_0', 'depth_imbalance_1'), ADDED('depth_imbalance_2', 'depth_imbalance_3'))))",
        "MRANK(20, DIV(DELTA(5, 'twap'), MSTD(25, 'twap')))",
        "ADDED(ADDED(MRANK(20, DIV(DELTA(5, 'close'), MSTD(20, 'close'))), MRANK(20, EMA(3, 'net_money_in'))), MRANK(20, 'depth_imbalance_0'))",
        "SUBBED(MRANK(15, MA(3, 'net_money_in')), MRANK(15, 'pct_change'))",
        "MRANK(25, DELTA(5, 'order_flow_imbanlace_weighted5'))",
        "MRANK(15, MSKEW(20, MSTD(5, 'pct_change')))",
        "MRANK(25, DIV(SUBBED('high', 'low'), EMA(20, SUBBED('high', 'low'))))",
        "MCORR(15, EMA(5, 'close'), EMA(20, 'close'))",
        "MRANK(25, DELTA(7, EMA(10, 'net_money_in')))",
        "MRANK(30, SUBBED(DIV('money', 'volume'), EMA(25, DIV('money', 'volume'))))",
        "MRANK(20, DIV(MA(3, 'volume'), EMA(20, 'volume')))",
        "ADDED(DIV(DELTA(3, 'close'), MSTD(10, 'close')), DIV(DELTA(10, 'close'), MSTD(30, 'close')))",
        "MRANK(30, EMA(15, 'mci_imbalance'))",
        "MRANK(25, ADDED(MUL(0.5, 'order_flow_imbanlace_1'), MUL(0.3, 'order_flow_imbanlace_avg5')))",
        "MRANK(25, MKURT(25, 'pct_change'))",
        "MRANK(35, DIV(SUBBED('high', 'low'), EMA(30, SUBBED('high', 'low'))))",
        "MRANK(30, MUL(DELTA(7, 'close'), DELTA(7, 'volume')))",
        "EMA(5, ADDED(MUL(0.4, 'depth_imbalance_0'), ADDED(MUL(0.3, 'depth_imbalance_1'), MUL(0.2, 'depth_imbalance_2'))))",

        # 第三梯队 (30个)
        "MRANK(20, EMA(12, 'ask_bid_press'))",
        "MRANK(15, DIV(SUBBED('realized_volatility', MA(15, 'realized_volatility')), MSTD(15, 'realized_volatility')))",
        "MCORR(20, DIV('money', 'volume'), 'pct_change')",
        "DIV(DELTA(5, SUBBED('high', 'low')), MA(15, SUBBED('high', 'low')))",
        "EMA(10, 'mci_imbalance')",
        "DIV('bid_ask_spread', MA(10, 'bid_ask_spread'))",
        "DIV('realized_volatility', MA(20, 'realized_volatility'))",
        "ADDED(ADDED('depth_imbalance_0', 'depth_imbalance_1'), ADDED('depth_imbalance_2', 'depth_imbalance_3'))",
        "MRANK(20, DIV('volume', MA(20, 'volume')))",
        "MCORR(15, DELTA(5, 'close'), SHIFT(5, DELTA(5, 'close')))",
        "SUBBED(MRANK(10, 'net_money_in'), MRANK(10, 'pct_change'))",
        "DELTA(3, 'order_flow_imbanlace_weighted5')",
        "MSKEW(20, 'pct_change')",
        "MRANK(30, DIV(MSTD(15, 'volume'), EMA(35, 'volume')))",
        "DIV(MA(10, 'pct_change'), MSTD(20, 'pct_change'))",
        "MADecay(15, 'pct_change')",
        "MRANK(20, DIV(SUBBED('close', MA(10, 'close')), MSTD(10, 'close')))",
        "DIV(MSUM(10, MUL('pct_change', 'volume')), MSUM(10, 'volume'))",
        "MRANK(25, DELTA(10, RSI(14, 'close')))",
        "MRANK(25, MSKEW(30, MSTD(5, 'pct_change')))",
        "MRANK(25, ADDED(MSKEW(30, 'pct_change'), MSKEW(15, 'pct_change')))",
        "MA(5, 'order_imbalance_ratio5')",
        "MRANK(25, DIV(SUBBED('smart_volume_in', 'smart_volume_out'), MSTD(15, 'volume')))",
        "MRANK(10, EMA(5, ADDED('price_imbalance_0', 'price_imbalance_1')))",
        "MRANK(30, MCORR(30, EMA(12, 'close'), EMA(35, 'close')))",
        "MRANK(30, ADDED(MSKEW(35, 'pct_change'), MSKEW(18, 'pct_change')))",
        "MRANK(30, MSKEW(35, MSTD(7, 'pct_change')))",
        "SUBBED(MRANK(30, EMA(10, 'net_money_in')), MRANK(30, 'pct_change'))",
        "MRANK(30, DELTA(7, 'order_flow_imbanlace_weighted5'))",
        "MRANK(30, MKURT(30, 'pct_change'))"
    ]
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
          expressions=expressions)
