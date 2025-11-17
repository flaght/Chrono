import os, math, hashlib
import pandas as pd
import numpy as np
from lumina.genetic.util import create_id

from dotenv import load_dotenv

load_dotenv()
from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns

from kdutils.tactix import Tactix
from lib.optim001.parallel import ParallelOptimizer
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
                                       top_n=top_n)
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
        "ABS(MMedian(96, ABS(SUBBED(MMIN(288, 'delta_volume_bid1'), MMedian(96, MMIN(288, 'delta_volume_bid1'))))))",
        "MRANK(288, MSKEW(192, 'delta_volume_ask1'))",
        "MUL('volume_in', 'smart_volume_in')",
        "MUL('money_in', 'smart_volume_in')",
        "MRANK(24, MSKEW(192, 'mid_price_bias_ratio'))",
        "MUL('smart_money_in', 'money_in')",
        "MUL(MUL(MT3(96, 'corr_vwap_ask_price_0'), 'bid_ask_spread'), 'smart_money_in')",
        "MRANK(192, MSKEW(192, MSKEW(96, 'corr_vwap_bid_price_0')))",
        "MUL('smart_money_in', 'smart_volume_in')",
        "MRANK(24, MSKEW(288, 'mid_price_bias_ratio'))",
        "MUL('smart_money_in', 'money')",
        "MUL('price_imbalance_2', 'smart_money_in')",
        "MUL('smart_money_in', MDEMA(192, 'delta_volume_bid5'))",
        "MUL('smart_volume_in', -1)", "ABS('smart_volume_in')",
        "MRANK(24, MSKEW(96, 'corr_ret_bid_ask_price_spread'))",
        "MRANK(48, MSKEW(192, 'volume_in'))",
        "DIV('volume_out', MUL('corr_money_bid_ask_price_spread', 'delta_volume_bid5'))",
        "RSI(24, MSKEW(192, 'pct_change_set'))",
        "ABS(MKURT(96, 'order_flow_imbanlace_1'))",
        "MUL('money_out', 'smart_volume_in')",
        "DIV('bid_ask_spread', DIV(SUBBED('smart_money_out_pct', SHIFT(1, 'smart_money_out_pct')), SHIFT(1, 'smart_money_out_pct')))",
        "MUL('smart_money_in', MSTD(288, 'tick_in_pct'))",
        "MUL('volume_in', 'smart_money_in_pct')",
        "ABS(DIV('net_money_in', 'smart_tick_out'))",
        "MRANK(192, MSKEW(288, 'mid_price_bias_ratio'))",
        "MUL('delta_volume_ask5', 'smart_volume_in')",
        "MRANK(192, MSKEW(192, 'vwap5_bid_div_mid_price'))",
        "DIV(MSTD(24, 'smart_money_out_pct'), DIV(SUBBED('smart_money_out_pct', SHIFT(1, 'smart_money_out_pct')), SHIFT(1, 'smart_money_out_pct')))",
        "MUL('smart_tick_in', 'smart_volume_in')",
        "MRANK(192, MKURT(48, 'price_imbalance_0'))",
        "MUL('delta_volume_ask1', 'smart_money_in')",
        "MRANK(96, MSKEW(96, 'net_tick_in'))",
        "MRANK(288, MSKEW(96, 'weighted_volume_ask5'))",
        "MRANK(192, MSKEW(96, 'delta_volume_ask5'))",
        "ABS(MSKEW(96, 'mid_price_bias_ratio'))",
        "DIV(WMA(192, 'mid_price_bias_ratio'), MRANK(24, 'tick_out'))",
        "MUL('smart_money_in', 'smart_tick_in')",
        "MRANK(192, MA(96, MA(96, 'corr_money_ask_size_0')))",
        "DIV('log_order_slope', DIV(SUBBED('smart_money_out_pct', SHIFT(1, 'smart_money_out_pct')), SHIFT(1, 'smart_money_out_pct')))",
        "MRANK(192, MSKEW(192, 'pct_change_set'))",
        "MRANK(192, MMedian(48, ABS(SUBBED('smart_tick_in', MMedian(48, 'smart_tick_in')))))",
        "MRANK(288, MUL(MMedian(24, ABS(SUBBED('weighted_volume_bid5', MMedian(24, 'weighted_volume_bid5')))), 'smart_money_out_pct'))",
        "MRANK(192, MSKEW(96, 'tick_in_pct'))",
        "MRANK(192, MSKEW(96, 'net_tick_in'))",
        "MKURT(48, 'delta_volume_bid1')",
        "MRANK(48, MSKEW(96, 'smart_tick_in'))",
        "DIV(MSUM(48, MUL('smart_money_in_pct', 'high')), MSUM(48, 'high'))",
        "MRANK(288, MSKEW(192, 'order_imbalance_ratio1'))",
        "MRANK(288, MSKEW(48, MSKEW(96, 'mid_price_bias_ratio')))",
        "DIV(MSUM(48, MUL('smart_money_in_pct', 'mid_price_bias_ratio')), MSUM(48, 'mid_price_bias_ratio'))",
        "MRANK(192, MSKEW(192, MT3(48, 'corr_money_bid_ask_price_spread')))",
        "RSI(96, MKURT(24, 'money'))",
        "MRANK(288, MSKEW(192, 'vwap5_bid_div_mid_price'))",
        "MSUM(48, 'smart_money_in_pct')", "ABS(MKURT(48, 'money_out'))",
        "ABS(MUL(MSKEW(96, 'mid_price_bias_ratio'), 'bid_ask_spread'))",
        "MKURT(24, EMA(288, 'mci_ask'))", "MKURT(48, 'low')",
        "MUL('smart_tick_in_pct', 'weighted_volume_bid5')",
        "MKURT(48, 'twap')",
        "MSUM(192, MUL(MUL('corr_vwap_ask_price_0', 'bid_ask_spread'), 'smart_money_out_pct'))",
        "MRANK(288, MCoef(96, MSKEW(96, 'tick_in'), 'corr_money_ask_price_0'))",
        "MUL('weighted_volume_bid5', 'smart_tick_in')",
        "MUL('volume_in', 'smart_tick_in')",
        "MUL('smart_tick_in', 'money_in')",
        "MRANK(96, MSKEW(96, RSI(288, 'smart_tick_in_pct')))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'corr_money_ask_size_0')), MSUM(96, 'corr_money_ask_size_0'))",
        "MRANK(48, 'tick_in_pct')", "MKURT(48, 'volume_out')",
        "MRANK(24, 'tick_in_pct')", "MUL(ABS('money'), 'smart_tick_in_pct')",
        "MSUM(96, MUL(MUL(MSUM(192, 'vwap5_bid_div_mid_price'), -1), 'smart_money_out_pct'))",
        "MRANK(192, MMedian(96, ABS(SUBBED('smart_tick_out', MMedian(96, 'smart_tick_out')))))",
        "RSI(96, MMedian(288, ABS(SUBBED('close', MMedian(288, 'close')))))",
        "MUL(ABS('money'), 'smart_tick_in')", "MRANK(48, 'net_tick_in')",
        "MMedian(48, ABS(SUBBED('smart_tick_out_pct', MMedian(48, 'smart_tick_out_pct'))))",
        "MRANK(24, 'net_tick_in')",
        "MSUM(192, MUL('corr_vwap_ask_price_0', 'smart_money_out_pct'))",
        "MSUM(192, MUL('price_imbalance_0', 'smart_money_out_pct'))",
        "MCORR(288, MA(192, 'pct_change_set'), 'vwap5_ask_div_mid_price')",
        "MUL(MKURT(48, 'order_flow_imbanlace_1'), -1)",
        "MKURT(48, 'order_flow_imbanlace_1')",
        "DIFF(EMA(288, 'smart_volume_in'))",
        "MT3(192, MA(96, MA(96, MSTD(48, 'mci_imbalance'))))",
        "MRANK(96, MSKEW(192, MMedian(192, ABS(SUBBED('corr_money_vwap', MMedian(192, 'corr_money_vwap'))))))",
        "WMA(24, DIV('money_in', SHIFT(1, 'money_in')))",
        "RSI(288, MKURT(96, 'order_flow_imbanlace_1'))",
        "MUL(MUL('net_tick_in', 'net_money_in_pct'), -1)",
        "MKURT(48, MSTD(192, 'pct_change'))",
        "MRANK(288, MSKEW(192, MSKEW(96, MMedian(96, ABS(SUBBED(RSI(96, 'depth_imbalance_4'), MMedian(96, RSI(96, 'depth_imbalance_4'))))))))",
        "MSUM(192, MUL('price_imbalance_1', 'smart_money_out_pct'))",
        "MRANK(24, MSKEW(288, 'corr_ret_bid_ask_price_spread'))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'delta_volume_ask5'))",
        "MKURT(24, 'order_flow_imbanlace_1')",
        "MRANK(192, MA(144, MA(144, 'order_flow_imbanlace_avg5')))",
        "MMedian(192, ABS(SUBBED(SQRT('smart_tick_out_pct'), MMedian(192, SQRT('smart_tick_out_pct')))))",
        "MSUM(192, MUL('vwap5_bid_div_mid_price', 'smart_money_out_pct'))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'delta_volume_bid1'))",
        "MA(96, MSTD(192, 'depth_imbalance_0'))",
        "MRANK(288, MSKEW(192, MUL(MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in')))), 'smart_money_out_pct')))",
        "MRANK(96, MSKEW(96, MMedian(96, ABS(SUBBED(RSI(96, 'depth_imbalance_4'), MMedian(96, RSI(96, 'depth_imbalance_4')))))))",
        "MSUM(192, MUL('mci_ask', 'smart_money_out_pct'))",
        "MSUM(192, MUL('price_imbalance_2', 'smart_money_out_pct'))",
        "MRANK(24, MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "MRANK(192, MA(144, MA(144, 'order_flow_imbanlace_weighted5')))",
        "MRANK(288, MRANK(96, 'open'))",
        "MSUM(192, MUL('price_imbalance_3', 'smart_money_out_pct'))",
        "MKURT(48, 'price_imbalance_3')",
        "MUL(MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in')))), -1)",
        "MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in'))))",
        "MSUM(192, MUL('price_imbalance_4', 'smart_money_out_pct'))",
        "MRANK(48, MSKEW(48, 'corr_money_ask_size_0'))",
        "MMedian(24, ABS(SUBBED(MSKEW(96, 'mid_price_bias_ratio'), MMedian(24, MSKEW(96, 'mid_price_bias_ratio')))))",
        "MRANK(24, MSKEW(96, 'mid_price_bias_ratio'))",
        "MRANK(96, MSKEW(192, 'tick_in'))", "MRANK(96, 'tick_in')",
        "MUL('low', MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in')))))",
        "MKURT(48, 'realized_volatility')",
        "MRANK(192, MSKEW(96, MRANK(288, 'mid_price_bias_ratio')))",
        "MKURT(48, 'price_imbalance_4')",
        "EMA(24, DIV(SUBBED('volume_in', SHIFT(1, 'volume_in')), SHIFT(1, 'volume_in')))",
        "MRANK(96, MSKEW(192, 'mid_price_bias_ratio'))",
        "MRANK(96, 'mid_price_bias')",
        "MCORR(96, 'smart_volume_out', 'corr_vwap_bid_ask_price_spread')",
        "EMA(48, DIV(SUBBED('volume_in', SHIFT(1, 'volume_in')), SHIFT(1, 'volume_in')))",
        "MRANK(96, MSKEW(96, 'weighted_volume_bid5'))",
        "MSUM(96, MUL(MUL('smart_money_in_pct', 'corr_vwap_ret'), 'twap'))",
        "MUL(MMedian(192, ABS(SUBBED('smart_tick_in', MMedian(192, 'smart_tick_in')))), MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in')))))",
        "MRANK(288, MSKEW(192, MSKEW(48, 'mci_bid')))",
        "MRANK(192, MSKEW(24, 'mid_price_bias_ratio'))",
        "MA(24, EMA(288, 'mci_ask'))", "EMA(288, 'log_order_slope')",
        "EMA(192, DIV(SUBBED('volume_in', SHIFT(1, 'volume_in')), SHIFT(1, 'volume_in')))",
        "MSUM(288, MUL('bid_ask_spread', 'smart_money_out_pct'))",
        "MRANK(48, 'smart_tick_out_pct')",
        "DIV(SQRT(EMA(288, 'smart_tick_in')), SHIFT(1, SQRT(EMA(288, 'smart_tick_in'))))",
        "MSUM(288, MUL('smart_money_out_pct', 'price_imbalance_3'))",
        "MSUM(96, MUL('corr_vwap_ask_price_0', 'smart_money_out_pct'))",
        "MRANK(192, MSKEW(96, RSI(48, 'weighted_volume_ask5')))",
        "MSUM(288, MUL('price_imbalance_4', 'log_order_slope'))",
        "MUL(MUL(DIV(MSUM(192, MUL('smart_money_out_pct', 'tick_in_pct')), MSUM(192, 'tick_in_pct')), -1), -1)",
        "MMedian(96, ABS(SUBBED(MMedian(96, ABS(SUBBED('corr_money_ret', MMedian(96, 'corr_money_ret')))), MMedian(96, MMedian(96, ABS(SUBBED('corr_money_ret', MMedian(96, 'corr_money_ret'))))))))",
        "MUL(DIV(MSUM(192, MUL('smart_money_out_pct', 'tick_in_pct')), MSUM(192, 'tick_in_pct')), -1)",
        "ABS(EMA(288, 'mci_ask'))", "MA(48, EMA(288, 'mci_ask'))",
        "MRANK(288, MSTD(288, 'smart_tick_out_pct'))",
        "RSI(192, 'smart_tick_in_pct')", "SHIFT(1, EMA(288, 'mci_ask'))",
        "MSUM(288, MUL('smart_money_out_pct', 'log_order_slope'))",
        "MMedian(96, EMA(288, 'price_imbalance_2'))",
        "MConVariance(96, 'money', 'corr_money_ret')",
        "MSUM(192, MUL('corr_money_ask_size_0', MUL('delta_volume_ask1', -1)))",
        "MRANK(96, MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "MRANK(192, MKURT(96, 'order_flow_imbanlace_1'))",
        "DIV('bid_ask_spread', 'corr_ret_bid_size_0')",
        "MRANK(288, MMedian(48, ABS(SUBBED('smart_tick_in', MMedian(48, 'smart_tick_in')))))",
        "MUL(MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))), MMedian(96, ABS(SUBBED('mid_price_bias_ratio', MMedian(96, 'mid_price_bias_ratio')))))",
        "MSUM(288, MUL('price_imbalance_4', 'tick_out'))",
        "RSI(192, MKURT(96, 'order_flow_imbanlace_1'))",
        "MKURT(48, 'price_imbalance_2')",
        "MKURT(48, 'vwap5_ask_div_mid_price')",
        "RSI(288, MMedian(96, ABS(SUBBED(WMA(48, 'corr_ret_bid_ask_price_spread'), MMedian(96, WMA(48, 'corr_ret_bid_ask_price_spread'))))))",
        "MRANK(48, MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "MMIN(96, MUL('vwap5_bid_div_mid_price', SHIFT(1, 'depth_imbalance_4')))",
        "MRANK(288, MSKEW(48, MSKEW(192, 'mid_price_bias_ratio')))",
        "MConVariance(96, 'volume_in', 'corr_money_ret')",
        "MUL(MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))), 'mci_ask')",
        "EMA(288, 'price_imbalance_0')", "RSI(288, 'money_in')",
        "MUL('smart_tick_in', 'mci_bid')",
        "MA(144, MA(144, 'vwap5_ask_div_mid_price'))",
        "MSUM(288, MUL('smart_money_out_pct', 'mci_bid'))",
        "MA(192, MA(96, 'log_order_slope'))", "EMA(288, 'price_imbalance_3')",
        "EMA(288, 'price_imbalance_2')", "RSI(192, 'smart_tick_in')",
        "SHIFT(1, EMA(288, 'price_imbalance_2'))", "MRANK(24, 'money_in_pct')",
        "ABS(MA(144, MA(144, 'net_money_in_pct')))",
        "RSI(24, MSKEW(96, MCoef(96, 'delta_volume_ask1', 'net_tick_in')))",
        "DIV(MSUM(24, MUL('corr_vwap_ret', 'corr_money_ret')), MSUM(24, 'corr_money_ret'))",
        "MRANK(48, MSKEW(192, 'mid_price_bias_ratio'))",
        "RSI(288, 'smart_tick_in_pct')",
        "ABS(MA(96, MA(96, 'net_money_in_pct')))",
        "MA(192, EMA(96, 'mci_ask'))",
        "SHIFT(1, MA(96, MA(96, 'smart_money_out_pct')))",
        "MA(144, MA(144, 'price_imbalance_2'))",
        "MSUM(192, 'smart_money_out_pct')",
        "MSUM(192, MUL('smart_money_out_pct', -1))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'mci_ask'))",
        "MA(96, MA(96, MUL('smart_money_out_pct', -1)))",
        "MSUM(96, WMA(288, 'price_imbalance_3'))",
        "ABS(MA(96, MA(96, 'smart_money_out_pct')))",
        "MMedian(24, ABS(SUBBED('smart_tick_in_pct', MMedian(24, 'smart_tick_in_pct'))))",
        "MSUM(192, MUL('corr_money_ask_size_0', ABS('mci_bid')))",
        "MSUM(288, MUL('price_imbalance_0', 'price_imbalance_3'))",
        "MRANK(192, MSKEW(48, MSKEW(192, 'mid_price_bias_ratio')))",
        "MRANK(24, MSKEW(192, 'smart_tick_in'))",
        "MSUM(96, MUL('corr_vwap_bid_price_0', 'smart_money_out_pct'))",
        "RSI(288, 'smart_tick_in')",
        "MRANK(192, MSKEW(48, MUL('corr_money_ask_size_0', MSKEW(96, 'mid_price_bias_ratio'))))",
        "MSUM(192, MUL('smart_money_in_pct', 'smart_tick_in'))",
        "MRANK(288, MSKEW(24, 'mid_price_bias_ratio'))",
        "MSUM(192, MUL('smart_money_in_pct', 'smart_tick_in_pct'))",
        "MT3(24, MA(96, MA(96, 'smart_money_out_pct')))",
        "MA(144, MA(144, 'price_imbalance_0'))",
        "MA(144, MA(144, 'bid_ask_spread'))",
        "MSUM(192, MUL('smart_money_in_pct', 'money_in_pct'))",
        "ABS(MA(144, MA(144, 'price_imbalance_1')))",
        "MUL(MKURT(24, 'smart_money_out_pct'), -1)",
        "MKURT(24, 'smart_money_out_pct')",
        "DIV(MSUM(192, MUL('smart_money_in_pct', 'money_in_pct')), MSUM(192, 'money_in_pct'))",
        "MSUM(192, MUL('smart_money_in_pct', 'open'))",
        "SHIFT(1, MA(144, MA(144, 'bid_ask_spread')))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'delta_volume_bid5'))",
        "MSTD(288, 'ask_bid_press')", "MRANK(24, 'net_money_in')",
        "MRANK(48, MSKEW(96, 'mid_price_bias_ratio'))",
        "MSUM(288, 'log_order_slope')", "RSI(96, 'smart_tick_in')",
        "RSI(96, MUL('smart_tick_in', -1))",
        "MRANK(288, MSKEW(192, MMedian(96, ABS(SUBBED('money_in', MMedian(96, 'money_in'))))))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'price_imbalance_2'))",
        "MT3(288, MUL(MT3(288, 'smart_money_in'), 'smart_tick_in_pct'))",
        "MSUM(192, MUL('corr_money_ask_size_0', MUL('price_imbalance_2', -1)))",
        "MRANK(96, MSKEW(24, 'mid_price_bias_ratio'))",
        "MSUM(192, MUL('corr_money_ask_size_0', MUL('price_imbalance_3', -1)))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'price_imbalance_3'))",
        "MRANK(24, 'net_volume_in')",
        "MSUM(192, MUL('corr_money_ask_size_0', 'vwap5_bid_div_mid_price'))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'price_imbalance_1'))",
        "MRANK(192, MSKEW(96, DIV(SUBBED('close', SHIFT(1, 'close')), SHIFT(1, 'close'))))",
        "MMedian(192, ABS(SUBBED('depth_imbalance_0', MMedian(192, 'depth_imbalance_0'))))",
        "ABS('net_tick_in')",
        "MSUM(288, MUL('corr_vwap_ask_price_0', 'mci_bid'))",
        "MRANK(192, MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "RSI(96, 'corr_money_ask_size_0')", "SQRT(EMA(288, 'mci_ask'))",
        "MUL('twap', MUL(MSTD(96, 'smart_tick_in_pct'), -1))",
        "MRANK(96, MSKEW(96, 'mid_price_bias_ratio'))",
        "RSI(96, 'realized_volatility')", "MRANK(192, 'close')",
        "MSUM(192, MUL('mid_price_bias_ratio', MUL('smart_money_out_pct', -1)))",
        "MSUM(192, MUL('mid_price_bias_ratio', 'smart_money_out_pct'))",
        "MUL('price_imbalance_3', 'smart_tick_in')", "MSUM(288, 'mci_bid')",
        "RSI(96, MKURT(24, 'smart_money_out_pct'))",
        "MCoef(192, 'price_imbalance_1', 'depth_imbalance_4')",
        "MA(96, EMA(288, 'mci_ask'))", "MRANK(192, 'high')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'volume')), MSUM(96, 'volume'))",
        "MSUM(192, MUL('corr_money_ask_size_0', SHIFT(1, 'price_imbalance_1')))",
        "MRANK(288, MSKEW(96, 'mid_price_bias_ratio'))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'vwap5_ask_div_mid_price'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'volume_out')), MSUM(96, 'volume_out'))",
        "RSI(192, MMedian(48, ABS(SUBBED('smart_tick_in', MMedian(48, 'smart_tick_in')))))",
        "MUL(MMedian(48, ABS(SUBBED('smart_tick_in', MMedian(48, 'smart_tick_in')))), MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "MRANK(192, MSKEW(96, 'mid_price_bias_ratio'))",
        "MSUM(192, MUL('corr_money_ask_size_0', MT3(96, 'price_imbalance_1')))",
        "MSUM(288, MUL(MT3(192, 'price_imbalance_1'), 'smart_money_out_pct'))",
        "MSUM(96, MUL('smart_money_in_pct', 'smart_money_out_pct'))",
        "MMedian(192, 'net_tick_in')",
        "MCoef(96, 'price_imbalance_0', 'delta_volume_ask1')",
        "MUL(MT3(192, 'smart_money_in_pct'), DIV(MSUM(48, MUL('open', 'corr_money_vwap')), MSUM(48, 'corr_money_vwap')))",
        "MRANK(192, MMedian(192, ABS(SUBBED('smart_tick_in', MMedian(192, 'smart_tick_in')))))",
        "SQRT(MMedian(48, ABS(SUBBED('smart_tick_in', MMedian(48, 'smart_tick_in')))))",
        "MCORR(288, 'smart_tick_in_pct', 'corr_money_bid_size_0')",
        "MSUM(288, 'vwap5_ask_div_mid_price')", "MA(288, 'price_imbalance_4')",
        "MRANK(192, MSKEW(96, 'weighted_volume_ask5'))",
        "SHIFT(1, DIV(MSUM(288, MUL('price_imbalance_1', 'money')), MSUM(288, 'money')))",
        "DIV(SUBBED(MSUM(48, MUL('price_imbalance_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in'))), SHIFT(1, MSUM(48, MUL('price_imbalance_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in'))))), SHIFT(1, MSUM(48, MUL('price_imbalance_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in')))))",
        "MSUM(288, 'price_imbalance_3')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'smart_tick_out_pct')), MSUM(96, 'smart_tick_out_pct'))",
        "MUL('vwap5_ask_div_mid_price', 'smart_tick_in')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'smart_tick_out')), MSUM(96, 'smart_tick_out'))",
        "DIV(MSUM(288, MUL('bid_ask_spread', 'mid_price_bias')), MSUM(288, 'mid_price_bias'))",
        "MConVariance(192, 'corr_money_bid_price_0', 'corr_money_vwap')",
        "MConVariance(96, 'smart_money_out', 'smart_money_in_pct')",
        "MRANK(192, 'twap')", "MA(288, 'price_imbalance_2')",
        "MMIN(24, MRANK(96, 'volume_order_imbanlace_1'))",
        "MRANK(192, 'mid_price_bias')",
        "DIV(SUBBED(MSUM(48, MUL('corr_money_ask_size_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in'))), SHIFT(1, MSUM(48, MUL('corr_money_ask_size_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in'))))), SHIFT(1, MSUM(48, MUL('corr_money_ask_size_0', MCoef(96, 'delta_volume_ask1', 'net_tick_in')))))",
        "MRANK(288, MSKEW(96, MMedian(96, ABS(SUBBED(RSI(288, 'depth_imbalance_4'), MMedian(96, RSI(288, 'depth_imbalance_4')))))))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'tick_out')), MSUM(96, 'tick_out'))",
        "MMedian(192, 'tick_in_pct')",
        "RSI(96, MKURT(48, 'order_flow_imbanlace_1'))",
        "MSUM(192, MA(96, MA(96, 'corr_money_ask_size_0')))",
        "MMAX(24, MA(144, MA(144, 'bid_ask_spread')))",
        "RSI(96, MA(144, MA(144, 'tick_out')))", "MRANK(48, 'net_money_in')",
        "MSUM(288, 'price_imbalance_1')",
        "MRANK(192, MUL(MMedian(288, 'mci_ask'), 'smart_money_out_pct'))",
        "MCoef(24, 'tick_out', 'depth_imbalance_0')",
        "MSUM(192, MUL('corr_money_ask_size_0', 'smart_tick_out'))",
        "RSI(24, MSKEW(96, SQRT('smart_tick_out_pct')))",
        "MMedian(96, ABS(SUBBED(RSI(48, 'low'), MMedian(96, RSI(48, 'low')))))",
        "MConVariance(96, 'volume', 'delta_volume_ask5')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', SQRT('smart_tick_out_pct'))), MSUM(96, SQRT('smart_tick_out_pct')))",
        "MMedian(96, 'pct_change_close')",
        "MA(288, 'vwap5_bid_div_mid_price')", "MRANK(192, 'low')",
        "DIV('mid_price_bias_ratio', MMAX(192, 'log_order_slope'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', EMA(24, MUL('corr_money_ask_size_0', -1)))), MSUM(96, EMA(24, MUL('corr_money_ask_size_0', -1))))",
        "MSUM(48, MMedian(48, 'smart_money_out_pct'))",
        "MA(288, MA(24, 'vwap5_ask_div_mid_price'))",
        "MConVariance(96, 'smart_tick_in_pct', 'smart_tick_out')",
        "MSTD(96, 'smart_tick_out_pct')",
        "MSUM(192, MUL('corr_money_ask_size_0', 'depth_imbalance_1'))",
        "SHIFT(1, MMedian(192, 'smart_money_out_pct'))",
        "MMedian(192, 'smart_money_out_pct')",
        "MUL(MMedian(192, 'smart_money_out_pct'), -1)",
        "MSUM(192, MUL('corr_money_ask_size_0', MUL('bid_ask_spread', -1)))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'price_imbalance_0'))",
        "MSUM(192, MUL('corr_money_ask_size_0', 'bid_ask_spread'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'log_order_slope')), MSUM(96, 'log_order_slope'))",
        "SHIFT(1, MUL(MSKEW(48, 'volume'), MKURT(24, 'smart_money_in')))",
        "MCORR(96, 'tick_out', 'corr_money_bid_price_0')",
        "MSTD(192, 'corr_money_bid_price_0')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'price_imbalance_4')), MSUM(96, 'price_imbalance_4'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'bid_ask_spread')), MSUM(96, 'bid_ask_spread'))",
        "MSKEW(48, 'corr_money_ask_size_0')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'mci_ask')), MSUM(96, 'mci_ask'))",
        "MT3(288, DIFF('corr_money_ask_size_0'))",
        "MMAX(24, MA(144, MA(144, 'mci_bid')))",
        "RSI(24, MMedian(96, ABS(SUBBED('smart_tick_out_pct', MMedian(96, 'smart_tick_out_pct')))))",
        "MRANK(48, 'net_volume_in')", "MRANK(96, 'net_money_in')",
        "DIFF(EMA(96, 'corr_money_ask_size_0'))",
        "MSUM(288, MUL('corr_money_ask_size_0', 'log_order_slope'))",
        "MDEMA(96, MA(48, 'pct_change_close'))",
        "MA(24, MA(24, DIV(MSUM(288, MUL('price_imbalance_1', 'money')), MSUM(288, 'money'))))",
        "DIFF('pct_change')",
        "MRANK(288, MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))))",
        "MRANK(96, 'twap')", "MSTD(288, 'order_imbalance_ratio5')",
        "RSI(192, DIV(MSUM(192, MUL('smart_money_out_pct', 'tick_in_pct')), MSUM(192, 'tick_in_pct')))",
        "MSUM(192, MUL('corr_money_ask_size_0', SQRT('weighted_volume_ask5')))",
        "MA(96, 'smart_money_in_pct')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'smart_money_out_pct')), MSUM(96, 'smart_money_out_pct'))",
        "MRANK(192, MSKEW(192, 'smart_money_in_pct'))",
        "MKURT(24, 'vwap5_bid_div_mid_price')",
        "MMedian(288, 'vwap5_ask_div_mid_price')",
        "MRANK(288, MSKEW(192, MSKEW(48, 'smart_money_out_pct')))",
        "MA(288, MMedian(24, 'bid_ask_spread'))",
        "ABS(MSTD(96, 'smart_tick_out'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'mid_price_bias_ratio')), MSUM(96, 'mid_price_bias_ratio'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'money_in_pct')), MSUM(96, 'money_in_pct'))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'high')), MSUM(96, 'high'))",
        "MRANK(96, 'net_volume_in')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', WMA(288, 'tick_in'))), MSUM(96, WMA(288, 'tick_in')))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'close')), MSUM(96, 'close'))",
        "MA(192, EMA(192, 'mci_ask'))",
        "MUL(MMedian(96, ABS(SUBBED('smart_tick_in', MMedian(96, 'smart_tick_in')))), MMedian(24, ABS(SUBBED('smart_tick_in', MMedian(24, 'smart_tick_in')))))",
        "MSTD(24, MSKEW(24, 'money'))",
        "MKURT(24, MA(96, MA(96, MT3(48, 'corr_money_ret'))))",
        "DIV(MSUM(96, MUL('smart_money_in_pct', EMA(288, 'smart_money_out_pct'))), MSUM(96, EMA(288, 'smart_money_out_pct')))",
        "MSUM(288, MUL('corr_money_ask_size_0', MSTD(192, 'corr_ret_bid_ask_price_spread')))",
        "MSUM(24, MDEMA(96, 'smart_tick_in'))",
        "DIV('mid_price_bias_ratio', SHIFT(1, 'mid_price_bias_ratio'))",
        "DIFF('mid_price_bias_ratio')",
        "MSUM(192, MUL('corr_money_ask_size_0', 'log_order_slope'))",
        "RSI(192, MKURT(96, 'net_money_in_pct'))",
        "ABS(MT3(192, 'smart_money_in_pct'))", "MRANK(96, 'low')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', 'delta_volume_ask1')), MSUM(96, 'delta_volume_ask1'))",
        "MConVariance(48, 'money', 'corr_money_ret')",
        "DIV(MSUM(96, MUL('smart_money_in_pct', EMA(192, 'smart_money_out_pct'))), MSUM(96, EMA(192, 'smart_money_out_pct')))",
        "DIV(SUBBED(MCORR(288, 'money_out', 'order_flow_imbanlace_weighted5'), SHIFT(1, MCORR(288, 'money_out', 'order_flow_imbanlace_weighted5'))), SHIFT(1, MCORR(288, 'money_out', 'order_flow_imbanlace_weighted5')))",
        "DIV('realized_volatility', MMAX(192, 'log_order_slope'))",
        "ABS(EMA(288, 'net_tick_in'))",
        "MSUM(288, MUL('corr_money_ask_size_0', MSTD(96, 'corr_ret_bid_ask_price_spread')))",
        "MSUM(192, MUL('smart_money_out_pct', 'smart_money_out_pct'))",
        "MSUM(192, MUL('smart_money_out_pct', MUL('smart_money_out_pct', -1)))",
        "RSI(24, MKURT(96, 'smart_tick_in'))",
        "MA(288, MA(192, 'vwap5_ask_div_mid_price'))",
        "MRANK(192, MSKEW(192, 'mid_price_bias_ratio'))",
        "MA(288, SHIFT(1, MMAX(192, MA(12, MA(12, 'price_imbalance_3')))))",
        "DIFF(SQRT(EMA(96, 'smart_tick_in')))", "MA(288, EMA(288, 'mci_ask'))",
        "DIV(WMA(192, 'corr_money_ask_price_0'), MRANK(24, 'tick_out'))",
        "ABS(MSUM(192, MUL(MMAX(192, 'ask_bid_press'), MCoef(96, 'delta_volume_ask1', 'net_tick_in'))))"
    ]
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session,
          expressions=expressions)
