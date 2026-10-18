import time, pdb, json, itertools
import pandas as pd
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.process import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl
from lumina.genetic.fusion.orders import *
from lumina.genetic.fusion.macro import EmpyricalTuple, EmpyoicalTuple
import ultron.factor.empyrical as empyrical
from ultron.factor.genetic.geneticist.operators import *


### 批量评估策略
@add_process_env_sig
def run_metrics(target_column, total_data1, total_data2, strategy_setting):
    metrics = run_process(target_column=target_column,
                          callback=create_metrics,
                          total_data1=total_data1,
                          total_data2=total_data2,
                          strategy_setting=strategy_setting)
    return metrics


def create_metrics(column, total_data1, total_data2, strategy_setting):
    total_dt = total_data2.copy()
    ### total_data1  dataframe index: trade_time
    ### total_data2  矩阵
    factors_data = calc_factor(expression=column.formual,
                               total_data=total_data1,
                               key='code',
                               indexs=[])
    factors_data1 = factors_data.reset_index().set_index(
        ['trade_time', 'code'])

    pos_data = eval(column.signal_method)(factor_data=factors_data1,
                                          **json.loads(column.signal_params))
    pos_data1 = eval(column.strategy_method)(signal=pos_data,
                                             total_data=total_dt,
                                             **json.loads(
                                                 column.strategy_params))

    if False:#"order" in strategy_setting['method']:
        orders = position_next_order_cy(
            pos_data=pos_data1,
            market_data=total_dt,
            commission=strategy_setting['commission'],
            slippage=strategy_setting['slippage'],
            name='open')

        win_rate1 = win_rate(orders=orders)
        profit_rate1 = profit_rate(orders=orders)
        profit_std1 = profit_std(orders=orders)
        metrics = EmpyoicalTuple(name=column.name,
                                 win_rate=win_rate1,
                                 profit_rate=profit_rate1,
                                 profit_std=profit_std1)

    else:
        total_dt['trade_vol',
                 total_dt['open'].columns[0]] = (strategy_setting['capital'] /
                                                 total_dt['open'] /
                                                 strategy_setting['size'])
        df = calculate_ful_ts_pnl(pos_data=pos_data1,
                                  total_data=total_dt,
                                  strategy_settings=strategy_setting)
        returns = df['ret']
        calmar_ratio = empyrical.calmar_ratio(returns=returns,
                                              period=empyrical.DAILY)
        sharpe_ratio = empyrical.sharpe_ratio(returns=returns,
                                              period=empyrical.DAILY)
        sortino_ratio = empyrical.sortino_ratio(returns=returns,
                                                period=empyrical.DAILY)
        max_drawdown = empyrical.max_drawdown(returns=returns)
        annual_return = empyrical.annual_return(returns=returns,
                                                period=empyrical.DAILY)
        annual_volatility = empyrical.annual_volatility(returns=returns,
                                                        period=empyrical.DAILY)

        metrics = EmpyricalTuple(name=column.name,
                                 annual_return=annual_return,
                                 annual_volatility=annual_volatility,
                                 calmar=calmar_ratio,
                                 sharpe=sharpe_ratio,
                                 max_drawdown=max_drawdown,
                                 sortino=sortino_ratio,
                                 returns_series=returns)
    return metrics


class Thruster(object):

    def __init__(self, k_split=1):
        self.k_split = k_split

    def calculate(self, strategies_infos, strategy_setting, total_data):
        total_data1 = total_data.set_index(['trade_time'])
        total_data2 = total_data.set_index(['trade_time', 'code']).unstack()
        process_list = split_k(self.k_split, strategies_infos)

        res = create_parellel(process_list=process_list,
                              callback=run_metrics,
                              total_data1=total_data1,
                              total_data2=total_data2,
                              strategy_setting=strategy_setting)
        res = list(itertools.chain.from_iterable(res))
        return res
