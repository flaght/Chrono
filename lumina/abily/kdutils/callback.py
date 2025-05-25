import json, pdb
import pandas as pd
import numpy as np
import ultron.factor.empyrical as empyrical
from ultron.factor.genetic.geneticist.operators import calc_factor
from .adapter import data_adapter
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl
from kdutils.orders import position_next_order_nb, pnl_std


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def callback_models(gen, rootid, best_programs, custom_params):
    g_instruments = custom_params['g_instruments']
    dethod = custom_params['dethod']
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)

    data_programs = best_programs.copy()
    data_programs['task_id'] = rootid
    data_programs['strategy_params'] = data_programs['strategy_params'].apply(
        lambda x: json.dumps(x, cls=NpEncoder))
    data_programs['signal_params'] = data_programs['signal_params'].apply(
        lambda x: json.dumps(x, cls=NpEncoder))

    data_adapter.refresh_data(total_data=data_programs.drop(
        ['update_time', 'features'], axis=1),
                              method='increment',
                              table_name='genetic_strategy')


def callback_fitness(factor_data, total_data, signal_method, strategy_method,
                     factor_sets, custom_params, default_value):
    pdb.set_trace()
    strategy_settings = custom_params['strategy_settings']
    factor_data = factor_data.reset_index().set_index(['trade_time', 'code'])
    total_data = total_data.set_index(['trade_time', 'code']).unstack()
    pos_data = signal_method.function(factor_data=factor_data,
                                      **signal_method.params)
    pos_data = strategy_method.function(signal=pos_data,
                                        total_data=total_data,
                                        **strategy_method.params)

    ### 包含order 则从订单开始评估
    if "order" in custom_params['method']:
        orders = position_next_order_nb(pos_data=pos_data,
                                        market_data=total_data)
        if 'order_std' in custom_params['method']:
            fitness = pnl_std(orders=orders)

    else:
        total_data['trade_vol', total_data['open'].columns[0]] = (
            custom_params['strategy_settings']['capital'] /
            total_data['open'] / custom_params['strategy_settings']['size'])
        df = calculate_ful_ts_pnl(pos_data=pos_data,
                                  total_data=total_data,
                                  strategy_settings=strategy_settings)
        returns = df['ret']
        if 'sharpe' in custom_params['method']:
            fitness = empyrical.sharpe_ratio(returns=returns,
                                             period=empyrical.DAILY)
            #method_funciton = getattr(empyrical,
            #                          '{0}_ratio'.format(custom_params['method']))
            #fitness = method_funciton(returns=returns)
    return fitness


def create_fitness(total_data, strategy, strategy_settings):
    total_dt = total_data.copy()
    factors_data = calc_factor(expression=strategy.formual,
                               total_data=total_dt.set_index(['trade_time']),
                               key='code',
                               indexs=[])
    factors_data1 = factors_data.reset_index().set_index(
        ['trade_time', 'code'])
    total_data1 = total_dt.set_index(['trade_time', 'code']).unstack()
    pos_data = eval(strategy.signal_method)(factor_data=factors_data1,
                                            **json.loads(
                                                strategy.signal_params))
    pos_data1 = eval(strategy.strategy_method)(signal=pos_data,
                                               total_data=total_data1,
                                               **json.loads(
                                                   strategy.strategy_params))

    total_dt['trade_vol',
             total_dt['open'].columns[0]] = (strategy_settings['capital'] /
                                             total_dt['open'] /
                                             strategy_settings['size'])
    df = calculate_ful_ts_pnl(pos_data=pos_data1,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['ret']
    fitness = empyrical.calmar_ratio(returns=returns, period=empyrical.DAILY)
    return fitness
