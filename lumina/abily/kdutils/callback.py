import json, pdb
import pandas as pd
import numpy as np
import ultron.factor.empyrical as empyrical
from .adapter import data_adapter
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_pnl


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

    strategy_settings = custom_params['strategy_settings']
    factor_data = factor_data.reset_index().set_index(['trade_time', 'code'])
    total_data = total_data.set_index(['trade_time', 'code']).unstack()
    pos_data = signal_method.function(factor_data=factor_data,
                                      **signal_method.params)
    pos_data = strategy_method.function(signal=pos_data,
                                        total_data=total_data,
                                        **strategy_method.params)
    pdb.set_trace()
    df = calculate_ful_ts_pnl(pos_data=pos_data,
                              total_data=total_data,
                              strategy_settings=strategy_settings)
    ### 值有异常 绝对值大于1
    returns = df['ret']
    fitness = empyrical.sharpe_ratio(returns=returns, period=empyrical.DAILY)
    return fitness
