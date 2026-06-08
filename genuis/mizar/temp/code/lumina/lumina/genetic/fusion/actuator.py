import time, pdb, json, itertools, warnings
import pandas as pd
import numpy as np
from lumina.genetic.signal.method import *
from lumina.genetic.strategy.method import *
from lumina.genetic.process import *
from ultron.factor.genetic.geneticist.operators import *

warnings.filterwarnings("ignore", category=DeprecationWarning)


### 批量生成策略
@add_process_env_sig
def run_position(target_column, total_data1, total_data2):
    position_data = run_process(target_column=target_column,
                                callback=create_position,
                                total_data1=total_data1,
                                total_data2=total_data2)
    return position_data


def create_position(column, total_data1, total_data2):
    factors_data = calc_factor(expression=column.formual,
                               total_data=total_data1,
                               key='code',
                               indexs=[])

    factors_data = factors_data.replace([np.inf, -np.inf], np.nan)
    factors_data['transformed'] = np.where(
        np.abs(factors_data.transformed.values) > 0.000001,
        factors_data.transformed.values, np.nan)
    factors_data = factors_data.loc[factors_data.index.unique()[1:]]

    factors_data1 = factors_data.reset_index().set_index(
        ['trade_time', 'code'])

    signal_params = column.signal_params if isinstance(
        column.signal_params, dict) else json.loads(column.signal_params)
    strategy_params = column.strategy_params if isinstance(
        column.strategy_params, dict) else json.loads(column.strategy_params)

    pos_data = eval(column.signal_method)(factor_data=factors_data1,
                                          **signal_params)
    pos_data1 = eval(column.strategy_method)(signal=pos_data,
                                             total_data=total_data2,
                                             **strategy_params)
    pos_data1 = pos_data1['pos']
    pos_data1 = pos_data1.stack()
    pos_data1.name = column.name
    pos_data1.index.names = [pos_data1.index.names[0], 'code']
    return pos_data1


class Actuator(object):

    def __init__(self, k_split=1):
        self.k_split = k_split

    def syntheti_signal(self, strategies_infos, total_data):
        strategies_data = self.calculate(strategies_infos, total_data)
        weights_data = pd.DataFrame(strategies_infos)[['name', 'fitness']]
        weights_data = weights_data.set_index('name')
        weights_data = weights_data / weights_data.sum()
        strategies_data.mul(weights_data['fitness'], axis=1)
        positions_data = strategies_data.mul(weights_data['fitness'],
                                             axis=1).sum(axis=1)
        return positions_data

    ### 权重
    def fitness_weight(self, strategies_infos):
        weights_data = pd.DataFrame(strategies_infos)[['name', 'fitness']]
        weights_data = weights_data.set_index('name')
        weights_data = weights_data / weights_data.sum()
        return weights_data

    ### 信号融合
    def fitness_signal(self,
                       method,
                       strategies_infos,
                       strategies_data,
                       weights_data=None):
        if method == 'weight' and weights_data is not None:
            weights_data = self.fitness_weight(strategies_infos)
            positions_data = strategies_data.mul(weights_data['fitness'],
                                                 axis=1).sum(axis=1)
            positions_data.name = 'fitness_weight'
        elif method == 'equal':
            positions_data = strategies_data.sum(axis=1)
            positions_data = positions_data / len(strategies_data.columns)
            positions_data.name = 'equal_weight'
        elif method == 'volatility':
            volatilities = strategies_data.diff().abs().mean()
            inverse_volatilities = 1 / (volatilities + 1e-8)
            total_inverse_vol = inverse_volatilities.sum()
            normalized_weights = inverse_volatilities / total_inverse_vol
            positions_data = strategies_data.mul(normalized_weights,
                                           axis='columns').sum(axis=1)
            positions_data.name = 'vol_inv_weight'
        return positions_data

    ## 计算各个策略的信号
    def calculate(self, strategies_infos, total_data):
        total_data1 = total_data.set_index(['trade_time'])
        total_data2 = total_data.set_index(['trade_time', 'code']).unstack()
        process_list = split_k(self.k_split, strategies_infos)
        res = create_parellel(process_list=process_list,
                              callback=run_position,
                              total_data1=total_data1,
                              total_data2=total_data2)
        res = list(itertools.chain.from_iterable(res))
        strategies_data = pd.concat(res, axis=1)
        return strategies_data
