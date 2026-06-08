import os, sys, pdb
import pandas as pd
import numpy as np

from ultron.factor.genetic.geneticist.operators import *
from create_data import load_random_data

sys.path.insert(0, os.path.abspath('../'))
from lumina.genetic.signal.method.divergence import divergence_signal

from lumina.genetic.strategy.method.atr import trailing_atr_strategy
from lumina.genetic.strategy.method.drawdown import drawdown_strategy
from lumina.genetic.strategy.method.gap import gap_strategy
from lumina.genetic.strategy.method.meanrevert import meanrevert_strategy
from lumina.genetic.strategy.method.momentum import momentum_strategy
from lumina.genetic.strategy.method.pricegap import pricegap_strategy
from lumina.genetic.strategy.method.quantile import quantile_strategy
from lumina.genetic.strategy.method.range import range_strategy
from lumina.genetic.strategy.method.rollingstddev import rollingstddev_strategy
from lumina.genetic.strategy.method.trend import trend_strategy
from lumina.genetic.strategy.method.turnover import turnover_strategy
from lumina.genetic.strategy.method.open_interest import open_interest_strategy


def create_basic_data():
    columns = [
        'close', 'low', 'high', 'open', 'volume', 'value', 'openint', 'chg',
        'price'
    ]
    data = load_random_data(ticker_dim=1,
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data  #.unstack()


def create_factors_data():
    columns = [
        'rv003_5_10_0_1', 'db007_5_10_1', 'dv008_5_10_1', 'ixy011_5_10_1'
    ]
    data = load_random_data(ticker_dim=1,
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data


def main():
    pdb.set_trace()
    backup_cycle = 1

    data1 = create_factors_data()
    data2 = create_basic_data()
    total_data = pd.concat([data1, data2], axis=1)
    total_data1 = total_data.reset_index().set_index('trade_time').sort_index()
    total_data2 = total_data.sort_index().unstack()
    pdb.set_trace()
    exp1 = "MADecay(18,MACCBands(14,MMAX(16,'rv003_5_10_0_1'),MIR(20,'db007_5_10_1')))"
    factor_data = calc_factor(expression=exp1,
                              total_data=total_data1,
                              indexs=[],
                              key='code')
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data['transformed'] = np.where(
        np.abs(factor_data.transformed.values) > 0.000001,
        factor_data.transformed.values, np.nan)
    factor_data = factor_data.loc[factor_data.index.unique()[backup_cycle:]]

    factors_data1 = factor_data.reset_index().set_index(['trade_time', 'code'])
    pdb.set_trace()
    signal = divergence_signal(factor_data=factors_data1.dropna(),
                               roll_num=100,
                               threshold=0.4)

    pos_data = open_interest_strategy(signal=signal, total_data=total_data2)
    '''
    pos_data = trailing_atr_strategy(signal=signal,
                          total_data=total_data2,
                          atr_period=5,
                          atr_multiplier=0.2,
                          max_volume=1)
    '''
    pdb.set_trace()
    print('-->')


main()
