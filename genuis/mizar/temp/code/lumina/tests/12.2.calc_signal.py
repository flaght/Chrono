import os, sys, pdb
import pandas as pd
import numpy as np

from ultron.factor.genetic.geneticist.operators import *
from create_data import load_random_data

sys.path.insert(0, os.path.abspath('../'))
from lumina.genetic.signal.method.divergence import divergence_signal
from lumina.genetic.signal.method.volatility import volatility_signal
from lumina.genetic.signal.method.autocorr import autocorr_signal
from lumina.genetic.signal.method.breakout import breakout_signal
from lumina.genetic.signal.method.entropy import entropy_signal
from lumina.genetic.signal.method.extreme import extreme_count_signal
from lumina.genetic.signal.method.kurtosis import kurtosis_signal
from lumina.genetic.signal.method.oscillator import oscillator_signal
from lumina.genetic.signal.method.regression import regression_signal
from lumina.genetic.signal.method.skewness import skewness_signal
from lumina.genetic.signal.method.icu import icu_signal
from lumina.genetic.signal.method.rsrs import rsrs_signal


def create_basic_data():
    columns = [
        'close', 'low', 'high', 'open', 'volume', 'value', 'openint', 'chg',
        'price'
    ]
    data = load_random_data(ticker_dim=4,
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data.unstack()


def create_factors_data():
    columns = ['rv003_5_10_0_1', 'db007_5_10_1', 'dv008_5_10_1', 'ixy011_5_10_1']
    data = load_random_data(ticker_dim=1,
                            factors_dim=len(columns) - 1,
                            res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data


def main():
    pdb.set_trace()
    backup_cycle = 1
    data = create_factors_data()
    data = data.reset_index().set_index('trade_time').sort_index()
    pdb.set_trace()
    exp1 = "MADecay(18,MACCBands(14,MMAX(16,'rv003_5_10_0_1'),MIR(20,'db007_5_10_1')))"
    factor_data = calc_factor(expression=exp1,
                              total_data=data,
                              indexs=[],
                              key='code')
    factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
    factor_data['transformed'] = np.where(
        np.abs(factor_data.transformed.values) > 0.000001,
        factor_data.transformed.values, np.nan)
    factor_data = factor_data.loc[factor_data.index.unique()[backup_cycle:]]

    factors_data1 = factor_data.reset_index().set_index(['trade_time', 'code'])
    pdb.set_trace()
    signal = rsrs_signal(factor_data=factors_data1.dropna(),
                               roll_num=100,
                               threshold=0.4)
    pdb.set_trace()
    print('-->')


main()
