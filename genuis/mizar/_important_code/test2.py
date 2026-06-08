import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()
from ultron.datasets.generate import create_data
from lib.optim002.calculator import ImpulseCalculator
from lib.optim002.optimizer import FactorsOptimizer

def create_factors(start_time, end_time, m=30, n=10, freq='T', res_name=None):
    date_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    date_index.name = 'trade_time'
    codes = ["code_" + str(i) for i in range(0, n)]

    factors_res = [
        create_data(date_index=date_index,
                    codes=codes,
                    name="factor_{}".format(str(i))) for i in range(0, m)
    ]
    if isinstance(res_name, str):
        factors_res.append(
            create_data(date_index=date_index, codes=codes, name=res_name))
    factors_data = pd.concat(factors_res, axis=1)

    factors_data = factors_data.reset_index().rename(
        columns={'level_1': 'code'})
    return factors_data

def load_random_data(ticker_dim,
                     factors_dim,
                     res_name=None,
                     start_time='2023-06-01 00:01:00',
                     end_time='2023-06-02 00:01:00'):
    data = create_factors(start_time,
                          end_time,
                          m=factors_dim,
                          n=ticker_dim,
                          res_name=res_name)
    data['price'] = np.abs((data['factor_0'] + data['factor_1']) / 2) * 10
    data.index = data['trade_time'].rank(method='dense').astype(int) - 1
    data.index.name = None
    return data

def create_data1():
    columns = ['close','low','high','open','volume','value','openint','chg', 'price','nxt1_ret_15h']
    data = load_random_data(ticker_dim=4, factors_dim=len(columns) - 1, res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data.unstack()


'''
impulse = ImpulseCalculator(impulse_version='i017')
market_data = create_data1()

impulse.calculate_with_class(impulse.get_class(name='kx001'),
            params=[(1, 2, 1), (2, 3, 1)], market_data=market_data)
'''

market_data = create_data1()
returns_data = market_data['nxt1_ret_15h']
returns_data = returns_data.unstack()
returns_data.name = 'nxt1_ret_15h'
returns_data = returns_data.reset_index()

fo = FactorsOptimizer(impulse_version='i017', n_jobs=1, evaluator_params={'scale_method':'roll_zscore','roll_win':15,
                                                                                    'resampling_win':15}, param_ranges={})
fo.optimize_parallel(factor_names=['kx001','kx002'],
                    market_data=market_data,
                    returns_data=returns_data,
                    period=15, n_trials=10,
                    top_n=0)

