import os, pdb, math, itertools
import pandas as pd
from ultron.factor.genetic.geneticist.operators import *
from lib.iux001 import fetch_data, aggregation_data, fetch_times
from lib.aux001 import calc_expression


def fetch_data1(method, instruments, datasets, period, expressions):
    total_data = fetch_data(method=method,
                            instruments=instruments,
                            datasets=datasets)
    program_list = list(expressions.keys())
    features = [eval(program)._dependency for program in program_list]
    features = list(itertools.chain.from_iterable(features))
    features = list(set(features))
    total_data = total_data[['trade_time', 'code'] + features +
                            ['nxt1_ret_{}h'.format(period)]]
    return total_data


def create_factors(total_data, expressions):
    res = []
    total_data1 = total_data.set_index('trade_time')
    for program, direction in expressions.items():
        print(program)
        factor_data = calc_expression(expression=program,
                                      total_data=total_data1)
        factor_data['transformed'] = factor_data['transformed'] * direction
        factor_data = factor_data.set_index(['trade_time', 'code'])
        factor_data.rename(columns={'transformed': program}, inplace=True)
        res.append(factor_data)
    factors_data = pd.concat(res, axis=1)
    return factors_data


## 因子等权合成降频
def create_equal(factors_data, total_data, period):
    final_data = factors_data.mean(axis=1)
    final_data.name = 'transformed'
    final_data = aggregation_data(final_data.reset_index(), total_data, period)
    return final_data












#### 
def equal_weight(method,
                 instruments,
                 period,
                 expressions,
                 datasets=['train', 'val', 'test']):
    total_data = fetch_data1(method=method,
                             instruments=instruments,
                             datasets=datasets,
                             period=period,
                             expressions=expressions)
    factors_data = create_factors(total_data=total_data,
                                  expressions=expressions)
    final_data = create_equal(factors_data=factors_data,
                              total_data=total_data,
                              period=period)
    return final_data
