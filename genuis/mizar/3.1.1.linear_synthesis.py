import pandas as pd
import numpy as np
import pdb, argparse
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()
from ultron.factor.genetic.geneticist.operators import *
from lib.iux001 import fetch_data, aggregation_data
from lib.aux001 import calc_expression

expressions = {
    "DELTA(4,MMASSI(2,'mid_price_bias_ratio',MSUM(3,'corr_money_ret')))": -1,
    "DELTA(3,DELTA(3,'twap'))": -1,
    "MMedian(2,MDPO(4,MT3(3,'pct_change')))": -1,
    "DELTA(3,DELTA(3,'mid_price_bias_ratio'))": 1,
    "MT3(2,EMA(2,MT3(2,MCPS(2,MT3(2,EMA(2,DIFF('twap')))))))": 1,
    "MDIFF(4,DIFF(MDEMA(4,MINIMUM('mid_price_bias_ratio','pct_change_set'))))":
    -1
}


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


def create_equal(factors_data, total_data, period):
    final_data = factors_data.mean(axis=1)
    final_data.name = 'transformed'
    final_data = aggregation_data(final_data.reset_index(), total_data, period)
    return final_data


def equal_weight(method,
                 instruments,
                 period,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--method',
                        type=str,
                        default='aicso0',
                        help='data method')
    parser.add_argument('--instruments',
                        type=str,
                        default='ims',
                        help='code or instruments')

    parser.add_argument('--period', type=int, default=5, help='period')

    parser.add_argument('--session',
                        type=str,
                        default=202509221,
                        help='session')

    args = parser.parse_args()
    equal_weight(method=args.method,
                 instruments=args.instruments,
                 period=args.period)
