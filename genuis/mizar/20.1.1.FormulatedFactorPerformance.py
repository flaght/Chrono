import os, pdb
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from lib.iux001 import fetch_data, aggregation_data
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1


def run(method, instruments, period, datasets):
    total_data = fetch_data(method=method,
                            instruments=instruments,
                            datasets=datasets)

    factor_data = calc_expression(
        expression=expression, total_data=total_data.set_index('trade_time'))

    dt = aggregation_data(factor_data=factor_data,
                          returns_data=total_data,
                          period=period)

    evaluate1 = FactorEvaluate1(factor_data=dt,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=240,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression=expression)
    stats_dt = evaluate1.run()


if __name__ == '__main__':
    method = 'bicso0'
    instruments = 'ims'
    period = 15
    datasets = ['train', 'val', 'test'][1:2]
    expression = "MADiff(2,'ixy007_1_2_1')"
    run(method, instruments, period, datasets)
