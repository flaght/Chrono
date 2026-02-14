import os, pdb
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from lib.iux001 import fetch_data, aggregation_data,merging_data1
from lib.aux001 import calc_expression
from lib.cux001 import FactorEvaluate1



def run(method, instruments, period, datasets, expression, task_id):
    total_data = fetch_data(method=method,
                        instruments=instruments,
                        task_id=task_id,
                        datasets=datasets)
    total_data.filter(regex="^nxt1").columns.to_list()
    nxt1_columns = total_data.filter(regex="^nxt1").columns.to_list()
    basic_columns = [
        'close', 'high', 'low', 'open', 'volume']

    regex_pattern = r'^[^_]+_(5|10|15)_.*'
    not_columns = total_data.columns[total_data.columns.str.contains(
        regex_pattern)]

    factor_columns = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns + basic_columns +
        not_columns.tolist()
        ]#[0:100]
    total_data = total_data[['trade_time', 'code'] + factor_columns + basic_columns + ["nxt1_ret_{}h".format(period)]]

    factor_data = calc_expression(expression=expression,
                              total_data=total_data.set_index('trade_time'))
    pdb.set_trace()
    dt = merging_data1(factor_data=factor_data,
                      returns_data=total_data,
                      period=period)
    evaluate1 = FactorEvaluate1(factor_data=dt,
                                factor_name='transformed',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='roll_zscore',
                                expression=expression,
                                resampling_win=period)
    pdb.set_trace()
    stats_dt = evaluate1.run()
    evaluate1.plot_results()
    evaluate1.save_results("./temp/2/")




if __name__ == '__main__':
    method = 'cicso1'
    instruments = 'ims'
    period = 15
    task_id = '200037'
    datasets = ['train','val']
    expression = "MMedian(120,MDPO(5,MSUM(30,'mid_price_bias')))"  #"MADiff(2,'ixy007_1_2_1')"
    run(method=method, instruments=instruments, 
            period=period, datasets=datasets, expression=expression, 
            task_id=task_id)
