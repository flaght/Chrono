import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

from lib.iux001 import fetch_data, aggregation_data,merging_data1
from lib.aux001 import calc_expression, extract_operators
from lib.cux001 import FactorEvaluate1

from kdutils.macro2 import *

def func1():
    method = 'cicso0'
    instruments = 'ims'
    period = 15
    task_id = '200037'
    datasets = ['train','val','test']

    total_data = fetch_data(method=method,
                        instruments=instruments,
                        task_id=task_id,
                        datasets=datasets)

    total_data.filter(regex="^nxt1").columns.to_list()
    nxt1_columns = total_data.filter(regex="^nxt1").columns.to_list()
    basic_columns = [
        'close', 'high', 'low', 'open', 'volume'
        #, 'value', 'openint'
    ]

    regex_pattern = r'^[^_]+_(5|10|15)_.*'
    not_columns = total_data.columns[total_data.columns.str.contains(
        regex_pattern)]

    factor_columns = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns + basic_columns +
        not_columns.tolist()
    ]#[0:100]

    total_data = total_data[['trade_time', 'code'] + factor_columns + basic_columns + ["nxt1_ret_{}h".format(period)]]

    expression = "DELTA(120,MSUM(120,DELTA(90,'high')))" # 反方向

    factor_data = calc_expression(expression=expression,
                              total_data=total_data.set_index('trade_time'))

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
    evaluate1.run(direction=-1)

def func2():
    method = 'cicso0'
    instruments = 'ims'
    period = 15
    form = 'linear_nav_1_90_data'
    name = 'final'
    task_id = str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])

    dirs = os.path.join(base_path, method, instruments, 'temp', "model", task_id,
                    str(period))

    filename = os.path.join(dirs, "{0}.feather".format(form,name))

    predict_data = pd.read_feather(filename) ## 计算时候已经调整了方向

    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)
    predict_data['predict'] = predict_data['predict']

    evaluate1 = FactorEvaluate1(factor_data=predict_data,
                                factor_name='predict',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.0,
                                scale_method='raw',
                                resampling_win=15,
                                expression="{0}_{1}".format(form, name))
    pdb.set_trace()
    stats_dt = evaluate1.run(direction=1)

func2()