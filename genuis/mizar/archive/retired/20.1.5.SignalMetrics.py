import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.aux001 import fetch_temp_data
from lib.svx001 import create_position, read_factors, scale_factors
from lib.cux002 import StrategyEvaluate1


def run(method, instruments, period, task_id, name, signal_method,
        strategy_settings, signal_params, strategy_method, strategy_params):

    basic_data = fetch_temp_data(method=method,
                                 instruments=instruments,
                                 task_id=task_id,
                                 datasets=['train', 'val', 'test'])
    predict_data = read_factors(method=method,
                                instruments=instruments,
                                task_id=task_id,
                                period=period,
                                name=name)
    scale_factors(predict_data,
                  method='roll_zscore',
                  win=240,
                  factor_name='predict')
    predict_data = predict_data.merge(basic_data[[
        'trade_time', 'code', 'open', 'high', 'low', 'low', 'close', 'volume'
    ]])

    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)

    eval1 = StrategyEvaluate1(pos_data=pos_data,
                              total_data=total_data2,
                              strategy_settings=strategy_settings,
                              strategy_name=signal_method,
                              ret_name='nxt1_ret_{0}h'.format(period))
    eval1.run()


if __name__ == '__main__':
    method = 'cicso0'
    instruments = 'ims'
    period = 5
    name = 'lgbm'
    task_id = '200037'

    signal_method = 'rollrank_signal'
    signal_params = {'roll_num': 24, 'threshold': 0.9}

    strategy_method = None
    strategy_params = None

    strategy_settings = {'commission': 0.000012, 'slippage': 0, 'size': 200}

    run(method=method,
        instruments=instruments,
        period=period,
        task_id=task_id,
        name=name,
        strategy_settings=strategy_settings,
        signal_method=signal_method,
        signal_params=signal_params,
        strategy_method=strategy_method,
        strategy_params=strategy_params)
