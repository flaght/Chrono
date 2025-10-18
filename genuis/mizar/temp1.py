import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.aux001 import fetch_temp_data
from lib.svx001 import create_position, read_factors
from lib.cux002 import StrategyEvaluate1


def run(method, instruments, period, task_id, name, signal_method,
        strategy_settings, signal_params, strategy_method, strategy_params):
    predict_data = read_factors(method=method,
                                instruments=instruments,
                                task_id=task_id,
                                period=period,
                                name=name)
    pdb.set_trace()
    data = fetch_temp_data(method=method,
                    instruments=instruments,
                    task_id=task_id,
                    datasets=['val', 'test'],
                    category='data')
    predict_data['transformed'] = predict_data['predict']
    predict_data = predict_data.merge(data[['trade_time','code','close']])
    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)

    eval1 = StrategyEvaluate1(pos_data=pos_data,
                              total_data=total_data2,
                              strategy_settings=strategy_settings,
                              strategy_name=signal_method,
                              ret_name = None)
                             # ret_name='nxt1_ret_{0}h'.format(period))
    print(eval1.run())


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
