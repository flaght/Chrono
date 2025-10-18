import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lib.cux001 import FactorEvaluate1
from lib.svx001 import create_position, read_factors, scale_factors


def run1(method, instruments, period, name, task_id):
    signal_method = 'rollrank_signal'
    signal_params = {'roll_num': 24, 'threshold': 0.7}
    strategy_method = None
    strategy_params = None
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        task_id, str(period))
    filename = os.path.join(dirs, "{0}_predict_data.feather".format(name))
    predict_data = pd.read_feather(filename)
    is_on_mark = predict_data['trade_time'].dt.minute % int(period) == 0
    predict_data = predict_data[is_on_mark]
    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)
    #scale_factors(predict_data,
    #              method='roll_zscore',
    #              win=240,
    #              factor_name='predict')
    pdb.set_trace()
    predict_data['transformed'] = predict_data['predict']
    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)
    pos_data = pos_data.stack()
    pos_data.name = 'pred_alpha_disc'
    pos_data = pos_data.reset_index()
    predict_data = predict_data.merge(pos_data, on=['trade_time', 'code'])
    predict_data = predict_data[[
        'trade_time', 'code', 'nxt1_ret_5h', 'predict', 'pred_alpha_disc'
    ]]
    pdb.set_trace()

    evaluate1 = FactorEvaluate1(factor_data=predict_data,
                                factor_name='pred_alpha_disc',
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=240,
                                fee=0.000012,
                                scale_method='raw',
                                expression=name)
    stats_dt = evaluate1.run()
    print('->')

def run2(method, instruments, period, name, task_id):
    signal_method = 'rollrank_signal'
    signal_params = {'roll_num': 24, 'threshold': 0.7}
    strategy_method = None
    strategy_params = None
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        task_id, str(period))
    filename = os.path.join(dirs, "{0}_predict_data.feather".format(name))
    predict_data = pd.read_feather(filename)
    is_on_mark = predict_data['trade_time'].dt.minute % int(period) == 0
    predict_data = predict_data[is_on_mark]
    predict_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_data.dropna(inplace=True)

    predict_data['transformed'] = predict_data['predict']
    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)
    pdb.set_trace()
    print()




if __name__ == '__main__':
    method = 'cicso0'
    instruments = 'ims'
    period = 5
    name = 'lgbm'
    task_id = '200037'
    run1(method=method,
         instruments=instruments,
         period=period,
         name=name,
         task_id=task_id)
