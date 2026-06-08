import os, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lib.cux001 import FactorEvaluate1
from lib.cux002 import StrategyEvaluate1
from lib.svx001 import create_position, read_factors, scale_factors


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
    strategy_settings = {'commission': 0.000012, 'slippage': 0, 'size': 200}
    predict_data['transformed'] = predict_data['predict']
    pos_data, total_data2 = create_position(predict_data=predict_data,
                                            signal_method=signal_method,
                                            signal_params=signal_params,
                                            strategy_method=strategy_method,
                                            strategy_params=strategy_params)
    eval1 = StrategyEvaluate1(
        pos_data=pos_data,
        total_data=total_data2,
        strategy_settings=strategy_settings,
        strategy_name=signal_method,
        ret_name='nxt1_ret_{0}h'.format(period),
    )
    pdb.set_trace()
    eval1.run()
    print()


if __name__ == '__main__':
    method = 'cicso0'
    instruments = 'ims'
    period = 5
    name = 'lgbm'
    task_id = '200037'
    run2(method=method,
         instruments=instruments,
         period=period,
         name=name,
         task_id=task_id)
