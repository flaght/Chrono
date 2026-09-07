import os, math, hashlib, pdb, time
import pandas as pd
import numpy as np
from lumina.genetic.util import create_id
from lib.logger import logger
from dotenv import load_dotenv
load_dotenv()

from kdutils.macro2 import *
from kdutils.common import fetch_temp_data, fetch_temp_returns
from lib.optim002.optimizer import FactorsOptimizer
from kdutils.tactix import Tactix
from lib.logger import logger

def save_result(dirs, factor_name, session, data):
    def create_params(params):
        m = hashlib.md5()
        # params可能是字典类型，需要转换为字符串
        if isinstance(params, dict):
            # 将字典按键排序后转换为字符串，确保相同参数组合产生相同hash
            params_str = str(sorted(params.items()))
        else:
            params_str = str(params)
        m.update(bytes(params_str, encoding='UTF-8'))
        return create_id(original=m.hexdigest(), digit=16)
    data2 = data
    data['session'] = session
    data['param_id'] = data['params'].apply(lambda x: create_params(x))
    filename = os.path.join(dirs, "{0}.feather".format(factor_name))
    if os.path.exists(filename):
        data1 = pd.read_feather(filename)
        data = pd.concat([data,data1],axis=0)
    ## 去重
    data = data.drop_duplicates(subset=['factor_name','param_id'])
    data.reset_index(drop=True).to_feather(filename)
    logger.set_log_file(os.path.join(dirs, "{0}_{1}.log".format(factor_name, int(time.time()))))
    logger.table(data=data2, title="{}".format(factor_name))



def train(method, instruments, period, session, task_id):
    dethod = 'ic'
    n_jobs = 4
    n_trials = 6000
    top_n = 150

    dirs = os.path.join(base_path, method, instruments, "lumina", dethod,
                        str(task_id), "nxt1_ret_{}h".format(period))

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    optimize_rule = {
        'ic_mean': 'maximize',
        'sharpe2': 'maximize',
        'profit_ratio': 'maximize'
    }

    total_factors = fetch_temp_data(method=method,
                                    task_id=task_id,
                                    instruments=instruments,
                                    datasets=['train', 'val'])

    total_returns = fetch_temp_returns(method=method,
                                       instruments=instruments,
                                       datasets=['train', 'val'],
                                       category='returns')
    market_data = total_factors.set_index(['trade_time','code']).unstack()
    returns_data = total_returns[['trade_time','code', 'nxt1_ret_{}h'.format(period)]]
    fo = FactorsOptimizer(impulse_version='i017', n_jobs=n_jobs, 
        evaluator_params={'scale_method':'roll_zscore','roll_win':period,
                            'resampling_win':period}, 
                            param_ranges={
                                'window': {'min': 3, 'max': 480, 'step': 3},
                                'weriod': {'min': 3, 'max': 480, 'step': 3}
                            })
    results = fo.optimize_parallel(factor_names=['kx001','kx002','kx003','kx004','kx005',
                                                    'kx006','kx007','kx008'],
                    market_data=market_data,
                    returns_data=returns_data,
                    optimize_rule=optimize_rule,
                    period=period, 
                    n_trials=n_trials,
                    top_n=top_n)
    grouped = results.groupby('factor_name')
    for k, v in grouped:
        save_result(dirs=dirs, factor_name=k, session=session, data=v)

if __name__ == '__main__':
    variant = Tactix().start()
    train(method=variant.method,
          instruments=variant.instruments,
          period=variant.period,
          task_id=variant.task_id,
          session=variant.session)