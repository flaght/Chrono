### 指定策略名称+合成方法生成最终信号，用于信号端使用

### 加载指定表达式合成

import os, pdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
## 批量计算因子
from lumina.formual.iactuator import Iactuator
from lumina.genetic.fusion import Actuator
from lumina.genetic.fusion.macro import StrategyTuple
from ultron.factor.genetic.geneticist.operators import *


def remove_none_from_dict(d):
    """如果输入是字典，则移除值为None的键值对"""
    if not isinstance(d, dict):
        return d  # 如果不是字典，原样返回
    return {key: value for key, value in d.items() if value is not None}


### 提取指定子策略的信息
def fetch_strategy1(task_id, **kwargs):
    dirs = os.path.join(
        os.path.join('temp', "{}".format(kwargs['method']), str(task_id),
                     'evolution'))

    positions_file = os.path.join(dirs, f'programs_{task_id}.feather')
    strategy_dt = pd.read_feather(positions_file)

    strategy_dt['strategy_params'] = strategy_dt['strategy_params'].apply(
        remove_none_from_dict)
    strategy_dt['signal_params'] = strategy_dt['signal_params'].apply(
        remove_none_from_dict)
    ## 参数修复
    strategy_dt['strategy_params'] = strategy_dt['strategy_params'].apply(
        lambda d: {
            **d, 'max_volume': 1
        } if isinstance(d, dict) else d)
    pdb.set_trace()
    strategy_dt = strategy_dt[strategy_dt.name.isin(
        kwargs['select_strategy_ids'])].reset_index(drop=True)
    return [
        StrategyTuple(name=dt.name,
                      formual=dt.formual,
                      signal_method=dt.signal_method,
                      signal_params=dt.signal_params,
                      strategy_method=dt.strategy_method,
                      strategy_params=dt.strategy_params,
                      fitness=dt.final_fitness)
        for dt in strategy_dt.itertuples()
    ]


## 批量计算基础因子
def calculate_factors(res, n_job):
    iactuator = Iactuator(k_split=n_job)
    impluse_data = iactuator.calculate(total_data=res)
    return impluse_data


## 批量计算信号
def calculate_signal(total_data, strategy_dt, n_job):
    actuator = Actuator(k_split=1)
    strategies_data = actuator.calculate(strategies_infos=strategy_dt,
                       total_data=total_data.reset_index())

    positions_data = actuator.fitness_signal(
            strategies_infos=strategy_dt, strategies_data=strategies_data,
            method='volatility')
    return positions_data


def load_file_data(base_path, method, instruments):
    filename = os.path.join(base_path, method, instruments, 'basic',
                            "train_data.feather")
    train_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])

    filename = os.path.join(base_path, method, instruments, 'basic',
                            "val_data.feather")
    val_data = pd.read_feather(filename).sort_values(by=['trade_time', 'code'])

    filename = os.path.join(base_path, method, instruments, 'basic',
                            "test_data.feather")
    test_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time', 'code'])
    total_data['trade_time'] = pd.to_datetime(total_data['trade_time'])
    total_data = total_data[[
        'trade_time', 'code', 'close', 'high', 'low', 'open', 'value',
        'volume', 'openint', 'vwap'
    ]]
    return total_data


if __name__ == '__main__':
    method = 'aicso0'
    instruments = 'ims'
    task_id = '200036'

    select_strategy_ids = [
        "ultron_1752352016796880", "ultron_1753594132891715",
        "ultron_1752346266304364", "ultron_1752145706357350",
        "ultron_1752385350601006"
    ]

    strategy_dt = fetch_strategy1(task_id=task_id,
                                  method=method,
                                  instruments=instruments,
                                  select_strategy_ids=select_strategy_ids)
    ### 加载数据
    total_data = load_file_data(base_path,
                                method=method,
                                instruments=instruments)
    total_data = total_data.reset_index(drop=True).loc[:60]
    total_data1 = total_data.set_index(['trade_time', 'code']).unstack()
    total_data2 = total_data.set_index(['trade_time', 'code'])
    impluse_data = calculate_factors(res=total_data1, n_job=16)
    pdb.set_trace()
    calculate_signal(total_data=pd.concat([impluse_data, total_data2], axis=1),
                     strategy_dt=strategy_dt,
                     n_job=4)
