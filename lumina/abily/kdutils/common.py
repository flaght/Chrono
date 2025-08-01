import os, pdb, sys, json, math, toml
import pandas as pd

from kdutils.macro2 import *


### 读取数据 计算训练集，校验集，测试集，总数集的绩效
def fetch_temp_data(method, instruments, task_id, datasets):

    res = []

    def fet(name):
        #filename = os.path.join(base_path, method, instruments, 'level2',
        #                        "{0}_data.feather".format(name))
        filename = os.path.join(
            base_path, method, instruments, DATAKIND_MAPPING[str(
                INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])],
            "{0}_data.feather".format(name))
        factors_data = pd.read_feather(filename).sort_values(
            by=['trade_time', 'code'])
        factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
        return factors_data

    for n in datasets:
        dt = fet(n)
        res.append(dt)

    res = pd.concat(res, axis=0)
    factors_data = res.sort_values(by=['trade_time', 'code'])
    factors_data['trade_time'] = pd.to_datetime(factors_data['trade_time'])
    factors_data = factors_data.sort_values(by=['trade_time', 'code'])
    return factors_data


### 读取 训练集 校验集，测试集的时间范围
def fetch_times(method, task_id, instruments):
    train_data = fetch_temp_data(method=method,
                                 task_id=task_id,
                                 instruments=instruments,
                                 datasets=['train'])
    val_data = fetch_temp_data(method=method,
                               task_id=task_id,
                               instruments=instruments,
                               datasets=['val'])
    test_data = fetch_temp_data(method=method,
                                task_id=task_id,
                                instruments=instruments,
                                datasets=['test'])
    return {
        'train_time':
        (train_data['trade_time'].min(), train_data['trade_time'].max()),
        'val_time':
        (val_data['trade_time'].min(), val_data['trade_time'].max()),
        'test_time':
        (test_data['trade_time'].min(), test_data['trade_time'].max())
    }


def remove_none_from_dict(d):
    """如果输入是字典，则移除值为None的键值对"""
    if not isinstance(d, dict):
        return d  # 如果不是字典，原样返回
    return {key: value for key, value in d.items() if value is not None}


### 根据阈值条件读取 策略评估信息
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
    ## 清理冗余参数
    strategy_dt = strategy_dt[strategy_dt.final_fitness >= kwargs['threshold']]
    return strategy_dt


## 加载总览fitness
def load_fitness(base_dirs):
    fitness_file = os.path.join(base_dirs, "fitness.feather")
    fitness_pd = pd.read_feather(fitness_file)

    return fitness_pd


## 加载不同时段的仓位数据
def load_positions(base_dirs, names):
    dirs = os.path.join(os.path.join(base_dirs, 'positions'))
    positions_res = {}
    for name in names:
        train_positions = pd.read_feather(
            os.path.join(dirs, "{0}_train.feather".format(name)))

        val_positions = pd.read_feather(
            os.path.join(dirs, "{0}_val.feather".format(name)))

        test_positions = pd.read_feather(
            os.path.join(dirs, "{0}_test.feather".format(name)))
        positions_res[name] = {
            'train': train_positions,
            'val': val_positions,
            'test': test_positions
        }
    return positions_res


## 加载不同时段的基础数据
def load_data(instruments, method, task_id, mode='train'):
    filename = os.path.join(base_path, method, instruments,
                            DATAKIND_MAPPING[task_id],
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


### 合并仓位数据
def merge_positions(positions_res, mode):
    res = []
    for name in positions_res:
        print(name)
        positions = positions_res[name][mode]
        positions = positions.rename(columns={'pos': name})
        res.append(positions.set_index('trade_time'))
    positions = pd.concat(res, axis=1).reset_index()
    return positions


## 加载策略配置文件
def fetch_experiment(config_file, method, task_id):
    strategy_pool = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        all_configs = toml.load(f)
    config = all_configs[method][task_id]
    benchmark = config['benchmark']
    alone_pools = config.get('alone', {})
    additive_pools = config.get('additive', {})
    active_pools = config['active_pools']
    for key in active_pools:
        if key in alone_pools:
            strategy_pool[key] = alone_pools[key]
        elif key in additive_pools:
            # 如果在 additive_pools 中找到，与 benchmark 合并
            strategy_pool[key] = benchmark + additive_pools[key]
            print(f"信息: 构建附加策略池 '{key}' (基于 benchmark)。")
        else:
            # 如果哪里都找不到，给出警告
            print(
                f"警告: 'active_pools' 中指定的池 '{key}' 在 'standalone_pools' 或 'additive_pools' 中均未定义，将被忽略。"
            )
    pdb.set_trace()
    return strategy_pool
