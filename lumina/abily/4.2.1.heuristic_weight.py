import os, pdb, sys, json, math, empyrical
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


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


def merge_positions(positions_res, mode):
    res = []
    for name in positions_res:
        #print(name)
        positions = positions_res[name][mode]
        positions = positions.rename(columns={'pos': name})
        res.append(positions.set_index('trade_time'))
    positions = pd.concat(res, axis=1).reset_index()
    return positions


## 加载总览fitness
def load_fitness(base_dirs):
    fitness_file = os.path.join(base_dirs, "fitness.feather")
    fitness_pd = pd.read_feather(fitness_file)

    return fitness_pd


def load_data(mode='train'):
    filename = os.path.join(base_path, method, g_instruments, 'level2',
                            '{0}_data.feather'.format(mode))
    factors_data = pd.read_feather(filename).sort_values(
        by=['trade_time', 'code'])
    return factors_data.set_index('trade_time')


def equal_weight_synthesis(positions: pd.DataFrame):
    # 求和，得到每个时间点的净信号强度
    net_positions = positions.sum(axis=1)
    # 归一化：将信号强度缩放到[-1, 1]区间
    # 除以子策略的总数，就得到了平均信号强度。
    # 将结果控制在[-1, 1]内
    meta_positions = net_positions / len(positions.columns)
    meta_positions.name = 'equal_weight'
    return meta_positions


def fitness_weight_synthesis(positions: pd.DataFrame, programs: pd.DataFrame,
                             fitness_name: str):
    programs = programs.set_index('name')
    weights = programs.loc[positions.columns, fitness_name]
    ## 处理负数情况(正常情况不会被选中)
    weights[weights < 0] = 0

    total_weight = weights.sum()
    normalized_weights = weights / total_weight
    meta_positions = positions.mul(normalized_weights,
                                   axis='columns').sum(axis=1)
    meta_positions.name = f'{fitness_name}_weight'
    return meta_positions


def volatility_weight_synthesis(positions: pd.DataFrame):
    # 1. 计算每个策略信号的波动率（日均换手强度）
    # diff()计算每日信号变化，abs()取绝对值，mean()求平均
    volatilities = positions.diff().abs().mean()

    # 2. 计算倒数权重
    # +1e-8是为了防止除以零（对于恒定信号）
    inverse_volatilities = 1 / (volatilities + 1e-8)

    # 3. 归一化权重
    total_inverse_vol = inverse_volatilities.sum()
    normalized_weights = inverse_volatilities / total_inverse_vol

    meta_positions = positions.mul(normalized_weights,
                                   axis='columns').sum(axis=1)

    meta_positions.name = 'vol_inv_weight'
    return meta_positions


def calcute_fitness(positions,
                    total_data,
                    strategy_settings,
                    base_dirs,
                    key=None):
    name = positions.name
    positions.name = 'pos'
    positions = positions.reset_index()
    positions['code'] = INSTRUMENTS_CODES[g_instruments]
    positions = positions.set_index(['trade_time', 'code']).unstack()
    pnl_in_window = calculate_ful_ts_ret(
        pos_data=positions,
        total_data=total_data,
        strategy_settings=strategy_settings,
        agg=True  # 确保按天聚合
    )
    dirs = os.path.join(os.path.join(base_dirs, 'returns', key)) if isinstance(
        key, str) else os.path.join(os.path.join(base_dirs, 'returns', key))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pnl_in_window.reset_index().to_feather(
        os.path.join(dirs, "{0}.feather".format(name)))


if __name__ == '__main__':
    method = 'aicso2'
    g_instruments = 'ims'
    task_id = '200037'
    threshold = 1.1

    base_dirs = os.path.join(
        os.path.join('temp', "{}".format(method),
                     PERFORMANCE_MAPPING[str(task_id)],
                     INSTRUMENTS_CODES[g_instruments]))

    ## 选中的策略
    benchmark = [
        'ultron_1751492470196206', 'ultron_1751388038959442',
        'ultron_1751431447109266'
    ]
    strategy_pool = {
        'bk':
        benchmark,
        'tst1':
        benchmark + ['ultron_1751397805025247', 'ultron_1751431447109266'],
        'tst2': [
            'ultron_1751401005542132', 'ultron_1751414027498169',
            'ultron_1751386610921461', 'ultron_1751388038959442',
            'ultron_1751397805025247', 'ultron_1751431447109266',
            'ultron_1751375993205158', 'ultron_1751460301087405',
            'ultron_1751492470196206', 'ultron_1751385107413126'
        ]
    }
    key = 'tst2'

    names = list(set(strategy_pool[key]))

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[g_instruments]] * 0.05,
        'slippage': 0,  #SLIPPAGE_MAPPING[INSTRUMENTS_CODES[g_instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[g_instruments]]
    }
    pdb.set_trace()
    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    programs = load_fitness(base_dirs=base_dirs)
    ## 加载仓位
    positions_res = load_positions(base_dirs=base_dirs, names=names)

    test_positions = merge_positions(positions_res=positions_res, mode='test')
    val_positions = merge_positions(positions_res=positions_res, mode='val')
    train_positions = merge_positions(positions_res=positions_res,
                                      mode='train')

    positions = pd.concat([train_positions, val_positions, test_positions],
                          axis=0).sort_values(by=['trade_time'])
    positions = positions.set_index('trade_time')

    val_data = load_data(mode='val')
    train_data = load_data(mode='train')
    test_data = load_data(mode='test')
    total_data = pd.concat([train_data, val_data, test_data],
                           axis=0).sort_values(by=['trade_time'])

    total_data = total_data.copy().reset_index().set_index(
        ['trade_time', 'code']).unstack()

    ## 等权合成
    equal_positions = equal_weight_synthesis(positions=positions)
    weight_positions = fitness_weight_synthesis(positions=positions,
                                                programs=programs,
                                                fitness_name='train_fitness')

    volatility_positions = volatility_weight_synthesis(positions=positions)

    ## 绩效计算
    calcute_fitness(positions=equal_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)

    calcute_fitness(positions=weight_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)

    calcute_fitness(positions=volatility_positions,
                    total_data=total_data,
                    strategy_settings=strategy_settings,
                    base_dirs=base_dirs,
                    key=key)
