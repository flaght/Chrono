import os, pdb, sys, json, math, empyrical
import pandas as pd
from pandas import Timestamp
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.common import *
from kdutils.composite.prepare import *
from lumina.genetic.metrics.ts_pnl import calculate_ful_ts_ret


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


if __name__ == '__main__':
    method = 'aicso0'
    instruments = 'ims'
    task_id = '200037'
    config_file = os.path.join('config', 'heuristic.toml')
    strategy_pool = fetch_experiment(config_file=config_file,
                                     method=method,
                                     task_id=task_id)
    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    strategy_settings = {
        'commission': COST_MAPPING[INSTRUMENTS_CODES[instruments]],
        'slippage': SLIPPAGE_MAPPING[INSTRUMENTS_CODES[instruments]],
        'size': CONT_MULTNUM_MAPPING[INSTRUMENTS_CODES[instruments]]
    }
    base_dirs = os.path.join(os.path.join('temp', "{}".format(method),
                                          task_id))
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    programs = load_fitness(base_dirs=base_dirs)

    all_possible_names = list(
        set([strat for pool in strategy_pool.values() for strat in pool]))
    all_positions_res = load_positions(base_dirs=base_dirs,
                                       names=all_possible_names)

    total_data = fetch_all_data(method=method,
                                instruments=instruments,
                                task_id=task_id)

    time_periods = fetch_times(method=method,
                               task_id=task_id,
                               instruments=instruments)

    for key, names in strategy_pool.items():
        print(f"\n{'='*20} 开始处理策略池: {key} {'='*20}")
        names = list(set(names))
        print(f"包含 {len(names)} 个独立策略。")
        positions_res = {
            k: v
            for k, v in all_positions_res.items() if k in names
        }

        positions = process_positions(positions_res=positions_res, key=key)

        synthesized_positions_list = [
            equal_weight_synthesis(positions=positions),
            fitness_weight_synthesis(positions=positions,
                                     programs=programs,
                                     fitness_name='train_fitness'),
            volatility_weight_synthesis(positions=positions)
        ]
        print(f"策略池 '{key}' 合成完毕，开始循环处理所有合成策略...")
        for synth_pos in synthesized_positions_list:
            print(f"  分割存储仓位: {synth_pos.name}...")
            split_positions(base_dirs=base_dirs,
                            positions=synth_pos,
                            time_periods=time_periods,
                            key=key)
            print(f"  计算绩效: {synth_pos.name}...")
            calcute_fitness(positions=synth_pos,
                            instruments=instruments,
                            total_data=total_data,
                            strategy_settings=strategy_settings,
                            base_dirs=base_dirs,
                            key=key)
