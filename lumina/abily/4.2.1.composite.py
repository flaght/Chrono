import os, pdb, sys, json, math, empyrical
import pandas as pd
from pandas import Timestamp
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.common import *
from kdutils.composite.prepare import *
from kdutils.composite.heurisitc import *
from kdutils.composite.model import *
from kdutils.composite.optn import *
from kdutils.composite.models.random_forest import RandomForestClassifier
from kdutils.composite.models.lightBGM import LightBGMClassifier
from kdutils.composite.models.bayes import NativeBayesClassifier

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

    total_data, train_data, val_data, test_data = fetch_all_data(
        method=method, instruments=instruments, task_id=task_id)

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

        positions, train_positions, val_positions, test_positions = process_positions(
            positions_res=positions_res, key=key)

        synthesized_positions_list = []

        lg1 = LightBGMClassifier(instruments=instruments,
                                 names=names,
                                 strategy_settings=strategy_settings,
                                 key=None)
        lg1.run(train_data=train_data,
                val_data=val_data,
                train_positions=train_positions,
                val_positions=val_positions,
                test_positions=test_positions)
        
        
        nb1 = NativeBayesClassifier(instruments=instruments,
                                    names=names,
                                    strategy_settings=strategy_settings,
                                    key=None)
        nb1.run(train_data=train_data,
                val_data=val_data,
                train_positions=train_positions,
                val_positions=val_positions,
                test_positions=test_positions)

        rf1 = RandomForestClassifier(instruments=instruments,
                                     names=names,
                                     strategy_settings=strategy_settings,
                                     key=None)
        rf1.run(train_data=train_data,
                val_data=val_data,
                train_positions=train_positions,
                val_positions=val_positions,
                test_positions=test_positions)

        pdb.set_trace()
        #f1 = lbgm_classifer_cv(train_data=train_data,
        #                       val_data=val_data,
        #                       test_data=test_data,
        #                       train_positions=train_positions,
        #                       val_positions=val_positions,
        #                       test_positions=test_positions,
        #                       instruments=instruments,
        #                       names=names,
        #                       strategy_settings=strategy_settings)
        print(f"策略池 '{key}' 合成完毕，开始循环处理所有合成策略...")
        for synth_pos in synthesized_positions_list:
            print(f"  分割存储仓位: {synth_pos.name}...")
            split_positions(base_dirs=base_dirs,
                            positions=synth_pos,
                            time_periods=time_periods,
                            key=key)
            print(f"计算绩效: {synth_pos.name}...")
            calcute_fitness(positions=synth_pos,
                            instruments=instruments,
                            total_data=total_data,
                            strategy_settings=strategy_settings,
                            base_dirs=base_dirs,
                            key=key)
