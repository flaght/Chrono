import os, pdb, copy, time, math
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from kichaos.envs.trader.cn_futures.hedge061 import Hedge061TraderEnv as HedgeTraderEnv
#from kichaos.envs.trader.cn_futures.hedge062 import Hedge062TraderEnv as  HedgeTraderEnv
from kichaos.agent.tessla.tessla0002 import Tessla0002 as Tessla
#from kichaos.agent.tessla.tessla0003 import Tessla0003 as Tessla
from kichaos.utils.env import *
from kdutils.tactix import Tactix
from kdutils.macro import *

from temp import create_data

### 特征池， 用于尝试不同特征

features_pool = {
    0: [],
    1: [
        'ixy006_5_10_1', 'tv017_5_10_1', 'tn004_10_15_1', 'tn003_5_5_10_15_1',
        'tc007_5_10_1', 'oi031_5_10_1', 'rv006_10_15_1_2', 'rv005_10_15_1_2',
        'rv011_75_5_10_1'
    ],
    2: [
        'fz002_5_10_1', 'ixy006_5_10_1', 'oi037_5_10_1', 'tv005_5_10_1',
        'rv005_5_10_1_1', 'rv005_10_15_1_2', 'db005_5_10_1', 'rv006_10_15_1_2',
        'tn003_5_5_10_15_1', 'ixy015_5_10_1', 'dv002_5_10_0', 'tc003_5_10_1'
    ]
}


def load_datasets(variant):
    dirs = os.path.join(
        base_path, variant['method'], 'normal', variant['g_instruments'],
        'rollings', 'normal_factors3',
        "{0}_{1}".format(variant['categories'], variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['swindow'])))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_filename = os.path.join(
            dirs, "normal_factors_train_{0}.feather".format(i))
        val_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        test_filename = os.path.join(
            dirs, "normal_factors_test_{0}.feather".format(i))

        train_data = pd.read_feather(train_filename)
        val_data = pd.read_feather(val_filename)
        test_data = pd.read_feather(test_filename)

        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data, test_data)
    return data_mapping


def fetch_features(columns, fid):
    selected_factors = features_pool[fid]
    selected_flag = 1 if len(selected_factors) > 0 else 0
    features = [
        col for col in columns
        if col not in ['trade_time', 'code'] + ['price', 'nxt1_ret']
    ] if not selected_flag else selected_factors
    return features


def tensorbdex(variant):
    data_mapping = load_datasets(variant)
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        _, _, test_data = data_mapping[i]
        tbdex(index=i, test_data=test_data, variant=variant)


def tbdex(index, test_data, variant):
    features = fetch_features(columns=test_data.columns,
                              fid=variant['feature_id'])

    ## 标的维度
    ticker_dimension = len(test_data.code.unique())
    state_space = ticker_dimension

    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[variant['code']]['buy']
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']  #0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(test_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(test_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    params['verbosity'] = 40
    del params['direction']
    params['direction'] = 1 if variant['direction'] == 'long' else -1
    params['step_len'] = test_data.shape[0] - 1

    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000

    tessla = Tessla(code=variant['code'],
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=variant['direction'],
                    cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
                    action_dim=2,
                    log_dir=g_log_path)

    name = "{0}c_{1}_{2}".format(variant['config_id'], Tessla.__name__, index)
    tessla.tbdex(tb_path=g_tensorboard_path,
                 out_path=g_tbdex_path,
                 tb_index=3,
                 name=name,
                 **params)


def evaluate(variant):
    data_mapping = load_datasets(variant)
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        _, _, test_data = data_mapping[i]
        educate(index=i,
                test_data=test_data,
                model_index='best_model',
                variant=variant)


def educate(index, test_data, model_index, variant):
    features = fetch_features(columns=test_data.columns,
                              fid=variant['feature_id'])
    ## 标的维度
    ticker_dimension = len(test_data.code.unique())
    state_space = ticker_dimension

    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[variant['code']]['buy']
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']  #0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(test_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(test_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    params['verbosity'] = 40
    del params['direction']
    params['direction'] = 1 if variant['direction'] == 'long' else -1
    params['step_len'] = test_data.shape[0] - 1

    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000

    tessla = Tessla(code=variant['code'],
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=variant['direction'],
                    cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
                    action_dim=2,
                    log_dir=g_log_path)
    name = "{0}c_{1}_{2}".format(variant['config_id'], Tessla.__name__, index)
    test_memory = tessla.evaluate(name=name,
                                  test_data=test_data,
                                  train_path=g_train_path,
                                  **params)
    tessla.illustrate(name=name,
                      memory_data=test_memory[0],
                      kl_pd=test_data,
                      illustrate_path=g_illustrate_path,
                      price_name='price',
                      trader_time=TRADE_TIME_MAPPING[variant['code']],
                      **params)


def fit(index, train_data, val_data, variant):
    features = fetch_features(columns=train_data.columns,
                              fid=variant['feature_id'])
    pdb.set_trace()
    ## 标的维度
    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension

    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[variant['code']]['buy']
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']  #0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    params['verbosity'] = 40
    del params['direction']
    params['direction'] = 1 if variant['direction'] == 'long' else -1
    params['step_len'] = train_data.shape[0] - 1

    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000

    if params['check_freq'] == 0:
        params['check_freq'] = params['step_len']  # 训练完一个周期评估一次

    total_timesteps = math.ceil(
        params['step_len'] * params['epchos'] / 10000) * 10000
    pdb.set_trace()
    tessla = Tessla(code=variant['code'],
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=variant['direction'],
                    cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
                    action_dim=2,
                    log_dir=g_log_path)

    tessla.train(name="{0}c_{1}_{2}".format(variant['config_id'],
                                            Tessla.__name__, index),
                 train_data=train_data,
                 val_data=val_data,
                 tensorboard_path=g_tensorboard_path,
                 total_timesteps=total_timesteps,
                 train_path=g_train_path,
                 **params)


def train(variant):
    data_mapping = load_datasets(variant)
    for i in range(variant['g_start_pos'],
                   variant['g_start_pos'] + variant['g_max_pos']):
        train_data, val_data, _ = data_mapping[i]
        fit(index=i, train_data=train_data, val_data=val_data, variant=variant)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant['methois'] == 'train':
        train(variant=variant)
    elif variant['methois'] == 'eval':
        evaluate(variant=variant)
    elif variant['methois'] == 'tbdex':
        tensorbdex(variant=variant)
