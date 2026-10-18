import os, copy, math
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from lib.rlx001 import normal1_factors

from kichaos.envs.trader.cn_futures.hedge061 import Hedge061TraderEnv as HedgeTraderEnv
from kichaos.agent.tessla.tessla0002 import Tessla0002 as Tessla
from kichaos.utils.env import *

import torch

torch.autograd.set_detect_anomaly(True)


def load_data(method, instruments, task_id, period, window):
    dirs = os.path.join(base_path, method, instruments, 'temp', "rl", "nos",
                        str(task_id), str(period), str(window))
    train_data = pd.read_feather(
        os.path.join(dirs, "normal_train_data.feather"))
    val_data = pd.read_feather(os.path.join(dirs, "normal_val_data.feather"))
    test_data = pd.read_feather(os.path.join(dirs, "normal_test_data.feather"))
    return train_data, val_data, test_data


def fit(train_data, val_data, variant):
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'nxt1_ret']
    ]  #[0:5]
    pdb.set_trace()
    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension

    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[
        variant.instruments]]  #['buy']
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[
        variant.instruments]]  #['sell']

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = vars(copy.deepcopy(variant))
    params['verbosity'] = 40
    del params['direction']
    params['direction'] = 1 if variant.direction == 'long' else -1
    params['step_len'] = train_data.shape[0] - 1

    params['close_times'] = []
    initial_amount = 2000000.0  #60000
    pdb.set_trace()
    if params['check_freq'] == 0:
        params['check_freq'] = int(params['step_len'] / 3)  # 训练完一个周期评估一次

    total_timesteps = math.ceil(
        params['step_len'] * params['epchos'] / 10000) * 1000
    pdb.set_trace()
    tessla = Tessla(code=variant.instruments,
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=params['direction'],
                    cont_multnum=200,
                    action_dim=2,
                    log_dir=g_log_path)
    tessla.train(name="{0}c_{1}".format(variant.config_id, 0),
                 train_data=train_data,
                 val_data=val_data,
                 tensorboard_path=g_tensorboard_path,
                 total_timesteps=total_timesteps,
                 train_path=g_train_path,
                 **params)

def educate(test_data, model_index, variant):
    features = [
        col for col in test_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'nxt1_ret']
    ] 
    ticker_dimension = len(test_data.code.unique())
    state_space = ticker_dimension

    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[
        variant.instruments]]  #['buy']
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[
        variant.instruments]]  #['sell']

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(test_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(test_data.code, [sell_cost_pct] * ticker_dimension))

    pdb.set_trace()
    params = vars(copy.deepcopy(variant))
    params['verbosity'] = 40
    del params['direction']
    params['direction'] = 1 if variant.direction == 'long' else -1
    params['step_len'] = test_data.shape[0] - 1

    params['close_times'] = []
    initial_amount = 2000000.0  #2000000.0  #60000

    tessla = Tessla(code=variant.instruments,
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=params['direction'],
                    cont_multnum=200,
                    action_dim=2,
                    log_dir=g_log_path)

    test_memory = tessla.evaluate(name="{0}c_{1}".format(variant.config_id, 0),
                                  test_data=test_data,
                                  train_path=g_train_path,
                                  **params)

    tessla.illustrate(name="{0}c_{1}".format(variant.config_id, 0),
                      memory_data=test_memory[0],
                      kl_pd=test_data,
                      illustrate_path=g_illustrate_path,
                      price_name='price',
                      trader_time=('09:00:00', '15:00:00'),
                      **params)

def train(method, instruments, task_id, period, window):
    train_data, val_data, _ = load_data(method=method,
                                        instruments=instruments,
                                        task_id=task_id,
                                        period=period,
                                        window=window)
    fit(train_data=train_data, val_data=val_data, variant=variant)


def evaluate(method, instruments, task_id, period, window):
    _, _, test_data = load_data(method=method,
                                instruments=instruments,
                                task_id=task_id,
                                period=period,
                                window=window)
    educate(test_data=test_data, model_index='best_model', variant=variant)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'normal':
        normal1_factors(method=variant.method,
                        instruments=variant.instruments,
                        task_id=variant.task_id,
                        period=variant.period,
                        window=variant.window)
    elif variant.form == 'train':
        train(method=variant.method,
              instruments=variant.instruments,
              task_id=variant.task_id,
              period=variant.period,
              window=variant.window)
    elif variant.form == 'evaluate':
        evaluate(method=variant.method,
              instruments=variant.instruments,
              task_id=variant.task_id,
              period=variant.period,
              window=variant.window)
