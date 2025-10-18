import os, pdb, copy, time, math
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kichaos.envs.trader.cn_futures.hedge061 import Hedge061TraderEnv as HedgeTraderEnv
from kichaos.agent.tessla.tessla0002 import Tessla0002 as Tessla
from kichaos.utils.env import *
from kdutils.macro2 import *
from kdutils.tactix import Tactix


def load_datasets(variant, start_pos, max_pos):
    pdb.set_trace()
    dirs = os.path.join(
        base_path, variant.method, variant.instruments, 'normal', 'rollings',
        'normal_factors3', "{0}".format(variant.horizon),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant.freq),
                                     str(variant.train_days),
                                     str(variant.val_days), str(variant.nc),
                                     str(variant.swindow)))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(start_pos, start_pos + max_pos):
        train_filename = os.path.join(
            dirs, "normal_factors_train_{0}.feather".format(i))
        val_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        test_filename = os.path.join(
            dirs, "normal_factors_test_{0}.feather".format(i))

        train_data = pd.read_feather(train_filename)
        val_data = pd.read_feather(val_filename)
        test_data = pd.read_feather(test_filename)

        train_data = train_data.rename(
            columns={"nxt1_ret_{0}h".format(variant.horizon): "nxt1_ret"})
        val_data = val_data.rename(
            columns={"nxt1_ret_{0}h".format(variant.horizon): "nxt1_ret"})
        test_data = test_data.rename(
            columns={"nxt1_ret_{0}h".format(variant.horizon): "nxt1_ret"})
        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data, test_data)
    return data_mapping


def fit(index, train_data, val_data, variant):
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] +
        ['price', 'nxt1_ret_{0}h'.format(variant.horizon)]
    ]

    ## 标的维度
    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension
    
    ## 开仓手续费
    buy_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[variant.instruments]]
    ## 平仓手续费
    sell_cost_pct = COST_MAPPING[INSTRUMENTS_CODES[variant.instruments]]

    ## 手续费字典
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(vars(variant))
    params['verbosity'] = 40
    params['code'] = INSTRUMENTS_CODES[variant.instruments]
    params['direction'] = 1 if params['direction'] == 'long' else -1
    params['step_len'] = train_data.shape[0] - 1
    params['close_times'] = CLOSE_TIME_MAPPING[params['code']]
    initial_amount = INIT_CASH_MAPPING[params['code']]  #2000000.0  #60000

    if params['check_freq'] == 0:
        params['check_freq'] = params['step_len']  # 训练完一个周期评估一次

    total_timesteps = math.ceil(
        params['step_len'] * params['epchos'] / 10000) * 10000
    pdb.set_trace()
    tessla = Tessla(code=params['code'],
                    env_class=HedgeTraderEnv,
                    features=features,
                    state_space=state_space,
                    buy_cost_pct=buy_cost_pct_sets,
                    sell_cost_pct=sell_cost_pct_sets,
                    ticker_dim=ticker_dimension,
                    initial_amount=initial_amount,
                    direction=params['direction'],
                    cont_multnum=CONT_MULTNUM_MAPPING[params['code']],
                    action_dim=2,
                    log_dir=g_log_path)
    tessla.train(name="{0}c_{1}".format(params['config_id'], index),
                 train_data=train_data,
                 val_data=val_data,
                 tensorboard_path=g_tensorboard_path,
                 total_timesteps=total_timesteps,
                 train_path=g_train_path,
                 **params)


def train(variant, start_pos, max_pos):
    data_mapping = load_datasets(variant=variant,
                                 start_pos=start_pos,
                                 max_pos=max_pos)
    for i in range(start_pos, start_pos + max_pos):
        train_data, val_data, _ = data_mapping[i]
        fit(index=i, train_data=train_data, val_data=val_data, variant=variant)


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'train':
        train(variant=variant, start_pos=48, max_pos=1)
    elif variant['methois'] == 'evaluate':
        evaluate(variant=variant)
    elif variant['methois'] == 'tbdex':
        tensorbdex(variant=variant)
