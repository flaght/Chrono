import argparse, os, pdb, copy, time, math
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from kichaos.envs.trader.cn_futures.hedge041 import Hedge041TraderEnv as HedgeTraderEnv
from kichaos.agent.tessla.tessla0001 import Tessla0001 as Tessla
from kdutils.macro import base_path
from kdutils.macro import *
from kichaos.utils.env import *


def load_datasets(variant):
    dirs = os.path.join(
        base_path, variant['method'], 'normal', variant['g_instruments'],
        'rolling', 'normal_factors3', "{0}_{1}".format(variant['categories'],
                                                       variant['horizon']),
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


def fit(index, train_data, val_data, variant):
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'close']
    ]
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
        params['check_freq'] = params['step_len'] # 训练完一个周期评估一次

    total_timesteps = math.ceil(
        params['step_len'] * params['epchos'] / 10000) * 10000

    pdb.set_trace()
    tessla = Tessla(
        code=variant['code'],
        env_class=HedgeTraderEnv,
        features=features,
        state_space=state_space,
        buy_cost_pct=buy_cost_pct_sets,
        sell_cost_pct=sell_cost_pct_sets,
        ticker_dim=ticker_dimension,
        initial_amount=initial_amount,
        cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
        open_threshold=THRESHOLD_MAPPING[variant['code']]['long_open'],
        close_threshold=THRESHOLD_MAPPING[variant['code']]['long_close'],
        action_dim=2,
        log_dir=g_log_path)
    tessla.train(name="{0}".format(index),
                 train_data=train_data,
                 val_data=val_data,
                 g_tensorboard_path=g_tensorboard_path,
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--freq', type=int, default=5)  ## 多少个周期训练一次
    parser.add_argument('--train_days', type=int, default=60)  ## 训练天数
    parser.add_argument('--val_days', type=int, default=5)  ## 验证天数

    parser.add_argument('--method', type=str, default='aicso3')  ## 方法
    parser.add_argument('--categories', type=str, default='o2o')  ## 类别
    parser.add_argument('--horizon', type=int, default=1)  ## 预测周期

    parser.add_argument('--nc', type=int, default=1)  ## 标准方式
    parser.add_argument('--swindow', type=int, default=0)  ## 滚动窗口

    parser.add_argument('--check_freq', type=int, default=0)  ## 每次模型保存的频率
    parser.add_argument('--batch_size', type=int, default=512)  ## 训练时数据大小
    parser.add_argument('-learning_starts', type=int,
                        default=1000)  ## 学习开始时间 目前该参数放入到tessla内部
    parser.add_argument('--epchos', type=int, default=10)  ## 迭代多少轮
    parser.add_argument('--window', type=int, default=3)  ## 开始周期，即多少个周期构建env数据

    parser.add_argument('--direction', type=str, default='long')  ## 方向
    parser.add_argument('--code', type=str, default='RB')  ## 代码
    parser.add_argument('--g_instruments', type=str, default='rbb')  ## 标的
    parser.add_argument('--param_index', type=int, default=1)  ## 参数索引

    parser.add_argument('--verbosity', type=int, default=40)  ## 训练时的输出频率
    parser.add_argument('--g_start_pos', type=int, default=47)
    parser.add_argument('--g_max_pos', type=int, default=1)

    args = parser.parse_args()

    variant = vars(args)
    train(variant=variant)
