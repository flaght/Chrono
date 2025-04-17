import sys, os, re, pdb, argparse, time, copy
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from sklearn.preprocessing import RobustScaler
from kichaos.envs.trader.cn_futures.hedge009c import Hedge009TraderEnv as HedgeTraderEnv
from kichaos.stable3.common.vec_env import VecMonitor
from kichaos.rl.agent import Agent
from kichaos.utils.env import *
from kdutils.macro import base_path, codes
from kdutils.data import fetch_main_market


os.environ['INSTRUMENTS'] = 'rbb'

g_instruments = os.environ['INSTRUMENTS']

g_start_pos = 0
g_max_pos = 1

def load_datasets1(variant):
    dirs = os.path.join(base_path, 'nicso', g_instruments, 'rolling',
                        'normal_factors3', str(variant['rootid']),
                        str(variant['freq']), str(variant['trade_dates']),
                        str(variant['val_dates']), str(variant['nc']),
                        str(variant['swindow']))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(g_start_pos, g_max_pos):
        test_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        
        test_data = pd.read_feather(test_filename)
        min_time = pd.to_datetime(test_data['trade_time']).min()
        max_time = pd.to_datetime(test_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (test_data)

    market_data = fetch_main_market(min_date.strftime('%Y-%m-%d'),
                             max_date.strftime('%Y-%m-%d'), codes)
    market_data = market_data[['trade_time', 'code',
                               'close']].rename(columns={'close': 'price'})
    for i in range(g_start_pos, g_max_pos):
        print(i)
        test_data = data_mapping[i]
        ## 去极值
        test_data = test_data.merge(market_data, on=['trade_time', 'code'])
        test_data = test_data.fillna(0)
        test_data.index = test_data['trade_time'].factorize()[0]
        features = [
            col for col in test_data.columns
            if col not in ['trade_time', 'code'] + ['price', 'close']
        ]
        test_data[features] = test_data[features].replace([np.inf, -np.inf],
                                                            np.nan)
        test_data = test_data.fillna(0)
        data_mapping[i] = (test_data)
    return data_mapping


def load_datasets(variant):
    dirs = os.path.join(base_path, 'nicso', g_instruments, 'rolling',
                        'normal_factors3', str(variant['rootid']),
                        str(variant['freq']), str(variant['trade_dates']),
                        str(variant['val_dates']), str(variant['nc']),
                        str(variant['swindow']))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(g_start_pos, g_max_pos):
        train_filename = os.path.join(
            dirs, "normal_factors_train_{0}.feather".format(i))
        val_filename = os.path.join(dirs,
                                    "normal_factors_val_{0}.feather".format(i))
        train_data = pd.read_feather(train_filename)
        val_data = pd.read_feather(val_filename)
        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data)

    market_data = fetch_main_market(min_date.strftime('%Y-%m-%d'),
                             max_date.strftime('%Y-%m-%d'), codes)
    market_data = market_data[['trade_time', 'code',
                               'close']].rename(columns={'close': 'price'})
    for i in range(g_start_pos, g_max_pos):
        print(i)
        train_data, val_data = data_mapping[i]
        ## 去极值
        pdb.set_trace()
        train_data = train_data.merge(market_data, on=['trade_time', 'code'])
        val_data = val_data.merge(market_data, on=['trade_time', 'code'])
        train_data = train_data.fillna(0)
        val_data = val_data.fillna(0)
        train_data.index = train_data['trade_time'].factorize()[0]
        val_data.index = val_data['trade_time'].factorize()[0]
        features = [
            col for col in train_data.columns
            if col not in ['trade_time', 'code'] + ['price', 'close']
        ]
        train_data[features] = train_data[features].replace([np.inf, -np.inf],
                                                            np.nan)
        val_data[features] = val_data[features].replace([np.inf, -np.inf],
                                                        np.nan)
        train_data = train_data.fillna(0)
        val_data = val_data.fillna(0)
        data_mapping[i] = (train_data, val_data)
    return data_mapping


def fit(index, train_data, val_data, variant):
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'close']
    ]
    fix_seed = 1999
    MODEL_PARAMS = {
        'learning_rate': lambda f: 1e-4 * f,
        'buffer_size': 100000,
        'ent_coef': "auto_0.01",
        'target_entropy': 'auto',
        'tau': 0.08,
        'batch_size': 256,
        'gamma': 0.75,
        "target_update_interval": 2
    }
    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension
    buy_cost_pct = 0.00012  #0.1 / 1000  #0.00001#0.000100#(0.0100 / 10000)
    sell_cost_pct = 0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    del params['direction']
    initial_amount = 60000
    env_train_gym = HedgeTraderEnv(df=train_data,
                                   features=features,
                                   state_space=state_space,
                                   action_dim=2,
                                   buy_cost_pct=buy_cost_pct_sets,
                                   sell_cost_pct=sell_cost_pct_sets,
                                   ticker_dim=ticker_dimension,
                                   direction=[variant['direction']],
                                   mode='train',
                                   cont_multnum=10,
                                   initial_amount=initial_amount,
                                   **params)
    env_train, _ = env_train_gym.get_env()

    env_val_gym = HedgeTraderEnv(df=val_data,
                                 features=features,
                                 state_space=state_space,
                                 action_dim=2,
                                 buy_cost_pct=buy_cost_pct_sets,
                                 sell_cost_pct=sell_cost_pct_sets,
                                 ticker_dim=ticker_dimension,
                                 direction=[variant['direction']],
                                 mode='eval',
                                 cont_multnum=10,
                                 initial_amount=initial_amount,
                                 **params)
    env_val, _ = env_val_gym.get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}_3".format(env_train_gym.name, variant['code'],
                                    variant['direction'], index)
    env_train_sac = VecMonitor(env_train, log_dir + '_{0}_train'.format(name))
    env_eval_sac = VecMonitor(env_val, log_dir + '_{0}_eval'.format(name))

    agent = Agent(env=env_train_sac)

    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)

    start = time.time()
    total_timesteps = 20000
    agent_model_name = model_name + "_{0}".format(name)
    pdb.set_trace()
    agent.train_model(
        model=model_sac,
        tb_log_name=agent_model_name,
        check_freq=500,  ## 控制每次train训练的次数
        ck_path=os.path.join(g_train_path, agent_model_name),
        log_path=None,
        eval_env=env_eval_sac,
        total_timesteps=total_timesteps)
    print("Training Done time: %.3f" % (time.time() - start))


def educate(index, val_data, variant):
    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'close']
    ]
    ticker_dimension = len(val_data.code.unique())
    state_space = ticker_dimension

    buy_cost_pct = 0.00012  #0.1 / 1000  #0.00001#0.000100#(0.0100 / 10000)
    sell_cost_pct = 0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    del params['direction']
    initial_amount = 60000

    env_val_gym = HedgeTraderEnv(df=val_data,
                                 features=features,
                                 state_space=state_space,
                                 action_dim=2,
                                 buy_cost_pct=buy_cost_pct_sets,
                                 sell_cost_pct=sell_cost_pct_sets,
                                 ticker_dim=ticker_dimension,
                                 direction=[variant['direction']],
                                 mode='eval',
                                 cont_multnum=10,
                                 initial_amount=initial_amount,
                                 **params)
    env_val, _ = env_val_gym.get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}".format(env_val_gym.name, variant['code'],
                                    variant['direction'], index)
    agent_model_name = model_name + "_{0}_3".format(name)

    model_path = os.path.join(g_train_path, agent_model_name, "best_model")
    pdb.set_trace()
    Agent.load_from_file(model_name=model_name,
                         environment=env_val_gym,
                         model_path=model_path,
                         log_dir=log_dir)


def train(variant):
    data_mapping = load_datasets(variant)
    pdb.set_trace()
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data = data_mapping[i]
        fit(index=i, train_data=train_data, val_data=val_data, variant=variant)


def perdict(variant):
    data_mapping = load_datasets1(variant)
    for i in range(g_start_pos, g_max_pos):
        pdb.set_trace()
        print("index===>{0}".format(i))
        test_data = data_mapping[i]
        educate(index=i, val_data=test_data, variant=variant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_len', type=int, default=4000)  ## 每次训练集要经过多少个周期
    parser.add_argument('--batch_size', type=int, default=512)  ## 训练时数据大小
    parser.add_argument('--window', type=int, default=3)  ## 开始周期，即多少个周期构建env数据
    parser.add_argument('--direction', type=str, default='long')  ## 方向
    parser.add_argument('--code', type=str, default='RB')  # 标的
    parser.add_argument('--learning_starts', type=int, default=500)  ## 学习开始时间
    parser.add_argument('--freq', type=int, default=5)  ## 多少个周期训练一次
    #parser.add_argument('--latest', type=int, default=180)  ## 过去多少个周期的数据
    parser.add_argument('--trade_dates', type=int, default=180)  ## 过去多少个周期的数据
    parser.add_argument('--val_dates', type=int, default=5)  ## 过去多少个周期的数据
    parser.add_argument('--nc', type=int, default=1)  ## 标准化模式
    parser.add_argument('--rootid', type=int, default=20241113001)  ## 特征集
    parser.add_argument('--swindow', type=int, default=0)  ## 标准化window
    args = parser.parse_args()
    train(vars(args))
    perdict(vars(args))

