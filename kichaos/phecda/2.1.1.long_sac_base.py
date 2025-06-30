import argparse, os, pdb, copy, time, math
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'rbb'
g_instruments = os.environ['INSTRUMENTS']
g_start_pos = 47  #44
g_max_pos = g_start_pos + 1

from kichaos.envs.trader.cn_futures.hedge041 import Hedge041TraderEnv as HedgeTraderEnv
from kichaos.stable3.common.vec_env import VecMonitor
from kichaos.rl.agent import Agent
from kichaos.utils.env import *

from kdutils.macro import base_path, codes
from kdutils.macro import *


def load_datasets(variant):
    dirs = os.path.join(
        base_path, variant['method'], 'normal', g_instruments, 'rolling',
        'normal_factors3', "{0}_{1}".format(variant['categories'],
                                            variant['horizon']),
        "{0}_{1}_{2}_{3}_{4}".format(str(variant['freq']),
                                     str(variant['train_days']),
                                     str(variant['val_days']),
                                     str(variant['nc']),
                                     str(variant['swindow'])))
    data_mapping = {}
    min_date = None
    max_date = None
    for i in range(g_start_pos, g_max_pos):
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

    fix_seed = 42
    MODEL_PARAMS = {
        'learning_rate': lambda f: 1e-4 * f,
        'buffer_size': 100000,
        'ent_coef': "auto_0.05",
        'target_entropy': 'auto',
        'tau': 0.08,
        'batch_size': 512,
        'gamma': 0.85,
        "target_update_interval": 2
    }
    '''
    fix_seed = 42
    POLICY_PARAMS = {
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {
            'weight_decay': 1e-5,
            'eps': 1e-5
        }
    }
    MODEL_PARAMS = {
        'learning_rate': lambda f: 1e-5 * f,
        'buffer_size': 100000,
        'ent_coef': "auto_0.05",
        'target_entropy': 'auto',
        'tau': 0.05,
        'batch_size': 256,
        'gamma': 0.99,
        "target_update_interval": 2,
    }
    '''
    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension
    buy_cost_pct = COST_MAPPING[variant['code']][
        'buy']  #0.00012  #0.1 / 1000  #0.00001#0.000100#(0.0100 / 10000)
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']  #0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    params['verbosity'] = 40
    del params['direction']
    params['step_len'] = train_data.shape[0] - 1
    params['direction'] = 1 if variant['direction'] == 'long' else -1
    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000
    env_train_gym = HedgeTraderEnv(
        df=train_data,
        features=features,
        state_space=state_space,
        action_dim=2,
        buy_cost_pct=buy_cost_pct_sets,
        sell_cost_pct=sell_cost_pct_sets,
        ticker_dim=ticker_dimension,
        mode='train',
        cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
        initial_amount=initial_amount,
        open_threshold=THRESHOLD_MAPPING[variant['code']]['long_open'],
        close_threshold=THRESHOLD_MAPPING[variant['code']]['long_close'],
        **params)
    env_train, _ = env_train_gym.get_env()

    env_val_gym = HedgeTraderEnv(
        df=val_data,
        features=features,
        state_space=state_space,
        action_dim=2,
        buy_cost_pct=buy_cost_pct_sets,
        sell_cost_pct=sell_cost_pct_sets,
        ticker_dim=ticker_dimension,
        mode='eval',
        cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
        initial_amount=initial_amount,
        open_threshold=THRESHOLD_MAPPING[variant['code']]['long_open'],
        close_threshold=THRESHOLD_MAPPING[variant['code']]['long_close'],
        **params)
    env_val, _ = env_val_gym.get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}".format(env_train_gym.name, variant['code'],
                                    variant['direction'], index)
    env_train_sac = VecMonitor(env_train, log_dir + '_{0}_train'.format(name))
    env_eval_sac = VecMonitor(env_val, log_dir + '_{0}_eval'.format(name))

    agent = Agent(env=env_train_sac)

    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)

    start = time.time()
    total_timesteps = math.ceil(
        params['step_len'] * params['check_freq'] / 10000) * 10000  #40000
    #total_timesteps = 40000
    agent_model_name = model_name + "_{0}".format(name)
    agent.train_model(
        model=model_sac,
        tb_log_name=agent_model_name,
        check_freq=int(total_timesteps /
                       params['check_freq']),  ## 控制每次train训练的次数
        ck_path=os.path.join(g_train_path, agent_model_name),
        log_path=None,
        eval_env=env_eval_sac,
        total_timesteps=total_timesteps)
    print("Training Done time: %.3f" % (time.time() - start))



def train(variant):
    data_mapping = load_datasets(variant)
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data, test_data = data_mapping[i]
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

    parser.add_argument('--check_freq', type=int, default=15)  ## 每次模型保存的频率
    parser.add_argument('--batch_size', type=int, default=512)  ## 训练时数据大小
    parser.add_argument('--window', type=int, default=3)  ## 开始周期，即多少个周期构建env数据

    parser.add_argument('--direction', type=str, default='long')  ## 方向
    parser.add_argument('--code', type=str, default='RB')  ## 代码

    args = parser.parse_args()
    train(vars(args))
