import argparse, os, pdb, copy, time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'rbb'
g_instruments = os.environ['INSTRUMENTS']
g_start_pos = 49
g_max_pos = 49 + 1

from kichaos.envs.trader.cn_futures.hedge011 import Hedge011TraderEnv as HedgeTraderEnv
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
    fix_seed = 1999
    MODEL_PARAMS = {
        'learning_rate': lambda f: 1e-4 * f,
        'buffer_size': 100000,
        'ent_coef': "auto_0.01",
        'target_entropy': 'auto',
        'tau': 0.08,
        'batch_size': 256,
        'gamma': 0.65,
        "target_update_interval": 2,
    }
    #train_data = train_data.loc[5:].reset_index(drop=True)
    #train_data = train_data.drop([104,105,106,107,108,107,108,109,110,111]).reset_index(drop=True)
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
    del params['direction']

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
        direction=[variant['direction']],
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
        direction=[variant['direction']],
        mode='eval',
        cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
        initial_amount=initial_amount,
        open_threshold=THRESHOLD_MAPPING[variant['code']]['long_open'],
        close_threshold=THRESHOLD_MAPPING[variant['code']]['long_close'],
        **params)
    env_val, _ = env_val_gym.get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}_{4}".format(env_train_gym.name, variant['code'],
                                        variant['direction'], index,
                                        variant['method'])
    env_train_sac = VecMonitor(env_train, log_dir + '_{0}_train'.format(name))
    env_eval_sac = VecMonitor(env_val, log_dir + '_{0}_eval'.format(name))

    agent = Agent(env=env_train_sac)

    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)

    start = time.time()
    total_timesteps = 40000
    agent_model_name = model_name + "_{0}".format(name)
    agent.train_model(
        model=model_sac,
        tb_log_name=agent_model_name,
        check_freq=2400,  ## 控制每次train训练的次数
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


def educate(index, test_data, model_index, variant):
    features = [
        col for col in test_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'close']
    ]
    ticker_dimension = len(test_data.code.unique())
    state_space = ticker_dimension

    buy_cost_pct = COST_MAPPING[variant['code']][
        'buy']  #0.00012  #0.1 / 1000  #0.00001#0.000100#(0.0100 / 10000)
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']  #0.00012  #0.1 / 1000  #0.000100#(0.0100 / 10000)

    buy_cost_pct_sets = dict(
        zip(test_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(test_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    del params['direction']
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000
    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    env_test_gym = HedgeTraderEnv(
        df=test_data,
        features=features,
        state_space=state_space,
        action_dim=2,
        buy_cost_pct=buy_cost_pct_sets,
        sell_cost_pct=sell_cost_pct_sets,
        ticker_dim=ticker_dimension,
        direction=[variant['direction']],
        mode='eval',
        cont_multnum=CONT_MULTNUM_MAPPING[variant['code']],
        initial_amount=initial_amount,
        open_threshold=THRESHOLD_MAPPING[variant['code']]['long_open'],
        close_threshold=THRESHOLD_MAPPING[variant['code']]['long_close'],
        **params)
    env_test, _ = env_test_gym.get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}_{4}".format(env_test_gym.name, variant['code'],
                                        variant['direction'], index,
                                        variant['method'])
    agent_model_name = model_name + "_{0}".format(name)

    model_path = os.path.join(g_train_path, agent_model_name, model_index)

    #pdb.set_trace()
    env_results = Agent.load_from_file(model_name=model_name,
                                       environment=env_test_gym,
                                       model_path=model_path,
                                       log_dir=log_dir)

    dirs = os.path.join(
        './record', 'agent', variant['method'], 'g_instruments',
        'rolling', 'normal_factors3', str(variant['train_days']),
        str(variant["val_days"]), "{0}_{1}".format(variant['categories'],
                                                   variant['horizon']),
        str(index), "{0}_{1}_{2}_{3}".format(model_name, env_test_gym.name,
                                             variant['code'],
                                             variant['direction']))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(
        dirs, "{0}_{1}_profit.feather".format(index, model_index))
    profit = env_results[0]['profit_memory']
    profit.index.name = 'trade_time'
    profit.reset_index().to_feather(filename)


def predict(variant):
    data_mapping = load_datasets(variant)
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data, test_data = data_mapping[i]
        educate(index=i,
                test_data=test_data,
                variant=variant,
                model_index='best_model')


def predict_all(variant):
    model_indexs = [
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000,
        12000, 13000, 14000, 15000, 16000
    ]
    data_mapping = load_datasets(variant)
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data, test_data = data_mapping[i]
        for j in model_indexs:
            educate(index=i,
                    test_data=test_data,
                    variant=variant,
                    model_index='model{0}'.format(j))
        educate(index=i,
                test_data=test_data,
                variant=variant,
                model_index='best_model')


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

    parser.add_argument('--step_len', type=int,
                        default=10130 - (1 + 5))  ## 每次训练集要经过多少个周期
    parser.add_argument('--batch_size', type=int, default=512)  ## 训练时数据大小
    parser.add_argument('--window', type=int, default=3)  ## 开始周期，即多少个周期构建env数据

    parser.add_argument('--direction', type=str, default='long')  ## 方向
    parser.add_argument('--code', type=str, default='RB')  ## 代码

    args = parser.parse_args()
    train(vars(args))
    #predict(vars(args))
    #predict_all(vars(args))
