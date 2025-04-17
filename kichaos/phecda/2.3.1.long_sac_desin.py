import argparse, os, pdb, copy, time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ['INSTRUMENTS'] = 'ifs'
g_instruments = os.environ['INSTRUMENTS']
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
g_start_pos = 46
g_max_pos = 48

from kichaos.agent.pawn.pawn0006 import Pawn0006  #as Pawn
from kichaos.envs.trader.cn_futures._hedge015 import Hedge015TraderEnv as HedgeTraderEnv
from kichaos.stable3.common.vec_env import VecMonitor
from kichaos.rl.agent import Agent
from kichaos.utils.env import *

from kdutils.macro import base_path, codes
from kdutils.macro import *

ACTOR_CLASS_MAPPING = {1: Pawn0006}
CRITIC_CLASS_MAPPING = {1: Pawn0006}


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

    ### 读取收益率
    yield_res = []
    for horizon in [1, 3, 5]:
        filename = os.path.join(
            base_path, variant['method'], g_instruments, 'yields',
            "{0}_{1}h.feather".format(variant['categories'], horizon))
        df = pd.read_feather(filename)
        df.rename(columns={'nxt1_ret': 'nxt{0}h_ret'.format(horizon)},
                  inplace=True)
        df.set_index(['trade_time', 'code'], inplace=True)
        yield_res.append(df)
    yield_data = pd.concat(yield_res, axis=1)

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
        train_data = train_data.merge(yield_data,
                                      on=['trade_time', 'code'],
                                      how='left')
        val_data = val_data.merge(yield_data,
                                  on=['trade_time', 'code'],
                                  how='left')
        test_data = test_data.merge(yield_data,
                                    on=['trade_time', 'code'],
                                    how='left')
        min_time = pd.to_datetime(train_data['trade_time']).min()
        max_time = pd.to_datetime(val_data['trade_time']).max()
        min_date = min_time if min_date is None else min(min_date, min_time)
        max_date = max_time if max_date is None else max(max_date, max_time)
        data_mapping[i] = (train_data, val_data, test_data)
    return data_mapping


def train(variant):
    data_mapping = load_datasets(variant)
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data, test_data = data_mapping[i]
        fit(index=i, train_data=train_data, val_data=val_data, variant=variant)


def fit(index, train_data, val_data, variant):
    nxt1_columns = ['nxt1h_ret', 'nxt3h_ret', 'nxt5h_ret']
    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + ['price', 'close'] +
        nxt1_columns
    ]
    fix_seed = 1999
    MODEL_PARAMS = {
        'learning_rate': lambda f: 1e-4 * f,
        'buffer_size': 100000,
        'ent_coef': "auto_0.01",
        'target_entropy': 'auto',
        'tau': 0.08,
        'batch_size': 32,
        'gamma': 0.75,
        "target_update_interval": 2,
        "train_freq": 60
    }

    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension
    buy_cost_pct = COST_MAPPING[variant['code']][
        'buy']
    sell_cost_pct = COST_MAPPING[variant['code']][
        'sell']
    buy_cost_pct_sets = dict(
        zip(val_data.code, [buy_cost_pct] * ticker_dimension))
    sell_cost_pct_sets = dict(
        zip(val_data.code, [sell_cost_pct] * ticker_dimension))

    params = copy.deepcopy(variant)
    del params['direction']

    state_class_index = 1
    actor_class_index = 1
    critic_class_index = 1
    #params['state_class'] = STATE_CLASS_MAPPING[state_class_index]
    params['actor_class'] = ACTOR_CLASS_MAPPING[actor_class_index]
    params['critic_class'] = CRITIC_CLASS_MAPPING[critic_class_index]
    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]
    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000

    env_train_gym = HedgeTraderEnv(
        df=train_data,
        features=features,
        targets=nxt1_columns,
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
        targets=nxt1_columns,
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

    model_name = 'sac_desin'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}_{4}_{5}state_{6}actor_{7}critic".format(
        env_train_gym.name, variant['code'], variant['direction'], index,
        variant['method'],
        state_class_index, actor_class_index, critic_class_index)
    env_train_sac = VecMonitor(env_train, log_dir + '_{0}_train'.format(name))
    env_eval_sac = VecMonitor(env_val, log_dir + '_{0}_eval'.format(name))

    agent = Agent(env=env_train_sac)


    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)

    start = time.time()
    total_timesteps = 8000
    agent_model_name = model_name + "_{0}".format(name)
    agent.train_model(
        model=model_sac,
        tb_log_name=agent_model_name,
        check_freq=500,  ## 控制每次train训练的次数
        ck_path=os.path.join(g_train_path, agent_model_name),
        log_path=None,
        eval_env=env_eval_sac,
        total_timesteps=total_timesteps)
    print("Training Done time: %.3f" % (time.time() - start))


def predict(variant):
    data_mapping = load_datasets(variant)
    for i in range(g_start_pos, g_max_pos):
        train_data, val_data, test_data = data_mapping[i]
        educate(index=i, test_data=test_data, variant=variant)


def educate(index, test_data, variant):
    nxt1_columns = ['nxt1h_ret', 'nxt3h_ret', 'nxt5h_ret']
    features = [
        col for col in test_data.columns if col not in ['trade_time', 'code'] +
        ['price', 'close'] + nxt1_columns
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
    state_class_index = 1
    actor_class_index = 1
    critic_class_index = 1

    params['actor_class'] = ACTOR_CLASS_MAPPING[actor_class_index]
    params['critic_class'] = CRITIC_CLASS_MAPPING[critic_class_index]
    params['close_times'] = CLOSE_TIME_MAPPING[variant['code']]

    initial_amount = INIT_CASH_MAPPING[variant['code']]  #2000000.0  #60000

    env_test_gym = HedgeTraderEnv(
        df=test_data,
        features=features,
        targets=nxt1_columns,
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

    model_name = 'sac_desin'
    log_dir = os.path.join("../../records/", "logs")
    name = "{0}_{1}_{2}_{3}_{4}_{5}state_{6}actor_{7}critic".format(
        env_test_gym.name, variant['code'], variant['direction'], index,
        variant['method'],
        state_class_index, actor_class_index, critic_class_index)
    
    agent_model_name = model_name + "_{0}".format(name)

    model_path = os.path.join(g_train_path, agent_model_name, "best_model")

    env_results = Agent.load_from_file(model_name=model_name,
                                       environment=env_test_gym,
                                       model_path=model_path,
                                       log_dir=log_dir)
    dirs = os.path.join(
        './record', 'agent', variant['method'], 'g_instruments', 'rolling',
        'normal_factors3', "{0}_{1}".format(variant['categories'],
                                            variant['horizon']),
        "{0}_{1}_{2}_{3}".format(model_name, env_test_gym.name,
                                 variant['code'], variant['direction']))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, "{0}_profit.feather".format(index))
    profit = env_results[0]['profit_memory']
    profit.index.name = 'trade_time'
    profit.reset_index().to_feather(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=int, default=5)  ## 多少个周期训练一次
    parser.add_argument('--train_days', type=int, default=180)  ## 训练天数
    parser.add_argument('--val_days', type=int, default=5)  ## 验证天数

    parser.add_argument('--method', type=str, default='aicso1')  ## 方法
    parser.add_argument('--categories', type=str, default='o2o')  ## 类别
    parser.add_argument('--horizon', type=int, default=1)  ## 预测周期

    parser.add_argument('--nc', type=int, default=1)  ## 标准方式
    parser.add_argument('--swindow', type=int, default=0)  ## 滚动窗口

    parser.add_argument('--step_len', type=int, default=4000)  ## 每次训练集要经过多少个周期
    parser.add_argument('--batch_size', type=int, default=512)  ## 训练时数据大小
    parser.add_argument('--window', type=int, default=3)  ## 开始周期，即多少个周期构建env数据

    parser.add_argument('--direction', type=str, default='long')  ## 方向
    parser.add_argument('--code', type=str, default='IF')  ## 代码

    args = parser.parse_args()
    #train(vars(args))
    predict(vars(args))
