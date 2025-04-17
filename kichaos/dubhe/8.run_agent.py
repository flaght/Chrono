import sys, os, re, pdb, argparse, time, copy
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.data import *
from kdutils.macro import base_path
from kichaos.envs.trader.cn_ashares.long005 import Long005TraderEnv
from kichaos.stable3.common.vec_env import VecMonitor
from kichaos.rl.agent import Agent
from kichaos.utils.env import *


def load_data(variant):
    ### 加载多周期因子数据
    horzions = [5, 10, 15, 20, 30]
    res = []
    for h in horzions[:3]:
        filename = os.path.join(base_path, variant['universe'],
                                "pawn0003_factors_{0}.feather".format(h))
        total_data = pd.read_feather(filename)
        total_data = total_data.sort_values(['trade_date', 'code'])
        total_data = total_data.set_index(['trade_date', 'code'])
        columns = total_data.columns
        factor_names = ["h{0}{1}".format(h, col) for col in columns]
        total_data.columns = factor_names
        total_data = total_data.sort_index()
        res.append(total_data)
    total_data = pd.concat(res, axis=1)
    features = total_data.columns.tolist()
    total_data = total_data.reset_index()
    dates = total_data['trade_date'].dt.strftime('%Y-%m-%d').unique().tolist()

    begin_date = total_data['trade_date'].min().strftime('%Y-%m-%d')
    end_date = total_data['trade_date'].max().strftime('%Y-%m-%d')

    market_data = fetch_daily(begin_date=begin_date,
                              end_date=end_date,
                              universe=variant['universe'])

    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]
    train_data.index = train_data['trade_date'].factorize()[0]
    val_data.index = val_data['trade_date'].factorize()[0]
    pdb.set_trace()
    return train_data.rename(
        columns={'trade_date': 'trade_time'}), val_data.rename(
            columns={'trade_date': 'trade_time'}), features


def train(variant):
    np.random.seed(42)
    fix_seed = 1999
    MODEL_PARAMS = {
        "batch_size": variant['batch_size'],
        "buffer_size": variant['buffer_size'],
        "learning_rate": 0.0001,
        "ent_coef": "auto_0.01",
        "learning_starts": variant['learning_starts']
    }

    train_data, val_data, features = load_data(variant)

    ticker_dimension = len(train_data.code.unique())
    state_space = ticker_dimension

    params = copy.deepcopy(variant)

    env_train, _ = Long005TraderEnv(id="hs300_dubbhe",
                                    df=train_data,
                                    features=features,
                                    action_dim=ticker_dimension,
                                    ticker_dim=ticker_dimension,
                                    state_space=state_space,
                                    step_len=variant['learning_starts'],
                                    mode='train',
                                    **params).get_env()

    env_val, _ = Long005TraderEnv(id="hs300_dubbhe",
                                  df=val_data,
                                  features=features,
                                  action_dim=ticker_dimension,
                                  ticker_dim=ticker_dimension,
                                  state_space=state_space,
                                  step_len=variant['learning_starts'],
                                  mode='val',
                                  **params).get_env()

    model_name = 'sac_base'
    log_dir = os.path.join("../../records/", "logs")

    name = "{0}_{1}".format(variant['universe'], "long005")

    env_train_sac = VecMonitor(env_train, log_dir + '_{0}_train'.format(name))
    env_eval_sac = VecMonitor(env_val, log_dir + '_{0}_eval'.format(name))

    agent = Agent(env=env_train_sac)

    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)
    pdb.set_trace()
    start = time.time()
    total_timesteps = 10000
    agent_model_name = model_name + "_{0}".format(name)
    agent.train_model(model=model_sac,
                      tb_log_name=agent_model_name,
                      check_freq=100,
                      ck_path=os.path.join(g_train_path, agent_model_name),
                      log_path=None,
                      eval_env=env_eval_sac,
                      total_timesteps=total_timesteps)
    pdb.set_trace()
    print("Training Done time: %.3f" % (time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--learning_starts', type=int, default=700)
    parser.add_argument('--temporal_size', type=int, default=5)
    parser.add_argument('--window', type=int, default=3)
    #parser.add_argument('--step_len', type=int, default=200)

    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--universe', type=str, default='hs300')

    args = parser.parse_args()

    train(vars(args))
