import sys, os, re, pdb, argparse, time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append('../../kichaos')

from agent.liaison import Liaison0004
from kichaos.stable3.common.vec_env import VecMonitor
from rl.agent import Agent
from utils.env import *

from sklearn.preprocessing import StandardScaler
from kdutils.data import *
from kdutils.macro import base_path
from kdutils.envst import create_env


def create_data(variant):
    pdb.set_trace()
    filename = os.path.join(
        base_path, variant['universe'],
        "{0}_model_normal.feather".format(variant['method']))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])
    ### 基本行情， 长短周期模型需要数据  MAE模型需要数据
    nxt1_columns = total_data.filter(regex="^nxt1_").columns.to_list()

    ## 特征周期滚动算
    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code'] + nxt1_columns
    ]

    total_data[nxt1_columns] = 0.0

    ### 标准化
    scaler = StandardScaler()
    total_data[features] = scaler.fit_transform(total_data[features].values)

    total_data = total_data.sort_values(['trade_date', 'code'])
    total_data["trade_date"] = pd.to_datetime(total_data["trade_date"])
    total_data.set_index(['trade_date', 'code'], inplace=True)
    raw = total_data[features]
    uraw = raw.unstack()
    res = []
    names = []
    for i in range(0, variant['window']):
        names += ["{0}_{1}d".format(c, i) for c in features]
        res.append(uraw.shift(i).stack())

    dt = pd.concat(res, axis=1)
    dt.columns = names
    dt = dt.reindex(raw.index)

    total_data = pd.concat([dt, total_data[nxt1_columns]], axis=1)
    begin_date = total_data.index.get_level_values(
        'trade_date').min().strftime('%Y-%m-%d')
    end_date = total_data.index.get_level_values('trade_date').max().strftime(
        '%Y-%m-%d')

    market_data = fetch_daily(begin_date,
                              end_date,
                              universe=variant['universe'])

    total_data1 = total_data.reset_index().merge(market_data.reset_index(),
                                                 on=['trade_date', 'code'])
    filename = os.path.join(
        base_path, variant['universe'],
        "{0}_{1}d_agent_factors.feather".format(variant['method'],
                                                variant['window']))
    total_data1.to_feather(filename)


def load_data(method, window, universe):
    filename = os.path.join(
        base_path, universe,
        "{0}_{1}d_agent_factors.feather".format(method, window))

    total_data = pd.read_feather(filename)
    total_data = total_data.rename(columns={'trade_date': 'trade_time'})
    total_data = total_data.sort_values(['trade_time', 'code'])
    total_data["trade_time"] = pd.to_datetime(total_data["trade_time"])

    ## 计算类VIX
    close_pd = total_data.set_index(['trade_time', 'code'])['close']
    close_pd = close_pd.unstack()
    ind1 = np.log(
        close_pd / close_pd.shift(1)).rolling(window=15).std() * np.sqrt(252)
    ind1 = ind1.fillna(method='ffill').fillna(method='bfill')
    ind1 = ind1.unstack()
    ind1.name = 'vix'

    total_data = total_data.merge(ind1.reset_index(),
                                  on=['trade_time', 'code'])

    features = total_data.filter(regex='_\d+d$').columns.to_list()

    total_data = total_data.dropna(subset=features)
    features = list(set([re.sub(r'_\d+d$', '', item) for item in features]))

    targets = total_data.filter(regex="^nxt1_").columns.to_list()
    dates = total_data['trade_time'].dt.strftime('%Y-%m-%d').unique().tolist()
    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_time'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_time'].isin(dates[pos:pos + 80])]

    train_data.index = train_data['trade_time'].factorize()[0]
    val_data.index = val_data['trade_time'].factorize()[0]
    return train_data, val_data, features, targets


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
    train_data, val_data, features, targets = load_data(
        variant['method'], variant['window'], variant['universe'])

    envory_id = "dubhe_{0}_{1}c_{2}".format('sicro', '5', 'hs300')
    liaison_id = "dubhe_{0}_{1}".format('sicro', 'hs300')

    initial_amount = 10000000
    buy_cost_pct = 1e-3
    sell_cost_pct = 1e-3
    ### 原始特征数
    pdb.set_trace()
    env_train, _ = create_env(
        id='hs300_dubhe',
        initial_amount=initial_amount,
        envory_window=3,
        envory_id=envory_id,
        liaison_window=2,
        liaison_id=liaison_id,
        state_transformer_class=Liaison0004,
        actor_transformer_class=None,
        critic_transformer_class=None,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        data=train_data,
        features=features,
        targets=targets,
        step_len=variant['step_len'],  #variant['learning_starts'],
        mode='train',
        variant=variant).get_env()

    env_val, _ = create_env(id='hs300_dubhe',
                            initial_amount=initial_amount,
                            envory_window=3,
                            envory_id=envory_id,
                            liaison_window=2,
                            liaison_id=liaison_id,
                            state_transformer_class=Liaison0004,
                            actor_transformer_class=None,
                            critic_transformer_class=None,
                            buy_cost_pct=buy_cost_pct,
                            sell_cost_pct=sell_cost_pct,
                            data=val_data,
                            features=features,
                            targets=targets,
                            step_len=40,
                            mode='eval',
                            variant=variant).get_env()

    model_name = 'sac_obs'
    log_dir = os.path.join("../../records/", "logs")

    env_train_sac = VecMonitor(env_train, log_dir + '_train')
    env_eval_sac = VecMonitor(env_val, log_dir + '_eval')

    pdb.set_trace()
    agent = Agent(env=env_train_sac)

    model_sac = agent.get_model(model_name=model_name,
                                model_kwargs=MODEL_PARAMS,
                                tensorboard_log=g_tensorboard_path,
                                seed=fix_seed)

    start = time.time()
    total_timesteps = 3600
    agent_model_name = model_name + "__0002"
    agent.train_model(model=model_sac,
                      tb_log_name=agent_model_name,
                      check_freq=100,
                      ck_path=os.path.join(g_train_path, agent_model_name),
                      log_path=None,
                      eval_env=env_eval_sac,
                      total_timesteps=total_timesteps)
    print("Training Done time: %.3f" % (time.time() - start))


def predict(variant):
    np.random.seed(42)
    fix_seed = 1999
    MODEL_PARAMS = {
        "batch_size": variant['batch_size'],
        "buffer_size": variant['buffer_size'],
        "learning_rate": 0.0001,
        "ent_coef": "auto_0.01",
        "learning_starts": variant['learning_starts'],
        'state_transformer_class': Liaison0004,
        'actor_transformer_class': Pawn0002,
        'critic_transformer_class': Pawn0002
    }

    train_data, val_data, features, targets = load_data(
        variant['method'], variant['window'], variant['universe'])

    envory_id = "dubhe_{0}_{1}c_{2}".format('sicro', '5', 'hs300')
    liaison_id = "dubhe_{0}_{1}".format('sicro', 'hs300')
    initial_amount = 1000000
    env_val_gym = create_env(id='hs300_test',
                             initial_amount=initial_amount,
                             envory_window=3,
                             envory_id=envory_id,
                             liaison_window=2,
                             liaison_id=liaison_id,
                             state_transformer_class=Liaison0004,
                             data=val_data,
                             features=features,
                             targets=targets,
                             step_len=40,
                             mode='eval',
                             variant=variant)

    model_name = 'sac_obs'
    agent_model_name = model_name + "__0002"

    log_dir = os.path.join("./", "logs")
    model_path = os.path.join(g_train_path, agent_model_name, "best_model")
    pdb.set_trace()
    Agent.load_from_file(model_name=model_name,
                         environment=env_val_gym,
                         model_path=model_path,
                         log_dir=log_dir)
    print("Predict Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--temporal_size', type=int, default=5)
    parser.add_argument('--step_len', type=int, default=200)

    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--universe', type=str, default='hs300')
    parser.add_argument('--window', type=int, default=5)

    args = parser.parse_args()
    create_data(vars(args))
    train(vars(args))
    #predict(vars(args))
