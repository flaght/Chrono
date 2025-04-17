import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from ultron.optimize.wisem import *
from kichaos.datasets.hyber.dataset1 import Dataset1
from kichaos.agent.pawn.pawn0008 import Pawn0008
from kichaos.agent.envoy import Envoy0002
from kichaos.agent import Liaison0004
from kichaos.utils.env import *
from kdutils.macro import base_path


def load_micro(method,
               window=3,
               seq_cycle=10,
               horizon=1,
               universe=None,
               time_format='%Y-%m-%d'):
    filename = os.path.join(base_path, universe,
                            "{0}_model_normal.feather".format(method))
    total_data = pd.read_feather(filename)
    total_data = total_data.sort_values(['trade_date', 'code'])
    nxt1_columns = total_data.filter(regex="^nxt1_").columns.to_list()

    columns = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code'] + nxt1_columns
    ]

    total_data = total_data[['trade_date', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_date', 'code'])

    total_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)
    total_data = total_data.sort_values(['trade_date', 'code'])

    dates = total_data['trade_date'].dt.strftime(time_format).unique().tolist()
    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]

    features = [
        col for col in total_data.columns
        if col not in ['trade_date', 'code', 'dummy', 'nxt1_ret']
    ]

    train_dataset = Dataset1.generate(train_data,
                                      codes=train_data['code'].unique(),
                                      seq_cycle=seq_cycle,
                                      features=features,
                                      window=window,
                                      target=['nxt1_ret'],
                                      time_name='trade_date',
                                      time_format=time_format)
    val_dataset = Dataset1.generate(val_data,
                                    codes=train_data['code'].unique(),
                                    seq_cycle=seq_cycle,
                                    features=features,
                                    window=window,
                                    target=['nxt1_ret'],
                                    time_name='trade_date',
                                    time_format=time_format)
    return train_dataset, val_dataset


def train(variant):
    batch_size = 16
    train_dataset, val_dataset = load_micro(method=variant['method'],
                                            window=variant['window'],
                                            seq_cycle=variant['seq_cycle'],
                                            horizon=variant['horizon'],
                                            universe=variant['universe'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    pawn = Pawn0008(id="dubhe_{0}_{1}_{2}h".format(variant['method'],
                                                   variant['universe'],
                                                   variant['horizon']),
                    features=train_dataset.features,
                    ticker_count=len(train_dataset.code.unique()),
                    window=variant['window'])

    long_envoy = Envoy0002(id="dubhe_{0}_{1}c_{2}".format(
        variant['method'], variant['seq_cycle'], variant['universe']),
                           features=train_dataset.features,
                           ticker_count=len(train_dataset.code.unique()),
                           window=variant['window'],
                           is_debug=True)
    long_envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(20))

    ##

    medium_envoy = Envoy0002(id="dubhe_{0}_{1}c_{2}".format(
        variant['method'], variant['seq_cycle'], variant['universe']),
                           features=train_dataset.features,
                           ticker_count=len(train_dataset.code.unique()),
                           window=variant['window'],
                           is_debug=True)
    medium_envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(10))
    

    short_envoy = Envoy0002(id="dubhe_{0}_{1}c_{2}".format(
        variant['method'], variant['seq_cycle'], variant['universe']),
                           features=train_dataset.features,
                           ticker_count=len(train_dataset.code.unique()),
                           window=variant['window'],
                           is_debug=True)
    short_envoy._create_custom_transient_hybrid_transformer(model_path=g_push_path,
                                                      id="{0}".format(5))
    

    envoy = Liaison0004(id="dubhe_{0}_{1}".format(variant['method'],
                                                  variant['universe']),
                        features=train_dataset.features,
                        targets=train_dataset.target,
                        ticker_count=len(train_dataset.code.unique()),
                        window=variant['window'],
                        is_load=True)
    
    
    max_episode = 110
    pdb.set_trace()
    pawn._meldures.train_model(train_loader=train_loader,
                               val_loader=val_loader,
                               is_state_dict=True,
                               model_dir=g_train_path,
                               tb_dir=g_tensorboard_path,
                               push_dir=g_push_path,
                               epochs=max_episode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sicro')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--seq_cycle', type=int, default=5)
    parser.add_argument('--universe', type=str, default='hs300')

    args = parser.parse_args()

    train(vars(args))
