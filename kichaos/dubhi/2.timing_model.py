import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultron.optimize.wisem import *
from kichaos.utils.env import *

from kichaos.datasets import CogniDataSet10
from kichaos.agent.envoy import Envoy0009
from kichaos.utils.env import *


def load_micro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    pdb.set_trace()
    train_filename = os.path.join(os.environ['BASE_PATH'], method,
                                  "train_model_normal.feather")
    train_data = pd.read_feather(train_filename).rename(
        columns={'trade_date': 'trade_time'})
    pdb.set_trace()
    val_filename = os.path.join(os.environ['BASE_PATH'], method,
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(
        columns={'trade_date': 'trade_time'})

    nxt1_columns = train_data.filter(regex="^nxt1_").columns.to_list()

    columns = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns
    ]
    train_data = train_data[['trade_time', 'code'] + columns +
                            ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                                ['trade_time', 'code'])

    val_data = val_data[['trade_time', 'code'] + columns +
                        ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                            ['trade_time', 'code'])

    train_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                      inplace=True)
    val_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                    inplace=True)

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'dummy', 'nxt1_ret']
    ]

    pdb.set_trace()
    train_dataset = CogniDataSet10.generate(train_data,
                                            seq_cycle=seq_cycle,
                                            features=features,
                                            window=window,
                                            target=['nxt1_ret'],
                                            time_name='trade_time',
                                            time_format=time_format)
    val_dataset = CogniDataSet10.generate(val_data,
                                          seq_cycle=seq_cycle,
                                          features=features,
                                          window=window,
                                          target=['nxt1_ret'],
                                          time_name='trade_time',
                                          time_format=time_format)
    return train_dataset, val_dataset, train_data, val_data


def train(variant):
    batch_size = 512
    train_dataset, val_dataset, _, _ = load_micro(
        method=variant['method'],
        window=variant['window'],
        seq_cycle=variant['seq_cycle'],
        horizon=variant['horizon'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    envoy = Envoy0009(id="dubhe_{0}_{1}c_{2}".format(variant['method'],
                                                     variant['seq_cycle'],
                                                     variant['universe']),
                      features=train_dataset.features,
                      ticker_count=len(train_dataset.code.unique()),
                      window=variant['window'],
                      is_debug=True)
    envoy._create_custom_sequential_hybrid_transformer(model_path=None,
                                                       id="{0}".format(
                                                           variant['horizon']))

    max_episode = 80
    envoy._custom_transient_hybrid_transformer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        is_state_dict=True,
        model_dir=envoy.train_path,
        tb_dir=envoy.tensorboard_path,
        push_dir=envoy.push_path,
        epochs=max_episode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=os.environ['DUMMY_NAME'])
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seq_cycle', type=int, default=10)
    parser.add_argument('--universe', type=str, default=os.environ['DUMMY_NAME'])

    args = parser.parse_args()

    train(vars(args))
