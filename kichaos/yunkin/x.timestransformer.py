import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
from ultron.optimize.wisem.utilz.optimizer import to_device
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from kdutils.macro import base_path
from ultron.optimize.wisem import *
from kichaos.utils.env import *

from kichaos.datasets import CogniDataSet10
from kichaos.agent.envoy import Envoy0009
from kichaos.utils.env import *


def load_misro(method,
               window,
               seq_cycle,
               horizon,
               categories,
               time_format='%Y-%m-%d %H:%M:%S'):
    test_filename = os.path.join(
        base_path, method, 'normal',
        "test_normal_{0}_{1}h.feather".format(categories, horizon))

    test_data = pd.read_feather(test_filename)

    features = [
        col for col in test_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]

    test_dataset = CogniDataSet10.build(data=test_data,
                                        seq_cycle=seq_cycle,
                                        features=features,
                                        window=window,
                                        time_name='trade_time',
                                        time_format=time_format)
    return test_dataset


def load_micro(method,
               window,
               seq_cycle,
               horizon,
               categories,
               time_format='%Y-%m-%d'):
    train_filename = os.path.join(
        base_path, method, 'normal',
        "train_normal_{0}_{1}h.feather".format(categories, horizon))
    train_data = pd.read_feather(train_filename)
    val_filename = train_filename = os.path.join(
        base_path, method, 'normal',
        "val_normal_{0}_{1}h.feather".format(categories, horizon))
    val_data = pd.read_feather(val_filename)

    features = [
        col for col in train_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret']
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
    return train_dataset, val_dataset


def predict(variant):
    test_datasets = load_misro(method=variant['method'],
                               window=variant['window'],
                               seq_cycle=variant['seq_cycle'],
                               horizon=variant['horizon'],
                               categories=variant['categories'],
                               time_format='%Y-%m-%d %H:%M:%S')
    features = [
        col for col in test_datasets.sfeatures
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]
    tickers = test_datasets.codes
    pdb.set_trace()
    envoy = Envoy0009(id="yunkin_{0}_{1}c_{2}".format(variant['method'],
                                                      variant['seq_cycle'],
                                                      variant['universe']),
                      features=features,
                      ticker_count=len(tickers),
                      window=variant['window'],
                      is_debug=True)
    envoy._create_custom_sequential_hybrid_transformer(model_path=g_push_path,
                                                       id="{0}".format(1))
    for data in test_datasets.samples:
        inputs = to_device(data['values'])
        outputs = envoy._custom_transient_hybrid_transformer.predict(inputs)
        outputs = pd.Series(
            outputs.detach().cpu().numpy().flatten(),
            index=pd.MultiIndex.from_product(
                [[pd.to_datetime(data['time'])], data['codes']],
                names=['trade_time', 'code']))
        print(outputs)


def train(variant):
    batch_size = 64
    train_dataset, val_dataset = load_micro(method=variant['method'],
                                            window=variant['window'],
                                            categories=variant['categories'],
                                            seq_cycle=variant['seq_cycle'],
                                            horizon=variant['horizon'])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    envoy = Envoy0009(id="yunkin_{0}_{1}c_{2}".format(variant['method'],
                                                      variant['seq_cycle'],
                                                      variant['universe']),
                      features=train_dataset.features,
                      ticker_count=len(train_dataset.code.unique()),
                      window=variant['window'],
                      is_debug=True)
    pdb.set_trace()
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
    parser.add_argument('--method', type=str, default='aicso1')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seq_cycle', type=int, default=3)
    parser.add_argument('--categories', type=str, default='o2o')
    parser.add_argument('--universe', type=str, default='all')
    args = parser.parse_args()

    #train(vars(args))
    predict(vars(args))
