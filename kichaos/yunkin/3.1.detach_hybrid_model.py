import sys, os, torch, pdb, argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from kdutils.macro import base_path
from ultron.optimize.wisem import *
from ultron.kdutils.progress import Progress

from kichaos.utils.env import *
from kichaos.datasets import CogniDataSet10
from kichaos.nn import SequentialHybridTransformer
from kichaos.nn.SimpleModel.linear1 import SimpleLinearModel


def create_prediction_model(features, window):
    params = {
        'd_model': 256,
        'n_heads': 2,
        'e_layers': 2,
        'd_layers': 2,
        'dropout': 0.35,
        'denc_dim': 1,
        'activation': 'gelu',
        'output_attention': True
    }
    model = SequentialHybridTransformer(enc_in=len(features) * window,
                                        dec_in=len(features) * window,
                                        c_out=1,
                                        **params)
    return model


def create_simple_model():
    params = {'input_size': 1, 'windows': 1, 'output_size': 1}
    model = SimpleLinearModel(**params)
    return model


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

    prediction_model = create_prediction_model(features=train_dataset.features,
                                               window=variant['window'])
    varinace_model = create_simple_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_model.to(device)
    varinace_model.to(device)

    criterion = nn.MSELoss()
    optimizer1 = torch.optim.AdamW(prediction_model.parameters(), lr=0.001)
    optimizer2 = torch.optim.AdamW(varinace_model.parameters(), lr=0.001)
    num_epochs = 10
    pdb.set_trace()
    for epoch in range(num_epochs):
        pdb.set_trace()
        prediction_model.train()
        varinace_model.train()
        train_loss = 0.0
        with Progress(len(train_loader),
                      0,
                      label="epoch {0}:train model".format(epoch)) as pg:
            for data in train_loader:
                X = to_device(data['values'])
                y = to_device(data['target'])

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                _, _, outputs1 = prediction_model(X)
                pdb.set_trace()
                outputs2 = varinace_model((outputs1.detach() - y).pow(2))

                loss1 = criterion(outputs1, y)
                loss2 = criterion(outputs2, (outputs1.detach() - y).pow(2))

                loss1.backward()
                loss2.backward()

                optimizer1.step()
                optimizer2.step()

                print("loss1 {0}, loss2 {1}".format(loss1.item(), loss2.item()))
        train_loss /= len(train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='aicso1')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seq_cycle', type=int, default=3)
    parser.add_argument('--categories', type=str, default='o2o')
    parser.add_argument('--universe', type=str, default='all')
    args = parser.parse_args()

    train(vars(args))
