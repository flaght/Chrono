import os, pdb
import pandas as pd
import numpy as np
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dotenv import load_dotenv

load_dotenv()

from lib.lsx001 import fetch_times
from kdutils.macro2 import *
from lib.svx001 import scale_factors
from kichaos.datasets import CogniDataSet1 as CogniDataSet
from kichaos.nn import TemporalConvolution


def load_data(method, instruments, task_id, period):
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))
    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename)  #.set_index(['trade_time', 'code'])

    features = [
        f for f in final_data.columns
        if f not in ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]
    ]
    new_features = ["feature_{0}".format(f) for f in range(0, len(features))]
    final_data.columns = ['trade_time', 'code'
                          ] + new_features + ['nxt1_ret_{0}h'.format(period)]

    return final_data.set_index(['trade_time', 'code'])


def standard_features(prepare_features, method, win):
    pdb.set_trace()
    features = prepare_features.columns
    predict_data = prepare_features.copy().dropna()
    for f in features:
        scale_factors(predict_data=predict_data,
                      method=method,
                      win=win,
                      factor_name=f)
        prepare_features[f] = predict_data['transformed']
    return prepare_features


def create_data(method, instruments, task_id, period):
    sethod = 'roll_zscore'
    win = 240
    #time_array = fetch_times(method=method,
    #                         task_id=task_id,
    #                         instruments=instruments)
    prepare_pd = load_data(method=method,
                           instruments=instruments,
                           task_id=task_id,
                           period=period)
    standard_features(prepare_features=prepare_pd, method=sethod, win=win)
    return prepare_pd.reset_index()


def process_data(prepare_pd, period):
    pdb.set_trace()
    features = [
        col for col in prepare_pd.columns
        if col not in ['trade_time', 'code', 'nxt1_ret_{0}h'.format(period)]
    ]
    cd_set = CogniDataSet.generate(data=prepare_pd.dropna(),
                                   features=features,
                                   window=1,
                                   time_name='trade_time',
                                   target=['nxt1_ret_{0}h'.format(period)])
    cd_set.array = cd_set.array.reshape(len(cd_set.array),
                                        len(cd_set.sfeatures), cd_set.window)
    return cd_set


def create_model(features):
    model = TemporalConvolution(input_size=len(features),
                                output_size=1,
                                num_channels=[16, 32, 64],
                                kernel_size=2,
                                dropout=0.2)
    return model.to(device='cuda')


def train_model(method, task_id, instruments, period):
    random_state = 42
    prepare_pd = create_data(method=method,
                             instruments=instruments,
                             task_id=task_id,
                             period=period)
    cd_set = process_data(prepare_pd=prepare_pd, period=period)
    model = create_model(features=cd_set.features)
    model_optim = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()

    train_loader = DataLoader(dataset=cd_set, batch_size=32, shuffle=False)
    train_loss = []
    pdb.set_trace()
    for data in train_loader:
        X = data['values'].to(device='cuda').float()
        y = data['target'].to(device='cuda').float()
        _, output = model(X)
        pred = output
        loss = criterion(pred, y)
        train_loss.append(loss.item())
        model_optim.zero_grad()
        loss.backward()
        model_optim.step()


train_model(method='bicso0', task_id='113001', instruments='rbb', period=5)
