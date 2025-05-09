import sys, os, torch, pdb, argparse, time
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from kichaos.nn import SequentialHybridTransformer
from ultron.optimize.wisem import *
from dataset10 import Dataset10 as CogniDataSet10
from dataset10 import Basic as CongniBasic10
from ultron.kdutils.progress import Progress
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()


def create_model(features, window):
    params = {
        'd_model': 256,
        'n_heads': 8,
        'e_layers': 8,
        'd_layers': 8,
        'dropout': 0.25,
        'denc_dim': 1,
        'activation': 'gelu',
        'output_attention': True
    }
    model = SequentialHybridTransformer(enc_in=len(features) * window,
                                        dec_in=len(features) * window,
                                        c_out=1,
                                        **params)
    return model


def load_misro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    pdb.set_trace()
    val_filename = os.path.join(os.environ['BASE_PATH'], method,
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(
        columns={'trade_date': 'trade_time'})

    nxt1_columns = val_data.filter(regex="^nxt1_").columns.to_list()
    columns = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code'] + nxt1_columns
    ]

    val_data = val_data[['trade_time', 'code'] + columns +
                        ['nxt1_ret_{0}h'.format(horizon)]].sort_values(
                            ['trade_time', 'code'])

    val_data.rename(columns={'nxt1_ret_{0}h'.format(horizon): 'nxt1_ret'},
                    inplace=True)

    features = [
        col for col in val_data.columns
        if col not in ['trade_time', 'code', 'dummy', 'nxt1_ret']
    ]
    val_dataset = CongniBasic10.generate(val_data,
                                         features=features,
                                         window=window,
                                         seq_cycle=seq_cycle,
                                         time_name='trade_time',
                                         time_format=time_format)
    return val_dataset


def predict(variant):
    task_id = '1746661049'
    test_datasets = load_misro(method=variant['method'],
                               window=variant['window'],
                               seq_cycle=variant['seq_cycle'],
                               horizon=variant['horizon'])
    features = [
        col for col in test_datasets.sfeatures
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]
    model = create_model(features=features, window=variant['window'])
    model_dir = os.path.join('runs/models/{0}'.format(task_id))
    model_name = os.path.join(model_dir, "{0}.pth".format('best_2'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model_dict = torch.load(model_name, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_dict.items():
        name = k[:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    res = []
    for data in test_datasets.samples:
        print(data['time'])
        X = to_device(data['values'])
        with torch.no_grad():
            _, _, outputs = model(X)
        outputs = pd.DataFrame(outputs.detach().cpu(), index=data['codes'])
        outputs = outputs.reset_index()
        outputs = outputs.rename(columns={'index': 'code', 0: 'value'})
        outputs['trade_date'] = data['time']
        outputs = outputs.set_index(['trade_date', 'code'])
        res.append(outputs)
    pdb.set_trace()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=os.environ['DUMMY_NAME'])
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seq_cycle', type=int, default=4)
    parser.add_argument('--universe',
                        type=str,
                        default=os.environ['DUMMY_NAME'])

    args = parser.parse_args()

    predict(vars(args))
