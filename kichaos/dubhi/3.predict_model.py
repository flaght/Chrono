import sys, os, torch, pdb, argparse, time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from kichaos.nn import SequentialHybridTransformer
from kichaos.nn import TemporalConvolution
from ultron.optimize.wisem import *
from dataset10 import Dataset10 as CogniDataSet10
from dataset10 import Basic as CongniBasic10
from ultron.kdutils.progress import Progress
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()


class VarianceModel(nn.Module):

    def __init__(self, input_channels=4, time_steps=3, num_filters=32):
        ### feature--> channel
        ### time_steps --> width 对应卷积核操作的sequence_length 长度
        ### batch_size --> batch

        super(VarianceModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.fc = nn.Linear(num_filters * 2 * (time_steps - 2), 1)
        self.softplus = nn.Softplus()  # 保证输出是正数

    def forward(self, x):
        ### 初始时间步骤： time_steps
        ### 第一层卷积后: time_steps - 1
        ### 第二层卷积后: time_steps - 2
        ### 全连接层输入维度: num_filters * 2 * (time_steps - 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.softplus(self.fc(x)) + 1e-6


def create_model3(features, window, seq_cycle):
    params = {
        'num_channels': [16, 32, 64],
        'kernel_size': 4,
        'dropout': 0.2,
        'softmax_output': False
    }
    model = TemporalConvolution(input_size=len(features) * window,
                                output_size=1,
                                **params)
    return model


def create_model2(features, window, seq_cycle):
    params = {
        'input_channels': len(features) * window,
        'time_steps': seq_cycle
    }
    model = VarianceModel(**params)
    return model


def create_model1(features, window, seq_cycle):
    params = {
        'd_model': 256,
        'n_heads': 2,
        'e_layers': 2,
        'd_layers': 2,
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
    task_id = '1746779220'
    test_datasets = load_misro(method=variant['method'],
                               window=variant['window'],
                               seq_cycle=variant['seq_cycle'],
                               horizon=variant['horizon'])
    features = [
        col for col in test_datasets.sfeatures
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]
    model = create_model1(features=features,
                          window=variant['window'],
                          seq_cycle=variant['seq_cycle'])
    #model = create_model2(features=features,
    #                      window=variant['window'],
    #                      seq_cycle=variant['seq_cycle'])
    #model = create_model3(features=features,
    #                      window=variant['window'],
    #                      seq_cycle=variant['seq_cycle'])
    model_dir = os.path.join('runs/models/{0}'.format(task_id))
    model_name = os.path.join(model_dir, "{0}.pth".format('best_1'))
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
            #outputs = model(X.permute(0, 2, 1))
            #_, outputs = model(X.permute(0, 2, 1))
        pdb.set_trace()
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
