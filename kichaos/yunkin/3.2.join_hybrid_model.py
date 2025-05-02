import sys, os, torch, pdb, argparse, time
import pandas as pd
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from kdutils.macro import base_path
from ultron.optimize.wisem import *
from ultron.kdutils.progress import Progress

from kichaos.utils.env import *
from kichaos.datasets import CogniDataSet10
from kichaos.nn import SequentialHybridTransformer
from kichaos.nn.SimpleModel.linear2 import SimpleLinearModel


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


def create_prediction_model(features, window):
    params = {
        'd_model': 256,
        'n_heads': 4,
        'e_layers': 4,
        'd_layers': 4,
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


def create_simple_model(features, window, seq_cycle):
    params = {
        'input_channels': len(features) * window,
        'time_steps': seq_cycle
    }
    model = VarianceModel(**params)
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

    #train_data = train_data.loc[:10000]
    #val_data = val_data.loc[:10000]
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


def nll_loss_with_two_models(pred, var, target):
    """
    基于两个模型的NLL损失计算
    pred: 主模型预测 [batch,1]
    var: 方差模型预测 [batch,1] 
    target: 真实值 [batch,1]
    """

    return 0.5 * (torch.log(var) + (target - pred).pow(2) / (2 * var)).mean()


def train(variant):

    writer = SummaryWriter(log_dir='runs/experiment')
    batch_size = 16
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
    variance_model = create_simple_model(features=train_dataset.features,
                                         window=variant['window'],
                                         seq_cycle=variant['seq_cycle'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_model.to(device)
    variance_model.to(device)

    optimizer1 = torch.optim.AdamW(prediction_model.parameters(), lr=0.001)
    optimizer2 = torch.optim.AdamW(variance_model.parameters(), lr=0.001)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=3,
                                                            verbose=True)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=3,
                                                            verbose=True)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        prediction_model.train()
        variance_model.train()
        train_batch_num = 0

        best_val_loss = None
        ## 训练集
        with Progress(len(train_loader),
                      0,
                      label="epoch {0}:train model".format(epoch)) as pg:
            for data in train_loader:
                train_batch_num += 1
                X = to_device(data['values'])
                y = to_device(data['target'])
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                _, _, pred = prediction_model(X)  ## [batch, time, features]

                var = variance_model(X.permute(0, 2,
                                               1))  ## [batch, features, time]

                loss = nll_loss_with_two_models(pred, var,
                                                y)  ## [batch, features, time]
                train_losses.append(loss.item())
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                pg.show(train_batch_num + 1)

        # 校验集
        val_batch_num = 0
        prediction_model.eval()
        variance_model.eval()

        with Progress(len(val_loader),
                      0,
                      label="epoch {0}:val model".format(epoch)) as pg:
            with torch.no_grad():
                for data in val_loader:
                    val_batch_num += 1
                    X = to_device(data['values'])
                    y = to_device(data['target'])
                    _, _, pred = prediction_model(X)
                    var = variance_model(X.permute(0, 2, 1))
                    loss = nll_loss_with_two_models(pred, var, y)
                    val_losses.append(loss.item())
                    pg.show(val_batch_num + 1)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        scheduler1.step(avg_val_loss)
        scheduler2.step(avg_val_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        #writer.add_scalars('Loss', {
        #    'train': avg_train_loss,
        #    'val': avg_val_loss
        #}, epoch)

        writer.add_scalar('LR/prediction_model', optimizer1.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/variance_model', optimizer2.param_groups[0]['lr'], epoch)

        for name, param in prediction_model.named_parameters():
            writer.add_histogram(f'PredictionModel/{name}', param, epoch)
        for name, param in variance_model.named_parameters():
            writer.add_histogram(f'VarianceModel/{name}', param, epoch)

        print('epoch {0}: train loss {1}, val loss {2}'.format(
            epoch,
            sum(train_losses) / len(train_losses),
            sum(val_losses) / len(val_losses)))

        if best_val_loss is None:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        model_dir = os.path.join('runs/models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                prediction_model.state_dict(),
                os.path.join(model_dir,
                             '{}_{}.pth'.format('prediction', epoch)))

            torch.save(
                variance_model.state_dict(),
                os.path.join(model_dir, '{}_{}.pth'.format('variance', epoch)))

    writer.close()


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
