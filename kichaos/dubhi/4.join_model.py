import sys, os, torch, pdb, argparse, time
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ultron.optimize.wisem import *
from ultron.kdutils.progress import Progress
from kichaos.nn import TemporalConvolution

from dataset10 import Dataset10 as CogniDataSet10
from dataset10 import Basic as CongniBasic10

from kdutils.process import *
from kdutils.data import fetch_basic
from dotenv import load_dotenv

load_dotenv()


class StandardDeviationModel(nn.Module):

    def __init__(self, input_channels=4, time_steps=3, num_filters=32):
        ### feature--> channel
        ### time_steps --> width 对应卷积核操作的sequence_length 长度
        ### batch_size --> batch
        super(StandardDeviationModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.fc = nn.Linear(num_filters * 2 * (time_steps - 2), 1)

        # 使用更稳定的激活函数组合
        self.log_softplus = nn.LogSigmoid()
        self.epsilon = 1e-6

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        # 修改输出处理流程
        log_var = self.log_softplus(self.fc(x))
        std = torch.sqrt(torch.exp(log_var) + self.epsilon)

        return std


def create_prediction_model(features, window, seq_cycle):
    params = {
        'num_channels': [16, 32, 64],
        'kernel_size': 8,
        'dropout': 0.2,
        'softmax_output': False
    }
    model = TemporalConvolution(input_size=len(features) * window,
                                output_size=1,
                                **params)
    return model


def create_volatility_model(features, window, seq_cycle):
    params = {
        'input_channels': len(features) * window,
        'time_steps': seq_cycle
    }
    model = StandardDeviationModel(**params)
    return model


def nll_loss_with_two_models(pred, var, target):
    """
    基于两个模型的NLL损失计算
    pred: 主模型预测 [batch,1]
    var: 方差模型预测 [batch,1] 
    target: 真实值 [batch,1]
    """

    return 0.5 * (torch.log(var) + (target - pred).pow(2) / (2 * var)).mean()


def load_misro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    val_filename = os.path.join(os.environ['BASE_PATH'], method,
                                "val_model_normal.feather")
    val_data = pd.read_feather(val_filename).rename(
        columns={'trade_date': 'trade_time'})

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


def load_micro(method, window, seq_cycle, horizon, time_format='%Y-%m-%d'):
    train_filename = os.path.join(os.environ['BASE_PATH'], method,
                                  "train_model_normal.feather")
    train_data = pd.read_feather(train_filename).rename(
        columns={'trade_date': 'trade_time'})
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
    batch_size = 512
    task_id = int(time.time())
    writer_dir = os.path.join('runs/experiment/join_model_{0}'.format(task_id))
    writer = SummaryWriter(log_dir=writer_dir)

    model_dir = os.path.join('runs/models/join_model_{0}'.format(task_id))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('model_dir:', model_dir)
    print('writer_dir:', writer_dir)
    train_dataset, val_dataset = load_micro(method=variant['method'],
                                            window=variant['window'],
                                            seq_cycle=variant['seq_cycle'],
                                            horizon=variant['horizon'])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 自定义批次大小
        shuffle=False,
        collate_fn=CogniDataSet10.collate_fn)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,  # 自定义批次大小
        shuffle=False,
        collate_fn=CogniDataSet10.collate_fn)

    ## 创建收益率模型
    prediction_model = create_prediction_model(features=train_dataset.features,
                                               window=variant['window'],
                                               seq_cycle=variant['seq_cycle'])

    ## 创建波动率模型
    volatility_model = create_volatility_model(features=train_dataset.features,
                                               window=variant['window'],
                                               seq_cycle=variant['seq_cycle'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_model.to(device)
    volatility_model.to(device)

    optimizer1 = torch.optim.AdamW(prediction_model.parameters(), lr=0.001)
    optimizer2 = torch.optim.AdamW(volatility_model.parameters(), lr=0.001)

    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lambda steps: min(
        (steps + 1) / 10000, 1.0))
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda steps: min(
        (steps + 1) / 10000, 1.0))
    '''
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                            mode='min',
                                                            factor=0.4,
                                                            patience=3,
                                                            verbose=True)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,
                                                            mode='min',
                                                            factor=0.4,
                                                            patience=3,
                                                            verbose=True)
    '''

    num_epochs = 200

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        prediction_model.train()
        volatility_model.train()
        train_batch_num = 0

        best_val_loss = None
        ## 训练集
        with Progress(len(train_loader),
                      0,
                      label="epoch {0}:train model".format(epoch)) as pg:
            for batch in train_loader:
                train_batch_num += 1
                for data in batch:
                    X = to_device(data['values'])
                    y = to_device(data['target'])
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    if X.shape[0] == 1:
                        continue
                    _, pred = prediction_model(X.permute(0, 2, 1))
                    #_, _, pred = prediction_model(
                    #    X)  ## [batch, time, features]

                    var = volatility_model(X.permute(
                        0, 2, 1))  ## [batch, features, time]

                    loss = nll_loss_with_two_models(
                        pred, var, y)  ## [batch, features, time]
                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer1.step()
                    optimizer2.step()
                pg.show(train_batch_num + 1)

        # 校验集
        val_batch_num = 0
        prediction_model.eval()
        volatility_model.eval()

        with Progress(len(val_loader),
                      0,
                      label="epoch {0}:val model".format(epoch)) as pg:
            with torch.no_grad():
                for batch in val_loader:
                    val_batch_num += 1
                    for data in batch:
                        X = to_device(data['values'])
                        y = to_device(data['target'])
                        _, pred = prediction_model(X.permute(0, 2, 1))
                        var = volatility_model(X.permute(0, 2, 1))
                        loss = nll_loss_with_two_models(pred, var, y)
                        val_losses.append(loss.item())
                    pg.show(val_batch_num + 1)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        scheduler1.step(avg_val_loss)
        scheduler2.step(avg_val_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        writer.add_scalar('LR/prediction_model',
                          optimizer1.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/volatility_model',
                          optimizer2.param_groups[0]['lr'], epoch)

        for name, param in prediction_model.named_parameters():
            writer.add_histogram(f'PredictionModel/{name}', param, epoch)
        for name, param in volatility_model.named_parameters():
            writer.add_histogram(f'VolatilityModel/{name}', param, epoch)

        print('epoch {0}: train loss {1}, val loss {2}'.format(
            epoch,
            sum(train_losses) / len(train_losses),
            sum(val_losses) / len(val_losses)))

        if best_val_loss is None:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        ## 保存每个epoch的模型
        torch.save(
            prediction_model.state_dict(),
            os.path.join(model_dir, '{}_{}.pth'.format('prediction', epoch)))

        torch.save(
            volatility_model.state_dict(),
            os.path.join(model_dir, '{}_{}.pth'.format('volatility', epoch)))

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(
                prediction_model.state_dict(),
                os.path.join(model_dir,
                             '{}_{}.pth'.format('best_prediction',
                                                best_epoch)))

            torch.save(
                volatility_model.state_dict(),
                os.path.join(model_dir,
                             '{}_{}.pth'.format('best_volatility',
                                                best_epoch)))
    writer.close()


def predict(variant):

    def load_state(model_dir, model_name):
        model_name = os.path.join(model_dir, "{0}.pth".format(model_name))
        model_dict = torch.load(model_name, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k[:]
            new_state_dict[name] = v
        return new_state_dict

    task_id = '1746792204'
    test_datasets = load_misro(method=variant['method'],
                               window=variant['window'],
                               seq_cycle=variant['seq_cycle'],
                               horizon=variant['horizon'])
    features = [
        col for col in test_datasets.sfeatures
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]

    ## 创建收益率模型
    prediction_model = create_prediction_model(
        features=test_datasets.sfeatures,
        window=variant['window'],
        seq_cycle=variant['seq_cycle'])

    ## 创建波动率模型
    volatility_model = create_volatility_model(
        features=test_datasets.sfeatures,
        window=variant['window'],
        seq_cycle=variant['seq_cycle'])

    ## 加载模型参数
    model_dir = os.path.join('runs/models/join_model_{0}'.format(task_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdb.set_trace()
    prediction_model = prediction_model.to(device=device)
    volatility_model = volatility_model.to(device=device)
    prediction_model.load_state_dict(
        load_state(model_dir, 'best_prediction_12'))
    volatility_model.load_state_dict(
        load_state(model_dir, 'best_volatility_12'))

    prediction_model.eval()
    volatility_model.eval()

    res = []
    for data in test_datasets.samples:
        print(data['time'])
        X = to_device(data['values'])
        with torch.no_grad():
            _, pred = prediction_model(X.permute(0, 2, 1))
            var = volatility_model(X.permute(0, 2, 1))
        output = (pred / var)
        output = pd.DataFrame(output.detach().cpu(), index=data['codes'])
        output = output.reset_index()
        output = output.rename(columns={'index': 'code', 0: 'value'})
        output['trade_date'] = data['time']
        output['trade_date'] = pd.to_datetime(output['trade_date'])
        output = output.set_index(['trade_date', 'code'])
        res.append(output)
    outputs = pd.concat(res)
    outputs = outputs['value'].unstack()
    min_date = outputs.index.get_level_values('trade_date').min().strftime(
        '%Y-%m-%d')
    max_date = outputs.index.get_level_values('trade_date').max().strftime(
        '%Y-%m-%d')
    val, ret, iret, ret_c2o, usedummy, vardummy = fetch_basic(
        begin_date=min_date, end_date=max_date)
    pdb.set_trace()
    weight = TopNWeight(vardummy, outputs, 1, 5, 0)
    out1, pnl1, tvs1 = CalRet(usedummy, weight, ret, None, iret['000300'], 252)
    out2, pnl2, tvs2 = CalRet(usedummy, weight, ret, ret_c2o, iret['000300'],
                              252)
    print(outputs)


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

    train(vars(args))
    #predict(vars(args))
