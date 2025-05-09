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
from dotenv import load_dotenv

load_dotenv()


def create_model(features, window):
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
    #train_data = train_data.loc[0:int(len(train_data) * 0.4)]
    #val_data = val_data.loc[0:int(len(val_data) * 0.4)]
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
    writer = SummaryWriter(log_dir='runs/experiment/{0}'.format(task_id))

    model_dir = os.path.join('runs/models/{0}'.format(task_id))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

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

    model = create_model(features=train_dataset.features,
                         window=variant['window'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=3,
                                                            verbose=True)

    num_epochs = 200

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []
        model.train()
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
                    if X.shape[0] == 1:
                        continue
                    _, _, pred = model(X)  ## [batch, time, features]

                    loss = criterion(pred, y)

                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer1.step()

                pg.show(train_batch_num + 1)

        # 校验集
        val_batch_num = 0
        model.eval()

        with Progress(len(val_loader),
                      0,
                      label="epoch {0}:val model".format(epoch)) as pg:
            with torch.no_grad():
                for batch in val_loader:
                    val_batch_num += 1
                    for data in batch:
                        X = to_device(data['values'])
                        y = to_device(data['target'])
                        _, _, pred = model(X)
                        loss = criterion(pred, y)
                        val_losses.append(loss.item())
                    pg.show(val_batch_num + 1)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        scheduler1.step(avg_val_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        writer.add_scalar('LR/model', optimizer1.param_groups[0]['lr'], epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f'Model/{name}', param, epoch)

        print('epoch {0}: train loss {1}, val loss {2}'.format(
            epoch,
            sum(train_losses) / len(train_losses),
            sum(val_losses) / len(val_losses)))

        if best_val_loss is None:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        ## 保存每个epoch的模型
        torch.save(model.state_dict(),
                   os.path.join(model_dir, '{}_{}.pth'.format('model', epoch)))

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, '{}_{}.pth'.format('best',
                                                           best_epoch)))
    writer.close()


def predict(variant):
    test_datasets = load_misro(method=variant['method'],
                               window=variant['window'],
                               seq_cycle=variant['seq_cycle'],
                               horizon=variant['horizon'])
    features = [
        col for col in test_datasets.sfeatures
        if col not in ['trade_time', 'code', 'nxt1_ret']
    ]


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
