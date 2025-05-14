import sys, os, torch, pdb, argparse, time, datetime
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
#from kichaos.nn import SequentialHybridTransformer
from kichaos.nn.HybridTransformer.transformer import Transformer_base
from kichaos.nn import TemporalConvolution
from ultron.optimize.wisem import *
from dataset10 import Dataset10 as CogniDataSet10
from dataset10 import Basic as CongniBasic10
from ultron.kdutils.progress import Progress
from dotenv import load_dotenv

load_dotenv()
'''
class SequentialHybridTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=-1,
                 output_attention=False):
        super(SequentialHybridTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)
        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim

    def hidden_size(self):
        return self.d_model

    def forward(self, inputs):
        # 将输入数据从四维变为三维
        enc_inp = inputs
        if self.denc_dim > 0:
            dec_inp = inputs[:, :, -self.denc_dim:]
        else:
            dec_inp = inputs

        # 去掉与 stock_num 相关的处理
        enc_out, dec_out, output = super(SequentialHybridTransformer,self).forward(enc_inp, dec_inp)
        return enc_out, dec_out, output
'''
loss_mapping = {
    'mse': nn.MSELoss(),
}


class SequentialHybridTransformer(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=False):
        super(SequentialHybridTransformer,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)

        self.d_model = d_model
        self.c_out = c_out

    def forward(self, inputs):
        """处理三维输入 [batch, feature, time]"""
        # 调整维度顺序为 [batch, time, features]
        #enc_inp = inputs.permute(0, 2, 1)  # [batch, time, features]
        enc_inp = inputs
        dec_inp = enc_inp  # 使用相同输入作为decoder输入

        enc_out, dec_out, output = super().forward(enc_inp, dec_inp)

        # 输出调整回 [batch, prediction_length]
        return enc_out, dec_out, output[:, -1, :]


class SequentialHybridTransformer1(Transformer_base):

    def __init__(self,
                 enc_in,
                 dec_in,
                 c_out,
                 d_model=128,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 denc_dim=1,
                 output_attention=False):
        super(SequentialHybridTransformer1,
              self).__init__(enc_in=enc_in,
                             dec_in=dec_in,
                             c_out=c_out,
                             d_model=d_model,
                             n_heads=n_heads,
                             e_layers=e_layers,
                             d_layers=d_layers,
                             d_ff=d_ff,
                             dropout=dropout,
                             activation=activation,
                             output_attention=output_attention)

        self.d_model = d_model
        self.c_out = c_out
        self.denc_dim = denc_dim

    def forward(self, inputs):
        # 输入为 [batch, time, features]
        enc_inp = inputs
        # 如果 denc_dim > 0，选择最后 denc_dim 个时间步作为解码器输入
        if self.denc_dim > 0:
            dec_inp = inputs[:, -self.denc_dim:, :]
        else:
            dec_inp = inputs

        # 使用父类的 forward 方法进行编码和解码
        enc_out, dec_out, output = super(SequentialHybridTransformer1,
                                         self).forward(enc_inp, dec_inp)
        # 对输出进行处理，确保输出为 [batch, 1]
        # 可以使用全连接层或其他方法进行降维
        output = output.mean(dim=1)  # 对时间维度进行平均，得到 [batch, features]
        #output = output.squeeze(-1)  # 压缩最后一个维度，得到 [batch]
        return enc_out, dec_out, output


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
        'n_heads': 4,
        'e_layers': 4,
        'd_layers':4,
        'dropout': 0.15,
        'denc_dim': 2,
        'activation': 'gelu',
        'output_attention': True
    }
    model = SequentialHybridTransformer1(enc_in=len(features) * window,
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
    task_id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    writer_dir = os.path.join('runs/experiment/time_model_{0}_{1}'.format(
        variant['loss'], task_id))
    writer = SummaryWriter(log_dir=writer_dir)

    model_dir = os.path.join('runs/models/time_model_{0}_{1}'.format(
        variant['loss'], task_id))
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

    #model = create_model(features=train_dataset.features,
    #                     window=variant['window'])
    model = create_model1(features=train_dataset.features,
                          window=variant['window'],
                          seq_cycle=variant['seq_cycle'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = loss_mapping[variant['loss']]  #nn.MSELoss()
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=0.005)
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
                    #pred = model(X.permute(0, 2, 1))
                    #_, pred = model(X.permute(0, 2, 1))
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
                pdb.set_trace()
                for batch in val_loader:
                    val_batch_num += 1
                    for data in batch:
                        X = to_device(data['values'])
                        y = to_device(data['target'])
                        _, _, pred = model(X)
                        #pred = model(X.permute(0, 2, 1))
                        #_, pred = model(X.permute(0, 2, 1))
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
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--universe',
                        type=str,
                        default=os.environ['DUMMY_NAME'])

    args = parser.parse_args()

    train(vars(args))
