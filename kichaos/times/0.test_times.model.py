import os, pdb, inspect, re, time, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from ultron.optimize.wisem import *
from ultron.optimize.wisem.utilz.scheduler import Scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

from ultron.strategy.models.processing import standardize as alk_standardize
from ultron.strategy.models.processing import winsorize as alk_winsorize

pd.options.mode.copy_on_write = True
load_dotenv()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lumina.techinical as lt
import lumina.features as lf
from ultron.tradingday import advanceDateByCalendar


class CustomStandardScaler:

    def __init__(self):
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.scale_ = np.std(X, axis=0, ddof=0)
        return self

    def transform(self, X):
        if self.scale_ is None:
            raise RuntimeError(
                "This CustomStandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X = np.asarray(X)
        return X / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_data(code, method):
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', code,
                             f'{method}.feather')
    print(f'Loading {file_path}')
    data = pd.read_feather(file_path)
    return data


def fetch_features():
    # FeatureExtMean FeatureExtVar 长度不一样，
    t1 = [
        'FeatureBase', 'FeatureAtr', 'FeatureDeg', 'FeaturePrice',
        'FeatureVWap', 'FeatureWave', 'FeatureExtVar', 'FeatureAnnealn',
        'FeatureMaximumSum', 'FeatureMaximumMean', 'FeatureMinimumMean',
        'FeatureMinimumSum', 'FeatureThick'
    ]
    features_list = [
        func for func, obj in inspect.getmembers(lf) if inspect.isclass(obj)
    ]
    features_list = [
        func for func in features_list if func.startswith('Feature')
    ]
    features_list = [func for func in features_list if func not in t1]
    return features_list


def calc_features(data, features_list):
    start_time = time.time()
    res = []
    indexs = data.set_index(['trade_time', 'code']).index
    count, _ = data.shape
    for feature in features_list:
        print(f'Calculating {feature}')
        keys_vars = {
            attr: value
            for attr, value in getattr(lf, feature)().__dict__.items()
            if attr.endswith('_keys')
        }
        key_name = list(keys_vars.keys())[0]
        keys = keys_vars[key_name]
        name = key_name.split('_keys')[0]
        i = 0
        for k in keys:
            i += 1
            params = k if isinstance(k, tuple) else (k, )
            ny = getattr(lt, "calc_{0}".format(name.lower()))(data, *params)
            if isinstance(ny, tuple):
                for j in range(len(ny)):
                    fname = "{0}_{1}_{2}".format(name, i, j)
                    dt = pd.Series(ny[j].tl, index=indexs, name=fname)
                    assert (dt.dropna().shape[0] == count)
                    res.append(dt)
            else:
                fname = "{0}_{1}".format(name, i)
                dt = pd.Series(ny.tl, index=indexs, name=fname)
                assert (dt.dropna().shape[0] == count)
                res.append(dt)
    end_time = time.time()
    print(f'Time: {end_time - start_time}')
    factors_data = pd.concat(res, axis=1).reset_index()
    factors_data = factors_data.merge(data, on=['trade_time', 'code'])
    return factors_data.sort_values(['trade_time', 'code'])


def update_features(code, method):
    features_list = fetch_features()
    data = load_data(code=code, method=method).dropna(subset=['chg'])
    factors = calc_features(data, features_list)
    return factors
    '''
    features = [
        col for col in factors.columns if col not in ['trade_time', 'code']
    ]
    
    filename = os.path.join(os.getenv('BASE_PATH'), 'times',
                            os.environ['CODE'],
                            f'{method}_lumina_features.feather')
    print(f'Saving {filename}')
    factors.reset_index(drop=True).to_feather(filename)
    '''


def process_features(data, scaler_name):
    if scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'custom':
        scaler = CustomStandardScaler()
    elif scaler_name == 'ultron':
        scaler = StandardScaler()

        #raise ValueError('Invalid scaler name')

    start_date = advanceDateByCalendar('china.sse', data['trade_time'].min(),
                                       '2b')
    data = data[data.trade_time >= start_date]
    features = [
        col for col in data.columns
        if col not in ['trade_time', 'code', 'price', 'chg']
    ]

    data[features] = scaler.fit_transform(data[features].values)
    assert (data.dropna().shape[0] == data.shape[0])
    return data


def create_yields(data, scaler_name, horizon, offset=1):
    
    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    
    if scaler_name in ['standard', 'minmax', 'custom']:
        if scaler_name == 'standard':
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_name == 'custom':
            scaler = CustomStandardScaler()
        dt.name = 'nxt1_ret'
        dt = dt.reset_index().dropna(subset=['nxt1_ret'])
        
        dt['nxt1_ret'] = scaler.fit_transform(dt[['nxt1_ret']].values)
        min_val = np.min(dt['nxt1_ret'])
        max_val = np.max(dt['nxt1_ret'])
        data_scaled = 2 * (dt['nxt1_ret']  - min_val) / (max_val - min_val) - 1
        dt['nxt1_ret'] = data_scaled

    elif scaler_name == 'ultron':
        dt = alk_standardize(alk_winsorize(dt.unstack())).unstack()
        dt.name = 'nxt1_ret'
    data = data.merge(dt, on=['trade_time',
                              'code']).sort_values(by=['trade_time', 'code'])
    return data


def calc_factors(method):
    
    horizon = 1
    codes = ['IM', 'IH', 'IC', 'IF']
    res = []
    for code in codes:
        data = update_features(code, method)
        res.append(data)
    data = pd.concat(res, axis=0).sort_values(['trade_time', 'code'])

    data = process_features(data, os.getenv('SCALER'))

    data = create_yields(data=data,
                         scaler_name=os.getenv('SCALER'),
                         horizon=int(horizon))
    
    data.to_feather("./temp/{}_{}.feather".format(method, os.getenv('SCALER')))



## 构建简单模型


## 截取一维张量的最后几个元素, TCN中，用于确保卷积操作的输出大小与输入大小一致
##  chomp_size，表示需要截取的大小
class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


## 用于捕获序列中的相关性
## in_channels，表示输入特征的通道数
class AttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels,
                                               num_heads=1)

    #  permute 方法将张量的维度调整为 (sequence_length, batch_size, features)
    #  最后调整回 (batch_size, features, sequence_length)
    def forward(self, x):
        x = x.permute(
            2, 0,
            1)  # Change the shape to (sequence_length, batch_size, features)
        x, _ = self.attention(x, x, x)
        x = x.permute(
            1, 2,
            0)  # Change it back to (batch_size, features, sequence_length)
        return x


# 两个卷积层、Chomp1d模块、ReLU 激活函数、Dropout层、残差连接和自注意力模块
class TemporalBlock(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.attention = AttentionBlock(n_outputs)  # Attention layer added

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2, self.attention)
        ## 匹配输入和输出通道数，通过卷积操作改变了特征图的深度（通道数）, 确保在残差连接中，两个张量的维度一致，从而能够进行元素级别的相加操作
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# 多个 TemporalBlock，构成了整个 TCN 的网络结构
class TemporalConvNet(nn.Module):
    # num_inputs 表示输入特征的通道数
    # 每个卷积层的输出通道数 num_channels
    # kernel_size 表示卷积核大小
    # dropout 表示 dropout 概率
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#在TemporalConvNet添加了一个线性层用于最终的预测
class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size,
                                   num_channels,
                                   kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: Tensor) -> Tensor:
        #output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output


class SimpleLinearModel(nn.Module):

    def __init__(self, input_size, windows, output_size):
        super(SimpleLinearModel, self).__init__()
        
        self.linear = nn.Linear(input_size * windows,
                                256)  # 输入维度为 6 (2*3)，输出维度为 1
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入张量展平成 (batch_size, 6)
        x = self.linear(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class CogniDataSet1(Dataset):

    @classmethod
    def generate(cls, data, features, window, target=None, start_date=None):
        names = []
        res = []
        start_date = data['trade_time'].min().strftime(
            '%Y-%m-%d %H:%M:%S') if start_date is None else start_date
        data_raw = data.set_index(['trade_time', 'code']).sort_index()
        raw = data_raw[features]
        uraw = raw.unstack()
        for i in range(0, window):
            names += ["{0}_{1}d".format(c, i) for c in features]
            res.append(uraw.shift(i).loc[start_date:].stack())

        dt = pd.concat(res, axis=1)
        dt.columns = names
        dt = dt.reindex(data_raw.index)
        dt = dt.loc[start_date:].dropna()
        if target is not None:
            dt = pd.concat(
                [dt, data.set_index(['trade_time', 'code'])[target]], axis=1)
            dt = dt.dropna().sort_index()
        return CogniDataSet(dt, features, window=window, target=target)

    def __init__(self, data=None, features=None, window=None, target=None):
        if data is None:
            return
        self.data = data
        #names = list(set([f.split('_')[0] for f in features]))
        #self.features = names
        self.features = features
        wfeatures = [f"{f}_{i}d" for f in features for i in range(window)]
        self.window = window
        self.wfeatures = wfeatures  # 时间滚动特征
        self.sfeatures = self.features  # 原始状态特征

        self.array = self.data[self.wfeatures].values
        self.array = torch.from_numpy(self.array).reshape(
            len(self.array), 1, len(self.sfeatures), self.window)
        self.targets = None
        if isinstance(target, list):
            self.targets = self.data[target].values
            self.targets = torch.from_numpy(np.array(self.targets)).reshape(
                len(self.targets), 1)
        self.trade_time = self.data.index.get_level_values(0).strftime(
            '%Y-%m-%d %H:%M:%S')
        self.code = self.data.index.get_level_values(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        date = self.trade_time[index]
        code = self.code[index]
        array = self.array[index]
        return {
            'trade_time': date,
            'code': code,
            'values': array
        } if self.targets is None else {
            'trade_time': date,
            'code': code,
            'values': array,
            'target': self.targets[index]
        }


train_data = pd.read_feather('./temp/min_train_standard.feather')
val_data = pd.read_feather('./temp/min_val_standard.feather')
features = [
    col for col in train_data.columns
    if col not in ['trade_time', 'code', 'price', 'chg', 'nxt1_ret']
]
features = [
    'bias_2', 'bias_3', 'pgo_2', 'pgo_3', 'psl_3', 'willr_3', 'rsi_2',
    'willr_2', 'pvol_1', 'bop_1'
]

train_data['nxt1_ret_1s'] = train_data['nxt1_ret']
train_data['nxt1_ret_2s'] = train_data['nxt1_ret']

val_data['nxt1_ret_1s'] = val_data['nxt1_ret']
val_data['nxt1_ret_2s'] = val_data['nxt1_ret']

features += ['nxt1_ret_1s','nxt1_ret_2s']

window = 4
train_dataset = CogniDataSet1.generate(train_data,
                                       window=window,
                                       features=features,
                                       target=['nxt1_ret'])
#train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=False)

val_dataset = CogniDataSet1.generate(val_data,
                                     window=window,
                                     features=features,
                                     target=['nxt1_ret'])

train_dataset.array = train_dataset.array.reshape(len(train_dataset.array),
                                                  len(train_dataset.sfeatures),
                                                  train_dataset.window)
val_dataset.array = val_dataset.array.reshape(len(val_dataset.array),
                                              len(val_dataset.sfeatures),
                                              val_dataset.window)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4000,
                          shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=4000, shuffle=False)

model = TCN(input_size=len(train_dataset.features),
            output_size=1,
            num_channels=[64, 128, 256, 512],
            kernel_size=8,
            dropout=0.05).to(device='cuda')

model = SimpleLinearModel(input_size=len(train_dataset.features),
                  windows=window,
                  output_size=1).to(device='cuda')
lossfn = {'mse': nn.MSELoss(), 'ccc': CCCLoss()}

model, optimizer = create_optimizer(model=model, optim_name='Adam', lr=1e-3)

writer = SummaryWriter(log_dir='./temp')
loss_func = lossfn['mse']
model_path = './temp'
model_scheduler = Scheduler(model=model,
                            optimizer=optimizer,
                            writer=writer,
                            loss_func=loss_func,
                            model_dir=model_path,
                            is_state_dict=False,
                            is_best_model=True)
model_scheduler.scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda steps: min((steps + 1) / 10000, 1.0))


losses = model_scheduler.train_model(max_epoch=50,
                                     train_loader=train_loader,
                                     val_loader=val_loader)
