from dotenv import load_dotenv

load_dotenv()
import torch,pdb,os,time
import numpy as np
import torch.nn as nn
from torchsummary import summary
from torch import Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from ultron.optimize.wisem import *
from ultron.optimize.wisem.utilz.scheduler import Scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import pandas as pd
from ultron.factor.fitness.metrics import Metrics
from jdwdata.RetrievalAPI import get_data_by_map


base_path = os.path.join('/workspace/data/dev/kd/evolution/nn', '9')



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
    
class CCCLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs:Tensor, targets:Tensor) -> Tensor:
        sigma = ((outputs - outputs.mean())*(targets - targets.mean())).mean()
        return -2*sigma/(self.mse(outputs, targets) + 2*sigma) # CCC越大越好，取相反数

class SharpeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, outputs:Tensor, targets:Tensor) -> Tensor:
        returns = (outputs - targets) / targets
        mask = torch.isfinite(returns)
        returns = returns[mask]    
        sharpe_ratio = torch.mean(returns) / torch.std(returns)
        return -sharpe_ratio

    
class ICLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        vx = outputs - outputs.mean()
        vy = targets - targets.mean()
        return -torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
lossfn = {
    'mse': nn.MSELoss(),
    'ccc': CCCLoss(),
    'sharpe': SharpeLoss(),
    'ic': ICLoss()
}

def load_train_micro(window=3, horizon=1, offset=1):
    train_data = pd.read_feather('./temp/min_train.feather')
    train_data['nxt1_ret_1s'] = train_data['nxt1_ret']
    train_data['nxt1_ret_2s'] = train_data['nxt1_ret']
    features = ['nxt1_ret_1s','nxt1_ret_2s']


    train_dataset = CogniDataSet1.generate(train_data,
                                       window=window,
                                       features=features,
                                       target=['nxt1_ret'])
    train_dataset.array = train_dataset.array.reshape(len(train_dataset.array),
                                                  len(train_dataset.sfeatures),
                                                  train_dataset.window)
    return train_dataset
    #filename = os.path.join(base_path, 'xy_factors_cogin_sets_micro_{0}_{1}_{2}.feather'.format(
    #     horizon, offset, window))
    #print("filename:{0}".format(filename))
    #dataset2 = CogniDataSet.load(filename)
    #dataset2.array = dataset2.array.reshape(len(dataset2.array), len(dataset2.sfeatures), dataset2.window)
    #return dataset2

def load_val_micro(window=3, horizon=1, offset=1):
    val_data = pd.read_feather('./temp/min_val.feather')
    val_data['nxt1_ret_1s'] = val_data['nxt1_ret']
    val_data['nxt1_ret_2s'] = val_data['nxt1_ret']
    features = ['nxt1_ret_1s','nxt1_ret_2s']
    val_dataset = CogniDataSet1.generate(val_data,
                                       window=window,
                                       features=features,
                                       target=['nxt1_ret'])
    val_dataset.array = val_dataset.array.reshape(len(val_dataset.array),
                                                  len(val_dataset.sfeatures),
                                                  val_dataset.window)
    return val_dataset
    #filename = os.path.join(base_path, 'xy_factors_cogin_sets_val_{0}_{1}_{2}.feather'.format(
    #    horizon, offset, window))
    #print("filename:{0}".format(filename))
    #dataset2 = CogniDataSet.load(filename)
    #dataset2.array = dataset2.array.reshape(len(dataset2.array), len(dataset2.sfeatures), dataset2.window)
    #return dataset2


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
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1)

    #  permute 方法将张量的维度调整为 (sequence_length, batch_size, features)
    #  最后调整回 (batch_size, features, sequence_length)
    def forward(self, x):
        x = x.permute(2, 0, 1)  # Change the shape to (sequence_length, batch_size, features)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # Change it back to (batch_size, features, sequence_length)
        return x


 # 两个卷积层、Chomp1d模块、ReLU 激活函数、Dropout层、残差连接和自注意力模块   
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.attention = AttentionBlock(n_outputs)  # Attention layer added

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2, self.attention)
        ## 匹配输入和输出通道数，通过卷积操作改变了特征图的深度（通道数）, 确保在残差连接中，两个张量的维度一致，从而能够进行元素级别的相加操作
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                          padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

#在TemporalConvNet添加了一个线性层用于最终的预测
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: Tensor) -> Tensor:
        #output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output
    
def main(variant):
    name = "{}{}{}{}".format("alphatcn8_", args.lossfn, args.window,
                             args.optim_name)
    
    log_dir = os.path.join(base_log, name)
    model_dir = os.path.join(base_model, name)
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset  = load_train_micro(window=variant['window'], horizon=variant['horizon'])
    val_dataset = load_val_micro(window=variant['window'], horizon=variant['horizon'])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=4000, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4000, shuffle=False)
    ## 特征数 * 窗口期 --> y, 且越近日期，排越前面，一列数据为一个特征
    #
    # 特征数
    #for data in train_loader:
    #    print(data['values'].shape)
    #    break
    
    model = TCN(input_size=len(train_dataset.features), 
                output_size=variant['output_size'], 
                num_channels=[32, 64, 128, 256], 
                kernel_size=variant['kernel_size'], 
                dropout=variant['dropout']).to(device='cuda')
    
    #input_data = torch.randn(1, variant['input_size'], 
    #                         variant['window']).to(device='cuda') 
    #summary(model, input_size=(variant['input_size'], variant['window']))
    #writer.add_graph(model, input_to_model=input_data)

    model, optimizer = create_optimizer(model=model, optim_name='Adam',
                                               lr=1e-3)
    
    loss_func = lossfn[variant['lossfn']]
    model_path = model_dir#os.path.join(model_dir, variant['lossfn'])
    model_scheduler = Scheduler(model=model, optimizer=optimizer, writer=writer, loss_func=loss_func, 
                                model_dir=model_path, is_state_dict=False, is_best_model=True)
    model_scheduler.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / 10000000, 1.0)
    )
    losses = model_scheduler.train_model(max_epoch=50, train_loader=train_loader, val_loader=val_loader)

def metrics(variant):
    name = "{}{}{}{}".format("alphatcn8_", args.lossfn, args.window,
                             args.optim_name)
    
    model_dir = os.path.join(base_model, name)
    filename = os.path.join(model_dir, 'best_model_49.model')
    model = torch.load(filename, map_location='cuda')
    val_dataset = load_val_micro(window=variant['window'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=4000, shuffle=False)
    res = []
    res1 = []
    
    with torch.no_grad():
        for data in val_loader:
            X = data['values'].to(device='cuda').float()
            y = data['target'].to(device='cuda').float()
            outputs = model(X)
            data = {
                'trade_date':data['trade_date'],'code':data['code'],
                'value':outputs.detach().cpu().numpy().reshape(-1)}
            data = pd.DataFrame(data).set_index(['trade_date','code'])
            '''
            data1 = {
                'trade_date':data['trade_date'],'code':data['code'],
                'value':y.detach().cpu().numpy().reshape(-1)
            }
            '''
            res.append(data)
            #res1.append(data1)
    factors_data = pd.concat(res, axis=0)
    factors_data.index.set_levels(pd.to_datetime(factors_data.index.levels[0]), level=0, inplace=True)

    #factors_data1 = pd.concat(res1, axis=0)
    #factors_data1.index.set_levels(pd.to_datetime(factors_data1.index.levels[0]), level=0, inplace=True)
    
    data = get_data_by_map(columns=['dummy120_fst','ret_f1r_oo'],
                    begin_date='2023-01-09',
                    end_date='2023-12-28')
    yields_data = data['ret_f1r_oo']
    dummy_data = data['dummy120_fst']
    yield_score = yields_data.reindex(index=dummy_data.index, columns=dummy_data.columns)
    factors_socre = factors_data['value'].unstack().reindex(index=dummy_data.index, columns=dummy_data.columns)
    ms = Metrics(returns=yield_score,
                 factors=factors_socre,hold=1,
                 is_series=True)
    retain_data = ms.fit_metrics()
    
    print(retain_data.long_evaluate)





if __name__ == '__main__':

    #log_dir = '../records/tcnet_{0}'.format(time.strftime("%Y%m%d%H%M%S"))
    #model_dir = '../models/tcnet_{0}'.format(time.strftime("%Y%m%d%H%M%S"))

    #writer = SummaryWriter(log_dir=log_dir)

    base_log = "../records"
    base_model = "../models"
    torch.cuda.set_device(2)

    parser = argparse.ArgumentParser()

    input_size = 3  # 窗口期
    #parser.add_argument('--input_size', type=int, default=input_size) # input_size == window
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=2)
    
    #### 相关参数
    parser.add_argument('--window', type=int, default=input_size)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--optim_name', type=str, default='Adam')
    parser.add_argument('--lossfn', type=str, default='mse')
    parser.add_argument('--horizon', type=int, default=1)

    args = parser.parse_args()

    main(vars(args))
    #metrics(vars(args))