from ultron.datasets.generate import create_factors
from ultron.optimize.wisem import *
import pdb, torch
## 创建因子数据
start_date = '2019-01-01'
end_date = '2019-07-01'

## 1. 默认8个特征，还有一个是什么？
data1 = create_factors(start_date, end_date, m=8, n=100, res_name='res')
data1.rename(columns={
    'factor_0': 'high',
    'factor_1': 'low',
    'factor_2': 'open',
    'factor_3': 'close',
    'factor_4': 'volume',
    'factor_5': 'amount',
    'factor_6': 'rate',
    'res': 'nxt1_ret'
},
             inplace=True)

## 2. 标准化 使用分位数方式
## y的排序是 每只股票当天排序（1-4000） 还是 所有日期放一起排序（1-100000），x呢，是跟y一样的处理方法么 还是不处理?
nxt1_ret = data1.set_index(
    ['trade_date',
     'code']).groupby(level='trade_date')['nxt1_ret'].rank(pct=True)
nxt1_ret = nxt1_ret.reset_index()
total_data = data1.drop(['nxt1_ret'], axis=1).merge(nxt1_ret,
                                                    on=['trade_date', 'code'])

## 3. 构建数据集   输入的时候带入过去多少期的数据， 设置一个范围，降低尝试消耗的时间
from kichaos.datasets import CogniDataSet3 as CogniDataSet

window = 1  # 把多少天数据拉下来和当期放在一起
seq_cycle = 4 # 构建序列矩阵的周期
dates = total_data['trade_date'].dt.strftime('%Y-%m-%d').unique().tolist()
pos = int(len(dates) * 0.7)
train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
val_data = total_data[total_data['trade_date'].isin(dates[pos - window -
                                                          seq_cycle + 1:])]

features = ['high', 'low', 'open', 'close', 'volume', 'amount', 'rate']

from torch.utils.data import DataLoader, Dataset


class DataSet1(Dataset):

    def __init__(self,
                 dt,
                 feature=features,
                 target_column='nxt1_ret',
                 window_size=3):
        self.dt = dt
        self.feature = feature
        self.target_column = target_column
        self.window_size = window_size
        self.codes = dt['code'].unique()
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据，创建特征矩阵和目标向量"""
        self.samples = []

        ## 简单处理
        for code in self.codes:
            # 获取当前股票的所有数据并按日期排序
            stock_data = self.dt[self.dt['code'] == code].sort_values(
                'trade_date')

            # 提取特征和目标
            features = stock_data[self.feature].values
            targets = stock_data[self.target_column].values
            # 为当前股票创建时间窗口样本
            for i in range(self.window_size, len(stock_data)):
                # 获取过去window_size天的特征
                x = features[i - self.window_size:i, :]
                # 获取当前的目标值(第window_size天后的nxt1_ret)
                y = targets[i - 1]  # 假设nxt1_ret是下一天的收益率

                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 转换为PyTorch张量
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor([y])

        # 调整形状为 (特征数, 时间步长) -> (4, 3)
        x_tensor = x_tensor.permute(1, 0)

        return x_tensor, y_tensor


batch_size = 32

train_loader = DataLoader(DataSet1(train_data, window_size=seq_cycle),
                          batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(DataSet1(val_data, window_size=seq_cycle),
                        batch_size=batch_size,
                        shuffle=False)

### 模型实现
### 损失函数是什么？ 预测收益率是transformer，预测波动率还是transformer么？
### 损失函数是两个模型共用吗？类似SAC。 那些tensor 不做梯度
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorModel(nn.Module):

    def __init__(self, input_channels, time_steps=3, num_filter=2):
        super(PredictorModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filter, kernel_size=2)
        self.conv2 = nn.Conv1d(num_filter, num_filter * 2, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_filter)
        self.bn2 = nn.BatchNorm1d(num_filter * 2)
        #self.fc = nn.Linear(num_filter * 2 * 1, 1)
        self.fc = nn.Linear(num_filter * 2 * (time_steps - 2), 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VarianceModel(nn.Module):

    def __init__(self, input_channels=4, time_steps=3, num_filters=32):
        super(VarianceModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.fc = nn.Linear(num_filters * 2 * (time_steps - 2), 1)
        self.softplus = nn.Softplus()  # 保证输出是正数

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.softplus(self.fc(x)) + 1e-6


class JointModel(nn.Module):

    def __init__(self, predictor, variance):
        super(JointModel, self).__init__()
        self.predictor = predictor
        self.variance = variance

    def forward(self, x):
        pred = self.predictor(x)  # 收益率预测
        var = self.variance(x)  # 波动率预测
        return torch.cat([pred, var], dim=1)


def nll_loss_with_two_models(pred, var, target):
    """
    基于两个模型的NLL损失计算
    pred: 主模型预测 [batch,1]
    var: 方差模型预测 [batch,1] 
    target: 真实值 [batch,1]
    """

    return 0.5 * (torch.log(var) + (target - pred).pow(2) / (2 * var)).mean()


num_epochs = 10
### 训练模型
predictor = PredictorModel(input_channels=len(features), time_steps=seq_cycle)
variance = VarianceModel(input_channels=len(features), time_steps=seq_cycle)
joint_model = JointModel(predictor, variance)

optimizer = torch.optim.Adam(joint_model.parameters(), lr=0.001)
for epoch in range(10):
    joint_model.train()
    loss1 = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = joint_model(x)  ## [batch,feature,time]
        pred, var = outputs[:, 0], outputs[:, 1]

        loss = nll_loss_with_two_models(pred, var, y)

        loss.backward()
        print(loss.item())
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss1/len(train_loader):.4f}')

###
