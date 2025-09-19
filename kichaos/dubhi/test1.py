import torch, pdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from kichaos.nn.HybridTransformer.transformer import Transformer_base

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
        super(SequentialHybridTransformer, self).__init__(
            enc_in=enc_in,
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
        enc_inp = inputs.permute(0, 2, 1)  # [batch, time, features]
        dec_inp = enc_inp  # 使用相同输入作为decoder输入
        
        enc_out, dec_out, output = super().forward(enc_inp, dec_inp)
        
        # 输出调整回 [batch, prediction_length]
        return enc_out, dec_out, output[:, -1, :]

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# 三维数据生成示例
class TimeSeries3DDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_features):
        self.data = np.random.randn(num_samples, num_features, seq_length)
        self.labels = np.random.randn(num_samples, 1)  # 假设预测单个值
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]), 
            torch.FloatTensor(self.labels[idx])
        )
# 参数设置
num_features = 5    # 特征数（如开盘价、收盘价、成交量等）
seq_length = 60     # 时间步长
batch_size = 32
# 创建数据集
train_dataset = TimeSeries3DDataset(
    num_samples=1000, 
    seq_length=seq_length,
    num_features=num_features
)
val_dataset = TimeSeries3DDataset(
    num_samples=200,
    seq_length=seq_length,
    num_features=num_features
)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
# 模型初始化
model = SequentialHybridTransformer(
    enc_in=num_features,
    dec_in=num_features,
    c_out=1,  # 预测单个值
    d_model=64,
    n_heads=2,
    e_layers=1,
    d_layers=1
)
# 训练循环示例
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            pdb.set_trace()
            _, _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                _, _, outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss/len(val_loader):.4f}")
# 开始训练
train_model(model, train_loader, val_loader, epochs=10)