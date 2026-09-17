"""与 rl016 TCN 特征层同口径的直接收益回归模型。"""

from typing import Iterable

import torch as th
from torch import nn


class CausalTCNBlock(nn.Module):
    """左侧填充保证卷积只能读取当前及过去数据。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_channels), int(out_channels), kernel_size=int(kernel_size),
            dilation=int(dilation), padding=0)
        self.norm = nn.GroupNorm(1, int(out_channels))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.residual = (nn.Conv1d(int(in_channels), int(out_channels), 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = self.residual(x)
        out = nn.functional.pad(x, (self.left_padding, 0))
        out = self.conv(out)
        out = self.norm(out)
        return self.dropout(self.activation(out + residual))


class TCNRegressor(nn.Module):
    """
    输入 (batch, lookback, n_features)，输出标准化未来收益 predicted_z。

    最后一层使用 tanh × prediction_bound_std，与 rl016 的 SAC 动作边界
    完全一致，避免模型比较时输出范围不同。
    """

    def __init__(self, n_features: int, channels: int = 64,
                 kernel_size: int = 3, num_blocks: int = 4,
                 features_dim: int = 64,
                 head_hidden: Iterable[int] = (128, 64),
                 dropout: float = 0.0,
                 prediction_bound_std: float = 3.0):
        super().__init__()
        if int(n_features) <= 0:
            raise ValueError("n_features 必须大于0")
        if float(prediction_bound_std) <= 0:
            raise ValueError("prediction_bound_std 必须大于0")
        blocks = []
        in_channels = int(n_features)
        for block_index in range(int(num_blocks)):
            blocks.append(CausalTCNBlock(
                in_channels, int(channels), int(kernel_size),
                dilation=2 ** block_index, dropout=float(dropout)))
            in_channels = int(channels)
        self.tcn = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(int(channels), int(features_dim)),
            nn.LayerNorm(int(features_dim)), nn.GELU())

        layers = []
        input_dim = int(features_dim)
        for hidden in head_hidden:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.GELU()])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            input_dim = int(hidden)
        layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*layers)
        self.prediction_bound_std = float(prediction_bound_std)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded = self.tcn(observations.transpose(1, 2))
        features = self.projection(encoded[:, :, -1])
        return th.tanh(self.head(features).squeeze(-1)) * self.prediction_bound_std


__all__ = ["CausalTCNBlock", "TCNRegressor"]
