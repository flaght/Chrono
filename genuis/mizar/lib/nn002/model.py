"""用于分钟收益预测的因果 Transformer 回归模型。"""

from typing import Iterable

import torch as th
from torch import nn


class TransformerRegressor(nn.Module):
    """
    输入 (batch, lookback, n_features)，输出标准化未来收益 predicted_z。

    每个观测窗口只包含 T 及之前的数据。注意力层再使用因果遮罩，保证
    任意时间 token 都不能访问其右侧 token，最后取 T 对应的末端 token。
    """

    def __init__(self, n_features: int, lookback: int = 20,
                 d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 head_hidden: Iterable[int] = (128, 64),
                 dropout: float = 0.1,
                 prediction_bound_std: float = 3.0):
        super().__init__()
        if int(n_features) <= 0 or int(lookback) <= 0:
            raise ValueError("n_features 和 lookback 必须大于0")
        if int(d_model) % int(nhead):
            raise ValueError("d_model 必须能被 nhead 整除")
        if float(prediction_bound_std) <= 0:
            raise ValueError("prediction_bound_std 必须大于0")
        self.lookback = int(lookback)
        self.input_projection = nn.Linear(int(n_features), int(d_model))
        self.input_norm = nn.LayerNorm(int(d_model))
        self.position_embedding = nn.Parameter(
            th.zeros(1, self.lookback, int(d_model)))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.register_buffer(
            "causal_mask",
            th.triu(th.ones(self.lookback, self.lookback, dtype=th.bool),
                    diagonal=1),
            persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model), nhead=int(nhead),
            dim_feedforward=int(dim_feedforward), dropout=float(dropout),
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_layers),
            norm=nn.LayerNorm(int(d_model)))

        layers = []
        input_dim = int(d_model)
        for hidden in head_hidden:
            layers.extend([nn.Linear(input_dim, int(hidden)), nn.GELU()])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            input_dim = int(hidden)
        layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*layers)
        self.prediction_bound_std = float(prediction_bound_std)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.ndim != 3 or observations.shape[1] != self.lookback:
            raise ValueError(
                f"输入必须是 (batch, {self.lookback}, n_features)，"
                f"实际为 {tuple(observations.shape)}")
        encoded = self.input_norm(self.input_projection(observations))
        encoded = encoded + self.position_embedding
        encoded = self.encoder(encoded, mask=self.causal_mask)
        output = self.head(encoded[:, -1, :]).squeeze(-1)
        return th.tanh(output) * self.prediction_bound_std


__all__ = ["TransformerRegressor"]
