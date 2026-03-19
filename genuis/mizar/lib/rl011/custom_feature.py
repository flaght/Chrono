import gym
import numpy as np
import torch as th
import torch.nn as nn
from typing import Any, Dict, List, Optional

from kichaos.stable3.common.torch_layers import BaseFeaturesExtractor

def _build_activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    return mapping.get(key, nn.ReLU)()

class _CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=self.pad
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=self.pad
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = _build_activation(activation)
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def _chomp(self, x: th.Tensor) -> th.Tensor:
        if self.pad <= 0:
            return x
        return x[:, :, :-self.pad]

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = self.resample(x)
        out = self.conv1(x)
        out = self._chomp(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self._chomp(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.drop(out)

        return self.act(out + residual)


class TCNFeatureExtractor(BaseFeaturesExtractor):
    """
    时序特征提取器（TCN）:
    - 输入: flatten 后的 [seq_len * feature_dim + 1(net_er)]
    - 输出: features_dim embedding
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        sequence_window: int = 15,
        feature_dim: Optional[int] = None,
        tcn_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(observation_space, int(features_dim))
        total_dim = int(np.prod(observation_space.shape))
        self.sequence_window = int(sequence_window)
        self.net_pos_dim = 1
        payload_dim = total_dim - self.net_pos_dim
        if self.sequence_window <= 0:
            raise ValueError(f"sequence_window 必须 >=1，当前值: {self.sequence_window}")

        if feature_dim is None:
            if payload_dim % self.sequence_window != 0:
                raise ValueError(
                    f"无法从观测维度推断 feature_dim: payload_dim={payload_dim}, "
                    f"sequence_window={self.sequence_window}"
                )
            feature_dim = payload_dim // self.sequence_window
        self.feature_dim = int(feature_dim)
        expected_payload_dim = self.sequence_window * self.feature_dim
        if payload_dim != expected_payload_dim:
            raise ValueError(
                "TCNFeatureExtractor 维度不匹配: "
                f"observation payload_dim={payload_dim}, "
                f"but sequence_window({self.sequence_window}) * feature_dim({self.feature_dim})="
                f"{expected_payload_dim}. 请检查 env.sequence_window 与 feature_dim 配置是否一致。"
            )

        if tcn_channels is None:
            tcn_channels = [64, 128]
        tcn_channels = [int(v) for v in tcn_channels if int(v) > 0]
        if not tcn_channels:
            tcn_channels = [64]

        # 将每个时间步的 feature 向量映射到一维通道序列: [B, T, F] -> [B, F, T]
        in_channels = self.feature_dim
        blocks: List[nn.Module] = []
        for i, out_channels in enumerate(tcn_channels):
            blocks.append(
                _CausalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int(kernel_size),
                    dilation=2 ** i,
                    dropout=float(dropout),
                    activation=activation,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.post = nn.Sequential(
            nn.Linear(in_channels + self.net_pos_dim, int(features_dim)),
            _build_activation(activation),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.view(observations.shape[0], -1)
        expected_total_dim = self.sequence_window * self.feature_dim + self.net_pos_dim
        if x.shape[1] != expected_total_dim:
            raise RuntimeError(
                "TCNFeatureExtractor 输入维度错误: "
                f"got={x.shape[1]}, expected={expected_total_dim} "
                f"(sequence_window={self.sequence_window}, feature_dim={self.feature_dim}, net_pos_dim={self.net_pos_dim})"
            )
        seq_payload = x[:, : self.sequence_window * self.feature_dim]
        net_pos = x[:, -self.net_pos_dim :]
        seq = seq_payload.view(x.shape[0], self.sequence_window, self.feature_dim)  # [B, T, F]
        seq = seq.transpose(1, 2)  # [B, F, T]
        h = self.tcn(seq)          # [B, C, T]
        h = self.pool(h).squeeze(-1)  # [B, C]
        out = th.cat([h, net_pos], dim=1)
        return self.post(out)



class RL011FeatureExtractor(BaseFeaturesExtractor):
    """
    适用于 RL011 的通用 MLP 特征提取器。
    - 输入: obs 向量
    - 输出: features_dim 维 embedding
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, int(features_dim))
        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = int(np.prod(observation_space.shape))
        layers: List[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            h = int(h)
            if h <= 0:
                continue
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(_build_activation(activation))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h

        layers.append(nn.Linear(prev, int(features_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.view(observations.shape[0], -1)
        return self.network(x)