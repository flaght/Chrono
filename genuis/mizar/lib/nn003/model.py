"""将 kichaos HybridTransformer 适配为分钟收益回归模型。"""

import torch as th
from torch import nn

from kichaos.nn.HybridTransformer import HybridTransformer
from kichaos.nn.HybridTransformer.masking import TriangularCausalMask


class HybridTransformerRegressor(nn.Module):
    """
    输入 (batch, lookback, n_features)，输出标准化未来收益 predicted_z。

    使用项目既有 HybridTransformer 的 Encoder–Decoder 和分头前馈层；
    Encoder 显式使用因果遮罩，Decoder 自注意力在原实现中已默认因果。
    """

    def __init__(self, n_features: int, lookback: int = 20,
                 d_model: int = 64, n_heads: int = 4,
                 e_layers: int = 2, d_layers: int = 1,
                 d_ff: int = 128, dropout: float = 0.1,
                 activation: str = "gelu",
                 prediction_bound_std: float = 3.0):
        super().__init__()
        if int(n_features) <= 0 or int(lookback) <= 0:
            raise ValueError("n_features 和 lookback 必须大于0")
        if int(d_model) <= 0 or int(n_heads) <= 0:
            raise ValueError("d_model 和 n_heads 必须大于0")
        if int(d_model) % int(n_heads):
            raise ValueError("d_model 必须能被 n_heads 整除")
        if int(lookback) > 5000:
            raise ValueError("HybridTransformer 位置编码要求 lookback <= 5000")
        if float(prediction_bound_std) <= 0:
            raise ValueError("prediction_bound_std 必须大于0")
        self.lookback = int(lookback)
        self.prediction_bound_std = float(prediction_bound_std)
        self.backbone = HybridTransformer(
            enc_in=int(n_features), dec_in=int(n_features), c_out=1,
            d_model=int(d_model), n_heads=int(n_heads),
            e_layers=int(e_layers), d_layers=int(d_layers),
            d_ff=int(d_ff), dropout=float(dropout),
            activation=str(activation), output_attention=False)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.ndim != 3 or observations.shape[1] != self.lookback:
            raise ValueError(
                f"输入必须是 (batch, {self.lookback}, n_features)，"
                f"实际为 {tuple(observations.shape)}")
        causal_mask = TriangularCausalMask(
            observations.shape[0], observations.shape[1],
            device=observations.device)
        _, _, sequence_output = self.backbone(
            observations, observations, enc_self_mask=causal_mask)
        raw = sequence_output[:, -1, 0]
        return th.tanh(raw) * self.prediction_bound_std


__all__ = ["HybridTransformerRegressor"]
