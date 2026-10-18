"""用于生成后缀公式的小型 actor-critic Transformer。"""

from __future__ import annotations

from typing import Any


class AlphaGPT:
    """延迟加载 PyTorch，导入挖掘器时不强制要求已安装 torch。"""

    def __new__(
        cls,
        vocab_size: int,
        max_formula_len: int,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> Any:
        try:
            import torch
            import torch.nn as nn
        except ImportError as error:
            raise RuntimeError(
                "AlphaGPT 挖掘器需要 PyTorch: pip install torch"
            ) from error

        class _AlphaGPT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token_emb = nn.Embedding(vocab_size + 1, d_model)
                self.pos_emb = nn.Parameter(
                    torch.zeros(1, max_formula_len + 1, d_model)
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                )
                self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
                self.norm = nn.LayerNorm(d_model)
                self.actor = nn.Linear(d_model, vocab_size)
                self.critic = nn.Linear(d_model, 1)
                self.bos_token = vocab_size

            def forward(self, tokens: Any) -> tuple[Any, Any]:
                length = tokens.size(1)
                hidden = self.token_emb(tokens) + self.pos_emb[:, :length, :]
                mask = torch.triu(
                    torch.full(
                        (length, length), float("-inf"), device=tokens.device
                    ),
                    diagonal=1,
                )
                hidden = self.norm(self.blocks(hidden, mask=mask))
                last = hidden[:, -1, :]
                return self.actor(last), self.critic(last)

        return _AlphaGPT()


__all__ = ["AlphaGPT"]
