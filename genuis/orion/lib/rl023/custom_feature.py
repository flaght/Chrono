"""

Only needed when using CrossSectionalSACPolicy with a deep encoder.
For most cases, the Actor's built-in stock_net is sufficient and
this module is NOT required.

Keep for reference or future use with very large feature spaces.
"""

import torch
import torch.nn as nn
from gym import spaces
from kichaos.stable3.common.torch_layers import BaseFeaturesExtractor


class CrossSectionalExtractor(BaseFeaturesExtractor):
    """
    Shared-encoder features extractor for cross-sectional obs.

    Takes obs = [n_assets * n_stock_features, portfolio_dim]
    and compresses stock features via a shared encoder before aggregating.

    Note: This is an alternative to the Actor's built-in stock_net.
    Use this when you want a deeper/wider encoder or when combining
    with MlpPolicy instead of CrossSectionalSACPolicy.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int,
        n_assets: int,
        n_stock_features: int,
        stock_encoder_mid_dim: int = 32,
        stock_encoder_out_dim: int = 16,
    ):
        super().__init__(observation_space, features_dim)

        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        self.hidden_dim = stock_encoder_out_dim

        # 共享股票编码器
        self.stock_encoder = nn.Sequential(
            nn.Linear(self.n_stock_features, stock_encoder_mid_dim),
            nn.LeakyReLU(),
            nn.Linear(stock_encoder_mid_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )

        # 组合特征维度
        self.portfolio_dim = observation_space.shape[0] - (self.n_assets * self.n_stock_features)
        total_extracted_dim = self.n_assets * self.hidden_dim + max(self.portfolio_dim, 0)

        # 汇总层
        self.final_aggregator = nn.Sequential(
            nn.Linear(total_extracted_dim, min(1024, total_extracted_dim)),
            nn.ReLU(),
            nn.Linear(min(1024, total_extracted_dim), features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        stock_obs_len = self.n_assets * self.n_stock_features

        stock_features = observations[:, :stock_obs_len]
        portfolio_features = observations[:, stock_obs_len:]

        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        encoded_stocks = self.stock_encoder(reshaped_stocks)
        flattened_encoded = encoded_stocks.reshape(batch_size, -1)

        if self.portfolio_dim > 0:
            combined_features = torch.cat([flattened_encoded, portfolio_features], dim=1)
        else:
            combined_features = flattened_encoded

        return self.final_aggregator(combined_features)
