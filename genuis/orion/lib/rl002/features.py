import torch
import torch.nn as nn
from gym import spaces
# 假设 kichaos 内部路径兼容 SB3
from kichaos.stable3.common.torch_layers import BaseFeaturesExtractor

class CrossSectionalExtractor(BaseFeaturesExtractor):
    """
    专为超大截面全市场选股设计的特征提取器
    核心逻辑：使用 1D 卷积 (Conv1d) 或共享线性层 (Linear)，
    对每只股票独立进行特征压缩，实现权重共享，拒绝维度爆炸！
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 n_assets: int = 5254, n_stock_features: int = 80):
        # 必须调用父类初始化，features_dim 是最终输出给 SAC 的特征长度
        super(CrossSectionalExtractor, self).__init__(observation_space, features_dim)
        
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        
        # 1. 单只股票的特征提取器 (共享大脑层)
        # 输入单只股票的 n_stock_features 个因子，输出压缩后的 hidden_dim 个高级特征
        self.hidden_dim = 16  # 你哪怕设成 8 都足够了，16 已经很丰富了
        self.stock_encoder = nn.Sequential(
            nn.Linear(self.n_stock_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.LeakyReLU()
        )
        
        # 2. 宏观环境特征维度评估
        self.portfolio_dim = observation_space.shape[0] - (self.n_assets * self.n_stock_features)
        
        # 3. 最终池化层 / 汇总层
        total_extracted_dim = self.n_assets * self.hidden_dim + self.portfolio_dim
        
        # 因为我们最终答应 BaseFeaturesExtractor 输出的是 features_dim (比如设成 256)
        # 所以用一层 Linear 把几万维的安全地压下来
        self.final_aggregator = nn.Sequential(
            nn.Linear(total_extracted_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (BatchSize, 1933475)
        
        # 1. 拆分股票特征和宏观组合特征
        stock_obs_len = self.n_assets * self.n_stock_features
        stock_features = observations[:, :stock_obs_len]       # (Batch, 5254*80)
        portfolio_features = observations[:, stock_obs_len:]   # (Batch, 3)
        
        # 2. Reshape 股票特征为三维矩阵，打通截面通道
        batch_size = observations.shape[0]
        # 变成: (BatchSize * 5254只股票, 80个因子)
        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        
        # 3. 让所有股票“排队”共用同一个大脑进行提炼
        # 一次性算出所有股票的高级特征。速度极快，因为是矩阵并行算！
        # encoded_stocks shape: (BatchSize * 5254, 16)
        encoded_stocks = self.stock_encoder(reshaped_stocks)
        
        # 4. 把提炼完的精简特征再平铺开组合回去
        # 变成: (BatchSize, 5254 * 16)
        flattened_encoded = encoded_stocks.reshape(batch_size, -1)
        
        # 5. 和宏观特征拼接
        # 拼接后 shape: (BatchSize, 5254 * 16 + 3) = (Batch, 84067)
        combined_features = torch.cat([flattened_encoded, portfolio_features], dim=1)
        
        # 6. 送进最终汇总池，压缩到最终的 features_dim (比如 256)
        final_out = self.final_aggregator(combined_features)
        
        return final_out
