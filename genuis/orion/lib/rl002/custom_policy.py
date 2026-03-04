import gym
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from kichaos.stable3.sac.policies import Actor, SACPolicy
from kichaos.stable3.common.policies import ContinuousCritic
from kichaos.stable3.common.preprocessing import get_action_dim

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class PermutationEquivariantActor(Actor):
    """
    置换不变 (Permutation Equivariant) Actor:
    让每只股票独立通过相同的小型网络打分，避免了 5000维 -> 128维 的维度坍缩灾难。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        n_assets: int = 5254,
        n_stock_features: int = 80,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            **kwargs
        )
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        self.portfolio_dim = max(
            int(observation_space.shape[0] - (self.n_assets * self.n_stock_features)), 0
        )
        
        # 抛弃 SB3 默认的全连接层
        self.latent_pi = nn.Identity()
        
        # 构建专属的独立股票打分网络 (Stock-specific Network)
        hidden_dim = 64
        self.stock_net = nn.Sequential(
            nn.Linear(self.n_stock_features + self.portfolio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 每只股票独立输出 Mu 和 LogStd
        self.mu_net = nn.Linear(hidden_dim, 1)
        self.log_std_net = nn.Linear(hidden_dim, 1)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        # 我们不需要 BaseFeaturesExtractor 进行降维，原样返回完整观测
        return obs

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, dict]:
        # obs shape: (Batch, N * n_features + portfolio_features)
        stock_obs_len = self.n_assets * self.n_stock_features
        stock_features = obs[:, :stock_obs_len]  # 提取股票特征
        portfolio_features = obs[:, stock_obs_len:] if self.portfolio_dim > 0 else None
        
        batch_size = obs.shape[0]
        
        # 核心：将 (Batch, 5254*80) 变形为 (Batch * 5254, 80)，让所有股票并行通过打分器
        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        
        if self.portfolio_dim > 0:
            expanded_portfolio = portfolio_features.unsqueeze(1).expand(
                batch_size, self.n_assets, self.portfolio_dim
            ).reshape(-1, self.portfolio_dim)
            reshaped_input = th.cat([reshaped_stocks, expanded_portfolio], dim=1)
        else:
            reshaped_input = reshaped_stocks

        # 独立特征提炼 -> (Batch * 5254, 64)
        latent = self.stock_net(reshaped_input)
        
        # 独立打分 -> (Batch * 5254, 1)
        mu_flat = self.mu_net(latent)
        log_std_flat = self.log_std_net(latent)
        
        # 重新还原回截面维度 -> (Batch, 5254)
        mean_actions = mu_flat.reshape(batch_size, self.n_assets)
        log_std = log_std_flat.reshape(batch_size, self.n_assets)
        
        # 限制方差防止梯度爆炸
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean_actions, log_std, {}


class PermutationInvariantCritic(ContinuousCritic):
    """
    置换不变 Critic (Deep Sets 思想):
    让所有股票 (状态+动作) 的组合分别提取 Q特征，然后通过 Mean Pooling 融合为一个整体的 Q_value。
    避免直接使用一个巨大的全连接网络导致维度爆炸和过拟合。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_assets: int = 5254,
        n_stock_features: int = 80,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        self.portfolio_dim = max(
            int(observation_space.shape[0] - (self.n_assets * self.n_stock_features)), 0
        )
        
        hidden_dim = 64
        # 处理单只股票的 State + Action
        self.q_net1 = nn.Sequential(
            nn.Linear(self.n_stock_features + self.portfolio_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.q_net2 = nn.Sequential(
            nn.Linear(self.n_stock_features + self.portfolio_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 汇总网络: 将 Pooling 后的特征打出最终的一个 Q 标量
        self.qf1_top = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qf2_top = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        stock_obs_len = self.n_assets * self.n_stock_features
        stock_features = obs[:, :stock_obs_len]
        portfolio_features = obs[:, stock_obs_len:] if self.portfolio_dim > 0 else None
        batch_size = obs.shape[0]
        
        # (Batch * 5254, 80)
        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        # (Batch * 5254, 1)
        reshaped_actions = actions.reshape(-1, 1)
        
        if self.portfolio_dim > 0:
            expanded_portfolio = portfolio_features.unsqueeze(1).expand(
                batch_size, self.n_assets, self.portfolio_dim
            ).reshape(-1, self.portfolio_dim)
            combined = th.cat([reshaped_stocks, expanded_portfolio, reshaped_actions], dim=1)
        else:
            combined = th.cat([reshaped_stocks, reshaped_actions], dim=1)
        
        # 独立评估
        latent_q1 = self.q_net1(combined)  # (Batch * 5254, 64)
        latent_q2 = self.q_net2(combined)
        
        # Deep Sets 核心: Sum Pooling (池化融合所有股票的特征，避免 Mean Pooling 完全吃掉梯度)
        pooled_q1 = latent_q1.reshape(batch_size, self.n_assets, -1).mean(dim=1)  # (Batch, 64)
        pooled_q2 = latent_q2.reshape(batch_size, self.n_assets, -1).mean(dim=1)  # (Batch, 64)
        
        # 产出最终 Q 值
        q1_value = self.qf1_top(pooled_q1)
        q2_value = self.qf2_top(pooled_q2)
        
        return q1_value, q2_value

class CrossSectionalSACPolicy(SACPolicy):
    """
    定制化 SAC Policy，将 Actor 和 Critic 替换为上述基于独立资产评估的模型。
    """
    def __init__(self, *args, **kwargs):
        # 我们需要在实例化前截取传入的 kwargs
        # 因为我们自己处理网络，不再依赖 net_arch 动态构建
        self.my_n_assets = kwargs.pop('n_assets', 5254)
        self.my_n_stock_features = kwargs.pop('n_stock_features', 80)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return PermutationEquivariantActor(
            n_assets=self.my_n_assets,
            n_stock_features=self.my_n_stock_features,
            **actor_kwargs
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PermutationInvariantCritic(
            n_assets=self.my_n_assets,
            n_stock_features=self.my_n_stock_features,
            **critic_kwargs
        ).to(self.device)
