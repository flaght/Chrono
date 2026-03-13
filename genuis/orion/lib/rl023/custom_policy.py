from typing import List, Optional, Tuple

import gym
import torch as th
from torch import nn

from kichaos.stable3.common.policies import ContinuousCritic
from kichaos.stable3.sac.policies import Actor, SACPolicy

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class PermutationEquivariantActor(Actor):
    """
    Shared-scorer actor for cross-sectional ranking.

    Training:
      - Receives flattened obs = [subset_features, portfolio_features].
      - Outputs per-stock scores via a shared stock_net.
    Inference:
      - score_assets() accepts arbitrary number of assets.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        n_assets: int,
        n_stock_features: int,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            **kwargs,
        )
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        self.portfolio_dim = max(
            int(observation_space.shape[0] - (self.n_assets * self.n_stock_features)), 0
        )

        # 共享打分网络（核心）
        self.latent_pi = nn.Identity()
        self.stock_net = nn.Sequential(
            nn.Linear(self.n_stock_features + self.portfolio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_net = nn.Linear(hidden_dim, 1)
        self.log_std_net = nn.Linear(hidden_dim, 1)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs

    def _prepare_input(
        self,
        stock_features: th.Tensor,
        portfolio_features: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """拼接 stock_features 和 portfolio_features。"""
        if self.portfolio_dim <= 0:
            return stock_features

        if portfolio_features is None:
            portfolio_features = th.zeros(
                (1, self.portfolio_dim), device=stock_features.device, dtype=stock_features.dtype
            )
        elif portfolio_features.ndim == 1:
            portfolio_features = portfolio_features.unsqueeze(0)

        if portfolio_features.shape[0] == 1:
            portfolio_features = portfolio_features.expand(stock_features.shape[0], -1)
        return th.cat([stock_features, portfolio_features], dim=1)

    def score_assets(
        self,
        stock_features: th.Tensor,
        portfolio_features: Optional[th.Tensor] = None,
        output_activation: str = "sigmoid",
    ) -> th.Tensor:
        """
        Inference: score arbitrary number of assets.
        Input: stock_features shape (N_assets, n_stock_features)
        Output: scores shape (N_assets,)
          - output_activation='sigmoid': [0, 1]
          - output_activation='tanh': [-1, 1]
          - output_activation='none': raw mu
        """
        prepared = self._prepare_input(stock_features, portfolio_features)
        latent = self.stock_net(prepared)
        mu = self.mu_net(latent).squeeze(-1)
        activation = str(output_activation).lower()
        if activation == "sigmoid":
            return th.sigmoid(mu)
        if activation == "tanh":
            return th.tanh(mu)
        if activation == "none":
            return mu
        raise ValueError(f"unsupported output_activation: {output_activation}")

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, dict]:
        """Training: parse flattened obs → per-stock mu, log_std."""
        stock_obs_len = self.n_assets * self.n_stock_features
        stock_features = obs[:, :stock_obs_len]
        portfolio_features = obs[:, stock_obs_len:] if self.portfolio_dim > 0 else None
        batch_size = obs.shape[0]

        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        if self.portfolio_dim > 0:
            expanded_portfolio = portfolio_features.unsqueeze(1).expand(
                batch_size, self.n_assets, self.portfolio_dim
            ).reshape(-1, self.portfolio_dim)
        else:
            expanded_portfolio = None

        prepared = self._prepare_input(reshaped_stocks, expanded_portfolio)
        latent = self.stock_net(prepared)
        mu_flat = self.mu_net(latent)
        log_std_flat = self.log_std_net(latent)

        mean_actions = mu_flat.reshape(batch_size, self.n_assets)
        log_std = log_std_flat.reshape(batch_size, self.n_assets)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class PermutationInvariantCritic(ContinuousCritic):
    """
    Per-stock Q-value estimation + mean-pooling.
    Each stock gets: concat(stock_features, portfolio_features, action_score) → Q.
    Final Q = mean over all stocks.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_assets: int,
        n_stock_features: int,
        hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        self.portfolio_dim = max(
            int(observation_space.shape[0] - (self.n_assets * self.n_stock_features)), 0
        )

        in_dim = self.n_stock_features + self.portfolio_dim + 1  # +1 for action score
        self.q_net1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.q_net2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.qf1_top = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qf2_top = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        stock_obs_len = self.n_assets * self.n_stock_features
        stock_features = obs[:, :stock_obs_len]
        portfolio_features = obs[:, stock_obs_len:] if self.portfolio_dim > 0 else None
        batch_size = obs.shape[0]

        reshaped_stocks = stock_features.reshape(-1, self.n_stock_features)
        reshaped_actions = actions.reshape(-1, 1)

        if self.portfolio_dim > 0:
            expanded_portfolio = portfolio_features.unsqueeze(1).expand(
                batch_size, self.n_assets, self.portfolio_dim
            ).reshape(-1, self.portfolio_dim)
            combined = th.cat([reshaped_stocks, expanded_portfolio, reshaped_actions], dim=1)
        else:
            combined = th.cat([reshaped_stocks, reshaped_actions], dim=1)

        latent_q1 = self.q_net1(combined)
        latent_q2 = self.q_net2(combined)
        pooled_q1 = latent_q1.reshape(batch_size, self.n_assets, -1).mean(dim=1)
        pooled_q2 = latent_q2.reshape(batch_size, self.n_assets, -1).mean(dim=1)
        return self.qf1_top(pooled_q1), self.qf2_top(pooled_q2)


class CrossSectionalSACPolicy(SACPolicy):
    """SAC Policy using PermutationEquivariant Actor + Invariant Critic."""

    def __init__(self, *args, **kwargs):
        self.my_n_assets = kwargs.pop("n_assets")
        self.my_n_stock_features = kwargs.pop("n_stock_features")
        self.my_hidden_dim = kwargs.pop("hidden_dim", 64)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return PermutationEquivariantActor(
            n_assets=self.my_n_assets,
            n_stock_features=self.my_n_stock_features,
            hidden_dim=self.my_hidden_dim,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PermutationInvariantCritic(
            n_assets=self.my_n_assets,
            n_stock_features=self.my_n_stock_features,
            hidden_dim=self.my_hidden_dim,
            **critic_kwargs,
        ).to(self.device)
