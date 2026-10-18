import gym
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type
from kichaos.stable3.common.distributions import SquashedDiagGaussianDistribution
from kichaos.stable3.common.policies import BasePolicy
from kichaos.stable3.common.preprocessing import get_action_dim
from kichaos.stable3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from kichaos.stable3.common.type_aliases import Schedule
from kichaos.stable3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, SACPolicy

class ResidualBlock(nn.Module):
    """
    一个残差块（Residual Block），比单纯的 MLP 拥有更好的梯度传播和特征提取能力。
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.relu = nn.GELU()  
        self.fc2 = nn.Linear(dim, dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.layer_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm2(out)
        return self.relu(out + residual)
    
class ResNetActor(BasePolicy):
    """
    加强版 SAC Actor：
    1. 抛弃传统的薄层 MLP，使用带有 LayerNorm 和残差连接的 ResNet 结构。
    2. 加强了对时序或者因子特征非线性组合的挖掘能力，避免梯度消失。
    3. 稳定输出连续的意图 Logits（交由特定设计环境进行分类或概率解析）。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -3,
        normalize_images: bool = True,
        dropout: float = 0.1,
        num_residual_blocks: int = 2,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.log_std_init = log_std_init
        self.dropout = dropout
        self.num_residual_blocks = num_residual_blocks
        self.activation_fn = activation_fn
        
        action_dim = get_action_dim(self.action_space)
        # 1. 投影层：将输入映射到统一维度的隐藏空间
        hidden_dim = net_arch[0] if len(net_arch) > 0 else 256
        self.projection = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 2. 残差提取层
        res_blocks = []
        for _ in range(self.num_residual_blocks):
            res_blocks.append(ResidualBlock(hidden_dim, dropout=self.dropout))
        self.latent_pi = nn.Sequential(*res_blocks)
        # 3. 输出层：分布头
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                log_std_init=self.log_std_init,
                features_extractor=self.features_extractor,
                dropout=self.dropout,
                num_residual_blocks=self.num_residual_blocks,
            )
        )
        return data
    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs)
        x = self.projection(features)
        latent_pi = self.latent_pi(x)
        
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)
    
class ResNetSACPolicy(SACPolicy):
    """
    定制化 SAC Policy，挂载强大的残差网络 Actor，用于输出更稳定的 Softmax 意图 Logits。
    Critic 保持稳定的 MLP 以评估 Q 值。
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        dropout: float = 0.1,
        num_residual_blocks: int = 2,
    ):
        self.dropout = float(dropout)
        self.num_residual_blocks = int(num_residual_blocks)
        if net_arch is None:
            net_arch = [256, 256]
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )
        
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ResNetActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(
            {
                "dropout": self.dropout,
                "num_residual_blocks": self.num_residual_blocks,
            }
        )
        return ResNetActor(**actor_kwargs).to(self.device)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                dropout=self.dropout,
                num_residual_blocks=self.num_residual_blocks,
            )
        )
        return data