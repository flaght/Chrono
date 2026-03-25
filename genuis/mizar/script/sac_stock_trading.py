"""
SAC Stock Trading - 基于SAC算法的截面选股强化学习实现

功能:
- 截面选股：每期从股票池中选出 batch_size 只股票
- 排列等变网络：Actor 为每只股票独立打分
- RankIC奖励: 预测得分与未来收益的相关性

不依赖任何自定义库,仅使用标准PyTorch/numpy/pandas/gymnasium
"""

import os
import json
import random
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Signal模块 - 信号处理和权重计算
# =============================================================================


def rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """Spearman-style rank IC."""
    s = pd.Series(scores).rank()
    r = pd.Series(returns).rank()
    corr = s.corr(r)
    return float(corr) if not np.isnan(corr) else 0.0



# =============================================================================
# 2. TradingEnv - 交易环境
# =============================================================================

class TradingEnv(gym.Env):
    """
    截面选股交易环境
    
    观测空间：[batch_size * n_features]
        - 股票特征：batch_size 只股票 × n_features 个特征
    
    动作空间：[batch_size] 每只股票原始得分 [-1, 1]
    
    奖励：RankIC
        
    数据采样：每步随机从全量数据中抽取 batch_size 只股票（有放回抽样）
    
    重置机制：每 steps_per_reset 步自动重置一次，开始新的训练回合
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        batch_size: int,
        steps_per_reset: int = 200,  # 默认 200 步重置一次
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.df = df.copy().reset_index(drop=True)
        self.features = list(features)
        self.n_features = len(self.features)
        self.total_rows = len(self.df)
        
        # 参数
        self.batch_size = int(batch_size)
        self.steps_per_reset = int(steps_per_reset)  # 重命名：更直观地表达"每多少步重置一次"
        
        # 数据验证
        if "nxt1_ret" not in self.df.columns:
            raise ValueError("data must contain 'nxt1_ret'")
        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"missing feature column: {f}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.total_rows < self.batch_size:
            raise ValueError(
                f"data has {self.total_rows} rows but batch_size={self.batch_size}"
            )
        
        # 预加载数据
        self._all_features = self.df[self.features].values.astype(np.float32)
        self._all_returns = self.df["nxt1_ret"].values.astype(np.float32)
        
        self.has_code = "code" in self.df.columns
        if self.has_code:
            self._all_codes = self.df["code"].values
        else:
            self._all_codes = None
                
        # 观测空间维度
        self.n_state_features = self.batch_size * self.n_features
                
        # 定义空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.batch_size,),
            dtype=np.float32,
        )
                
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32,
        )
                
        # 状态
        self.episode_step_count = 0
        self.terminal = False
                
        # 累计指标
        self.total_rank_ic = 0.0
                
        # 当前 batch 缓存
        self._cur_features = np.zeros((self.batch_size, self.n_features), dtype=np.float32)
        self._cur_returns = np.zeros(self.batch_size, dtype=np.float32)
                
        # 随机种子
        self.np_random = np.random.RandomState(seed)
                
        # Gymnasium 兼容
        self.spec = None
        self.metadata = {"render_modes": []}
    
    def _load_batch(self):
        """随机采样 batch 数据：从全量数据中有放回地抽取 batch_size 只股票"""
        # 随机选择 batch_size 个索引（有放回抽样）
        batch_indices = self.np_random.choice(
            self.total_rows,
            size=self.batch_size,
            replace=True  # 有放回，允许重复
        )
            
        self._cur_features[:] = self._all_features[batch_indices]
        self._cur_returns[:] = self._all_returns[batch_indices]
    
    def _build_observation(self) -> np.ndarray:
        """构建观测向量"""
        return self._cur_features.flatten().astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
            
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        self.total_rank_ic = 0.0
        self.episode_step_count = 0
        self.terminal = False
            
        # 随机采样第一个 batch
        self._load_batch()
        obs = self._build_observation()
            
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行动作：每步都会随机采样新的 batch 数据"""
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = action.astype(np.float32).flatten()
            
        # 维度对齐
        if action.size < self.batch_size:
            action = np.pad(action, (0, self.batch_size - action.size))
        elif action.size > self.batch_size:
            action = action[:self.batch_size]
            
        # 已终止
        if self.terminal:
            return self._build_observation(), 0.0, True, False, {}
            
        # 计算 RankIC
        ic_value = rank_ic(action, self._cur_returns) if action.size > 1 else 0.0
        reward = ic_value
            
        self.total_rank_ic += ic_value
            
        self.episode_step_count += 1
            
        # 终止条件：达到重置步数
        if self.episode_step_count >= self.steps_per_reset:
            self.terminal = True
            
        # 随机采样下一个 batch（不再使用 cursor）
        if not self.terminal:
            self._load_batch()
            
        obs = self._build_observation()
            
        if self.terminal:
            return obs, reward, True, False, {}
            
        info = {
            "episode_step": int(self.episode_step_count),
            "rank_ic": float(ic_value),
            "reward": float(reward),
            "total_rank_ic": float(self.total_rank_ic),
        }
        return obs, reward, False, False, info
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
    
    def close(self):
        """关闭环境"""
        pass
    
    def __repr__(self) -> str:
        return (
            f"TradingEnv("
            f"total_rows={self.total_rows}, "
            f"batch_size={self.batch_size}, "
            f"steps_per_reset={self.steps_per_reset}, "
            f"n_features={self.n_features}, "
            f"state_dim={self.n_state_features}, "
            f"sample_mode='random')"
        )


# =============================================================================
# 3. 网络结构 - Actor & Critic
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class StockActor(nn.Module):
    """
    排列等变Actor网络
    
    为每只股票独立打分,共享网络参数
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_assets: int,
        n_stock_features: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        
        # 共享打分网络
        self.stock_net = nn.Sequential(
            nn.Linear(n_stock_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_net = nn.Linear(hidden_dim, 1)
        self.log_std_net = nn.Linear(hidden_dim, 1)
        
        # 动作维度
        self.action_dim = action_space.shape[0]
    
    def score_assets(
        self,
        stock_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理: 为任意数量的股票打分
        
        Args:
            stock_features: (N_assets, n_stock_features)
        
        Returns:
            scores: (N_assets,) in [-1, 1]
        """
        latent = self.stock_net(stock_features)
        mu = self.mu_net(latent).squeeze(-1)
        return torch.tanh(mu)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            obs: (batch_size, n_assets * n_stock_features)
                 n_assets 可以是任意值（动态计算）
        
        Returns:
            mean_actions: (batch_size, n_assets)
            log_std: (batch_size, n_assets)
        """
        batch_size = obs.shape[0]
        # 动态计算 n_assets：从观测维度推断
        n_assets = obs.shape[1] // self.n_stock_features
        
        reshaped_stocks = obs.reshape(-1, self.n_stock_features)
        
        latent = self.stock_net(reshaped_stocks)
        mu_flat = self.mu_net(latent)
        log_std_flat = self.log_std_net(latent)
        
        # 使用动态计算的 n_assets
        mean_actions = mu_flat.reshape(batch_size, n_assets)
        log_std = log_std_flat.reshape(batch_size, n_assets)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean_actions, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """获取动作"""
        mean_actions, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean_actions)
        
        std = log_std.exp()
        normal = Normal(mean_actions, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        return action


class StockCritic(nn.Module):
    """
    排列不变Critic网络
    
    为每只股票独立估计Q値,然名mean-pooling
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_assets: int,
        n_stock_features: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_stock_features = n_stock_features
        
        # Q网络输入: stock_features + action
        in_dim = self.n_stock_features + 1
        
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
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: (batch_size, n_assets * n_stock_features)
                 n_assets 可以是任意值（动态计算）
            actions: (batch_size, n_assets)
        
        Returns:
            q1, q2: (batch_size, 1)
        """
        batch_size = obs.shape[0]
        # 动态计算 n_assets：从观测维度推断
        n_assets = obs.shape[1] // self.n_stock_features
        
        reshaped_stocks = obs.reshape(-1, self.n_stock_features)
        reshaped_actions = actions.reshape(-1, 1)
        
        combined = torch.cat([reshaped_stocks, reshaped_actions], dim=1)
        
        latent_q1 = self.q_net1(combined)
        latent_q2 = self.q_net2(combined)
        
        # Mean-pooling over assets（使用动态计算的 n_assets）
        pooled_q1 = latent_q1.reshape(batch_size, n_assets, -1).mean(dim=1)
        pooled_q2 = latent_q2.reshape(batch_size, n_assets, -1).mean(dim=1)
        
        return self.qf1_top(pooled_q1), self.qf2_top(pooled_q2)


# =============================================================================
# 4. 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
    ):
        self.buffer_size = buffer_size
        self.device = device
        
        self.obs_shape = observation_space.shape
        self.action_shape = action_space.shape
        
        self.observations = np.zeros((buffer_size,) + self.obs_shape, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + self.action_shape, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """添加经验"""
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        data = {
            "observations": self.observations[batch_inds],
            "next_observations": self.next_observations[batch_inds],
            "actions": self.actions[batch_inds],
            "rewards": self.rewards[batch_inds],
            "dones": self.dones[batch_inds],
        }
        
        return {k: torch.as_tensor(v, device=self.device) for k, v in data.items()}
    
    def __len__(self):
        return self.buffer_size if self.full else self.pos


# =============================================================================
# 5. SAC算法主类
# =============================================================================

class SAC:
    """
    Soft Actor-Critic (SAC) 算法实现
    
    支持自定义排列等变Actor和排列不变Critic
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str = "auto",
        target_update_interval: int = 1,
        target_entropy: Optional[float] = None,
        hidden_dim: int = 64,
        device: str = "auto",
        verbose: int = 0,
    ):
        self.env = env
        self.verbose = verbose
        
        # 自动选择设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.verbose > 0:
            logger.info(f"Using device: {self.device}")
        
        # 超参数
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        
        # 获取环境信息
        self.observation_space = env.observation_space
        self.action_space = env.action_space
                
        # 使用自定义排列等变/不变网络
        n_assets = env.batch_size if hasattr(env, 'batch_size') else env.action_space.shape[0]
        n_features = env.n_features if hasattr(env, 'n_features') else env.observation_space.shape[0] // n_assets
                
        self.actor = StockActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_assets=n_assets,
            n_stock_features=n_features,
            hidden_dim=hidden_dim,
        ).to(self.device)
                
        self.critic = StockCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_assets=n_assets,
            n_stock_features=n_features,
            hidden_dim=hidden_dim,
        ).to(self.device)
                
        self.critic_target = StockCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_assets=n_assets,
            n_stock_features=n_features,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 熵系数
        self.ent_coef = ent_coef
        if ent_coef == "auto":
            if target_entropy is None:
                target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
            self.target_entropy = target_entropy
            self.log_ent_coef = torch.zeros(1, requires_grad=True, device=self.device)
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=learning_rate)
            self.ent_coef_tensor = self.log_ent_coef.exp().detach()
        else:
            self.ent_coef_tensor = torch.tensor(float(ent_coef), device=self.device)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
        )
        
        # 训练状态
        self.num_timesteps = 0
        self._n_updates = 0
        
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """预测动作"""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            
            action = self.actor.get_action(obs_tensor, deterministic=deterministic)
            
            return action.cpu().numpy().flatten()
    
    def _update_critic(self, data: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """更新Critic"""
        obs = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"].unsqueeze(1)
        next_obs = data["next_observations"]
        dones = data["dones"].unsqueeze(1)
        
        with torch.no_grad():
            # 下一状态的动作和 log_prob
            next_actions = self.actor.get_action(next_obs, deterministic=False)
                    
            # 计算目标 Q 值
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(obs, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), current_q1.mean().item()
    
    def _update_actor(self, obs: torch.Tensor) -> Tuple[float, float]:
        """更新 Actor"""
        actions = self.actor.get_action(obs, deterministic=False)
            
        q1, q2 = self.critic(obs, actions)
        q = torch.min(q1, q2)
            
        actor_loss = -q.mean()
            
        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
            
        return actor_loss.item(), q.mean().item()
    
    def _update_entropy(self, obs: torch.Tensor) -> Tuple[float, float]:
        """更新熵系数 (如果使用 auto)"""
        if self.ent_coef != "auto":
            return 0.0, self.ent_coef_tensor.item()
            
        with torch.no_grad():
            actions = self.actor.get_action(obs, deterministic=False)
        
        # 这里简化处理,实际应该计算log_prob
        ent_coef_loss = -self.log_ent_coef * self.target_entropy
        
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
        self.ent_coef_tensor = self.log_ent_coef.exp().detach()
        
        return ent_coef_loss.item(), self.ent_coef_tensor.item()
    
    def train_step(self) -> Dict[str, float]:
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样
        data = self.replay_buffer.sample(self.batch_size)
        
        # 更新Critic
        critic_loss, current_q = self._update_critic(data)
        
        # 更新Actor (延迟更新)
        if self._n_updates % self.target_update_interval == 0:
            actor_loss, actor_q = self._update_actor(data["observations"])
            ent_coef_loss, ent_coef = self._update_entropy(data["observations"])
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = 0.0
            actor_q = 0.0
            ent_coef_loss = 0.0
            ent_coef = self.ent_coef_tensor.item()
        
        self._n_updates += 1
        
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "current_q": current_q,
            "actor_q": actor_q,
            "ent_coef": ent_coef,
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
    ) -> "SAC":
        """训练模型"""
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for timestep in range(total_timesteps):
            # 选择动作
            if timestep < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.predict(obs, deterministic=False)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验
            self.replay_buffer.add(obs, next_obs, action, reward, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1
            
            # 训练
            if timestep >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    train_info = self.train_step()
                
                # 日志
                if log_interval > 0 and timestep % log_interval == 0 and train_info:
                    logger.info(
                        f"Step {timestep}/{total_timesteps} | "
                        f"Reward: {reward:.4f} | "
                        f"Critic Loss: {train_info.get('critic_loss', 0):.4f} | "
                        f"Actor Loss: {train_info.get('actor_loss', 0):.4f}"
                    )
            
            # 回调
            if callback is not None:
                callback.on_step()
            
            # 回合结束
            if done:
                if self.verbose > 0:
                    avg_reward = episode_reward / episode_length if episode_length > 0 else 0.0
                    logger.info(
                        f"Episode finished: step={self.num_timesteps}, "
                        f"length={episode_length}, "
                        f"total_reward={episode_reward:.4f}, "
                        f"avg_reward={avg_reward:.4f}"
                    )
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
        
        return self
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
        }, os.path.join(path, "model.pt"))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.num_timesteps = checkpoint["num_timesteps"]
        logger.info(f"Model loaded from {path}")


# =============================================================================
# 6. 训练和预测函数
# =============================================================================

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    env_config: Dict[str, Any],
    sac_config: Dict[str, Any],
    output_dir: str,
    total_timesteps: int,
    eval_n_episodes: int = 5,
    verbose: int = 1,
) -> Tuple[SAC, Dict[str, Any]]:
    """
    训练 SAC 模型
    
    Args:
        train_df: 训练数据
        val_df: 验证数据
        features: 特征列表
        env_config: 环境配置
        sac_config: SAC 算法配置
        output_dir: 输出目录
        total_timesteps: 总训练步数
        eval_n_episodes: 评估回合数
        verbose: 日志级别
    
    Returns:
        model: 训练好的模型
        training_info: 训练信息
    """
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建训练环境
    train_env = TradingEnv(
        df=train_df,
        features=features,
        **env_config
    )
    
    # 创建验证环境
    val_env = TradingEnv(
        df=val_df,
        features=features,
        **env_config
    )
    
    logger.info(f"训练环境: {train_env}")
    logger.info(f"验证环境: {val_env}")
    logger.info(f"动作空间: {train_env.action_space}")
    logger.info(f"观测空间: {train_env.observation_space}")
    
    # 创建模型
    model = SAC(
        env=train_env,
        verbose=verbose,
        **sac_config
    )
    
    # 保存配置
    config_to_save = {
        'env_config': env_config,
        'sac_config': sac_config,
        'features': features,
        'total_timesteps': total_timesteps,
        'eval_n_episodes': eval_n_episodes,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'training_start': datetime.now().isoformat(),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)
    
    # 训练
    logger.info(f"开始训练 SAC 截面选股模型...")
    logger.info(f"  数据行数 (训练): {len(train_df)}")
    logger.info(f"  数据行数 (验证): {len(val_df)}")
    logger.info(f"  总步数：{total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100 if verbose > 0 else 0,
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    
    training_info = {
        'model_path': final_model_path,
        'config_path': config_path,
        'output_dir': output_dir,
        'total_timesteps': total_timesteps,
    }
    
    logger.info(f"训练完成！最终模型: {final_model_path}")
    
    return model, training_info


def predict_test_set(
    model_path: str,
    config_path: str,
    test_df: pd.DataFrame,
    output_path: Optional[str] = None,
    deterministic: bool = True,
    save_stock_scores: bool = True,
) -> pd.DataFrame:
    """
    使用训练好的模型进行预测，并按日期计算每日 RankIC
    
    Args:
        model_path: 模型路径
        config_path: 配置文件路径
        test_df: 测试数据（必须包含 'trade_time', 'code', 'nxt1_ret' 列）
        output_path: 输出文件路径
        deterministic: 是否确定性预测
        save_stock_scores: 是否保存每只股票的预测得分
    
    Returns:
        daily_rank_ic: 按日期统计的每日 RankIC DataFrame
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    env_config = config['env_config']
    features = config['features']
    
    # 检查是否有 code 和 trade_time 列
    has_code = 'code' in test_df.columns
    has_time = 'trade_time' in test_df.columns
    has_return = 'nxt1_ret' in test_df.columns
    
    if not has_time or not has_code:
        raise ValueError("测试数据必须包含 'trade_time' 和 'code' 列")
    if not has_return:
        raise ValueError("测试数据必须包含 'nxt1_ret' 列")
    
    # 加载模型
    env = TradingEnv(
        df=test_df,
        features=features,
        **env_config
    )
    
    model = SAC(
        env=env,
        **sac_config
    )
    model.load(model_path)
    
    # 按日期分组预测（使用动态 n_assets，支持任意股票数量）
    unique_dates = test_df['trade_time'].unique()
    stock_predictions = []  # 存储每只股票的预测得分
    daily_results = []
    
    for date in unique_dates:
        date_data = test_df[test_df['trade_time'] == date].copy()
        if len(date_data) == 0:
            continue
        
        # 构建观测（展平为一维向量）
        obs = date_data[features].values.flatten().astype(np.float32)
        
        # 预测（网络自动适配股票数量）
        action = model.predict(obs, deterministic=deterministic)
        
        # 获取真实收益
        returns = date_data['nxt1_ret'].values
        
        # 计算当日 RankIC
        daily_ic = rank_ic(action, returns)
        
        # 记录每只股票的预测
        for i, (_, row) in enumerate(date_data.iterrows()):
            stock_predictions.append({
                'trade_time': date,
                'code': row['code'],
                'score': float(action[i]),
                'nxt1_ret': float(returns[i]),
            })
        
        daily_results.append({
            'trade_time': date,
            'rank_ic': daily_ic,
            'n_stocks': len(date_data),
        })
    
    # 保存股票预测得分
    if save_stock_scores and stock_predictions:
        stock_pred_df = pd.DataFrame(stock_predictions)
        if output_path:
            stock_output_path = output_path.replace('.csv', '_stock_scores.csv')
            os.makedirs(os.path.dirname(stock_output_path), exist_ok=True)
            stock_pred_df.to_csv(stock_output_path, index=False)
            logger.info(f"股票预测得分保存至：{stock_output_path}")
    
    # 构建每日 RankIC 结果
    daily_rank_ic = pd.DataFrame(daily_results)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        daily_output_path = output_path.replace('.csv', '_daily_rankic.csv')
        daily_rank_ic.to_csv(daily_output_path, index=False)
        logger.info(f"每日 RankIC 保存至：{daily_output_path}")
    
    # 打印统计信息
    logger.info(f"预测完成，共 {len(daily_rank_ic)} 个交易日")
    if len(daily_rank_ic) > 0:
        mean_ic = daily_rank_ic['rank_ic'].mean()
        std_ic = daily_rank_ic['rank_ic'].std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0  # IC_IR：IC 均值/IC 标准差，衡量 IC 稳定性
        
        logger.info(f"平均每日 RankIC: {mean_ic:.6f} ± {std_ic:.6f}")
        logger.info(f"IC_IR (稳定性): {ic_ir:.4f}")
        logger.info(f"最大 RankIC: {daily_rank_ic['rank_ic'].max():.6f}")
        logger.info(f"最小 RankIC: {daily_rank_ic['rank_ic'].min():.6f}")
        logger.info(f"RankIC > 0 的天数占比：{(daily_rank_ic['rank_ic'] > 0).sum() / len(daily_rank_ic):.2%}")
    
    return daily_rank_ic


# =============================================================================
# 7. 主程序入口
# =============================================================================

if __name__ == "__main__":
    """
    示例运行代码
    
    创建模拟数据进行训练和预测演示
    """
    logger.info("=" * 60)
    logger.info("SAC Stock Trading - 示例运行")
    logger.info("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
   
    base_path = './rl/records'
    # features = ['LLM_000006', 'LLM_000016', 'LLM_000081','LLM_000085', 'LLM_000108', 'LLM_000119', 'LLM_000137', 'LLM_000143']
    # ret_name = 'abret_market'
    
    train_data = pd.read_feather(os.path.join(base_path,'eicso0','ashare','rl','1010101301',
                                              "train_data.feather"))
    
    features = train_data.columns.to_list()[2:-8]
    ret_name = 'abret_market'

    val_data = pd.read_feather(os.path.join(base_path,'eicso0','ashare','rl','1010101301',
                                              "val_data.feather"))
    
    test_data = pd.read_feather(os.path.join(base_path,'eicso0','ashare','rl','1010101301',
                                              "test_data.feather"))
    
    train_data = train_data[['trade_time','code',ret_name] + features].rename(columns={ret_name:'nxt1_ret'})
    val_data = val_data[['trade_time','code',ret_name] + features].rename(columns={ret_name:'nxt1_ret'})
    test_data = test_data[['trade_time','code',ret_name] + features].rename(columns={ret_name:'nxt1_ret'})
    
    # 环境配置
    env_config = {
        'batch_size': 5000,          # 批次大小：每步随机采样 5000 只股票作为一个截面（类似 mini-batch）
        'steps_per_reset': 300,      # 重置间隔：每 300 步自动重置一次，开始新的训练回合（覆盖更长市场周期）
        'seed': 42,                  # 随机种子：保证实验可复现性
    }
    
    # SAC 算法配置
    sac_config = {
        'learning_rate': 1e-4,           # 学习率：Actor/Critic/熵系数的优化器学习率（从 6e-5 提升以加快收敛）
        'buffer_size': 100000,           # 回放缓冲区容量：存储最多 100000 条经验 (obs, action, reward, next_obs, done)
        'learning_starts': 10000,        # 开始学习步数：先收集 10000 条经验后再开始训练（更充分的 warm-up）
        'batch_size': 256,               # 训练批次大小：每次从缓冲区采样 256 条经验进行梯度更新
        'tau': 0.005,                    # 软更新系数：目标网络更新时的平滑系数 (θ_target ← τ·θ + (1-τ)·θ_target)
        'gamma': 0.97,                   # 折扣因子：未来奖励的折现率（从 0.99 降低以平衡短中期收益）
        'train_freq': 1,                 # 训练频率：每执行 1 步就训练一次（on-policy 风格）
        'gradient_steps': 1,             # 梯度更新步数：每次执行 train_step() 时做 1 次梯度下降
        'ent_coef': 'auto',              # 熵系数：'auto' 表示自动调节探索力度（最大化策略熵）
        'target_update_interval': 1,     # 目标网络更新间隔：每 1 步更新一次目标网络（延迟更新策略）
        'hidden_dim': 128,               # 隐藏层维度：Actor 和 Critic 网络的隐藏层神经元数量（从 64 增加以增强表达能力）
    }
    
    # 训练控制参数
    output_dir = './sac_stock_output'   # 输出目录：保存模型、配置文件和日志的根目录
    total_timesteps = 10000            # 总训练步数：整个训练过程执行 100000 个 timestep（从 3000 大幅增加以充分训练）
    eval_n_episodes = 20                # 评估回合数：验证阶段运行 20 个 episode 评估模型性能
    
    # 训练模型
    logger.info("\n开始训练...")
    model, training_info = train_model(
        train_df=train_data,
        val_df=val_data,
        features=features,
        env_config=env_config,
        sac_config=sac_config,
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        eval_n_episodes=eval_n_episodes,
        verbose=1,
    )
    
    # 测试预测
    logger.info("\n开始预测...")
   
    
    signals_df = predict_test_set(
        model_path=training_info['model_path'],
        config_path=training_info['config_path'],
        test_df=test_data,
        output_path=os.path.join(output_dir, 'predictions.csv'),
        deterministic=True,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("示例运行完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
