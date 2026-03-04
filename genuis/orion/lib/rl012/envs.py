import pdb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from gym import spaces

from lib.rl012.signal import *

class TradingEnv:
    def __init__(self,
        df: pd.DataFrame,
        features: List[str],
        subset_size: int,
        episode_len: int,
        seed: Optional[int] = None,
        reward_scale: float = 1000.0,
        signal_config: Optional[Config] = None,
        ic_scale: float = 1.0,
        negative_ic_penalty: float = 0.0,
        use_turnover_proxy: bool = False,
        turnover_proxy_coef: float = 0.0):
        
        self.df = df.copy().reset_index(drop=True)
        self.features = list(features)
        self.n_features = len(self.features)
        self.total_rows = len(self.df)
        
        # ── 参数 ──
        self.subset_size = int(subset_size)
        self.episode_len = int(episode_len)
        self.reward_scale = float(reward_scale)
        self.ic_scale = float(ic_scale)
        self.negative_ic_penalty = float(negative_ic_penalty)
        self.use_turnover_proxy = bool(use_turnover_proxy)
        self.turnover_proxy_coef = float(turnover_proxy_coef)
        
        if signal_config is None:
            raise ValueError("signal_config must be provided")
        self.signal_config = signal_config
        
        if "nxt1_ret" not in self.df.columns:
            raise ValueError("data must contain 'nxt1_ret'")
        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"missing feature column: {f}")
        if self.subset_size <= 0:
            raise ValueError(f"subset_size must be > 0, got {self.subset_size}")
        if self.total_rows < self.subset_size:
            raise ValueError(
                f"data has {self.total_rows} rows but subset_size={self.subset_size}"
            )
            
        self._all_features = self.df[self.features].values.astype(np.float32)
        self._all_returns = self.df["nxt1_ret"].values.astype(np.float32)
        
        
        self.has_code = "code" in self.df.columns
        if self.has_code:
            self._all_codes = self.df["code"].values
        else:
            self._all_codes = None

        self.max_steps_in_data = self.total_rows // self.subset_size
        
        self.n_portfolio_features = 3  # [turnover, hhi, holding_ratio]
        self.n_state_features = self.subset_size * self.n_features + self.n_portfolio_features
        
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.subset_size,),
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32,
        )
        
        # ── 状态 ──
        self.cursor = 0              # 当前读取位置
        self.episode_step_count = 0
        self.terminal = False
        self.last_turnover = 0.0
        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)

        # 累计指标
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_rank_ic = 0.0

        # 当前 batch 缓存
        self._cur_features = np.zeros((self.subset_size, self.n_features), dtype=np.float32)
        self._cur_returns = np.zeros(self.subset_size, dtype=np.float32)

        # ── 随机 ──
        self.np_random = np.random.RandomState(seed)

        # SB3 兼容
        self.spec = None
        self.metadata = {"render_modes": []}
        
    def _load_batch(self):
        end = self.cursor + self.subset_size
        if end > self.total_rows:
            # 数据不够一个完整 batch，截断 + 零填充
            avail = self.total_rows - self.cursor
            self._cur_features[:avail] = self._all_features[self.cursor:self.total_rows]
            self._cur_features[avail:] = 0.0
            self._cur_returns[:avail] = self._all_returns[self.cursor:self.total_rows]
            self._cur_returns[avail:] = 0.0
        else:
            self._cur_features[:] = self._all_features[self.cursor:end]
            self._cur_returns[:] = self._all_returns[self.cursor:end]
            
    def _build_observation(self) -> np.ndarray:
        turnover = 0.0 if self.episode_step_count <= 0 else float(self.last_turnover)
        hhi = float(np.sum(self.prev_weights ** 2))
        holding_ratio = float(np.sum(self.prev_weights > 1e-6) / max(self.subset_size, 1))
        portfolio_features = np.array([turnover, hhi, holding_ratio], dtype=np.float32)
        return np.concatenate([self._cur_features.flatten(), portfolio_features]).astype(np.float32)
        
    def reset(self, start_row: Optional[int] = None) -> np.ndarray:
        
        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)
        self.last_turnover = 0.0
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_rank_ic = 0.0
        self.episode_step_count = 0
        self.terminal = False
        
        # 确定随机起点
        # 需要留够 episode_len 个 batch 的空间
        rows_needed = self.episode_len * self.subset_size
        max_start = max(0, self.total_rows - rows_needed)
        
        if start_row is not None:
            self.cursor = int(np.clip(start_row, 0, max_start))
        else:
            self.cursor = int(self.np_random.randint(0, max(1, max_start + 1)))
            
        # 对齐到 subset_size 的整数倍（可选，保证 batch 完整）
        self.cursor = (self.cursor // self.subset_size) * self.subset_size

        self._load_batch()
        obs = self._build_observation()

        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = action.astype(np.float32).flatten()
        
        # 维度对齐
        if action.size < self.subset_size:
            action = np.pad(action, (0, self.subset_size - action.size))
        elif action.size > self.subset_size:
            action = action[:self.subset_size]
            
        # 已终止
        if self.terminal:
            return self._build_observation(), 0.0, True, {}
        
        ## 计算RankIC
        ic_value = rank_ic(action, self._cur_returns) if action.size > 1 else 0.0
        
        reward = self.ic_scale * ic_value
        
        ### 惩罚
        if self.negative_ic_penalty > 0 and ic_value < 0:
            reward -= self.negative_ic_penalty * abs(ic_value)
            
        ### 换手率
        old_weights = self.prev_weights.copy()
        should_rebalance = (
            self.signal_config.rebalance_window <= 1
            or (self.episode_step_count % self.signal_config.rebalance_window == 0)
        )
        if should_rebalance and action.size > 0:
            new_weights = scores_to_weights(action, self.signal_config)
        else:
            new_weights = old_weights.copy()
        
        
        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights, self.signal_config)
        
        
        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights, self.signal_config)

        self.last_turnover = turnover
        self.total_turnover += turnover
        self.total_cost += cost
        self.total_rank_ic += ic_value

        if self.use_turnover_proxy and self.turnover_proxy_coef > 0:
            reward -= self.turnover_proxy_coef * turnover
        if self.signal_config.turnover_penalty > 0:
            reward -= self.signal_config.turnover_penalty * turnover

        self.prev_weights = new_weights
        
        self.episode_step_count += 1
        self.cursor += self.subset_size

        # 终止条件
        if self.episode_step_count >= self.episode_len:
            self.terminal = True
        if self.cursor + self.subset_size > self.total_rows:
            self.terminal = True

        # ── 4. 加载下一个 batch ──
        if not self.terminal:
            self._load_batch()

        obs = self._build_observation()
        reward_scaled = reward * self.reward_scale

        if self.terminal:
            return obs, reward_scaled, True, {}

        info = {
            "episode_step": int(self.episode_step_count),
            "cursor": int(self.cursor),
            "rank_ic": float(ic_value),
            "reward_raw": float(reward),
            "reward_scaled": float(reward_scaled),
            "turnover": float(turnover),
            "cost": float(cost),
            "total_rank_ic": float(self.total_rank_ic),
            "total_turnover": float(self.total_turnover),
            "total_cost": float(self.total_cost),
            "rebalanced": bool(should_rebalance),
        }
        return obs, reward_scaled, False, info
    
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

    def close(self):
        pass

    def __repr__(self) -> str:
        return (
            f"TradingEnv("
            f"total_rows={self.total_rows}, "
            f"subset_size={self.subset_size}, "
            f"episode_len={self.episode_len}, "
            f"n_features={self.n_features}, "
            f"state_dim={self.n_state_features}, "
            f"reward_scale={self.reward_scale})"
        )