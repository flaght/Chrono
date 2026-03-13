import logging, pdb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gym import spaces

from lib.rl023.signal import (
    Config,
    rank_ic,
)
from kdutils.logger import logger


class TradingEnv:

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        subset_size: int,
        episode_len: int,
        seed: Optional[int] = None,
        reward_scale: float = 1000.0,
        signal_config: Optional[Config] = None,
        ic_scale: float = 1.0,
        negative_ic_penalty: float = 0.0,
        reward_mode: str = "rank_ic",
        reward_top_k: int = 20,
        enable_step_logging: bool = True,
        log_every_n_steps: int = 200,
        warn_turnover_threshold: float = 0.80,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.features = list(features)
        self.n_features = len(self.features)
        self.total_rows = len(self.df)

        self.subset_size = int(subset_size)
        self.episode_len = int(episode_len)
        self.reward_scale = float(reward_scale)
        self.ic_scale = float(ic_scale)

        self.negative_ic_penalty = float(negative_ic_penalty)
        self.reward_mode = str(reward_mode)
        self.reward_top_k = int(reward_top_k)

        self.enable_step_logging = bool(enable_step_logging)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.warn_turnover_threshold = float(warn_turnover_threshold)

        self.signal_config = signal_config

        self._all_features = self.df[self.features].values.astype(np.float32)
        self._all_returns = self.df["nxt1_ret"].values.astype(np.float32)

        self.max_steps_in_data = self.total_rows // self.subset_size

        self.n_state_features = self.subset_size * self.n_features

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.subset_size, ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_features, ),
            dtype=np.float32,
        )

        self.cursor = 0
        self.episode_step_count = 0
        self.terminal = False

        self.total_rank_ic = 0.0
        self.total_reward_raw = 0.0
        self.total_rebalances = 0
        self.nan_or_inf_events = 0

        self._cur_features = np.zeros((self.subset_size, self.n_features),
                                      dtype=np.float32)
        self._cur_returns = np.zeros(self.subset_size, dtype=np.float32)

        self.np_random = np.random.RandomState(seed)

        # SB3 兼容
        self.spec = None
        self.metadata = {"render_modes": []}

    def _load_batch(self):
        batch_indices = self.np_random.choice(
            self.total_rows,
            size=self.subset_size,
            replace=True,
        )
        self._cur_features[:] = self._all_features[batch_indices]
        self._cur_returns[:] = self._all_returns[batch_indices]

        # end = self.cursor + self.subset_size
        # if end > self.total_rows:
        #     avail = self.total_rows - self.cursor
        #     self._cur_features[:avail] = self._all_features[self.cursor:self.
        #                                                     total_rows]
        #     self._cur_features[avail:] = 0.0
        #     self._cur_returns[:avail] = self._all_returns[self.cursor:self.
        #                                                   total_rows]
        #     self._cur_returns[avail:] = 0.0
        # else:
        #     self._cur_features[:] = self._all_features[self.cursor:end]
        #     self._cur_returns[:] = self._all_returns[self.cursor:end]

    def _build_observation(self) -> np.ndarray:

        # turnover = 0.0 if self.episode_step_count <= 0 else float(
        #     self.last_turnover)
        # hhi = float(np.sum(self.prev_weights**2))
        # holding_ratio = float(
        #     np.sum(self.prev_weights > 1e-6) / max(self.subset_size, 1))
        # portfolio_features = np.array([turnover, hhi, holding_ratio],
        #                               dtype=np.float32)
        return np.concatenate([self._cur_features.flatten()
                               ]).astype(np.float32)

    def reset(self, start_row: Optional[int] = None) -> np.ndarray:
        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)
        self.last_turnover = 0.0
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_rank_ic = 0.0
        self.total_reward_raw = 0.0
        self.total_portfolio_return = 0.0
        self.total_rebalances = 0
        self.nan_or_inf_events = 0
        self.episode_step_count = 0
        self.terminal = False

        self.cursor = 0

        self._load_batch()
        obs = self._build_observation()
        return obs

    def _calculate_topk_profit_reward(self, action: np.ndarray, top_k: int) -> Tuple[float, Dict[str, float]]:
        """
        Directly rewards the model based on the absolute return of its TOP K choices.
        """
        # Get indices of the highest K scored coins
        candidate_indices = np.argsort(action)[-top_k:]
        
        # Calculate actual absolute mean return of these held coins
        top_k_returns = self._cur_returns[candidate_indices]
        portfolio_abs_return = np.mean(top_k_returns)
        
        # Base reward
        reward = portfolio_abs_return
        
        # Extra penalty for losing money (force risk-aversion)
        if portfolio_abs_return < 0:
            reward *= 2.0  
            
        return float(reward), {"topk_abs_return": portfolio_abs_return}

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = action.astype(np.float32).flatten()

        if action.size < self.subset_size:
            action = np.pad(action, (0, self.subset_size - action.size))
        elif action.size > self.subset_size:
            action = action[:self.subset_size]

        if not np.all(np.isfinite(action)):
            self.nan_or_inf_events += 1
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.terminal:
            return self._build_observation(), 0.0, True, {
                "episode_step": int(self.episode_step_count),
                "cursor": int(self.cursor),
                "already_terminal": True,
            }

        # ==== 新增：多模式 Reward 计算路由 ====
        info_metrics = {}
        if self.reward_mode == "topk_profit":
            # 策略 B: 专注于 Top-K 绝对收益
            reward_base, topk_info = self._calculate_topk_profit_reward(action, top_k=self.reward_top_k)
            reward = reward_base * self.ic_scale  # 借用 ic_scale 作为调整系数
            ic_value = rank_ic(action, self._cur_returns) if action.size > 1 else 0.0 # 仅作记录，不参与奖惩
            info_metrics.update(topk_info)
            
        else:
            # 默认模式: 全截面 Spearman Rank IC
            ic_value = rank_ic(action, self._cur_returns) if action.size > 1 else 0.0
            reward = self.ic_scale * ic_value
            if self.negative_ic_penalty > 0 and ic_value < 0:
                reward -= self.negative_ic_penalty * abs(ic_value)

        should_rebalance = False
        # old_weights = np.zeros(self.subset_size, dtype=np.float32)
        # new_weights = np.zeros(self.subset_size, dtype=np.float32)

        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)
        self.last_turnover = 0.0

        self.total_rank_ic += ic_value
        self.total_rebalances += int(should_rebalance)
        self.total_reward_raw += reward

        self.episode_step_count += 1

        if self.episode_step_count >= self.episode_len:
            self.terminal = True

        if not self.terminal:
            self._load_batch()

        obs = self._build_observation()
        reward_scaled = reward * self.reward_scale
        done = bool(self.terminal)

        # weight_hhi = float(np.sum(new_weights**2))
        # active_positions = int(np.sum(new_weights > 1e-8))

        finite_ok = (np.isfinite(reward_scaled) and np.isfinite(ic_value))
        if not finite_ok:
            self.nan_or_inf_events += 1

        info = {
            "episode_step": int(self.episode_step_count),
            "cursor": int(self.cursor),
            "rank_ic": float(ic_value),
            "reward_raw": float(reward),
            "reward_scaled": float(reward_scaled),
            "total_rank_ic": float(self.total_rank_ic),
            "rebalanced": bool(should_rebalance),
            "finite_ok": bool(finite_ok),
        }

        if self.enable_step_logging and (done or
                                         (self.episode_step_count %
                                          self.log_every_n_steps == 0)):
            logger.info(
                "TradingEnv step=%d done=%s ic=%.6f reward=%.6f  finite_ok=%s",
                self.episode_step_count, done, ic_value, reward, finite_ok)

        return obs, reward_scaled, done, info

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

    def close(self):
        pass

    def __repr__(self) -> str:
        return ("TradingEnv("
                f"total_rows={self.total_rows}, "
                f"subset_size={self.subset_size}, "
                f"episode_len={self.episode_len}, "
                f"n_features={self.n_features}, "
                f"state_dim={self.n_state_features}, "
                f"reward_scale={self.reward_scale})")
