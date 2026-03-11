import pdb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gym import spaces

from lib.rl023.signal import (
    Config,
    calculate_portfolio_return,
    calculate_transaction_cost,
    calculate_turnover,
    rank_ic,
    scores_to_weights,
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
        use_turnover_proxy: bool = False,
        turnover_proxy_coef: float = 0.0,
        use_fee_in_reward: bool = True,
        fee_coef: float = 1.0,
        sampling_mode: str = "sequential",  # sequential | random
        action_mode: str = "weights",  # weights | raw_ic
        include_portfolio_state: bool = True,
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
        self.use_turnover_proxy = bool(use_turnover_proxy)
        self.turnover_proxy_coef = float(turnover_proxy_coef)
        self.use_fee_in_reward = bool(use_fee_in_reward)
        self.fee_coef = float(fee_coef)

        self.enable_step_logging = bool(enable_step_logging)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.warn_turnover_threshold = float(warn_turnover_threshold)

        self.sampling_mode = str(sampling_mode).lower()
        self.action_mode = str(action_mode).lower()

        # random 模式下强制关闭持仓状态特征，确保无路径依赖
        self.include_portfolio_state = bool(include_portfolio_state)

        self.signal_config = signal_config

        self._all_features = self.df[self.features].values.astype(np.float32)
        self._all_returns = self.df["nxt1_ret"].values.astype(np.float32)

        self.max_steps_in_data = self.total_rows // self.subset_size

        self.n_portfolio_features = 3 if self.include_portfolio_state else 0
        self.n_state_features = self.subset_size * self.n_features + self.n_portfolio_features

        # --- 强制逻辑约束 ---
        if self.action_mode == "weights" and self.sampling_mode != "sequential":
            raise ValueError(
                "逻辑冲突: `action_mode='weights'` 必须搭配 `sampling_mode='sequential'`。\n"
                "因为计算换手率(turnover)和资金状态(portfolio)依赖于时间序列的物理连续性。")

        if self.sampling_mode == "random" and self.action_mode != "raw_ic":
            raise ValueError(
                "逻辑冲突: `sampling_mode='random'` 必须搭配 `action_mode='raw_ic'`。\n"
                "因为随机抽样打破了时间连续性，无法进行模拟交易，只适合用于纯截面因子的预测打分。")

        action_low = -1.0 if self.action_mode == "raw_ic" else 0.0
        action_high = 1.0
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
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
        self.last_turnover = 0.0
        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)

        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_rank_ic = 0.0
        self.total_reward_raw = 0.0
        self.total_portfolio_return = 0.0
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
        if self.sampling_mode == "random":
            batch_indices = self.np_random.choice(
                self.total_rows,
                size=self.subset_size,
                replace=True,
            )
            self._cur_features[:] = self._all_features[batch_indices]
            self._cur_returns[:] = self._all_returns[batch_indices]
            return

        end = self.cursor + self.subset_size
        if end > self.total_rows:
            avail = self.total_rows - self.cursor
            self._cur_features[:avail] = self._all_features[self.cursor:self.
                                                            total_rows]
            self._cur_features[avail:] = 0.0
            self._cur_returns[:avail] = self._all_returns[self.cursor:self.
                                                          total_rows]
            self._cur_returns[avail:] = 0.0
        else:
            self._cur_features[:] = self._all_features[self.cursor:end]
            self._cur_returns[:] = self._all_returns[self.cursor:end]

    def _build_observation(self) -> np.ndarray:
        if not self.include_portfolio_state:
            return self._cur_features.flatten().astype(np.float32)

        turnover = 0.0 if self.episode_step_count <= 0 else float(
            self.last_turnover)
        hhi = float(np.sum(self.prev_weights**2))
        holding_ratio = float(
            np.sum(self.prev_weights > 1e-6) / max(self.subset_size, 1))
        portfolio_features = np.array([turnover, hhi, holding_ratio],
                                      dtype=np.float32)
        return np.concatenate(
            [self._cur_features.flatten(),
             portfolio_features]).astype(np.float32)

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

        if self.sampling_mode == "sequential":
            rows_needed = self.episode_len * self.subset_size
            max_start = max(0, self.total_rows - rows_needed)
            if start_row is not None:
                self.cursor = int(np.clip(start_row, 0, max_start))
            else:
                self.cursor = int(
                    self.np_random.randint(0, max(1, max_start + 1)))
            self.cursor = (self.cursor // self.subset_size) * self.subset_size
        else:
            self.cursor = 0

        self._load_batch()
        obs = self._build_observation()
        return obs

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

        ic_value = rank_ic(action,
                           self._cur_returns) if action.size > 1 else 0.0

        reward = self.ic_scale * ic_value

        ### 惩罚
        if self.negative_ic_penalty > 0 and ic_value < 0:
            reward -= self.negative_ic_penalty * abs(ic_value)

        should_rebalance = False
        old_weights = np.zeros(self.subset_size, dtype=np.float32)
        new_weights = np.zeros(self.subset_size, dtype=np.float32)
        turnover = 0.0
        cost = 0.0
        portfolio_return = 0.0

        if self.action_mode == "weights":
            if self.sampling_mode == "sequential":
                old_weights = self.prev_weights.copy()
                should_rebalance = (self.signal_config.rebalance_window <= 1 or
                                    (self.episode_step_count %
                                     self.signal_config.rebalance_window == 0))
                if should_rebalance and action.size > 0:
                    new_weights = scores_to_weights(action, self.signal_config)
                else:
                    new_weights = old_weights.copy()
            else:
                should_rebalance = True
                old_weights = np.zeros(self.subset_size, dtype=np.float32)
                new_weights = scores_to_weights(action, self.signal_config)

            turnover = calculate_turnover(old_weights, new_weights)
            cost = calculate_transaction_cost(old_weights, new_weights,
                                              self.signal_config)
            portfolio_return = calculate_portfolio_return(
                new_weights, self._cur_returns)

            if self.use_fee_in_reward and self.fee_coef > 0:
                reward -= self.fee_coef * cost
            if self.use_turnover_proxy and self.turnover_proxy_coef > 0:
                reward -= self.turnover_proxy_coef * turnover
            if self.signal_config.turnover_penalty > 0:
                reward -= self.signal_config.turnover_penalty * turnover

            if self.sampling_mode == "sequential":
                self.prev_weights = new_weights

            self.last_turnover = float(turnover)
        else:
            # raw_ic 模式：直接用原始 action 计算 IC，不走权重交易路径
            self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)
            self.last_turnover = 0.0

        self.total_turnover += turnover
        self.total_cost += cost
        self.total_rank_ic += ic_value
        self.total_portfolio_return += portfolio_return
        self.total_rebalances += int(should_rebalance)
        self.total_reward_raw += reward

        self.episode_step_count += 1
        if self.sampling_mode == "sequential":
            self.cursor += self.subset_size

        if self.episode_step_count >= self.episode_len:
            self.terminal = True
        if self.sampling_mode == "sequential" and self.cursor + self.subset_size > self.total_rows:
            self.terminal = True

        if not self.terminal:
            self._load_batch()

        obs = self._build_observation()
        reward_scaled = reward * self.reward_scale
        done = bool(self.terminal)

        weight_hhi = float(np.sum(new_weights**2))
        active_positions = int(np.sum(new_weights > 1e-8))

        finite_ok = (np.isfinite(reward_scaled) and np.isfinite(turnover)
                     and np.isfinite(cost) and np.isfinite(ic_value)
                     and np.all(np.isfinite(new_weights)))
        if not finite_ok:
            self.nan_or_inf_events += 1

        info = {
            "episode_step": int(self.episode_step_count),
            "cursor": int(self.cursor),
            "sampling_mode": self.sampling_mode,
            "action_mode": self.action_mode,
            "include_portfolio_state": bool(self.include_portfolio_state),
            "rank_ic": float(ic_value),
            "reward_raw": float(reward),
            "reward_scaled": float(reward_scaled),
            "turnover": float(turnover),
            "cost": float(cost),
            "portfolio_return": float(portfolio_return),
            "total_rank_ic": float(self.total_rank_ic),
            "total_turnover": float(self.total_turnover),
            "total_cost": float(self.total_cost),
            "total_portfolio_return": float(self.total_portfolio_return),
            "rebalanced": bool(should_rebalance),
            "weight_hhi": weight_hhi,
            "active_positions": active_positions,
            "finite_ok": bool(finite_ok),
        }

        if self.enable_step_logging and (done or
                                         (self.episode_step_count %
                                          self.log_every_n_steps == 0)):
            logger.info(
                "TradingEnv mode=%s action_mode=%s step=%d done=%s ic=%.6f reward=%.6f "
                "ret=%.6f cost=%.6f turnover=%.6f hhi=%.6f active=%d",
                self.sampling_mode,
                self.action_mode,
                self.episode_step_count,
                done,
                ic_value,
                reward,
                portfolio_return,
                cost,
                turnover,
                weight_hhi,
                active_positions,
            )

        if self.enable_step_logging and (done or
                                         (self.episode_step_count %
                                          self.log_every_n_steps == 0)):
            logger.info(
                "TradingEnv anomaly step=%d finite_ok=%s turnover=%.6f threshold=%.6f",
                self.episode_step_count,
                finite_ok,
                turnover,
                self.warn_turnover_threshold,
            )

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
                f"sampling_mode={self.sampling_mode}, "
                f"action_mode={self.action_mode}, "
                f"reward_scale={self.reward_scale})")
