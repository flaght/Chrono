import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from gym import spaces

from lib.rl013.signal import *

LOGGER = logging.getLogger(__name__)


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
                 turnover_proxy_coef: float = 0.0,
                 use_fee_in_reward: bool = True,
                 fee_coef: float = 1.0,
                 enable_step_logging: bool = True,
                 log_every_n_steps: int = 200,
                 warn_turnover_threshold: float = 0.80):

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
        self.use_fee_in_reward = bool(use_fee_in_reward)
        self.fee_coef = float(fee_coef)
        self.enable_step_logging = bool(enable_step_logging)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.warn_turnover_threshold = float(warn_turnover_threshold)

        if signal_config is None:
            raise ValueError("signal_config must be provided")
        self.signal_config = signal_config

        if "nxt1_ret" not in self.df.columns:
            raise ValueError("data must contain 'nxt1_ret'")
        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"missing feature column: {f}")
        if self.subset_size <= 0:
            raise ValueError(
                f"subset_size must be > 0, got {self.subset_size}")
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
            low=0.0,
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

        # ── 状态 ──
        self.cursor = 0  # 当前读取位置
        self.episode_step_count = 0
        self.terminal = False
        self.last_turnover = 0.0
        self.prev_weights = np.zeros(self.subset_size, dtype=np.float32)

        # 累计指标
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_rank_ic = 0.0
        self.total_reward_raw = 0.0
        self.total_portfolio_return = 0.0
        self.total_rebalances = 0
        self.nan_or_inf_events = 0

        # 当前 batch 缓存
        self._cur_features = np.zeros((self.subset_size, self.n_features),
                                      dtype=np.float32)
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

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = action.astype(np.float32).flatten()

        # 维度对齐
        if action.size < self.subset_size:
            action = np.pad(action, (0, self.subset_size - action.size))
        elif action.size > self.subset_size:
            action = action[:self.subset_size]

        if not np.all(np.isfinite(action)):
            self.nan_or_inf_events += 1
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)

        # 已终止
        if self.terminal:
            return self._build_observation(), 0.0, True, {
                "episode_step": int(self.episode_step_count),
                "cursor": int(self.cursor),
                "already_terminal": True,
            }

        action_mean = float(np.mean(action))
        action_std = float(np.std(action))
        action_min = float(np.min(action))
        action_max = float(np.max(action))
        returns_mean = float(np.mean(self._cur_returns))
        returns_std = float(np.std(self._cur_returns))
        returns_min = float(np.min(self._cur_returns))
        returns_max = float(np.max(self._cur_returns))

        ## 计算RankIC
        ic_value = rank_ic(action,
                           self._cur_returns) if action.size > 1 else 0.0

        reward = self.ic_scale * ic_value

        ### 惩罚
        if self.negative_ic_penalty > 0 and ic_value < 0:
            reward -= self.negative_ic_penalty * abs(ic_value)

        ### 换手率
        old_weights = self.prev_weights.copy()
        should_rebalance = (self.signal_config.rebalance_window <= 1
                            or (self.episode_step_count %
                                self.signal_config.rebalance_window == 0))
        if should_rebalance and action.size > 0:
            new_weights = scores_to_weights(action, self.signal_config)
        else:
            new_weights = old_weights.copy()

        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights,
                                          self.signal_config)
        portfolio_return = calculate_portfolio_return(new_weights,
                                                      self._cur_returns)

        self.last_turnover = turnover
        self.total_turnover += turnover
        self.total_cost += cost
        self.total_rank_ic += ic_value
        self.total_portfolio_return += portfolio_return
        self.total_rebalances += int(should_rebalance)

        # 手续费率惩罚
        if self.use_fee_in_reward and self.fee_coef > 0:
            reward -= self.fee_coef * cost

        # 训练过程的行为约束/代理正则
        if self.use_turnover_proxy and self.turnover_proxy_coef > 0:
            reward -= self.turnover_proxy_coef * turnover
        #  策略层面的固定换手惩罚
        if self.signal_config.turnover_penalty > 0:
            reward -= self.signal_config.turnover_penalty * turnover

        self.prev_weights = new_weights
        self.total_reward_raw += reward

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

        if self.enable_step_logging and (done or
                                         (self.episode_step_count %
                                          self.log_every_n_steps == 0)):
            LOGGER.info(
                "TradingEnv step=%d done=%s ic=%.6f reward=%.6f "
                "ret=%.6f cost=%.6f turnover=%.6f hhi=%.6f active=%d",
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

        return obs, reward_scaled, done, info

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

    def close(self):
        pass

    def __repr__(self) -> str:
        return (f"TradingEnv("
                f"total_rows={self.total_rows}, "
                f"subset_size={self.subset_size}, "
                f"episode_len={self.episode_len}, "
                f"n_features={self.n_features}, "
                f"state_dim={self.n_state_features}, "
                f"reward_scale={self.reward_scale})")
