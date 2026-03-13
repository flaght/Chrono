import logging
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
        
        
        action_low = -1.0 if self.action_mode == "raw_ic" else 0.0
        action_high = 1.0
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.subset_size,),
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_features,),
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

        self._cur_features = np.zeros((self.subset_size, self.n_features), dtype=np.float32)
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
            self._cur_features[:avail] = self._all_features[self.cursor:self.total_rows]
            self._cur_features[avail:] = 0.0
            self._cur_returns[:avail] = self._all_returns[self.cursor:self.total_rows]
            self._cur_returns[avail:] = 0.0
        else:
            self._cur_features[:] = self._all_features[self.cursor:end]
            self._cur_returns[:] = self._all_returns[self.cursor:end]