"""
期现正套交易环境

核心设计:
  - 正套 = 做多现货 + 做空期货 (方向固定, 不可反套)
  - 截面环境: 每个时间步同时处理 N 个交易对
  - Agent 输出 N 维权重向量 [0,1]
    - 收益定义 (预处理好):
      nxt_ret = 组合简单收益率转对数 (包含基差收益 + 资金费率收益)
  - 成本 = 现货手续费 + 期货手续费 (基于权重变化, 在 env 内部扣除)

数据格式:
  面板数据 DataFrame 包含:
    - trade_time: 时间
    - code: 交易对 (如 'BTC', 'ETH')
    - nxt1_ret: 下期综合收益率 (包含基差和资金费率)
    - basis_pct: 基差率 (可选, 用于观测)
    - feature1, feature2, ...: 因子特征
    
"""
import pdb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces

from .signal import (Config, action_to_weights, calculate_turnover,
                     calculate_transaction_cost, calculate_arbitrage_return)


class TradingEnv:
    """
    期现正套截面环境
    
    方向固定: 做多现货 + 做空期货 (正套)
    Agent 只决定: 哪些交易对入场, 分配多少仓位
    """

    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str],
                 n_pairs: int = 0,
                 episode_len: int = 500,
                 start_time: Optional[int] = None,
                 seed: Optional[int] = None,
                 reward_scale: float = 10000.0,
                 signal_config: Optional[Config] = None):
        """
        Args:
            df: 面板数据, 必须包含 [trade_time, code, nxt1_ret] + features
            features: 因子特征列名
            n_pairs: 交易对数量 (0 = 自动检测)
            episode_len: 每个 episode 步数
            start_time: 起始时间索引
            seed: 随机种子
            reward_scale: 奖励缩放
            signal_config: 套利配置
        """
        self.df = df.copy()
        self.features = features
        self.episode_len = episode_len
        self.reward_scale = reward_scale

        # 配置
        if signal_config is None:
            self.signal_config = Config()
        else:
            self.signal_config = signal_config

        # 验证数据
        required_cols = ['trade_time', 'nxt1_ret']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"数据必须包含 '{col}' 列")
        for f in features:
            if f not in self.df.columns:
                raise ValueError(f"数据缺少特征列: {f}")

        # 解析面板结构
        self.unique_times = sorted(self.df['trade_time'].unique())
        self.max_time_index = len(self.unique_times) - 1

        # 交易对
        if 'code' in self.df.columns:
            self.pair_ids = sorted(self.df['code'].unique())
            self.n_pairs = len(self.pair_ids) if n_pairs == 0 else n_pairs
        else:
            self.pair_ids = ['code']
            self.n_pairs = 1

        # 构建面板索引
        self._build_panel_index()

        # 维度
        self.n_features = len(features)

        # 每个交易对: n_features 个因子 + 1 个当前权重
        # 组合特征: 换手率, 集中度(HHI), 持仓数比, 加权基差
        self.n_portfolio_features = 4
        self.n_state_features = self.n_pairs * (self.n_features +
                                                1) + self.n_portfolio_features

        # 动作空间: N 个交易对的权重 [0, 1]
        self.action_space = spaces.Box(low=0.0,
                                       high=1.0,
                                       shape=(self.n_pairs, ),
                                       dtype=np.float32)

        # 观测空间
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.n_state_features, ),
                                            dtype=np.float32)
        # 状态
        self.current_step = 0
        self.current_time_index = 0
        self.episode_step_count = 0
        self.terminal = False
        self.current_weights = np.zeros(self.n_pairs, dtype=np.float32)

        # 统计
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_arb_return = 0.0

        if seed is not None:
            np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)

        self.start_time_index = start_time

        # SB3 兼容
        self.spec = None
        self.metadata = {'render_modes': []}

    def _build_panel_index(self):
        """构建面板索引"""
        self._time_to_data = {}

        if 'code' in self.df.columns:
            for t_idx, t in enumerate(self.unique_times):
                time_data = self.df[self.df['trade_time'] == t].copy()
                time_data = time_data.sort_values('code').reset_index(
                    drop=True)
                self._time_to_data[t_idx] = time_data
        else:
            for t_idx, t in enumerate(self.unique_times):
                time_data = self.df[self.df['trade_time'] ==
                                    t].copy().reset_index(drop=True)
                self._time_to_data[t_idx] = time_data

    def _get_cross_section(self, time_index: int) -> pd.DataFrame:
        time_index = min(time_index, self.max_time_index)
        return self._time_to_data.get(time_index, pd.DataFrame())

    def _pad_or_truncate(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """填充或截断到目标长度"""
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        elif len(arr) > target_len:
            return arr[:target_len]
        return arr

    def observation(self, time_index: int) -> np.ndarray:
        """
        构建观测向量
        
        观测 = [
            pair_1_features, pair_1_current_weight,
            pair_2_features, pair_2_current_weight,
            ...,
            pair_N_features, pair_N_current_weight,
            portfolio_turnover, portfolio_hhi, holding_ratio, weighted_basis
        ]
        """
        cs_data = self._get_cross_section(time_index)

        if len(cs_data) == 0:
            return np.zeros(self.n_state_features, dtype=np.float32)

        # 交易对因子特征
        pair_features = cs_data[self.features].values.astype(
            np.float32)  # (N, n_features)

        # 填充或截断
        if len(pair_features) < self.n_pairs:
            pad = np.zeros(
                (self.n_pairs - len(pair_features), self.n_features),
                dtype=np.float32)
            pair_features = np.vstack([pair_features, pad])
        elif len(pair_features) > self.n_pairs:
            pair_features = pair_features[:self.n_pairs]

        # 每个交易对拼接当前权重: [features, current_weight]
        weights_col = self.current_weights.reshape(-1, 1)  # (N, 1)
        pair_obs = np.hstack([pair_features,
                              weights_col])  # (N, n_features + 1)
        pair_obs_flat = pair_obs.flatten()

        # 组合状态特征
        hhi = float(np.sum(self.current_weights**2))
        holding_ratio = float(np.sum(self.current_weights > 0.001)) / max(
            self.n_pairs, 1)

        # 上一步换手率
        turnover = 0.0  # 会在 step 中更新

        # 加权基差 (当前持仓的加权平均基差)
        if 'basis_pct' in cs_data.columns:
            basis_vals = cs_data['basis_pct'].values.astype(np.float32)
            basis_vals = self._pad_or_truncate(basis_vals, self.n_pairs)
            weighted_basis = float(np.dot(self.current_weights, basis_vals))
        else:
            weighted_basis = 0.0

        portfolio_features = np.array(
            [turnover, hhi, holding_ratio, weighted_basis], dtype=np.float32)

        observation = np.concatenate([pair_obs_flat, portfolio_features])

        assert observation.shape == (self.n_state_features,), \
            f"Observation shape mismatch: expected ({self.n_state_features},), got {observation.shape}"

        return observation

    def reset(self, start_time_index: Optional[int] = None):
        """
        重置环境
        
        注意: kichaos.stable3 使用旧版 gym API, reset() 只返回 obs
        """
        self.current_weights = np.zeros(self.n_pairs, dtype=np.float32)

        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_arb_return = 0.0

        if start_time_index is not None:
            self.start_time_index = start_time_index
        elif self.start_time_index is not None:
            pass
        else:
            max_start = self.max_time_index - self.episode_len - 1
            self.start_time_index = self.np_random.randint(
                0, max(max_start, 1))

        self.current_time_index = self.start_time_index
        self.current_step = 0
        self.episode_step_count = 0
        self.terminal = False

        return self.observation(self.current_time_index)

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: shape=(N,), 每个交易对的正套仓位权重, 范围 [0, 1]
        
        Returns:
            observation, reward, done, info
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        else:
            action = action.astype(np.float32)

        if action.ndim > 1:
            action = action.flatten()

        action = self._pad_or_truncate(action, self.n_pairs)

        # Episode 结束检查
        self.episode_step_count += 1
        if self.episode_step_count >= self.episode_len:
            self.terminal = True
        if self.current_time_index >= self.max_time_index:
            self.terminal = True

        if self.terminal:
            observation = self.observation(
                min(self.current_time_index, self.max_time_index))
            return observation, 0.0, True, {}

        # ========== 1. 转换动作为权重 ==========
        new_weights = action_to_weights(action, self.signal_config)
        old_weights = self.current_weights.copy()

        # ========== 2. 计算交易成本 ==========
        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights,
                                          self.signal_config)

        # ========== 3. 获取综合对数收益 (包含了基差和资金费率) ==========
        cs_data = self._get_cross_section(self.current_time_index)
        if len(cs_data) > 0 and 'nxt1_ret' in cs_data.columns:
            log_returns = cs_data['nxt1_ret'].values.astype(np.float32)
            log_returns = self._pad_or_truncate(log_returns, self.n_pairs)
        else:
            log_returns = np.zeros(self.n_pairs, dtype=np.float32)

        # ========== 4. 计算组合收益 ==========
        # arb_return 现在包含了基差和资金费率收益
        arb_return = calculate_arbitrage_return(new_weights, log_returns)

        # ========== 5. 计算奖励 (收益 - 成本) ==========
        reward = arb_return - cost

        if self.signal_config.turnover_penalty > 0:
            reward -= self.signal_config.turnover_penalty * turnover

        # ========== 6. 更新状态 ==========
        self.current_weights = new_weights
        self.total_turnover += turnover
        self.total_cost += cost
        self.total_arb_return += arb_return

        self.current_step += 1
        self.current_time_index += 1

        if self.current_time_index > self.max_time_index:
            self.terminal = True
            observation = self.observation(
                min(self.current_time_index, self.max_time_index))
        else:
            observation = self.observation(self.current_time_index)

        reward_scaled = reward * self.reward_scale

        # 当前基差统计
        if 'basis_pct' in cs_data.columns:
            basis_vals = cs_data['basis_pct'].values.astype(np.float32)
            basis_vals = self._pad_or_truncate(basis_vals, self.n_pairs)
            weighted_basis = float(np.dot(new_weights, basis_vals))
            avg_basis = float(np.mean(basis_vals))
        else:
            weighted_basis = 0.0
            avg_basis = 0.0

        top_k_weights = np.sort(new_weights)[::-1][:5]
        n_holdings = int(np.sum(new_weights > 0.001))

        info = {
            'current_step': int(self.current_step),
            'time_index': int(self.current_time_index),
            'arb_return': float(arb_return),  # 包含了基差和资金费率
            'cost': float(cost),
            'turnover': float(turnover),
            'reward_raw': float(reward),
            'reward_scaled': float(reward_scaled),
            'n_holdings': n_holdings,
            'top_weights': top_k_weights.tolist(),
            'hhi': float(np.sum(new_weights**2)),
            'weighted_basis': weighted_basis,
            'avg_basis': avg_basis,
            'total_turnover': float(self.total_turnover),
            'total_cost': float(self.total_cost),
            'total_arb_return': float(self.total_arb_return),
        }

        return observation, reward_scaled, self.terminal, info

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)

    def close(self):
        pass

    def __repr__(self) -> str:
        return (f"TradingEnv(期现正套, "
                f"n_pairs={self.n_pairs}, "
                f"n_features={self.n_features}, "
                f"n_state_dim={self.n_state_features}, "
                f"spot_fee={self.signal_config.spot_fee}, "
                f"futures_fee={self.signal_config.futures_fee}, "
                f"reward_scale={self.reward_scale})")
