import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from gym import spaces

from lib.rl002.signal import (Config, calculate_turnover, action_to_weights, 
                              calculate_transaction_cost, calculate_portfolio_return)


class TradingEnv:
    def __init__(self, 
                 df: pd.DataFrame,
                 features: List[str],
                 n_assets: int = 0,
                 episode_len: int = 500,
                 start_time: Optional[int] = None,
                 seed: Optional[int] = None,
                 reward_scale: float = 10000.0,
                 signal_config: Optional[Config] = None,
                 strict_asset_alignment: bool = True):
        
        """
        Args:
            df: 面板数据, 必须包含 [trade_time, asset_id, ret_1min] + features
                  按 trade_time 排序, 每个 trade_time 有相同数量的 asset 行
            features: 特征列名列表
            n_assets: 股票数量 (0 = 自动检测)
            episode_len: 每个 episode 的时间步数
            start_time: 起始时间索引
            seed: 随机种子
            reward_scale: 奖励缩放因子
            signal_config: A股截面配置
        """
        self.df = df.copy()
        self.features = features
        self.episode_len = episode_len
        self.reward_scale = reward_scale
        self.strict_asset_alignment = strict_asset_alignment
        
        # 信号配置
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
        
        # 确定股票数量
        if 'code' in self.df.columns:
            self.asset_ids = sorted(self.df['code'].unique())
            self.n_assets = len(self.asset_ids) if n_assets == 0 else n_assets
        else:
            # 没有 code 列, 假设每个时间步只有一行
            self.asset_ids = ['code']
            self.n_assets = 1
            
        # 构建面板索引
        self._build_panel_index()
        self._validate_asset_alignment()
        
        # 维度
        self.n_features = len(features)
        # 观测维度 = N只股票 * 每只特征数 + 组合状态特征
        self.n_portfolio_features = 3  # 当前换手率, 持仓集中度, 持仓数量比
        self.n_state_features = self.n_assets * self.n_features + self.n_portfolio_features
        
        # 动作空间: N 只股票的权重 [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32
        )
        
        # 状态变量
        self.current_step = 0
        self.current_time_index = 0
        self.episode_step_count = 0
        self.terminal = False
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.last_turnover = 0.0
        
        # 统计
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_portfolio_return = 0.0
        
        if seed is not None:
            np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        
        self.fixed_start_time = start_time
        self.start_time_index = 0
        
        # SB3 兼容
        self.spec = None
        self.metadata = {'render_modes': []}
        
        
        
    def _build_panel_index(self):
        """构建面板索引 (极致提速版)"""
        self._time_to_data = {}
        
        # 1. 一次性对整个大表完成排序！这比在循环里排 750 次要快得多
        if 'code' in self.df.columns:
            sorted_df = self.df.sort_values(by=['trade_time', 'code'])
        else:
            sorted_df = self.df.sort_values(by=['trade_time'])
            
        # 2. 利用 Pandas 底层极其高效的 C 语言 groupby 取出所有的截面切片
        # sort=False 是因为我们上面已经全局排过序了，不需要再次做代价昂贵的排序
        grouped_data = sorted_df.groupby('trade_time', sort=False)
        
        # 3. 将切片按顺序写入字典
        for t_idx, (t_val, time_data) in enumerate(grouped_data):
            # time_data 已经是按 code 排好序的截面数据，直接 drop index 即可
            self._time_to_data[t_idx] = time_data.reset_index(drop=True)
                
    def _get_cross_section(self, time_index: int) -> pd.DataFrame:
        """获取某个时间步的截面数据"""
        time_index = min(time_index, self.max_time_index)
        return self._time_to_data.get(time_index, pd.DataFrame())

    def _validate_asset_alignment(self):
        """验证每个时间步资产集合/顺序一致，避免动作权重与收益错位。"""
        if not self.strict_asset_alignment or 'code' not in self.df.columns:
            return

        counts = self.df.groupby('trade_time')['code'].nunique()
        if counts.empty:
            raise ValueError("数据为空，无法构建截面环境")
        if counts.nunique() != 1 or int(counts.iloc[0]) != self.n_assets:
            raise ValueError(
                f"资产数量在不同 trade_time 不一致，期望每期 {self.n_assets}，实际范围 [{int(counts.min())}, {int(counts.max())}]"
            )

        expected_codes = tuple(self.asset_ids)
        for t_idx, cs_data in self._time_to_data.items():
            codes = tuple(cs_data['code'].tolist())
            if codes != expected_codes:
                sample_expected = list(expected_codes[:5])
                sample_actual = list(codes[:5])
                raise ValueError(
                    f"trade_time 索引 {t_idx} 的 code 顺序/集合与全局不一致。"
                    f" expected_head={sample_expected}, actual_head={sample_actual}"
                )
    
    def observation(self, time_index: int) -> np.ndarray:
        """
        构建观测向量
        
        观测 = [stock_1_features, stock_2_features, ..., stock_N_features, portfolio_features]
        
        每只股票的特征: feature1, feature2, ... (n_features 维)
        组合状态特征: 当前换手率, 持仓集中度(HHI), 持仓数量比
        """
        cs_data = self._get_cross_section(time_index)
        
        if len(cs_data) == 0:
            return np.zeros(self.n_state_features, dtype=np.float32)
        
        # 股票因子特征 (N * n_features)
        stock_features = cs_data[self.features].values.astype(np.float32)  # (N, n_features)
        
        # 如果实际股票数和预期不一致, 填充或截断
        if len(stock_features) < self.n_assets:
            pad = np.zeros((self.n_assets - len(stock_features), self.n_features), dtype=np.float32)
            stock_features = np.vstack([stock_features, pad])
        elif len(stock_features) > self.n_assets:
            stock_features = stock_features[:self.n_assets]
            
        stock_features_flat = stock_features.flatten()  # (N * n_features,)
        
        # 1. 当前换手率 (用最近一步的来近似)
        turnover = 0.0 if self.episode_step_count <= 0 else float(self.last_turnover)
        
        # 2. 持仓集中度 (HHI)
        hhi = np.sum(self.current_weights ** 2)  # HHI ∈ [1/N, 1], 越大越集中
        
        # 3. 持仓数量比 (有多少只股票权重 > 0)
        holding_ratio = np.sum(self.current_weights > 0.001) / max(self.n_assets, 1)
        
        portfolio_features = np.array([turnover, hhi, holding_ratio], dtype=np.float32)
        
        # 拼接
        observation = np.concatenate([stock_features_flat, portfolio_features])
        
        
        assert observation.shape == (self.n_state_features,), \
            f"Observation shape mismatch: expected ({self.n_state_features},), got {observation.shape}"
        return observation
    
    def reset(self, start_time_index: Optional[int] = None):
        """重置环境"""
        # 重置权重
        self.current_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.last_turnover = 0.0
        
        # 重置统计
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_portfolio_return = 0.0
        
        # 确定起始时间
        if start_time_index is not None:
            episode_start = int(start_time_index)
        elif self.fixed_start_time is not None:
            # Explicit fixed start is mainly for inference/replay.
            episode_start = int(self.fixed_start_time)
        else:
            # Sample a fresh random start every reset during training/eval.
            max_start = self.max_time_index - self.episode_len
            if max_start <= 0:
                episode_start = 0
            else:
                episode_start = int(self.np_random.randint(0, max_start + 1))
        
        self.start_time_index = max(0, min(episode_start, self.max_time_index))
        
        self.current_time_index = self.start_time_index
        self.current_step = 0
        self.episode_step_count = 0
        self.terminal = False
        
        observation = self.observation(self.current_time_index)
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: shape=(N,), 每只股票的目标权重, 范围 [0, 1]
        
        Returns:
            observation, reward, done, info
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        else:
            action = action.astype(np.float32)
        
        if action.ndim > 1:
            action = action.flatten()
        
        # 截断或填充到 n_assets
        if len(action) < self.n_assets:
            action = np.pad(action, (0, self.n_assets - len(action)))
        elif len(action) > self.n_assets:
            action = action[:self.n_assets]
            
        # 检查是否结束（仅时间边界提前终止，episode 长度在执行后判断）
        if self.current_time_index >= self.max_time_index:
            self.terminal = True
        
        if self.terminal:
            observation = self.observation(min(self.current_time_index, self.max_time_index))
            return observation, 0.0, True, {}
            
        # ========== 1. 转换动作为权重 ==========
        old_weights = self.current_weights.copy()
        should_rebalance = (
            self.signal_config.rebalance_window <= 1
            or (self.current_step % self.signal_config.rebalance_window == 0)
        )
        if should_rebalance:
            new_weights = action_to_weights(action, self.signal_config)
        else:
            # Keep previous target weights when not on a rebalance step.
            new_weights = old_weights.copy()
        
        
        # ========== 2. 计算交易成本 (基于权重变化) ==========
        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights, self.signal_config)
        self.last_turnover = float(turnover)
        
        
        # ========== 3. 获取当期截面收益率 ==========
        cs_data = self._get_cross_section(self.current_time_index)
        if len(cs_data) > 0:
            returns = cs_data['nxt1_ret'].values.astype(np.float32)
            # 填充或截断
            if len(returns) < self.n_assets:
                returns = np.pad(returns, (0, self.n_assets - len(returns)))
            elif len(returns) > self.n_assets:
                returns = returns[:self.n_assets]
        else:
            returns = np.zeros(self.n_assets, dtype=np.float32)
    
    
        # ========== 4. 计算组合收益 ==========
        portfolio_return = calculate_portfolio_return(new_weights, returns)
        
        # ========== 5. 计算奖励 (增加非对称下跌惩罚以降低回撤) ==========
        reward = portfolio_return - cost
        if portfolio_return < 0:
            reward -= 2.0 * abs(portfolio_return)  # 如果当日亏损，额外双倍惩罚下行风险
        
        # 额外的换手惩罚 (可选)
        if self.signal_config.turnover_penalty > 0:
            reward -= self.signal_config.turnover_penalty * turnover
            
        # ========== 6. 更新权重 (引入价格波动带来的持仓比例漂移) ==========
        # 当期投资组合赚了 returns，各个股票仓位的自然末期价值按收益率膨胀
        drifted_weights = new_weights * (1.0 + returns)
        if np.sum(drifted_weights) > 0:
            drifted_weights = drifted_weights / np.sum(drifted_weights)
        else:
            drifted_weights = np.zeros_like(drifted_weights)
            
        # 将被动漂移后的权重作为下期调仓的起点 (old_weights)
        self.current_weights = drifted_weights
        
        # ========== 7. 更新统计 ==========
        self.total_turnover += turnover
        self.total_cost += cost
        self.total_portfolio_return += portfolio_return
        
        # ========== 8. 前进一步 ==========
        self.episode_step_count += 1
        self.current_step += 1
        self.current_time_index += 1
        if self.episode_step_count >= self.episode_len:
            self.terminal = True
        
        # ========== 9. 获取下一步观测 ==========
        if self.current_time_index > self.max_time_index:
            self.terminal = True
            observation = self.observation(min(self.current_time_index, self.max_time_index))
        else:
            observation = self.observation(self.current_time_index)
            
        # 奖励缩放
        reward_scaled = reward * self.reward_scale
        
        # 信息字典 (使用调仓后的 target: new_weights 来计算指标)
        top_k_weights = np.sort(new_weights)[::-1][:5]
        # 调低精度阈值，防止大量极其小的大盘等仓被吃掉判断为 0
        n_holdings = int(np.sum(new_weights > 0.00001))
        
        info = {
            'current_step': int(self.current_step),
            'time_index': int(self.current_time_index),
            'portfolio_return': float(portfolio_return),
            'cost': float(cost),
            'turnover': float(turnover),
            'reward_raw': float(reward),
            'reward_scaled': float(reward_scaled),
            'n_holdings': n_holdings,
            'top_weights': top_k_weights.tolist(),
            'hhi': float(np.sum(new_weights ** 2)),
            'total_turnover': float(self.total_turnover),
            'total_cost': float(self.total_cost),
            'total_portfolio_return': float(self.total_portfolio_return),
            'rebalanced': bool(should_rebalance),
        }
        
        return observation, reward_scaled, self.terminal, info
        
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
    
    def close(self):
        pass
    
    def __repr__(self) -> str:
        return (
            f"TradingEnv(A股截面选股, "
            f"n_assets={self.n_assets}, "
            f"n_features={self.n_features}, "
            f"n_state_dim={self.n_state_features}, "
            f"cost_rate={self.signal_config.cost_rate}, "
            f"stamp_duty={self.signal_config.stamp_duty}, "
            f"reward_scale={self.reward_scale})"
        )
