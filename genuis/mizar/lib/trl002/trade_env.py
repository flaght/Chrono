import pdb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gym
from gym import spaces

from lib.trl002.signal import Config, calculate_turnover, action_to_weights, calculate_transaction_cost, calculate_portfolio_return


class TradingEnv:
    """
    A股截面选股环境
    
    关键区别:
      - 动作空间: Box(shape=(N,)), N = 股票数量
      - 观测空间: Box(shape=(N * n_features + portfolio_features,))
      - 奖励: 组合收益 - 换手成本
    """
    def __init__(self, 
                 df: pd.DataFrame,
                 features: List[str],
                 n_assets: int = 0,
                 episode_len: int = 500,
                 start_time: Optional[int] = None,
                 seed: Optional[int] = None,
                 reward_scale: float = 10000.0,
                 signal_config: Optional[Config] = None):
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
        
        # 信号配置
        if signal_config is None:
            self.signal_config = Config()
        else:
            self.signal_config = signal_config
            
        # 验证数据
        for col in ['trade_time', 'ret_1min']:
            if col not in self.df.columns:
                raise ValueError(f"数据必须包含 '{col}' 列")
        for f in features:
            if f not in self.df.columns:
                raise ValueError(f"数据缺少特征列: {f}")
            
        # 解析面板结构
        self.unique_times = sorted(self.df['trade_time'].unique())
        self.max_time_index = len(self.unique_times) - 1
        
        # 确定股票数量
        if 'asset_id' in self.df.columns:
            self.asset_ids = sorted(self.df['asset_id'].unique())
            self.n_assets = len(self.asset_ids) if n_assets == 0 else n_assets
        else:
            # 没有 asset_id 列, 假设每个时间步只有一行
            self.asset_ids = ['asset_0']
            self.n_assets = 1
            
        # 构建面板索引: time_index -> DataFrame (N行)
        self._build_panel_index()
        
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
        
        # 统计
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_portfolio_return = 0.0
        
        if seed is not None:
            np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        
        self.start_time_index = start_time
        
        # SB3 兼容
        self.spec = None
        self.metadata = {'render_modes': []}
            
        
        
    def _build_panel_index(self):
        """构建面板索引, 将数据按时间步组织"""
        self._time_to_data = {}
        
        if 'asset_id' in self.df.columns:
            for t_idx, t in enumerate(self.unique_times):
                time_data = self.df[self.df['trade_time'] == t].copy()
                # 按 asset_id 排序, 确保顺序一致
                time_data = time_data.sort_values('asset_id').reset_index(drop=True)
                self._time_to_data[t_idx] = time_data
        else:
            for t_idx, t in enumerate(self.unique_times):
                time_data = self.df[self.df['trade_time'] == t].copy().reset_index(drop=True)
                self._time_to_data[t_idx] = time_data
    
    def _get_cross_section(self, time_index: int) -> pd.DataFrame:
        """获取某个时间步的截面数据"""
        time_index = min(time_index, self.max_time_index)
        return self._time_to_data.get(time_index, pd.DataFrame())
           
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
        
        # 组合状态特征
        # 1. 当前换手率 (用最近一步的来近似)
        turnover = calculate_turnover(
            self.current_weights, self.current_weights  # 初始时为 0
        ) if self.episode_step_count > 0 else 0.0
        
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
        
        # 重置统计
        self.total_turnover = 0.0
        self.total_cost = 0.0
        self.total_portfolio_return = 0.0
        
        # 确定起始时间
        if start_time_index is not None:
            self.start_time_index = start_time_index
        elif self.start_time_index is not None:
            pass
        else:
            max_start = self.max_time_index - self.episode_len - 1
            self.start_time_index = self.np_random.randint(
                0, max(max_start, 1)
            )
        
        self.current_time_index = self.start_time_index
        self.current_step = 0
        self.episode_step_count = 0
        self.terminal = False
        
        observation = self.observation(self.current_time_index)
        
        return observation
    
    # 观测 (153维)  →  隐藏层1 (256)  →  隐藏层2 (256)  →  动作 (50维)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: shape=(N,), 每只股票的目标权重, 范围 [0, 1]
        
        Returns:
            observation, reward, done, info
        """
        # 确保 action 格式
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
            
        # 检查是否结束
        self.episode_step_count += 1
        if self.episode_step_count >= self.episode_len:
            self.terminal = True
        if self.current_time_index >= self.max_time_index:
            self.terminal = True
        
        if self.terminal:
            observation = self.observation(min(self.current_time_index, self.max_time_index))
            return observation, 0.0, True, {}
        
        # ========== 1. 转换动作为权重 ==========
        new_weights = action_to_weights(action, self.signal_config)
        old_weights = self.current_weights.copy()
        
        
        # ========== 2. 计算交易成本 (基于权重变化) ==========
        turnover = calculate_turnover(old_weights, new_weights)
        cost = calculate_transaction_cost(old_weights, new_weights, self.signal_config)
        
        # ========== 3. 获取当期截面收益率 ==========
        cs_data = self._get_cross_section(self.current_time_index)
        if len(cs_data) > 0:
            returns = cs_data['ret_1min'].values.astype(np.float32)
            # 填充或截断
            if len(returns) < self.n_assets:
                returns = np.pad(returns, (0, self.n_assets - len(returns)))
            elif len(returns) > self.n_assets:
                returns = returns[:self.n_assets]
        else:
            returns = np.zeros(self.n_assets, dtype=np.float32)
            
        # ========== 4. 计算组合收益 ==========
        portfolio_return = calculate_portfolio_return(new_weights, returns)
        
        # ========== 5. 计算奖励 ==========
        reward = portfolio_return - cost
        
        # 额外的换手惩罚 (可选)
        if self.signal_config.turnover_penalty > 0:
            reward -= self.signal_config.turnover_penalty * turnover
            
        # ========== 6. 更新权重 ==========
        self.current_weights = new_weights
        
        # ========== 7. 更新统计 ==========
        self.total_turnover += turnover
        self.total_cost += cost
        self.total_portfolio_return += portfolio_return
        
        # ========== 8. 前进一步 ==========
        self.current_step += 1
        self.current_time_index += 1
        
        # ========== 9. 获取下一步观测 ==========
        if self.current_time_index > self.max_time_index:
            self.terminal = True
            observation = self.observation(min(self.current_time_index, self.max_time_index))
        else:
            observation = self.observation(self.current_time_index)
            
        # 奖励缩放
        reward_scaled = reward * self.reward_scale
        
        # 信息字典
        top_k_weights = np.sort(new_weights)[::-1][:5]
        n_holdings = int(np.sum(new_weights > 0.001))
        
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
