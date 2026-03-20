import numpy as np
import pandas as pd
import gym,pdb
import random
from gym import spaces
from typing import List, Dict, Any


class TradingEnv(gym.Env):
    """
    基于预期收益率 合成的强化学习环境
    逻辑（完全契合 15 分钟累计收益率筛选）：
    1. 模型输出连续值 action [-1, 1]。
    2. action 被压入一个持有 15 分钟的队列（队列里最多有 15 个 active action）。
    3. 每分钟的净头寸 (Net ER) = 队列里所有未到期的 action 之和。
    4. 单步 Reward = Reward-Net ER * 这 1 分钟内的真实市场收益 (nxt1_ret)。
       其中 Reward-Net ER 支持幂次放大（reward_action_power）以强调大动作。
    
    在数学上，这等价于评价每一笔 action（预测预测准确度）时，让它完整吃到了未来 15 分钟的复利涨跌。
    """
    
    def __init__(self, df: pd.DataFrame, features: List[str], config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.config = config
        
        self.env_config = config.get("env_config", {})
        self.signal_config = config.get("signal_config", {})
        
        self.holding_period = int(self.env_config["holding_period"])
        self.reward_scale = float(self.env_config["reward_scale"])
        self.reward_action_power = float(self.env_config["reward_action_power"])
        if self.reward_action_power <= 0:
            raise ValueError(
                f"reward_action_power 必须大于 0，当前值: {self.reward_action_power}"
            )
        self.last_open_step = len(self.df) - self.holding_period - 1
        if self.last_open_step < 0:
            raise ValueError(
                f"数据长度不足以支持完整持有 {self.holding_period} 期: len(df)={len(self.df)}"
            )
            
        ## 预留以后转信号
        self.discrete_mode = bool(self.signal_config["discrete_mode"])
        self.discrete_threshold = float(self.signal_config["discrete_threshold"])
        
        # 动作空间：一维连续值 [-1, 1] - 这代表 SAC 吐出的 Raw Score
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 状态空间：因子特征 + 当前队列汇总的总 ER (作为状态告知网络目前手里堆了多少多空倾向)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.features) + 1,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.active_queue = [] # 格式: [{'expire_step': int, 'er_value': float, 'reward_er_value': float}]
        
        # 记录每步历史细节，用于最后生成 results.csv 分析跑批结果
        self.history = []

        # 保持与 stable-baselines/gym 的种子接口兼容
        init_seed = self.env_config["seed"]
        if init_seed is not None:
            self.seed(init_seed)
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.active_queue = []
        self.history = []
        return self._get_obs()
    
    def _get_obs(self):
        # 1. 提取因子值
        row = self.df.iloc[self.current_step]
        feat_vals = pd.to_numeric(row[self.features], errors='coerce').values.astype(np.float32)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 累加计算当前的 Net ER
        current_net_er = sum(item['er_value'] for item in self.active_queue)
        
        # 3. 拼接作为特征状态
        obs = np.concatenate([feat_vals, [current_net_er]], axis=0).astype(np.float32)
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _reward_weight(self, er_value: float) -> float:
        # p=1 时等价线性 reward；p>1 时增强大动作贡献，抑制中间小动作。
        return float(np.sign(er_value) * (abs(er_value) ** self.reward_action_power))
    
    def step(self, action: np.ndarray):
        raw_action = float(action[0])
        if not np.isfinite(raw_action):
            raw_action = 0.0
        # 截断保持在 [-1, 1] 之间 # 模型已经控制[-1, 1], 这里在做异常处理
        raw_action = max(min(raw_action, 1.0), -1.0)
        
        if self.discrete_mode:
            if raw_action > self.discrete_threshold:
                er_value = 1.0
            elif raw_action < -self.discrete_threshold:
                er_value = -1.0
            else:
                er_value = 0.0
        else:
            # 核心模式：保留原汁原味的连续值
            er_value = raw_action
        
        
        # -------------------------------------------------------------------
        # 1. 开仓 (将当前时刻的预测值压入队列)
        # -------------------------------------------------------------------
        can_open = self.current_step <= self.last_open_step
        if can_open and er_value != 0:
            self.active_queue.append({
                'expire_step': self.current_step + self.holding_period,
                'er_value': er_value,
                'reward_er_value': self._reward_weight(er_value)
            })
        opened = can_open and er_value != 0
        
        # -------------------------------------------------------------------
        # 2. 清理到期旧仓 (时间推进)
        # -------------------------------------------------------------------
        # 当前步骤结束后，只有大于当前时间的预测才能被带入下一个时刻
        self.active_queue = [item for item in self.active_queue if item['expire_step'] > self.current_step]
        
        
        # -------------------------------------------------------------------
        # 3. 汇总当前净头寸 
        # -------------------------------------------------------------------
        net_er = sum(item['er_value'] for item in self.active_queue)
        reward_net_er = sum(item.get('reward_er_value', item['er_value']) for item in self.active_queue)
        
        # -------------------------------------------------------------------
        # 4. 获取当前这 1 分钟的真实市场环境收益 (即原代码的 nxt1_ret)
        # 数据集在构造中，应该确保 nxt1_ret 是从 t时刻到 t+1时刻的收益
        # -------------------------------------------------------------------
        nxt1_ret = float(self.df.iloc[self.current_step].get('nxt1_ret', 0.0))
        if not np.isfinite(nxt1_ret):
            nxt1_ret = 0.0
        
        # -------------------------------------------------------------------
        # 5. 计算奖励 (Reward)：在每一分钟，让过去15分钟内的所有有效预测共同来分配奖惩
        # 如果不开 discrete_mode，这实质上是不带手续费的纯 IC 评价函数！
        # -------------------------------------------------------------------
        step_reward = reward_net_er * nxt1_ret
        if not np.isfinite(step_reward):
            step_reward = 0.0
        scaled_reward = step_reward * self.reward_scale
        if not np.isfinite(scaled_reward):
            scaled_reward = 0.0
        
        # ==================== 记录给 Analysis 分析的指标 ====================
        trade_time = self.df.iloc[self.current_step].get('trade_time', self.current_step)
        
        
        net_position = net_er # 这里的净头寸不再是手数，而是累加的连续得分
        direction = 1 if er_value > 0 else (-1 if er_value < 0 else 0)
        confidence = abs(er_value)
        signal = er_value
        
        self.history.append({
            'trade_time': trade_time,
            'action': f"[{raw_action}]", 
            'signal': signal,
            'direction': direction,
            'confidence': confidence,
            'net_position': net_position, # 这列用于后续分析净预测力
            'reward_net_position': reward_net_er,
            'active_signals': len(self.active_queue),
            'opened': opened,
            'current_ret': nxt1_ret,
            'reward': step_reward,
            'reward_scaled': scaled_reward,
            # 将 trade_cost 设为 0（因做纯因子预测时不在此纳入交易摩擦）
            'trade_cost': 0.0 
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), scaled_reward, done, {}
        
