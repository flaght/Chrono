import numpy as np
import pandas as pd
import gym, random
from gym import spaces
from typing import List, Dict, Any

class TradingEnv(gym.Env):
    """
    基于预期收益率合成的连续到离散分类的强化学习环境 (rl012版本)。
    核心逻辑：
    1. 自定义策略模型 (如 ResNet) 每步输出表示意图维度的 3D 原始连续 Logits [-1, 1]。
    2. 环境内部通过缩放与 Softmax 将其转换为 `[观望, 开多, 开空]` 的全集概率权重 [0, 1]。
    3. 环境使用 Argmax 将概率强迫确定具体的分类方向，并乘以其最大概率值作为动态开仓仓位/信号。
    当前 action 的单步加权直接对应未来 N 分钟累计收益标签。
    单步奖励是 action * future_ret_h，使得在评价每一笔预测准确度时，让它完整吃到了未来 N 分钟的涨跌。
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
        
        
        self.last_open_step = len(self.df) - self.holding_period - 1
        if self.last_open_step < 0:
            raise ValueError(
                f"数据长度不足以支持完整持有 {self.holding_period} 期: len(df)={len(self.df)}"
            )
            
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.features),), 
            dtype=np.float32
        )
        
        
        self.current_step = 0
        self.future_ret_h = self._build_future_horizon_returns()
        
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.history = []
        return self._get_obs()
    
    
    def _build_future_horizon_returns(self) -> np.ndarray:
        """
        对每个 t 预计算 sum(ret[t : t+holding_period])。
        对于末尾不足 holding_period 的位置，保留 NaN。
        """
        ret = pd.to_numeric(self.df.get("nxt1_ret", 0.0), errors="coerce").astype(float).to_numpy()
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(ret)
        out = np.full(n, np.nan, dtype=np.float64)
        if self.holding_period <= 0 or n < self.holding_period:
            return out
        kernel = np.ones(self.holding_period, dtype=np.float64)
        valid = np.convolve(ret, kernel, mode="valid")
        out[: len(valid)] = valid
        return out
    
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        feat_vals = pd.to_numeric(row[self.features], errors='coerce').values.astype(np.float32)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.concatenate([feat_vals], axis=0).astype(np.float32) 
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs
    
    def step(self, action: np.ndarray):
        raw_action = action.astype(float)
        if not np.isfinite(raw_action).all():
            raw_action = np.zeros(3, dtype=float)

        # 核心：将 raw_action (3维 logits，本身被 SAC 限制在 [-1, 1] 之间) 放大
        # 放大的目的是打破 Tanh [-1, 1] 造成的数学封锁，让 Softmax 可以达到 >90% 的真正高置信度 (否则最大只能达到 78%)
        temperature_scaled_action = raw_action * 5.0
        exp_action = np.exp(temperature_scaled_action - np.max(temperature_scaled_action)) # 减最大值防止溢出
        softmax_probs = exp_action / np.sum(exp_action)
    
        chosen_action = int(np.argmax(softmax_probs))
        if chosen_action == 0:
            er_value = 0.0
            confidence = float(softmax_probs[0])
        elif chosen_action == 1:
            confidence = float(softmax_probs[1])
            er_value = confidence
        else:
            confidence = float(softmax_probs[2])
            er_value = -confidence
        
        raw_action_str = f"[{raw_action[0]:.4f},{raw_action[1]:.4f},{raw_action[2]:.4f}]"
        soft_action_str = f"[{softmax_probs[0]:.4f},{softmax_probs[1]:.4f},{softmax_probs[2]:.4f}]"
        
        can_open = self.current_step <= self.last_open_step
        opened = can_open and er_value != 0
        
        net_er_out = er_value if opened else 0.0
        active_count = 1 if opened else 0
        future_ret_h = float(self.future_ret_h[self.current_step]) if can_open else np.nan
        
        nxt1_ret = float(self.df.iloc[self.current_step].get('nxt1_ret', 0.0))
        if not np.isfinite(nxt1_ret):
            nxt1_ret = 0.0
        
        
        target_ret_raw = future_ret_h if np.isfinite(future_ret_h) else 0.0
        target_ret = target_ret_raw
        # baseline_ret = (
        #     float(self.future_ret_h_baseline[self.current_step])
        #     if (self.future_ret_h_baseline is not None and can_open)
        #     else 0.0
        # )
        # target_ret_excess = target_ret_raw - baseline_ret
        
        # if self.target_mode == "raw":
        #     target_ret = target_ret_raw
        # elif self.target_mode == "excess":
        #     target_ret = target_ret_excess
        # else:
        #     target_ret = (
        #         self.target_mix_alpha * target_ret_raw
        #         + (1.0 - self.target_mix_alpha) * target_ret_excess
        #     ) 
        
        step_reward = net_er_out * target_ret
        
        # if self.exposure_penalty > 0:
        #     step_reward -= self.exposure_penalty * (reward_net_er ** 2)
        if not np.isfinite(step_reward):
            step_reward = 0.0
        scaled_reward = step_reward * self.reward_scale
        
        
        trade_time = self.df.iloc[self.current_step].get('trade_time', self.current_step)
        
        direction = 1 if er_value > 0 else (-1 if er_value < 0 else 0)
        signal = er_value
        
        self.history.append({
            'trade_time': trade_time,
            'raw_action': raw_action_str, 
            'soft_action': soft_action_str,
            'signal': signal,
            'direction': direction,
            'confidence': confidence,
            'net_er_out': net_er_out,
            'er_value': er_value,
            'active_signals': active_count,
            'opened': opened,
            'current_ret': nxt1_ret,
            'target_ret_raw': target_ret_raw,
            'target_ret': target_ret,
            'reward': step_reward,
            'reward_scaled': scaled_reward,
            'future_ret_h': future_ret_h,
            # 将 trade_cost 设为 0（因做纯因子预测时不在此纳入交易摩擦）
            'trade_cost': 0.0 
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), scaled_reward, done, {}
        
        
