import numpy as np
import pandas as pd
import gym
import random
from gym import spaces
from typing import List, Dict, Any

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, features: List[str], config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.config = config
        
        self.env_config = config.get("env_config", {})
        self.signal_config = config.get("signal_config", {})
        
        self.holding_period = int(self.env_config["holding_period"])
        self.reward_scale = float(self.env_config["reward_scale"])
        
    
    def step(self, action: np.ndarray):
        raw_action = action.astype(float)
        if not np.isfinite(raw_action).all():
            raw_action = np.zeros(3, dtype=float)
        
        # 截断保持在 [-1, 1] 之间 (SAC 输出自带 tanh，此处作最后保护)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        
        # 核心：将 raw_action (3维 logits) 通过 softmax 得到类别概率分布
        exp_action = np.exp(raw_action - np.max(raw_action)) # 减最大值防止溢出
        softmax_probs = exp_action / np.sum(exp_action)