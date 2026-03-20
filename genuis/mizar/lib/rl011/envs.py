import numpy as np
import pandas as pd
import gym
import random
from gym import spaces
from typing import List, Dict, Any


class TradingEnv(gym.Env):
    """
    基于预期收益率 合成的强化学习环境
    逻辑（完全契合 15 分钟累计收益率筛选）：
    1. 模型输出连续值 action [-1, 1]。
    reward_mode = "queue" 时：
    1. action 被压入一个持有 N 分钟的队列。
    2. 每分钟净头寸由队列聚合得到，奖励使用队列聚合权重 * 当前 1 分钟收益。

    reward_mode = "single_horizon" 时：
    1. 当前 action 直接对应未来 N 分钟累计收益标签。
    2. 单步奖励是 action_weight * future_ret_h，不再与其它时刻 action 混合。
    
    在数学上，这等价于评价每一笔 action（预测预测准确度）时，让它完整吃到了未来 15 分钟的复利涨跌。
    """
    
    def __init__(self, df: pd.DataFrame, features: List[str], config: Dict[str, Any]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.config = config
        
        self.env_config = config.get("env_config", {})
        self.signal_config = config.get("signal_config", {})
        
        self.holding_period = int(self.env_config.get("holding_period", 15))
        self.reward_scale = float(self.env_config.get("reward_scale", 10000.0))
        self.reward_action_power = float(self.env_config.get("reward_action_power", 1.0))
        if self.reward_action_power <= 0:
            raise ValueError(
                f"reward_action_power 必须大于 0，当前值: {self.reward_action_power}"
            )
        self.reward_mode = str(self.env_config.get("reward_mode", "queue")).strip().lower()
        if self.reward_mode not in {"queue", "single_horizon"}:
            raise ValueError(
                f"reward_mode 仅支持 queue/single_horizon，当前值: {self.reward_mode}"
            )
        self.exposure_penalty = float(self.env_config.get("exposure_penalty", 0.0))
        if self.exposure_penalty < 0:
            raise ValueError(
                f"exposure_penalty 必须 >= 0，当前值: {self.exposure_penalty}"
            )
        # 目标收益模式:
        # raw    : target = future_ret_h
        # excess : target = future_ret_h - baseline
        # mix    : target = a*raw + (1-a)*excess
        self.target_demean = bool(self.env_config.get("target_demean", False))
        target_mode_cfg = str(self.env_config.get("target_mode", "")).strip().lower()
        if not target_mode_cfg:
            target_mode_cfg = "excess" if self.target_demean else "raw"
        if target_mode_cfg not in {"raw", "excess", "mix"}:
            raise ValueError(
                f"target_mode 仅支持 raw/excess/mix，当前值: {target_mode_cfg}"
            )
        self.target_mode = target_mode_cfg
        self.target_mix_alpha = float(self.env_config.get("target_mix_alpha", 0.5))
        if not (0.0 <= self.target_mix_alpha <= 1.0):
            raise ValueError(
                f"target_mix_alpha 必须在 [0,1]，当前值: {self.target_mix_alpha}"
            )
        self.target_demean_window = int(self.env_config.get("target_demean_window", 240))
        self.baseline_window = int(self.env_config.get("baseline_window", self.target_demean_window))
        if self.baseline_window < 0:
            raise ValueError(
                f"baseline_window 必须 >= 0，当前值: {self.baseline_window}"
            )
        # 是否对队列累计的 net_er 做归一化（除以队列长度）
        # True: net_er 永远在 [-1, 1]，不随 holding_period 膨胀（推荐）
        # False: 原始累加模式（holding_period 越长杠杆越大）
        self.reward_normalize = bool(self.env_config.get("reward_normalize", False))
        self.last_open_step = len(self.df) - self.holding_period - 1
        if self.last_open_step < 0:
            raise ValueError(
                f"数据长度不足以支持完整持有 {self.holding_period} 期: len(df)={len(self.df)}"
            )
            
        ## 预留以后转信号
        self.discrete_mode = bool(self.signal_config.get("discrete_mode", False))
        self.discrete_threshold = float(self.signal_config.get("discrete_threshold", 0.5))
        
        # 动作空间：一维连续值 [-1, 1] - 这代表 SAC 吐出的 Raw Score
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        
        # self.sequence_window = int(self.env_config.get("sequence_window", 1))
        # if self.sequence_window <= 0:
        #     raise ValueError(f"sequence_window 必须 >= 1，当前值: {self.sequence_window}")
        
        # obs_feature_dim = len(self.features) * self.sequence_window
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, 
        #     shape=(obs_feature_dim + 1,),
        #     dtype=np.float32
        # )

        
        # 状态空间：因子特征 + 当前队列汇总的总 ER (作为状态告知网络目前手里堆了多少多空倾向)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.features) + 1,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.active_queue = [] # 格式: [{'expire_step': int, 'er_value': float, 'reward_er_value': float}]
        self.future_ret_h = self._build_future_horizon_returns() if self.reward_mode != 'queue' else None
        self.future_ret_h_baseline = (
            self._build_future_horizon_baseline()
            if (self.reward_mode == "single_horizon" and self.target_mode in {"excess", "mix"})
            else None
        )
        
        # 记录每步历史细节，用于最后生成 results.csv 分析跑批结果
        self.history = []

        # 保持与 stable-baselines/gym 的种子接口兼容
        init_seed = self.env_config.get("seed", None)
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

    def _build_future_horizon_baseline(self) -> np.ndarray:
        """
        构造 single_horizon 目标的历史基线(仅使用过去1分钟收益)。
        baseline[t] = mean(nxt1_ret[t-window : t-1]) * holding_period。
        这样避免直接用 historical future_ret_h 作为基线。
        """
        ret1 = pd.to_numeric(self.df.get("nxt1_ret", 0.0), errors="coerce").astype(float).to_numpy()
        ret1 = np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        n = ret1.shape[0]
        baseline = np.zeros(n, dtype=np.float64)
        prefix_sum = np.concatenate([[0.0], np.cumsum(ret1)])

        for t in range(n):
            end = t - 1
            if end < 0:
                baseline[t] = 0.0
                continue

            if self.baseline_window > 0:
                start = max(0, end - self.baseline_window + 1)
            else:
                start = 0

            s = prefix_sum[end + 1] - prefix_sum[start]
            c = end - start + 1
            mean_ret1 = (s / c) if c > 0 else 0.0
            baseline[t] = mean_ret1 * float(self.holding_period)

        return baseline
    
    def _get_obs(self):
        # 1. 提取因子值
        row = self.df.iloc[self.current_step]
        feat_vals = pd.to_numeric(row[self.features], errors='coerce').values.astype(np.float32)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 累加计算当前的 Net ER（归一化后传给网络，让状态值域稳定）
        raw_net_er = sum(item['er_value'] for item in self.active_queue)
        n_active = max(len(self.active_queue), 1)
        current_net_er = raw_net_er / n_active if self.reward_normalize else raw_net_er
        
        # 3. 拼接作为特征状态
        obs = np.concatenate([feat_vals, [current_net_er]], axis=0).astype(np.float32)
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # def _get_obs(self):
    #     # 1. 提取因子值（支持时序窗口）
    #     if self.sequence_window <= 1:
    #         row = self.df.iloc[self.current_step]
    #         feat_vals = pd.to_numeric(row[self.features], errors='coerce').values.astype(np.float32)
    #         feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
    #     else:
    #         n_feat = len(self.features)
    #         start = max(0, self.current_step - self.sequence_window + 1)
    #         window_df = self.df.iloc[start : self.current_step + 1]
    #         window_vals = window_df[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    #         window_vals = np.nan_to_num(window_vals, nan=0.0, posinf=0.0, neginf=0.0)

    #         if window_vals.shape[0] < self.sequence_window:
    #             pad_len = self.sequence_window - window_vals.shape[0]
    #             pad = np.zeros((pad_len, n_feat), dtype=np.float32)
    #             window_vals = np.concatenate([pad, window_vals], axis=0)
    #         feat_vals = window_vals.reshape(-1).astype(np.float32)
        
    #     # 2. 累加计算当前的 Net ER（归一化后传给网络，让状态值域稳定）
    #     raw_net_er = sum(item['er_value'] for item in self.active_queue)
    #     n_active = max(len(self.active_queue), 1)
    #     current_net_er = raw_net_er / n_active if self.reward_normalize else raw_net_er
        
    #     # 3. 拼接作为特征状态
    #     obs = np.concatenate([feat_vals, [current_net_er]], axis=0).astype(np.float32)
    #     if not np.isfinite(obs).all():
    #         obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    #     return obs
    
    def _reward_weight(self, er_value: float) -> float:
        # p=1 时等价线性 reward；p>1 时增强大动作贡献，抑制中间小动作。
        return float(np.sign(er_value) * (abs(er_value) ** self.reward_action_power))
    
    def step(self, action: np.ndarray):
        raw_action = float(action[0])
        if not np.isfinite(raw_action):
            raw_action = 0.0
        # 截断保持在 [-1, 1] 之间 # 模型已经控制[-1, 1], 这里在做异常处理
        raw_action = max(min(raw_action, 1.0), -1.0)
        
        if False: ## 转信号版本不生效
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
        opened = can_open and er_value != 0

        if self.reward_mode == "queue":
            if opened:
                self.active_queue.append({
                    'expire_step': self.current_step + self.holding_period,
                    'er_value': er_value,
                    'reward_er_value': self._reward_weight(er_value)
                })

            # 当前步骤结束后，只有大于当前时间的预测才能被带入下一个时刻 # 解决持仓时间不足
            self.active_queue = [item for item in self.active_queue if item['expire_step'] > self.current_step]

            net_er = sum(item['er_value'] for item in self.active_queue) # 净仓位
            reward_net_er_raw = sum(item.get('reward_er_value', item['er_value']) for item in self.active_queue)
            n_active = max(len(self.active_queue), 1)
            ## 标准化 防止模型训练梯度爆炸
            if self.reward_normalize:
                reward_net_er = reward_net_er_raw / n_active
                net_er_out = net_er / n_active
            else:
                reward_net_er = reward_net_er_raw
                net_er_out = net_er
            active_count = len(self.active_queue)
            # future_ret_h = float(self.future_ret_h[self.current_step]) if can_open else np.nan
        else:
            # single_horizon：当前动作直接对未来 holding_period 累计收益负责
            self.active_queue = []
            reward_net_er = self._reward_weight(er_value) if opened else 0.0
            net_er_out = er_value if opened else 0.0
            active_count = 1 if opened else 0
            future_ret_h = float(self.future_ret_h[self.current_step]) if can_open else np.nan
        
        # -------------------------------------------------------------------
        # 4. 获取当前这 1 分钟的真实市场环境收益 (即原代码的 nxt1_ret)
        # 数据集在构造中，应该确保 nxt1_ret 是从 t时刻到 t+1时刻的收益
        # -------------------------------------------------------------------
        nxt1_ret = float(self.df.iloc[self.current_step].get('nxt1_ret', 0.0))
        if not np.isfinite(nxt1_ret):
            nxt1_ret = 0.0
        
        # -------------------------------------------------------------------
        # 5. 计算奖励 (Reward)
        # queue: 用队列聚合权重对当前1步收益打分
        # single_horizon: 用当前动作直接对未来累计收益标签打分
        # -------------------------------------------------------------------
        if self.reward_mode == "single_horizon":
            target_ret_raw = future_ret_h if np.isfinite(future_ret_h) else 0.0
            baseline_ret = (
                float(self.future_ret_h_baseline[self.current_step])
                if (self.future_ret_h_baseline is not None and can_open)
                else 0.0
            )
            target_ret_excess = target_ret_raw - baseline_ret
            if self.target_mode == "raw":
                target_ret = target_ret_raw
            elif self.target_mode == "excess":
                target_ret = target_ret_excess
            else:
                target_ret = (
                    self.target_mix_alpha * target_ret_raw
                    + (1.0 - self.target_mix_alpha) * target_ret_excess
                )
        else:
            baseline_ret = 0.0
            target_ret_raw = nxt1_ret
            target_ret_excess = target_ret_raw
            target_ret = target_ret_raw

        step_reward = reward_net_er * target_ret
        if self.exposure_penalty > 0:
            step_reward -= self.exposure_penalty * (reward_net_er ** 2)
        if not np.isfinite(step_reward):
            step_reward = 0.0
        scaled_reward = step_reward * self.reward_scale
        if not np.isfinite(scaled_reward):
            scaled_reward = 0.0
        
        # ==================== 记录给 Analysis 分析的指标 ====================
        trade_time = self.df.iloc[self.current_step].get('trade_time', self.current_step)
        
        
        net_position = net_er_out
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
            'active_signals': active_count,
            'opened': opened,
            'current_ret': nxt1_ret,
            'reward_mode': self.reward_mode,
            'target_mode': self.target_mode,
            'target_ret_raw': target_ret_raw,
            'target_ret_excess': target_ret_excess,
            'target_ret': target_ret,
            'target_baseline': baseline_ret,
            'target_ret_baseline': baseline_ret,
            'reward': step_reward,
            'reward_scaled': scaled_reward,
            'future_ret_h': future_ret_h if self.reward_mode == "single_horizon" else 0,
            # 将 trade_cost 设为 0（因做纯因子预测时不在此纳入交易摩擦）
            'trade_cost': 0.0 
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), scaled_reward, done, {}
        
