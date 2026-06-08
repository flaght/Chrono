import numpy as np
import pandas as pd
import gym, random, pdb
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
        self.action_change_penalty = float(self.env_config.get("action_change_penalty", 0.0))
        self.action_change_deadzone = float(self.env_config.get("action_change_deadzone", 0.0))
        if self.action_change_deadzone < 0.0:
            self.action_change_deadzone = 0.0
        
        self.min_open_signal_abs = max(0.0, float(self.signal_config.get("min_open_signal_abs", 0.0)))
        self.mode = str(self.env_config["mode"]).strip().lower()
        self.target_mode = str(self.env_config["target_mode"]).strip().lower()
        self.target_cost_rate = self.env_config["target_cost_rate"]
        self.target_cost_mult = self.env_config["target_cost_mult"]
        
        
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
        self.target_ret_h = self._build_target_returns(self.future_ret_h)
        
        
        valid_target = self.target_ret_h[np.isfinite(self.target_ret_h)]
        self._ret_scale = float(np.std(valid_target)) + 1e-8

        self.prev_net_er_out = 0.0
        self.history = []
        
        
        self.max_episode_steps = int(self.env_config["max_episode_steps"])
        if self.max_episode_steps < 0:
            self.max_episode_steps = 0
            
        
        self.random_start_min_step = 0
        max_valid_start = self.last_open_step
        if self.max_episode_steps > 0:
            max_valid_start = max(0, self.last_open_step - self.max_episode_steps + 1)
        self.random_start_max_step = max_valid_start
        
        self.softmax_temperature = float(self.env_config["softmax_temperature"])
        
        self.train_scheme = str(self.env_config["train_scheme"]).strip().lower()
        
        self.train_window_starts: List[int] = []
        self.train_window_order: List[int] = []
        self.train_window_cursor = 0
        
        self.episode_end_step_exclusive = self.last_open_step + 1
        self.seed(seed=42)
        
        
        self.reset_count = 0  
        self.debug_reset_log = True
        self.debug_reset_log_every = 5
        
        
        if self.mode == "train":
            if self.train_scheme == "full":
                self.train_window_starts = self._build_window_starts_full_stride()
            elif self.train_scheme == "half":
                self.train_window_starts = self._build_window_starts_half_stride()
            elif self.train_scheme == "holding":
                self.train_window_starts = self._build_window_starts_holding_stride()
            if self.train_window_starts:
                self._reset_train_window_order()
                
            if self.debug_reset_log:
                print(
                    "[ENV_INIT][train] scheme={0} max_episode_steps={1} windows={2} start_range=[{3},{4}]".format(
                    self.train_scheme,
                    self.max_episode_steps,
                    len(self.train_window_starts),
                    self.random_start_min_step,
                    self.random_start_max_step)
                )
        
        elif self.debug_reset_log:
            print(
                    "[ENV_INIT][{0}] full_eval=True start=0 end_exclusive={1} tradable_len={2}".format(
                    self.mode,
                    self.last_open_step + 1,
                    self.last_open_step + 1)
            )
        
    
    def _build_window_starts_by_stride(self, stride: int) -> List[int]:
        stride = max(1, int(stride))
        starts = list(range(self.random_start_min_step, self.random_start_max_step + 1, stride))
        if not starts:
            starts = [self.random_start_min_step]
        if starts[-1] != self.random_start_max_step:
            starts.append(self.random_start_max_step)
        return starts
    
    def _build_window_starts_full_stride(self) -> List[int]:
        # 方案1：stride = max_episode_steps（不重叠）
        return self._build_window_starts_by_stride(self.max_episode_steps)

    def _build_window_starts_half_stride(self) -> List[int]:
        # 方案2：stride = max_episode_steps // 2（半重叠）
        return self._build_window_starts_by_stride(max(1, self.max_episode_steps // 2))

    def _build_window_starts_holding_stride(self) -> List[int]:
        # 方案3：stride = holding_period（按持有周期）
        return self._build_window_starts_by_stride(max(1, self.holding_period))
    
    def _reset_train_window_order(self):
        n = len(self.train_window_starts)
        self.train_window_order = list(range(n))
        if n > 1:
            self.np_random.shuffle(self.train_window_order)
        self.train_window_cursor = 0
        
        
    def _next_train_window_start(self) -> int:
        if not self.train_window_starts:
            raise ValueError("train_window_starts 为空，无法获取下一个训练窗口起点。")
        if self.train_window_cursor >= len(self.train_window_order):
            self._reset_train_window_order()
        if not self.train_window_order:
            raise ValueError("train_window_order 为空，无法获取下一个训练窗口起点。")
        idx = self.train_window_order[self.train_window_cursor]
        self.train_window_cursor += 1
        return int(self.train_window_starts[idx])
        
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        # if seed is not None:
        #     self.seed(seed)
        # self.current_step = 0
        # # self.current_step = int(self.np_random.randint(self.random_start_min_step, self.random_start_max_step + 1))
        # self.history = []
        # return self._get_obs()
        if seed is not None:
            self.seed(seed)
        
        if self.mode == "train":
            self.current_step = self._next_train_window_start()
        else:
            self.current_step = 0
        
        
        if self.max_episode_steps > 0 and self.mode == "train":
            self.episode_end_step_exclusive = min(
                self.last_open_step + 1,
                self.current_step + self.max_episode_steps
            )
        else:
            #self.episode_end_step_exclusive = len(self.df) - 1
            self.episode_end_step_exclusive = self.last_open_step + 1 # 这个更合理， 因为数据尾部 有无效奖励
        
        self.reset_count += 1
        self._log_reset_window()
        self.history = []
        self.prev_net_er_out = 0.0
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
    
    def _build_target_returns(self, future_ret_h: np.ndarray) -> np.ndarray:
        """
        构建训练用 target。默认 raw；effective 模式只保留扣除成本门槛后仍有幅度的收益。
        """
        target = np.array(future_ret_h, dtype=np.float64, copy=True)
        if self.target_mode == "raw":
            return target
        if self.target_mode == "effective":
            threshold = self.target_cost_rate * self.target_cost_mult
            if threshold <= 0.0:
                return target
            abs_ret = np.abs(target)
            target = np.sign(target) * np.maximum(abs_ret - threshold, 0.0)
            return target
        raise ValueError(
            "target_mode must be one of {'raw', 'effective'}"
        )

        
    def _get_obs(self):
        pdb.set_trace()
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
        temperature_scaled_action = raw_action * self.softmax_temperature
        exp_action = np.exp(temperature_scaled_action - np.max(temperature_scaled_action)) # 减最大值防止溢出
        softmax_probs = exp_action / np.sum(exp_action)
    
        # chosen_action = int(np.argmax(softmax_probs))
        # if chosen_action == 0:
        #     er_value = 0.0
        #     confidence = float(softmax_probs[0])
        # elif chosen_action == 1:
        #     confidence = float(softmax_probs[1])
        #     er_value = confidence
        # else:
        #     confidence = float(softmax_probs[2])
        #     er_value = -confidence
        
        ## er_value 为 多空之差，单方越强。
        if float(softmax_probs[0]) > float(softmax_probs[1]) and float(softmax_probs[0]) > float(softmax_probs[2]):
            er_value = 0.0
            confidence = float(softmax_probs[0])
        else:
            er_long_short = float(softmax_probs[1]) - float(softmax_probs[2])
            er_value = er_long_short
            confidence = float(abs(er_value))
        
        raw_action_str = f"[{raw_action[0]:.4f},{raw_action[1]:.4f},{raw_action[2]:.4f}]"
        soft_action_str = f"[{softmax_probs[0]:.4f},{softmax_probs[1]:.4f},{softmax_probs[2]:.4f}]"
        
        can_open = self.current_step <= self.last_open_step
        opened = can_open and (abs(float(er_value)) >= self.min_open_signal_abs) and (er_value != 0) ## 多空必须差很多，才能开仓
        # opened = can_open and er_value != 0
        
        net_er_out = er_value if opened else 0.0
        active_count = 1 if opened else 0
        future_ret_h = float(self.future_ret_h[self.current_step]) if can_open else np.nan
        
        nxt1_ret = float(self.df.iloc[self.current_step].get('nxt1_ret', 0.0))
        if not np.isfinite(nxt1_ret):
            nxt1_ret = 0.0
        
        target_ret_h = float(self.target_ret_h[self.current_step]) if can_open else np.nan
        target_ret_raw = target_ret_h if np.isfinite(target_ret_h) else 0.0
        # target_ret = target_ret_raw
        target_ret = target_ret_raw / self._ret_scale

        
        step_reward = net_er_out * target_ret
        
        if not np.isfinite(step_reward):
            step_reward = 0.0
        
        ## 诱导模型只对方向开仓抑制，从而降低换手率
        if self.action_change_penalty > 0:
            prev_sign = 1 if self.prev_net_er_out > 0 else (-1 if self.prev_net_er_out < 0 else 0)
            curr_sign = 1 if net_er_out > 0 else (-1 if net_er_out < 0 else 0)
            is_flip = (prev_sign != 0) and (curr_sign != 0) and (prev_sign != curr_sign)
            delta = float(abs(net_er_out - self.prev_net_er_out))
            delta_excess = max(0.0, delta - self.action_change_deadzone)
            if is_flip and delta_excess > 0.0:
                step_reward -= self.action_change_penalty * delta_excess
        
            
        scaled_reward = step_reward * self.reward_scale
        self.prev_net_er_out = float(net_er_out)
        
        
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
            'target_mode': self.target_mode,
            'target_cost_rate': self.target_cost_rate,
            'target_cost_mult': self.target_cost_mult,
            'target_ret_raw': target_ret_raw,
            'target_ret': target_ret,
            'reward': step_reward,
            'reward_scaled': scaled_reward,
            'future_ret_h': future_ret_h,
            # 将 trade_cost 设为 0（因做纯因子预测时不在此纳入交易摩擦）
            'trade_cost': 0.0 
        })
        
        self.current_step = min(self.current_step + 1, len(self.df) - 1)
        done = self.current_step >= self.episode_end_step_exclusive
        
        return self._get_obs(), scaled_reward, done, {}
        
        

    def _log_reset_window(self):
        if not self.debug_reset_log:
            return
        if self.reset_count % self.debug_reset_log_every != 0:
            return

        start_idx = int(self.current_step)
        end_exclusive = int(self.episode_end_step_exclusive)
        window_len = int(max(0, end_exclusive - start_idx))
        start_time = self.df.iloc[start_idx].get("trade_time", start_idx)
        end_label_idx = min(max(0, end_exclusive - 1), len(self.df) - 1)
        end_time = self.df.iloc[end_label_idx].get("trade_time", end_label_idx)

        if self.mode == "train":
            print(
                "[ENV_RESET][train] reset={0} start={1} end_exclusive={2} len={3} "
                "cursor={4}/{5} time=[{6} ->{7}]".format(
                self.reset_count,
                start_idx,
                end_exclusive,
                window_len,
                self.train_window_cursor,
                len(self.train_window_order),
                start_time,
                end_time)
            )
        else:
            print(
                "[ENV_RESET][{0}] reset={1} full_eval=True start={2} end_exclusive={3} len={4} "
                "time=[{5} -> {6}]".format(
                self.mode,
                self.reset_count,
                start_idx,
                end_exclusive,
                window_len,
                start_time,
                end_time)
            )