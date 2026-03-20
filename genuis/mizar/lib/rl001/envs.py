import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import gym
from gym import spaces

from lib.rl001.signal import calculate_cost, to_signal, Config
from lib.logger import logger


class TradingMode(Enum):
    """交易模式"""
    LOCKED = "LOCKED" ## 锁仓模式， 股指期货对冲模式
    UNLOCK = "UNLOCK"
    
    
@dataclass
class PendingClose:
    """待平仓任务"""
    open_step: int              # 开仓时刻
    close_step: int             # 平仓时刻 = open_step + 15
    direction: int              # +1 多头, -1 空头
    signal_id: str              # 信号ID（用于追踪）
    is_locked_restore: bool = False # 锁仓模式：是否为恢复操作
    
    
@dataclass
class Signal:
    """活跃信号"""
    signal_id: str              # 唯一标识
    open_step: int              # 开仓时刻
    direction: int              # +1 多头, -1 空头
    confidence: float = 1.0     # 置信度
    

class PositionManager(object):
    def __init__(self, mode, max_pairs: int=50, 
                 max_allowed_position: int =10, 
                 holding_period: int = 15):
        self.mode = mode
        self.max_pairs = max_pairs
        self.max_allowed_position = max_allowed_position
        self.holding_period = holding_period

        self.remaining_pairs = max_pairs if self.mode == TradingMode.LOCKED else None
        
        # 持仓状态
        self.pending_list: List[PendingClose] = []  # 待平仓队列
        self.active_signals: List[Signal] = []       # 活跃信号列表
        self.long_positions: int = 0                  # 多头持仓数
        self.short_positions: int = 0                 # 空头持仓数
        self.net_position: int = 0                   # 净持仓 = long - short

        self.last_trade_step: Dict[str, Optional[int]] = {
            'long': None,
            'short': None
        }
        
        
        if mode == TradingMode.LOCKED:
            self.long_positions = max_pairs
            self.short_positions = max_pairs
            self.net_position = 0
            
    def state_features(self)->Dict[str, float]:
        """获取状态特征"""
        features = {
            'net_position_ratio': self.net_position / self.max_allowed_position,
            'active_signals_count': len(self.active_signals) / self.max_allowed_position,  # 归一化
        }

        if self.mode == TradingMode.LOCKED:
            features['remaining_pairs_ratio'] = self.remaining_pairs / self.max_pairs
        return features
    
    def can_open_long(self, current_step:int, cooldown_steps:int =0,
                      effective_max_position: Optional[int] = None) -> bool:
        """检查是否可以开多头仓"""
        ## 锁仓模式： 检查剩余昨仓
        if self.mode == TradingMode.LOCKED:
            if self.remaining_pairs == 0:
                return False
        
        # 风险控制：检查净持仓上限（使用调整后的阈值）
        max_pos = effective_max_position if effective_max_position is not None else self.max_allowed_position
        if self.net_position >= max_pos:
            return False
        
        # Cooldown机制
        if cooldown_steps > 0 and self.last_trade_step['long'] is not None:
            if current_step - self.last_trade_step['long'] < cooldown_steps:
                return False
        return True
    
    def can_open_short(self, current_step: int, cooldown_steps: int = 0, 
                       effective_max_position: Optional[int] = None):
        """检查是否可以开空头仓"""
        # 锁仓模式：检查剩余昨仓
        if self.mode == TradingMode.LOCKED:
            if self.remaining_pairs == 0:
                return False
        
        # 风险控制：检查净持仓上限（使用调整后的阈值）
        max_pos = effective_max_position if effective_max_position is not None else self.max_allowed_position
        if self.net_position <= -max_pos:
            return False
        
        # Cooldown机制
        if cooldown_steps > 0 and self.last_trade_step['short'] is not None:
            if current_step - self.last_trade_step['short'] < cooldown_steps:
                return False
        
        return True
    
    def open_position(self, direction: int, current_step: int,
                      signal_id: Optional[str] = None,
                      confidence: float = 1.0) -> bool:
        """开仓"""
        if direction == 1:  # 多头
            if not self.can_open_long(current_step):
                return False
            ## 锁仓模式: 平一手空仓(昨仓)
            if self.mode == TradingMode.LOCKED:
                if self.short_positions > 0:
                    self.short_positions -= 1
                    self.remaining_pairs -= 1
                    self.net_position += 1
                else:
                    return False
            elif self.mode == TradingMode.UNLOCK:
                self.long_positions += 1
                self.net_position += 1
            
            self.last_trade_step['long'] = current_step
        elif direction == -1:  # 空头
            if not self.can_open_short(current_step):
                return False
            
            ## 锁仓模式: 平一手空仓(昨仓)
            if self.mode == TradingMode.LOCKED:
                if self.long_positions > 0:
                    self.long_positions -= 1
                    self.remaining_pairs -= 1
                    self.net_position -= 1
                else:
                    return False
            elif self.mode == TradingMode.UNLOCK:
                self.short_positions += 1
                self.net_position -= 1

            self.last_trade_step['short'] = current_step
        
        else:
            return False
        
        # 创建待平仓任务
        if signal_id is None:
            signal_id = f"signal_{current_step}_{direction}"
        
        pending_close = PendingClose(
            open_step=current_step,
            close_step=current_step + self.holding_period,
            direction=direction,
            signal_id=signal_id,
            is_locked_restore=False
        )
        self.pending_list.append(pending_close)
        
        # 创建活跃信号
        signal = Signal(
            signal_id=signal_id,
            open_step=current_step,
            direction=direction,
            confidence=confidence
        )
        self.active_signals.append(signal)
        
        return True
    
    def close_position(self, pending_close: PendingClose, current_step: int) -> bool:
        """平仓（处理到期平仓任务）"""
        direction = pending_close.direction
        if self.mode == TradingMode.LOCKED:
            # 锁仓模式（日内）：到期平仓可恢复净敞口，但不恢复 remaining_pairs
            # 这样每个交易对在当日只可“消耗”一次，避免日内循环交易。
            if direction == 1:  # 平多头 -> 开回空仓
                self.short_positions += 1
                self.net_position -= 1
            elif direction == -1:  # 平空头 -> 开回多仓
                self.long_positions += 1
                self.net_position += 1
        else:
            if direction == 1:  # 平多仓
                if self.long_positions > 0:
                    self.long_positions -= 1
                    self.net_position -= 1
            elif direction == -1: # 平空仓
                if self.short_positions > 0:
                    self.short_positions -= 1
                    self.net_position += 1
        
        # 移除待平仓任务和活跃信号
        if pending_close in self.pending_list:
            self.pending_list.remove(pending_close)
        
        # 移除对应的活跃信号
        self.active_signals = [
            s for s in self.active_signals
            if s.signal_id != pending_close.signal_id
        ]
        
        return True
    
    def process_expired_positions(self, current_step: int) -> List[PendingClose]:
        """处理到期的持仓"""
        expired = []
        remaining = []
        
        for pending in self.pending_list:
            if pending.close_step <= current_step:
                expired.append(pending)
            else:
                remaining.append(pending)
        
        self.pending_list = remaining
        
        # 平仓处理
        for pending in expired:
            self.close_position(pending, current_step)
        
        return expired

    def replenish_daily_pairs(self):
        """
        仅用于 LOCKED 模式的日切补仓：
        - 日内 remaining_pairs 只减不增
        - 到新交易日时统一恢复到 max_pairs
        """
        if self.mode != TradingMode.LOCKED:
            return

        self.pending_list = []
        self.active_signals = []
        self.last_trade_step = {'long': None, 'short': None}
        self.remaining_pairs = self.max_pairs
        self.long_positions = self.max_pairs
        self.short_positions = self.max_pairs
        self.net_position = 0
    
    def reset(self):
        """重置持仓状态"""
        self.pending_list = []
        self.active_signals = []
        self.last_trade_step = {'long': None, 'short': None}

        if self.mode == TradingMode.LOCKED:
            self.remaining_pairs = self.max_pairs
            self.long_positions = self.max_pairs
            self.short_positions = self.max_pairs
            self.net_position = 0
        else:
            self.long_positions = 0
            self.short_positions = 0
            self.net_position = 0
            
class TradingEnv:
    """
    支持 SAC 的交易环境
    """
    def __init__(self, df:pd.DataFrame,
                 features: List[str],
                 mode: str = "LOCKED",
                 holding_period:int= 15,
                 max_pairs: int =50,
                 max_allowed_position: int=10,
                 use_cooldown: bool = True,
                 cooldown_steps: int = 3,
                 masking_threshold_multiplier:float = 1.0,
                 episode_len: int = 500,
                 start_time: Optional[int] = None,
                 seed: Optional[int] = None,
                 cost_rate: Optional[float] = None,
                 reward_scale: float = 10000.0,
                 signal_config: Optional[Config] = None,
                 obs_noise_std: float = 0.0,
                 debug_daily_logs: bool = False
        ):
        """
        初始化 SAC 交易环境
        
        Args:
            signal_config: 信号转换配置，如果为None，使用默认配置
        """
        self.df = df.copy()
        self.df = self.df.sort_values('trade_time').reset_index(drop=True)
        self.df["trade_time"] = pd.to_datetime(self.df["trade_time"])
        self.features = features
        self.mode = TradingMode.LOCKED if mode.upper() == "LOCKED" else TradingMode.UNLOCK
        self.holding_period = holding_period
        self.max_pairs = max_pairs
        self.max_allowed_position = max_allowed_position
        self.use_cooldown = use_cooldown
        self.cooldown_steps = cooldown_steps if use_cooldown else 0
        self.masking_threshold_multiplier = masking_threshold_multiplier
        self.episode_len = episode_len
        
        self.reward_scale = reward_scale
        
        if cost_rate is None:
            fallback_cost = signal_config.base_cost if signal_config is not None else 0.0001
            self.cost_rate = float(fallback_cost)
        else:
            self.cost_rate = float(cost_rate)
        
        self.signal_config = signal_config
        self.obs_noise_std = obs_noise_std  # 观测噪声标准差（防过拟合）
        self.debug_daily_logs = bool(debug_daily_logs)
        self.signal_config.base_cost = self.cost_rate
        self.signal_config.min_confidence = self.signal_config.cost_multiplier * self.cost_rate
        
        if 'nxt1_ret' not in self.df.columns:
            raise ValueError("数据必须包含 'nxt1_ret' 列（分钟收益率）")
        if self.df['trade_time'].duplicated().any():
            raise ValueError("trade_time 存在重复值；当前环境要求单标的且每个 trade_time 唯一一行")
        
        self.effective_max_position = int(max_allowed_position * masking_threshold_multiplier)
        
        self.masking_stats = {
            'total_steps': 0,
            'masking_triggered': {
                'long': 0,
                'short': 0,
                'locked_pairs': 0,
                'cooldown': 0
            }
        }
        
        self.max_time_index = len(self.df) - 1
        self.trade_dates = self.df["trade_time"].dt.date.values
        self.day_end_index = np.zeros(len(self.df), dtype=np.int32)
        day_end = self.max_time_index
        for i in range(self.max_time_index, -1, -1):
            if i == self.max_time_index or self.trade_dates[i] != self.trade_dates[i + 1]:
                day_end = i
            self.day_end_index[i] = day_end

        self.position_manager = PositionManager(
            mode=self.mode, max_pairs=max_pairs,
            max_allowed_position=self.effective_max_position,
            holding_period=holding_period
        )
        
        self.current_step = 0
        self.current_time_index = 0
        self.episode_step_count = 0
        self.terminal = False
        self._last_trade_date = None
        self._daily_opened_total = 0
        self._daily_opened_long = 0
        self._daily_opened_short = 0
        self._daily_blocked_eod = 0
        self._daily_signal_nonzero = 0
        self._daily_remaining_pairs_start = 0

        self.n_features = len(features)
        self.n_state_features = self.state_dim()
        
        
        # SAC 动作空间：连续 [long_score, short_score]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32
        )
        
        if seed is not None:
            np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        
        self.start_time_index = start_time
        
        # 添加 spec 和 metadata 属性以兼容 Stable-Baselines3
        self.spec = None
        self.metadata = {'render_modes': []}
        
    def state_dim(self)->int:
        dim = self.n_features
        
        if self.mode == TradingMode.LOCKED: ## 锁仓状态
            dim += 1 
            
        dim += 2 ## 持仓状态特征
        
        return dim
        
        
    def observation(self, time_index: int)->np.ndarray:
        row = self.df.iloc[time_index]
        
        # 确保 factor_features 是1D数组
        factor_features = row[self.features].values.astype(np.float32)
        if factor_features.ndim > 1:
            factor_features = factor_features.flatten()
        
        # 训练时加入观测噪声（防过拟合）
        if self.obs_noise_std > 0:
            noise = self.np_random.randn(*factor_features.shape).astype(np.float32) * self.obs_noise_std
            factor_features = factor_features + noise
            
        position_features = self.position_manager.state_features()
        
        state_list = [factor_features]
        
        if self.mode == TradingMode.LOCKED:
            state_list.append(np.array([position_features['remaining_pairs_ratio']], dtype=np.float32))
            
        state_list.append(np.array([
            position_features['net_position_ratio'],
            position_features['active_signals_count']
        ], dtype=np.float32))
        
        observation = np.concatenate([arr.flatten() for arr in state_list]).astype(np.float32)
        # 验证维度
        expected_dim = self.n_state_features
        if observation.shape != (expected_dim,):
            raise ValueError(
                f"Observation shape mismatch: expected ({expected_dim},), "
                f"got {observation.shape}. State dim calculation may be incorrect."
            )
        
        return observation
    
    def reset(self, start_time_index:Optional[int] = None):
        """重置环境"""
        self.position_manager.reset()
        self.reset_masking_stats()
        self._last_direction = 0

        if start_time_index is not None:
            self.start_time_index = start_time_index
        elif self.start_time_index is not None:
            pass
        else:
            max_start = self.max_time_index - self.episode_len - self.holding_period
            self.start_time_index = self.np_random.randint(
                self.holding_period,
                max(max_start, self.holding_period + 1)
            )
            
        self.current_time_index = self.start_time_index
        self.current_step = 0
        self.episode_step_count = 0
        self.terminal = False
        self._last_trade_date = self.trade_dates[self.current_time_index]
        self._init_daily_debug(self._last_trade_date)
        
        observation = self.observation(self.current_time_index)
        
        info = {
            'current_step': int(self.current_step),
            'start_time_index': int(self.start_time_index) if self.start_time_index is not None else 0,
            'masking_threshold_multiplier': float(self.masking_threshold_multiplier),
            'effective_max_position': int(self.effective_max_position),
            'cost_rate': float(self.cost_rate),
            'reward_scale': float(self.reward_scale),
            'threshold_mode': str(self.signal_config.threshold_mode),
            'threshold': float(self.signal_config.threshold),
            'base_cost': float(self.signal_config.base_cost),
            'cost_mode': str(self.signal_config.cost_mode)
        }
        
        return observation, info

    def _init_daily_debug(self, trade_date):
        self._daily_opened_total = 0
        self._daily_opened_long = 0
        self._daily_opened_short = 0
        self._daily_blocked_eod = 0
        self._daily_signal_nonzero = 0
        if self.mode == TradingMode.LOCKED:
            self._daily_remaining_pairs_start = int(self.position_manager.remaining_pairs)
        else:
            self._daily_remaining_pairs_start = 0

    def _log_daily_debug(self, trade_date):
        if not self.debug_daily_logs:
            return

        if self.mode == TradingMode.LOCKED:
            remaining_pairs_end = int(self.position_manager.remaining_pairs)
            consumed_pairs = int(self._daily_remaining_pairs_start - remaining_pairs_end)
        else:
            remaining_pairs_end = 0
            consumed_pairs = 0

        logger.info(
            f"[LOCKED-DAY] date={trade_date} "
            f"非 0 信号次数={self._daily_signal_nonzero} "
            f"实际成功开出的仓位数量={self._daily_opened_total} "
            f"成功执行的“开多方向”次数={self._daily_opened_long} "
            f"成功执行的“开空方向”次数={self._daily_opened_short} "
            f"hold_period 不足拦截信号={self._daily_blocked_eod} "
            f"开始时可用的配对额度={self._daily_remaining_pairs_start} "
            f"当天结束时剩余可用配对额度={remaining_pairs_end} "
            f"当天消耗的配对数={consumed_pairs} "
            f"结束时净仓位={int(self.position_manager.net_position)}"
        )
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # 确保 action 是正确的 numpy 数组格式
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        else:
            action = action.astype(np.float32)
            
        # 确保 action 是1D数组且长度为2
        if action.ndim > 1:
            action = action.flatten()
        if len(action) != 2:
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        
         # 检查是否结束
        self.episode_step_count += 1
        if self.episode_step_count > self.episode_len:
            self.terminal = True
            
        if self.current_time_index > self.max_time_index:
            self.terminal = True
        
        if self.terminal:
            observation = self.observation(self.current_time_index)
            return observation, 0.0, True, {}
        
        
        # 获取当前时刻的 nxt1_ret
        current_ret = self.df.iloc[self.current_time_index]['nxt1_ret']
        
        # ========== 1. 计算持仓即时收益 ==========
        reward = 0.0
        step_trade_cost = 0.0
        for signal in self.position_manager.active_signals:
            # 使用信号强度（confidence）加权收益
            reward += signal.direction * signal.confidence * current_ret
        
        # ========== 2. 处理到期平仓 ==========
        # 在处理到期平仓之前，先保存 confidence 信息
        expired_pending = []
        for pending in self.position_manager.pending_list:
            if pending.close_step <= self.current_step:
                # 找到对应的信号以获取 confidence
                signal = next((s for s in self.position_manager.active_signals 
                              if s.signal_id == pending.signal_id), None)
                confidence = signal.confidence if signal is not None else 1.0
                expired_pending.append((pending, confidence))
        
        # 处理到期平仓
        expired = self.position_manager.process_expired_positions(self.current_step)
        
        # 计算平仓成本（支持比例成本）
        for pending, confidence in expired_pending:
            # 计算成本（支持固定或比例）
            close_cost = calculate_cost(
                signal=pending.direction * confidence,
                confidence=confidence,
                config=self.signal_config
            )
            reward -= close_cost
            step_trade_cost += close_cost
            
        # ========== 3. 将 SAC 动作转换为交易信号 ==========
        # 获取净持仓（用于 position_risk 模式）
        net_position = float(self.position_manager.net_position)
        max_position = float(self.effective_max_position)
        
        signal, confidence, direction = to_signal(
            action=action,
            config=self.signal_config,
            net_position=net_position if self.signal_config.threshold_mode == 'position_risk' else None,
            max_position=max_position if self.signal_config.threshold_mode == 'position_risk' else None
        )
        if direction != 0:
            self._daily_signal_nonzero += 1
        
        # ========== 3.5 方向翻转惩罚 ==========
        if not hasattr(self, '_last_direction'):
            self._last_direction = 0
        if direction != 0 and self._last_direction != 0 and direction != self._last_direction:
            reward -= 0.0005  # 从做多翻做空（或反过来），额外扣罚
        if direction != 0:
            self._last_direction = direction
        
        # ========== 4. 处理新信号开仓 ==========
        opened = False
        if direction != 0:
            # 检查是否可以开仓（基于方向）
            can_open = False
            # LOCKED 模式：确保新开仓可以完整持有 holding_period（不跨日）
            if self.mode == TradingMode.LOCKED:
                bars_left_same_day = int(self.day_end_index[self.current_time_index] - self.current_time_index)
                if bars_left_same_day < self.holding_period:
                    can_open = False
                    self._daily_blocked_eod += 1
                else:
                    if direction == 1:
                        can_open = self.position_manager.can_open_long(
                            self.current_step,
                            self.cooldown_steps,
                            self.effective_max_position
                        )
                    elif direction == -1:
                        can_open = self.position_manager.can_open_short(
                            self.current_step,
                            self.cooldown_steps,
                            self.effective_max_position
                        )
            else:
                if direction == 1:
                    can_open = self.position_manager.can_open_long(
                        self.current_step,
                        self.cooldown_steps,
                        self.effective_max_position
                    )
                elif direction == -1:
                    can_open = self.position_manager.can_open_short(
                        self.current_step,
                        self.cooldown_steps,
                        self.effective_max_position
                    )
            
            if can_open:
                success = self.position_manager.open_position(
                    direction=direction,
                    current_step=self.current_step,
                    signal_id=f"signal_{self.current_step}_{direction}",
                    confidence=confidence
                )
                if success:
                    opened = True
                    self._daily_opened_total += 1
                    if direction == 1:
                        self._daily_opened_long += 1
                    elif direction == -1:
                        self._daily_opened_short += 1
                    # 计算开仓成本（支持固定或比例）
                    open_cost = calculate_cost(
                        signal=signal,
                        confidence=confidence,
                        config=self.signal_config
                    )
                    reward -= open_cost
                    step_trade_cost += open_cost
        
        # ========== 5. 前进一步 ==========
        prev_time_index = self.current_time_index
        self.current_step += 1
        self.current_time_index += 1
        
        # ========== 6. 获取下一步数据 ==========
        if self.current_time_index > self.max_time_index:
            self.terminal = True
            observation = self.observation(min(self.current_time_index, self.max_time_index))
        else:
            # 日切时：LOCKED 模式统一补足交易对库存，清空日内任务
            prev_date = self.trade_dates[prev_time_index]
            next_date = self.trade_dates[self.current_time_index]
            if self.mode == TradingMode.LOCKED and next_date != prev_date:
                self._log_daily_debug(prev_date)
                self.position_manager.replenish_daily_pairs()
                self._init_daily_debug(next_date)

            observation = self.observation(self.current_time_index)
        
        # 应用奖励缩放
        reward_scaled = reward * self.reward_scale
        
        # 信息字典 - 确保所有值都是基本类型（int, float, bool, str），避免嵌套结构
        info = {
            'current_step': int(self.current_step),
            'time_index': int(self.current_time_index),
            'net_position': int(self.position_manager.net_position),
            'active_signals': int(len(self.position_manager.active_signals)),
            'opened': bool(opened),
            'expired_count': int(len(expired)),
            'signal': float(signal),                    # 转换后的信号
            'confidence': float(confidence),            # 置信度
            'direction': int(direction),              # 方向
            'reward_raw': float(reward),                # 原始奖励
            'reward_scaled': float(reward_scaled),      # 缩放后的奖励
            'current_ret': float(current_ret),
            'cost_rate': float(self.cost_rate),
            'trade_cost': float(step_trade_cost),       # 当步真实手续费（开+平）
        }
        
        if self.mode == TradingMode.LOCKED:
            info['remaining_pairs'] = int(self.position_manager.remaining_pairs)
            if self.debug_daily_logs:
                info["daily_opened_total"] = int(self._daily_opened_total)
                info["daily_opened_long"] = int(self._daily_opened_long)
                info["daily_opened_short"] = int(self._daily_opened_short)
                info["daily_blocked_eod"] = int(self._daily_blocked_eod)
                info["daily_signal_nonzero"] = int(self._daily_signal_nonzero)
                info["daily_remaining_pairs_start"] = int(self._daily_remaining_pairs_start)

        # 回合终止时，输出最后一个交易日的汇总
        if self.terminal and self.mode == TradingMode.LOCKED:
            final_idx = min(prev_time_index, self.max_time_index)
            self._log_daily_debug(self.trade_dates[final_idx])
        
        return observation, reward_scaled, self.terminal, info
            
            
    def reset_masking_stats(self):
        """重置Action Masking统计信息"""
        self.masking_stats = {
            'total_steps': 0,
            'masking_triggered': {
                'long': 0,
                'short': 0,
                'locked_pairs': 0,
                'cooldown': 0
            }
        }
        
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
    
    def close(self):
        """关闭环境"""
        pass
    
    def __repr__(self) -> str:
        return (
            f"TradingEnvSAC(mode={self.mode.value}, "
            f"n_features={self.n_features}, "
            f"n_state_dim={self.n_state_features}, "
            f"holding_period={self.holding_period}, "
            f"cost_rate={self.cost_rate}, "
            f"reward_scale={self.reward_scale}, "
            f"threshold_mode={self.signal_config.threshold_mode})"
        )
