import numpy as np
from collections import deque
from typing import List
from strategy import Strategy  # 引用您原有的 Strategy 基类

class RLStrategyAdapter(Strategy):
    """
    一个适配器策略，继承自您原有的Strategy基类。
    它不包含任何交易逻辑，而是作为RL Agent在回测框架中的代理。
    """
    def __init__(self, strategy_id, params={}, state_features: List[str] = None, lookback_window: int = 30):
        super().__init__(strategy_id, params)
        
        self.state_features = state_features or ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        self.lookback_window = lookback_window
        
        # 内部状态，用于RL
        self.history = deque(maxlen=self.lookback_window)
        self.last_net_value = self.portfolio.initial_capital
        self.reward = 0.0

    def reset_for_new_episode(self):
        """为新的交易日重置状态"""
        self.portfolio.__init__(
            initial_capital=self._params.get('initial_capital', 1_000_000),
            multiplier=self._params.get('multiplier', 1),
            commission_rate=self._params.get('commission_rate', {})
        )
        self.history.clear()
        self.last_net_value = self.portfolio.initial_capital
        self.reward = 0.0

    def on_bar(self, bar):
        """
        这个 on_bar 将被 RLAdapter 动态替换为一个生成器。
        原始的 on_bar 主要负责结算。
        """
        # 1. 首先执行父类的结算逻辑
        super().on_bar(bar)
        
        # 2. 计算本步的奖励（由上一步动作和本bar价格变化产生）
        current_net_value = self.portfolio.get_current_net_value()
        self.reward = current_net_value - self.last_net_value
        self.last_net_value = current_net_value
        
        # 3. 记录历史数据用于生成 state
        self.history.append(bar)

    def on_tick(self, tick):
        # RL决策是基于Bar的，所以tick级别通常什么都不做
        pass

    def get_observation(self):
        """根据历史数据生成RL的State"""
        if len(self.history) < self.lookback_window:
            return np.zeros((self.lookback_window, len(self.state_features) + 1))

        obs_data = []
        # 使用deque的切片功能来获取最新的 lookback_window 条数据
        for bar in list(self.history)[-self.lookback_window:]:
            features = [getattr(bar, f, 0) for f in self.state_features]
            obs_data.append(features)
        
        obs = np.array(obs_data, dtype=np.float32)
        
        pos_state = 0
        if len(self._long_position) > 0: pos_state = 1
        if len(self._short_position) > 0: pos_state = -1
        pos_feature = np.full((self.lookback_window, 1), pos_state, dtype=np.float32)
        
        return np.concatenate([obs, pos_feature], axis=1)
    
    def execute_action(self, action: int, current_bar):
        """执行来自RL Agent的动作"""
        if current_bar is None: return

        is_long = len(self._long_position) > 0
        is_short = len(self._short_position) > 0

        if action == 1: # Go Long
            if is_short:
                for tid, pos in list(self._short_position.items()):
                    self.order_cover(pos.symbol, current_bar.create_time, tid, current_bar.close_price, pos.volume)
            if not is_long:
                self.order_buy(current_bar.symbol, current_bar.create_time, current_bar.close_price, volume=1)
        
        elif action == 2: # Go Short
            if is_long:
                for tid, pos in list(self._long_position.items()):
                    self.order_sell(pos.symbol, current_bar.create_time, current_bar.close_price, pos.volume, tid)
            if not is_short:
                self.order_short(current_bar.symbol, current_bar.create_time, current_bar.close_price, volume=1)
