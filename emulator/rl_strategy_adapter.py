import numpy as np
from typing import List
from collections import deque
from strategy import Strategy  # 引用您原有的 Strategy 基类

class RLStrategyAdapter(Strategy):
    """
    一个适配器策略，继承自您原有的Strategy基类。
    它不包含任何交易逻辑，而是作为RL Agent在回测框架中的代理。
    """
    def __init__(self, strategy_id, params={}, state_features: List[str] = None, lookback_window: int = 30):
        super().__init__(strategy_id, params)
        