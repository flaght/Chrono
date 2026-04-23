import pandas as pd
import numpy as np
from typing import Tuple, Optional

class PositionBacktester(object):
    
    def __init__(self, market_data: pd.DataFrame,
                 contract_multiplier: float,
                 slippage: float = 0.2,
                 initial_capital: float = 5e7,
                 base_position: int = 10):
        
        self.market_data = market_data.copy()
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        self.base_position = base_position  # 对锁底仓手数
        
        self.trade_records = None
        self.daily_stats = None
      
    def _prepare_data(self):
        pass
    
    
    def fetch_daily_open(self, date: int) -> Optional[float]:
        pass
        
    def run(self, position_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        position_df = position_df.copy()
        
        # 交易记录
        all_trade_records = []
        
        # 日度统计
        all_daily_stats = []
        
        # 累计 PnL
        cumulative_pnl = 0.0
        
        