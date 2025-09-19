import pdb
from datetime import date
import pandas as pd
import numpy as np
from typing import Dict, Annotated, Union


class Portfolio(object):

    def __init__(self, symbol: str, lookback_window_size: int = 7) -> None:
        self.cur_date = None
        self.symbol = symbol
        self.action_series = {}
        self.price_series = {}
        self.rets_series = {}
        self.share_series = {}
        self.market_price = None
        self.day_count = 0
        self.holding_shares = 0
        self.lookback_window_size = lookback_window_size

    def update_market_info(self, cur_date: date, market_price: float,
                           rets: float) -> None:
        self.market_price = market_price
        self.rets = rets
        self.cur_date = cur_date
        self.day_count += 1
        self.price_series[cur_date] = market_price
        self.rets_series[cur_date] = rets

    def record_action(self, action: Dict[str, int]) -> None:
        self.holding_shares += action["direction"]
        self.action_series[self.cur_date] = action["direction"]
        self.share_series[self.cur_date] = action[
            "direction"]  # 不做累计#self.holding_shares

    def feedback(self) -> Union[Dict[str, Union[int, date]], None]:
        if self.day_count <= self.lookback_window_size:
            return None
        rets = np.array(list(self.rets_series.values()))
        share = np.array(list(self.share_series.values()))
        temp = np.cumsum(rets[-self.lookback_window_size:] *
                         share[-self.lookback_window_size:])

        if temp[-1] > 0:
            return {"feedback": 1, "date": self.cur_date}
        elif temp[-1] < 0:
            return {"feedback": -1, "date": self.cur_date}
        else:
            return {"feedback": 0, "date": self.cur_date}

    def is_refresh(self) -> bool:
        if self.day_count > self.lookback_window_size:
            return True
        else:
            return False
