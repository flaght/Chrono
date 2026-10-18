import pandas as pd
import pdb


class BaseOnline1(object):

    def __init__(self,
                 hold_bars,
                 bar_index=-1,
                 long_expire_bar=None,
                 short_expire_bar=None,
                 last_trade_time=None) -> None:
        self.hold_bars = hold_bars
        self.bar_index = bar_index
        self.long_expire_bar = long_expire_bar
        self.short_expire_bar = short_expire_bar
        self.last_trade_time = last_trade_time

    def advance_bar(self, trade_time):
        trade_time = pd.Timestamp(trade_time)

        if self.last_trade_time is not None:
            if trade_time == self.last_trade_time:
                return False  # 重复bar，不推进
            if trade_time < self.last_trade_time:
                raise ValueError("trade_time reversed")

        self.bar_index += 1
        self.last_trade_time = trade_time
        return True
    
    def _direction_name(self, direction: int) -> str:
        return {1: "long", -1: "short", 0: "flat"}[direction]


    ### 只有到期时出现同方向信号，才延长持仓时间
class TradeOnline1(BaseOnline1):

    def __init__(self,
                 hold_bars,
                 bar_index=-1,
                 long_expire_bar=None,
                 short_expire_bar=None,
                 last_trade_time=None) -> None:
        super(TradeOnline1, self).__init__(hold_bars=hold_bars,
                                          bar_index=bar_index,
                                          long_expire_bar=long_expire_bar,
                                          short_expire_bar=short_expire_bar,
                                          last_trade_time=last_trade_time)

    def on_bar(self, trade_time, code, raw_signal):
        signal = 0 if pd.isna(raw_signal) else int(raw_signal)
        if signal not in (-1, 0, 1):
            raise ValueError(f"invalid signal: {signal}")

        if not self.advance_bar(trade_time):
            return []
        
        events = []

        def expire_bar(direction):
            return (self.long_expire_bar
                    if direction == 1 else self.short_expire_bar)

        def set_expire_bar(direction, value):
            if direction == 1:
                self.long_expire_bar = value
            else:
                self.short_expire_bar = value
                
        if signal != 0 and expire_bar(signal) is not None:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": 0,
                "numbers": 0,
                "position_direction": signal,
                "signal_type": "extend",
                "reason": f"extend_{self._direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })
            
        for direction in (1, -1):  ## 多仓空仓是否到期
            if expire_bar(direction) != self.bar_index:
                continue
            set_expire_bar(direction, None)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": -direction,
                "numbers": 1,
                "position_direction": direction,
                "signal_type": "close",
                "reason": f"expire_close_{self._direction_name(direction)}",
                "expire_bar": self.bar_index,
            })

        if signal != 0 and expire_bar(signal) is None:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": signal,
                "numbers": 1,
                "position_direction": signal,
                "signal_type": "open",
                "reason": f"open_{self._direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        return events
    
    
### 若在持仓期间同方向有新信号，则延长持仓时间
class TradeOnline2(BaseOnline1):
    def __init__(self,
                 hold_bars,
                 bar_index=-1,
                 long_expire_bar=None,
                 short_expire_bar=None,
                 last_trade_time=None) -> None:
        super(TradeOnline2, self).__init__(hold_bars=hold_bars,
                                          bar_index=bar_index,
                                          long_expire_bar=long_expire_bar,
                                          short_expire_bar=short_expire_bar,
                                          last_trade_time=last_trade_time)
        
    def on_bar(self, trade_time, code, raw_signal):
        signal = 0 if pd.isna(raw_signal) else int(raw_signal)
        if signal not in (-1, 0, 1):
            raise ValueError(f"invalid signal: {signal}")

        if not self.advance_bar(trade_time):
            return []

        events = []

        def expire_bar(direction):
            return (self.long_expire_bar
                    if direction == 1 else self.short_expire_bar)

        def set_expire_bar(direction, value):
            if direction == 1:
                self.long_expire_bar = value
            else:
                self.short_expire_bar = value

        if signal != 0 and expire_bar(signal) == self.bar_index:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": 0,
                "numbers": 0,
                "position_direction": signal,
                "signal_type": "extend",
                "reason": f"extend_{self._direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        for direction in (1, -1):  ## 多仓空仓是否到期
            if expire_bar(direction) != self.bar_index:
                continue
            set_expire_bar(direction, None)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": -direction,
                "numbers": 1,
                "position_direction": direction,
                "signal_type": "close",
                "reason": f"expire_close_{self._direction_name(direction)}",
                "expire_bar": self.bar_index,
            })

        if signal != 0 and expire_bar(signal) is None:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": signal,
                "numbers": 1,
                "position_direction": signal,
                "signal_type": "open",
                "reason": f"open_{self._direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        return events
