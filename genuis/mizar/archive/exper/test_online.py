import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
from lib.uvx import *
from kdutils.macro2 import base_path


def fetch_raw_signal(method, instruments, task_id, period, composite_method,
                     composite_id):

    base_path1 = os.path.join(base_path, method, instruments, 'temp', 'model',
                              str(task_id), str(period), 'rl')
    dirs1 = os.path.join(base_path1, "signal", composite_method,
                         str(composite_id))
    file_path = Path(dirs1)
    res = []
    for feat_file in file_path.rglob('*.feather'):
        print(feat_file)
        signal_data = pd.read_feather(feat_file)
        name = feat_file.parts[-1].split('.')[0]
        if 'test' not in name:
            continue
        parts = feat_file.parts
        res.append((name, parts, signal_data))
    return res


def _direction_name(direction: int) -> str:
    return {1: "long", -1: "short", 0: "flat"}[direction]


@dataclass  ## 仅在平仓时判断，当前开仓信号是否同方向
class OnlineTradeRuleState2:
    hold_bars: int
    bar_index: int = -1
    long_expire_bar: int = None
    short_expire_bar: int = None

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

    def on_bar(self, trade_time, code, raw_signal):
        signal = 0 if pd.isna(raw_signal) else int(raw_signal)
        if signal not in (-1, 0, 1):
            raise ValueError(f"invalid signal: {signal}")

        if not self.advance_bar(self, trade_time):
            return

        events = []

        def expire_bar(direction):
            return (self.long_expire_bar
                    if direction == 1 else self.short_expire_bar)

        def set_expire_bar(direction, value):
            if direction == 1:
                self.long_expire_bar = value
            else:
                self.short_expire_bar = value

        # 只有当同侧仓位到期时才续仓。
        if signal != 0 and expire_bar(signal) == self.bar_index:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": 0,
                "numbers": 0,
                "position_direction": signal,
                "signal_type": "extend",
                "reason": f"extend_{_direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        # 仅平仓续期到期后仍需支付的仓位。
        for direction in (1, -1):
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
                "reason": f"expire_close_{_direction_name(direction)}",
                "expire_bar": self.bar_index,
            })

        # 仅当该信号的方向没有有效位置时才打开。
        if signal != 0 and expire_bar(signal) is None:
            set_expire_bar(signal, self.bar_index + self.hold_bars)
            events.append({
                "trade_time": trade_time,
                "code": code,
                "direction": signal,
                "numbers": 1,
                "position_direction": signal,
                "signal_type": "open",
                "reason": f"open_{_direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        return events


### 若在持仓期间同方向有新信号，则延长持仓时间
@dataclass
class OnlineTradeRuleState1:
    hold_bars: int
    bar_index: int = -1
    long_expire_bar: int = None
    short_expire_bar: int = None
    last_trade_time: str = None  ## 存储和加载要进行时间格式转化

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

    def on_bar(self, trade_time, code, raw_signal):
        signal = 0 if pd.isna(raw_signal) else int(raw_signal)
        if signal not in (-1, 0, 1):
            raise ValueError(f"invalid signal: {signal}")

        if not self.advance_bar(self, trade_time):
            return

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
                "reason": f"extend_{_direction_name(signal)}",
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
                "reason": f"expire_close_{_direction_name(direction)}",
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
                "reason": f"open_{_direction_name(signal)}",
                "expire_bar": expire_bar(signal),
            })

        return events


method = 'ricso2'
instruments = 'rbb'
task_id = '113001'
period = '5'
composite_method = 'rl'
composite_id = '1018806311332385'

raw_signal_pools = fetch_raw_signal(method=method,
                                    instruments=instruments,
                                    task_id=task_id,
                                    period=period,
                                    composite_method=composite_method,
                                    composite_id=composite_id)

raw_signal = raw_signal_pools[0][-1]

rule2 = OnlineTradeRuleState2(hold_bars=5)
rule1 = OnlineTradeRuleState1(hold_bars=5)
res1 = []
res2 = []
for signal in raw_signal.itertuples():
    rt1 = rule1.on_bar(trade_time=signal.trade_time,
                       code=signal.code,
                       raw_signal=signal.signal)
    res1.extend(rt1)

    rt2 = rule2.on_bar(trade_time=signal.trade_time,
                       code=signal.code,
                       raw_signal=signal.signal)
    res2.extend(rt2)
dt1 = pd.DataFrame(res1)
dt2 = pd.DataFrame(res2)
pdb.set_trace()
print('-->')

print('-->')
