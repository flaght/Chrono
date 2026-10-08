"""同品种两个期限角色的价差均值回归；真实合约目标按日切换。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from calendar_signal import CalendarSpreadSignal
from bomber.framework.market.basic.base import Bar, InstrumentId
from bomber.framework.trader.template import StrategyTemplate


@dataclass(frozen=True)
class CalendarSelection:
    """每日两腿真实合约；默认第一腿是次主力，第二腿是 far，不存在第三条 near 腿。"""
    day: object
    effective_ns: int
    leg1_symbol: str
    leg2_symbol: str
    old_pair_bars_available: bool = True

    def __post_init__(self) -> None:
        if self.effective_ns < 0 or not self.leg1_symbol or not self.leg2_symbol:
            raise ValueError("期限角色记录无效")
        if self.leg1_symbol.lower() == self.leg2_symbol.lower():
            raise ValueError("两个期限角色不能指向同一真实合约")


@dataclass(frozen=True)
class CalendarSpreadConfig:
    """角色按第一腿、第二腿配置；角色名称与实际到期月份的先后不等价。"""
    leg1_role: str = "secondary"
    leg2_role: str = "far"
    quantity: Decimal = Decimal(1)
    lookback: int = 120
    entry_z: float = 2.0
    exit_z: float = 0.5
    rebalance_interval: int = 5

    def __post_init__(self) -> None:
        amount = Decimal(str(self.quantity))
        if not amount.is_finite() or amount <= 0 or amount != amount.to_integral_value():
            raise ValueError("CTP 目标手数必须为正整数")
        if (self.leg1_role not in {"main", "secondary"} or self.leg2_role not in {"secondary", "far"}
                or self.leg1_role == self.leg2_role or self.lookback < 2
                or self.rebalance_interval < 1):
            raise ValueError("期限角色须不同，窗口至少2根，调仓间隔须为正")
        if not isfinite(self.entry_z) or not isfinite(self.exit_z) or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("阈值须满足 0 <= exit_z < entry_z 且有限")
        object.__setattr__(self, "quantity", amount)


class CalendarSpreadStrategy(StrategyTemplate):
    """价差低时多第一腿空第二腿；价差高时空第一腿多第二腿；换约先清旧仓。"""

    def __init__(self, strategy_id: str, config: CalendarSpreadConfig,
                 selections: tuple[CalendarSelection, ...],
                 instruments: Mapping[str, InstrumentId]) -> None:
        super().__init__(strategy_id)
        if not selections:
            raise ValueError("期限角色日程不能为空")
        self.config = config
        self.selections = tuple(sorted(selections, key=lambda row: row.effective_ns))
        self._times = tuple(row.effective_ns for row in self.selections)
        self.instruments = dict(instruments)
        self.active_pair: tuple[str, str] | None = None
        self.latest: dict[str, Bar] = {}
        self.signal = CalendarSpreadSignal(config.lookback, config.entry_z, config.exit_z)
        self.last_frame_ns = -1
        self.complete_frames = 0
        self.submissions = 0
        self.rolls = 0
        self.direction = 0
        self.last_z: float | None = None
        self.last_targets: Mapping[str, Decimal] | None = None
        self._old_pair: tuple[str, str] | None = None
        self._last_roll_zero_ns = -1

    def _selection(self, timestamp: int) -> CalendarSelection | None:
        index = bisect_right(self._times, timestamp) - 1
        return None if index < 0 else self.selections[index]

    def _exposure(self, pair: tuple[str, str]) -> bool:
        return any(self.account_position(str(self.instruments[symbol])) != 0
                   or self.working_quantity(str(self.instruments[symbol])) != 0
                   for symbol in pair)

    def _targets(self, leg1: str | None = None, leg2: str | None = None,
                 direction: int = 0) -> dict[str, Decimal]:
        targets = {str(item): Decimal(0) for item in self.instruments.values()}
        if direction:
            assert leg1 is not None and leg2 is not None
            targets[str(self.instruments[leg1])] = self.config.quantity * direction
            targets[str(self.instruments[leg2])] = -self.config.quantity * direction
        return targets

    def _submit(self, targets: dict[str, Decimal], timestamp: int,
                *, reason: str, z: float | None = None) -> None:
        self.set_targets(targets, timestamp, metadata={
            "reason": reason, "z_score": z,
            "leg1_role": self.config.leg1_role, "leg2_role": self.config.leg2_role,
            "active_pair": self.active_pair,
        })
        self.last_targets = MappingProxyType(targets)
        self.submissions += 1

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"Bar与data_key不匹配: {data_key}")
        timestamp = bar.ts_event
        selection = self._selection(timestamp)
        if selection is None:
            return
        pair = (selection.leg1_symbol.lower(), selection.leg2_symbol.lower())
        symbol = str(bar.bar_type.instrument_id.symbol).lower()
        if self.active_pair is None:
            self.active_pair = pair
        elif pair != self.active_pair and self._old_pair is None:
            if self._exposure(self.active_pair) and not selection.old_pair_bars_available:
                raise RuntimeError(
                    f"{selection.day} 旧期限组合 {self.active_pair} 仍有仓位或在途订单，"
                    "但换月日缺少旧合约 Bar，无法安全清仓；请检查数据或提前平仓"
                )
            self._old_pair = self.active_pair
            self.signal.reset()
            self.direction = 0
            self.latest.clear()
            self.rolls += 1

        if self._old_pair is not None:
            # 先用旧双腿的同步 Bar 清仓；归零后再等待新双腿的同步 Bar。
            needed = set(self._old_pair) if self._exposure(self._old_pair) else set(pair)
            if symbol in needed:
                self.latest[symbol] = bar
            if (timestamp > self._last_roll_zero_ns and all(
                item in self.latest and self.latest[item].ts_event == timestamp for item in needed
            )):
                if self._exposure(self._old_pair):
                    self._submit(self._targets(), timestamp, reason="ROLL_CLOSE")
                    self._last_roll_zero_ns = timestamp
                else:
                    self.active_pair = pair
                    self._old_pair = None
                    self.latest.clear()
            return

        if symbol not in pair:
            return
        old = self.latest.get(symbol)
        if old is not None and timestamp <= old.ts_event:
            if timestamp < old.ts_event:
                raise ValueError(f"{symbol} Bar时间回退")
            return
        self.latest[symbol] = bar
        if (timestamp <= self.last_frame_ns or any(
            item not in self.latest or self.latest[item].ts_event != timestamp for item in pair
        )):
            return
        self.last_frame_ns = timestamp
        self.complete_frames += 1
        spread = float(self.latest[pair[0]].close.as_decimal()
                       - self.latest[pair[1]].close.as_decimal())
        # 两腿必须同步后才更新纯信号；这里仅将方向转为真实合约目标。
        signal = self.signal.update(spread, self.direction)
        if signal is not None:
            z, direction = signal.z_score, signal.direction
            self.last_z = z
            if (direction != self.direction or
                    (direction and self.complete_frames % self.config.rebalance_interval == 0)):
                self.active_pair = pair
                self._submit(self._targets(*pair, direction), timestamp,
                             reason="SIGNAL", z=z)
                self.direction = direction
