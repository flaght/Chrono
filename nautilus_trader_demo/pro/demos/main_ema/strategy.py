"""品种主力 EMA 信号；连续复权价仅用于信号，真实合约由动态路由成交。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from bomber.indicators import ExponentialMovingAverage

from datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from market.basic.base import Bar
from strategy.template import StrategyTemplate


class RoleSnapshotPort(Protocol):
    def snapshot(self, as_of_ns: int) -> SectorRoleAssignment: ...


@dataclass(frozen=True)
class MainEmaConfig:
    product: str
    venue: str
    fast_period: int = 3
    slow_period: int = 5
    quantity: Decimal = Decimal(1)
    target_key: str = ""

    def __post_init__(self) -> None:
        quantity = Decimal(str(self.quantity))
        if not 0 < self.fast_period < self.slow_period:
            raise ValueError("EMA 周期须满足 0 < fast_period < slow_period")
        if not quantity.is_finite() or quantity <= 0:
            raise ValueError("目标手数须为正且有限")
        product = self.product.strip().upper()
        venue = self.venue.strip().upper()
        if not product.isalpha() or not venue:
            raise ValueError("品种代码须为字母，交易所不能为空")
        target_key = self.target_key.strip() or f"{product.lower()}_main"
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "quantity", quantity)


class MainEmaStrategy(StrategyTemplate):
    """每根主力真实 Bar 用截至该时刻的复权收盘价更新 EMA。"""

    def __init__(self, strategy_id: str, data_hub: RoleSnapshotPort,
                 config: MainEmaConfig) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config
        self.fast = ExponentialMovingAverage(self.config.fast_period)
        self.slow = ExponentialMovingAverage(self.config.slow_period)
        self.bars_used = 0
        self.unavailable_events = 0
        self.last_target: Decimal | None = None
        self.last_processed_ns = -1
        self.last_main: str | None = None

    def on_bar(self, data_key: str, bar: Bar) -> None:
        del data_key
        timestamp = int(bar.ts_event)
        if timestamp <= self.last_processed_ns:
            return
        try:
            assignment = self.data_hub.snapshot(timestamp)
            main = assignment.instrument(self.config.product, "main").lower()
            cumulative = assignment.factor(self.config.product, "main")
        except SectorDataUnavailable:
            self.unavailable_events += 1
            return
        if (str(bar.bar_type.instrument_id.symbol).lower() != main
                or str(bar.bar_type.instrument_id.venue).upper() != self.config.venue):
            return
        # 因子只来自此前交易日；当前价格直接取已送达的真实 Bar。
        adjusted_close = bar.close.as_decimal() * cumulative
        value = float(adjusted_close)
        self.fast.update_raw(value)
        self.slow.update_raw(value)
        self.last_processed_ns = timestamp
        self.last_main = main
        self.bars_used += 1
        if not self.slow.initialized:
            return
        target = self.config.quantity if self.fast.value >= self.slow.value else -self.config.quantity
        if target == self.last_target:
            return
        self.last_target = target
        self.set_target(
            self.config.target_key, target, timestamp,
            metadata={
                "signal": "LONG" if target > 0 else "SHORT",
                "research_main": main,
                "adjusted_close": str(adjusted_close),
                "fast_ema": str(self.fast.value),
                "slow_ema": str(self.slow.value),
                "signal_ts": timestamp,
            },
        )
