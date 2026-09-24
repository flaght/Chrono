"""示例业务策略：单标的EMA交叉目标仓位。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bomber.indicators import ExponentialMovingAverage

from market.basic.base import Bar
from trader.template import StrategyTemplate


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class EmaCrossConfig:
    """EMA信号参数；data_key和target_key均为与平台无关的逻辑名称。"""

    data_key: str = "primary_bar"
    target_key: str = "position"
    fast_period: int = 3
    slow_period: int = 5
    long_quantity: Decimal = Decimal(1)
    short_quantity: Decimal = Decimal(-1)
    skip_single_price: bool = True

    def __post_init__(self) -> None:
        if not self.data_key.strip():
            raise ValueError("data_key 不能为空")
        if not self.target_key.strip():
            raise ValueError("target_key 不能为空")
        if self.fast_period < 1 or self.slow_period < 1:
            raise ValueError("EMA周期必须为正整数")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period 必须小于 slow_period")
        long_quantity = _decimal(self.long_quantity)
        short_quantity = _decimal(self.short_quantity)
        if long_quantity <= 0:
            raise ValueError("long_quantity 必须大于0")
        if short_quantity >= 0:
            raise ValueError("short_quantity 必须小于0")
        object.__setattr__(self, "long_quantity", long_quantity)
        object.__setattr__(self, "short_quantity", short_quantity)


class EmaCrossTargetStrategy(StrategyTemplate):
    """用原生EMA计算方向，只输出目标仓位，不直接创建订单。"""

    def __init__(self, strategy_id: str, config: EmaCrossConfig | None = None) -> None:
        super().__init__(strategy_id)
        self.config = config or EmaCrossConfig()
        self._fast = ExponentialMovingAverage(self.config.fast_period)
        self._slow = ExponentialMovingAverage(self.config.slow_period)
        self._last_target: Decimal | None = None
        self._bars_seen = 0
        self._bars_used = 0

    @property
    def fast_ema(self) -> Decimal | None:
        return Decimal(str(self._fast.value)) if self._fast.has_inputs else None

    @property
    def slow_ema(self) -> Decimal | None:
        return Decimal(str(self._slow.value)) if self._slow.has_inputs else None

    @property
    def is_warmed_up(self) -> bool:
        return self._slow.initialized

    @property
    def last_target(self) -> Decimal | None:
        return self._last_target

    @property
    def bars_seen(self) -> int:
        return self._bars_seen

    @property
    def bars_used(self) -> int:
        return self._bars_used

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != self.config.data_key:
            return
        self._bars_seen += 1
        if self.config.skip_single_price and bar.is_single_price():
            return

        close = bar.close.as_decimal()
        self._fast.update_raw(bar.close.as_double())
        self._slow.update_raw(bar.close.as_double())
        self._bars_used += 1
        if not self.is_warmed_up:
            return

        target = (
            self.config.long_quantity
            if self._fast.value >= self._slow.value
            else self.config.short_quantity
        )
        if target == self._last_target:
            return
        self._last_target = target
        self.set_target(
            self.config.target_key,
            target,
            bar.ts_event,
            metadata={
                "signal": "LONG" if target > 0 else "SHORT",
                "close": str(close),
                "fast_ema": str(self._fast.value),
                "slow_ema": str(self._slow.value),
                "fast_period": self.config.fast_period,
                "slow_period": self.config.slow_period,
                "signal_ts": bar.ts_event,
            },
        )
