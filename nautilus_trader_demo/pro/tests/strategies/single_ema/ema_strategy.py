"""单标的 EMA 交叉策略的统一目标仓位实现。

该实现吸 共同信号逻辑，但不读取Portfolio、不创建订单。策略只消费逻辑 data_key，并输出逻辑 target_key。
"""



from dataclasses import dataclass
from decimal import Decimal

from bomber.indicators import ExponentialMovingAverage
from market.basic.base import Bar
from trader import StrategyTemplate

def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))

@dataclass(frozen=True)
class EmaCrossConfig:
    """EMA策略参数。

    ``data_key`` 和 ``target_key`` 都是策略内部逻辑名称。具体行情源、合约和执行
    客户端由 Runner 的 DataBinding/ExecutionRoute 在外部装配。
    """

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
    """EMA方向变化时提交最终目标仓位。

    快线大于等于慢线时输出 ``long_quantity``，否则输出 ``short_quantity``。
    慢线完成预热前不产生目标；方向没有变化时也不重复提交相同目标。

    策略不知道Bar来自文件、CTP、DolphinDB还是Binance，也不知道目标最终由
    NT、Bomber还是vn.py执行。
    """

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
        if not self._fast.has_inputs:
            return None
        return Decimal(str(self._fast.value))

    @property
    def slow_ema(self) -> Decimal | None:
        if not self._slow.has_inputs:
            return None
        return Decimal(str(self._slow.value))

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
        # 同一个策略实例可以绑定其他辅助数据；只有primary data_key更新EMA。
        if data_key != self.config.data_key:
            return
        self._bars_seen += 1

        if self.config.skip_single_price and bar.is_single_price():
            return

        close = bar.close.as_decimal()
        close_raw = bar.close.as_double()
        self._fast.update_raw(close_raw)
        self._slow.update_raw(close_raw)
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
            },
        )
