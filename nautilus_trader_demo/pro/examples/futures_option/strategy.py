"""example05的信号层：连续合约只产生方向，真实选约留在执行装配层。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from market.basic.base import Bar
from trader import StrategyTemplate, TargetUpdateMode


@dataclass(frozen=True)
class ProductSignalConfig:
    data_key: str
    future_key: str
    call_key: str
    put_key: str
    trade_futures: bool
    future_quantity: Decimal = Decimal(2)
    option_quantity: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        keys = (self.future_key, self.call_key, self.put_key)
        if not self.data_key.strip() or len(set(keys)) != 3 or any(not key.strip() for key in keys):
            raise ValueError("连续合约数据键和三条逻辑目标腿必须明确且互不重复")
        future_quantity = Decimal(str(self.future_quantity))
        option_quantity = Decimal(str(self.option_quantity))
        if future_quantity <= 0 or option_quantity <= 0:
            raise ValueError("目标数量必须为正")
        object.__setattr__(self, "future_quantity", future_quantity)
        object.__setattr__(self, "option_quantity", option_quantity)


class FuturesOptionSignalStrategy(StrategyTemplate):
    """只计算快慢均线并提交期货/Call/Put逻辑目标，不查选约、不创建订单。"""

    def __init__(
        self,
        strategy_id: str,
        products: tuple[ProductSignalConfig, ...],
        *,
        fast_window: int = 3,
        slow_window: int = 8,
    ) -> None:
        super().__init__(strategy_id)
        if not products or fast_window < 1 or slow_window <= fast_window:
            raise ValueError("至少配置一个产品，且0 < fast_window < slow_window")
        if len({item.data_key for item in products}) != len(products):
            raise ValueError("产品的data_key不能重复")
        all_keys = [key for item in products for key in (item.future_key, item.call_key, item.put_key)]
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("不同产品的目标键不能重叠")
        self.products = {item.data_key: item for item in products}
        self.fast_window = fast_window
        self.slow_window = slow_window
        self._history = {key: deque(maxlen=slow_window) for key in self.products}
        self._signals: dict[str, int] = {}

    def on_bar(self, data_key: str, bar: Bar) -> None:
        product = self.products.get(data_key)
        if product is None:
            return
        history = self._history[data_key]
        history.append(bar.close.as_decimal())
        if len(history) < self.slow_window:
            return
        fast = sum(tuple(history)[-self.fast_window:]) / self.fast_window
        slow = sum(history) / self.slow_window
        signal = 1 if fast > slow else -1
        if self._signals.get(data_key) == signal:
            return
        self._signals[data_key] = signal
        self.set_targets(
            {
                product.future_key: (
                    product.future_quantity * signal if product.trade_futures else Decimal(0)
                ),
                product.call_key: product.option_quantity if signal > 0 else Decimal(0),
                product.put_key: product.option_quantity if signal < 0 else Decimal(0),
            },
            bar.ts_event,
            update_mode=TargetUpdateMode.PATCH,
            metadata={"signal_data_key": data_key, "signal_direction": signal},
        )
