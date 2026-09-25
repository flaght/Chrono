"""三品种主力相对价值组合；信号用连续复权价，成交用真实合约。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from datahub.sector_roles import SectorRoleStore
from market.basic.base import Bar, InstrumentId
from strategy.template import StrategyTemplate

from .spread_signal import direction_for, normalized_log_spread, score


@dataclass(frozen=True)
class TriangleSpreadConfig:
    anchor: str
    hedges: tuple[str, str]
    hedge_weights: tuple[Decimal, Decimal]
    lookback: int
    entry_z: float
    exit_z: float
    target_notional: Decimal
    rebalance_interval: int = 1

    def __post_init__(self) -> None:
        products = (self.anchor, *self.hedges)
        if len(set(products)) != 3 or any(not item.isalpha() or item != item.upper()
                                          for item in products):
            raise ValueError("需要三个不同的品种字母代码")
        weights = tuple(Decimal(str(item)) for item in self.hedge_weights)
        if len(weights) != 2 or any(not item.is_finite() or item <= 0 for item in weights):
            raise ValueError("两个对冲权重必须为有限正数")
        if sum(weights) != 1:
            raise ValueError("两个对冲权重之和必须为1")
        if self.lookback < 2 or self.rebalance_interval < 1:
            raise ValueError("回看窗口至少2根，同步调仓间隔必须为正")
        if not isfinite(self.entry_z) or not isfinite(self.exit_z) or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("阈值必须满足 0 <= exit_z < entry_z")
        amount = Decimal(str(self.target_notional))
        if not amount.is_finite() or amount <= 0:
            raise ValueError("目标名义金额必须为有限正数")
        object.__setattr__(self, "hedge_weights", weights)
        object.__setattr__(self, "target_notional", amount)

    @property
    def products(self) -> tuple[str, str, str]:
        return (self.anchor, *self.hedges)


class TriangleSpreadStrategy(StrategyTemplate):
    """价差低做多主腿/做空两对冲腿；价差高反向，回归均值平仓。"""

    def __init__(self, strategy_id: str, config: TriangleSpreadConfig,
                 roles: SectorRoleStore,
                 instruments: Mapping[str, InstrumentId],
                 multipliers: Mapping[InstrumentId, Decimal]) -> None:
        super().__init__(strategy_id)
        self.config = config
        self.roles = roles
        self.instruments = dict(instruments)
        self.multipliers = dict(multipliers)
        self.latest: dict[str, Bar] = {}
        self.current_main: dict[str, str] = {}
        self.baseline: dict[str, Decimal] = {}
        self.history: deque[float] = deque(maxlen=config.lookback)
        self.last_frame_ns = -1
        self.complete_frames = 0
        self.submissions = 0
        self.direction = 0  # +1: 多主腿、空两对冲腿；-1: 相反。
        self.roll_pending = False
        self.last_z: float | None = None
        self.last_targets: Mapping[str, Decimal] | None = None

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"Bar与data_key不匹配: {data_key}")
        snapshot = self.roles.snapshot(bar.ts_event)
        symbol = str(bar.bar_type.instrument_id.symbol).lower()
        product = next((item for item in self.config.products
                        if snapshot.instrument(item, "main").lower() == symbol), None)
        if product is None:
            return  # 换月日旧合约 Bar 仅作为模拟平仓行情。
        previous_symbol = self.current_main.get(product)
        if previous_symbol is not None and previous_symbol != symbol:
            self.latest.pop(product, None)
            self.roll_pending = True
        self.current_main[product] = symbol
        old = self.latest.get(product)
        if old is not None and bar.ts_event <= old.ts_event:
            if bar.ts_event < old.ts_event:
                raise ValueError(f"{product} Bar时间回退")
            return
        self.latest[product] = bar
        timestamp = bar.ts_event
        if (timestamp <= self.last_frame_ns or len(self.latest) != 3
                or any(self.latest[item].ts_event != timestamp for item in self.config.products)):
            return
        self.last_frame_ns = timestamp
        self.complete_frames += 1
        adjusted = {item: self.latest[item].close.as_decimal() * snapshot.factor(item, "main")
                    for item in self.config.products}
        if any(not price.is_finite() or price <= 0 for price in adjusted.values()):
            raise ValueError("复权价格必须为有限正数")
        if not self.baseline:
            self.baseline = adjusted
        spread = normalized_log_spread(
            {item: float(value) for item, value in adjusted.items()},
            {item: float(value) for item, value in self.baseline.items()},
            self.config.anchor, self.config.hedges,
            tuple(float(item) for item in self.config.hedge_weights),
        )
        # 用之前 lookback 根计算均值和标准差，再评估当前帧。
        if len(self.history) == self.config.lookback:
            z = score(spread, tuple(self.history))
            self.last_z = z
            direction = direction_for(z, self.direction,
                                      self.config.entry_z, self.config.exit_z)
            if (direction != self.direction or self.roll_pending
                    or (direction and self.complete_frames % self.config.rebalance_interval == 0)):
                self._submit(direction, timestamp, z)
                self.direction = direction
                self.roll_pending = False
        self.history.append(spread)

    def _submit(self, direction: int, timestamp: int, z: float) -> None:
        targets = {str(item): Decimal(0) for item in self.instruments.values()}
        if direction:
            legs = ((self.config.anchor, Decimal(1)),
                    (self.config.hedges[0], -self.config.hedge_weights[0]),
                    (self.config.hedges[1], -self.config.hedge_weights[1]))
            for product, signed_weight in legs:
                symbol = self.current_main[product]
                instrument = self.instruments[symbol]
                one_lot = self.latest[product].close.as_decimal() * self.multipliers[instrument]
                lots = (self.config.target_notional * abs(signed_weight) / one_lot).to_integral_value(
                    rounding=ROUND_DOWN,
                )
                if lots < 1:
                    raise ValueError(f"{product} 的目标名义金额不足一手；增加 --target-notional")
                targets[str(instrument)] = lots * (1 if signed_weight * direction > 0 else -1)
        self.set_targets(targets, timestamp, metadata={
            "z_score": z, "direction": direction,
            "main_contracts": dict(self.current_main),
            "complete_frame": self.complete_frames,
        })
        self.last_targets = MappingProxyType(targets)
        self.submissions += 1
