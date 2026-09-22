"""第五类纯信号核：双品种收益率均值与比较品种收益率对照。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True)
class SectorSignal:
    ts_event: int
    direction: int
    comparison_return: Decimal
    sector_ma: Decimal
    leader_return_mas: tuple[Decimal, Decimal]


class BlackSectorSignal:
    def __init__(self, return_period: int = 30, sector_period: int = 15,
                 *, leader_products: tuple[str, str] = ("JM", "I"),
                 comparison_product: str = "RB") -> None:
        if return_period < 1 or sector_period < 1:
            raise ValueError("均线周期必须为正整数")
        if len(leader_products) != 2 or len(set((*leader_products, comparison_product))) != 3:
            raise ValueError("必须指定两个不同的领头品种和另一个比较品种")
        self.return_period = return_period
        self.sector_period = sector_period
        self.leader_products = leader_products
        self.comparison_product = comparison_product
        self.products = (*leader_products, comparison_product)
        self._previous: dict[str, Decimal] = {}
        self._returns = {key: deque(maxlen=return_period) for key in leader_products}
        self._sector = deque(maxlen=sector_period)
        self.last_timestamp = -1
        self.direction = 0

    def update(self, ts_event: int, adjusted_closes: Mapping[str, Decimal]) -> SectorSignal | None:
        if ts_event <= self.last_timestamp:
            raise ValueError("同步帧时间必须严格递增")
        if set(adjusted_closes) != set(self.products):
            raise ValueError(f"同步帧必须含{self.products}")
        prices = {key: Decimal(str(value)) for key, value in adjusted_closes.items()}
        if any(not value.is_finite() or value <= 0 for value in prices.values()):
            raise ValueError("复权收盘价必须为正且有限")
        self.last_timestamp = ts_event
        if not self._previous:
            self._previous = prices
            return None
        returns = {key: prices[key] / self._previous[key] - 1 for key in self.products}
        self._previous = prices
        for key in self.leader_products:
            self._returns[key].append(returns[key])
        if len(self._returns[self.leader_products[0]]) < self.return_period:
            return None
        leader_mas = tuple(
            sum(self._returns[key]) / Decimal(self.return_period)
            for key in self.leader_products
        )
        self._sector.append(sum(leader_mas) / 2)
        if len(self._sector) < self.sector_period:
            return None
        sector_ma = sum(self._sector) / Decimal(self.sector_period)
        if returns[self.comparison_product] > sector_ma:
            self.direction = 1
        elif returns[self.comparison_product] < sector_ma:
            self.direction = -1
        return SectorSignal(
            ts_event, self.direction, returns[self.comparison_product],
            sector_ma, leader_mas,
        )
