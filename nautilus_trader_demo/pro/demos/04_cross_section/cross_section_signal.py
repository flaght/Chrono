"""纯横截面动量信号：同步复权收盘价的收益率排名。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class CrossSectionSignal:
    """保存每个品种的动量分数及从强到弱的排名结果。"""
    scores: Mapping[str, Decimal]
    ranking: tuple[str, ...]
    long_products: tuple[str, ...]
    short_products: tuple[str, ...]


class CrossSectionMomentumSignal:
    """回看若干个同步分钟间隔，需要比间隔数多一个的价格点。"""

    def __init__(self, products: tuple[str, ...], lookback: int,
                 group_fraction: Decimal = Decimal("0.30")) -> None:
        if len(products) < 2 or len(set(products)) != len(products) or any(not item for item in products):
            raise ValueError("至少需要两个不重复的品种")
        if lookback < 1:
            raise ValueError("回看窗口必须为正整数")
        fraction = Decimal(str(group_fraction))
        if not fraction.is_finite() or not 0 < fraction <= Decimal("0.5"):
            raise ValueError("每侧选仓比例必须为有限数且满足 0 < 比例 <= 0.5")
        self.group_fraction = fraction
        # 按当前品种数取整，每侧至少一个；比例上限保证多空组不重叠。
        self.group_size = max(1, int(len(products) * fraction))
        self.products = tuple(products)
        self.lookback = lookback
        self.history = {item: deque(maxlen=lookback + 1) for item in products}

    def update(self, adjusted_closes: Mapping[str, Decimal]) -> CrossSectionSignal | None:
        """接收完整同步复权价格；窗口未预热时返回空值，否则返回排名。"""
        if set(adjusted_closes) != set(self.products):
            raise ValueError("信号必须接收完整品种集合的同步复权收盘价")
        prices = {item: Decimal(str(adjusted_closes[item])) for item in self.products}
        if any(not value.is_finite() or value <= 0 for value in prices.values()):
            raise ValueError("复权收盘价必须为有限正数")
        for item, value in prices.items():
            # 先校验全部价格，再一起推进窗口，避免无效帧造成品种窗口错位。
            self.history[item].append(value)
        if any(len(values) < self.lookback + 1 for values in self.history.values()):
            return None
        scores = {item: self.history[item][-1] / self.history[item][0] - 1
                  for item in self.products}
        # 分数相同时按品种代码排序，不依赖行情到达顺序。
        ranking = tuple(sorted(self.products, key=lambda item: (-scores[item], item)))
        return CrossSectionSignal(MappingProxyType(scores), ranking,
                                  ranking[:self.group_size], ranking[-self.group_size:])


def target_quantity(target_notional: Decimal, raw_price: Decimal,
                    multiplier: Decimal) -> Decimal:
    """按真实价格向下取整；预算不足一手时返回零，不突破分配预算。"""
    values = tuple(Decimal(str(value)) for value in (target_notional, raw_price, multiplier))
    if any(not value.is_finite() or value <= 0 for value in values):
        raise ValueError("名义金额、真实价格和合约乘数必须为有限正数")
    notional, price, factor = values
    units = (notional / (price * factor)).to_integral_value(rounding=ROUND_DOWN)
    return units


def allocate_group_targets(signal: CrossSectionSignal, side_notional: Decimal,
                           raw_prices: Mapping[str, Decimal],
                           multipliers: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """多空两侧使用相等预算，组内等额分配；整数手数余款保留现金。"""
    if set(raw_prices) != set(signal.ranking) or set(multipliers) != set(signal.ranking):
        raise ValueError("配置真实价格和乘数的品种集合必须与信号一致")
    budget = Decimal(str(side_notional))
    if not budget.is_finite() or budget <= 0:
        raise ValueError("每侧预算必须为有限正数")
    targets = {item: Decimal(0) for item in signal.ranking}
    for products, direction in ((signal.long_products, 1), (signal.short_products, -1)):
        per_product = budget / len(products)
        for item in products:
            targets[item] = target_quantity(per_product, raw_prices[item], multipliers[item]) * direction
    return targets
