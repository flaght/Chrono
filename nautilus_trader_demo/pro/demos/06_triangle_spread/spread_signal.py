"""不依赖行情或撮合框架的三腿价差计算。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite, log, sqrt
from typing import Mapping


def normalized_log_spread(prices: Mapping[str, float], baseline: Mapping[str, float],
                          anchor: str, hedges: tuple[str, str],
                          weights: tuple[float, float]) -> float:
    """计算主腿相对初始价格的对数变化，减去两条对冲腿的加权变化。"""
    spread = log(prices[anchor] / baseline[anchor])
    for product, weight in zip(hedges, weights):
        spread -= weight * log(prices[product] / baseline[product])
    return spread


def score(current: float, previous: tuple[float, ...]) -> float:
    """使用此前价差窗口的总体标准差评价当前价差，零方差时返回零。"""
    mean = sum(previous) / len(previous)
    variance = sum((value - mean) ** 2 for value in previous) / len(previous)
    return (current - mean) / sqrt(variance) if variance > 0 else 0.0


def direction_for(z: float, previous_direction: int, entry_z: float,
                  exit_z: float) -> int:
    """低价差做多主腿，高价差反向；回归出场区间则归零，否则保持方向。"""
    if abs(z) <= exit_z:
        return 0
    if z <= -entry_z:
        return 1
    if z >= entry_z:
        return -1
    return previous_direction


@dataclass(frozen=True)
class TriangleSignal:
    """完整预热后的一次价差信号，不包含真实合约或订单。"""
    spread: float
    z_score: float
    direction: int


class TriangleSpreadSignal:
    """维护首帧归一化基准和此前价差窗口，复权连续线换约时不重置。"""

    def __init__(self, anchor: str, hedges: tuple[str, str],
                 weights: tuple[Decimal, Decimal], lookback: int,
                 entry_z: float, exit_z: float) -> None:
        products = (anchor, *hedges)
        weights = tuple(Decimal(str(value)) for value in weights)
        if len(products) != 3 or len(set(products)) != 3:
            raise ValueError("价差信号需要三个不同品种")
        if len(weights) != 2 or any(not value.is_finite() or value <= 0 for value in weights) or sum(weights) != 1:
            raise ValueError("两条对冲腿权重须为有限正数且合计为1")
        if lookback < 2 or not all(isfinite(value) for value in (entry_z, exit_z)) or not 0 <= exit_z < entry_z:
            raise ValueError("窗口至少2根，阈值须满足 0 <= exit_z < entry_z 且有限")
        self.anchor, self.hedges = anchor, hedges
        self.products = products
        self.weights = tuple(float(value) for value in weights)
        self.lookback, self.entry_z, self.exit_z = lookback, entry_z, exit_z
        self.baseline: dict[str, float] = {}
        self.history: deque[float] = deque(maxlen=lookback)

    def update(self, adjusted_closes: Mapping[str, Decimal],
               previous_direction: int) -> TriangleSignal | None:
        """接收完整同步复权价；先计算当前信号，再将当前价差加入窗口。"""
        if set(adjusted_closes) != set(self.products) or previous_direction not in {-1, 0, 1}:
            raise ValueError("需要完整三品种价格及有效前一方向")
        prices = {item: float(adjusted_closes[item]) for item in self.products}
        if any(not isfinite(value) or value <= 0 for value in prices.values()):
            raise ValueError("复权价格必须为有限正数")
        if not self.baseline:
            self.baseline = dict(prices)
        spread = normalized_log_spread(prices, self.baseline, self.anchor, self.hedges, self.weights)
        result = None
        if len(self.history) == self.lookback:
            z = score(spread, tuple(self.history))
            result = TriangleSignal(spread, z, direction_for(z, previous_direction, self.entry_z, self.exit_z))
        # 当前帧不能进入自身的均值和标准差基准。
        self.history.append(spread)
        return result
