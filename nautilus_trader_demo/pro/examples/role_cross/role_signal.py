"""四角色复权价穿越规则；不依赖行情源、DataHub实现或交易客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from datahub.core import RoleSnapshot
from datahub.role_prices import ROLES


@dataclass(frozen=True)
class RoleCrossEvent:
    ts_event: int
    direction: int
    spread: Decimal
    main_instrument: str


class RoleCrossSignal:
    """main复权价穿越四角色复权均价时输出方向，第一帧仅预热。"""

    def __init__(self) -> None:
        self.previous_spread: Decimal | None = None
        self.last_processed_ns = -1
        self.first_spread: Decimal | None = None
        self.min_spread: Decimal | None = None
        self.max_spread: Decimal | None = None
        self.positive_frames = 0
        self.negative_frames = 0
        self.zero_frames = 0

    def update(self, snapshot: RoleSnapshot) -> RoleCrossEvent | None:
        if snapshot.as_of_ns <= self.last_processed_ns:
            return None
        prices = snapshot.prices
        if set(prices) != set(ROLES) or set(snapshot.contracts) != set(ROLES):
            raise ValueError("四角色复权价必须同时齐备")
        values = {role: prices[role].adjusted_close for role in ROLES}
        if any(not value.is_finite() or value <= 0 for value in values.values()):
            raise ValueError("复权研究价必须为正且有限")
        spread = values["main"] - sum(values.values()) / Decimal(4)
        previous = self.previous_spread
        if self.first_spread is None:
            self.first_spread = spread
        self.min_spread = spread if self.min_spread is None else min(self.min_spread, spread)
        self.max_spread = spread if self.max_spread is None else max(self.max_spread, spread)
        if spread > 0:
            self.positive_frames += 1
        elif spread < 0:
            self.negative_frames += 1
        else:
            self.zero_frames += 1
        self.previous_spread = spread
        self.last_processed_ns = snapshot.as_of_ns
        direction = 0
        if previous is not None:
            if previous <= 0 < spread:
                direction = 1
            elif previous >= 0 > spread:
                direction = -1
        if not direction:
            return None
        return RoleCrossEvent(
            snapshot.as_of_ns, direction, spread, snapshot.contracts["main"],
        )
