"""多品种合约角色与复权因子的因果快照；不预设产业链或执行标的。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


class SectorDataUnavailable(LookupError):
    """决策时刻缺少已经发布的角色或因子。"""


@dataclass(frozen=True)
class SectorRoleAssignment:
    """某日生效的角色映射；哪些品种/角色参与策略由调用方决定。"""

    trading_day: date
    source_day: date
    effective_ns: int
    available_ns: int
    contracts: Mapping[str, Mapping[str, str]]
    cumulative_factors: Mapping[str, Mapping[str, Decimal]]

    def __post_init__(self) -> None:
        if self.source_day >= self.trading_day or min(self.effective_ns, self.available_ns) < 0:
            raise ValueError("角色只能来自此前交易日，时间必须非负")
        if not self.contracts or any(not product or not roles for product, roles in self.contracts.items()):
            raise ValueError("至少需要一个品种及其合约角色")
        contracts: dict[str, Mapping[str, str]] = {}
        for product, roles in self.contracts.items():
            if any(not role or not symbol for role, symbol in roles.items()):
                raise ValueError(f"{product}角色名与真实合约不能为空")
            contracts[product] = MappingProxyType(dict(roles))
        factors: dict[str, Mapping[str, Decimal]] = {}
        for product, roles in self.cumulative_factors.items():
            if product not in contracts or not set(roles).issubset(contracts[product]):
                raise ValueError(f"{product}因子必须对应已有的品种/角色")
            converted = {role: Decimal(str(value)) for role, value in roles.items()}
            if any(not value.is_finite() or value <= 0 for value in converted.values()):
                raise ValueError("累计复权因子必须为正且有限")
            factors[product] = MappingProxyType(converted)
        object.__setattr__(self, "contracts", MappingProxyType(contracts))
        object.__setattr__(self, "cumulative_factors", MappingProxyType(factors))

    def instrument(self, product: str, role: str) -> str:
        try:
            return self.contracts[product][role]
        except KeyError as exc:
            raise SectorDataUnavailable(f"{product}/{role}没有真实合约") from exc

    def factor(self, product: str, role: str) -> Decimal:
        try:
            return self.cumulative_factors[product][role]
        except KeyError as exc:
            raise SectorDataUnavailable(f"{product}/{role}没有复权因子") from exc


class SectorRoleStore:
    """通用品种/角色日程；按生效时间与实际发布时间查询。"""

    def __init__(self, assignments: tuple[SectorRoleAssignment, ...]) -> None:
        if not assignments:
            raise ValueError("角色日程不能为空")
        rows = tuple(sorted(assignments, key=lambda item: item.effective_ns))
        if len({item.trading_day for item in rows}) != len(rows):
            raise ValueError("交易日角色记录重复")
        if any(right.effective_ns <= left.effective_ns or right.trading_day <= left.trading_day
               for left, right in zip(rows, rows[1:])):
            raise ValueError("角色生效时间及交易日必须递增")
        self.assignments = rows
        self._times = tuple(item.effective_ns for item in rows)

    def snapshot(self, as_of_ns: int) -> SectorRoleAssignment:
        if as_of_ns < 0:
            raise ValueError("查询时间不能为负")
        index = bisect_right(self._times, as_of_ns) - 1
        if index < 0:
            raise SectorDataUnavailable("尚无生效的品种角色")
        row = self.assignments[index]
        if row.available_ns > as_of_ns:
            raise SectorDataUnavailable("最新角色版本尚未发布，不可退回旧版本")
        return row
