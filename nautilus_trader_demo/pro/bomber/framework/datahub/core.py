"""最小DataHub查询契约：数据源与策略、执行系统分离。

当前仅承载四角色研究快照；将来文件、数据库或服务端Provider均实现同一
as-of协议，不应在策略中直接读取Feather或构造复权Bar。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping, Protocol

from .role_prices import RoleAssignment, RolePrice


@dataclass(frozen=True)
class RoleSnapshot:
    as_of_ns: int
    trading_day: date
    assignment_source_day: date
    contracts: Mapping[str, str]
    prices: Mapping[str, RolePrice]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))


class RoleResearchProvider(Protocol):
    """允许换成正式DataHub后端；缺失或未发布数据必须显式拒绝。"""

    def assignment_at(self, as_of_ns: int) -> RoleAssignment: ...

    def snapshot(self, as_of_ns: int, *, max_source_age_ns: int | None = None) -> Mapping[str, RolePrice]: ...


class MinimalDataHub:
    """稳定的研究数据门面，不负责仓位、订单、撮合或期权选约。"""

    def __init__(self, roles: RoleResearchProvider) -> None:
        self._roles = roles

    def snapshot(self, as_of_ns: int, *, max_source_age_ns: int | None = None) -> RoleSnapshot:
        # 先校验角色版本在此刻已可用，再查询角色真实价格及复权研究价格。
        assignment = self._roles.assignment_at(as_of_ns)
        prices = self._roles.snapshot(as_of_ns, max_source_age_ns=max_source_age_ns)
        return RoleSnapshot(
            as_of_ns=as_of_ns,
            trading_day=assignment.trading_day,
            assignment_source_day=assignment.source_day,
            contracts=assignment.contracts,
            prices=prices,
        )
