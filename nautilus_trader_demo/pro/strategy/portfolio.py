"""策略目标、组合目标和仓位状态的通用核心组件。

本模块不创建订单：TargetStore 保存策略目标，PortfolioCoordinator 汇总已经
解析为真实账户腿的策略贡献，PositionManager 保存仓位和在途数量。
"""
from __future__ import annotations


import threading
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from market.basic.base import InstrumentId
from strategy.contracts import TargetPortfolio, TargetUpdateMode


class StaleRevisionError(ValueError):
    """收到重复或倒序的状态版本。"""


class TargetExpiredError(ValueError):
    """目标在进入存储时已经超过执行截止时间。"""


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class TargetStore:
    """按策略保存最新的物化目标快照。

    PATCH 与上一份目标合并；对外返回的一律是 REPLACE 形式的完整快照。
    同一策略的 revision 必须严格递增。
    """

    def __init__(self) -> None:
        self._targets: dict[str, TargetPortfolio] = {}
        self._lock = threading.RLock()

    def apply(
        self,
        intent: TargetPortfolio,
        *,
        now_ns: int | None = None,
    ) -> TargetPortfolio:
        if now_ns is not None and now_ns < 0:
            raise ValueError("now_ns 不能为负数")
        if (
            now_ns is not None
            and intent.deadline_ns is not None
            and intent.deadline_ns < now_ns
        ):
            raise TargetExpiredError(
                f"目标已过期: strategy={intent.strategy_id} "
                f"deadline={intent.deadline_ns} now={now_ns}",
            )

        with self._lock:
            previous = self._targets.get(intent.strategy_id)
            if previous is not None and intent.revision <= previous.revision:
                raise StaleRevisionError(
                    f"目标版本必须递增: strategy={intent.strategy_id} "
                    f"current={previous.revision} received={intent.revision}",
                )
            if intent.update_mode is TargetUpdateMode.PATCH and previous is not None:
                targets = dict(previous.targets)
                targets.update(intent.targets)
            else:
                targets = dict(intent.targets)

            materialized = TargetPortfolio(
                strategy_id=intent.strategy_id,
                revision=intent.revision,
                ts_event=intent.ts_event,
                targets=targets,
                update_mode=TargetUpdateMode.REPLACE,
                execution_policy=intent.execution_policy,
                deadline_ns=intent.deadline_ns,
                metadata=intent.metadata,
            )
            self._targets[intent.strategy_id] = materialized
            return materialized

    def get(self, strategy_id: str) -> TargetPortfolio | None:
        with self._lock:
            return self._targets.get(strategy_id)

    def all(self) -> Mapping[str, TargetPortfolio]:
        with self._lock:
            return MappingProxyType(dict(self._targets))

    def remove(self, strategy_id: str) -> TargetPortfolio | None:
        with self._lock:
            return self._targets.pop(strategy_id, None)

    def clear(self) -> None:
        with self._lock:
            self._targets.clear()


@dataclass(frozen=True)
class AccountTargetKey:
    """一个交易客户端账户中的真实合约腿。"""

    client_id: str
    instrument_id: InstrumentId

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id 不能为空")


@dataclass(frozen=True)
class PortfolioTargetSnapshot:
    """多策略净额后的账户目标快照。"""

    revision: int
    ts_event: int
    targets: Mapping[AccountTargetKey, Decimal]
    contributions: Mapping[str, Mapping[AccountTargetKey, Decimal]]

    def __post_init__(self) -> None:
        frozen = {
            strategy_id: MappingProxyType(dict(values))
            for strategy_id, values in self.contributions.items()
        }
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "contributions", MappingProxyType(frozen))


class PortfolioCoordinator:
    """汇总多个策略对真实账户腿的目标贡献。

    输入必须已由静态或动态 TargetResolver 解析成 AccountTargetKey。不同
    client_id 永不相互净额。同一客户端、同一真实合约的目标会求和，同时保留
    每个策略的独立贡献用于归属和审计。
    """

    def __init__(self) -> None:
        self._contributions: dict[str, dict[AccountTargetKey, Decimal]] = {}
        self._strategy_revisions: dict[str, int] = {}
        self._known_keys: set[AccountTargetKey] = set()
        self._revision = 0
        self._lock = threading.RLock()

    def update(
        self,
        *,
        strategy_id: str,
        revision: int,
        ts_event: int,
        targets: Mapping[AccountTargetKey, Decimal | int | float | str],
        update_mode: TargetUpdateMode | str = TargetUpdateMode.REPLACE,
    ) -> PortfolioTargetSnapshot:
        if not strategy_id.strip():
            raise ValueError("strategy_id 不能为空")
        if revision < 1:
            raise ValueError("revision 必须为正整数")
        if ts_event < 0:
            raise ValueError("ts_event 不能为负数")
        mode = TargetUpdateMode(update_mode)
        normalized = {key: _decimal(value) for key, value in targets.items()}

        with self._lock:
            current_revision = self._strategy_revisions.get(strategy_id, 0)
            if revision <= current_revision:
                raise StaleRevisionError(
                    f"组合贡献版本必须递增: strategy={strategy_id} "
                    f"current={current_revision} received={revision}",
                )
            previous = self._contributions.get(strategy_id, {})
            self._known_keys.update(previous)
            self._known_keys.update(normalized)
            if mode is TargetUpdateMode.PATCH:
                materialized = dict(previous)
                materialized.update(normalized)
            else:
                materialized = dict(normalized)
            self._contributions[strategy_id] = materialized
            self._strategy_revisions[strategy_id] = revision
            self._revision += 1
            return self._snapshot_locked(ts_event)

    def remove_strategy(
        self,
        strategy_id: str,
        *,
        ts_event: int,
    ) -> PortfolioTargetSnapshot:
        if ts_event < 0:
            raise ValueError("ts_event 不能为负数")
        with self._lock:
            previous = self._contributions.pop(strategy_id, None)
            self._strategy_revisions.pop(strategy_id, None)
            if previous is not None:
                self._known_keys.update(previous)
                self._revision += 1
            return self._snapshot_locked(ts_event)

    def snapshot(self, *, ts_event: int = 0) -> PortfolioTargetSnapshot:
        if ts_event < 0:
            raise ValueError("ts_event 不能为负数")
        with self._lock:
            return self._snapshot_locked(ts_event)

    def strategy_contribution(
        self,
        strategy_id: str,
    ) -> Mapping[AccountTargetKey, Decimal]:
        with self._lock:
            return MappingProxyType(dict(self._contributions.get(strategy_id, {})))

    def _snapshot_locked(self, ts_event: int) -> PortfolioTargetSnapshot:
        totals = {key: Decimal(0) for key in self._known_keys}
        for contribution in self._contributions.values():
            for key, quantity in contribution.items():
                totals[key] = totals.get(key, Decimal(0)) + quantity
        return PortfolioTargetSnapshot(
            revision=self._revision,
            ts_event=ts_event,
            targets=totals,
            contributions=self._contributions,
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """仓位管理器的不可变状态快照。"""

    strategy_positions: Mapping[tuple[str, str], Decimal]
    account_positions: Mapping[AccountTargetKey, Decimal]
    working_quantities: Mapping[AccountTargetKey, Decimal]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy_positions",
            MappingProxyType(dict(self.strategy_positions)),
        )
        object.__setattr__(
            self,
            "account_positions",
            MappingProxyType(dict(self.account_positions)),
        )
        object.__setattr__(
            self,
            "working_quantities",
            MappingProxyType(dict(self.working_quantities)),
        )


class PositionManager:
    """保存策略归属仓位、账户真实仓位和在途数量。

    account_positions 必须由撮合器或交易客户端查询/回报更新，是账户级权威
    状态；working_quantities 是尚未完全成交的有符号数量，买入为正、卖出为负。
    本类实现现有 PositionProvider 协议，但不会自行把账户成交分配给策略。
    """

    def __init__(self) -> None:
        self._strategy_positions: dict[tuple[str, str], Decimal] = {}
        self._account_positions: dict[AccountTargetKey, Decimal] = {}
        self._working_quantities: dict[AccountTargetKey, Decimal] = {}
        self._account_revisions: dict[AccountTargetKey, int] = {}
        self._strategy_revisions: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def position(self, strategy_id: str, target_key: str) -> Decimal:
        with self._lock:
            return self._strategy_positions.get((strategy_id, target_key), Decimal(0))

    def set_strategy_position(
        self,
        strategy_id: str,
        target_key: str,
        quantity: Decimal | int | float | str,
        *,
        revision: int | None = None,
    ) -> None:
        if not strategy_id.strip() or not target_key.strip():
            raise ValueError("strategy_id 和 target_key 不能为空")
        key = (strategy_id, target_key)
        normalized = _decimal(quantity)
        with self._lock:
            self._guard_revision(self._strategy_revisions, key, revision, "策略仓位")
            self._strategy_positions[key] = normalized

    def adjust_strategy_position(
        self,
        strategy_id: str,
        target_key: str,
        delta: Decimal | int | float | str,
    ) -> Decimal:
        if not strategy_id.strip() or not target_key.strip():
            raise ValueError("strategy_id 和 target_key 不能为空")
        key = (strategy_id, target_key)
        with self._lock:
            value = self._strategy_positions.get(key, Decimal(0)) + _decimal(delta)
            self._strategy_positions[key] = value
            return value

    def account_position(self, client_id: str, instrument_id: InstrumentId) -> Decimal:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            return self._account_positions.get(key, Decimal(0))

    def set_account_position(
        self,
        client_id: str,
        instrument_id: InstrumentId,
        quantity: Decimal | int | float | str,
        *,
        revision: int | None = None,
    ) -> None:
        key = AccountTargetKey(client_id, instrument_id)
        normalized = _decimal(quantity)
        with self._lock:
            self._guard_revision(self._account_revisions, key, revision, "账户仓位")
            self._account_positions[key] = normalized

    def adjust_account_position(
        self,
        client_id: str,
        instrument_id: InstrumentId,
        delta: Decimal | int | float | str,
    ) -> Decimal:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            value = self._account_positions.get(key, Decimal(0)) + _decimal(delta)
            self._account_positions[key] = value
            return value

    def replace_account_positions(
        self,
        client_id: str,
        positions: Mapping[InstrumentId, Decimal | int | float | str],
    ) -> None:
        """用权威快照替换客户端全部仓位，未出现的旧合约归零。"""
        if not client_id.strip():
            raise ValueError("client_id 不能为空")
        normalized = {
            AccountTargetKey(client_id, instrument_id): _decimal(quantity)
            for instrument_id, quantity in positions.items()
        }
        with self._lock:
            for key in tuple(self._account_positions):
                if key.client_id == client_id:
                    self._account_positions[key] = Decimal(0)
            self._account_positions.update(normalized)

    def set_working_quantity(
        self,
        client_id: str,
        instrument_id: InstrumentId,
        quantity: Decimal | int | float | str,
    ) -> None:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            self._working_quantities[key] = _decimal(quantity)

    def adjust_working_quantity(
        self,
        client_id: str,
        instrument_id: InstrumentId,
        delta: Decimal | int | float | str,
    ) -> Decimal:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            value = self._working_quantities.get(key, Decimal(0)) + _decimal(delta)
            self._working_quantities[key] = value
            return value

    def working_quantity(self, client_id: str, instrument_id: InstrumentId) -> Decimal:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            return self._working_quantities.get(key, Decimal(0))

    def effective_position(self, client_id: str, instrument_id: InstrumentId) -> Decimal:
        key = AccountTargetKey(client_id, instrument_id)
        with self._lock:
            return (
                self._account_positions.get(key, Decimal(0))
                + self._working_quantities.get(key, Decimal(0))
            )

    def clear_working(self, *, client_id: str | None = None) -> None:
        with self._lock:
            if client_id is None:
                self._working_quantities.clear()
            else:
                for key in tuple(self._working_quantities):
                    if key.client_id == client_id:
                        del self._working_quantities[key]

    def snapshot(self) -> PositionSnapshot:
        with self._lock:
            return PositionSnapshot(
                strategy_positions=self._strategy_positions,
                account_positions=self._account_positions,
                working_quantities=self._working_quantities,
            )

    @staticmethod
    def _guard_revision(
        revisions: dict,
        key: object,
        revision: int | None,
        name: str,
    ) -> None:
        if revision is None:
            return
        if revision < 0:
            raise ValueError("revision 不能为负数")
        current = revisions.get(key)
        if current is not None and revision <= current:
            raise StaleRevisionError(
                f"{name}版本必须递增: current={current} received={revision}",
            )
        revisions[key] = revision
