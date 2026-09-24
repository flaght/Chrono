"""逻辑目标的因果选约与安全换月；不依赖DataHub或具体交易客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol

from market.basic.base import InstrumentId


class ContractUnavailable(ValueError):
    """决策时刻尚无可用的真实合约；必须拒绝交易，不得使用未来记录。"""


@dataclass(frozen=True)
class ContractAssignment:
    target_key: str
    instrument_id: InstrumentId
    effective_ns: int
    available_ns: int
    revision: int

    def __post_init__(self) -> None:
        if not self.target_key.strip() or self.effective_ns < 0 or self.available_ns < 0:
            raise ValueError("合约角色和时间必须有效")
        if self.revision < 1:
            raise ValueError("合约表版本必须为正整数")


class ContractResolverPort(Protocol):
    """将来DataHub只须实现本接口，策略和换月逻辑不变。"""

    def resolve(self, target_key: str, as_of_ns: int) -> ContractAssignment: ...


class ScheduledContractResolver:
    """测试用的显式合约表；按生效时间和实际可用时间进行因果查询。"""

    def __init__(self, assignments: tuple[ContractAssignment, ...]) -> None:
        grouped: dict[str, list[ContractAssignment]] = {}
        for entry in assignments:
            grouped.setdefault(entry.target_key, []).append(entry)
        for key, entries in grouped.items():
            entries.sort(key=lambda item: item.effective_ns)
            if len({item.effective_ns for item in entries}) != len(entries):
                raise ValueError(f"同一目标存在重复生效时间: {key}")
            if any(b.revision <= a.revision for a, b in zip(entries, entries[1:])):
                raise ValueError(f"合约表版本必须递增: {key}")
        self._assignments = MappingProxyType({key: tuple(rows) for key, rows in grouped.items()})

    def resolve(self, target_key: str, as_of_ns: int) -> ContractAssignment:
        if as_of_ns < 0:
            raise ValueError("as_of_ns不能为负数")
        eligible = [item for item in self._assignments.get(target_key, ()) if item.effective_ns <= as_of_ns]
        if not eligible:
            raise ContractUnavailable(f"{target_key}在{as_of_ns}尚无生效合约")
        latest = eligible[-1]
        # 最新生效版本尚不可用时必须fail-closed，不能退回可能已到期的旧合约。
        if latest.available_ns > as_of_ns:
            raise ContractUnavailable(f"{target_key}的合约版本{latest.revision}尚不可用")
        return latest


@dataclass(frozen=True)
class DynamicExecutionRoute:
    """固定客户端、动态真实合约；只用于账户独占的换月试验。"""

    target_key: str
    client_id: str
    resolver: ContractResolverPort

    def __post_init__(self) -> None:
        if not self.target_key.strip() or not self.client_id.strip():
            raise ValueError("动态路由的目标键和客户端不能为空")


class RollPhase(str, Enum):
    ACTIVE = "ACTIVE"
    CANCELING = "CANCELING"
    CLOSING = "CLOSING"


@dataclass(frozen=True)
class RollState:
    active: ContractAssignment
    desired: Decimal
    phase: RollPhase = RollPhase.ACTIVE
    pending: ContractAssignment | None = None


@dataclass(frozen=True)
class RollDecision:
    targets: Mapping[InstrumentId, Decimal] | None = None
    cancel_strategy_orders: bool = False
    phase: RollPhase = RollPhase.ACTIVE

    def __post_init__(self) -> None:
        if self.targets is not None:
            object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))


class SafeRollCoordinator:
    """先撤、再平、确认旧仓和在途均为零，最后执行最新目标。

    只返回动作，不直接发送订单。调用方应在每次权威订单/仓位回报或行情时推进；
    中途新信号只覆盖desired，不会提前打开新合约。换月阶段可导出/恢复。
    """

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], RollState] = {}

    def state(self, strategy_id: str, target_key: str) -> RollState | None:
        return self._states.get((strategy_id, target_key))

    def snapshot(self) -> Mapping[tuple[str, str], RollState]:
        return MappingProxyType(dict(self._states))

    def restore(self, states: Mapping[tuple[str, str], RollState]) -> None:
        if any(not strategy_id or target_key != state.active.target_key for (strategy_id, target_key), state in states.items()):
            raise ValueError("换月状态与目标键不一致")
        self._states = dict(states)

    def step(
        self,
        strategy_id: str,
        selection: ContractAssignment,
        desired: Decimal | int | str,
        *,
        old_position: Decimal | int | str = 0,
        old_working: Decimal | int | str = 0,
        allow_open: bool = True,
    ) -> RollDecision:
        if not strategy_id.strip():
            raise ValueError("strategy_id不能为空")
        quantity = Decimal(str(desired))
        if not quantity.is_finite():
            raise ValueError("目标数量必须为有限值")
        key = (strategy_id, selection.target_key)
        current = self._states.get(key)
        if current is None:
            if not allow_open and quantity:
                raise RuntimeError("尚未完成账户恢复，禁止打开新合约")
            self._states[key] = RollState(selection, quantity)
            return RollDecision({selection.instrument_id: quantity})

        latest_revision = max(
            current.active.revision,
            current.pending.revision if current.pending is not None else 0,
        )
        if selection.revision < latest_revision:
            raise ValueError("不能用较旧合约表版本回退换月状态")

        # 若选约连续变化，只保留最新版本；换月中的目标数量也永远采用最新信号。
        pending = current.pending
        if selection.instrument_id != current.active.instrument_id:
            pending = selection
        elif current.phase is not RollPhase.ACTIVE:
            pending = selection
        if current.phase is RollPhase.ACTIVE:
            if pending is None:
                self._states[key] = RollState(selection, quantity)
                return RollDecision({selection.instrument_id: quantity})
            self._states[key] = RollState(current.active, quantity, RollPhase.CANCELING, pending)
            return RollDecision(cancel_strategy_orders=True, phase=RollPhase.CANCELING)

        if current.phase is RollPhase.CANCELING:
            self._states[key] = RollState(current.active, quantity, RollPhase.CANCELING, pending)
            if Decimal(str(old_working)):
                return RollDecision(phase=RollPhase.CANCELING)
            self._states[key] = RollState(current.active, quantity, RollPhase.CLOSING, pending)
            return RollDecision({current.active.instrument_id: Decimal(0)}, phase=RollPhase.CLOSING)

        self._states[key] = RollState(current.active, quantity, RollPhase.CLOSING, pending)
        if Decimal(str(old_working)):
            return RollDecision(phase=RollPhase.CLOSING)
        if Decimal(str(old_position)):
            # 撤单/平仓被拒后可在下一次推进重发0目标；不会提前开新合约。
            return RollDecision({current.active.instrument_id: Decimal(0)}, phase=RollPhase.CLOSING)
        if pending is None:
            raise RuntimeError("换月状态缺少待切换合约")
        if not allow_open and quantity:
            return RollDecision(phase=RollPhase.CLOSING)
        self._states[key] = RollState(pending, quantity)
        return RollDecision(
            {current.active.instrument_id: Decimal(0), pending.instrument_id: quantity},
        )
