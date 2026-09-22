"""外部目标计划的最小 DataHub 契约；不读取文件、不依赖行情或交易客户端。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping


class TargetPlanUnavailable(LookupError):
    """目标计划在触发时刻尚未发布。"""


@dataclass(frozen=True)
class TargetPlan:
    """一个时点的一份完整组合目标；零数量表示明确平仓。"""

    slot_ns: int
    available_ns: int
    targets: Mapping[str, Decimal]
    source_revision: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slot_ns <= 0 or self.available_ns < 0:
            raise ValueError("目标时点必须为正，发布时间不能为负")
        if self.available_ns > self.slot_ns:
            raise ValueError("目标计划不能在触发时刻之后才发布")
        if not self.targets or any(not key.strip() for key in self.targets):
            raise ValueError("完整目标组合至少需要一个非空目标键")
        normalized = {key: Decimal(str(value)) for key, value in self.targets.items()}
        if any(not value.is_finite() for value in normalized.values()):
            raise ValueError("目标数量必须为有限数值")
        object.__setattr__(self, "targets", MappingProxyType(normalized))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class TargetScheduleStore:
    """按时间索引的不可变目标计划；完整快照语义由每个 TargetPlan 保证。"""

    def __init__(self, plans: tuple[TargetPlan, ...]) -> None:
        ordered = tuple(sorted(plans, key=lambda plan: plan.slot_ns))
        if len({plan.slot_ns for plan in ordered}) != len(ordered):
            raise ValueError("同一目标时点不能有两个版本；先在 Provider 完成版本选择")
        self.plans = ordered
        self.slots = tuple(plan.slot_ns for plan in ordered)
        self._by_slot = MappingProxyType({plan.slot_ns: plan for plan in ordered})

    def at(self, slot_ns: int, *, as_of_ns: int | None = None) -> TargetPlan | None:
        plan = self._by_slot.get(slot_ns)
        if plan is None:
            return None
        observed_ns = slot_ns if as_of_ns is None else as_of_ns
        if observed_ns < plan.available_ns:
            raise TargetPlanUnavailable(f"{slot_ns}目标版本尚未发布")
        if observed_ns < slot_ns:
            raise ValueError("不能在目标时点之前触发计划")
        return plan

    def slots_between(self, after_ns: int, through_ns: int) -> tuple[int, ...]:
        if through_ns < after_ns:
            raise ValueError("时钟不能倒退")
        left = bisect_right(self.slots, after_ns)
        right = bisect_right(self.slots, through_ns)
        return self.slots[left:right]
