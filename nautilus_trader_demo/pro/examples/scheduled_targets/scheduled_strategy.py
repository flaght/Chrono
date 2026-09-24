"""目标文件驱动策略；只处理时点与完整目标，不读文件、不直接下单。"""

from __future__ import annotations

from dataclasses import dataclass

from datahub.target_schedule import TargetScheduleStore
from trader.contracts import TargetUpdateMode
from trader.template import StrategyTemplate


@dataclass(frozen=True)
class ScheduleAudit:
    slot_ns: int
    status: str
    detail: str = ""


class ScheduledTargetStrategy(StrategyTemplate):
    """在独立时钟上逐时点提交完整组合；迟到时点禁止补发过期订单。"""

    def __init__(self, strategy_id: str, schedule: TargetScheduleStore) -> None:
        super().__init__(strategy_id)
        self.schedule = schedule
        self._last_clock_ns = -1
        self._handled: set[int] = set()
        self._audit: list[ScheduleAudit] = []

    @property
    def audit(self) -> tuple[ScheduleAudit, ...]:
        return tuple(self._audit)

    def on_time(self, ts_event: int) -> None:
        if ts_event < self._last_clock_ns:
            raise ValueError("策略时钟不能倒退")
        if ts_event == self._last_clock_ns:
            return
        for slot in self.schedule.slots_between(self._last_clock_ns, ts_event):
            if slot in self._handled:
                continue
            if slot < ts_event:
                self._audit.append(ScheduleAudit(slot, "MISSED", "时钟越过计划时点；未追单"))
                self._handled.add(slot)
                continue
            plan = self.schedule.at(slot, as_of_ns=ts_event)
            assert plan is not None
            # 一次 set_targets 即一个完整目标版本；下游 Bomber 批次协议不进入策略。
            self.set_targets(
                plan.targets,
                ts_event,
                update_mode=TargetUpdateMode.REPLACE,
                metadata={"schedule_slot_ns": slot, "source_revision": plan.source_revision,
                          **plan.metadata},
            )
            self._audit.append(ScheduleAudit(slot, "SUBMITTED"))
            self._handled.add(slot)
        self._last_clock_ns = ts_event

    def on_stop(self) -> None:
        for slot in self.schedule.slots:
            if slot not in self._handled:
                self._audit.append(ScheduleAudit(slot, "NOT_REACHED", "回放结束前未到该时点"))
