"""外部目标计划的时点信号与审计，不读取文件或提交订单。"""

from __future__ import annotations

from dataclasses import dataclass

from bomber.framework.datahub.target_schedule import TargetPlan, TargetScheduleStore


@dataclass(frozen=True)
class ScheduleAudit:
    """记录计划时点是否提交、错过或在回放结束前未到达。"""
    slot_ns: int
    status: str
    detail: str = ""


class ScheduledTargetSignal:
    """只在计划的精确时点输出目标，时钟越过的计划不补发。"""

    def __init__(self, schedule: TargetScheduleStore) -> None:
        self.schedule = schedule
        self._last_clock_ns = -1
        self._handled: set[int] = set()
        self._audit: list[ScheduleAudit] = []

    @property
    def audit(self) -> tuple[ScheduleAudit, ...]:
        """返回不可修改的当前审计快照。"""
        return tuple(self._audit)

    def update(self, ts_event: int) -> tuple[TargetPlan, ...]:
        """推进递增时钟，输出正好到达且已可用的计划，重复时钟不重复输出。"""
        if ts_event < self._last_clock_ns:
            raise ValueError("策略时钟不能倒退")
        if ts_event == self._last_clock_ns:
            return ()
        due = []
        for slot in self.schedule.slots_between(self._last_clock_ns, ts_event):
            if slot in self._handled:
                continue
            if slot < ts_event:
                self._audit.append(ScheduleAudit(slot, "MISSED", "时钟越过计划时点；未追单"))
                self._handled.add(slot)
                continue
            plan = self.schedule.at(slot, as_of_ns=ts_event)
            assert plan is not None
            due.append(plan)
        self._last_clock_ns = ts_event
        return tuple(due)

    def mark_submitted(self, slot_ns: int) -> None:
        """策略成功提交目标后才记为已提交；并不表示已经成交。"""
        if slot_ns not in self.schedule.slots or slot_ns != self._last_clock_ns:
            raise ValueError("只能确认当前时钟对应的计划时点")
        if slot_ns not in self._handled:
            self._audit.append(ScheduleAudit(slot_ns, "SUBMITTED"))
            self._handled.add(slot_ns)

    def finalize_audit(self) -> None:
        """回放结束后补齐未处理时点；重复调用不重复记录。"""
        for slot in self.schedule.slots:
            if slot not in self._handled:
                self._audit.append(ScheduleAudit(slot, "NOT_REACHED", "回放结束前未到该时点或未完成提交"))
                self._handled.add(slot)
