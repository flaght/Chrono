"""独立时钟触发外部完整目标组合；不读取 CSV 或 Bar。"""

from __future__ import annotations

from bomber.framework.datahub.target_schedule import TargetScheduleStore
from bomber.framework.trader.contracts import TargetUpdateMode
from bomber.framework.trader.template import StrategyTemplate
from schedule_signal import ScheduleAudit, ScheduledTargetSignal


class ScheduledTargetStrategy(StrategyTemplate):
    """在计划信号到达时提交完整组合，未列出的旧目标按替换语义归零。"""
    def __init__(self, strategy_id: str, schedule: TargetScheduleStore) -> None:
        super().__init__(strategy_id)
        self.schedule = schedule
        self.signal = ScheduledTargetSignal(schedule)

    @property
    def audit(self) -> tuple[ScheduleAudit, ...]:
        """提供信号模块的计划执行审计。"""
        return self.signal.audit

    def on_time(self, ts_event: int) -> None:
        """由独立时钟触发；计划选择由信号模块负责，策略仅提交目标。"""
        for plan in self.signal.update(ts_event):
            self.set_targets(
                plan.targets, ts_event, update_mode=TargetUpdateMode.REPLACE,
                metadata={"schedule_slot_ns": plan.slot_ns, "source_revision": plan.source_revision,
                          **plan.metadata},
            )
            self.signal.mark_submitted(plan.slot_ns)

    def finalize_audit(self) -> None:
        """回放结束后补齐未到达时点；重复调用不重复记录。"""
        self.signal.finalize_audit()

    def on_stop(self) -> None:
        """停止时确保审计覆盖所有计划时点。"""
        self.finalize_audit()
