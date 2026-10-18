"""外部完整目标计划的 CSV 读取、时区及整数手数校验。"""

from io import StringIO
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bomber.framework.dataprep.schedules import load_target_csv
from bomber.framework.datahub.target_schedule import TargetPlan, TargetScheduleStore

ScheduledTargetSignal = import_module("demos.07_scheduled_targets.schedule_signal").ScheduledTargetSignal


class MemoryCsv:
    name = "targets.csv"

    def __init__(self, content: str) -> None:
        self.content = content

    def open(self, *args, **kwargs):
        return StringIO(self.content)


def test_complete_target_plan() -> None:
    schedule = load_target_csv(MemoryCsv(
        "timestamp,target_key,target_qty\n"
        "2026-05-06 09:40:00,rb2610.SHFE,1\n"
        "2026-05-06 09:40:00,hc2610.SHFE,-1\n"
        "2026-05-06 09:45:00,rb2610.SHFE,0\n"
    ))
    first, second = schedule.plans
    assert len(first.targets) == 2
    assert second.targets == {"rb2610.SHFE": 0}
    assert second.slot_ns - first.slot_ns == 300_000_000_000
    assert schedule.at(first.slot_ns).targets["hc2610.SHFE"] == -1


def test_reject_fractional_lots() -> None:
    try:
        load_target_csv(MemoryCsv(
            "timestamp,instrument_id,target_qty\n"
            "2026-05-06 09:40:00,rb2610.SHFE,0.5\n"
        ))
    except ValueError as exc:
        assert "手数" in str(exc)
    else:
        raise AssertionError("非整数期货手数不应被接受")


def test_exact_clock_and_duplicate_submission() -> None:
    schedule = TargetScheduleStore((TargetPlan(100, 0, {"rb2610.SHFE": 1}),))
    signal = ScheduledTargetSignal(schedule)
    assert signal.update(99) == ()
    assert signal.update(100) == schedule.plans
    assert signal.audit == ()
    signal.mark_submitted(100)
    signal.mark_submitted(100)
    assert signal.update(100) == ()
    assert [item.status for item in signal.audit] == ["SUBMITTED"]
    signal.finalize_audit()
    assert len(signal.audit) == 1


def test_missed_and_not_reached_slots() -> None:
    schedule = TargetScheduleStore(tuple(TargetPlan(slot, 0, {"rb2610.SHFE": 1})
                                         for slot in (100, 200, 300)))
    signal = ScheduledTargetSignal(schedule)
    assert signal.update(150) == ()
    assert signal.update(200)[0].slot_ns == 200
    signal.mark_submitted(200)
    signal.finalize_audit()
    signal.finalize_audit()
    assert [(item.slot_ns, item.status) for item in signal.audit] == [
        (100, "MISSED"), (200, "SUBMITTED"), (300, "NOT_REACHED"),
    ]
    try:
        signal.update(199)
    except ValueError:
        pass
    else:
        raise AssertionError("倒退时钟必须拒绝")


if __name__ == "__main__":
    test_complete_target_plan()
    test_reject_fractional_lots()
    test_exact_clock_and_duplicate_submission()
    test_missed_and_not_reached_slots()
    print("scheduled targets CSV and clock signal: OK")
