"""外部完整目标计划的 CSV 读取、时区及整数手数校验。"""

from io import StringIO

from demos.scheduled_targets.local_input import load_target_csv


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


if __name__ == "__main__":
    test_complete_target_plan()
    test_reject_fractional_lots()
    print("scheduled targets CSV: OK")
