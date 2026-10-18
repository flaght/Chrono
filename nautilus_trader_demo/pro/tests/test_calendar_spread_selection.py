"""跨期选约范围及纯信号窗口规则；在部署环境手动执行。"""

from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = import_module("demos.03_calendar_spread.calendar_signal")
available_pair = module.available_pair
CalendarSpreadSignal = module.CalendarSpreadSignal


def test_secondary_far_pair_ignores_recent() -> None:
    symbols = {"main": "i2605", "secondary": "i2609", "far": "i2701"}
    present = {"i2605", "i2609", "i2701"}
    assert available_pair(symbols, "secondary", "far", present) == ("i2609", "i2701")
    assert available_pair(symbols, "main", "secondary", present) == ("i2605", "i2609")


def test_no_two_contracts() -> None:
    symbols = {"main": "i2605", "secondary": "i2609", "far": "i2701"}
    assert available_pair(symbols, "secondary", "far", {"i2605"}) is None


def test_selection_new_old_and_default_roles() -> None:
    from datetime import date
    import pandas as pd

    day = date(2026, 1, 5)
    roles = pd.DataFrame([dict(trade_date="2025-12-30", code="RB",
                               second="rb2610", far="rb2701")])
    available = {day: {"rb2610", "rb2701"}}
    expected = (((day, "rb2610", "rb2701"),), ())
    for arguments in ({}, dict(leg1_role="secondary", leg2_role="far"),
                      dict(near_role="secondary", far_role="far")):
        assert module.select_roles(roles, (day,), "RB", available=available,
                                   **arguments) == expected
    try:
        module.select_roles(roles, (day,), "RB", available=available,
                            leg1_role="secondary", near_role="main")
    except ValueError as exc:
        assert "冲突" in str(exc)
    else:
        raise AssertionError("冲突的角色参数应被拒绝")


def test_past_window_and_reverse() -> None:
    signal = CalendarSpreadSignal(2, 2.0, 0.5)
    assert signal.update(0.0, 0) is None
    assert signal.update(2.0, 0) is None
    # 此前窗口 [0, 2] 的均值/标准差都是 1；当前 4 不进入基准。
    result = signal.update(4.0, 0)
    assert result.z_score == 3.0 and result.direction == -1
    result = signal.update(0.0, -1)
    assert result.z_score == -3.0 and result.direction == 1


def test_flat_window_and_reset() -> None:
    signal = CalendarSpreadSignal(2, 2.0, 0.5)
    signal.update(1.0, 0)
    signal.update(1.0, 0)
    result = signal.update(1.0, 1)
    assert result.z_score == 0.0 and result.direction == 0
    signal.reset()
    assert signal.update(10.0, 0) is None
    assert signal.update(12.0, 0) is None
    assert signal.update(14.0, 0).z_score == 3.0


def test_keep_direction_between_thresholds() -> None:
    signal = CalendarSpreadSignal(2, 2.0, 0.5)
    signal.update(0.0, 0)
    signal.update(2.0, 0)
    result = signal.update(2.0, -1)
    assert result.z_score == 1.0 and result.direction == -1


if __name__ == "__main__":
    test_secondary_far_pair_ignores_recent()
    test_no_two_contracts()
    test_selection_new_old_and_default_roles()
    test_past_window_and_reverse()
    test_flat_window_and_reset()
    test_keep_direction_between_thresholds()
    print("calendar selection and signal: OK")
