"""三角色演示信号与四角色 DataHub 默认行为的离线回归检查。"""

from datetime import date
from decimal import Decimal
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datahub import MinimalDataHub, ObservedClose, RoleAssignment, RolePriceStore

signal_module = import_module("demos.05_role_cross.role_signal")
SIGNAL_ROLES = signal_module.SIGNAL_ROLES
RoleCrossSignal = signal_module.RoleCrossSignal


def test_three_role_signal_ignores_recent() -> None:
    day = date(2026, 5, 18)
    contracts = {"main": "rb2610", "secondary": "rb2612", "far": "rb2701"}
    assignment = RoleAssignment(day, date(2026, 5, 15), 100, 100,
                                contracts, roles=SIGNAL_ROLES)
    closes = tuple(
        ObservedClose(symbol, day, timestamp, Decimal(price))
        for timestamp, prices in ((100, (100, 100, 100)),
                                  (200, (105, 100, 100)),
                                  (300, (95, 100, 100)))
        for symbol, price in zip(contracts.values(), prices)
    )
    store = RolePriceStore((assignment,), closes)
    signal = RoleCrossSignal()
    hub = MinimalDataHub(store)
    assert set(hub.snapshot(100).prices) == set(SIGNAL_ROLES)
    assert signal.update(hub.snapshot(100)) is None
    assert signal.update(hub.snapshot(200)).direction == 1
    assert signal.update(hub.snapshot(300)).direction == -1


def test_four_role_default_unchanged() -> None:
    contracts = {"main": "rb2610", "secondary": "rb2612",
                 "near": "rb2605", "far": "rb2701"}
    assignment = RoleAssignment(date(2026, 5, 18), date(2026, 5, 15),
                                100, 100, contracts)
    closes = tuple(ObservedClose(symbol, assignment.trading_day, 100, Decimal(100))
                   for symbol in contracts.values())
    store = RolePriceStore((assignment,), closes)
    assert set(store.snapshot(100)) == set(contracts)


def test_no_cross_and_duplicate_timestamp() -> None:
    contracts = {"main": "rb2610", "secondary": "rb2612", "far": "rb2701"}
    day = date(2026, 5, 18)
    assignment = RoleAssignment(day, date(2026, 5, 15), 100, 100,
                                contracts, roles=SIGNAL_ROLES)
    closes = tuple(ObservedClose(symbol, day, timestamp, Decimal(price))
                   for timestamp, prices in ((100, (105, 100, 100)),
                                              (200, (106, 100, 100)),
                                              (300, (95, 100, 100)))
                   for symbol, price in zip(contracts.values(), prices))
    hub = MinimalDataHub(RolePriceStore((assignment,), closes))
    signal = RoleCrossSignal()
    assert signal.update(hub.snapshot(100)) is None
    assert signal.update(hub.snapshot(200)) is None
    assert signal.update(hub.snapshot(200)) is None
    assert signal.positive_frames == 2
    assert signal.update(hub.snapshot(300)).direction == -1


if __name__ == "__main__":
    test_three_role_signal_ignores_recent()
    test_four_role_default_unchanged()
    test_no_cross_and_duplicate_timestamp()
    print("three-role signal and four-role compatibility: OK")
