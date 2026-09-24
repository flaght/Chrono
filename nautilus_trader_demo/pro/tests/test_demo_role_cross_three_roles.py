"""三角色演示信号与四角色 DataHub 默认行为的离线回归检查。"""

from datetime import date
from decimal import Decimal

from datahub import MinimalDataHub, ObservedClose, RoleAssignment, RolePriceStore
from demos.role_cross.local_input import SIGNAL_ROLES
from demos.role_cross.role_signal import RoleCrossSignal


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


if __name__ == "__main__":
    test_three_role_signal_ignores_recent()
    test_four_role_default_unchanged()
    print("three-role signal and four-role compatibility: OK")
