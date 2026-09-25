"""Offline checks for the one-shot SimNow account guards."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

# Direct execution must import the probe from this checkout, not an installed
# examples package in the active Python environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.single_ema.ctp_simnow_order_probe import (
    assert_account_ready,
    wait_target_position,
)
from market.basic.base import InstrumentId


class Driver:
    def __init__(self, active_orders=()):
        self.active_orders = active_orders

    def reconcile_active_orders(self):
        return SimpleNamespace(orders=self.active_orders)

    def reconcile_account_state(self):
        return None


class Transport:
    def __init__(self, positions):
        self.positions = positions

    def query_gross_positions(self):
        return self.positions


def rejected(driver, transport, target, allow_other_positions):
    try:
        assert_account_ready(
            driver, transport, target,
            allow_other_positions=allow_other_positions,
        )
    except RuntimeError:
        return
    raise AssertionError("unsafe account passed the order guard")


def main():
    target = InstrumentId.from_str("rb2701.SHFE")
    old_position = {"rb2610.SHFE": (Decimal(1), Decimal(0))}
    rejected(Driver(), Transport(old_position), target, False)
    assert assert_account_ready(
        Driver(), Transport(old_position), target,
        allow_other_positions=True,
    ) == old_position
    rejected(
        Driver(), Transport({"rb2701.SHFE": (Decimal(0), Decimal(1))}),
        target, True,
    )
    rejected(Driver(active_orders=(object(),)), Transport({}), target, True)
    assert assert_account_ready(
        Driver(), Transport({}), target,
        allow_other_positions=False,
    ) == {}
    filled = dict(old_position, **{"rb2701.SHFE": (Decimal(1), Decimal(0))})
    assert wait_target_position(
        Transport(filled), target, (Decimal(1), Decimal(0)),
        old_position, 0.1,
    ) == filled
    try:
        wait_target_position(
            Transport({"rb2610.SHFE": (Decimal(2), Decimal(0))}),
            target, (Decimal(0), Decimal(0)), old_position, 0.1,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("other contract changed during round-trip")
    print("SimNow单笔探针账户保护通过")


if __name__ == "__main__":
    main()
