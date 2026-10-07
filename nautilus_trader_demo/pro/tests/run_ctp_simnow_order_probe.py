"""单次 SimNow 报单账户保护条件的离线检查。"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

# 直接执行时必须从当前源码目录导入探针，不能使用环境中已安装的
# examples 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.single_ema.ctp_simnow_order_probe import (
    assert_account_ready,
    assert_close_ready,
    wait_target_position,
)
from market.basic.base import InstrumentId
from trader.execution.contracts import OrderSide


class Driver:
    def __init__(self, active_orders=(), trading_day="20260923"):
        self.active_orders = active_orders
        self.trading_day = trading_day

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
    assert assert_close_ready(
        Driver(), Transport(filled), target, side=OrderSide.SELL,
        expected_trading_day="20260923", allow_other_positions=True,
    ) == old_position
    for driver, positions, side, day in (
        (Driver(trading_day="20260924"), filled, OrderSide.SELL, "20260923"),
        (Driver(active_orders=(object(),)), filled, OrderSide.SELL, "20260923"),
        (Driver(), filled, OrderSide.BUY, "20260923"),
        (Driver(), old_position, OrderSide.SELL, "20260923"),
    ):
        try:
            assert_close_ready(
                driver, Transport(positions), target, side=side,
                expected_trading_day=day, allow_other_positions=True,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe close-only account passed the guard")
    print("SimNow单笔探针账户保护通过")


if __name__ == "__main__":
    main()
