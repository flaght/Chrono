"""P3活动订单权威恢复安全边界；使用内存柜台，不连接交易账户。"""

from __future__ import annotations

from decimal import Decimal

from strategy import ActiveOrder, ActiveOrderSnapshot, ExecutionReportType
from tests.run_p3_controlled_live import BTC, build_client, report, request


def _snapshot(*orders: ActiveOrder, revision: int = 1) -> ActiveOrderSnapshot:
    return ActiveOrderSnapshot(
        client_id="demo-client", account_id="demo-account",
        revision=revision, ts_event=10, orders=orders,
    )


def _active(*, filled: int = 1, quantity: int = 2, order_id: str = "p3-order-1") -> ActiveOrder:
    return ActiveOrder(
        client_order_id=order_id, instrument_id=str(BTC), side="BUY",
        order_quantity=quantity, cumulative_filled=filled,
        remaining_quantity=quantity - filled,
    )


def _disconnected_with_partial():
    client, driver, positions, _ = build_client()
    client.start()
    client.arm_demo(client.DEMO_CONFIRMATION)
    client.submit_targets(request(2))
    driver.emit(report(ExecutionReportType.ACCEPTED, 1))
    driver.emit(report(ExecutionReportType.PARTIALLY_FILLED, 2, filled=1, trade_id="fill-1"))
    driver.positions = {BTC: Decimal(1)}
    client.mark_disconnected("fake_disconnect")
    return client, driver, positions


def test1_exact_snapshot_recovers_without_arming() -> None:
    client, driver, positions = _disconnected_with_partial()
    driver.reconcile_active_orders = lambda: _snapshot(_active())
    client.recover_active_orders()
    assert not positions.is_recovery_required("demo-client")
    assert positions.working_quantity("demo-client", BTC) == 1
    client.reconcile()
    assert positions.account_position("demo-client", BTC) == 1
    assert not client.is_armed
    client.stop()
    print("P3e1通过：已知活动订单完全一致时恢复在途量，但不会自动重新授权")


def test2_mismatch_and_unknown_fail_closed() -> None:
    cases = (
        (lambda: _snapshot(), "旧订单消失"),
        (lambda: _snapshot(_active(filled=0)), "断线期间成交量变化"),
        (lambda: _snapshot(_active(), _active(order_id="unknown", quantity=1, filled=0)),
         "未知外部订单"),
    )
    for snapshot, label in cases:
        client, driver, positions = _disconnected_with_partial()
        driver.reconcile_active_orders = snapshot
        try:
            client.recover_active_orders()
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{label}必须拒绝自动恢复")
        assert positions.is_recovery_required("demo-client")
        assert not client.is_reconciled and not client.is_armed
        client.stop()
    print("P3e2通过：消失订单、断线成交和未知订单均保持闭闸")


def test3_start_requires_authoritative_orders() -> None:
    client, driver, positions = _disconnected_with_partial()
    client.stop()
    try:
        client.start()
    except RuntimeError as error:
        assert "活动订单" in str(error) or "reconcile_active_orders" in str(error)
    else:
        raise AssertionError("没有权威订单查询时重启不能开放闸门")
    assert positions.is_recovery_required("demo-client")
    driver.reconcile_active_orders = lambda: _snapshot(_active())
    client.start()
    assert client.is_reconciled and not client.is_armed
    assert positions.account_position("demo-client", BTC) == 1
    assert positions.working_quantity("demo-client", BTC) == 1
    client.stop()
    print("P3e3通过：重启必须先恢复权威活动订单，再恢复仓位与资金")


def test4_manual_working_clear_cannot_bypass() -> None:
    client, driver, positions = _disconnected_with_partial()
    # 旧底层API可以直接改在途量；受控客户端仍保留本次断线待验证标志。
    positions.complete_working_recovery("demo-client", {BTC: 0})
    try:
        client.reconcile()
    except RuntimeError as error:
        assert "活动订单" in str(error)
    else:
        raise AssertionError("手工清零在途量不得绕过权威订单快照")
    driver.reconcile_active_orders = lambda: _snapshot(_active())
    client.recover_active_orders()
    client.reconcile()
    assert positions.working_quantity("demo-client", BTC) == 1
    client.stop()
    print("P3e4通过：底层手工清零在途量无法绕过受控客户端订单验证")


def test5_stale_snapshot_cannot_reopen() -> None:
    client, driver, positions = _disconnected_with_partial()
    driver.reconcile_active_orders = lambda: _snapshot(_active(), revision=1)
    client.recover_active_orders()
    client.reconcile()
    client.mark_disconnected("second_disconnect")
    try:
        client.recover_active_orders()
    except ValueError as error:
        assert "版本" in str(error)
    else:
        raise AssertionError("旧订单快照版本不得再次开闸")
    assert positions.is_recovery_required("demo-client")
    client.stop()
    print("P3e5通过：重复旧活动订单快照版本不能再次解锁")


def main() -> None:
    test1_exact_snapshot_recovers_without_arming()
    test2_mismatch_and_unknown_fail_closed()
    test3_start_requires_authoritative_orders()
    test4_manual_working_clear_cannot_bypass()
    test5_stale_snapshot_cannot_reopen()


if __name__ == "__main__":
    main()
