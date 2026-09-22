"""P3受控在线执行：全程内存Driver，不连接账户、不发送真实/DEMO订单。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import InstrumentId
from strategy import (
    AccountStateEvent,
    ActiveOrderSnapshot,
    ControlledLiveExecutionClient,
    CurrencyBalance,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRequest,
    MarketReferencePriceStore,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    OrderSide,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RiskRejected,
)


BTC = InstrumentId.from_str("BTCUSDT.BINANCE")


class FakeDemoDriver:
    driver_id = "demo-client"

    def __init__(self) -> None:
        self.report_sink = None
        self.orders = []
        self.positions = {}
        self.fail_account_query = False
        self.account_revision = 1

    def start(self, report_sink) -> None:
        self.report_sink = report_sink

    def stop(self) -> None:
        self.report_sink = None

    def submit_order(self, order) -> None:
        assert self.report_sink is not None
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        return dict(self.positions)

    def reconcile_account_state(self) -> AccountStateEvent:
        if self.fail_account_query:
            raise RuntimeError("账户资金查询失败")
        self.account_revision += 1
        return AccountStateEvent(
            client_id=self.driver_id,
            account_id="demo-account",
            revision=self.account_revision,
            ts_event=100,
            balances={"USDT": CurrencyBalance("USDT", equity=100_000, available=90_000)},
        )

    def emit(self, report: ExecutionReport) -> None:
        assert self.report_sink is not None
        self.report_sink(report)


def build_client(*, demo_verified: bool = True):
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    prices.update(BTC, 80_000, 100)
    risk = PreTradeRiskManager(
        "demo-client", positions, prices,
        default_limits=RiskLimits(
            max_order_quantity=2,
            max_abs_position=2,
            max_order_notional=160_000,
            max_market_age_ns=10,
        ),
    )
    driver = FakeDemoDriver()
    backend = NautilusLiveExecutionBackend("demo-client", driver)
    client = ControlledLiveExecutionClient(
        "demo-client", NetTargetOrderPlanner(positions), backend, positions, risk,
        account_id="demo-account",
        demo_environment_check=lambda: demo_verified,
    )
    return client, driver, positions, risk


def request(target: int, revision: int = 1) -> ExecutionRequest:
    return ExecutionRequest(
        strategy_id="p3-alpha", revision=revision, client_id="demo-client",
        ts_event=105, targets={BTC: Decimal(target)}, execution_policy="DIRECT",
    )


def report(kind, sequence: int, *, filled: int = 0, trade_id: str | None = None,
           order_id: str = "p3-order-1", quantity: int = 2, reason: str | None = None):
    metadata = {"strategy_id": "p3-alpha", "account_id": "demo-account"}
    if trade_id:
        metadata["trade_id"] = trade_id
    return ExecutionReport(
        backend_id="demo-client", client_order_id=order_id, instrument_id=BTC,
        report_type=kind, ts_event=sequence, filled_quantity=filled,
        fill_price=80_000 if filled else None, order_side=OrderSide.BUY,
        order_quantity=quantity, report_id=f"p3-report-{sequence}", sequence=sequence,
        reason=reason,
        metadata=metadata,
    )


def test1_read_only_and_authorization() -> None:
    """P3a：只读对账成功也不能下单；必须核实DEMO且显式授权。"""
    client, driver, positions, _ = build_client()
    client.start()
    assert client.is_reconciled and client.account_state is not None
    assert client.account_state.balances["USDT"].equity == 100_000
    try:
        client.submit_targets(request(1))
    except PermissionError:
        pass
    else:
        raise AssertionError("只读状态不得下单")
    assert not driver.orders and positions.working_quantity("demo-client", BTC) == 0
    try:
        client.arm_demo("wrong")
    except PermissionError:
        pass
    else:
        raise AssertionError("错误确认语不得开闸")
    client.arm_demo(ControlledLiveExecutionClient.DEMO_CONFIRMATION)
    assert client.is_armed
    client.disarm()
    assert not client.is_armed
    client.stop()

    unsafe, _, _, _ = build_client(demo_verified=False)
    unsafe.start()
    try:
        unsafe.arm_demo(ControlledLiveExecutionClient.DEMO_CONFIRMATION)
    except PermissionError:
        pass
    else:
        raise AssertionError("未核实DEMO环境不得开闸")
    unsafe.stop()
    print("P3a通过：权威资金/仓位只读对账、DEMO环境核验和显式授权正常")


def test2_report_and_risk() -> None:
    """P3b：假柜台回报经过状态机；风控拒单无副作用。"""
    client, driver, positions, risk = build_client()
    events = []
    client.register_execution_event_handler(events.append)
    client.start()
    client.arm_demo(ControlledLiveExecutionClient.DEMO_CONFIRMATION)
    try:
        client.submit_targets(request(3))
    except RiskRejected:
        pass
    else:
        raise AssertionError("超出数量/仓位上限必须拒单")
    assert not driver.orders
    client.submit_targets(request(2))
    assert len(driver.orders) == 1
    driver.emit(report(ExecutionReportType.ACCEPTED, 1))
    driver.emit(report(ExecutionReportType.PARTIALLY_FILLED, 2, filled=1, trade_id="p3-trade-1"))
    driver.emit(report(ExecutionReportType.PARTIALLY_FILLED, 2, filled=1, trade_id="p3-trade-1"))
    assert positions.account_position("demo-client", BTC) == 1
    assert positions.working_quantity("demo-client", BTC) == 1
    driver.emit(report(ExecutionReportType.CANCELED, 3))
    assert positions.working_quantity("demo-client", BTC) == 0
    assert [event.status.value for event in events if hasattr(event, "status")] == [
        "ACCEPTED", "PARTIALLY_FILLED", "CANCELED",
    ]
    assert len([event for event in events if hasattr(event, "trade_id")]) == 1
    client.submit_targets(request(2, revision=2))
    assert len(driver.orders) == 2
    driver.emit(report(ExecutionReportType.REJECTED, 4, order_id="p3-order-2",
                       quantity=1, reason="fake venue rejection"))
    assert positions.account_position("demo-client", BTC) == 1
    assert positions.working_quantity("demo-client", BTC) == 0
    assert [event.status.value for event in events if hasattr(event, "status")][-1] == "REJECTED"
    assert not client.report_errors and client.is_armed
    assert any(item.action == "ORDER_REJECTED" for item in client.audit_records)
    client.stop()
    print("P3b通过：部分成交/撤单/去重、风控拒单和审计记录正常")


def test3_disconnect_and_failed_query() -> None:
    """P3c：断线撤销授权，活动订单必须权威恢复，资金查询失败闭闸。"""
    client, driver, positions, _ = build_client()
    client.start()
    client.arm_demo(ControlledLiveExecutionClient.DEMO_CONFIRMATION)
    client.mark_disconnected("test_disconnect")
    assert not client.is_armed and not client.is_reconciled
    assert positions.is_recovery_required("demo-client")
    try:
        client.reconcile()
    except RuntimeError as error:
        assert "活动订单" in str(error)
    else:
        raise AssertionError("缺少活动订单权威恢复不得重连开闸")
    driver.reconcile_active_orders = lambda: ActiveOrderSnapshot(
        client_id="demo-client", account_id="demo-account", revision=1,
        ts_event=100, orders=(),
    )
    client.recover_active_orders()
    driver.fail_account_query = True
    try:
        client.reconcile()
    except RuntimeError as error:
        assert "资金查询失败" in str(error)
    else:
        raise AssertionError("资金查询失败不得重连开闸")
    assert not client.is_reconciled and not client.is_armed
    driver.fail_account_query = False
    driver.reconcile_active_orders = lambda: ActiveOrderSnapshot(
        client_id="demo-client", account_id="demo-account", revision=2,
        ts_event=101, orders=(),
    )
    client.recover_active_orders()
    client.reconcile()
    assert client.is_reconciled and not client.is_armed
    client.stop()
    print("P3c通过：断线/活动订单/资金失败均闭闸，重连仍需重新授权")


STAGES = {"read_only": test1_read_only_and_authorization,
          "reports": test2_report_and_risk,
          "reconnect": test3_disconnect_and_failed_query}


def main() -> None:
    parser = argparse.ArgumentParser(description="P3受控在线执行无网络测试")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for name, function in STAGES.items():
        if args.stage in (name, "all"):
            function()


if __name__ == "__main__":
    main()
