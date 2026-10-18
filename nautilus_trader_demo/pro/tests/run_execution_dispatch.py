"""P1：假Live Backend验证状态优先、策略隔离和回报去重，不访问网络。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader import (
    BackendExecutionClient,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRoute,
    NetTargetOrderPlanner,
    OrderSide,
    PositionManager,
    RuntimeMode,
    StrategyTemplate,
    UnifiedStrategyRunner,
)
from bomber.framework.trader.execution.contracts import AccountPositionSnapshot


RB = InstrumentId.from_str("rb2704.SHFE")
SA = InstrumentId.from_str("SA703.CZCE")


class FakeBackend:
    backend_id = "fake-live"

    def __init__(self) -> None:
        self.handler = None
        self.orders = []

    def register_report_handler(self, handler) -> None:
        self.handler = handler

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reconcile(self) -> AccountPositionSnapshot:
        return AccountPositionSnapshot(self.backend_id, 1, 0, {})

    def submit_order(self, order) -> None:
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def emit(self, report: ExecutionReport) -> None:
        assert self.handler is not None
        self.handler(report)


class Observer(StrategyTemplate):
    def __init__(self, strategy_id: str) -> None:
        super().__init__(strategy_id)
        self.events: list[tuple] = []

    def on_order_update(self, event) -> None:
        self.events.append(("order", event.status.value, self.account_position("leg"),
                            self.working_quantity("leg")))

    def on_fill(self, event) -> None:
        self.events.append(("fill", event.trade_id, self.account_position("leg"),
                            self.working_quantity("leg")))


class OrderObserver(Observer):
    def on_order(self, event) -> None:
        Observer.on_order_update(self, event)


class BothOrderObserver(OrderObserver):
    def on_order_update(self, event) -> None:
        raise AssertionError("同时重写时不应自动二次调用旧订单入口")


def _setup(*, legacy: bool = False, same_instrument: bool = False, observer_type=Observer):
    positions = PositionManager()
    backend = FakeBackend()
    client = BackendExecutionClient(
        "fake-live", NetTargetOrderPlanner(positions), backend, positions, account_id="account-a",
    )
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE, position_manager=positions)
    runner.add_execution_client(client)
    alpha = observer_type("alpha")
    beta = StrategyTemplate("beta") if legacy else Observer("beta")
    runner.add_strategy(alpha, data_bindings=(),
                        execution_routes=(ExecutionRoute("leg", "fake-live", RB),))
    runner.add_strategy(beta, data_bindings=(),
                        execution_routes=(ExecutionRoute(
                            "leg", "fake-live", RB if same_instrument else SA,
                        ),))
    runner.start()
    alpha.set_target("leg", 2, 1)
    assert len(backend.orders) == 1
    return runner, client, backend, alpha, beta, positions


def _report(status, sequence: int, *, filled: int = 0, strategy_id: str | None = "alpha",
            order_metadata=None,
            trade_id: str | None = None, reason: str | None = None) -> ExecutionReport:
    metadata = dict(order_metadata or {})
    if strategy_id is not None:
        metadata["strategy_id"] = strategy_id
    if trade_id is not None:
        metadata["trade_id"] = trade_id
    return ExecutionReport(
        backend_id="fake-live", client_order_id="order-1", instrument_id=RB,
        report_type=status, ts_event=sequence, filled_quantity=filled,
        fill_price=3126 if filled else None, order_side=OrderSide.BUY,
        order_quantity=2, report_id=f"order-1:{sequence}", sequence=sequence,
        metadata=metadata, reason=reason,
    )


def test1_ordering_and_isolation() -> None:
    runner, client, backend, alpha, beta, positions = _setup(same_instrument=True)
    try:
        metadata = backend.orders[0].metadata
        backend.emit(_report(ExecutionReportType.ACCEPTED, 1, order_metadata=metadata))
        partial = _report(ExecutionReportType.PARTIALLY_FILLED, 2, filled=1, trade_id="trade-1", order_metadata=metadata)
        backend.emit(partial)
        backend.emit(partial)  # 重发不得二次通知。
        backend.emit(_report(ExecutionReportType.ACCEPTED, 1))  # 乱序回报不得重放。
        backend.emit(_report(ExecutionReportType.ACCEPTED, 4))  # 新序号的旧状态也不通知。
        backend.emit(_report(ExecutionReportType.FILLED, 5, filled=1, trade_id="trade-2"))
        assert alpha.events == [
            ("order", "ACCEPTED", Decimal(0), Decimal(2)),
            ("order", "PARTIALLY_FILLED", Decimal(1), Decimal(1)),
            ("fill", "trade-1", Decimal(1), Decimal(1)),
            ("order", "FILLED", Decimal(2), Decimal(0)),
            ("fill", "trade-2", Decimal(2), Decimal(0)),
        ]
        assert not beta.events
        assert positions.account_position("fake-live", RB) == 2
        assert not client.report_errors
    finally:
        runner.stop()
    print("P1a通过：先更新仓位再通知、订单先于成交、跨策略隔离及重复乱序抑制")


def test2_legacy_and_fail_closed() -> None:
    runner, client, backend, alpha, beta, positions = _setup(legacy=True)
    try:
        assert beta.is_started  # 旧策略不实现新回调仍可启动。
        backend.emit(_report(ExecutionReportType.ACCEPTED, 1, order_metadata=backend.orders[0].metadata))
        backend.emit(_report(ExecutionReportType.PARTIALLY_FILLED, 2,
                             filled=1, trade_id="trade-1", strategy_id="beta"))
        assert not client.is_reconciled and client.report_errors
        assert positions.account_position("fake-live", RB) == 1
        assert len(alpha.events) == 1  # 错误归属不能投递给其他策略。
    finally:
        runner.stop()

    runner, client, backend, alpha, _, _ = _setup(legacy=True)
    try:
        backend.emit(_report(ExecutionReportType.ACCEPTED, 1, strategy_id=None))
        assert not alpha.events and client.report_errors and not client.is_reconciled
    finally:
        runner.stop()
    runner, client, backend, alpha, _, _ = _setup(legacy=True)
    try:
        backend.emit(_report(ExecutionReportType.PARTIALLY_FILLED, 1, filled=1))
        assert not alpha.events and client.report_errors and not client.is_reconciled
    finally:
        runner.stop()
    print("P1b通过：旧策略兼容；错误归属或缺失关联键关闭闸门而不串投")


def test3_order_callback_compatibility() -> None:
    for observer_type in (Observer, OrderObserver, BothOrderObserver):
        runner, client, backend, alpha, beta, positions = _setup(observer_type=observer_type)
        try:
            metadata = backend.orders[0].metadata
            backend.emit(_report(ExecutionReportType.ACCEPTED, 1, order_metadata=metadata))
            partial = _report(ExecutionReportType.PARTIALLY_FILLED, 2,
                filled=1, trade_id="trade-1", order_metadata=metadata)
            backend.emit(partial)
            backend.emit(partial)
            backend.emit(_report(ExecutionReportType.CANCELED, 3, order_metadata=metadata))
            assert alpha.events == [
                ("order", "ACCEPTED", Decimal(0), Decimal(2)),
                ("order", "PARTIALLY_FILLED", Decimal(1), Decimal(1)),
                ("fill", "trade-1", Decimal(1), Decimal(1)),
                ("order", "CANCELED", Decimal(1), Decimal(0)),
            ], observer_type.__name__
            assert not beta.events
            assert positions.position("alpha", "leg") == 1
            assert not client.report_errors
        finally:
            runner.stop()
        runner, client, backend, alpha, _, positions = _setup(observer_type=observer_type)
        try:
            backend.emit(_report(ExecutionReportType.REJECTED, 1,
                order_metadata=backend.orders[0].metadata, reason="测试柜台拒单"))
            assert alpha.events == [("order", "REJECTED", Decimal(0), Decimal(0))]
            assert positions.position("alpha", "leg") == 0
            assert not client.report_errors
        finally:
            runner.stop()
    print("P1c通过：on_order与旧on_order_update兼容、双入口不重复、订单先于成交")


STAGES = {"dispatch": test1_ordering_and_isolation, "failure": test2_legacy_and_fail_closed,
          "callbacks": test3_order_callback_compatibility}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for name, function in STAGES.items():
        if args.stage in (name, "all"):
            function()
    print("P1 execution dispatch tests OK")


if __name__ == "__main__":
    main()
