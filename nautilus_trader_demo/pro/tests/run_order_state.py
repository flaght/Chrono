"""F1b：部分成交、终态释放、重复及乱序执行回报测试。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import InstrumentId
from trader import (
    BackendExecutionClient,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRequest,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    OrderIntent,
    OrderLifecycleStatus,
    OrderReportStateMachine,
    OrderSide,
    OrderStateError,
    PositionManager,
)


BTC = InstrumentId.from_str("BTCUSDT.BINANCE")


def _report(
    report_type: ExecutionReportType,
    *,
    order_id: str = "F1B-1",
    sequence: int,
    quantity: int = 5,
    filled: int = 0,
) -> ExecutionReport:
    return ExecutionReport(
        backend_id="live-account",
        client_order_id=order_id,
        instrument_id=BTC,
        report_type=report_type,
        ts_event=sequence,
        filled_quantity=filled,
        fill_price=Decimal("80000") if filled else None,
        order_side=OrderSide.BUY,
        order_quantity=quantity,
        report_id=f"{order_id}:{sequence}",
        sequence=sequence,
    )


def test1_state_machine_idempotency() -> None:
    """F1b1：部分成交只能应用一次，重复和倒序回报不重复记仓。"""

    machine = OrderReportStateMachine("live-account")
    accepted = machine.apply(_report(ExecutionReportType.ACCEPTED, sequence=1))
    partial_report = _report(
        ExecutionReportType.PARTIALLY_FILLED,
        sequence=2,
        filled=2,
    )
    partial = machine.apply(partial_report)
    duplicate = machine.apply(partial_report)
    out_of_order = machine.apply(
        _report(ExecutionReportType.ACCEPTED, sequence=1),
    )
    filled = machine.apply(
        _report(ExecutionReportType.FILLED, sequence=3, filled=3),
    )

    assert accepted.applied and accepted.fill_delta == 0
    assert partial.applied and partial.fill_delta == 2
    assert partial.state.remaining_quantity == 3
    assert not duplicate.applied and duplicate.fill_delta == 0
    assert not out_of_order.applied
    assert filled.fill_delta == 3
    assert filled.state.status is OrderLifecycleStatus.FILLED
    assert filled.state.filled_quantity == 5

    # 终态之后到达的旧撤单不能把已经成交的订单重新释放一次。
    late_cancel = machine.apply(
        _report(ExecutionReportType.CANCELED, sequence=4),
    )
    assert not late_cancel.applied and late_cancel.release_delta == 0

    rejected_machine = OrderReportStateMachine("live-account")
    rejected_machine.apply(
        _report(
            ExecutionReportType.ACCEPTED,
            order_id="F1B-R",
            sequence=1,
            quantity=4,
        ),
    )
    rejected = rejected_machine.apply(
        _report(
            ExecutionReportType.REJECTED,
            order_id="F1B-R",
            sequence=2,
            quantity=4,
        ),
    )
    assert rejected.release_delta == 4
    assert rejected.state.status is OrderLifecycleStatus.REJECTED
    print("F1b1通过：部分成交、拒单、重复、乱序和终态回报均保持幂等")


class _ReportDriver:
    driver_id = "live-account"

    def __init__(self) -> None:
        self.sink = None
        self.orders: list[OrderIntent] = []

    def start(self, report_sink) -> None:
        self.sink = report_sink

    def stop(self) -> None:
        self.sink = None

    def submit_order(self, order: OrderIntent) -> None:
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        return {}

    def emit(self, report: ExecutionReport) -> None:
        if self.sink is None:
            raise RuntimeError("Driver尚未启动")
        self.sink(report)


def _request(target: int) -> ExecutionRequest:
    return ExecutionRequest(
        strategy_id="f1b-alpha",
        revision=1,
        client_id="live-account",
        ts_event=1,
        targets={BTC: Decimal(target)},
        execution_policy="DIRECT",
    )


def _client():
    positions = PositionManager()
    driver = _ReportDriver()
    backend = NautilusLiveExecutionBackend("live-account", driver)
    client = BackendExecutionClient(
        "live-account",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
    )
    client.start()
    return client, driver, positions


def test2_partial_fill_then_cancel() -> None:
    """F1b2：部分成交写真实仓位，撤单只释放未成交的剩余在途量。"""

    client, driver, positions = _client()
    client.submit_targets(_request(5))
    assert positions.working_quantity("live-account", BTC) == 5

    accepted = _report(ExecutionReportType.ACCEPTED, sequence=1)
    partial = _report(
        ExecutionReportType.PARTIALLY_FILLED,
        sequence=2,
        filled=2,
    )
    driver.emit(accepted)
    driver.emit(partial)
    driver.emit(partial)  # 模拟网络重发，同一成交不能重复入账。
    assert positions.account_position("live-account", BTC) == 2
    assert positions.working_quantity("live-account", BTC) == 3

    driver.emit(_report(ExecutionReportType.CANCELED, sequence=3))
    assert positions.account_position("live-account", BTC) == 2
    assert positions.working_quantity("live-account", BTC) == 0
    state = client.order_state("F1B-1")
    assert state is not None
    assert state.status is OrderLifecycleStatus.CANCELED
    assert state.filled_quantity == 2 and state.remaining_quantity == 3
    assert client.is_reconciled and not client.report_errors
    client.stop()
    print("F1b2通过：部分成交后撤单只释放剩余量，账户仓位和在途数量正确")


def test3_conflict_closes_order_gate() -> None:
    """F1b3：数量矛盾不猜测修复，必须关闭下单闸门等待重新核对。"""

    client, driver, positions = _client()
    client.submit_targets(_request(3))
    driver.emit(_report(ExecutionReportType.ACCEPTED, sequence=1, quantity=3))

    # FILLED只报告2手但原订单为3手，说明中间成交丢失或柜台字段语义错误。
    driver.emit(
        _report(
            ExecutionReportType.FILLED,
            sequence=2,
            quantity=3,
            filled=2,
        ),
    )
    assert not client.is_reconciled
    assert len(client.report_errors) == 1
    assert positions.account_position("live-account", BTC) == 0
    assert positions.working_quantity("live-account", BTC) == 3
    try:
        client.submit_targets(_request(1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("状态冲突后必须关闭下单闸门")

    machine = OrderReportStateMachine("live-account")
    machine.apply(_report(ExecutionReportType.ACCEPTED, sequence=1, quantity=1))
    try:
        machine.apply(
            _report(
                ExecutionReportType.FILLED,
                sequence=2,
                quantity=1,
                filled=2,
            ),
        )
    except OrderStateError:
        pass
    else:
        raise AssertionError("过量成交必须被识别为状态冲突")
    client.stop()
    print("F1b3通过：终态数量矛盾和过量成交会关闭闸门，不会污染仓位")


STAGES = {
    1: test1_state_machine_idempotency,
    2: test2_partial_fill_then_cancel,
    3: test3_conflict_closes_order_gate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1b执行回报状态机测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
