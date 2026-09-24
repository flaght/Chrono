"""F1a：权威账户仓位对账与实盘下单闸门测试。

全部测试使用内存Driver，不连接柜台，也不会发送真实订单。
"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import InstrumentId
from trader import (
    BackendExecutionClient,
    ExecutionRequest,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    OrderIntent,
    OrderSide,
    PositionManager,
    StaleRevisionError,
)


BTC = InstrumentId.from_str("BTCUSDT.BINANCE")
ETH = InstrumentId.from_str("ETHUSDT.BINANCE")


class _PositionDriver:
    """可控的权威仓位柜台替身。"""

    driver_id = "live-account"

    def __init__(self, positions=None, *, fail_reconcile: bool = False) -> None:
        self.positions = {} if positions is None else dict(positions)
        self.fail_reconcile = fail_reconcile
        self.started = False
        self.stopped = False
        self.orders: list[OrderIntent] = []

    def start(self, report_sink) -> None:
        del report_sink
        self.started = True
        self.stopped = False

    def stop(self) -> None:
        self.started = False
        self.stopped = True

    def submit_order(self, order: OrderIntent) -> None:
        if not self.started:
            raise RuntimeError("Driver尚未启动")
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        if self.fail_reconcile:
            raise RuntimeError("模拟柜台对账失败")
        return dict(self.positions)


def _request(target: int) -> ExecutionRequest:
    return ExecutionRequest(
        strategy_id="f1-alpha",
        revision=1,
        client_id="live-account",
        ts_event=1,
        targets={BTC: Decimal(target)},
        execution_policy="DIRECT",
    )


def test1_authoritative_snapshot_contract() -> None:
    """F1a1：Backend把Driver原始仓位标准化为带单调版本的快照。"""

    driver = _PositionDriver({"BTCUSDT.BINANCE": "2.5", ETH: -1})
    backend = NautilusLiveExecutionBackend("live-account", driver)
    backend.start()
    first = backend.reconcile()
    second = backend.reconcile()
    backend.stop()

    assert first.backend_id == "live-account"
    assert first.revision == 1 and second.revision == 2
    assert first.positions[BTC] == Decimal("2.5")
    assert first.positions[ETH] == Decimal(-1)
    assert second.ts_event >= first.ts_event
    print("F1a1通过：Live Backend可生成标准、不可变、单调版本的权威仓位快照")


def test2_reconciliation_gate_and_atomic_replace() -> None:
    """F1a2：启动前禁止下单；对账后才规划，并清除快照缺失的旧仓。"""

    positions = PositionManager()
    positions.set_account_position("live-account", ETH, 9)
    driver = _PositionDriver({BTC: 2})
    backend = NautilusLiveExecutionBackend("live-account", driver)
    client = BackendExecutionClient(
        "live-account",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
    )

    try:
        client.submit_targets(_request(3))
    except RuntimeError as error:
        assert "尚未完成" in str(error)
    else:
        raise AssertionError("对账前必须拒绝下单")

    client.start()
    assert client.is_reconciled
    assert positions.is_account_reconciled("live-account")
    assert positions.account_position("live-account", BTC) == 2
    assert positions.account_position("live-account", ETH) == 0

    client.submit_targets(_request(3))
    assert len(driver.orders) == 1
    assert driver.orders[0].side is OrderSide.BUY
    assert driver.orders[0].quantity == 1
    assert positions.working_quantity("live-account", BTC) == 1

    state = positions.account_reconciliation("live-account")
    assert state is not None and state.revision == 1
    try:
        positions.apply_account_snapshot(
            "live-account",
            {BTC: 100},
            revision=1,
            ts_event=state.ts_event,
        )
    except StaleRevisionError:
        pass
    else:
        raise AssertionError("重复权威快照版本必须被拒绝")

    client.stop()
    assert not client.is_reconciled
    assert not positions.is_account_reconciled("live-account")
    print("F1a2通过：权威快照原子替仓，下单闸门及停止后失效规则正常")


def test3_failed_reconciliation_keeps_gate_closed() -> None:
    """F1a3：柜台对账失败时回滚启动并保持禁止下单。"""

    positions = PositionManager()
    driver = _PositionDriver(fail_reconcile=True)
    backend = NautilusLiveExecutionBackend("live-account", driver)
    client = BackendExecutionClient(
        "live-account",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
    )
    try:
        client.start()
    except RuntimeError as error:
        assert "对账失败" in str(error)
    else:
        raise AssertionError("对账失败必须终止启动")

    assert driver.stopped
    assert not backend.is_started
    assert not client.is_reconciled
    assert not positions.is_account_reconciled("live-account")
    try:
        client.submit_targets(_request(1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("失败后的下单闸门必须保持关闭")
    print("F1a3通过：对账失败会停止Backend，且不会开放下单闸门")


STAGES = {
    1: test1_authoritative_snapshot_contract,
    2: test2_reconciliation_gate_and_atomic_replace,
    3: test3_failed_reconciliation_keeps_gate_closed,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1a权威仓位对账测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
