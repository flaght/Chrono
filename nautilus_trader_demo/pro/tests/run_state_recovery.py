"""F1d：状态文件、完整快照恢复及重启安全闸门测试。"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import tempfile

from market.basic.base import InstrumentId
from trader import (
    AccountTargetKey,
    BackendExecutionClient,
    ConcurrentStateWriteError,
    CtpPositionLedger,
    ExecutionReport,
    ExecutionReportType,
    JsonStateStore,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    OrderReportStateMachine,
    OrderSide,
    PortfolioCoordinator,
    PositionEffect,
    PositionManager,
    RuntimeStateManager,
    StateCorruptionError,
    TargetPortfolio,
    TargetStore,
)


BTC = InstrumentId.from_str("BTCUSDT.BINANCE")
RB = InstrumentId.from_str("rb2704.SHFE")


def test1_atomic_file_and_corruption_detection() -> None:
    """F1d1：generation防并发覆盖，校验和发现人工修改或半文件。"""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runtime-state.json"
        repository = JsonStateStore(path)
        first = repository.save({"probe": 1}, expected_generation=0)
        assert first.generation == 1
        assert path.stat().st_mode & 0o777 == 0o600

        try:
            JsonStateStore(path).save({"probe": 2}, expected_generation=0)
        except ConcurrentStateWriteError:
            pass
        else:
            raise AssertionError("旧generation不能覆盖新状态")

        document = json.loads(path.read_text(encoding="utf-8"))
        document["payload"]["probe"] = 999
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            repository.load()
        except StateCorruptionError:
            pass
        else:
            raise AssertionError("校验和必须识别被修改的状态文件")
    print("F1d1通过：原子状态文件、generation CAS和损坏检测正常")


def _partial_report() -> ExecutionReport:
    return ExecutionReport(
        backend_id="live-account",
        client_order_id="RECOVER-1",
        instrument_id=BTC,
        report_type=ExecutionReportType.PARTIALLY_FILLED,
        ts_event=2,
        filled_quantity=2,
        fill_price=80_000,
        order_side=OrderSide.BUY,
        order_quantity=5,
        report_id="RECOVER-1:2",
        sequence=2,
    )


def _build_populated_state(path: Path):
    targets = TargetStore()
    targets.apply(
        TargetPortfolio(
            strategy_id="alpha",
            revision=3,
            ts_event=100,
            targets={"btc": 5},
            metadata={"threshold": Decimal("1.25")},
        ),
    )
    portfolio = PortfolioCoordinator()
    portfolio.update(
        strategy_id="alpha",
        revision=3,
        ts_event=100,
        targets={AccountTargetKey("live-account", BTC): 5},
    )
    positions = PositionManager()
    positions.set_strategy_position("alpha", "btc", 2, revision=3)
    positions.apply_account_snapshot(
        "live-account",
        {BTC: 2},
        revision=1,
        ts_event=100,
    )
    positions.set_working_quantity("live-account", BTC, 3)

    orders = OrderReportStateMachine("live-account")
    orders.apply(
        ExecutionReport(
            backend_id="live-account",
            client_order_id="RECOVER-1",
            instrument_id=BTC,
            report_type=ExecutionReportType.ACCEPTED,
            ts_event=1,
            order_side=OrderSide.BUY,
            order_quantity=5,
            report_id="RECOVER-1:1",
            sequence=1,
        ),
    )
    orders.apply(_partial_report())

    ledger = CtpPositionLedger("20260918")
    ledger.apply_fill(RB, OrderSide.BUY, PositionEffect.OPEN, 2, 3100, 10)
    manager = RuntimeStateManager(
        JsonStateStore(path),
        targets,
        portfolio,
        positions,
        order_machines={"live-account": orders},
        ctp_ledgers={"ctp-account": ledger},
    )
    manager.save()


class _RecoveryDriver:
    driver_id = "live-account"

    def __init__(self) -> None:
        self.stops = 0

    def start(self, report_sink) -> None:
        self.report_sink = report_sink

    def stop(self) -> None:
        self.stops += 1

    def submit_order(self, order) -> None:
        del order

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        return {BTC: 2}


def _empty_recovery(path: Path):
    targets = TargetStore()
    portfolio = PortfolioCoordinator()
    positions = PositionManager()
    driver = _RecoveryDriver()
    backend = NautilusLiveExecutionBackend("live-account", driver)
    client = BackendExecutionClient(
        "live-account",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
    )
    ledger = CtpPositionLedger()
    manager = RuntimeStateManager(
        JsonStateStore(path),
        targets,
        portfolio,
        positions,
        order_machines={"live-account": client.order_state_machine},
        ctp_ledgers={"ctp-account": ledger},
    )
    return manager, targets, portfolio, positions, client, driver, ledger


def test2_full_state_round_trip() -> None:
    """F1d2：Target、组合、仓位、订单幂等键及CTP今昨仓完整恢复。"""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runtime-state.json"
        _build_populated_state(path)
        manager, targets, portfolio, positions, client, _, ledger = _empty_recovery(path)
        summary = manager.restore()
        assert summary is not None and summary.generation == 1
        assert summary.targets == 1 and summary.orders == 1
        assert summary.ctp_accounts == 1
        assert summary.requires_authoritative_reconciliation

        target = targets.get("alpha")
        assert target is not None and target.revision == 3
        assert target.metadata["threshold"] == Decimal("1.25")
        key = AccountTargetKey("live-account", BTC)
        assert portfolio.snapshot().targets[key] == 5
        assert positions.account_position("live-account", BTC) == 2
        assert positions.working_quantity("live-account", BTC) == 3
        # 进程重启后，本地保存的已对账标志必须失效。
        assert not positions.is_account_reconciled("live-account")
        assert positions.is_recovery_required("live-account")

        order = client.order_state("RECOVER-1")
        assert order is not None and order.filled_quantity == 2
        # 同一report_id/sequence在恢复后重放，仍不得重复记账。
        duplicate = client.order_state_machine.apply(_partial_report())
        assert not duplicate.applied
        ctp = ledger.snapshot(RB)
        assert ctp.long_today == 2 and ctp.long_today_basis == 3100

        second = manager.save()
        assert second.generation == 2
    print("F1d2通过：Target、组合、仓位、订单幂等状态和CTP账本可完整恢复")


def test3_restart_reconciliation_gate() -> None:
    """F1d3：持仓对账不能替代活动订单恢复，二者完成前保持禁单。"""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runtime-state.json"
        _build_populated_state(path)
        manager, _, _, positions, client, driver, _ = _empty_recovery(path)
        manager.restore()

        # start已查询权威持仓，但仍检测到重启前活动订单，所以启动必须失败。
        try:
            client.start()
        except RuntimeError as error:
            assert "活动订单" in str(error)
        else:
            raise AssertionError("未恢复活动订单时不能开放执行客户端")
        assert driver.stops == 1
        assert not client.is_reconciled

        # 模拟柜台查询确认仍有3手买单在途；只有该权威结果可以解除恢复闸门。
        positions.complete_working_recovery("live-account", {BTC: 3})
        client.start()
        assert client.is_reconciled
        assert positions.is_account_reconciled("live-account")
        assert positions.working_quantity("live-account", BTC) == 3
        client.stop()
    print("F1d3通过：重启后必须分别完成账户仓位和活动订单权威恢复")


STAGES = {
    1: test1_atomic_file_and_corruption_detection,
    2: test2_full_state_round_trip,
    3: test3_restart_reconciliation_gate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1d状态持久化与恢复测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
