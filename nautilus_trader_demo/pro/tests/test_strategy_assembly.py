"""公共组装与受管理实盘Runtime验收；假通道，不联网、不读取真实凭据。"""

from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from bomber.framework.market.basic.base import DataType, InstrumentId, InstrumentMeta, MarketDataFeed, make_bar
from bomber.framework.trader import (
    DataBinding, ExecutionBackendKind, ExecutionReport, ExecutionReportType, ExecutionRoute,
    MarketStreamBinding, NautilusMarketFeedAdapter, PositionManager, RiskLimits, RuntimeMode,
    StrategyTemplate, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)
from bomber.framework.trader.assembly import StrategyBindings, assemble_strategy
from bomber.framework.trader.execution.builders import build_simulation_execution
from bomber.framework.trader.runtime.managed import LiveRunContext, ManagedLiveRuntime
from bomber.framework.trader.runtime.reports import LiveRunReport
from bomber.framework.datahub.sector_roles import SectorRoleAssignment, SectorRoleStore


class Controller:
    event_label = "测试通道"

    def __init__(self, trace, fail=None):
        self.trace, self.fail = trace, fail

    def step(self, name):
        self.trace.append(name)
        if name == self.fail:
            raise RuntimeError(name)

    def prepare(self, resources):
        resources.callback(lambda: self.trace.append("release"))
        self.step("prepare")
        return LiveRunContext("20260922", resources)

    def finish_preparation(self, context):
        self.step("finish_preparation")

    def start(self, session, context):
        self.step("start")

    def poll(self, session):
        self.step("poll")
        return False

    def verify(self, session):
        self.step("verify")

    def shutdown(self, session, context):
        self.step("shutdown")
        return {"final_active_orders": 0, "cleanup_errors": []}

    def snapshot(self, session, context):
        return {"orders_submitted": 0}

    def events(self, session):
        return ()


class ManagedRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.trace = []

    def runtime(self, fail=None, mode=RuntimeMode.LIVE):
        controller = Controller(self.trace, fail)

        def inputs(context):
            controller.step("inputs")
            return object()

        def assemble(inputs, context):
            controller.step("assemble")
            return SimpleNamespace(runner=SimpleNamespace(mode=mode))

        return ManagedLiveRuntime("other-strategy", controller=controller,
            prepare_inputs=inputs, assemble_session=assemble,
            report=LiveRunReport(self.root, clock_ns=lambda: 1), seconds=1,
            monotonic=lambda: 0, sleep=lambda seconds: None)

    def summary(self):
        return json.loads((self.root / "live-1/summary.json").read_text())

    def test_success_order_cached_result_and_idempotent_stop(self):
        runtime = self.runtime()
        self.assertEqual(self.trace, [])
        result = runtime.run()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.trace, ["prepare", "inputs", "finish_preparation", "assemble",
            "start", "poll", "verify", "shutdown", "release"])
        self.assertIs(runtime.run(), result)
        runtime.stop()
        self.assertEqual(self.trace.count("shutdown"), 1)
        with self.assertRaisesRegex(RuntimeError, "不能重新启动"):
            runtime.start()

    def test_prepare_inputs_assembly_start_poll_and_verify_failures_release_resources(self):
        for phase in ("prepare", "inputs", "finish_preparation", "assemble", "start", "poll", "verify"):
            with self.subTest(phase=phase), TemporaryDirectory() as root:
                self.root, self.trace = Path(root), []
                runtime = self.runtime(fail=phase)
                with self.assertRaisesRegex(RuntimeError, phase):
                    runtime.run()
                self.assertEqual(self.summary()["status"], "failed")
                self.assertEqual(self.summary()["failure"], phase)
                self.assertEqual(self.trace.count("release"), 1)
                self.assertEqual(self.trace.count("shutdown"), 1)
                runtime.stop()
                with self.assertRaisesRegex(RuntimeError, "不能重新启动"):
                    runtime.run()

    def test_shutdown_failure_cannot_report_passed_and_still_releases(self):
        runtime = self.runtime(fail="shutdown")
        with self.assertRaisesRegex(RuntimeError, "停机查询"):
            runtime.run()
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["cleanup_errors"], ["shutdown"])
        self.assertEqual(self.trace.count("release"), 1)
        runtime.stop()

    def test_wrong_mode_does_not_start_controller(self):
        runtime = self.runtime(mode=RuntimeMode.HISTORICAL)
        with self.assertRaisesRegex(ValueError, "仅接受LIVE"):
            runtime.run()
        self.assertNotIn("start", self.trace)
        self.assertEqual(self.trace[-1], "release")

    def test_input_resource_close_failure_marks_failed_without_skipping_other_cleanup(self):
        runtime = self.runtime()
        prepare = runtime.prepare_inputs

        def close_source():
            raise RuntimeError("source_close_failed")

        def inputs(context):
            context.resources.callback(close_source)
            return prepare(context)

        runtime.prepare_inputs = inputs
        with self.assertRaisesRegex(RuntimeError, "source_close_failed"):
            runtime.run()
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["cleanup_errors"], ["source_close_failed"])
        self.assertEqual(self.trace.count("shutdown"), 1)
        self.assertEqual(self.trace.count("release"), 1)

    def test_stop_before_start_never_connects_or_writes(self):
        runtime = self.runtime()
        runtime.stop()
        runtime.stop()
        self.assertEqual(self.trace, [])
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaisesRegex(RuntimeError, "不能重新启动"):
            runtime.start()

    def test_report_collision_preserves_existing_evidence(self):
        path = self.root / "live-1"
        path.mkdir()
        marker = path / "summary.json"
        marker.write_text("original")
        runtime = self.runtime()
        with self.assertRaises(FileExistsError):
            runtime.run()
        self.assertEqual(marker.read_text(), "original")
        self.assertEqual(self.trace, [])


RB = InstrumentId.from_str("rb2704.SHFE")


class ReplayFeed(MarketDataFeed):
    def __init__(self, count=2):
        super().__init__("ASSEMBLY_REPLAY")
        meta = InstrumentMeta(RB, price_precision=0, size_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10), exchange="SHFE")
        self.register_instrument(meta)
        base = int(datetime(2026, 9, 22, tzinfo=UTC).timestamp() * 1_000_000_000)
        self.events = tuple(make_bar(RB, 100, 101, 99, 100, 10, base + index * 60_000_000_000,
            meta=meta, bar_type="1-MINUTE") for index in range(count))

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def _on_subscription_added(self, request):
        pass

    def _on_subscription_removed(self, request):
        pass

    def replay(self):
        for event in self.events:
            self._emit_bar(event)
        return {"bars": len(self.events)}


class Backend:
    kind = ExecutionBackendKind.SIMULATION
    backend_id = "assembly-sim"

    def __init__(self, trace):
        self.trace, self.handlers, self.orders = trace, [], []
        self.started, self.filled = False, 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def register_report_handler(self, handler):
        self.handlers.append(handler)

    def cancel_strategy(self, strategy_id):
        pass

    def submit_order(self, order):
        self.orders.append(order)
        self.trace.append("submit")

    def process_market_event(self, event):
        self.trace.append(("market", event.ts_event))
        reports = []
        while self.filled < len(self.orders):
            index = self.filled
            order = self.orders[index]
            report = ExecutionReport(backend_id=self.backend_id,
                client_order_id=f"assembly-{index}", instrument_id=order.instrument_id,
                report_type=ExecutionReportType.FILLED, ts_event=event.ts_event,
                filled_quantity=order.quantity, fill_price=100, order_side=order.side,
                order_quantity=order.quantity, position_effect=order.position_effect,
                report_id=f"fill-{index}", sequence=1,
                metadata={"strategy_id": order.strategy_id, "trade_id": f"trade-{index}"})
            for handler in tuple(self.handlers):
                handler(report)
            self.trace.append(("fill", event.ts_event))
            reports.append(report)
            self.filled += 1
        return tuple(reports)

    def result(self):
        return {"orders": len(self.orders), "fills": self.filled}

    def finish(self):
        return self.result()


class OneShot(StrategyTemplate):
    def __init__(self, trace):
        super().__init__("other-strategy")
        self.trace, self.sent, self.observed = trace, False, []

    def on_bar(self, data_key, bar):
        self.trace.append(("strategy", bar.ts_event))
        self.observed.append(self.account_position("position"))
        if not self.sent:
            self.sent = True
            self.set_target("position", 1, bar.ts_event)


class AssemblyTests(unittest.TestCase):
    def test_shared_assembly_preserves_next_event_fill_and_position_accounting(self):
        trace = []
        backend, feed, strategy = Backend(trace), ReplayFeed(), OneShot(trace)
        execution = build_simulation_execution(backend, instrument_limits={RB: RiskLimits(
            max_order_quantity=Decimal(1), max_abs_position=Decimal(1), contract_multiplier=Decimal(10))})
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=execution.positions)
        bindings = StrategyBindings(
            data=(DataBinding("bar", "bars", RB, DataType.BAR, "1-MINUTE"),),
            execution=(ExecutionRoute("position", execution.client.client_id, RB),))
        assemble_strategy(runner, strategy, feeds={"bars": feed}, execution=execution, bindings=bindings)
        adapter = NautilusMarketFeedAdapter("clock", feed, backend,
            (MarketStreamBinding(RB, DataType.BAR, "1-MINUTE"),), manage_lifecycle=False)
        runtime = UnifiedHistoricalRuntime("assembled", runner, adapter)
        self.addCleanup(runtime.stop)
        self.assertFalse(backend.started)
        self.assertFalse(feed.is_connected)
        result = runtime.run()
        first, second = (event.ts_event for event in feed.events)
        self.assertEqual(trace, [("market", first), ("strategy", first), "submit",
            ("market", second), ("fill", second), ("strategy", second)])
        self.assertEqual(result.backend_result, {"orders": 1, "fills": 1})
        self.assertEqual(strategy.observed, [Decimal(0), Decimal(1)])
        self.assertEqual(execution.positions.account_position(backend.backend_id, RB), 1)
        self.assertEqual(execution.positions.working_quantity(backend.backend_id, RB), 0)
        self.assertEqual(execution.client.report_errors, ())
        runtime.stop()
        self.assertFalse(feed.is_connected)
        self.assertFalse(backend.started)

    def test_split_position_managers_are_rejected_before_registration(self):
        execution = build_simulation_execution(Backend([]), instrument_limits={})
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=PositionManager())
        with self.assertRaisesRegex(ValueError, "共享PositionManager"):
            assemble_strategy(runner, OneShot([]), feeds={}, execution=execution,
                bindings=StrategyBindings((), ()))

    def test_current_backtest_entry_assembles_and_runs_current_strategy(self):
        backtest = import_module("demos.01_main_ema.run_backtest")
        from bomber.framework.trader import ContractAssignment, ScheduledContractResolver
        references = SectorRoleStore((SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21),
            0, 0, {"RB": {"main": "rb2704"}}, {"RB": {"main": Decimal(2)}}),))
        backend, feed = Backend([]), ReplayFeed(count=3)
        resolver = ScheduledContractResolver((ContractAssignment("rb_main", RB, 0, 0, 1),))
        runtime, strategy, client = backtest.assemble(config=backtest.MainEmaConfig("RB", "SHFE", 1, 2),
            references=references, feed=feed, backend=backend,
            instruments={"rb2704": RB}, multipliers={RB: Decimal(10)},
            resolver=resolver, max_notional=Decimal(50000))
        self.assertIs(strategy.data_hub, references)
        self.assertIs(runtime.backend, backend)
        self.assertIs(client.position_manager, runtime.runner.position_manager)
        self.assertIs(runtime.mode, RuntimeMode.HISTORICAL)
        self.assertFalse(backend.started)
        self.assertFalse(feed.is_connected)
        self.addCleanup(runtime.stop)
        result = runtime.run()
        self.assertEqual(strategy.bars_used, 3)
        self.assertEqual(strategy.last_target, Decimal(1))
        self.assertEqual(strategy.fast.value, 200.0)
        self.assertEqual(result.backend_result, {"orders": 1, "fills": 1})
        self.assertEqual(client.position_manager.account_position(backend.backend_id, RB), 1)
        self.assertEqual(client.report_errors, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
