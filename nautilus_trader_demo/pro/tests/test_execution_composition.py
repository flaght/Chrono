"""实际公共执行链无网络回归；仅替换SDK行情标识，不替换账本/状态机/风控。"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from importlib import import_module
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
import os
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Instrument:
    value: str

    @classmethod
    def from_str(cls, value):
        return cls(value)

    def __str__(self):
        return self.value


class DataType(str, Enum):
    BAR = "bar"
    CUSTOM_BAR = "custom_bar"
    TRADE_TICK = "trade_tick"
    QUOTE_TICK = "quote_tick"


class ExecutionCompositionTests(unittest.TestCase):
    def setUp(self):
        replacements = {}
        for name, path in (
                ("bomber", "bomber"), ("bomber.framework", "bomber/framework"),
                ("bomber.framework.market", "bomber/framework/market"),
                ("bomber.framework.market.basic", "bomber/framework/market/basic"),
                ("bomber.framework.trader", "bomber/framework/trader"),
                ("bomber.framework.trader.execution", "bomber/framework/trader/execution"),
                ("bomber.framework.trader.execution.live", "bomber/framework/trader/execution/live")):
            module = ModuleType(name)
            module.__path__ = [str(ROOT / path)]
            replacements[name] = module
        # 使用临时包命名空间，结束后完整恢复已有SDK模块，不污染其余测试组。
        for name in tuple(sys.modules):
            if name.startswith(("bomber.framework.trader.", "bomber.framework.market.basic.")):
                replacements[name] = None
        base = ModuleType("bomber.framework.market.basic.base")
        base.InstrumentId, base.DataType = Instrument, DataType
        for name in ("Bar", "CustomBar", "QuoteTick", "TradeTick"):
            setattr(base, name, type(name, (), {}))
        replacements[base.__name__] = base
        context = patch.dict(sys.modules, replacements)
        context.start()
        self.addCleanup(context.stop)
        for name, value in replacements.items():
            if value is None:
                del sys.modules[name]
        self.contracts = import_module("bomber.framework.trader.execution.contracts")
        self.policy = import_module("bomber.framework.trader.execution.policy")
        self.client_type = import_module("bomber.framework.trader.execution.live.client").BackendExecutionClient
        self.portfolio = import_module("bomber.framework.trader.portfolio")
        self.planner = import_module("bomber.framework.trader.execution.planner")
        self.risk = import_module("bomber.framework.trader.execution.risk")
        self.model = import_module("bomber.framework.trader.execution.model")
        self.fanout = import_module("bomber.framework.trader.execution.fanout")
        self.account = import_module("bomber.framework.trader.execution.account")
        self.request_type = import_module("bomber.framework.trader.contracts").ExecutionRequest
        self.instrument = Instrument("RB.SHFE")
        self.timestamp = 1791622919999999999

    def request(self, cid, *, strategy="s", quantity=1, revision=1, timestamp=None):
        return self.request_type(strategy, revision, cid, self.timestamp if timestamp is None else timestamp,
            {self.instrument: Decimal(quantity)}, "DIRECT", logical_targets={"main": Decimal(quantity)},
            metadata={"position_attribution": ({"strategy_id": strategy, "target_key": "main",
                "instrument_id": str(self.instrument), "quantity": str(quantity)},)})

    def client(self, cid, *, mode="off", real=20, model=False):
        positions = self.portfolio.PositionManager()
        prices = self.risk.MarketReferencePriceStore()
        prices.update(self.instrument, 3000, self.timestamp)
        owner = self

        class Backend:
            backend_id = cid
            kind = owner.contracts.ExecutionBackendKind.LIVE
            def __init__(self):
                self.handlers, self.sent = [], []
                self.queries = self.starts = self.stops = 0
                self.running = False
                self.revision = 0
            def register_report_handler(self, handler):
                self.handlers.append(handler)
            def start(self):
                self.starts += 1
                self.running = True
            def stop(self):
                self.stops += 1
                self.running = False
            def reconcile(self):
                self.queries += 1
                self.revision += 1
                return owner.contracts.AccountPositionSnapshot(cid, self.revision, owner.timestamp,
                    {owner.instrument: Decimal(real)})
            def submit_order(self, order):
                self.sent.append(order)
            def cancel_strategy(self, sid):
                pass
            def fill(self, index=0, *, duplicate=False):
                order = self.sent[index]
                report = owner.contracts.ExecutionReport(cid, f"{cid}-{index}", owner.instrument,
                    "FILLED", owner.timestamp, filled_quantity=order.quantity, fill_price=3000,
                    order_side=order.side, order_quantity=order.quantity,
                    position_effect=order.position_effect, report_id=f"{cid}-{index}:fill", sequence=1,
                    metadata={**order.metadata, "strategy_id": order.strategy_id, "trade_id": f"{cid}-{index}:trade"})
                for handler in self.handlers:
                    handler(report)
                    if duplicate:
                        handler(report)
        backend = self.model.ReferencePriceExecutionBackend(cid, prices) if model else Backend()
        client = self.client_type(cid, self.planner.NetTargetOrderPlanner(positions), backend, positions,
            self.risk.PreTradeRiskManager(cid, positions, prices,
                instrument_limits={self.instrument: self.risk.RiskLimits(max_order_quantity=5, max_abs_position=10)}),
            account_id=f"account:{cid}", account_reconciliation=mode)
        self.addCleanup(client.stop)
        return client

    def test_strict_default_uses_authoritative_position(self):
        c = self.client("td", mode="strict", real=20)
        c.start()
        self.assertEqual(c.backend.queries, 1)
        self.assertEqual(c.position_manager.account_position("td", self.instrument), 20)
        with self.assertRaises(Exception):
            c.submit_targets(self.request("td"))
        self.assertFalse(c.backend.sent)

    def test_off_never_queries_real_positions_and_tracks_only_own_fills(self):
        c = self.client("td", real=-20)
        c.backend.reconcile = lambda: self.fail("off不应查询柜台仓位")
        c.start()
        c.submit_targets(self.request("td"))
        self.assertEqual(c.position_manager.position("s", "main"), 0)  # 报单不等于成交。
        c.backend.fill(duplicate=True)
        self.assertEqual(c.position_manager.position("s", "main"), 1)
        c.reconcile()
        self.assertEqual(c.position_manager.account_position("td", self.instrument), 1)
        self.assertEqual(c.position_manager.unassigned_position("td", self.instrument), 0)

    def test_observe_does_not_override_own_fills(self):
        c = self.client("td", mode="observe", real=-20)
        c.start()
        c.submit_targets(self.request("td"))
        c.backend.fill()
        c.reconcile()
        self.assertEqual(c.backend.queries, 2)
        self.assertEqual(c.account_observation.positions[self.instrument], -20)
        self.assertEqual(c.position_manager.position("s", "main"), 1)
        self.assertEqual(c.position_differences[self.instrument], -21)

    def test_observation_query_failure_does_not_gate_independent_book(self):
        c = self.client("td", mode="observe")
        def fail():
            raise RuntimeError("仓位查询不可用")
        c.backend.reconcile = fail
        c.start()
        c.submit_targets(self.request("td"))
        c.backend.fill()
        self.assertEqual(c.position_manager.position("s", "main"), 1)
        self.assertIsNone(c.account_observation)
        self.assertEqual(c.account_observation_error, "仓位查询不可用")

    def test_fills_source_uses_only_selected_execution_callbacks(self):
        primary, follower = self.client("primary"), self.client("follower")
        front = self.fanout.TargetFanoutClient(primary, (self.fanout.AccountExecutionRoute(follower),))
        self.addCleanup(front.stop)
        events = []
        front.register_execution_event_handler(events.append)
        front.start()
        front.submit_targets(self.request("primary"))
        follower.backend.fill()
        self.assertEqual(primary.position_manager.position("s", "main"), 0)
        self.assertFalse(events)
        primary.backend.fill()
        self.assertEqual(primary.position_manager.position("s", "main"), 1)
        self.assertEqual(len(events), 2)

    def test_follower_start_failure_does_not_stop_model_or_other_account(self):
        model, failed, good = self.client("model", model=True), self.client("bad"), self.client("good")
        def fail():
            raise RuntimeError("账户登录失败")
        failed.start = fail
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(failed), self.fanout.AccountExecutionRoute(good)))
        self.addCleanup(front.stop)
        front.start()
        front.submit_targets(self.request("model"))
        self.assertEqual(model.position_manager.position("s", "main"), 1)
        self.assertEqual(len(good.backend.sent), 1)
        self.assertEqual(front.dispatch_status["bad"]["status"], "blocked")

    def test_model_generates_labeled_fills_and_is_not_a_real_account(self):
        c = self.client("model", model=True)
        events = []
        c.register_execution_event_handler(events.append)
        c.start()
        c.submit_targets(self.request("model"))
        self.assertEqual(c.position_manager.position("s", "main"), 1)
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e.metadata["execution_origin"] == "model" for e in events))
        self.assertFalse(c.report_errors)

    def test_model_rejects_future_price_without_filling(self):
        c = self.client("model", model=True)
        c.start()
        c.backend.prices.update(self.instrument, 3000, self.timestamp + 1)
        with self.assertRaises(ValueError):
            c.submit_targets(self.request("model"))
        self.assertEqual(c.position_manager.position("s", "main"), 0)
        self.assertEqual(c.position_manager.working_quantity("model", self.instrument), 0)

    def test_one_to_many_scaling_and_account_fills_do_not_pollute_strategy(self):
        model, a, b = self.client("model", model=True), self.client("a"), self.client("b")
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a),
            self.fanout.AccountExecutionRoute(b, 2)))
        self.addCleanup(front.stop)
        events = []
        front.register_execution_event_handler(events.append)
        front.start()
        front.submit_targets(self.request("model"))
        self.assertEqual(a.backend.sent[0].quantity, 1)
        self.assertEqual(b.backend.sent[0].quantity, 2)
        self.assertEqual(b.backend.sent[0].metadata["allocation_legs"][0]["quantity"], "2")
        before = len(events)
        a.backend.fill(duplicate=True)
        b.backend.fill(duplicate=True)
        self.assertEqual(len(events), before)
        self.assertEqual(model.position_manager.position("s", "main"), 1)
        self.assertEqual(a.position_manager.position("s", "main"), 1)
        self.assertEqual(b.position_manager.position("s", "main"), 2)
        self.assertEqual(front.dispatch_status["a"]["status"], "submitted")

    def test_failed_follower_is_isolated_and_not_blindly_retried(self):
        model, a, b = self.client("model", model=True), self.client("a"), self.client("b")
        calls = []
        def fail(request):
            calls.append(request)
            raise RuntimeError("第三方发送未决")
        a.submit_targets = fail
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a), self.fanout.AccountExecutionRoute(b)))
        self.addCleanup(front.stop)
        front.start()
        front.submit_targets(self.request("model"))
        front.submit_targets(self.request("model", revision=2))
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.position_manager.position("s", "main"), 1)
        self.assertEqual(front.dispatch_status["a"]["status"], "blocked")
        self.assertEqual(len(b.backend.sent), 1)  # 未成交在途量避免重复发送。

    def test_pending_dispatch_restore_blocks_account_before_resend(self):
        model, a = self.client("model", model=True), self.client("a")
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a),))
        state = front.snapshot_state()
        state["journal"] = {"a": {"status": "pending", "revision": 1, "strategy_id": "s", "ts_event": self.timestamp}}
        front.restore_state(state)
        self.addCleanup(front.stop)
        front.start()
        front.submit_targets(self.request("model"))
        self.assertFalse(a.backend.sent)
        self.assertEqual(model.position_manager.position("s", "main"), 1)

    def test_write_before_dispatch_failure_does_not_reach_account(self):
        model, a = self.client("model", model=True), self.client("a")
        def fail():
            raise OSError("checkpoint failed")
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a),), checkpoint=fail)
        self.addCleanup(front.stop)
        front.start()
        with self.assertRaises(OSError):
            front.submit_targets(self.request("model"))
        self.assertFalse(a.backend.sent)

    def test_fanout_rejects_duplicate_account_and_shared_strategy_book(self):
        model, a = self.client("model", model=True), self.client("a")
        with self.assertRaises(ValueError):
            self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a), self.fanout.AccountExecutionRoute(a)))
        a.account_id = model.account_id
        with self.assertRaises(ValueError):
            self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a),))
        a.account_id = "a"
        a.position_manager = model.position_manager
        with self.assertRaises(ValueError):
            self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(a),))

    def test_primary_callback_failure_prevents_follower_submission(self):
        model, follower = self.client("model", model=True), self.client("a")
        def fail(event):
            raise OSError("strategy checkpoint failed")
        model.register_execution_event_handler(fail)
        front = self.fanout.TargetFanoutClient(model, (self.fanout.AccountExecutionRoute(follower),))
        front.start()
        self.addCleanup(front.stop)
        with self.assertRaises(RuntimeError):
            front.submit_targets(self.request("model"))
        self.assertTrue(model.report_errors)
        self.assertFalse(follower.backend.sent)

    def test_fills_source_allows_async_checkpoint_snapshot_while_submitting(self):
        import threading
        primary, follower = self.client("source"), self.client("a")
        front = self.fanout.TargetFanoutClient(primary, (self.fanout.AccountExecutionRoute(follower),))
        front.start()
        self.addCleanup(front.stop)
        observed, threads = threading.Event(), []
        submit = primary.backend.submit_order
        def send(order):
            def callback():
                front.snapshot_state()
                observed.set()
            worker = threading.Thread(target=callback, daemon=True)
            threads.append(worker)
            worker.start()
            if not observed.wait(1):
                raise RuntimeError("源成交检查点被发送锁阻塞")
            submit(order)
        primary.backend.submit_order = send
        try:
            front.submit_targets(self.request("source"))
        finally:
            for worker in threads:
                worker.join(2)
        self.assertTrue(observed.is_set())
        self.assertEqual(len(follower.backend.sent), 1)

    def test_account_service_shares_connection_and_aggregates_multiple_strategies(self):
        account = self.client("a")
        service = self.account.AccountExecutionService(account)
        first, second = self.client("first", model=True), self.client("second", model=True)
        fronts = [self.fanout.TargetFanoutClient(c, (self.fanout.AccountExecutionRoute(service),)) for c in (first, second)]
        for front in fronts:
            self.addCleanup(front.stop)
            front.start()
        self.assertEqual(account.backend.starts, 1)
        fronts[0].submit_targets(self.request("first", strategy="one"))
        account.backend.fill(0)
        fronts[1].submit_targets(self.request("second", strategy="two", quantity=2))
        self.assertEqual(account.backend.sent[1].quantity, 2)
        account.backend.fill(1)
        self.assertEqual(account.position_manager.account_position("a", self.instrument), 3)
        self.assertEqual(account.position_manager.position("one", "main"), 1)
        self.assertEqual(account.position_manager.position("two", "main"), 2)
        fronts[0].submit_targets(self.request("first", strategy="one", quantity=-1, revision=2))
        account.backend.fill(2)
        self.assertEqual(account.position_manager.account_position("a", self.instrument), 1)
        self.assertEqual(account.position_manager.position("one", "main"), -1)
        self.assertEqual(account.position_manager.position("two", "main"), 2)
        fronts[0].stop()
        self.assertTrue(account.backend.running)
        fronts[1].stop()
        self.assertEqual(account.backend.stops, 1)

    def test_account_service_rejects_stale_and_conflicting_target_versions(self):
        client = self.client("a")
        service = self.account.AccountExecutionService(client)
        service.start()
        self.addCleanup(service.stop)
        service.submit_targets(self.request("a", revision=2))
        for req in (self.request("a"), self.request("a", revision=2, quantity=2)):
            with self.assertRaises(ValueError):
                service.submit_targets(req)
        self.assertEqual(len(client.backend.sent), 1)

    def test_account_service_checkpoint_restore_keeps_other_strategy_contributions(self):
        original = self.account.AccountExecutionService(self.client("a"))
        original.start()
        self.addCleanup(original.stop)
        original.submit_targets(self.request("a", strategy="one"))
        original.submit_targets(self.request("a", strategy="two", quantity=2))
        restored = self.account.AccountExecutionService(self.client("a"))
        restored.restore_state(original.snapshot_state())
        targets = restored.coordinator.snapshot().targets
        self.assertEqual(next(iter(targets.values())), 3)

    def test_account_service_rejects_duplicate_attribution_before_mutation_and_copies_snapshot(self):
        from dataclasses import replace
        client = self.client("a")
        service = self.account.AccountExecutionService(client)
        service.start()
        self.addCleanup(service.stop)
        request = self.request("a", quantity=2)
        row = {**request.metadata["position_attribution"][0], "quantity": "1"}
        duplicate = replace(request, metadata={"position_attribution": (row, row)})
        with self.assertRaises(ValueError):
            service.submit_targets(duplicate)
        self.assertFalse(service.snapshot_state()["sources"])
        self.assertFalse(client.backend.sent)
        service.submit_targets(request)
        snapshot = service.snapshot_state()
        snapshot["sources"]["s"]["rows"][0]["quantity"] = "999"
        self.assertEqual(service.snapshot_state()["sources"]["s"]["rows"][0]["quantity"], "2")

    def test_account_service_pending_restore_never_resends_unknown_submission(self):
        client = self.client("a")
        service = self.account.AccountExecutionService(client)
        state = service.snapshot_state()
        state["pending"] = True
        service.restore_state(state)
        service.start()
        self.addCleanup(service.stop)
        with self.assertRaisesRegex(RuntimeError, "发送未决"):
            service.submit_targets(self.request("a"))
        self.assertFalse(client.backend.sent)

    def test_account_service_allows_async_report_snapshot_while_submitting(self):
        import threading
        client = self.client("a")
        service = self.account.AccountExecutionService(client)
        service.start()
        self.addCleanup(service.stop)
        observed, threads, snapshots = threading.Event(), [], []
        submit = client.backend.submit_order
        def send(order):
            def callback():
                snapshots.append(service.snapshot_state())
                observed.set()
            worker = threading.Thread(target=callback, daemon=True)
            threads.append(worker)
            worker.start()
            if not observed.wait(1):
                raise RuntimeError("异步回报被发送状态锁阻塞")
            submit(order)
        client.backend.submit_order = send
        try:
            service.submit_targets(self.request("a"))
        finally:
            for thread in threads:
                thread.join(2)
        self.assertTrue(observed.is_set())
        self.assertTrue(snapshots[0]["pending"])
        self.assertFalse(service.snapshot_state()["pending"])

    def test_model_state_file_lock_is_per_strategy_not_per_account(self):
        import_module('tests.test_live_common')  # 初始化独立的Runtime测试包。
        module = import_module('_test_live_runtime.live.local')
        with TemporaryDirectory() as root, ExitStack() as first, ExitStack() as second, ExitStack() as conflict:
            one = module.LocalExecutionLifecycle('20261009', state_file=Path(root) / '01.json')
            two = module.LocalExecutionLifecycle('20261009', state_file=Path(root) / '04.json')
            one.prepare(first)
            two.prepare(second)
            with self.assertRaisesRegex(RuntimeError, '状态文件已有进程'):
                module.LocalExecutionLifecycle('20261009', state_file=Path(root) / '01.json').prepare(conflict)

    def test_failed_local_resume_preserves_state_file(self):
        import_module('tests.test_live_common')
        module = import_module('_test_live_runtime.live.local')
        with TemporaryDirectory() as root:
            path = Path(root) / '01.json'
            path.write_text('original')
            calls = []
            session = NS(runner=NS(accept_bars=True, stop=lambda: calls.append('stop')),
                manager=NS(save=lambda: path.write_text('overwritten')),
                client=NS(backend=NS(reports=())))
            result = module.LocalExecutionLifecycle('20261009', resume=True).shutdown(session, None)
            self.assertTrue(result['resume_state_preserved'])
            self.assertEqual(path.read_text(), 'original')
            self.assertEqual(calls, ['stop'])

    def test_ctp_model_report_preserves_observed_md_day_after_feed_stop(self):
        import_module('tests.test_live_common')
        module = import_module('_test_live_runtime.live.channels.ctp_market')
        env = {'CTP_MD_ADDRESS': 'tcp://fake:1234', 'CTP_ACCOUNT_ID': 'demo', 'CTP_PASSWORD': 'fake'}
        with patch.dict(os.environ, env, clear=True):
            lifecycle = module.CtpMarketSessionLifecycle(expected_trading_day='20261009')
        upstream = NS(latest_trading_day='20261009')
        session = NS(upstream=upstream, manager=None,
            runner=NS(accept_bars=True, stop=lambda: setattr(upstream, 'latest_trading_day', None)),
            client=NS(backend=NS(reports=())))
        lifecycle.shutdown(session, None)
        self.assertIsNone(upstream.latest_trading_day)
        self.assertEqual(lifecycle.snapshot(session, None)['observed_md_trading_day'], '20261009')

    def test_ctp_model_report_never_substitutes_expected_day_when_md_is_missing(self):
        import_module('tests.test_live_common')
        module = import_module('_test_live_runtime.live.channels.ctp_market')
        env = {'CTP_MD_ADDRESS': 'tcp://fake:1234', 'CTP_ACCOUNT_ID': 'demo', 'CTP_PASSWORD': 'fake'}
        with patch.dict(os.environ, env, clear=True):
            lifecycle = module.CtpMarketSessionLifecycle(expected_trading_day='20261009')
        session = NS(upstream=NS(latest_trading_day=None), manager=None,
            runner=NS(accept_bars=False, stop=lambda: None), client=NS(backend=NS(reports=())))
        lifecycle.shutdown(session, None)
        self.assertIsNone(lifecycle.snapshot(session, None)['observed_md_trading_day'])

    def test_policy_defaults_and_mode_switch_recovery_rejected(self):
        strict = self.policy.ExecutionPolicy()
        self.assertEqual(strict.snapshot_state()["account_reconciliation"], "strict")
        with self.assertRaises(ValueError):
            strict.restore_state(self.policy.ExecutionPolicy("off").snapshot_state())
        with self.assertRaises(ValueError):
            self.policy.ExecutionPolicy("strict", "model")
        self.assertEqual(self.policy.execution_policy_from_args(NS(mode="model")), self.policy.ExecutionPolicy("off", "model"))

    def test_account_observation_reporting_never_labels_model_book_as_real(self):
        client = self.client("model", model=True)
        client.start()
        client.submit_targets(self.request("model"))
        self.assertIsNone(self.policy.observed_account_positions(client, (self.instrument,)))
        observed = self.client("observe", mode="observe", real=-20)
        observed.start()
        self.assertEqual(self.policy.observed_account_positions(observed, (self.instrument,)), {"RB.SHFE": -20})


if __name__ == "__main__":
    unittest.main()
