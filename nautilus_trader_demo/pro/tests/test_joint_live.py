"""GAP-10无网络验收：双端分腿、闭闸、权威恢复、检查点与账户租约。"""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.assembly import StrategyBindings
from bomber.framework.trader.contracts import ExecutionRoute
from bomber.framework.trader.execution.builders import ExecutionComponents
from bomber.framework.trader.execution.contracts import AccountPositionSnapshot, ExecutionReport, ExecutionReportType
from bomber.framework.trader.execution.events import AccountStateEvent, ActiveOrder, ActiveOrderSnapshot, CurrencyBalance
from bomber.framework.trader.execution.live.binance_account import BinanceHttpAccountReader
from bomber.framework.trader.execution.live.controlled import ControlledLiveExecutionClient
from bomber.framework.trader.execution.planner import NetTargetOrderPlanner
from bomber.framework.trader.execution.risk import MarketReferencePriceStore, PreTradeRiskManager, RiskLimits
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager
from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.runtime.joint import JointExecutionBinding, JointLiveRunner, assemble_joint_strategy
from bomber.framework.trader.runtime.ownership import AccountWriterIdentity, account_writer_lease
from bomber.framework.trader.template import StrategyTemplate

CTP = InstrumentId.from_str("cu2611.SHFE")
BN = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")


class Backend:
    def __init__(self, name, instrument):
        self.backend_id, self.account_id, self.instrument = name, f"account-{name}", instrument
        self.orders, self.handlers, self.rows = [], [], {}
        self.positions, self.extras = {}, ()
        self.revision = 0
        self.started = False
        self.fail_start = self.fail_send = self.fail_query = False
        self.reject_on_send = False
        self.stop_calls = 0

    def register_report_handler(self, handler):
        self.handlers.append(handler)

    def start(self):
        if self.fail_start:
            raise RuntimeError("startup failure")
        self.started = True

    def stop(self):
        self.started = False
        self.stop_calls += 1

    def reconcile(self):
        self.revision += 1
        return AccountPositionSnapshot(self.backend_id, self.revision, time.time_ns(), self.positions)

    def reconcile_account_state(self):
        if self.fail_query:
            raise RuntimeError("HTTP query failed")
        self.revision += 1
        return AccountStateEvent(self.backend_id, self.account_id, self.revision, time.time_ns(),
            {"USD": CurrencyBalance("USD", total=1000, available=1000)})

    def reconcile_active_orders(self):
        self.revision += 1
        orders = []
        for index, row in self.rows.items():
            order = self.orders[index]
            if row["terminal"]:
                continue
            orders.append(ActiveOrder(f"{self.backend_id}-{index}", str(order.instrument_id),
                order.side.value, order.quantity, row["filled"], order.quantity - row["filled"]))
        return ActiveOrderSnapshot(self.backend_id, self.account_id, self.revision, time.time_ns(),
            tuple(orders) + self.extras)

    def submit_order(self, order):
        if self.fail_send:
            raise RuntimeError("definitely unsent")
        self.orders.append(order)
        index = len(self.orders) - 1
        self.rows[index] = {"filled": Decimal(0), "terminal": False, "sequence": 0}
        self.emit(index, ExecutionReportType.REJECTED if self.reject_on_send else ExecutionReportType.ACCEPTED)

    def emit(self, index, status, quantity=0):
        order, row = self.orders[index], self.rows[index]
        row["sequence"] += 1
        quantity = Decimal(quantity)
        row["filled"] += quantity
        row["terminal"] = status in {ExecutionReportType.FILLED, ExecutionReportType.CANCELED, ExecutionReportType.REJECTED}
        if quantity:
            signed = quantity if order.side.value == "BUY" else -quantity
            self.positions[order.instrument_id] = self.positions.get(order.instrument_id, Decimal(0)) + signed
        report = ExecutionReport(self.backend_id, f"{self.backend_id}-{index}", order.instrument_id,
            status, time.time_ns(), filled_quantity=quantity, fill_price=100 if quantity else None,
            order_side=order.side, order_quantity=order.quantity,
            reason="rejected" if status is ExecutionReportType.REJECTED else None,
            report_id=f"{self.backend_id}-{index}:{row['sequence']}", sequence=row["sequence"],
            metadata={**order.metadata, "strategy_id": order.strategy_id,
                "trade_id": f"trade-{index}-{row['sequence']}"})
        for handler in self.handlers:
            handler(report)
        return report

    def cancel_strategy(self, strategy_id):
        for index, row in tuple(self.rows.items()):
            if not row["terminal"] and self.orders[index].strategy_id == strategy_id:
                self.emit(index, ExecutionReportType.CANCELED)


class Probe(StrategyTemplate):
    def __init__(self):
        super().__init__("joint-probe")
        self.fills = []

    def on_fill(self, event):
        key = event.metadata["attributed_target_key"]
        self.fills.append((event.identity.client_id, key, self.position(key), self.account_position(key)))


class JointTests(unittest.TestCase):
    def create(self, *, auto_start=True, positions=None):
        positions = positions or PositionManager()
        prices = MarketReferencePriceStore()
        items, backends, clients = [], [], []
        ready = {"a-ctp": True, "b-binance": True}
        for name, instrument in (("a-ctp", CTP), ("b-binance", BN)):
            backend = Backend(name, instrument)
            prices.update(instrument, 100, time.time_ns())
            client = ControlledLiveExecutionClient(name, NetTargetOrderPlanner(positions), backend, positions,
                PreTradeRiskManager(name, positions, prices, default_limits=RiskLimits(max_order_quantity=3)),
                account_id=backend.account_id, demo_environment_check=lambda: True)
            execution = ExecutionComponents(client, positions, prices, backend=backend)
            items.append(JointExecutionBinding(execution, AccountWriterIdentity(name, "demo", backend.account_id),
                lambda name=name: ready[name]))
            backends.append(backend)
            clients.append(client)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runner = JointLiveRunner(position_manager=positions, bindings=items,
            lease_factory=lambda identities: account_writer_lease(identities, lock_directory=Path(temporary.name)))
        strategy = Probe()
        assemble_joint_strategy(runner, strategy, feeds={}, bindings=StrategyBindings(data=(), execution=(
            ExecutionRoute("ctp", "a-ctp", CTP), ExecutionRoute("bn", "b-binance", BN))))
        self.addCleanup(runner.stop)
        if auto_start:
            runner.start()
            runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
        return runner, strategy, backends, clients, ready

    def test_two_legs_callbacks_and_duplicate_fill_are_isolated(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 1, "bn": -1}, time.time_ns())
        for backend in backends:
            report = backend.emit(0, ExecutionReportType.FILLED, 1)
            backend.handlers[0](report)
        self.assertEqual(strategy.fills, [("a-ctp", "ctp", 1, 1), ("b-binance", "bn", -1, -1)])
        self.assertFalse(any(client.report_errors for client in clients))
        for backend in backends:
            self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, backend.instrument), 0)

    def test_endpoint_not_ready_rejects_before_target_store(self):
        runner, strategy, backends, _, ready = self.create()
        ready["b-binance"] = False
        with self.assertRaises(RuntimeError):
            strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        self.assertIsNone(runner.target_store.get(strategy.strategy_id))
        self.assertFalse(any(backend.orders for backend in backends))

    def test_default_start_is_disarmed_and_wrong_confirmation_fails(self):
        runner, strategy, backends, clients, _ = self.create(auto_start=False)
        runner.start()
        with self.assertRaises(PermissionError):
            runner.arm("AUTHORIZE_DEMO_ORDERS")
        with self.assertRaises(RuntimeError):
            strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        self.assertFalse(any(client.is_armed for client in clients))

    def test_second_send_failure_preserves_first_order_and_retained_target(self):
        runner, strategy, backends, _, _ = self.create()
        backends[1].fail_send = True
        with self.assertRaises(RuntimeError):
            strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        self.assertEqual([len(b.orders) for b in backends], [1, 0])
        self.assertEqual(runner.position_manager.working_quantity("a-ctp", CTP), 1)
        self.assertEqual(runner.target_store.get(strategy.strategy_id).targets["bn"], 1)
        self.assertTrue(runner.failure)
        with self.assertRaises(RuntimeError):
            runner.retry_remaining(strategy.strategy_id, now_ns=time.time_ns())

    def test_synchronous_first_rejection_blocks_second_endpoint(self):
        runner, strategy, backends, _, _ = self.create()
        backends[0].reject_on_send = True
        with self.assertRaises(RuntimeError):
            strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        self.assertEqual([len(b.orders) for b in backends], [1, 0])
        self.assertIn("order_rejected", runner.failure)

    def test_partial_fill_disconnect_recovery_replans_only_remaining(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 2, "bn": -1}, time.time_ns())
        backends[0].emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        clients[0].mark_disconnected("test")
        runner.trip("ctp_disconnected")
        runner.recover(operator="test", reason="full snapshots verified")
        self.assertFalse(any(client.is_armed for client in clients))
        self.assertEqual([len(b.orders) for b in backends], [1, 1])
        runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
        runner.retry_remaining(strategy.strategy_id, now_ns=time.time_ns())
        self.assertEqual([len(b.orders) for b in backends], [1, 1])  # 在途已占全部剩余量。
        backends[0].emit(0, ExecutionReportType.CANCELED)
        runner.refresh_authority()
        runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
        runner.retry_remaining(strategy.strategy_id, now_ns=time.time_ns())
        self.assertEqual([len(b.orders) for b in backends], [2, 1])
        self.assertEqual(backends[0].orders[1].quantity, 1)

    def test_missing_original_order_keeps_recovery_closed(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        clients[0].mark_disconnected("test")
        runner.trip("disconnect")
        backends[0].rows[0]["terminal"] = True  # 柜台消失但本地尚无终态回报。
        with self.assertRaises(RuntimeError):
            runner.recover(operator="test", reason="cannot guess missing order")
        self.assertTrue(runner.failure)
        self.assertTrue(runner.position_manager.is_recovery_required("a-ctp"))

    def test_unknown_external_order_blocks_start_and_releases_both_clients(self):
        runner, _, backends, _, _ = self.create(auto_start=False)
        backends[1].extras = (ActiveOrder("external", str(BN), "BUY", 1, 0, 1),)
        with self.assertRaises(RuntimeError):
            runner.start()
        self.assertFalse(any(backend.started for backend in backends))
        self.assertFalse(runner._lease_held)

    def test_second_start_failure_stops_first_endpoint(self):
        runner, _, backends, _, _ = self.create(auto_start=False)
        backends[1].fail_start = True
        with self.assertRaises(RuntimeError):
            runner.start()
        self.assertFalse(any(backend.started for backend in backends))
        self.assertGreater(backends[0].stop_calls, 0)
        self.assertFalse(runner._lease_held)

    def test_funds_query_failure_halts_every_endpoint(self):
        runner, _, backends, clients, _ = self.create()
        backends[1].fail_query = True
        with self.assertRaises(RuntimeError):
            runner.refresh_authority()
        self.assertTrue(runner.failure)
        self.assertFalse(any(client.is_armed for client in clients))

    def test_query_empty_position_cannot_erase_owned_position_silently(self):
        runner, strategy, backends, _, _ = self.create()
        strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        backends[0].emit(0, ExecutionReportType.FILLED, 1)
        backends[0].positions.clear()
        with self.assertRaises(RuntimeError):
            runner.refresh_authority()
        self.assertTrue(runner.failure)

    def test_cross_client_same_instrument_does_not_net(self):
        runner, strategy, backends, _, _ = self.create(auto_start=False)
        registration = runner._registrations[strategy.strategy_id]
        runner._registrations[strategy.strategy_id] = replace(registration, execution_routes={
            "ctp": ExecutionRoute("ctp", "a-ctp", CTP), "bn": ExecutionRoute("bn", "b-binance", CTP)})
        runner.start()
        runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
        strategy.set_targets({"ctp": 1, "bn": -1}, time.time_ns())
        self.assertEqual([b.orders[0].quantity for b in backends], [1, 1])
        self.assertEqual([b.orders[0].side.value for b in backends], ["BUY", "SELL"])

    def test_checkpoint_restores_both_clients_but_never_authorizes_or_replays(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 2, "bn": -1}, time.time_ns())
        backends[0].emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        with TemporaryDirectory() as root:
            manager = RuntimeStateManager(JsonStateStore(Path(root) / "state.json"), runner.target_store,
                runner.portfolio_coordinator, runner.position_manager,
                order_machines={c.client_id: c.order_state_machine for c in clients})
            manager.save()
            positions = PositionManager()
            restarted, _, new_backends, new_clients, _ = self.create(auto_start=False, positions=positions)
            restored = RuntimeStateManager(JsonStateStore(Path(root) / "state.json"), restarted.target_store,
                restarted.portfolio_coordinator, positions,
                order_machines={c.client_id: c.order_state_machine for c in new_clients})
            restored.restore()
            self.assertEqual(positions.position(strategy.strategy_id, "ctp"), 1)
            with self.assertRaises(RuntimeError):
                restarted.start()  # 权威订单为空，不能复用本地状态直接发送。
            self.assertFalse(any(b.orders for b in new_backends))
            self.assertFalse(any(c.is_armed for c in new_clients))

    def test_cancel_confirmation_preserves_filled_position(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 2, "bn": -1}, time.time_ns())
        backends[0].emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        runner.cancel_and_confirm(timeout_seconds=1)
        self.assertEqual(strategy.position("ctp"), 1)
        self.assertFalse(any(runner.position_manager.snapshot().working_quantities.values()))
        self.assertFalse(any(c.is_armed for c in clients))

    def test_restored_known_orders_reconcile_without_automatic_resend(self):
        runner, strategy, backends, clients, _ = self.create()
        strategy.set_targets({"ctp": 2, "bn": -1}, time.time_ns())
        backends[0].emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        with TemporaryDirectory() as root:
            repository = JsonStateStore(Path(root) / "state.json")
            RuntimeStateManager(repository, runner.target_store, runner.portfolio_coordinator,
                runner.position_manager, order_machines={c.client_id: c.order_state_machine for c in clients}).save()
            restarted, _, new_backends, new_clients, _ = self.create(auto_start=False)
            for previous, current in zip(backends, new_backends):
                current.orders = list(previous.orders)
                current.rows = {key: dict(value) for key, value in previous.rows.items()}
                current.positions = dict(previous.positions)
            RuntimeStateManager(repository, restarted.target_store, restarted.portfolio_coordinator,
                restarted.position_manager,
                order_machines={c.client_id: c.order_state_machine for c in new_clients}).restore()
            restarted.start()
            self.assertFalse(any(c.is_armed for c in new_clients))
            restarted.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
            restarted.retry_remaining(strategy.strategy_id, now_ns=time.time_ns())
            self.assertEqual([len(b.orders) for b in new_backends], [1, 1])

    def test_live_node_thread_must_end_before_disposal(self):
        from types import SimpleNamespace
        from bomber.framework.trader.execution.live.driver import NautilusTradingNodeDriver
        disposed = []
        driver = NautilusTradingNodeDriver("test-node", SimpleNamespace(
            stop=lambda: None, dispose=lambda: disposed.append(True)))
        driver._started = True
        driver._thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
        with self.assertRaises(RuntimeError):
            driver.stop()
        self.assertTrue(driver._started)
        self.assertFalse(disposed)
        driver._thread = SimpleNamespace(is_alive=lambda: False)
        driver.stop()
        self.assertEqual(disposed, [True])

    def test_same_physical_account_under_different_client_ids_rejected(self):
        runner, _, _, _, _ = self.create(auto_start=False)
        duplicate = replace(runner.bindings[1], identity=runner.bindings[0].identity)
        with self.assertRaises(ValueError):
            JointLiveRunner(position_manager=runner.position_manager, bindings=(runner.bindings[0], duplicate))

    def test_session_order_cap_remains_after_cancel_and_recovery(self):
        runner, strategy, backends, _, _ = self.create()
        runner.max_orders_per_client = 1
        strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        runner.cancel_and_confirm()
        runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
        with self.assertRaises(RuntimeError):
            runner.retry_remaining(strategy.strategy_id, now_ns=time.time_ns())
        self.assertEqual([len(b.orders) for b in backends], [1, 1])
        self.assertIn("session_order_cap", runner.failure)

    def test_cleanup_error_preserves_lease_until_stop_is_confirmed(self):
        runner, _, backends, _, _ = self.create()
        normal_stop = backends[0].stop
        def failed_stop():
            raise RuntimeError("stop failed")
        backends[0].stop = failed_stop
        with self.assertRaises(RuntimeError):
            runner.stop()
        self.assertTrue(runner._lease_held)
        backends[0].stop = normal_stop
        runner.stop()
        self.assertFalse(runner._lease_held)
        self.assertFalse(backends[0].started)

    def test_account_lease_conflict_and_release(self):
        identity = AccountWriterIdentity("binance", "DEMO", "same-owner")
        with TemporaryDirectory() as root:
            with account_writer_lease((identity,), lock_directory=Path(root)):
                with self.assertRaises(RuntimeError):
                    with account_writer_lease((identity,), lock_directory=Path(root)):
                        self.fail("duplicate owner acquired lease")
            with account_writer_lease((identity,), lock_directory=Path(root)):
                pass

    def test_http_positions_reject_hedge_missing_and_duplicates(self):
        row = {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0.001"}
        self.assertEqual(BinanceHttpAccountReader.parse_positions([row]), {BN: Decimal("0.001")})
        for rows in ([{**row, "positionSide": "LONG"}], [row, row], [{"symbol": "BTCUSDT"}], None,
                [{**row, "positionAmt": "NaN"}]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                BinanceHttpAccountReader.parse_positions(rows)

    def test_http_funds_require_complete_balances_and_trading_permission(self):
        from types import SimpleNamespace
        asset = {"asset": "USDT", "walletBalance": "1000", "marginBalance": "1001",
            "availableBalance": "900", "initialMargin": "100"}
        raw = {"canTrade": True, "assets": [asset]}
        reader = BinanceHttpAccountReader("bn", "owner", lambda: None,
            SimpleNamespace(account=lambda: None))
        reader._read = lambda operation: raw
        state = reader.account()
        self.assertEqual(state.balances["USDT"].available, 900)
        self.assertEqual(state.revision, 1)
        for invalid in ({"canTrade": False, "assets": [asset]}, {"canTrade": True, "assets": []},
                {"canTrade": True, "assets": [{"asset": "USDT"}]}):
            raw = invalid
            with self.assertRaises(ValueError if invalid.get("canTrade") else RuntimeError):
                reader.account()
        self.assertEqual(reader.revision, 1)

    def test_joint_cli_requires_both_authorizations_and_new_checkpoint(self):
        from contextlib import redirect_stderr
        import io
        from scripts.integration.ctp_binance.run import parse_args
        base = ["--connect", "--product", "CU", "--expected-trading-day", "20261009",
            "--expected-source-day", "20261008", "--ctp-target", "1", "--bn-target", "0.001",
            "--ctp-max-notional", "600000", "--bn-max-notional", "200"]
        self.assertEqual(parse_args(base).mode, "readonly")
        self.assertEqual(parse_args(base).reference_source, "dolphindb")
        with self.assertRaises(SystemExit):
            parse_args([*base, "--reference-source", "file"])
        self.assertTrue(parse_args([*base, "--reference-source", "file",
            "--allow-file-reference-test"]).allow_file_reference_test)
        with TemporaryDirectory() as root:
            path = Path(root) / "state.json"
            orders = [*base, "--mode", "orders", "--state-file", str(path)]
            with redirect_stderr(io.StringIO()):
                for flags in ([], ["--enable-orders", "--confirm-simnow"], ["--enable-orders", "--confirm-demo"]):
                    with self.assertRaises(SystemExit):
                        parse_args([*orders, *flags])
            authorized = [*orders, "--enable-orders", "--confirm-simnow", "--confirm-demo"]
            self.assertEqual(parse_args(authorized).mode, "orders")
            path.write_text("preserve existing checkpoint", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(authorized)

    def test_stale_funds_unsupported_policy_and_expired_target_never_send(self):
        runner, strategy, backends, clients, _ = self.create()
        for kwargs in ({"execution_policy": "ATOMIC"}, {"deadline_ns": 1}):
            with self.assertRaises((ValueError, RuntimeError)):
                strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns(), **kwargs)
        clients[1]._account_state = replace(clients[1].account_state, ts_event=1)
        with self.assertRaises(RuntimeError):
            strategy.set_targets({"ctp": 1, "bn": 1}, time.time_ns())
        self.assertFalse(any(b.orders for b in backends))


if __name__ == "__main__":
    unittest.main(verbosity=2)
