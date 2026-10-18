"""持续会话的无网络验证；远程执行，本机只核对源码。"""
from datetime import datetime
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch, Mock
import json
import unittest

from bomber.framework.trader import TargetStore, PortfolioCoordinator, PositionManager
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager, StatePersistenceError
from bomber.framework.trader.runtime.trading_sessions import CopperSessions, SHANGHAI
from bomber.framework.trader.execution.ctp.native_driver import CtpNativeTraderDriver, CtpDriverCheckpoint
from tests.run_p4_ctp_recovery import RecoveryTransport, _active

ema = import_module("demos.01_main_ema.strategy")
checkpoint = import_module("demos.01_main_ema.checkpoint")
continuous = import_module("demos.01_main_ema.continuous")
ROOT = Path(__file__).resolve().parents[1]
MINUTE = 60_000_000_000


def ns(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=SHANGHAI).timestamp() * 1e9)


def make_checkpoint():
    strategy = ema.MainEmaStrategy("main-ema-simnow", None, ema.MainEmaConfig("CU", "SHFE"))
    return checkpoint.EmaCheckpoint(strategy, "cu2611.SHFE")


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.calendar = CopperSessions(ROOT / "demos/01_main_ema/shfe_cu_2026.json")
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def window(self, text):
        return self.calendar.window(datetime.fromisoformat(text).replace(tzinfo=SHANGHAI))

    def test_intraday_pause_and_open_boundaries(self):
        self.assertIsNotNone(self.window("2026-10-09T10:14:59"))
        self.assertIsNone(self.window("2026-10-09T10:15:00"))
        self.assertIsNotNone(self.window("2026-10-09T10:30:00"))
        self.assertIsNone(self.window("2026-10-09T11:30:00"))
        self.assertIsNone(self.window("2026-10-09T15:00:00"))

    def test_friday_night_belongs_to_monday_and_saturday_tail(self):
        self.assertEqual(self.window("2026-10-09T21:00:00").trading_day, "20261012")
        self.assertEqual(self.window("2026-10-10T00:59:59").trading_day, "20261012")
        self.assertIsNone(self.window("2026-10-10T01:00:00"))
        self.assertIsNone(self.window("2026-10-11T21:00:00"))

    def test_holidays_and_previous_source_day(self):
        self.assertIsNone(self.window("2026-09-30T21:00:00"))
        self.assertIsNone(self.window("2026-10-07T09:00:00"))
        self.assertEqual(self.calendar.previous_trading_day("20261008"), "20260930")
        self.assertEqual(self.calendar.previous_trading_day("20260105"), "20251231")

    def test_unknown_calendar_and_naive_clock_fail(self):
        with self.assertRaises(RuntimeError):
            self.window("2027-01-04T09:00:00")
        with self.assertRaises(ValueError):
            self.calendar.window(datetime(2026, 10, 9, 9))

    def test_missing_minutes_excludes_break(self):
        after = ns("2026-10-09T10:15:00") - 1
        self.assertEqual(list(self.calendar.missing_minutes(after, ns("2026-10-09T10:31:00"))),
            [ns("2026-10-09T10:31:00") - 1])

    def test_close_drain_keeps_unfinished_minute_and_emits_no_callbacks(self):
        from tests.run_main_ema_live import LiveTests, ManualMd, BASE
        from bomber.framework.market.stream.aggregation import TradeTickBarFeed
        from bomber.framework.market.basic.base import DataType, make_trade_tick
        fixture = LiveTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        references = fixture.reference()
        feed = TradeTickBarFeed("closed", ManualMd())
        meta = references.instrument_meta()
        feed.register_instrument(meta)
        feed.subscribe(references.instrument_id, DataType.BAR, "1-MINUTE")
        feed._on_trade_tick(make_trade_tick(instrument_id=references.instrument_id,
            price=3100, size=1, ts_event=BASE + 1000, ts_init=BASE + 1000, meta=meta, trade_id="last"))
        self.assertEqual(feed.drain_completed(BASE + MINUTE - 1), ())
        bars = feed.drain_completed(BASE + MINUTE)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].ts_event, BASE + MINUTE - 1)
        self.assertEqual(bars[0].close.as_decimal(), Decimal(3100))
        self.assertEqual(feed.drain_completed(BASE + 2 * MINUTE), ())

    def test_indicator_restart_matches_uninterrupted_and_partial_warmup(self):
        for count in (2, 5, 8):
            original = make_checkpoint()
            for index in range(count):
                original.warm_closed(index * MINUTE, 110000 + index * 10)
            resumed = make_checkpoint()
            resumed.restore_state(original.snapshot_state())
            for index in range(count, count + 7):
                original.warm_closed(index * MINUTE, 110000 - index * 20)
                resumed.warm_closed(index * MINUTE, 110000 - index * 20)
                self.assertEqual(resumed.strategy.fast.value, original.strategy.fast.value)
                self.assertEqual(resumed.strategy.slow.value, original.strategy.slow.value)
                self.assertEqual(resumed.strategy.slow.initialized, original.strategy.slow.initialized)
                self.assertEqual(resumed.strategy._revision, 0)
                self.assertIsNone(resumed.strategy.last_target)

    def test_restore_rejects_wrong_contract_and_nonfinite(self):
        saved = make_checkpoint().snapshot_state()
        for replacement in ({"instrument": "cu2612.SHFE"}, {"fast": "nan"}, {"bars_used": -1}):
            with self.assertRaises(ValueError):
                make_checkpoint().restore_state({**saved, **replacement})

    def test_restore_preserves_saved_double_without_initial_input_rounding(self):
        original = make_checkpoint()
        original.warm_closed(MINUTE, "11235.989822638036")
        saved = original.snapshot_state()
        saved.update(fast="11235.005399922291", slow="11235.989822638036")
        restored = make_checkpoint()
        restored.restore_state(saved)
        self.assertEqual(restored.snapshot_state(), saved)

    def test_empty_indicator_restore_matches_engine_from_first_real_input(self):
        original, restored = make_checkpoint(), make_checkpoint()
        restored.restore_state(original.snapshot_state())
        for index, price in enumerate(("11303.1461600", "11333.1519346", "11324.1210704", "11235.989822638036")):
            original.warm_closed(index * MINUTE, price)
            restored.warm_closed(index * MINUTE, price)
            self.assertEqual(restored.snapshot_state(), original.snapshot_state())

    def test_indicator_revision_must_match_published_target(self):
        component = make_checkpoint()
        component.strategy._revision = 1
        component.strategy.last_target = Decimal(-1)
        with self.assertRaisesRegex(ValueError, "缺少对应"):
            component.validate_targets(TargetStore())

    def test_component_checkpoint_saved_in_same_generation(self):
        component = make_checkpoint()
        manager = RuntimeStateManager(JsonStateStore(self.root / "state.json"), TargetStore(),
            PortfolioCoordinator(), PositionManager(), state_components={"ema": component})
        component.warm_closed(1, 110000)
        manager.save()
        component.warm_closed(2, 110100)
        manager.restore()
        self.assertEqual(component.strategy.bars_used, 1)
        self.assertEqual(component.strategy.fast.value, 110000)
        self.assertEqual(manager.generation, 1)

    def test_legacy_component_cannot_silently_restart_empty(self):
        store = JsonStateStore(self.root / "old.json")
        RuntimeStateManager(store, TargetStore(), PortfolioCoordinator(), PositionManager()).save()
        manager = RuntimeStateManager(store, TargetStore(), PortfolioCoordinator(), PositionManager(),
            state_components={"ema": make_checkpoint()})
        with self.assertRaises(StatePersistenceError):
            manager.restore()

    def test_restore_error_rolls_back_previously_restored_component(self):
        class Broken:
            value = 0
            def snapshot_state(self):
                return {"value": self.value}
            def restore_state(self, state):
                if state["value"] == 1:
                    raise ValueError("injected component failure")
                self.value = state["value"]
        component, broken = make_checkpoint(), Broken()
        manager = RuntimeStateManager(JsonStateStore(self.root / "rollback.json"), TargetStore(),
            PortfolioCoordinator(), PositionManager(), state_components={"ema": component, "broken": broken})
        component.warm_closed(1, 110000)
        broken.value = 1
        manager.save()
        broken.value = 0
        component.warm_closed(2, 110100)
        with self.assertRaises(ValueError):
            manager.restore()
        self.assertEqual(component.strategy.bars_used, 2)
        self.assertEqual(broken.value, 0)

    def test_gap_blocks_without_mutating_indicator_or_emitting_target(self):
        component = make_checkpoint()
        component.warm_closed(ns("2026-10-09T09:01:00") - 1, 110000)
        before = component.snapshot_state()
        with self.assertRaisesRegex(RuntimeError, "缺少2根"):
            continuous.recover_minutes(component, self.calendar, ns("2026-10-09T09:03:00"), None)
        self.assertEqual(component.snapshot_state(), before)

    def test_gap_replay_is_indicator_only_and_validates_all_rows_first(self):
        component = make_checkpoint()
        component.warm_closed(ns("2026-10-09T09:01:00") - 1, 110000)
        path = self.root / "minutes.jsonl"
        rows = [{"instrument_id": "cu2611.SHFE", "ts_event": ns(f"2026-10-09T09:0{i}:00") - 1,
                 "adjusted_close": "110100"} for i in (2, 3)]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        continuous.recover_minutes(component, self.calendar, ns("2026-10-09T09:03:00"), path)
        self.assertEqual(component.strategy.bars_used, 3)
        self.assertEqual(component.strategy._revision, 0)
        self.assertIsNone(component.strategy.last_target)
        self.assertEqual(component.strategy.fills_received, 0)

    def test_daily_order_budget_includes_terminal_orders(self):
        payload = {"orders": {"client": [{"client_order_id": "CTP-9999-demo-20261009-1", "status": "FILLED"},
            {"client_order_id": "CTP-9999-demo-20261012-2", "status": "CANCELED"}]}}
        self.assertEqual(continuous.orders_for_day(payload, "20261009"), 1)
        self.assertEqual(continuous.orders_for_day(payload, "20261012"), 1)

    def test_cross_day_empty_orders_requires_current_authoritative_query(self):
        for day, active, accepted in (("20260921", (), True), ("20260923", (), False),
                                     ("20260921", (_active(),), False)):
            driver = CtpNativeTraderDriver("ctp-demo", "demo-account", RecoveryTransport(active),
                disconnect_handler=lambda reason: None)
            driver.stage_recovery(CtpDriverCheckpoint("ctp-demo", "demo-account", "9999", "demo", day, 8, ()))
            driver.start(lambda report: None)
            try:
                if accepted:
                    driver.reconcile_active_orders()
                    self.assertEqual(driver.checkpoint().trading_day, "20260922")
                else:
                    with self.assertRaises(RuntimeError):
                        driver.reconcile_active_orders()
            finally:
                driver.stop()

    def test_shutdown_bar_failure_still_runs_cancel_query_and_release(self):
        lifecycle = object.__new__(continuous.ContinuousLifecycle)
        lifecycle.window = SimpleNamespace(end=datetime(2026, 1, 1, tzinfo=SHANGHAI))
        session = SimpleNamespace(runner=SimpleNamespace(failure=None, accept_bars=True),
            client=SimpleNamespace(disarm=Mock()), bar_feed=SimpleNamespace(
                drain_completed=Mock(side_effect=RuntimeError("close bar failed"))))
        with patch.object(continuous.CtpSessionLifecycle, "shutdown",
                return_value={"cleanup_errors": [], "final_active_orders": 0}) as cleanup:
            result = lifecycle.shutdown(session, None)
        cleanup.assert_called_once_with(session, None)
        self.assertEqual(result["cleanup_errors"], ["close bar failed"])

    def test_supervisor_waits_through_rest_and_stops_after_failed_worker(self):
        state = self.root / "service.json"
        moments = [datetime(2026, 10, 9, 10, 20, tzinfo=SHANGHAI)]
        class Stop:
            def is_set(self):
                return False
            def wait(self, seconds):
                moments[0] = datetime(2026, 10, 9, 10, 30, tzinfo=SHANGHAI)
        args = SimpleNamespace(trading_calendar=ROOT / "demos/01_main_ema/shfe_cu_2026.json",
            state_file=state, max_session_orders=4)
        factory = Mock(return_value=SimpleNamespace(run=lambda: {"status": "failed"}))
        live = import_module("demos.01_main_ema.run_live")
        with patch.object(continuous, "account_lock", return_value=nullcontext()), \
                patch.object(live, "required", return_value="demo"):
            with self.assertRaisesRegex(RuntimeError, "不自动重启"):
                continuous.run_continuous(args, factory, clock=lambda: moments[0], stop_event=Stop())
        self.assertEqual(factory.call_count, 1)
        worker = factory.call_args.args[0]
        self.assertEqual(worker.expected_trading_day, "20261009")
        self.assertEqual(worker.expected_source_day, "20261008")
        self.assertFalse(worker.resume)


if __name__ == "__main__":
    unittest.main(verbosity=2)
