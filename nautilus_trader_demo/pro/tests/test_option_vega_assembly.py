"""在目标原生环境验证真实Runner/归属/持久化，行情和执行仍为无网络夹具。"""
from contextlib import ExitStack
from datetime import date, timedelta
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta, MarketDataFeed, make_bar
from bomber.framework.dataprep.contracts import InstrumentSpec
from bomber.framework.market.frames import MINUTE_NS
from bomber.framework.trader.runtime.live.contracts import LiveRunContext
from bomber.framework.trader.contracts import ExecutionRequest
from bomber.framework.trader.execution.contracts import OrderSide, PositionEffect
from bomber.framework.trader.execution.ctp import CtpPositionLedger
from bomber.framework.trader.execution.ctp.planner import CtpClosePlanner
runtime = import_module("demos.08_option_vega.live_runtime")
recovery = import_module("demos.08_option_vega.recovery")
signal = import_module("demos.08_option_vega.vega_signal")
from tests.test_demo_option_vega import ns


class Feed(MarketDataFeed):
    def _on_subscription_added(self, request):
        pass
    def _on_subscription_removed(self, request):
        pass
    def connect(self):
        self._is_connected = True
    def disconnect(self):
        self._is_connected = False


class ReferenceService:
    valuation_day = date(2026, 8, 13)
    future_product = "IM"
    fingerprint = "fixture-options"

    def __init__(self):
        option = InstrumentSpec("option", "MO2609-C-6600", "MO", "CFFEX", "CNY",
            date(2026, 6, 1), date(2026, 9, 18), Decimal(".2"), Decimal(100),
            date(2026, 9, 18), "C", Decimal(6600), "000852", "202609")
        self.data = NS(options={option.symbol: option}, fingerprint=self.fingerprint,
            futures={"IM2609": dict(contMultNum=Decimal(200), minChgPriceNum=Decimal(".2"),
                listDate=date(2026, 6, 1), lastTradeDate=date(2026, 9, 18))})
    def snapshot(self, now=None):
        return self.data
    def refresh(self):
        return self.data
    def manifest(self):
        return {"fixture": True, "fingerprint": self.fingerprint}


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.now = ns("2026-10-11 18:00")
        self.sessions = []
        self.addCleanup(lambda: [session.runner.stop() for session in self.sessions])

    def test_cli_separates_market_time_and_execution_authorization(self):
        from contextlib import redirect_stderr
        from io import StringIO
        entry = import_module("demos.08_option_vega.run_live")
        common = ["--connect", "--expected-trading-day", "20261009",
            "--calendar-id", "china.cffex", "--simnow-environment", "replay",
            "--replay-md-trading-day", "20260813"]
        recording = entry.parse_args(common)
        self.assertEqual(recording.mode, "recording")
        self.assertEqual(recording.valuation_day, "20260813")
        self.assertEqual(recording.execution_policy.account_reconciliation.value, "off")
        self.assertEqual(recording.calendar_id, "china.cffex")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(common + ["--calendar", "legacy.csv"])
        model = entry.parse_args(common + ["--mode", "model", "--state-file", str(self.path)])
        self.assertEqual(model.execution_policy.position_source.value, "model")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(common + ["--mode", "simnow", "--state-file", str(self.path)])
        native = entry.parse_args(common + ["--mode", "simnow", "--state-file", str(self.path),
            "--enable-orders", "--confirm-simnow"])
        self.assertEqual(native.execution_policy.position_source.value, "fills")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(common + ["--valuation-day", "20261009"])
        mapped = entry.parse_args(common + ["--index-source", "dolphindb-dfs", "--replay-test-start-time", "13:55"])
        self.assertEqual(mapped.replay_test_start_time, "13:55")
        for extras in (["--replay-test-start-time", "13:55"], ["--replay-test-source-time", "16:00"],
                ["--index-source", "dolphindb-dfs", "--replay-test-start-time", "25:00"],
                ["--index-source", "dolphindb-dfs", "--replay-test-start-time", "13:55", "--resume"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(common + extras)

    def test_native_audit_assembly_replaces_frozen_components_and_preserves_client(self):
        originals = []
        def native_fixture(driver, **kwargs):
            execution = runtime.build_model_execution(runtime.CLIENT_ID,
                instrument_limits=kwargs["instrument_limits"], positions=kwargs["positions"])
            originals.append(execution)
            return execution
        with patch.object(runtime, "build_ctp_portfolio_execution", side_effect=native_fixture):
            session, _ = self.build(mode="simnow")
        original, = originals
        self.assertIsNot(session.execution, original)
        self.assertIs(session.execution.client, original.client)
        self.assertIs(session.execution.positions, original.positions)
        self.assertIs(session.execution.prices, original.prices)
        self.assertIs(session.execution.ledger, original.ledger)
        self.assertIs(session.execution.driver, original.driver)
        self.assertIs(session.execution.backend._backend, original.backend)
        self.assertIs(session.client.backend, session.execution.backend)
        self.assertEqual(session.execution.backend.snapshot, [])

    def test_index_missing_logs_once_then_reports_recovered_projection(self):
        session, _ = self.build(mode="recording")
        event = str(ns("2026-08-13 13:58"))
        row = {"status": "MISSING", "latest_event_ns": None}
        feed = NS(key="000852", source_id="fixture-dfs", poll=lambda pending: (),
            manifest={"minute_lookups": {event: row}})
        session.runner.feature_query_feeds = (feed,)
        with self.assertLogs("OptionMarketInputs", level="INFO") as logs:
            session.runner.poll_features()
            session.runner.poll_features()
            row.update(status="PROJECTED", received_ns=self.now)
            session.runner.poll_features()
        self.assertEqual(sum("指数分钟缺失" in line for line in logs.output), 1)
        self.assertEqual(sum("此前缺失" in line for line in logs.output), 1)

    def test_index_query_error_logged_and_propagated(self):
        session, _ = self.build(mode="recording")
        def fail(pending):
            raise RuntimeError("fixture query timeout")
        session.runner.feature_query_feeds = (NS(key="000852", poll=fail),)
        with self.assertLogs("OptionMarketInputs", level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "fixture query timeout"):
                session.runner.poll_features()
        self.assertIn("指数DFS查询失败", logs.output[0])

    def build(self, *, resume=False, mode="model", config=None, dfs_index=False, test_clock=False):
        service = ReferenceService()
        calendar = tuple(date(2026, 8, 13) + timedelta(days=n) for n in range(50)
            if (date(2026, 8, 13) + timedelta(days=n)).weekday() < 5)
        refs = runtime.OptionReferences(service, "20261009", calendar)
        market, index = Feed("test-market"), Feed("test-index")
        if test_clock:
            mapping = import_module("demos.08_option_vega.replay_test")
            market = mapping.mapped_bar_feed("test-market", Feed("raw-test"), price_type="last",
                time_map=mapping.ReplayTestTimeMap(service.valuation_day, "13:58"))
        index_id = InstrumentId.from_str("000852.XSHG")
        query_index = None
        if dfs_index:
            from bomber.framework.dataprep.features import FeatureQueryCache
            from bomber.framework.market.stream.feature_query import FeatureQueryBarFeed
            from tests.test_dolphindb_index_features import Session, row, index_bundle
            source = Session([row(event=ns("2026-08-13 13:58"), published=ns("2026-08-13 13:58")+100)])
            bundle = index_bundle()
            provider = bundle.provider("index.minute",
                session_factory=lambda **kwargs: source, clock_ns=lambda: self.now)
            cache = FeatureQueryCache(provider, dataset_id="index.minute", schema_version=1, keys=("000852",),
                start_ns=ns("2026-08-13 00:00"), end_ns=ns("2026-08-14 00:00")-1,
                clock_ns=lambda: self.now, source_id="fixture-query")
            query_index = index = FeatureQueryBarFeed(cache, key="000852", replay=True,
                projection=bundle.binding("index.minute").bar_projection)
        for instrument in refs.instrument_ids.values():
            market.register_instrument(InstrumentMeta(instrument, price_precision=1))
        index.register_instrument(InstrumentMeta(index_id, price_precision=4, size_precision=4))
        args = NS(mode=mode, simnow_environment="replay", replay_md_trading_day="20260813",
            expected_trading_day="20261009", frame_wait_seconds=2, max_market_age_seconds=300,
            instrument_max_age_seconds=30, config=config or signal.OptionVegaConfig(min_time_value=0),
            max_notional=Decimal("100000000"), limit_offset_ticks=1, flatten_day=None,
            index_code="000852", state_file=self.path if mode == "model" else None,
            account_reconciliation="off", strategy_position_source="model" if mode == "model" else "fills",
            ctp_price_type="last")
        inputs = NS(references=refs, feeds={"market": market, "index": index}, labels={"market": "inclusive-end" if test_clock else "end", "index": "end"},
            index_id=index_id, sources={**{key: "market" for key in refs.instrument_ids}, "000852": "index"},
            ctp_upstream=None, ctp_bar_feed=market if test_clock else None, dolphin_feeds=[], feature_query_feeds=(query_index,) if query_index else (),
            source_identity={"fixture": "memory"})
        session = runtime.assemble(args, inputs)
        session.runner.clock_ns = lambda: self.now
        if query_index:
            for channel in session.readiness.channels:
                if channel.identity.channel == "dolphindb-raw":
                    channel.clock_ns = lambda: self.now
        lifecycle = recovery.OptionLocalLifecycle("20261009", mode=mode, valuation_day="20260813",
            state_file=args.state_file, resume=resume, require_fills=mode == "model")
        self.sessions.append(session)
        return session, lifecycle

    def bars(self, session, clock, *, omit=(), index_delay=0, flush=True):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        stamp = ns(clock)
        expiry = date(2026, 9, 18)
        premium = signal.call_price(6500, 6600,
            signal.tau(datetime.fromtimestamp(stamp / 1e9, ZoneInfo("Asia/Shanghai")), expiry), .02, .25)
        values = {"MO2609-C-6600": premium, "IM2609": 6500, "000852": 6400}
        for key, price in values.items():
            if key in omit:
                continue
            if key == "000852":
                self.now += index_delay
            instrument = session.runner.key_by_instrument
            identity = next(i for i, name in instrument.items() if name == key)
            meta = session.feeds[session.runner.sources[key]].get_instrument_meta(identity)
            session.runner.publish(session.runner.sources[key], make_bar(instrument_id=identity,
                open=price, high=price, low=price, close=price, volume=1, ts_event=stamp,
                ts_init=self.now, meta=meta, bar_type="1-MINUTE"))
        if flush:
            self.now += 2_000_000_000
            session.runner.flush_frames()

    def test_dfs_index_enters_actual_recording_frame_without_execution_route(self):
        s, lifecycle = self.build(mode="recording", dfs_index=True)
        self.start(s, lifecycle)
        self.assertNotIn("000852", s.runner.observations)
        self.bars(s, "2026-08-13 13:58", omit=("000852",), flush=False)
        s.runner.poll_features()
        self.now += 2_000_000_000
        s.runner.flush_frames()
        self.assertEqual(len(s.client.requests), 1)
        self.assertEqual(s.strategy.fills_received, 0)
        self.assertEqual(s.strategy.entry_audit[0]["outcome"], "TARGET")
        self.assertEqual(s.runner.frame_audit[0]["status"], "CONSUMED")
        description = import_module("demos.08_option_vega.run_live").describe(s)
        self.assertEqual(description["recorded_requests"][0]["signal_ns"], s.strategy.entry_audit[0]["signal_ns"])
        self.assertTrue(all(str(i) != "000852.XSHG" for i in s.client.requests[0].targets))

    def test_mapped_replay_reaches_recording_target_and_model_owned_fill(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from decimal import Decimal
        for mode in ("recording", "model"):
            with self.subTest(mode=mode):
                s, lifecycle = self.build(mode=mode, dfs_index=True, test_clock=True,
                    config=signal.OptionVegaConfig(min_time_value=0, max_option_lots=1,
                        max_total_option_lots=2, max_future_lots=1))
                self.start(s, lifecycle)
                valuation = ns("2026-08-13 13:58")
                premium = signal.call_price(6500, 6600,
                    signal.tau(datetime.fromtimestamp(valuation/1e9, ZoneInfo("Asia/Shanghai")), date(2026, 9, 18)), .02, .25)
                for clock in ("2026-08-12 16:00", "2026-08-12 16:01"):
                    for key, price in (("MO2609-C-6600", premium), ("IM2609", 6500)):
                        value = NS(as_decimal=lambda price=price: Decimal(str(price)))
                        raw = NS(instrument_id=s.references.instrument_ids[key], price=value,
                            size=NS(as_decimal=lambda: Decimal(1)), ts_event=ns(clock), ts_init=self.now)
                        s.bar_feed._on_trade_tick(raw)
                        self.assertEqual(raw.ts_event, ns(clock))
                s.runner.poll_features()
                self.now += 2_000_000_000
                s.runner.flush_frames()
                lifecycle.verify(s)
                self.assertEqual(s.strategy.entry_audit[0]["outcome"], "TARGET")
                self.assertEqual(s.runner.frame_audit[0]["status"], "CONSUMED")
                self.assertEqual(s.runner.frame_audit[0]["valuation_ns"], valuation)
                description = import_module("demos.08_option_vega.run_live").describe(s)
                self.assertEqual(description["validation_scope"], "engineering_chain")
                self.assertEqual(description["replay_test_time_mapping"]["source_day"], "2026-08-12")
                self.assertEqual(lifecycle.snapshot(s, NS(trading_day="20261009", references=s.references))["bar_time_basis"], "replay_test_mapped")
                if mode == "recording":
                    self.assertEqual(len(s.client.requests), 1)
                    self.assertEqual(s.strategy.fills_received, 0)
                else:
                    self.assertGreater(s.strategy.fills_received, 0)
                    self.assertEqual(s.runner.position_manager.position(s.strategy.strategy_id,
                        str(s.references.instrument_ids["MO2609-C-6600"])), Decimal(-1))

    def test_recording_wait_over_reference_ttl_refreshes_before_gate(self):
        from tests.test_option_vega_live import ReferenceTests
        fixture = ReferenceTests(); fixture.setUp()
        service = fixture.service()
        service.clock_ns = lambda: self.now
        s, lifecycle = self.build(mode="recording")
        s.references = s.runner.references = runtime.OptionReferences(service, "20261009", s.references.calendar)
        started = self.now
        s.runner.readiness = lambda: self.now-started >= 31_000_000_000
        with patch.object(recovery.time, "monotonic", side_effect=lambda: (self.now-started)/1e9), \
                patch.object(recovery.time, "sleep", side_effect=lambda seconds: setattr(self, "now", self.now+int(seconds*1e9))):
            self.start(s, lifecycle)
        self.assertTrue(lifecycle.resume_ready)
        self.assertGreaterEqual(self.now-started, 31_000_000_000)
        self.assertTrue(service.manifest()["ready"])

    def test_expired_reference_report_keeps_day_and_zero_order_count(self):
        from tests.test_option_vega_live import ReferenceTests
        fixture = ReferenceTests(); fixture.setUp()
        service = fixture.service(); service.clock_ns = lambda: self.now
        s, lifecycle = self.build(mode="recording")
        s.references = runtime.OptionReferences(service, "20261009", s.references.calendar)
        self.now += 31_000_000_000
        fields = lifecycle.snapshot(s, LiveRunContext("20261009", ExitStack(), s.references))
        self.assertEqual(fields["trading_day"], "20261009")
        self.assertEqual(fields["orders_submitted"], 0)
        self.assertFalse(fields["reference_manifest"]["ready"])

    def test_dfs_index_after_frame_deadline_does_not_create_recording_goal(self):
        s, lifecycle = self.build(mode="recording", dfs_index=True)
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58", omit=("000852",), flush=False)
        self.now += 2_000_000_001
        s.runner.poll_features()
        s.runner.flush_frames()
        self.assertEqual(s.client.requests, ())
        self.assertEqual(s.runner.frames.late_count, 1)
        self.assertEqual(s.runner.skipped_feature_frames, 1)
        self.assertEqual(s.strategy.frames, 0)

    def start(self, session, lifecycle):
        lifecycle.start(session, LiveRunContext("20261009", ExitStack(), session.references))

    def test_model_owns_option_fills_then_hedges_actual_position(self):
        s, lifecycle = self.build()
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58", index_delay=1_000_000_000)
        option = s.references.instrument_ids["MO2609-C-6600"]
        self.assertLess(s.strategy.position(str(option)), 0)
        self.assertEqual(s.strategy.position(str(s.references.instrument_ids["IM2609"])), 0)
        self.now += 60_000_000_000
        self.bars(s, "2026-08-13 14:00")
        self.assertGreater(s.strategy.position(str(s.references.instrument_ids["IM2609"])), 0)
        self.assertEqual(s.runner.position_manager.unassigned_position(s.client.client_id, option), 0)
        lifecycle.verify(s)
        # 模型仓位不允许通过账户接口读取或伪装成物理仓位。
        with self.assertRaises(RuntimeError):
            s.strategy.account_position(str(option))

    def test_recording_target_has_no_fake_fill_or_index_execution_route(self):
        s, lifecycle = self.build(mode="recording")
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58")
        self.assertEqual(len(s.client.requests), 1)
        self.assertEqual(s.strategy.fills_received, 0)
        self.assertEqual(s.strategy.position(str(s.references.instrument_ids["MO2609-C-6600"])), 0)
        self.assertTrue(all(str(i) != "000852.XSHG" for i in s.client.requests[0].targets))

    def test_model_restore_preserves_position_and_does_not_reopen_same_day(self):
        s, lifecycle = self.build()
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58")
        self.bars(s, "2026-08-13 14:00")
        owned = dict(s.runner.position_manager.snapshot().strategy_positions)
        lifecycle.shutdown(s, None)
        self.now += 60_000_000_000
        restored, controller = self.build(resume=True)
        self.start(restored, controller)
        self.assertEqual(dict(restored.runner.position_manager.snapshot().strategy_positions), owned)
        self.assertFalse(restored.strategy.latest)
        self.assertFalse(restored.strategy.iv_cache)
        self.bars(restored, "2026-08-13 14:01")
        self.assertEqual(restored.strategy.entry_decisions, 1)
        self.assertEqual(restored.client.backend.reports, ())

    def test_failed_resume_preserves_file(self):
        s, lifecycle = self.build()
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58")
        lifecycle.shutdown(s, None)
        original = self.path.read_bytes()
        changed = signal.OptionVegaConfig(min_time_value=0, nav=Decimal("80000000"))
        restored, controller = self.build(resume=True, config=changed)
        with self.assertRaises(ValueError):
            self.start(restored, controller)
        result = controller.shutdown(restored, None)
        self.assertTrue(result["resume_state_preserved"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_wrong_index_date_and_missing_candidate(self):
        s, lifecycle = self.build()
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58", omit=("MO2609-C-6600",))
        self.assertEqual(s.strategy.entry_decisions, 0)
        self.assertFalse(s.client.backend.reports)
        with self.assertRaisesRegex(ValueError, "估值日期"):
            self.bars(s, "2026-08-14 13:58")

    def test_realtime_stale_optional_candidate_is_skipped_without_polluting_frame(self):
        s, lifecycle = self.build(mode="recording")
        self.start(s, lifecycle)
        s.runner.replay = False
        self.now = ns("2026-08-13 13:58")
        with self.assertLogs("OptionMarketInputs", level="WARNING"):
            self.bars(s, "2026-08-13 13:52", omit=("IM2609", "000852"), flush=False)
        self.assertIsNone(s.runner.failure)
        self.assertNotIn("MO2609-C-6600", s.runner.observations)
        self.assertNotIn("MO2609-C-6600", s.strategy.latest)
        self.assertEqual(s.runner.frames.pending, {})
        self.assertEqual(s.runner.ignored_stale_candidate_bars, 1)
        # 新鲜同分钟行情仍能选约并把非零目标送到Recording。
        self.bars(s, "2026-08-13 13:58")
        self.assertEqual(s.strategy.entry_audit[-1]["outcome"], "TARGET")
        self.assertEqual(len(s.client.requests), 1)
        description = import_module("demos.08_option_vega.run_live").describe(s)
        self.assertEqual(description["bar_time_audit"][-1]["action"], "IGNORED_CANDIDATE")
        self.assertEqual(description["ignored_stale_candidate_bars"], 1)

    def test_realtime_stale_target_option_and_its_future_remain_fatal(self):
        for key, omit in (("MO2609-C-6600", ("IM2609", "000852")),
                          ("IM2609", ("MO2609-C-6600", "000852"))):
            with self.subTest(key=key):
                s, lifecycle = self.build(mode="recording")
                self.start(s, lifecycle)
                s.runner.replay = False
                self.now = ns("2026-08-13 13:58")
                s.strategy.option_goals["MO2609-C-6600"] = Decimal(-1)
                with self.assertLogs("OptionMarketInputs", level="ERROR"):
                    with self.assertRaisesRegex(ValueError, f"key={key}.*required=True"):
                        self.bars(s, "2026-08-13 13:52", omit=omit, flush=False)
                self.assertIsNotNone(s.runner.failure)
                self.assertFalse(s.runner.accept_bars)
                self.assertEqual(s.runner.ignored_stale_candidate_bars, 0)
                self.assertEqual(s.runner.bar_time_audit[-1]["action"], "REJECTED")

    def test_realtime_stale_index_remains_fatal(self):
        s, lifecycle = self.build(mode="recording")
        self.start(s, lifecycle)
        s.runner.replay = False
        self.now = ns("2026-08-13 13:58")
        with self.assertLogs("OptionMarketInputs", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "key=000852.*required=True"):
                self.bars(s, "2026-08-13 13:52", omit=("MO2609-C-6600", "IM2609"), flush=False)
        self.assertFalse(s.runner.accept_bars)

    def test_realtime_stale_actual_position_remains_required_with_zero_goal(self):
        s, lifecycle = self.build(mode="model")
        self.start(s, lifecycle)
        self.bars(s, "2026-08-13 13:58")
        option = s.references.instrument_ids["MO2609-C-6600"]
        self.assertLess(s.strategy.position(str(option)), 0)
        s.strategy.option_goals["MO2609-C-6600"] = Decimal(0)
        s.runner.replay = False
        self.now = ns("2026-08-13 14:04")
        with self.assertLogs("OptionMarketInputs", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "key=MO2609-C-6600.*required=True"):
                self.bars(s, "2026-08-13 13:58", omit=("IM2609", "000852"), flush=False)
        self.assertEqual(s.runner.bar_time_audit[-1]["action"], "REJECTED")
        self.assertLess(s.strategy.position(str(option)), 0)

    def test_realtime_future_optional_candidate_is_not_ignored(self):
        s, lifecycle = self.build(mode="recording")
        self.start(s, lifecycle)
        s.runner.replay = False
        self.now = ns("2026-08-13 13:58")
        with self.assertLogs("OptionMarketInputs", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "age_seconds=-60.0"):
                self.bars(s, "2026-08-13 13:59", omit=("IM2609", "000852"), flush=False)
        self.assertEqual(s.runner.bar_time_audit[-1]["reason"], "FUTURE_BAR")
        self.assertEqual(s.runner.ignored_stale_candidate_bars, 0)

    def test_missing_held_leg_waits_then_fails_without_erasing_position(self):
        s, controller = self.build()
        self.start(s, controller)
        self.bars(s, "2026-08-13 13:58")
        instrument = s.references.instrument_ids["MO2609-C-6600"]
        position = s.strategy.position(str(instrument))
        for minute in range(4, 10):
            self.bars(s, f"2026-08-13 14:{minute:02d}", omit=("MO2609-C-6600",))
        self.assertEqual(s.strategy.signals[-1]["action"], "RESUME_WAIT")
        with self.assertRaisesRegex(RuntimeError, "缺少新鲜行情"):
            self.bars(s, "2026-08-13 14:10", omit=("MO2609-C-6600",))
        self.assertEqual(s.strategy.position(str(instrument)), position)
        self.assertIsNotNone(s.runner.failure)
        self.assertFalse(s.runner.accept_bars)

    def test_quote_bar_is_mid_with_zero_volume_and_never_a_trade_tick(self):
        from bomber.framework.market.stream.aggregation import QuoteMidBarFeed
        instrument = InstrumentId.from_str("MO2609-C-6600.CFFEX")
        upstream = Feed("quote-source")
        feed = QuoteMidBarFeed("quote-bars", upstream)
        feed.register_instrument(InstrumentMeta(instrument, price_precision=1))
        from bomber.framework.market.basic.base import DataType
        feed.subscribe(instrument, DataType.BAR, "1-MINUTE")
        bars, ticks = [], []
        feed.register_bar_handler(bars.append)
        feed.register_trade_tick_handler(ticks.append)
        feed.connect()
        self.addCleanup(feed.disconnect)
        class Number:
            def __init__(self, value):
                self.value = Decimal(value)
            def as_decimal(self):
                return self.value
        for clock, bid, ask in (("2026-08-13 13:57", "20", "20.1"),
                                ("2026-08-13 13:58", "21", "21.2")):
            upstream._emit_quote_tick(NS(instrument_id=instrument, bid_price=Number(bid),
                ask_price=Number(ask), ts_event=ns(clock), ts_init=self.now))
        self.assertEqual(len(bars), 1)
        self.assertIn("-MID-EXTERNAL", str(bars[0].bar_type))
        self.assertEqual(bars[0].volume.as_decimal(), 0)
        self.assertEqual(bars[0].close.as_decimal(), Decimal("20.05"))
        self.assertFalse(ticks)

    def test_delayed_partial_fill_hedges_filled_lots_without_resending_option(self):
        from bomber.framework.trader.execution.contracts import ExecutionReport, ExecutionReportType
        s, controller = self.build()
        self.start(s, controller)
        sent = []
        backend = s.client.backend
        backend.submit_order = sent.append  # 接受异步委托；实际状态机/预留量保持原样。
        self.bars(s, "2026-08-13 13:58")
        order = sent[0]
        key = str(order.instrument_id)
        self.assertEqual(s.strategy.position(key), 0)
        self.assertEqual(s.strategy.strategy_working_quantity(key), -order.quantity)
        partial = ExecutionReport(backend.backend_id, "partial-option", order.instrument_id,
            ExecutionReportType.PARTIALLY_FILLED, self.now, filled_quantity=Decimal(3),
            fill_price=order.metadata.get("price", Decimal(20)), order_side=order.side,
            order_quantity=order.quantity, sequence=1, report_id="partial:1",
            metadata={**order.metadata, "strategy_id": order.strategy_id, "trade_id": "partial-trade"})
        for handler in tuple(backend._handlers):
            handler(partial)
            handler(partial)
        self.assertEqual(s.strategy.position(key), -3)
        self.assertEqual(s.strategy.fills_received, 1)
        self.assertEqual(s.strategy.strategy_working_quantity(key), -(order.quantity - 3))
        self.bars(s, "2026-08-13 14:00")
        self.assertEqual(len([item for item in sent if item.instrument_id == order.instrument_id]), 1)
        self.assertTrue(any(item.instrument_id != order.instrument_id and item.side is OrderSide.BUY for item in sent))

    def test_generic_close_is_explicitly_selected_for_cffex(self):
        ledger = CtpPositionLedger("20261009")
        instrument = InstrumentId.from_str("MO2610-C-6600.CFFEX")
        ledger.apply_fill(instrument, OrderSide.SELL, PositionEffect.OPEN, 2, 20, 100)
        request = ExecutionRequest("test", 1, "test", 1, {instrument: Decimal(0)}, "DIRECT")
        orders = CtpClosePlanner(ledger, generic_close_venues=frozenset(("CFFEX",))).plan(request)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].position_effect, PositionEffect.CLOSE)
        self.assertEqual(orders[0].quantity, 2)


if __name__ == "__main__":
    unittest.main()
