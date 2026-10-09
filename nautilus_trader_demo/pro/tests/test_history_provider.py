"""历史行情工厂、文件/数据库切换、窗口策略及EMA预热的无网络验收。"""
from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch, Mock
import json
import unittest
import pandas as pd

from bomber.framework.dataprep.history import *
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, DataSourcePurpose
from bomber.framework.dataprep.sources.policy import validate_source
from bomber.framework.trader.runtime.trading_sessions import CopperSessions, SHANGHAI

ROOT = Path(__file__).resolve().parents[1]
MINUTE = 60_000_000_000
START = pd.Timestamp("2026-10-09T09:00:00+08:00").value
STAMPS = (START + MINUTE - 1, START + 2 * MINUTE - 1)
INSTRUMENT = "cu2611.SHFE"


def bar(stamp, price="100"):
    return HistoryBar(INSTRUMENT, stamp, adjusted_close=Decimal(price))


class MemoryProvider(HistoryProvider):
    def __init__(self, config):
        self.config = config
    def open(self):
        self.config["opens"] = self.config.get("opens", 0) + 1
    def read(self, request):
        if self.config.get("fail"):
            raise HistoryUnavailable("injected outage")
        return tuple(self.config.get("bars", ()))
    def close(self):
        self.config["closes"] = self.config.get("closes", 0) + 1


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "minutes.jsonl"
        class Factory(HistoryProviderFactory):
            _providers = dict(HistoryProviderFactory._providers, dolphindb=MemoryProvider, mongodb=MemoryProvider)
        self.factory = Factory
        self.config = {"bars": tuple(bar(t) for t in STAMPS)}
        self.calendar = CopperSessions(ROOT / "demos/01_main_ema/shfe_cu_2026.json")

    def write_file(self, stamps):
        self.file.write_text("\n".join(json.dumps({"instrument_id": INSTRUMENT,
            "ts_event": t, "adjusted_close": "100"}) for t in stamps))

    def service(self, **kwargs):
        return HistoryService(database_config=self.config, factory=self.factory, **kwargs)

    def test_factory_construction_never_imports_sdk_or_connects(self):
        config = DolphinDbHistoryConfig(DolphinDbReferenceConfig("fake", 8848, "fake", "fake"),
            "dfs://test", "minutes")
        with patch("bomber.framework.dataprep.history.dolphindb._session_factory") as sdk:
            HistoryProviderFactory.create("dolphindb", config)
            sdk.assert_not_called()

    def test_registered_mongodb_uses_same_service_contract(self):
        result = self.service(database_backend="mongodb").load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(result.sources, ("mongodb",))
        self.assertEqual(result.missing, ())
        self.assertEqual(self.config["closes"], 1)

    def test_new_database_registration_does_not_change_service_or_strategy(self):
        self.factory.register("customdb", MemoryProvider)
        result = self.service(database_backend="customdb").load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(result.sources, ("customdb",))
        with self.assertRaises(ValueError):
            self.factory.create_for(DataSourcePurpose.OFFLINE_MARKET_HISTORY, "customdb", self.config)

    def test_unimplemented_mongodb_is_explicit_and_does_not_fallback(self):
        with self.assertRaises(NotImplementedError):
            HistoryProviderFactory.create("mongodb", {}).open()

    def test_unknown_factory_and_nonprovider_registration_rejected(self):
        with self.assertRaises(ValueError):
            self.factory.create("typo", {})
        with self.assertRaises(TypeError):
            self.factory.register("custom", dict)
        with self.assertRaises(ValueError):
            self.factory.register("dolphindb", MemoryProvider)
        self.factory.register("mongodb", MemoryProvider, replace=True)

    def test_purpose_matrix_keeps_offline_file_only_and_allows_live_databases(self):
        for backend in ("file", "dolphindb", "mongodb"):
            validate_source(backend, DataSourcePurpose.LIVE_MARKET_HISTORY)
        for backend in ("dolphindb", "mongodb"):
            validate_source(backend, DataSourcePurpose.LIVE_RECOVERY)
            with self.assertRaises(ValueError):
                validate_source(backend, DataSourcePurpose.OFFLINE_MARKET_HISTORY)
        with self.assertRaises(ValueError):
            validate_source("file", DataSourcePurpose.LIVE_RECOVERY)
        with self.assertRaises(ValueError):
            validate_source("mongodb", DataSourcePurpose.LIVE_REFERENCE)

    def test_complete_preopen_file_never_opens_database(self):
        self.write_file(STAMPS)
        result = self.service(file_path=self.file).load(INSTRUMENT, STAMPS, phase="preopen")
        self.assertEqual(result.sources, ("file",))
        self.assertNotIn("opens", self.config)

    def test_csv_text_file_uses_same_canonical_minute_contract(self):
        path = self.root / "bars.csv"
        path.write_text("instrument_id,ts_event,close,cumulative_factor\n" +
            "\n".join(f"{INSTRUMENT},{stamp},50,2" for stamp in STAMPS))
        result = self.service(file_path=path).load(INSTRUMENT, STAMPS, phase="preopen")
        self.assertEqual(result.sources, ("file",))
        self.assertEqual(result.bars[0].adjusted_close, Decimal(100))

    def test_absent_preopen_file_falls_back_to_database(self):
        result = self.service(file_path=self.file).load(INSTRUMENT, STAMPS, phase="preopen")
        self.assertEqual(result.sources, ("dolphindb",))
        self.assertIn("file-not-found", result.unavailable_sources)

    def test_incomplete_file_is_filled_from_database(self):
        self.write_file(STAMPS[:1])
        result = self.service(file_path=self.file).load(INSTRUMENT, STAMPS, phase="preopen")
        self.assertEqual(result.sources, ("file", "dolphindb"))
        self.assertEqual(len(result.bars), 2)

    def test_recovery_ignores_local_file_even_when_complete(self):
        self.file.write_text("invalid file should not be read")
        result = self.service(file_path=self.file).load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(result.sources, ("dolphindb",))

    def test_live_buffer_completes_gap_without_query_or_duplicate_indicator_rows(self):
        result = self.service().load(INSTRUMENT, STAMPS, phase="recovery", live_bars=self.config["bars"])
        self.assertEqual(result.sources, ("live-buffer",))
        self.assertEqual(result.missing, ())
        self.assertNotIn("opens", self.config)

    def test_missing_strict_rejects_and_records_coverage(self):
        self.config["bars"] = ()
        service = self.service()
        with self.assertRaises(HistoryUnavailable):
            service.load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(service.last_audit["missing"], 2)
        self.assertEqual(self.config["closes"], 1)

    def test_missing_allow_returns_real_rows_and_gaps(self):
        self.config["bars"] = (bar(STAMPS[0]),)
        result = self.service(missing_policy="allow").load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(result.missing, STAMPS[1:])
        self.assertEqual(len(result.bars), 1)
        self.assertTrue(result.allowed_missing)

    def test_allow_empty_without_database_config_is_audited(self):
        result = HistoryService(missing_policy="allow").load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertEqual(result.bars, ())
        self.assertEqual(result.missing, STAMPS)
        self.assertIn("database-unconfigured", result.unavailable_sources)

    def test_database_outage_policy_closes_source_in_both_modes(self):
        self.config["fail"] = True
        with self.assertRaises(HistoryUnavailable):
            self.service().load(INSTRUMENT, STAMPS, phase="recovery")
        result = self.service(missing_policy="allow").load(INSTRUMENT, STAMPS, phase="recovery")
        self.assertIn("dolphindb", result.unavailable_sources)
        self.assertEqual(self.config["opens"], self.config["closes"])

    def test_file_database_conflict_remains_fatal_even_when_allow_missing(self):
        self.write_file(STAMPS[:1])
        self.config["bars"] = (bar(STAMPS[0], "101"), bar(STAMPS[1]))
        with self.assertRaises(HistoryError):
            self.service(file_path=self.file, missing_policy="allow").load(INSTRUMENT, STAMPS, phase="preopen")

    def test_complete_duplicates_deduplicate_and_conflicts_reject(self):
        from bomber.framework.dataprep.history.base import select_bars
        request = HistoryRequest(INSTRUMENT, START, START + 2 * MINUTE)
        self.assertEqual(select_bars((bar(STAMPS[0]), bar(STAMPS[0])), request), (bar(STAMPS[0]),))
        with self.assertRaises(HistoryError):
            select_bars((bar(STAMPS[0]), bar(STAMPS[0], "101")), request)

    def test_future_rows_other_instrument_filtered(self):
        from bomber.framework.dataprep.history.base import select_bars
        request = HistoryRequest(INSTRUMENT, START, START + MINUTE)
        self.assertEqual(select_bars((bar(STAMPS[1]), HistoryBar("rb2704.SHFE", STAMPS[0], close=1)), request), ())

    def test_invalid_price_ohlc_or_minute_boundary_rejected(self):
        for kwargs in ({"close": "nan"}, {"close": "0"}, {"close": "10", "high": "9"},
                       {"close": "10", "volume": "-1"}):
            with self.assertRaises(HistoryError):
                HistoryBar(INSTRUMENT, STAMPS[0], **kwargs)
        with self.assertRaises(HistoryError):
            HistoryBar(INSTRUMENT, START, close=10)

    def test_event_specific_factor_derives_adjusted_close_without_current_factor_guess(self):
        point = HistoryBar(INSTRUMENT, STAMPS[0], close="100", cumulative_factor="1.234")
        self.assertEqual(point.adjusted_close, Decimal("123.400"))

    def test_240_minute_window_excludes_intraday_rest(self):
        end = pd.Timestamp("2026-10-09T14:47:00+08:00").value
        stamps = self.calendar.last_minutes(end, 240)
        self.assertEqual(len(stamps), 240)
        self.assertEqual(stamps[-1], end - 1)
        for stamp in stamps:
            self.assertIsNotNone(self.calendar.window(datetime.fromtimestamp(stamp // 1000000000, SHANGHAI)))

    def test_dolphindb_config_rejects_injection_and_unknown_columns(self):
        connection = DolphinDbReferenceConfig("fake", 8848, "fake", "fake")
        for kwargs in ({"table": "minutes;dropTable"}, {"database": "bad"},
                       {"columns": {"instrument_id": "Code", "ts_event": "ts"}}, {"time_label": "guess"}):
            values = {"database": "dfs://test", "table": "minutes", **kwargs}
            with self.assertRaises(HistoryError):
                DolphinDbHistoryConfig(connection, **values)

    def test_dolphindb_query_bounded_and_timestamp_unit_explicit(self):
        connection = DolphinDbReferenceConfig("fake", 8848, "fake", "fake")
        config = DolphinDbHistoryConfig(connection, "dfs://test", "minutes", time_unit="ms", time_label="start")
        sdk = Mock()
        sdk.connect.return_value = True
        sdk.run.return_value = pd.DataFrame([{"instrument_id": INSTRUMENT, "ts_event": STAMPS[0], "close": "100"}])
        provider = DolphinDbHistoryProvider(config, session_factory=lambda **kwargs: sdk)
        provider.open()
        bars = provider.read(HistoryRequest(INSTRUMENT, START, START + MINUTE))
        script = sdk.run.call_args.args[0]
        self.assertIn("long(ts_event)*1000000", script)
        self.assertIn(str(MINUTE - 1), script)
        self.assertEqual(bars[0].ts_event, STAMPS[0])
        provider.close()
        sdk.close.assert_called_once()

    def test_dolphindb_connection_failure_releases_and_sanitizes_error(self):
        sdk = Mock()
        sdk.connect.side_effect = RuntimeError("secret password must not leak")
        config = DolphinDbHistoryConfig(DolphinDbReferenceConfig("fake", 8848, "fake", "fake"), "dfs://test", "minutes")
        provider = DolphinDbHistoryProvider(config, session_factory=lambda **kwargs: sdk)
        with self.assertRaisesRegex(HistoryUnavailable, "历史库连接失败") as caught:
            provider.open()
        self.assertNotIn("secret", str(caught.exception))
        sdk.close.assert_called_once()

    def test_symbol_database_format_requires_explicit_venue_and_normalizes_identity(self):
        connection = DolphinDbReferenceConfig("fake", 8848, "fake", "fake")
        config = DolphinDbHistoryConfig(connection, "dfs://test", "minutes", instrument_format="symbol", venue="SHFE")
        sdk = Mock()
        sdk.connect.return_value = True
        sdk.run.return_value = pd.DataFrame([{"instrument_id": "cu2611", "ts_event": STAMPS[0], "close": "100"}])
        provider = DolphinDbHistoryProvider(config, session_factory=lambda **kwargs: sdk)
        provider.open()
        point = provider.read(HistoryRequest(INSTRUMENT, START, START + MINUTE))[0]
        self.assertEqual(point.instrument_id, INSTRUMENT)
        self.assertIn('instrument_id="cu2611"', sdk.run.call_args.args[0])
        with self.assertRaises(HistoryError):
            provider._script(HistoryRequest("cu2611.INE", START, START + MINUTE))
        provider.close()

    def test_history_warm_only_updates_indicator_not_targets_or_positions(self):
        module = import_module("demos.01_main_ema.history")
        ema = import_module("demos.01_main_ema.strategy")
        checkpoint = import_module("demos.01_main_ema.checkpoint")
        strategy = ema.MainEmaStrategy("ema", None, ema.MainEmaConfig("CU", "SHFE"))
        cp = checkpoint.EmaCheckpoint(strategy, INSTRUMENT)
        runner = SimpleNamespace(ema_checkpoint=cp, calendar=self.calendar, history_service=self.service(),
            history_minutes=2, _submit_lock=RLock(), _clients={"client": SimpleNamespace(_submit_lock=RLock())})
        module.warm_history(runner, START + 2 * MINUTE, phase="preopen")
        self.assertEqual(strategy.bars_used, 2)
        self.assertEqual(strategy._revision, 0)
        self.assertIsNone(strategy.last_target)
        self.assertEqual(strategy.fills_received, 0)

    def test_invalid_unadjusted_history_does_not_partially_warm(self):
        module = import_module("demos.01_main_ema.history")
        ema = import_module("demos.01_main_ema.strategy")
        checkpoint = import_module("demos.01_main_ema.checkpoint")
        strategy = ema.MainEmaStrategy("ema", None, ema.MainEmaConfig("CU", "SHFE"))
        self.config["bars"] = (bar(STAMPS[0]), HistoryBar(INSTRUMENT, STAMPS[1], close=100))
        runner = SimpleNamespace(ema_checkpoint=checkpoint.EmaCheckpoint(strategy, INSTRUMENT), calendar=self.calendar,
            history_service=self.service(), history_minutes=2)
        with self.assertRaises(HistoryError):
            module.warm_history(runner, START + 2 * MINUTE, phase="preopen")
        self.assertEqual(strategy.bars_used, 0)

    def test_recovery_keeps_published_target_revision_and_never_reprocesses_old_bar(self):
        module = import_module("demos.01_main_ema.history")
        ema = import_module("demos.01_main_ema.strategy")
        checkpoint = import_module("demos.01_main_ema.checkpoint")
        strategy = ema.MainEmaStrategy("ema", None, ema.MainEmaConfig("CU", "SHFE"))
        cp = checkpoint.EmaCheckpoint(strategy, INSTRUMENT)
        for index in range(5):
            cp.warm_closed(STAMPS[0] - (4 - index) * MINUTE, 100)
        strategy._revision, strategy.last_target = 3, Decimal(-1)
        runner = SimpleNamespace(ema_checkpoint=cp, calendar=self.calendar, history_service=self.service(),
            history_minutes=240, _submit_lock=RLock(), _clients={"client": SimpleNamespace(_submit_lock=RLock())})
        module.warm_history(runner, START + 2 * MINUTE, phase="recovery")
        self.assertEqual(strategy.bars_used, 6)
        self.assertEqual(strategy._revision, 3)
        self.assertEqual(strategy.last_target, Decimal(-1))

    def test_cli_exposes_explicit_240_and_allow_and_rejects_unknown_backend(self):
        from io import StringIO
        from contextlib import redirect_stderr
        live = import_module("demos.01_main_ema.run_live")
        base = ["--connect", "--product", "CU", "--expected-source-day", "20261009"]
        args = live.parse_args([*base, "--history-minutes", "240", "--history-missing-policy", "allow"])
        self.assertEqual(args.history_minutes, 240)
        self.assertEqual(args.history_missing_policy, "allow")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            live.parse_args([*base, "--history-db", "typo"])

    def test_ctp_history_failure_never_arms_and_does_not_submit(self):
        from tests.run_main_ema_live import LiveTests, live
        fixture = LiveTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        session = fixture.session(orders=True)
        session.client.disarm("test-history")
        session.runner.accept_bars = False
        session.runner.mark_history_live_start = Mock()
        session.runner.prepare_history = Mock(side_effect=HistoryUnavailable("missing history"))
        controller = live.MainEmaSessionLifecycle(session.transport, md_front="tcp://fake:1234", orders=True)
        controller.driver = session.driver
        with patch.object(session.client, "arm_demo") as arm:
            with self.assertRaises(HistoryUnavailable):
                controller.start(session, SimpleNamespace(trading_day="20260922"))
        arm.assert_not_called()
        self.assertEqual(session.transport._api.sent, [])

    def test_ctp_history_allow_does_not_bypass_md_health_gate(self):
        from tests.run_main_ema_live import LiveTests, live
        fixture = LiveTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        session = fixture.session(orders=True)
        session.client.disarm("test-history")
        session.runner.accept_bars = False
        session.runner.mark_history_live_start = Mock()
        session.runner.prepare_history = lambda: setattr(session.upstream, "latest_trading_day", "wrong-day")
        controller = live.MainEmaSessionLifecycle(session.transport, md_front="tcp://fake:1234", orders=True)
        controller.driver = session.driver
        with patch.object(session.client, "arm_demo") as arm:
            with self.assertRaisesRegex(RuntimeError, "历史加载后MD/TD未就绪"):
                controller.start(session, SimpleNamespace(trading_day="20260922"))
        arm.assert_not_called()
        self.assertEqual(session.transport._api.sent, [])

    def test_live_buffer_warms_without_orders_then_first_new_bar_publishes_current_target(self):
        from tests.run_main_ema_live import LiveTests, ManualMd, BASE, live
        from bomber.framework.market.basic.base import make_bar
        fixture = LiveTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        references = fixture.reference()
        path = fixture.root / "warm.jsonl"
        path.write_text(json.dumps({"instrument_id": str(references.instrument_id),
            "ts_event": BASE + MINUTE - 1, "adjusted_close": "6200"}))
        args = SimpleNamespace(mode="recording", product="RB", fast=2, slow=3, quantity=Decimal(1),
            limit_offset_ticks=1, max_notional=Decimal(50000), state_file=None, history_minutes=2,
            history_config=None, history_db="dolphindb", history_file=path, history_missing_policy="fail",
            history_stage="preopen", reference_timeout=15,
            trading_calendar=ROOT / "demos/01_main_ema/shfe_cu_2026.json")
        transport, driver, _ = fixture.transport_driver()
        driver.start(lambda report: None)
        session = live.assemble(args, references, driver, ManualMd())
        session.runner.start()
        fixture.addCleanup(session.runner.stop)
        session.runner.mark_history_live_start()
        fixture.now = BASE + 2 * MINUTE + 1000
        event = make_bar(instrument_id=references.instrument_id, open=3100, high=3100, low=3100,
            close=3100, volume=1, ts_event=BASE + 2 * MINUTE - 1, ts_init=fixture.now,
            meta=references.instrument_meta())
        session.runner.publish("ctp-bars", event)
        self.assertEqual(session.client.requests, ())
        session.runner.prepare_history()
        session.runner.begin_bars()
        self.assertEqual(session.strategy.bars_used, 2)
        self.assertEqual(session.strategy._revision, 0)
        self.assertEqual(session.client.requests, ())
        fixture.tick(session, 2)
        fixture.tick(session, 3)
        self.assertEqual(session.strategy.bars_used, 3)
        self.assertEqual(len(session.client.requests), 1)
        self.assertEqual(session.strategy.last_target, Decimal(1))
        self.assertEqual(transport._api.sent, [])


    def im_config(self, **kwargs):
        values = json.loads((ROOT / "demos/01_main_ema/history_im_raw.json").read_text())
        values.update(kwargs)
        return DolphinDbHistoryConfig(DolphinDbReferenceConfig("fake", 8848, "fake", "fake"), **values)

    def test_im_composite_time_query_normalizes_shanghai_and_preserves_raw_prices(self):
        stamp = pd.Timestamp("2026-10-09T09:30:59.999999999+08:00").value
        request = HistoryRequest("IM2611.CFFEX", stamp - MINUTE + 1, stamp + 1)
        sdk = Mock()
        sdk.connect.return_value = True
        sdk.run.return_value = pd.DataFrame([dict(instrument_id="IM2611", ts_event=stamp,
            open=7059, high=7059, low=7059, close=7059)])
        provider = DolphinDbHistoryProvider(self.im_config(), session_factory=lambda **kwargs: sdk)
        provider.open()
        bars = provider.read(request)
        provider.close()
        script = sdk.run.call_args.args[0]
        self.assertIn("long(nanotimestamp(concatDateTime(date,minTime)))", script)
        self.assertIn("-28800000000000l", script)
        self.assertIn('Code="IM2611"', script)
        self.assertIn("date>=2026.10.09,date<=2026.10.09", script)
        self.assertIn("order by date,minTime", script)
        self.assertIn(str(stamp + 1) + "l", script)  # LONG UTC boundary, no default INT overflow.
        self.assertEqual(bars[0].instrument_id, "IM2611.CFFEX")
        self.assertEqual(bars[0].ts_event, stamp)
        self.assertEqual(bars[0].close, Decimal(7059))
        self.assertIsNone(bars[0].adjusted_close)
        self.assertIsNone(bars[0].volume)
        sdk.close.assert_called_once()

    def test_composite_time_rejects_partial_unsafe_or_ambiguous_configuration(self):
        for changes in ({"time_column": None}, {"date_column": "date;delete"},
                {"columns": {"instrument_id": "Code", "close": "close", "ts_event": "ts"}},
                {"source_timezone": "guess"}, {"time_label": "end_ns"}, {"time_unit": "ms"}):
            with self.subTest(changes=changes), self.assertRaises(HistoryError):
                self.im_config(**changes)

    def test_composite_utc_does_not_apply_shanghai_shift(self):
        provider = DolphinDbHistoryProvider(self.im_config(source_timezone="UTC"))
        script = provider._script(HistoryRequest("IM2611.CFFEX", START, START + MINUTE))
        self.assertNotIn("28800000000000", script)
        self.assertIn("59999999999", script)

    def test_readonly_probe_uses_factory_and_never_promotes_raw_close_to_adjusted(self):
        from scripts.integration.history.readonly import probe
        provider, factory = Mock(), Mock()
        factory.create_for.return_value = provider
        provider.read.return_value = (HistoryBar("IM2611.CFFEX", STAMPS[0],
            open=7059, high=7059, low=7059, close=7059),)
        request = HistoryRequest("IM2611.CFFEX", START, START + MINUTE)
        config = self.im_config()
        result = probe(config, request, expected_bars=1, factory=factory)
        factory.create_for.assert_called_once_with(DataSourcePurpose.LIVE_RECOVERY, "dolphindb", config)
        provider.close.assert_called_once()
        self.assertEqual(result["price_basis"], "raw")
        self.assertFalse(result["ema_warmed"])
        self.assertEqual(result["orders_submitted"], 0)
        self.assertEqual(result["first"]["minute_start"], "2026-10-09T01:00:00+00:00")

    def test_readonly_probe_empty_count_and_ohlc_failures_always_release_provider(self):
        from scripts.integration.history.readonly import probe
        point = HistoryBar("IM2611.CFFEX", STAMPS[0], open=1, high=1, low=1, close=1)
        for rows, expected in (((), None), ((point,), 2),
                ((HistoryBar("IM2611.CFFEX", STAMPS[0], close=1),), 1)):
            provider, factory = Mock(), Mock()
            provider.read.return_value = rows
            factory.create_for.return_value = provider
            with self.assertRaises(HistoryError):
                probe(self.im_config(), HistoryRequest("IM2611.CFFEX", START, START + MINUTE),
                    expected_bars=expected, factory=factory)
            provider.close.assert_called_once()

    def test_readonly_probe_query_failure_releases_provider(self):
        from scripts.integration.history.readonly import probe
        provider, factory = Mock(), Mock()
        factory.create_for.return_value = provider
        provider.read.side_effect = HistoryUnavailable("query failure")
        with self.assertRaises(HistoryUnavailable):
            probe(self.im_config(), HistoryRequest("IM2611.CFFEX", START, START + MINUTE), factory=factory)
        provider.close.assert_called_once()

    def test_readonly_cli_requires_explicit_connect_aware_complete_historical_boundaries(self):
        from scripts.integration.history.readonly import parse_args
        from contextlib import redirect_stderr
        from io import StringIO
        base = ["--config", "unused.json", "--instrument", "IM2611.CFFEX"]
        times = ["--start", "2026-09-22T09:30:00+08:00", "--end", "2026-09-22T09:40:00+08:00"]
        args = parse_args([*base, *times, "--connect", "--expected-bars", "10"])
        self.assertEqual(args.request.end_ns - args.request.start_ns, 10 * MINUTE)
        for invalid in ([*base, *times], [*base, "--connect", "--start", "2026-09-22T09:30:00",
                "--end", "2026-09-22T09:40:00+08:00"], [*base, "--connect", "--start",
                "2026-09-22T09:30:01+08:00", "--end", "2026-09-22T09:40:00+08:00"],
                [*base, *times, "--connect", "--expected-bars", "0"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(invalid)

    def test_readonly_report_collision_preserves_existing_evidence(self):
        from scripts.integration.history.readonly import write_report
        path = self.root / "report.json"
        path.write_text("original")
        with self.assertRaises(FileExistsError):
            write_report(path, {"status": "passed"})
        self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main(verbosity=2)
