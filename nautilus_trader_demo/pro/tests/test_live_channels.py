"""Live渠道分离的纯Python验收；原生SDK、网络和真实凭据均不参与。"""

import ast
from contextlib import ExitStack
from decimal import Decimal
from enum import Enum
from importlib import import_module
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tests.test_live_common import ROOT, NOW, MONO, load

contracts = import_module("_test_live_runtime.live.contracts")
profiles = import_module("_test_live_runtime.live.profiles")
health = import_module("_test_live_runtime.live.health")
ctp = import_module("_test_live_runtime.live.channels.ctp")
binance = import_module("_test_live_runtime.live.channels.binance")


def feed():
    return NS(source_id="market-source", latest_trading_day="20261008",
        latest_receive_monotonic_ns=MONO,
        health_snapshot=NS(state="ready", reason="healthy", last_event_ts=NOW,
                           last_event_received_ns=NOW))


class ChannelTests(unittest.TestCase):
    def test_ctp_replay_md_day_is_independent_from_td_day(self):
        upstream = feed()
        md = ctp.CtpMarketChannel(upstream, "20261009",
            profile=ctp.CtpLiveProfile("replay", "20261008"), monotonic_ns=lambda: MONO)
        driver = NS(driver_id="native-td", trading_day="20261009", is_simnow_session=True)
        td = ctp.CtpExecutionChannel(driver, "20261009", require_orders=True)
        check = contracts.SessionReadiness(md, td)
        self.assertTrue(check())
        driver.trading_day = "20261008"
        result = check.check()
        self.assertFalse(result)
        self.assertEqual([(i.source.domain, i.code) for i in result.issues],
                         [("execution", "TD_TRADING_DAY_MISMATCH")])
        self.assertIs(check.last_result, result)

    def test_collects_all_failures_with_channel_and_source(self):
        upstream = feed()
        upstream.latest_trading_day = "20261007"
        upstream.latest_receive_monotonic_ns -= 11_000_000_000
        upstream.health_snapshot.state = "degraded"
        driver = NS(driver_id="third-party-ctp", trading_day=None, is_simnow_session=False)
        check = contracts.SessionReadiness(
            ctp.CtpMarketChannel(upstream, "20261008", monotonic_ns=lambda: MONO),
            ctp.CtpExecutionChannel(driver, "20261008", require_orders=True))
        result = check.check()
        self.assertEqual({issue.code for issue in result.issues}, {
            "STREAM_UNHEALTHY", "MD_TRADING_DAY_MISMATCH", "RECEIVE_STALE",
            "TD_TRADING_DAY_MISMATCH", "EXECUTION_NOT_AUTHORIZED", "EXECUTION_DISCONNECTED"})
        saved = json.loads(json.dumps(result.as_dict()))
        self.assertEqual({s["source_id"] for s in saved["sources"]},
                         {"market-source", "third-party-ctp"})
        exposed = result.as_dict()
        exposed["sources"][0]["channel"] = "other"
        self.assertEqual(result.sources[0].channel, "ctp")

    def test_ctp_market_never_reads_execution_driver(self):
        self.assertTrue(ctp.CtpMarketChannel(feed(), "20261008",
            monotonic_ns=lambda: MONO).check())

    def test_ctp_execution_never_reads_market_or_replay(self):
        self.assertTrue(ctp.CtpExecutionChannel(NS(driver_id="td", trading_day="20261009"),
                                               "20261009").check())

    def test_ctp_disconnected_driver_cannot_pass_with_retained_trading_day(self):
        driver = NS(driver_id="td", trading_day="20261009", _connected=False)
        channel = ctp.CtpExecutionChannel(driver, "20261009")
        self.assertEqual(channel.check().issues[0].code, "EXECUTION_DISCONNECTED")

    def test_binance_market_can_compose_with_ctp_execution(self):
        check = contracts.SessionReadiness(
            binance.BinanceMarketChannel(feed(), clock_ns=lambda: NOW, monotonic_ns=lambda: MONO),
            ctp.CtpExecutionChannel(NS(driver_id="ctp-hedge", trading_day="20261009"), "20261009"))
        self.assertTrue(check())
        self.assertEqual([(s.domain, s.channel) for s in check.last_result.sources],
                         [("market", "binance"), ("execution", "ctp")])

    def test_dolphin_stream_can_compose_with_binance_execution(self):
        identity = contracts.ChannelIdentity("market", "dolphindb", "ddb-stream")
        execution = NS(driver_id="bn-third-party", node=NS(is_running=lambda: True))
        check = contracts.SessionReadiness(
            health.StreamHealthCheck(feed(), identity, clock_ns=lambda: NOW, monotonic_ns=lambda: MONO),
            binance.BinanceExecutionChannel(execution, account_ready=lambda: True))
        self.assertTrue(check())
        self.assertEqual(check.last_result.sources[0].source_id, "ddb-stream")

    def test_binance_rejects_online_replay_profile(self):
        with self.assertRaisesRegex(ValueError, "只支持realtime"):
            binance.BinanceMarketChannel(feed(), profile=profiles.MarketTimeProfile("replay"))

    def test_binance_account_gate_independent_from_market(self):
        running, reconciled = [True], [False]
        driver = NS(driver_id="bn-exec", node=NS(is_running=lambda: running[0]))
        channel = binance.BinanceExecutionChannel(driver, account_ready=lambda: reconciled[0])
        self.assertEqual(channel.check().issues[0].code, "ACCOUNT_NOT_READY")
        reconciled[0] = True
        self.assertTrue(channel.check())
        running[0] = False
        self.assertEqual(channel.check().issues[0].code, "EXECUTION_DISCONNECTED")
        vendor = binance.BinanceExecutionChannel(NS(driver_id="bn-vendor"),
            connection_ready=lambda: True, account_ready=lambda: True)
        self.assertTrue(vendor.check())
        self.assertEqual(vendor.identity.source_id, "bn-vendor")

    def test_stream_event_age_and_receive_age_have_different_semantics(self):
        upstream = feed()
        upstream.health_snapshot.last_event_ts -= 2 * 86_400_000_000_000
        identity = contracts.ChannelIdentity("market", "dolphindb", "historical-stream")
        realtime = health.StreamHealthCheck(upstream, identity, clock_ns=lambda: NOW,
                                            monotonic_ns=lambda: MONO)
        replay = health.StreamHealthCheck(upstream, identity, clock_ns=lambda: NOW,
            monotonic_ns=lambda: MONO, profile=profiles.MarketTimeProfile("replay"))
        self.assertEqual(realtime.check().issues[0].code, "EVENT_STALE")
        self.assertTrue(replay.check())
        upstream.latest_receive_monotonic_ns -= 11_000_000_000
        self.assertEqual(replay.check().issues[0].code, "RECEIVE_STALE")

    def test_wall_receive_fallback_is_checked_when_no_monotonic_interface(self):
        upstream = feed()
        del upstream.latest_receive_monotonic_ns
        upstream.health_snapshot.last_event_received_ns -= 11_000_000_000
        channel = binance.BinanceMarketChannel(upstream, clock_ns=lambda: NOW)
        self.assertEqual(channel.check().issues[0].code, "RECEIVE_STALE")

    def test_future_missing_and_upstream_rollbacks_are_not_waived_by_replay(self):
        for stamp, code in ((None, "EVENT_MISSING"), (0, "EVENT_MISSING"),
                            (NOW + 1, "EVENT_FUTURE")):
            upstream = feed()
            upstream.health_snapshot.last_event_ts = stamp
            check = health.StreamHealthCheck(upstream,
                contracts.ChannelIdentity("market", "dolphindb", "replay"),
                clock_ns=lambda: NOW, monotonic_ns=lambda: MONO,
                profile=profiles.MarketTimeProfile("replay"))
            self.assertEqual(check.check().issues[0].code, code)
        upstream.health_snapshot.last_event_ts = NOW
        upstream.health_snapshot.state = "degraded"
        upstream.health_snapshot.reason = "timestamp_rollback"
        self.assertEqual(check.check().issues[0].code, "STREAM_UNHEALTHY")

    def test_ctp_replay_zero_source_time_is_missing(self):
        upstream = feed()
        upstream.depth_observations = {"AU": NS(ts_event=0, received_monotonic_ns=MONO,
            trading_day="20261008", timestamp_regressed=False)}
        gate = ctp.InstrumentHealthGate(upstream, ("AU",), "20261009",
            policy=ctp.CtpLiveProfile("replay", "20261008"),
            clock_ns=lambda: NOW, monotonic_ns=lambda: MONO)
        self.assertIn("AU", gate.stale_instruments())

    def test_channel_check_exception_is_closed_and_identified(self):
        channel = NS(identity=contracts.ChannelIdentity("execution", "vendor", "vendor-account"),
                     check=lambda: (_ for _ in ()).throw(RuntimeError("account lost")))
        result = contracts.SessionReadiness(channel).check()
        self.assertFalse(result)
        self.assertEqual((result.issues[0].code, result.issues[0].detail), ("CHECK_ERROR", "account lost"))

    def test_adapter_cannot_claim_another_source(self):
        a = contracts.ChannelIdentity("market", "ctp", "md-a")
        b = contracts.ChannelIdentity("market", "binance", "md-b")
        adapter = NS(identity=a, check=lambda: contracts.ReadinessResult((b,)))
        self.assertEqual(contracts.SessionReadiness(adapter).check().issues[0].code, "CHECK_ERROR")

    def test_empty_and_duplicate_sessions_rejected(self):
        channel = ctp.CtpMarketChannel(feed(), "20261008")
        for channels in ((), (channel, channel)):
            with self.assertRaises(ValueError):
                contracts.SessionReadiness(*channels)
        self.assertFalse(contracts.ReadinessResult(()))

    def test_trading_window_is_injected_and_has_its_own_failure_source(self):
        opened = [False]
        channel = ctp.CtpMarketChannel(feed(), "20261008", monotonic_ns=lambda: MONO)
        profile = profiles.LiveProfile(session_window=lambda stamp: opened[0] and stamp == NOW)
        check = contracts.SessionReadiness(channel, profile=profile, clock_ns=lambda: NOW)
        self.assertEqual(check.check().issues[0].code, "OUTSIDE_SESSION")
        opened[0] = True
        self.assertTrue(check())

    def test_trading_window_error_also_fails_closed(self):
        channel = ctp.CtpMarketChannel(feed(), "20261008", monotonic_ns=lambda: MONO)
        profile = profiles.LiveProfile(session_window=lambda stamp: 1 / 0)
        result = contracts.SessionReadiness(channel, profile=profile).check()
        self.assertFalse(result)
        self.assertEqual((result.issues[0].source.domain, result.issues[0].code),
                         ("runtime", "CHECK_ERROR"))

    def test_replay_aggregation_uses_declared_receive_age_and_clock(self):
        with patch.dict("sys.modules", {"bomber.framework.market.stream":
                NS(TradeTickBarFeed=NS, ReceiveTimeTradeTickBarFeed=lambda *args, **kwargs: kwargs)}):
            clock = lambda: NOW
            result = profiles.build_minute_feed("stream", feed(), profiles.MarketTimeProfile("replay"),
                                                max_age_seconds=30, clock_ns=clock)
        self.assertEqual(result["max_receive_age_ns"], 30_000_000_000)
        self.assertIs(result["clock_ns"], clock)

    def test_old_imports_are_aliases_to_single_implementation(self):
        old = import_module("_test_live_runtime.ctp")
        self.assertIs(old.CtpSessionLifecycle, ctp.CtpSessionLifecycle)
        policy = import_module("_test_live_runtime.live_policy")
        self.assertIs(policy.LiveTimePolicy, ctp.CtpLiveProfile)

    def test_common_live_modules_do_not_import_channels_or_demos(self):
        for path in ("__init__.py", "contracts.py", "profiles.py", "health.py", "runtime.py"):
            tree = ast.parse((ROOT / "bomber/framework/trader/runtime/live" / path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("channels", node.module or "", path)
                    self.assertFalse((node.module or "").startswith("demos"), path)


class RuntimeTests(unittest.TestCase):
    """仅替换原生模块依赖的RuntimeMode枚举，实际公共Runtime代码正常执行。"""
    @classmethod
    def setUpClass(cls):
        mode = Enum("RuntimeMode", {"LIVE": "live", "HISTORICAL": "historical"}, type=str)
        with patch.dict("sys.modules", {"bomber.framework.trader.contracts": NS(RuntimeMode=mode)}):
            cls.runtime = load("_test_live_runtime.live.runtime", "bomber/framework/trader/runtime/live/runtime.py")
        sys.modules[cls.runtime.__name__] = cls.runtime

    def build(self, failed_stage=None):
        trace, results = [], []
        def step(stage):
            trace.append(stage)
            if stage == failed_stage:
                raise RuntimeError(stage)
        def prepare(resources):
            resources.callback(lambda: trace.append("release"))
            step("prepare")
            return contracts.LiveRunContext(None, resources)
        controller = NS(event_label="test", prepare=prepare,
            finish_preparation=lambda context: step("finish"),
            start=lambda session, context: step("start"),
            poll=lambda session: step("poll") or False,
            verify=lambda session: step("verify"),
            shutdown=lambda session, context: step("shutdown") or {"cleanup_errors": []},
            snapshot=lambda session, context: {}, events=lambda session: ())
        identity = contracts.ChannelIdentity("market", "file", "offline-live-test")
        readiness = contracts.SessionReadiness(NS(identity=identity,
            check=lambda: contracts.ReadinessResult((identity,))))
        readiness.check()
        report = NS(path=None)
        report.begin = lambda: setattr(report, "path", "memory")
        def write(**kwargs):
            results.append(kwargs)
            return kwargs
        report.write = write
        def inputs(context):
            step("inputs")
            return NS()
        def assemble(inputs, context):
            step("assemble")
            return NS(runner=NS(mode=self.runtime.RuntimeMode.LIVE), readiness=readiness)
        runtime = self.runtime.ManagedLiveRuntime("test", controller=controller,
            prepare_inputs=inputs, assemble_session=assemble, report=report, seconds=1,
            monotonic=lambda: 0)
        return runtime, trace, results

    def test_success_releases_resources_once_and_reports_channel_provenance(self):
        runtime, trace, results = self.build()
        result = runtime.run()
        runtime.stop()
        self.assertEqual(trace, ["prepare", "inputs", "finish", "assemble", "start", "poll", "verify", "shutdown", "release"])
        self.assertEqual(len(results), 1)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["fields"]["session_readiness"]["sources"][0]["channel"], "file")
        legacy = import_module("_test_live_runtime.managed")
        self.assertIs(legacy.ManagedLiveRuntime, self.runtime.ManagedLiveRuntime)

    def test_prepare_assembly_start_and_poll_errors_all_release_resources(self):
        for stage in ("prepare", "inputs", "assemble", "start", "poll"):
            with self.subTest(stage=stage):
                runtime, trace, results = self.build(stage)
                with self.assertRaisesRegex(RuntimeError, stage):
                    runtime.run()
                runtime.stop()
                self.assertEqual(trace.count("release"), 1)
                self.assertEqual(trace.count("shutdown"), 1)
                self.assertEqual(results[0]["status"], "failed")

    def test_direct_live_runtime_still_implements_start_stop_without_selecting_channels(self):
        trace = []
        runner = NS(mode=self.runtime.RuntimeMode.LIVE,
                    start=lambda: trace.append("start"), stop=lambda: trace.append("stop"))
        direct = self.runtime.DirectLiveRuntime("direct", runner)
        direct.start()
        direct.stop()
        self.assertEqual(trace, ["start", "stop"])


class AggregationTests(unittest.TestCase):
    """执行真实聚合/时间适配代码；只替换本机不可加载的原生实体与Feed基类。"""
    @classmethod
    def setUpClass(cls):
        class Base:
            def __init__(self, source_id):
                self.source_id = source_id
                self._subscriptions = {"AU"}
            def get_instrument_meta(self, instrument):
                return NS() if instrument == "AU" else None
            def _emit_bar(self, bar):
                self.trace.append(("bar", bar.ts_event))
        class Number:
            def __init__(self, value):
                self.value = Decimal(value)
            def as_decimal(self):
                return self.value
        def tick(**kwargs):
            kwargs.pop("meta", None)
            kwargs["price"], kwargs["size"] = Number(kwargs["price"]), Number(kwargs["size"])
            return NS(**kwargs)
        cls.make_tick = staticmethod(tick)
        base = NS(MarketDataFeed=Base, BarType=NS,
            DataType=NS(BAR="BAR", TRADE_TICK="TRADE_TICK", QUOTE_TICK="QUOTE_TICK"), InstrumentId=str,
            InstrumentMeta=NS, SubscriptionRequest=NS, QuoteTick=NS, TradeTick=NS,
            make_trade_tick=tick, make_bar=lambda **kwargs: NS(**kwargs))
        with patch.dict("sys.modules", {"bomber.framework.market.basic.base": base}):
            cls.module = load("_test_live_aggregation", "bomber/framework/market/stream/aggregation.py")

    def tick(self, event, received):
        return self.make_tick(instrument_id="AU", price="894.5", size="1", trade_id="t",
                              ts_event=event, ts_init=received, aggressor_side="BUYER")

    def test_replay_bar_and_execution_tick_share_receive_time_without_mutating_source(self):
        minute = 60_000_000_000
        received = NOW // minute * minute
        trace, callbacks = [], []
        def after(tick):
            trace.append(("tick", tick.ts_event))
            callbacks.append(tick)
        now = [received]
        aggregate = self.module.ReceiveTimeTradeTickBarFeed("bars", NS(),
            clock_ns=lambda: now[0], after_tick=after)
        aggregate.trace = trace
        source = self.tick(received - 2 * 86_400_000_000_000, received)
        aggregate._on_trade_tick(source)
        now[0] += minute
        aggregate._on_trade_tick(self.tick(source.ts_event + minute, now[0]))
        self.assertEqual(trace, [("tick", received), ("bar", received + minute - 1),
                                 ("tick", received + minute)])
        self.assertEqual(callbacks[0].aggressor_side, "BUYER")
        self.assertEqual(source.ts_event, received - 2 * 86_400_000_000_000)
        self.assertEqual(aggregate.timing_snapshot["first_source_event_ns"], source.ts_event)

    def test_realtime_keeps_source_event_for_bar_and_execution(self):
        trace = []
        aggregate = self.module.TradeTickBarFeed("bars", NS(),
            after_tick=lambda tick: trace.append(("tick", tick.ts_event)))
        aggregate.trace = trace
        start = NOW // 60_000_000_000 * 60_000_000_000
        aggregate._on_trade_tick(self.tick(start, start + 1))
        aggregate._on_trade_tick(self.tick(start + 60_000_000_000, start + 60_000_000_001))
        self.assertEqual(trace, [("tick", start), ("bar", start + 60_000_000_000 - 1),
                                 ("tick", start + 60_000_000_000)])

    def test_replay_receive_age_obeys_configured_threshold_and_rejects_regression(self):
        aggregate = self.module.ReceiveTimeTradeTickBarFeed("bars", NS(), clock_ns=lambda: NOW,
                                                          max_receive_age_ns=30_000_000_000)
        aggregate.trace = []
        aggregate._on_trade_tick(self.tick(NOW - 86_400_000_000_000, NOW - 20_000_000_000))
        for received in (NOW - 31_000_000_000, NOW + 1, NOW - 21_000_000_000):
            with self.subTest(received=received), self.assertRaisesRegex(RuntimeError, "接收时间"):
                aggregate._on_trade_tick(self.tick(NOW - 86_400_000_000_000, received))


if __name__ == "__main__":
    unittest.main(verbosity=2)
