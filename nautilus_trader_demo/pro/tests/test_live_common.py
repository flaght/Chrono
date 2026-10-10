"""公共Live机制无网络回归；按文件加载纯Python模块，不要求原生引擎。"""

import argparse
from importlib.util import spec_from_file_location, module_from_spec
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 用独立命名空间加载公共模块与旧导入兼容层，不导入依赖原生引擎的trader包。
namespace = ModuleType("_test_live_runtime")
namespace.__path__ = [str(ROOT / "bomber/framework/trader/runtime")]
sys.modules[namespace.__name__] = namespace
policy = load("_test_live_runtime.live_policy", "bomber/framework/trader/runtime/live_policy.py")
setup = load("_test_live_runtime.ctp_setup", "bomber/framework/trader/runtime/ctp_setup.py")
account = load("_test_ctp_account", "bomber/framework/trader/runtime/ctp_account.py")
cache_module = load("_test_live_cache", "bomber/framework/dataprep/live_cache.py")
NOW = 1_000_000_000_000_000
MONO = 1_000_000_000_000


class PolicyTests(unittest.TestCase):
    def test_default_and_replay_time_bases(self):
        tick = NS(ts_event=1, ts_init=2)
        real = policy.LiveTimePolicy()
        replay = policy.LiveTimePolicy("replay", "20261008")
        self.assertEqual((real.bar_time_basis, real.tick_timestamp(tick)), ("event", 1))
        self.assertEqual((replay.bar_time_basis, replay.tick_timestamp(tick)), ("receive", 2))
        self.assertEqual(replay.expected_md_day("20261009"), "20261008")

    def test_invalid_modes_dates_and_age_limits(self):
        for environment, day in (("other", None), ("replay", None), ("replay", "20261032"),
                                 ("replay", "2026108"), ("realtime", "20261008")):
            with self.subTest(environment=environment, day=day), self.assertRaises(ValueError):
                policy.LiveTimePolicy(environment, day)
        for value in (0, -1, 121, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.validate_instrument_max_age(value)

    def test_shared_cli(self):
        parser = argparse.ArgumentParser()
        policy.add_environment_arguments(parser)
        args = parser.parse_args(["--simnow-environment", "replay", "--replay-md-trading-day",
                                  "20261008", "--instrument-max-age-seconds", "30"])
        self.assertEqual(policy.validate_environment_arguments(args).bar_time_basis, "receive")
        self.assertEqual(parser.parse_args([]).instrument_max_age_seconds, 10)

    def test_replay_endpoint_pair_and_production(self):
        policy.validate_replay_environment("tcp://182.254.243.31:40011", "tcp://182.254.243.31:40001", True)
        for md, td, production in (("tcp://fake:40011", "tcp://182.254.243.31:40001", True),
                ("tcp://182.254.243.31:40011/path", "tcp://182.254.243.31:40001", True),
                ("tcp://182.254.243.31:40011", "tcp://182.254.243.31:30001", True),
                ("tcp://182.254.243.31:40011", "tcp://182.254.243.31:40001", False)):
            with self.subTest(md=md, td=td, production=production), self.assertRaises(ValueError):
                policy.validate_replay_environment(md, td, production)


class DepthTests(unittest.TestCase):
    def gate(self, *, replay=False):
        self.upstream = NS(depth_observations={i: NS(ts_event=NOW, received_monotonic_ns=MONO,
            trading_day="20261008", timestamp_regressed=False) for i in ("AU", "AG")})
        return policy.InstrumentHealthGate(self.upstream, ("AU", "AG"), "20261008",
            policy=policy.LiveTimePolicy("replay", "20261008") if replay else None,
            clock_ns=lambda: NOW, monotonic_ns=lambda: MONO)

    def test_all_contracts_required(self):
        gate = self.gate()
        self.assertFalse(gate.stale_instruments())
        del self.upstream.depth_observations["AG"]
        self.assertEqual(set(gate.stale_instruments()), {"AG"})
        self.upstream.depth_observations.clear()
        self.assertEqual(set(gate.stale_instruments()), {"AU", "AG"})

    def test_replay_accepts_old_event_but_real_rejects(self):
        for replay in (False, True):
            gate = self.gate(replay=replay)
            self.upstream.depth_observations["AU"].ts_event -= 172_352_000_000_000
            self.assertEqual(bool(gate.stale_instruments()), not replay)

    def test_both_environments_reject_wrong_day_regression_future_or_stale_receive(self):
        for replay in (False, True):
            for key, value in (("trading_day", "20261007"), ("timestamp_regressed", True),
                    ("ts_event", NOW + 1), ("received_monotonic_ns", MONO + 1),
                    ("received_monotonic_ns", MONO - 11_000_000_000)):
                with self.subTest(replay=replay, key=key, value=value):
                    gate = self.gate(replay=replay)
                    setattr(self.upstream.depth_observations["AU"], key, value)
                    self.assertEqual(set(gate.stale_instruments()), {"AU"})

    def test_non_ctp_feed_and_explicit_fallback(self):
        upstream = NS()
        gate = policy.InstrumentHealthGate(upstream, ("AU",), "20261008")
        self.assertFalse(gate.stale_instruments())
        gate = policy.InstrumentHealthGate(upstream, ("AU",), "20261008",
            clock_ns=lambda: NOW, monotonic_ns=lambda: MONO, fallback_events={}, fallback_receives={})
        self.assertEqual(set(gate.stale_instruments()), {"AU"})
        gate.fallback_events["AU"], gate.fallback_receives["AU"] = NOW, MONO
        self.assertFalse(gate.stale_instruments())


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = NOW
        self.cache = cache_module.LiveReferenceCache(clock_ns=lambda: self.now)
        self.assignment = NS(effective_ns=NOW, available_ns=NOW)
        self.reads = 0
        self.cache.refresh(self.loader)

    def loader(self):
        self.reads += 1
        return self.assignment, "spec", {"factor_date": "2026-09-30", "observed_at_ns": self.now,
            "freshness_policy": {"max_observation_age_seconds": 30}}

    def test_snapshot_reads_without_query_and_manifest_is_copied(self):
        for _ in range(5):
            self.assertIs(self.cache.snapshot(self.now), self.assignment)
        self.assertEqual(self.reads, 1)
        manifest = self.cache.manifest
        manifest["freshness_policy"]["max_observation_age_seconds"] = 500
        self.assertEqual(self.cache.manifest["freshness_policy"]["max_observation_age_seconds"], 30)

    def test_not_published_at_bar_time(self):
        with self.assertRaisesRegex(RuntimeError, "尚不可见"):
            self.cache.snapshot(NOW - 1)

    def test_expiry_and_successful_refresh(self):
        self.now += 31_000_000_000
        with self.assertRaisesRegex(RuntimeError, "过期"):
            self.cache.snapshot(self.now)
        self.cache.refresh(self.loader)
        self.assertIs(self.cache.snapshot(self.now), self.assignment)

    def test_failure_closes_access_until_successful_refresh(self):
        def failed():
            raise RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.cache.refresh(failed)
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.cache.snapshot(self.now)
        self.cache.refresh(self.loader)
        self.assertIs(self.cache.snapshot(self.now), self.assignment)

    def test_clock_regression_cannot_reuse_snapshot(self):
        self.cache.snapshot(self.now)
        self.now -= 1
        with self.assertRaisesRegex(RuntimeError, "时钟回退"):
            self.cache.snapshot(NOW)

    def test_slow_loader_does_not_hold_submit_lock_and_publication_waits(self):
        entered, release, loaded = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def slow():
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test loader timeout")
            loaded.set()
            return NS(effective_ns=NOW, available_ns=NOW), "new", self.loader()[2]
        def refresh():
            try:
                self.cache.refresh(slow)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=refresh)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.cache.publication_lock:
                self.assertIs(self.cache.snapshot(NOW), self.assignment)
                release.set()
                self.assertTrue(loaded.wait(1))
                self.assertEqual(self.cache.value[1], "spec")
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors)
            self.assertEqual(self.cache.value[1], "new")
        finally:
            release.set()
            worker.join(2)


class AccountTests(unittest.TestCase):
    def row(self, **kwargs):
        return {"InstrumentID": "au2612", "ExchangeID": "SHFE", "Position": "1",
                "HedgeFlag": "1", "PosiDirection": "2", "PositionDate": "1",
                "PositionCost": "1000", **kwargs}

    def test_today_yesterday_long_short_and_cost(self):
        rows = [self.row(PosiDirection=d, PositionDate=p) for d in ("2", "3") for p in ("1", "2")]
        result = account.position_buckets(rows, ("au2612.SHFE",), require_cost=True, unique_buckets=True)
        self.assertEqual(result["au2612.SHFE"].quantities, (1, 1, 1, 1))
        self.assertEqual(result["au2612.SHFE"].costs, (1000,) * 4)

    def test_duplicate_bucket_policy(self):
        rows = [self.row(), self.row()]
        self.assertEqual(account.position_buckets(rows, ("au2612.SHFE",))["au2612.SHFE"].quantities, (2, 0, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "重复"):
            account.position_buckets(rows, ("au2612.SHFE",), unique_buckets=True)

    def test_invalid_authoritative_rows(self):
        for change in ({"InstrumentID": "other"}, {"HedgeFlag": "2"}, {"PosiDirection": "1"},
                {"PositionDate": "3"}, {"Position": "NaN"}, {"Position": "Infinity"},
                {"Position": "-1"}, {"Position": "0.5"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                account.position_buckets([self.row(**change)], ("au2612.SHFE",))
        for cost in ("0", "-1", "NaN", "Infinity"):
            with self.subTest(cost=cost), self.assertRaisesRegex(RuntimeError, "成本"):
                account.position_buckets([self.row(PositionCost=cost)], ("au2612.SHFE",), require_cost=True)

    def test_zero_rows_do_not_introduce_unknown_position(self):
        self.assertEqual(account.position_buckets([self.row(Position="0", InstrumentID="other")], ()), {})


class SetupTests(unittest.TestCase):
    def test_minute_factory_uses_public_feed_and_keeps_callback(self):
        def real(*args, **kwargs):
            return NS(basis="event", args=args, kwargs=kwargs)
        def replay(*args, **kwargs):
            return NS(basis="receive", args=args, kwargs=kwargs)
        module = NS(TradeTickBarFeed=real, ReceiveTimeTradeTickBarFeed=replay)
        callback = lambda tick: None
        with patch.dict(sys.modules, {"bomber.framework.market.stream": module}):
            for value, basis in ((policy.LiveTimePolicy(), "event"),
                    (policy.LiveTimePolicy("replay", "20261008"), "receive")):
                feed = setup.build_minute_feed("test", "upstream", value, after_tick=callback)
                self.assertEqual(feed.basis, basis)
                self.assertIs(feed.kwargs["after_tick"], callback)

    def test_transport_factory_constructs_without_connecting(self):
        env = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo", "CTP_TD_ADDRESS": "tcp://fake:1",
               "CTP_PASSWORD": "test-password", "CTP_PRODUCTION_MODE": "true"}
        with patch.dict(os.environ, env, clear=True):
            result = setup.build_simnow_transport(NS, client_id="test", flow_path="/tmp/test", timeout_seconds=15)
            self.assertEqual((result.investor_id, result.flow_path, result.timeout_seconds), ("demo", "/tmp/test", 15))
            self.assertTrue(result.production_mode)
            config = setup.build_md_config(NS, NS(md_front="tcp://fake:2", transport=result), flow_path="/tmp/md")
            self.assertEqual((config.user_id, config.front, config.flow_path), ("demo", "tcp://fake:2", "/tmp/md"))

    def test_missing_credentials_and_wrong_broker(self):
        for env in ({}, {"CTP_BROKER_ID": "8888"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                setup.build_simnow_transport(NS, client_id="test", flow_path="/tmp/test", timeout_seconds=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
