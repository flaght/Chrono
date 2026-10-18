"""实际CTP深度回调与健康状态机；无SDK、网络和伪造派生成交。"""
import ast
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_depth_health_contract",
    ROOT / "bomber/framework/market/stream/health.py")
health = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = health
spec.loader.exec_module(health)
source = ast.parse((ROOT / "bomber/framework/market/stream/ctp/feed.py").read_text())
observation = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "CtpDepthObservation")
feed = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "CtpLiveDataFeed")
callback = next(n for n in feed.body if isinstance(n, ast.FunctionDef) and n.name == "on_depth_market_data")
namespace = dict(dataclass=dataclass, Mapping=Mapping, Any=Any, time=time, logger=logging.getLogger(__name__))
exec(compile(ast.Module([observation, callback], type_ignores=[]), "<actual CTP depth callback>", "exec"), namespace)
Handler = type("Handler", (), {"on_depth_market_data": namespace["on_depth_market_data"]})


class DepthHealthTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.monitor = health.StreamHealthMonitor("ctp", monotonic_ns=lambda: self.now, wall_time_ns=lambda: self.now)
        self.monitor.on_connected()
        self.received = []
        self.handler = Handler()
        self.handler._health_monitor = self.monitor
        self.handler._symbol_to_instrument = {"IM2610": "IM2610.CFFEX"}
        self.handler.get_instrument_meta = lambda i: SimpleNamespace(exchange="CFFEX")
        self.handler._converter = SimpleNamespace(convert=lambda *args: (),
            _timestamp_ns=lambda data, exchange: data["ts"])
        self.handler._lock = threading.RLock()
        self.handler._depth_observations = {}
        self.handler.enqueue_event = self.received.append
        self.handler._event_is_subscribed = lambda *args: True

    def depth(self, stamp=100):
        with patch.object(time, "monotonic_ns", return_value=self.now):
            self.handler.on_depth_market_data({"InstrumentID": "IM2610", "TradingDay": "20261012", "ts": stamp})

    def test_unchanged_depth_keeps_health_ready_without_derived_ticks_and_still_times_out(self):
        for seconds in range(0, 50, 5):
            self.now = seconds * 1_000_000_000
            self.depth()
            self.assertEqual(self.monitor.check_timeout().state, health.MarketHealthState.READY)
        self.assertEqual(self.received, [])
        self.assertEqual(self.handler._depth_observations["IM2610.CFFEX"].trading_day, "20261012")
        self.now += 31_000_000_000
        self.assertEqual(self.monitor.check_timeout().reason, health.MarketHealthReason.STALE)

    def test_raw_timestamp_rollback_remains_sticky(self):
        self.depth(100)
        self.depth(99)
        self.assertEqual(self.monitor.snapshot.reason, health.MarketHealthReason.TIMESTAMP_ROLLBACK)
        self.assertTrue(self.handler._depth_observations["IM2610.CFFEX"].timestamp_regressed)
        self.depth(101)
        self.assertEqual(self.monitor.snapshot.state, health.MarketHealthState.DEGRADED)
        self.assertEqual(self.received, [])

    def test_source_receipt_does_not_clear_dispatch_failure(self):
        self.monitor.on_dispatch_error(RuntimeError("broken handler"))
        self.depth()
        self.assertEqual(self.monitor.snapshot.reason, health.MarketHealthReason.DISPATCH_ERROR)

    def test_unregistered_or_invalid_depth_cannot_make_source_ready(self):
        with self.assertLogs(__name__, level="WARNING"):
            self.handler.on_depth_market_data({"InstrumentID": "UNKNOWN"})
        self.handler._converter.convert = lambda *args: (_ for _ in ()).throw(ValueError("invalid depth"))
        with self.assertLogs(__name__, level="ERROR"):
            self.depth()
        self.assertEqual(self.monitor.snapshot.reason, health.MarketHealthReason.WAITING_FOR_FIRST_EVENT)
        self.assertEqual(self.handler._depth_observations, {})


if __name__ == "__main__":
    unittest.main()
