"""08的时间、来源、归属与信号恢复离线验证，不连接任何服务。"""
from dataclasses import replace
from datetime import date
from decimal import Decimal
from importlib import import_module
from pathlib import Path
import sys
import types
import unittest

from tests.test_demo_option_vega import ns
fixtures = import_module("tests.test_demo_option_vega")
from bomber.framework.market.frames import MinuteFrameBuffer, minute_close_ns, MINUTE_NS
from bomber.framework.dataprep.live_options import LiveOptionReferences
from bomber.framework.dataprep.sources.base import ReferenceBatch, ReferenceDataset
checkpoint = import_module("demos.08_option_vega.checkpoint")


class FrameTests(unittest.TestCase):
    def test_dual_time_and_delayed_sources_do_not_leak_next_minute(self):
        frame_ns, wall = ns("2026-08-13 13:58"), ns("2026-10-11 17:00")
        buffer = MinuteFrameBuffer({"option": "ctp", "index": "dolphin"}, grace_seconds=2)
        buffer.add("option", 20, frame_ns, wall, source_id="ctp")
        buffer.add("option", 22, frame_ns + MINUTE_NS, wall + 1, source_id="ctp")
        buffer.add("index", 6400, frame_ns, wall + 1_000_000_000, source_id="dolphin")
        self.assertEqual(buffer.close_due(wall + 1_000_000_000), ())
        frame, = buffer.close_due(wall + 2_000_000_000)
        self.assertEqual(frame.valuation_ns, frame_ns)
        self.assertEqual(frame.request_ns, wall + 2_000_000_000)
        self.assertEqual(frame.observations["option"].price, 20)
        self.assertEqual(frame.observations["index"].source_id, "dolphin")
        self.assertIn(frame_ns + MINUTE_NS, buffer.pending)

    def test_late_candidate_never_changes_closed_decision(self):
        buffer = MinuteFrameBuffer({"option": "ctp", "index": "dolphin"}, grace_seconds=0)
        buffer.add("index", 6400, MINUTE_NS, 1, source_id="dolphin")
        frame, = buffer.close_due(1)
        self.assertNotIn("option", frame.observations)
        self.assertFalse(buffer.add("option", 20, MINUTE_NS, 2, source_id="ctp"))
        self.assertEqual(buffer.late_count, 1)

    def test_sources_conflicts_and_timestamp_labels(self):
        self.assertEqual(minute_close_ns(MINUTE_NS - 1, "inclusive-end"), MINUTE_NS)
        self.assertEqual(minute_close_ns(MINUTE_NS, "end"), MINUTE_NS)
        self.assertEqual(minute_close_ns(0, "start"), MINUTE_NS)
        with self.assertRaises(ValueError):
            minute_close_ns(1, "end")
        b = MinuteFrameBuffer({"x": "ctp"})
        with self.assertRaises(ValueError):
            b.add("x", 1, MINUTE_NS, 1, source_id="dolphin")
        b.add("x", 1, MINUTE_NS, 1, source_id="ctp")
        with self.assertRaises(ValueError):
            b.add("x", 2, MINUTE_NS, 2, source_id="ctp")


# 使用现有夹具，但不重复运行其父类的用例。
class PositionAndRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.StrategyTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.StrategyTests()
        self.fixture.setUp()
        self.s, self.ctx = self.fixture.strategy, self.fixture.ctx

    def frame(self, clock):
        self.fixture.frame(clock)

    def test_unrelated_physical_position_never_enters_hedge_or_diagnostics(self):
        self.ctx.account_position = lambda key: (_ for _ in ()).throw(AssertionError("读取了物理仓位"))
        self.ctx.position = lambda key: self.ctx.actual.get(key, Decimal(0))
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        self.assertLess(self.ctx.intents[-1].targets[self.fixture.ids[self.fixture.info.symbol]], 0)
        self.ctx.actual[self.fixture.ids[self.fixture.info.symbol]] = Decimal(-2)
        self.frame("2026-08-13 14:00")
        self.assertEqual(self.s.entry_decisions, 1)

    def test_strategy_pending_does_not_read_other_strategy_orders(self):
        self.ctx.working_quantity = lambda key: (_ for _ in ()).throw(AssertionError("读取了账户在途"))
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        self.assertTrue(self.ctx.intents)

    def test_restart_clears_cached_iv_and_does_not_repeat_daily_entry(self):
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        state = checkpoint.OptionVegaCheckpoint(self.s).snapshot_state()
        original_revision = self.ctx.intents[-1].revision
        self.assertTrue(self.s.iv_cache)
        checkpoint.OptionVegaCheckpoint(self.s).restore_state(state)
        self.assertFalse(self.s.latest)
        self.assertFalse(self.s.iv_cache)
        self.assertIn(date(2026, 8, 13), self.s.sent_days)
        self.assertTrue(self.s.restored)
        self.assertEqual(self.s._revision, original_revision)
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        self.assertEqual(self.s.entry_decisions, 1)

    def test_frame_valuation_is_historical_and_request_is_wall_time(self):
        self.s.final_day = None
        self.frame("2026-08-13 13:58")  # 获取可信价格作为输入夹具，不触发决策。
        b = MinuteFrameBuffer({key: "test" for key in self.s.latest}, grace_seconds=0)
        stamp, wall = ns("2026-08-13 13:58"), ns("2026-10-11 18:00")
        for key, (_, price) in self.s.latest.items():
            b.add(key, price, stamp, wall, source_id="test")
        self.s.latest.clear()
        self.s.last_event_ns = -1
        self.s.on_market_frame(b.close_due(wall)[0])
        intent = self.ctx.intents[-1]
        self.assertEqual(intent.ts_event, wall)
        self.assertEqual(intent.metadata["signal_ts"], stamp)
        self.assertIn(date(2026, 8, 13), self.s.sent_days)
        self.assertNotIn(date(2026, 10, 11), self.s.sent_days)
        self.assertFalse(self.s.flatten_requested)

    def test_restore_rebases_unsent_open_goal_to_filled_position(self):
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        key = self.fixture.info.symbol
        self.ctx.actual[self.fixture.ids[key]] = Decimal(-1)
        state = checkpoint.OptionVegaCheckpoint(self.s).snapshot_state()
        checkpoint.OptionVegaCheckpoint(self.s).restore_state(state)
        b = MinuteFrameBuffer({key: "test" for key in (*self.s.instruments, self.s.index_id)}, grace_seconds=0)
        # 原始价格来自全新的估值分钟；旧目标即使超过实际成交也不被续发。
        self.fixture.frame("2026-08-13 14:01")
        for name, (_, price) in self.s.latest.items():
            b.add(name, price, ns("2026-08-13 14:01"), ns("2026-10-11 18:00"), source_id="test")
        self.s.last_event_ns = -1
        self.s.restored = True
        self.s.on_market_frame(b.close_due(ns("2026-10-11 18:00"))[0])
        self.assertEqual(self.s.option_goals[key], -1)
        self.assertEqual(self.s.entry_decisions, 1)

    def test_changed_contract_or_oversized_checkpoint_rejected(self):
        c = checkpoint.OptionVegaCheckpoint(self.s)
        state = c.snapshot_state()
        state["option_goals"]["UNKNOWN"] = "-1"
        with self.assertRaises(ValueError):
            c.restore_state(state)
        state = c.snapshot_state()
        state["option_goals"][self.fixture.info.symbol] = "-1000"
        with self.assertRaises(ValueError):
            c.restore_state(state)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.now = ns("2026-10-11 18:00")
        self.option = dict(symbol="MO2609-C-6600", contractType="CO", strikePrice=6600,
            contMultNum=100, varTicker="000852", exchangeCD="CFFEX", currencyCD="CNY", tickNum=.2,
            listDate=date(2026, 6, 1), lastTradeDate=date(2026, 9, 18), expDate=date(2026, 9, 18))
        self.future = dict(symbol="IM2609", code="IM", exchangeCD="CFFEX", currencyCD="CNY",
            contMultNum=200, minChgPriceNum=.2, listDate=date(2026, 6, 1), lastTradeDate=date(2026, 9, 18))

    def batch(self, dataset, row):
        return ReferenceBatch(dataset, (row,), tuple(row), "memory")

    def service(self):
        source = types.SimpleNamespace(options_basic=lambda q: self.batch(ReferenceDataset.OPTIONS_BASIC, self.option),
            futures_basic=lambda q: self.batch(ReferenceDataset.FUTURES_BASIC, self.future))
        return LiveOptionReferences(source, option_product="MO", future_product="IM", index_code="000852",
            valuation_day=date(2026, 8, 13), clock_ns=lambda: self.now)

    def test_historical_lifecycle_not_confused_with_observation_freshness(self):
        s = self.service()
        self.assertEqual(s.valuation_day, date(2026, 8, 13))
        self.assertIn("MO2609-C-6600", s.snapshot().options)
        self.now += 31_000_000_000
        with self.assertRaisesRegex(RuntimeError, "过期"):
            s.snapshot()

    def test_expired_manifest_remains_available_without_opening_gate(self):
        s = self.service()
        old = s.snapshot()
        self.now += 31_000_000_000
        manifest = s.manifest()
        self.assertFalse(manifest["ready"])
        self.assertEqual(manifest["observed_at_ns"], old.observed_at_ns)
        self.assertIn("过期", manifest["failure"])
        with self.assertRaisesRegex(RuntimeError, "过期"):
            s.snapshot()

    def test_changed_terms_fail_closed_and_future_publication_rejected(self):
        s = self.service()
        self.option["contMultNum"] = 200
        with self.assertRaisesRegex(RuntimeError, "版本变化"):
            s.refresh(force=True)
        with self.assertRaises(RuntimeError):
            s.snapshot()
        self.option["available_ns"] = self.now + 1
        with self.assertRaisesRegex(ValueError, "尚未发布"):
            self.service()


if __name__ == "__main__":
    unittest.main()
