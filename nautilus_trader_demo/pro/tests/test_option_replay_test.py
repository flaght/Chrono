"""08专用时间映射保留源值、统一偏移和来源健康；无网络。"""
from datetime import date
from importlib import import_module
from types import SimpleNamespace
import unittest

from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta, MarketDataFeed, make_bar
from tests.test_demo_option_vega import ns

module = import_module("demos.08_option_vega.replay_test")


class Feed(MarketDataFeed):
    def _on_subscription_added(self, request): pass
    def _on_subscription_removed(self, request): pass
    def connect(self): self._is_connected = True
    def disconnect(self): self._is_connected = False


class ReplayTestMappingTests(unittest.TestCase):
    def bar(self, clock="2026-10-08 16:29", symbol="IM2610.CFFEX"):
        identity = InstrumentId.from_str(symbol)
        return make_bar(instrument_id=identity, open="7080.6", high="7080.8", low="7080.4",
            close="7080.6", volume=3, ts_event=ns(clock)-1, ts_init=ns("2026-10-12 16:21"),
            meta=InstrumentMeta(identity, price_precision=1), bar_type="1-MINUTE")

    def test_explicit_16_to_09_changes_date_and_time_without_changing_source(self):
        mapping = module.ReplayTestTimeMap(date(2026, 10, 9), "09:00", "16:00")
        original = self.bar()
        result = mapping.map_bar(original)
        self.assertEqual(result.ts_event, ns("2026-10-09 09:29")-1)
        self.assertEqual(original.ts_event, ns("2026-10-08 16:29")-1)
        self.assertEqual(result.ts_init, original.ts_init)
        self.assertEqual(str(result.bar_type), str(original.bar_type))
        for name in ("open", "high", "low", "close", "volume"):
            self.assertEqual(getattr(result, name).as_decimal(), getattr(original, name).as_decimal())
        self.assertEqual(mapping.snapshot["source_day"], "2026-10-08")
        self.assertEqual(mapping.snapshot["samples"][0]["original_event_ns"], original.ts_event)

    def test_automatic_anchor_is_shared_by_all_contracts_and_keeps_minute_gaps(self):
        mapping = module.ReplayTestTimeMap(date(2026, 10, 9), "13:55")
        first = mapping.map_bar(self.bar())
        second = mapping.map_bar(self.bar("2026-10-08 16:32", "MO2610-C-7000.CFFEX"))
        self.assertEqual(first.ts_event, ns("2026-10-09 13:55")-1)
        self.assertEqual(second.ts_event, ns("2026-10-09 13:58")-1)
        self.assertEqual(mapping.snapshot["bars_mapped"], 2)

    def test_feed_reuses_health_and_emits_mapped_bar_for_last_and_mid(self):
        for price_type in ("last", "mid"):
            upstream = Feed("raw")
            upstream.health_snapshot = SimpleNamespace(marker="original")
            mapping = module.ReplayTestTimeMap(date(2026, 10, 9), "13:55")
            feed = module.mapped_bar_feed("mapped", upstream, price_type=price_type, time_map=mapping)
            delivered = []
            feed.register_bar_handler(delivered.append)
            original = self.bar()
            feed._emit_bar(original)
            self.assertIs(feed.health_snapshot, upstream.health_snapshot)
            self.assertEqual(delivered[0].ts_event, ns("2026-10-09 13:55")-1)
            self.assertEqual(original.ts_event, ns("2026-10-08 16:29")-1)

    def test_cross_day_and_uncompleted_bar_rejected_without_resetting_anchor(self):
        mapping = module.ReplayTestTimeMap(date(2026, 10, 9), "13:55")
        mapping.map_bar(self.bar())
        with self.assertRaisesRegex(ValueError, "跨日"):
            mapping.map_bar(self.bar("2026-10-09 16:29"))
        raw = self.bar()
        invalid = SimpleNamespace(ts_event=raw.ts_event-1)
        with self.assertRaisesRegex(ValueError, "已完成"):
            mapping.map_bar(invalid)
        self.assertEqual(mapping.snapshot["bars_mapped"], 1)

    def test_mapping_outside_valuation_day_rejected(self):
        mapping = module.ReplayTestTimeMap(date(2026, 10, 9), "23:59")
        mapping.map_bar(self.bar())
        with self.assertRaisesRegex(ValueError, "跨出估值日"):
            mapping.map_bar(self.bar("2026-10-08 16:31"))

    def test_clock_inputs_are_explicit_minutes(self):
        for value in ("9:00", "24:00", "13:55:01", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.parse_minute(value)


if __name__ == "__main__":
    unittest.main()
