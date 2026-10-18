"""原生Bar环境下DFS指数通道的分钟送达、期限及精度测试；数据库为夹具。"""
from datetime import datetime
from decimal import Decimal
from importlib import import_module
import unittest
from zoneinfo import ZoneInfo

from bomber.framework.dataprep.features import FeatureQueryCache
from bomber.framework.market.basic.base import DataType, InstrumentId, InstrumentMeta
from bomber.framework.market.frames import MinuteFrameBuffer
from bomber.framework.market.stream.feature_query import FeatureQueryBarFeed
from bomber.framework.trader.runtime.live.channels.features import FeatureQueryChannel
from tests.test_dolphindb_index_features import END, Session, row, index_bundle, M


class QueryFeedTests(unittest.TestCase):
    def setUp(self):
        self.now = END+1000
        self.source = Session()
        bundle = index_bundle()
        provider = bundle.provider("index.minute",
            session_factory=lambda **kwargs: self.source, clock_ns=lambda: self.now)
        day = datetime.fromtimestamp(END/1e9, ZoneInfo("Asia/Shanghai")).date()
        start = int(datetime.combine(day, datetime.min.time(), ZoneInfo("Asia/Shanghai")).timestamp())*1_000_000_000
        cache = FeatureQueryCache(provider, dataset_id="index.minute", schema_version=1, keys=("000852",),
            start_ns=start, end_ns=start+24*60*M-1, clock_ns=lambda: self.now)
        self.feed = FeatureQueryBarFeed(cache, key="000852", projection=bundle.binding("index.minute").bar_projection)
        self.addCleanup(self.feed.disconnect)
        self.instrument = InstrumentId.from_str("000852.XSHG")
        self.feed.register_instrument(InstrumentMeta(self.instrument, price_precision=4, size_precision=4))
        self.feed.subscribe(self.instrument, DataType.BAR, bar_spec="1-MINUTE")
        self.bars = []
        self.feed.register_bar_handler(self.bars.append)

    def test_bootstrap_does_not_replay_day_only_pending_minute_delivered_once(self):
        self.feed.connect()
        self.assertEqual(self.bars, [])
        self.assertTrue(FeatureQueryChannel(self.feed.cache, clock_ns=lambda: self.now).check())
        self.assertEqual(self.feed.poll((END,)), (END,))
        bar, = self.bars
        self.assertEqual(bar.ts_event, END)
        self.assertEqual(bar.close.as_decimal(), Decimal("6401.2"))
        self.assertEqual(bar.volume.as_decimal(), Decimal("1000.25"))
        self.assertEqual(self.feed.poll((END,)), ())
        self.assertEqual(len(self.bars), 1)
        self.assertEqual(self.feed.manifest["minute_lookups"][str(END)]["status"], "PROJECTED")

    def test_late_missing_row_can_arrive_within_deadline(self):
        self.source.rows = []
        self.feed.connect()
        self.assertFalse(FeatureQueryChannel(self.feed.cache, clock_ns=lambda: self.now).check())
        self.assertEqual(self.feed.poll((END,)), ())
        self.assertEqual(self.feed.manifest["minute_lookups"][str(END)]["status"], "MISSING")
        self.now += 300_000_000
        self.source.rows = [row()]
        self.assertEqual(self.feed.poll((END,)), (END,))

    def test_expired_pending_minute_rejected_by_public_buffer(self):
        buffer = MinuteFrameBuffer({"future": "future-bars", "index": "index-bars"}, grace_seconds=2)
        self.assertTrue(buffer.add("future", 6400, END, END, source_id="future-bars"))
        accepted = []
        self.feed.register_bar_handler(lambda bar: accepted.append(buffer.add("index",
            bar.close.as_decimal(), bar.ts_event, self.now, source_id="index-bars")))
        self.feed.connect()
        self.now = END+2_000_000_001
        self.feed.poll((END,))
        self.assertEqual(accepted, [False])
        self.assertEqual(buffer.late_count, 1)

    def test_revisions_update_cache_without_reopening_already_delivered_minute(self):
        self.feed.connect()
        self.feed.poll((END,))
        self.now += 300_000_000
        self.source.rows = [row(close=6401.3)]
        self.feed.poll((END,))
        self.assertEqual(len(self.bars), 1)
        self.assertEqual(self.feed.manifest["cached_records"], 2)

    def test_precision_loss_fails_explicitly(self):
        self.feed.register_instrument(InstrumentMeta(self.instrument, price_precision=0, size_precision=0))
        self.feed.connect()
        with self.assertRaisesRegex(ValueError, "精度"):
            self.feed.poll((END,))
        self.assertEqual(self.bars, [])
        self.assertFalse(FeatureQueryChannel(self.feed.cache, clock_ns=lambda: self.now).check())

    def test_stale_source_not_refreshed_into_fresh_bar(self):
        self.feed.connect()
        self.now = END+301_000_000_000
        self.assertEqual(self.feed.poll((END,)), ())
        self.assertEqual(self.bars, [])
        self.assertEqual(self.feed.manifest["latest_event_ns"], END)
        self.assertTrue(self.feed.manifest["latest_event_time"].endswith("+08:00"))
        self.assertEqual(self.feed.manifest["cached_slots"], 1)
        self.assertEqual(self.feed.manifest["minute_lookups"][str(END)]["status"], "SOURCE_STALE")
        self.feed.replay = True
        self.assertEqual(self.feed.poll((END,)), (END,))

    def test_subscription_validation_at_connection(self):
        self.feed.subscribe("000300.XSHG", DataType.BAR, bar_spec="1-MINUTE")
        with self.assertRaisesRegex(ValueError, "业务键"):
            self.feed.connect()

    def test_cli_dfs_selection_explicit_availability_default(self):
        entry = import_module("demos.08_option_vega.run_live")
        args = entry.parse_args(["--connect", "--expected-trading-day", "20261012", "--calendar-id",
            "china.cffex", "--index-source", "dolphindb-dfs"])
        binding = index_bundle().binding(args.index_dataset, args.index_schema_version)
        self.assertEqual(binding.source_config.table, "index_1min")
        self.assertEqual(binding.source_config.timestamp_label, "end")
        self.assertEqual(binding.source_config.availability, "observed")
        self.assertEqual(args.mode, "recording")

    def test_another_bar_dataset_renamed_fields_five_minutes_same_feed_and_channel(self):
        import json
        from bomber.framework.dataprep.features import DEFAULT_FEATURE_CONFIG, bundle_from_definition
        from bomber.framework.dataprep.sources.dolphindb import DolphinDbReferenceConfig
        definition = json.loads(DEFAULT_FEATURE_CONFIG.read_text(encoding="utf-8"))
        item = definition["datasets"][0]
        names = {"open": "opening", "high": "maximum", "low": "minimum", "close": "closing", "volume": "quantity"}
        item["schema"]["dataset_id"] = "market.five_minute"
        item["schema"]["frequency"]["period_ns"] = 5*M
        for field in item["schema"]["fields"]:
            field["name"] = names.get(field["name"], field["name"])
        for relation in item["schema"]["relations"]:
            for side in ("left", "right"):
                relation[side] = names.get(relation[side], relation[side])
        item["source"]["payload_columns"] = {names.get(k, k): v for k, v in item["source"]["payload_columns"].items()}
        item["bar_projection"] = {"bar_spec": "5-MINUTE", "fields": names}
        event = END//(5*M)*(5*M)
        class RenamedSession(Session):
            def run(self, script):
                result = super().run(script)
                return result if script.startswith("schema(") else result.rename(
                    columns={"p_"+old: "p_"+new for old, new in names.items()})
        source = RenamedSession([row(event=event, published=event+100, key="FIVE")])
        bundle = bundle_from_definition(definition, DolphinDbReferenceConfig("fixture", 8848, "user", "secret"))
        provider = bundle.provider("market.five_minute", session_factory=lambda **kwargs: source, clock_ns=lambda: self.now)
        cache = FeatureQueryCache(provider, dataset_id="market.five_minute", schema_version=1, keys=("FIVE",),
            start_ns=event, end_ns=event, clock_ns=lambda: self.now, source_id="five-minute-source")
        feed = FeatureQueryBarFeed(cache, key="FIVE", projection=bundle.binding("market.five_minute").bar_projection)
        self.addCleanup(feed.disconnect)
        instrument = InstrumentId.from_str("FIVE.XSHG")
        feed.register_instrument(InstrumentMeta(instrument, price_precision=4, size_precision=4))
        feed.subscribe(instrument, DataType.BAR, bar_spec="5-MINUTE")
        bars = []
        feed.register_bar_handler(bars.append)
        feed.connect()
        self.assertIs(type(feed), type(self.feed))
        self.assertTrue(FeatureQueryChannel(cache, clock_ns=lambda: self.now).check())
        feed.poll((event,))
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].ts_event, event)
        self.assertEqual(bars[0].close.as_decimal(), Decimal("6401.2"))


if __name__ == "__main__":
    unittest.main()
