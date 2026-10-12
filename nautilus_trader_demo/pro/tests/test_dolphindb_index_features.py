"""指数原表映射/可见性/修订及文件往返；不联网、不依赖原生Bar。"""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from bomber.framework.datahub import DataHub
from bomber.framework.datahub.features import FeatureError, FeatureHub, FeatureQuery, FeatureRequirement, FeatureUnavailable
from bomber.framework.dataprep.features import (DEFAULT_FEATURE_CONFIG, load_feature_bundle,
    DolphinDbRawFeatureProvider, FeatureRequest, FeatureRefreshService, FileFeatureProvider, export_features)
from bomber.framework.dataprep.features.base import FeatureSourceError
from bomber.framework.dataprep.sources.dolphindb import DolphinDbReferenceConfig

M = 60_000_000_000
END = 1_791_767_460_000_000_000 // M * M


def index_bundle(connection=None):
    connection = connection or DolphinDbReferenceConfig("fixture", 8848, "user", "fixture-secret")
    return load_feature_bundle(DEFAULT_FEATURE_CONFIG, connection)


def definitions(min_type="TIME", update_type="TIMESTAMP"):
    types = {"date": "DATE", "minTime": min_type, "Code": "SYMBOL", "updateTime": update_type}
    return pd.DataFrame([{"name": name, "typeString": types.get(name, "DOUBLE")} for name in
        ("minTime", "date", "Code", "open", "high", "low", "close", "prev_close", "volume",
         "turnover", "pct_change", "updateTime")])


def row(event=END, published=END+100, **changes):
    value = dict(key="000852", event_ns=event, published_ns=published, open=6400.1, high=6402.5,
        low=6398.3, close=6401.2, prev_close=6390, volume=1000.25, turnover=6401200.1, pct_change=.175)
    value.update(changes)
    return value


class Session:
    def __init__(self, rows=None, schema=None):
        self.rows = [row()] if rows is None else rows
        self.schema = definitions() if schema is None else schema
        self.scripts, self.connected, self.closed = [], False, False

    def connect(self, *args, **kwargs):
        self.connected = True
        return True

    def run(self, script):
        self.scripts.append(script)
        if script.startswith("schema("):
            return self.schema
        envelope = {"key", "event_ns", "published_ns"}
        return pd.DataFrame([{(name if name in envelope else "p_"+name): value for name, value in r.items()}
            for r in self.rows], columns=tuple(name if name in envelope else "p_"+name for name in row()))

    def close(self):
        self.closed = True


class IndexProviderTests(unittest.TestCase):
    def setUp(self):
        self.bundle = index_bundle()
        self.registry = self.bundle.registry
        self.now = END+1000
        self.session = Session()
        self.config = self.bundle.binding("index.minute").source_config

    def provider(self, **changes):
        if "timestamp_label" in changes:
            changes["event_time"] = replace(self.config.event_time, label=changes.pop("timestamp_label"))
        if "source_timezone" in changes:
            zone = changes.pop("source_timezone")
            changes["event_time"] = replace(changes.get("event_time", self.config.event_time), timezone=zone)
            changes["publication_time"] = replace(self.config.publication_time, timezone=zone)
        if changes.get("availability") == "update-time":
            changes["availability"] = "source-time"
        provider = DolphinDbRawFeatureProvider(replace(self.config, **changes), self.registry,
            session_factory=lambda **kwargs: self.session, clock_ns=lambda: self.now)
        self.addCleanup(provider.close)
        provider.open()
        return provider

    def request(self, *, start=END, end=END, asof=None, max_rows=10):
        return FeatureRequest("index.minute", 1, ("000852",), start, end,
            self.now if asof is None else asof, max_rows=max_rows)

    def snapshot(self, rows, asof):
        hub = FeatureHub(self.registry)
        hub.store.publish(rows)
        return hub.snapshot((FeatureRequirement("index", "index.minute", 1, "000852"),),
            FeatureQuery(END, asof)).latest("index")

    def test_end_label_date_time_and_actual_ohlc_preserved(self):
        provider = self.provider()
        record, = provider.read(self.request()).records
        self.assertEqual(record.ts_event, END)
        self.assertEqual(record.payload.close, Decimal("6401.2"))
        self.assertEqual(record.payload.volume, Decimal("1000.25"))
        sql = self.session.scripts[-1]
        self.assertIn("concatDateTime(date,minTime)", sql)
        self.assertIn("28800000000000l", sql)
        self.assertNotIn("+60000000000l", sql)
        self.assertNotIn("settlement", sql)
        self.assertIn('Code in ["000852"]', sql)

    def test_schema_minute_and_absolute_timestamp_explicit_conversion(self):
        self.session.schema = definitions("MINUTE", "NANOTIMESTAMP")
        provider = self.provider()
        provider.read(self.request())
        self.assertIn("concatDateTime(date,time(minTime))", self.session.scripts[-1])
        self.session.schema = definitions("TIMESTAMP")
        provider.close()
        provider.open()
        provider.read(self.request())
        self.assertIn("nanotimestamp(minTime)", self.session.scripts[-1])

    def test_start_label_and_utc_configuration(self):
        provider = self.provider(timestamp_label="start", source_timezone="UTC")
        provider.read(self.request())
        self.assertIn("+60000000000l", self.session.scripts[-1])
        self.assertNotIn("28800000000000l", self.session.scripts[-1])

    def test_observed_not_visible_before_query_completion_and_stable_on_repeat(self):
        provider = self.provider()
        first, = provider.read(self.request(asof=END)).records
        self.assertEqual(first.available_ns, self.now)
        self.assertEqual(first.quality_flags, 2)
        with self.assertRaises(FeatureUnavailable):
            self.snapshot((first,), self.now-1)
        self.assertEqual(self.snapshot((first,), self.now), first)
        self.now += 1000
        repeated, = provider.read(self.request()).records
        self.assertEqual(first, repeated)

    def test_observed_changed_content_has_new_revision_without_backdating(self):
        provider = self.provider()
        first, = provider.read(self.request()).records
        self.now += 1000
        self.session.rows = [row(close=6401.3)]
        revised, = provider.read(self.request()).records
        self.assertGreater(revised.revision, first.revision)
        self.assertEqual(self.snapshot((first, revised), first.available_ns), first)
        self.assertEqual(self.snapshot((first, revised), revised.available_ns), revised)

    def test_observed_same_clock_change_rejected(self):
        provider = self.provider()
        provider.read(self.request())
        self.session.rows = [row(close=6401.3)]
        with self.assertRaisesRegex(FeatureError, "严格递增"):
            provider.read(self.request())

    def test_conflicting_observed_batch_is_atomic(self):
        provider = self.provider()
        self.session.rows = [row(), row(close=6401.3)]
        with self.assertRaisesRegex(FeatureError, "冲突快照"):
            provider.read(self.request())
        self.assertEqual(provider._observed, {})

    def test_explicit_publication_controls_visibility_and_revisions(self):
        self.session.rows = [row(published=END+100), row(published=END+200, close=6401.3)]
        provider = self.provider(availability="update-time")
        records = provider.read(self.request()).records
        self.assertEqual(self.snapshot(records, END+150).payload.close, Decimal("6401.2"))
        self.assertEqual(self.snapshot(records, END+200).payload.close, Decimal("6401.3"))
        self.assertEqual(records[0].quality_flags, 0)
        self.assertIn(f"<={self.now}l", self.session.scripts[-1])

    def test_future_publication_or_publication_before_close_rejected(self):
        provider = self.provider(availability="update-time")
        for published in (self.now+1, END-1):
            self.session.rows = [row(published=published)]
            with self.assertRaises(FeatureError):
                provider.read(self.request())

    def test_source_schema_rejects_numeric_codes_and_unknown_time(self):
        for column, dtype in (("Code", "INT"), ("minTime", "STRING"), ("updateTime", "LONG")):
            self.session.schema = definitions()
            self.session.schema.loc[self.session.schema.name == column, "typeString"] = dtype
            with self.assertRaises(FeatureError):
                self.provider()
            self.assertTrue(self.session.closed)

    def test_rows_wrong_key_future_unaligned_or_invalid_prices_rejected(self):
        provider = self.provider()
        for value in (row(key=852), row(event=END+M), row(event=END+1),
                      row(high=6400), row(volume=-1), row(close=float("nan"))):
            self.session.rows = [value]
            with self.assertRaises(FeatureError):
                provider.read(self.request(end=END+M, asof=END+M))

    def test_query_limit_empty_and_incremental_late_revision(self):
        provider = self.provider()
        self.session.rows = [row(), row(event=END-M)]
        with self.assertRaises(FeatureSourceError):
            provider.read(self.request(start=END-M, max_rows=1))
        self.session.rows = []
        self.assertEqual(provider.read(self.request()).records, ())
        self.session.rows = [row()]
        first = provider.read_updates(self.request())
        self.now += 1000
        self.assertEqual(provider.read_updates(self.request(), first.cursor).records, ())
        self.session.rows = [row(close=6401.3)]
        self.now += 1000
        self.assertEqual(len(provider.read_updates(self.request(), first.cursor).records), 1)

    def test_refresh_records_query_completion_receipt(self):
        provider = self.provider()
        hub = DataHub(features=FeatureHub(self.registry))
        service = FeatureRefreshService(hub, {("index.minute", 1): provider})
        clock = iter((self.now-1, self.now+1, self.now+2))
        service.refresh((self.request(asof=self.now-1),), clock_ns=lambda: next(clock))
        record = hub.feature_snapshot((FeatureRequirement("i", "index.minute", 1, "000852"),),
            FeatureQuery(END, self.now+2)).latest("i")
        self.assertEqual(record.available_ns, self.now)
        self.assertEqual(record.received_ns, self.now+1)

    def test_standard_file_preserves_prices_versions_and_visibility(self):
        provider = self.provider()
        first, = provider.read(self.request()).records
        self.now += 1000
        self.session.rows = [row(close=6401.3)]
        second, = provider.read(self.request()).records
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"index.csv"
            export_features(path, self.registry, (first, second))
            with FileFeatureProvider(path, self.registry) as file:
                self.assertEqual(file.read(self.request()).records, (first, second))
                self.assertEqual(file.read(self.request(asof=first.available_ns)).records, (first,))

    def test_query_errors_sanitized_and_no_connection_on_construction(self):
        def fail_factory(**kwargs):
            raise RuntimeError("fixture-secret")
        provider = DolphinDbRawFeatureProvider(self.config, self.registry, session_factory=fail_factory)
        with self.assertRaises(FeatureSourceError) as error:
            provider.open()
        self.assertNotIn("fixture-secret", str(error.exception))
        with self.assertRaises(FeatureError):
            replace(self.config, table='index_1min;delete from t')

    def test_nullable_auxiliary_fields_preserved_in_parquet(self):
        provider = self.provider()
        self.session.rows = [row(prev_close=None, pct_change=None)]
        batch = provider.read(self.request())
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"index.parquet"
            export_features(path, self.registry, batch.records)
            with FileFeatureProvider(path, self.registry) as file:
                self.assertEqual(file.read(self.request()).records, batch.records)

    def test_null_source_update_time_only_allowed_with_observation_evidence(self):
        self.session.rows = [row(published=None)]
        provider = self.provider()
        record, = provider.read(self.request()).records
        self.assertEqual(record.available_ns, self.now)
        explicit = self.provider(availability="update-time")
        with self.assertRaisesRegex(FeatureError, "空发布时间"):
            explicit.read(self.request())


if __name__ == "__main__":
    unittest.main()
