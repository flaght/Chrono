"""GAP08无网络契约测试；运行于用户Linux环境，不加载策略或交易扩展。"""
from dataclasses import replace
from decimal import Decimal
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from bomber.framework.datahub import DataHub
from bomber.framework.datahub.features import (
    Alignment, DatasetRegistry, DatasetSpec, ExternalDataRecord, FeatureError,
    FeatureFrameAssembler, FeatureHub, FeatureQuery, FeatureRequirement, FeatureUnavailable,
    FrequencySpec, IndexPrice, MarketFrameInput, MinuteSignal, PayloadField, replay_feature_frames,
)
from bomber.framework.datahub.features.contracts import MINUTE_NS as M
from bomber.framework.dataprep.features import (
    DatasetTable, DolphinDbFeatureConfig, DolphinDbFeatureProvider, FeatureColumnMapping,
    FeatureProvider, FeatureProviderFactory, FeatureRefreshService, FeatureRequest, FeatureSourceError,
    FileFeatureProvider, MongoDbFeatureConfig, MongoDbFeatureProvider, decode_record, encode_record,
    export_features,
)
from bomber.framework.dataprep.features.base import select_records


def registry(frequency=None):
    return DatasetRegistry((
        DatasetSpec("minute.signal", 1, MinuteSignal, (PayloadField("score", "float"),),
            frequency=frequency or FrequencySpec()),
        DatasetSpec("index.price", 1, IndexPrice, (PayloadField("close", "decimal"),)),
    ))


def row(event=M, available=None, revision=0, score=1.0, **kwargs):
    return ExternalDataRecord("minute.signal", 1, "IM", MinuteSignal(score), event,
        event if available is None else available, event, revision, "fixture", **kwargs)


def requirement(mode=Alignment.EXACT, **kwargs):
    return FeatureRequirement("signal", "minute.signal", 1, "IM", mode, **kwargs)


def hub(rows=(), reg=None):
    features = FeatureHub(reg or registry())
    features.store.publish(rows)
    return DataHub(features=features)


class FeatureHubTests(unittest.TestCase):
    def test_exact_has_no_implicit_forward_fill(self):
        data = hub((row(),))
        with self.assertRaises(FeatureUnavailable) as error:
            data.feature_snapshot((requirement(),), FeatureQuery(2*M, 2*M))
        self.assertEqual(error.exception.code, "MISSING")
        self.assertEqual(data.feature_snapshot((requirement(Alignment.ASOF),),
            FeatureQuery(2*M, 2*M)).latest("signal").ts_event, M)

    def test_visible_revision_and_unpublished_business_slot(self):
        data = hub((row(), row(revision=1, available=3*M, score=2.0), row(2*M, available=4*M)))
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(M, 2*M)).latest("signal").payload.score, 1.0)
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(M, 3*M)).latest("signal").revision, 1)
        self.assertEqual(data.feature_snapshot((requirement(Alignment.ASOF),), FeatureQuery(2*M, 2*M)).latest("signal").ts_event, M)

    def test_future_business_time_hidden_even_after_publication(self):
        data = hub((row(), row(2*M)))
        self.assertEqual(data.feature_snapshot((requirement(Alignment.ASOF),), FeatureQuery(M, 3*M)).latest("signal").ts_event, M)

    def test_real_receive_time_and_first_observation_preserved(self):
        data = hub((row(received_ns=2*M),))
        with self.assertRaises(FeatureUnavailable):
            data.feature_snapshot((requirement(),), FeatureQuery(M, M))
        data.features.store.publish((row(received_ns=3*M),))
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(M, 2*M)).latest("signal").received_ns, 2*M)

    def test_window_requires_each_business_slot(self):
        data = hub((row(M), row(3*M)))
        with self.assertRaises(FeatureError):
            data.feature_snapshot((requirement(Alignment.WINDOW, window_ns=3*M+1),), FeatureQuery(4*M, 4*M))
        demand = requirement(Alignment.WINDOW, window_ns=3*M)
        with self.assertRaises(FeatureUnavailable):
            data.feature_snapshot((demand,), FeatureQuery(3*M, 3*M))
        data.features.store.publish((row(2*M),))
        snap = data.feature_snapshot((demand,), FeatureQuery(3*M, 3*M))
        self.assertEqual(tuple(item.ts_event for item in snap.values["signal"]), (M, 2*M, 3*M))

    def test_window_freshness_checks_last_slot_not_entire_history(self):
        data = hub((row(M), row(2*M), row(3*M)))
        snap = data.feature_snapshot((requirement(Alignment.WINDOW, window_ns=3*M, max_source_age_ns=0),), FeatureQuery(3*M, 3*M))
        self.assertEqual(len(snap.values["signal"]), 3)
        with self.assertRaises(FeatureUnavailable) as error:
            data.feature_snapshot((requirement(max_source_age_ns=0),), FeatureQuery(3*M, 3*M+1))
        self.assertEqual(error.exception.code, "STALE")

    def test_calendar_slots_skip_breaks_and_holidays_without_fabrication(self):
        frequency = FrequencySpec(calendar_id="injected.cffex", slots=lambda start, end:
            tuple(slot for slot in (M, 2*M, 5*M) if start < slot <= end))
        data = hub((row(M), row(2*M), row(5*M)), registry(frequency))
        snap = data.feature_snapshot((requirement(Alignment.WINDOW, window_ns=5*M),), FeatureQuery(5*M, 5*M))
        self.assertEqual(len(snap.values["signal"]), 3)
        with self.assertRaises(FeatureError):
            FrequencySpec(calendar_id="exchange-without-slots")

    def test_frequency_normalizes_all_labels_and_supports_five_minutes(self):
        frequency = FrequencySpec(period_ns=5*M)
        self.assertEqual(frequency.close_ns(0, "start"), 5*M)
        self.assertEqual(frequency.close_ns(5*M-1, "inclusive-end"), 5*M)
        data = hub((row(5*M),), registry(frequency))
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(5*M, 5*M)).latest("signal").ts_event, 5*M)

    def test_batch_conflict_does_not_partially_publish(self):
        data = hub((row(),))
        generation, rows = data.features.store.capture()
        with self.assertRaises(FeatureError):
            data.features.store.publish((row(2*M), row(score=5.0)))
        self.assertEqual(data.features.store.capture(), (generation, rows))

    def test_quality_bad_latest_revision_does_not_fall_back(self):
        data = hub((row(), row(revision=1, available=2*M, quality_flags=1)))
        with self.assertRaises(FeatureUnavailable) as error:
            data.feature_snapshot((requirement(),), FeatureQuery(M, 2*M))
        self.assertEqual(error.exception.code, "QUALITY")

    def test_schema_nan_and_revision_order_are_rejected(self):
        data = hub()
        with self.assertRaises(FeatureError):
            IndexPrice(Decimal("-1"))
        for invalid in (row(score=float("nan")), replace(row(), schema_version=2),
                        replace(row(), payload=IndexPrice(Decimal("1")))):
            with self.assertRaises(FeatureError):
                data.features.store.publish((invalid,))
        with self.assertRaises(FeatureError):
            data.features.store.publish((row(available=2*M), row(available=M, revision=1)))

    def test_snapshot_is_immutable_and_pinned_across_publication(self):
        data = hub((row(),))
        snapshot = data.feature_snapshot((requirement(),), FeatureQuery(M, 2*M))
        with self.assertRaises(TypeError):
            snapshot.values["signal"] = ()
        data.features.store.publish((row(available=2*M, revision=1, score=9.0),))
        self.assertEqual(snapshot.latest("signal").payload.score, 1.0)

    def test_index_identity_transform_and_leading_zero_key(self):
        record = ExternalDataRecord("index.price", 1, "000852", IndexPrice(Decimal("6400.1250")), M, M, M, 0, "index")
        data = hub((record,))
        snap = data.feature_snapshot((FeatureRequirement("index", "index.price", 1, "000852"),), FeatureQuery(M, M))
        self.assertEqual(snap.latest("index").payload.close, Decimal("6400.1250"))

    def test_daily_spot_same_day_publication_and_revision_asof(self):
        from bomber.framework.dataprep.features.samples import sample_registry
        from bomber.framework.datahub.features import CommoditySpot
        reg = sample_registry()
        day_start = 16*60*M
        payload = CommoditySpot(Decimal("3500"), "CNY", "tonne", "Shanghai", "grade3")
        first = ExternalDataRecord("commodity.spot.daily", 1, "RB.SHANGHAI.GRADE3", payload,
            day_start, day_start+11*60*M, day_start+11*60*M, 0, "spot.fixture")
        second = replace(first, revision=1, available_ns=day_start+12*60*M,
            payload=replace(payload, price=Decimal("3510")))
        data = hub((first, second), reg)
        demand = FeatureRequirement("spot", "commodity.spot.daily", 1, "RB.SHANGHAI.GRADE3")
        with self.assertRaises(FeatureUnavailable):
            data.feature_snapshot((demand,), FeatureQuery(day_start+10*60*M, day_start+10*60*M))
        snap = data.feature_snapshot((demand,), FeatureQuery(day_start+11*60*M, day_start+11*60*M))
        self.assertEqual(snap.latest("spot").payload.price, Decimal("3500"))

    def test_default_schema_and_payload_unknown_fields_fail_explicitly(self):
        encoded = encode_record(row())
        encoded["payload.extra"] = "unexpected"
        with self.assertRaises(FeatureError): decode_record(encoded, registry())
        reg = registry()
        with self.assertRaises(FeatureError): reg.register(reg.get("minute.signal", 1))


class FrameTests(unittest.TestCase):
    def test_wait_then_ready_at_actual_visibility_once(self):
        data = hub((row(available=M+200),))
        assembler = FeatureFrameAssembler(data, (requirement(),), wait_ns=500)
        assembler.offer("bar-1", M, M+50, {"close": 6400})
        self.assertEqual(assembler.poll(M+100), ())
        outcome, = assembler.poll(M+200)
        self.assertEqual((outcome.status, outcome.frame.decision_ns), ("READY", M+200))
        self.assertEqual(assembler.poll(M+300), ())

    def test_timeout_and_delayed_poll_never_emit_old_goal(self):
        for rows in ((row(available=M+600),), (row(),)):
            assembler = FeatureFrameAssembler(hub(rows), (requirement(),), wait_ns=500)
            assembler.offer("one", M, M, {"close": 1})
            outcome, = assembler.poll(M+501)
            self.assertEqual(outcome.status, "SKIPPED")
            self.assertEqual(assembler.poll(M+700), ())

    def test_deadline_boundary_ready_and_missing_skip(self):
        for records, status in (((row(available=M+500),), "READY"), ((), "SKIPPED")):
            assembler = FeatureFrameAssembler(hub(records), (requirement(),), wait_ns=500)
            assembler.offer("one", M, M, {"close": 1})
            self.assertEqual(assembler.poll(M+500)[0].status, status)

    def test_replay_and_live_use_same_clock_contract(self):
        records = (row(available=M+200), row(2*M, available=2*M+600))
        market = tuple(MarketFrameInput(str(event), event, event+50, {"close": 1}) for event in (M, 2*M))
        outcomes = replay_feature_frames(hub(), (requirement(),), market, records, wait_ns=500)
        self.assertEqual(tuple(item.status for item in outcomes), ("READY", "SKIPPED"))
        live = hub()
        assembler = FeatureFrameAssembler(live, (requirement(),), wait_ns=500)
        assembler.offer(str(M), M, M+50, {"close": 1})
        live.features.store.publish((replace(records[0], received_ns=M+200),))
        result = assembler.poll(M+200)[0]
        self.assertEqual(result.frame.features.latest("signal").payload,
            outcomes[0].frame.features.latest("signal").payload)
        self.assertEqual(result.frame.decision_ns, outcomes[0].frame.decision_ns)

    def test_clock_rollback_and_duplicate_frame_rejected(self):
        assembler = FeatureFrameAssembler(hub(), (requirement(),))
        assembler.offer("one", M, M, {"close": 1})
        with self.assertRaises(FeatureError):
            assembler.offer("two", M, M, {"close": 1})
        assembler.poll(M+1)
        with self.assertRaises(FeatureError):
            assembler.poll(M)

    def test_legacy_market_buffer_rejects_expired_not_yet_flushed_row(self):
        from bomber.framework.market.frames import MinuteFrameBuffer
        buffer = MinuteFrameBuffer({"opt": "ctp", "index": "ddb"}, grace_seconds=2)
        buffer.add("opt", 20, M, M, source_id="ctp")
        self.assertFalse(buffer.add("index", 6400, M, M+2_000_000_001, source_id="ddb"))
        self.assertNotIn("index", buffer.close_due(M+3_000_000_000)[0].observations)


class FileTests(unittest.TestCase):
    def test_csv_roundtrip_preserves_revisions_and_publication(self):
        with TemporaryDirectory() as directory:
            reg = registry()
            path = Path(directory) / "features.csv"
            records = (row(), row(available=2*M, revision=1, score=5.0))
            export_features(path, reg, records)
            with FileFeatureProvider(path, reg) as provider:
                batch = provider.read(FeatureRequest("minute.signal", 1, ("IM",), 0, M, 2*M))
                self.assertEqual(batch.records, records)
            with self.assertRaises(FeatureError):
                export_features(path, reg, records)

    def test_parquet_roundtrip_preserves_large_nanoseconds_and_null_receipt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            event = 1_790_000_000_000_000_000 // M * M
            records = (row(event, available=event+123), row(event+M, received_ns=event+M+456))
            reg = registry()
            export_features(path, reg, records)
            with FileFeatureProvider(path, reg) as provider:
                batch = provider.read(FeatureRequest("minute.signal", 1, ("IM",), event, event+M, event+2*M))
                self.assertEqual(batch.records, records)

    def test_csv_tamper_and_schema_change_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            reg = registry()
            export_features(path, reg, (row(),))
            path.write_text(path.read_text().replace("1.0", "2.0"))
            with self.assertRaises(FeatureError):
                FileFeatureProvider(path, reg).open()

    def test_float_timestamp_and_missing_publication_rejected(self):
        for field, value in (("available_ns", float(M)), ("available_ns", None), ("payload.score", "nan")):
            encoded = encode_record(row())
            encoded[field] = value
            with self.assertRaises(FeatureError):
                decode_record(encoded, registry())

    def test_column_mapping_start_label_and_index_key(self):
        record = ExternalDataRecord("index.price", 1, "000852", IndexPrice(Decimal("6400")), M, M+1, M, 0, "index")
        encoded = encode_record(record)
        encoded["bar_start"] = encoded.pop("ts_event") - M
        encoded["Code"] = encoded.pop("key")
        decoded = decode_record(encoded, registry(), FeatureColumnMapping({"ts_event": "bar_start", "key": "Code"}, "start"))
        self.assertEqual(decoded, record)


class MemoryProvider(FeatureProvider):
    def __init__(self, reg):
        super().__init__(reg)
        self.rows = [row()]
        self.fail = False
    def open(self): pass
    def close(self): pass
    def read(self, request):
        if self.fail:
            raise FeatureSourceError("read failed")
        return select_records(self.registry, self.rows, request, "database:fixture")


class RefreshTests(unittest.TestCase):
    def test_slow_query_is_visible_only_after_completed_receive(self):
        reg = registry()
        data = hub(reg=reg)
        provider = MemoryProvider(reg)
        service = FeatureRefreshService(data, {("minute.signal", 1): provider})
        times = iter((M, M+100, M+100))
        service.refresh((FeatureRequest("minute.signal", 1, ("IM",), 0, M, M),), clock_ns=lambda: next(times))
        with self.assertRaises(FeatureUnavailable):
            data.feature_snapshot((requirement(),), FeatureQuery(M, M+50))
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(M, M+100)).latest("signal").received_ns, M+100)

    def test_old_slot_new_revision_and_late_insert_not_lost_by_cursor(self):
        reg = registry()
        provider = MemoryProvider(reg)
        request = FeatureRequest("minute.signal", 1, ("IM",), 0, M, 3*M)
        batch = provider.read_updates(request)
        self.assertEqual(len(batch.records), 1)
        provider.rows.append(row(available=2*M, revision=1, score=9.0))
        update = provider.read_updates(request, batch.cursor)
        self.assertEqual(update.records[0].revision, 1)
        provider.rows.append(replace(row(), key="OTHER"))
        self.assertEqual(provider.read_updates(request, update.cursor).records, ())

    def test_overwrite_and_cursor_source_scope_rollback_rejected(self):
        provider = MemoryProvider(registry())
        request = FeatureRequest("minute.signal", 1, ("IM",), 0, M, 3*M)
        cursor = provider.read_updates(request).cursor
        provider.rows = [row(score=9.0)]
        with self.assertRaises(FeatureError):
            provider.read_updates(request, cursor)
        provider.rows = [row()]
        for invalid in (replace(cursor, source_identity="other"), replace(cursor, as_of_ns=4*M)):
            with self.assertRaises(FeatureError):
                provider.read_updates(request, invalid)

    def test_poll_does_not_refresh_source_age_and_error_blocks_cache(self):
        reg = registry()
        data = hub(reg=reg)
        provider = MemoryProvider(reg)
        service = FeatureRefreshService(data, {("minute.signal", 1): provider}, max_observation_age_ns=M)
        request = FeatureRequest("minute.signal", 1, ("IM",), 0, M, M)
        service.refresh((request,), clock_ns=lambda: M)
        service.refresh((replace(request, as_of_ns=M+10),), clock_ns=lambda: M+10)
        snap = data.feature_snapshot((requirement(),), FeatureQuery(M, M+10))
        self.assertEqual((snap.latest("signal").source_ns, snap.latest("signal").received_ns), (M, M))
        provider.fail = True
        with self.assertRaises(FeatureSourceError):
            service.refresh((request,), clock_ns=lambda: M+20)
        with self.assertRaises(FeatureUnavailable) as error:
            data.feature_snapshot((requirement(),), FeatureQuery(M, M+20))
        self.assertEqual(error.exception.code, "SOURCE_ERROR")
        provider.fail = False
        service.refresh((replace(request, as_of_ns=M+30),), clock_ns=lambda: M+30)
        self.assertEqual(data.feature_snapshot((requirement(),), FeatureQuery(M, M+30)).latest("signal").revision, 0)
        with self.assertRaises(FeatureUnavailable) as error:
            data.feature_snapshot((requirement(),), FeatureQuery(M, 2*M+31))
        self.assertEqual(error.exception.code, "REFRESH_STALE")

    def test_failed_multi_dataset_refresh_does_not_publish_partial_records(self):
        reg = registry()
        data = hub(reg=reg)
        provider = MemoryProvider(reg)
        service = FeatureRefreshService(data, {("minute.signal", 1): provider})
        requests = (FeatureRequest("minute.signal", 1, ("IM",), 0, M, M),
                    FeatureRequest("index.price", 1, ("000852",), 0, M, M))
        with self.assertRaises(FeatureError): service.refresh(requests, clock_ns=lambda: M)
        self.assertEqual(data.features.store.capture()[1], ())


class FakeSession:
    def __init__(self, rows):
        self.rows, self.calls = rows, []
        self.fail = False
    def connect(self, *args, **kwargs):
        self.calls.append("connect")
        return True
    def close(self): self.calls.append("close")
    def run(self, script):
        self.calls.append(script)
        if self.fail:
            raise RuntimeError("secret must not leak")
        import pandas as pd
        return pd.DataFrame(self.rows)


class FakeCursor(list):
    def limit(self, limit): return FakeCursor(self[:limit])
    def close(self): pass


class FakeCollection:
    def __init__(self, rows): self.rows, self.calls = rows, []
    def find(self, query, **kwargs):
        self.calls.append(query)
        return FakeCursor(self.rows)


class FakeClient:
    def __init__(self, collection):
        self.collection, self.closed = collection, False
        self.admin = self
    def __getitem__(self, key):
        return self if key == "features" else self.collection
    def command(self, name): return {"ok": 1}
    def close(self): self.closed = True


def database_row():
    encoded = encode_record(row())
    encoded["score"] = encoded.pop("payload.score")
    return encoded


class DatabaseTests(unittest.TestCase):
    def binding(self):
        return DatasetTable("minute.signal", 1, "signals", FeatureColumnMapping({"payload.score": "score"}))

    def request(self):
        return FeatureRequest("minute.signal", 1, ("IM",), 0, M, M)

    def test_database_adapters_are_lazy_and_return_same_records_as_file(self):
        reg = registry()
        sdk = FakeSession([database_row()])
        ddb = DolphinDbFeatureProvider(DolphinDbFeatureConfig("host", 8848, "user", "secret",
            "dfs://features", (self.binding(),)), reg, session_factory=lambda **kwargs: sdk)
        mongo_collection = FakeCollection([database_row()])
        client = FakeClient(mongo_collection)
        mongo = MongoDbFeatureProvider(MongoDbFeatureConfig("mongodb://user:secret@host", "features",
            (self.binding(),), "primary"), reg, client_factory=lambda *args, **kwargs: client)
        self.assertEqual(sdk.calls, [])
        for provider in (ddb, mongo):
            with self.assertRaises(FeatureSourceError): provider.read(self.request())
            with provider:
                self.assertEqual(provider.read(self.request()).records, (row(),))
            self.assertNotIn("secret", repr(provider.config))
        self.assertTrue(client.closed)
        self.assertIn("available_ns <=", sdk.calls[1])
        self.assertIn("$lte", mongo_collection.calls[0]["available_ns"])

    def test_ddb_error_is_not_empty_and_credentials_do_not_leak(self):
        sdk = FakeSession([database_row()])
        config = DolphinDbFeatureConfig("host", 8848, "user", "secret", "dfs://features", (self.binding(),))
        with DolphinDbFeatureProvider(config, registry(), session_factory=lambda **kwargs: sdk) as provider:
            sdk.fail = True
            with self.assertRaises(FeatureSourceError) as error: provider.read(self.request())
            self.assertNotIn("secret", str(error.exception))

    def test_row_limit_and_sql_mapping_injection_rejected(self):
        sdk = FakeSession([database_row(), database_row()])
        config = DolphinDbFeatureConfig("host", 8848, "user", "secret", "dfs://features", (self.binding(),))
        with DolphinDbFeatureProvider(config, registry(), session_factory=lambda **kwargs: sdk) as provider:
            with self.assertRaises(FeatureSourceError): provider.read(replace(self.request(), max_rows=1))
            with self.assertRaises(FeatureError): provider.read(replace(self.request(), keys=('IM";drop',)))
        with self.assertRaises(FeatureError): DatasetTable("minute.signal", 1, "table;drop")

    def test_purpose_matrix_has_no_silent_backend_fallback(self):
        from bomber.framework.dataprep.sources.policy import DataSourcePurpose as Purpose
        for purpose, backend in ((Purpose.LIVE_FEATURE, "file"), (Purpose.OFFLINE_FEATURE, "mongodb"),
                                (Purpose.OFFLINE_FEATURE, "dolphindb")):
            with self.assertRaises(ValueError):
                FeatureProviderFactory.create_for(purpose, backend, {}, registry())
        provider = FeatureProviderFactory.create_for(Purpose.OFFLINE_FEATURE, "file", "unused.csv", registry())
        self.assertIsInstance(provider, FileFeatureProvider)


if __name__ == "__main__":
    unittest.main()
