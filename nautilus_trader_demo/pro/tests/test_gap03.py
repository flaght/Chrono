"""GAP-03来源隔离、参考时效及导出可复现性验收；仅假SDK及临时文件。"""
from datetime import date
from contextlib import ExitStack
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.dataprep.live_references import LiveFuturesReferences
from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
from bomber.framework.dataprep.sources import (
    DataSourcePurpose as Purpose, DolphinDbReferenceConfig, ReferenceDataset as Dataset,
    ReferenceQuery, ReferenceSourceFactory, ReferenceSourceError)
from bomber.framework.dataprep.sources.artifacts import ExportedReferenceSource, export_reference_bundle
from .test_reference_sources import BASE, DAY, FakeSession


class Gap03Tests(unittest.TestCase):
    def setUp(self):
        self.now = BASE
        self.sdk = FakeSession()
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
        self.config = DolphinDbReferenceConfig("fake", 8848, "fake", "never-export-this-secret")
        self.source = ReferenceSourceFactory.create_for(Purpose.EXPORT, "dolphindb", self.config,
            session_factory=lambda **kwargs: self.sdk)
        self.source.open()
        self.addCleanup(self.source.close)
        self.queries = {
            Dataset.FUTURES_BASIC: ReferenceQuery(products=("RB",), active_on=DAY),
            Dataset.ADJUSTMENT_FACTORS: ReferenceQuery(products=("RB",), start_date=date(2026, 9, 21), end_date=date(2026, 9, 21)),
            Dataset.CONTRACT_STRUCTURE: ReferenceQuery(products=("RB",), end_date=date(2026, 9, 21))}
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "references.json"

    def explicit(self):
        for frame in self.sdk.data.values():
            frame["available_ns"] = BASE - 1
            frame["revision"] = 1
            frame["source_version"] = "fixture-v1"

    def references(self, **kwargs):
        return LiveFuturesReferences(self.source, products=("RB",), trading_day=DAY,
            started_ns=BASE, factor_date_basis="source", factor_availability="observed-on-read",
            clock_ns=lambda: self.now, **kwargs)

    def test_purpose_matrix_rejects_before_constructing_or_connecting(self):
        local = (Purpose.OFFLINE_REFERENCE, Purpose.OFFLINE_MARKET_HISTORY, Purpose.FILE_REFERENCE_TEST)
        for purpose in Purpose:
            if purpose in (Purpose.LIVE_MARKET_HISTORY, Purpose.LIVE_RECOVERY):
                continue  # 历史用途新矩阵由history-provider组覆盖。
            backend = "file" if purpose in local else "dolphindb"
            wrong = "dolphindb" if backend == "file" else "file"
            with self.subTest(purpose=purpose), patch.object(ReferenceSourceFactory, "create") as create:
                with self.assertRaisesRegex(ValueError, "禁止静默回退"):
                    ReferenceSourceFactory.create_for(purpose, wrong, {})
                create.assert_not_called()
                ReferenceSourceFactory.create_for(purpose, backend, {})
                create.assert_called_once_with(backend, {})

    def test_allowed_database_factory_is_lazy(self):
        with patch("bomber.framework.dataprep.sources.dolphindb._session_factory") as sdk:
            ReferenceSourceFactory.create_for(Purpose.LIVE_REFERENCE, "dolphindb", self.config)
            ReferenceSourceFactory.create_for(Purpose.EXPORT, "dolphindb", self.config)
            sdk.assert_not_called()

    def test_online_cli_defaults_to_database_and_requires_declared_day(self):
        live = import_module("demos.01_main_ema.run_live")
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921"])
        self.assertEqual(args.reference_source, "dolphindb")
        self.assertEqual((args.factor_date_basis, args.reference_max_source_age_days), ("source", 14))
        with self.assertRaises(SystemExit):
            live.parse_args(["--connect", "--product", "RB"])

    def test_file_reference_requires_explicit_test_mode(self):
        live = import_module("demos.01_main_ema.run_live")
        base = ["--connect", "--product", "RB", "--reference-source", "file"]
        with self.assertRaises(SystemExit):
            live.parse_args(base)
        self.assertTrue(live.parse_args([*base, "--allow-file-reference-test"]).allow_file_reference_test)
        with self.assertRaises(SystemExit):
            live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921",
                "--allow-file-reference-test"])

    def test_database_failure_never_falls_back_to_existing_files_or_starts_feed(self):
        live = import_module("demos.01_main_ema.run_live")
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921"])
        self.sdk.fail = True
        with ExitStack() as resources, patch.dict("os.environ", {
                "DDB_HOST": "fake", "DDB_USERNAME": "fake", "DDB_PASSWORD": "fake"}), \
                patch.object(live.ReferenceSourceFactory, "create_for", return_value=self.source), \
                patch.object(live, "resolve_paths", side_effect=AssertionError("禁止回退本地")) as files, \
                patch.object(live, "CtpLiveDataFeed") as feed:
            with self.assertRaises(ReferenceSourceError):
                live.prepare_inputs(args, SimpleNamespace(trading_day="20260922", resources=resources),
                    SimpleNamespace(transport=None))
            files.assert_not_called()
            feed.assert_not_called()

    def test_continuous_cli_defers_source_day_to_each_session_calendar(self):
        live = import_module("demos.01_main_ema.run_live")
        calendar = Path(self.directory.name) / "calendar.json"
        calendar.write_text("{}")
        args = live.parse_args(["--connect", "--product", "CU", "--mode", "simnow", "--run-forever",
            "--enable-orders", "--confirm-simnow", "--state-file", str(self.path),
            "--trading-calendar", str(calendar)])
        self.assertEqual(args.reference_source, "dolphindb")
        self.assertIsNone(args.expected_source_day)
        with self.assertRaisesRegex(ValueError, "先由会话日历"):
            live.prepare_inputs(args, SimpleNamespace(trading_day="20261009"), None)

    def test_freshness_config_rejects_invalid_values(self):
        for kwargs in ({"max_source_age_days": 0}, {"max_observation_age_seconds": -1},
                       {"max_refresh_duration_seconds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReferenceFreshnessPolicy(**kwargs)

    def test_expected_day_mismatch_rejects_even_when_age_is_small(self):
        with self.assertRaisesRegex(SectorDataUnavailable, "来源日不符"):
            self.references(freshness=ReferenceFreshnessPolicy(date(2026, 9, 18)))

    def test_source_age_limit_and_holiday_gap_are_separate_from_expected_day(self):
        policy = ReferenceFreshnessPolicy(date(2026, 9, 30), max_source_age_days=14)
        policy.validate_day(date(2026, 9, 30), date(2026, 10, 8))
        with self.assertRaises(SectorDataUnavailable):
            policy.validate_day(date(2026, 9, 30), date(2026, 10, 15))
        with self.assertRaises(SectorDataUnavailable):
            policy.validate_day(date(2026, 10, 8), date(2026, 10, 8))

    def test_cached_observation_age_and_clock_rollback_close_readiness(self):
        refs = self.references(refresh_seconds=60,
            freshness=ReferenceFreshnessPolicy(max_observation_age_seconds=30))
        self.now += 30_000_000_000
        refs.snapshot(self.now)
        self.now += 1
        with self.assertRaisesRegex(SectorDataUnavailable, "最近成功观测"):
            refs.snapshot(self.now)
        self.assertFalse(refs.manifest["ready"])
        calls_before = len(self.sdk.calls)
        # 既覆盖早于首次观测，也覆盖仍晚于首次观测但早于最近运行时刻的回退。
        for rolled_back in (BASE - 1, BASE + 10_000_000_000):
            with self.subTest(rolled_back=rolled_back):
                self.now = rolled_back
                with self.assertRaisesRegex(SectorDataUnavailable, "时钟回退"):
                    refs.snapshot(BASE)
                with self.assertRaisesRegex(SectorDataUnavailable, "时钟回退"):
                    refs.refresh(force=True)
                self.assertEqual(len(self.sdk.calls), calls_before)
                self.assertFalse(refs.manifest["ready"])
                self.assertEqual(refs.manifest["observed_at_ns"], BASE)
                self.assertEqual(refs.manifest["first_observed_ns"], BASE)
        # 时钟恢复本身不能释放旧缓存；成功刷新后才能恢复就绪。
        self.now = BASE + 30_000_000_001
        with self.assertRaisesRegex(SectorDataUnavailable, "时钟回退"):
            refs.snapshot(self.now)
        self.assertEqual(len(self.sdk.calls), calls_before)
        self.now = BASE + 60_000_000_000
        refs.snapshot(self.now)
        self.assertGreater(len(self.sdk.calls), calls_before)
        self.assertTrue(refs.manifest["ready"])
        self.assertEqual(refs.manifest["observed_at_ns"], self.now)
        self.assertEqual(refs.manifest["first_observed_ns"], BASE)
        # 仅查询诊断manifest也必须发现回退，且强制刷新仍须等时钟恢复。
        calls_before = len(self.sdk.calls)
        self.now -= 1
        self.assertFalse(refs.manifest["ready"])
        self.assertIn("时钟回退", refs.manifest["failure"])
        with self.assertRaisesRegex(SectorDataUnavailable, "时钟回退"):
            refs.refresh(force=True)
        self.assertEqual(len(self.sdk.calls), calls_before)
        self.now += 1
        refs.refresh(force=True)
        self.assertTrue(refs.manifest["ready"])

    def test_long_refresh_blocks_old_snapshot(self):
        refs = self.references(refresh_seconds=0)
        def delayed(table):
            self.now += 6_000_000_000
        self.sdk.on_read = delayed
        with self.assertRaisesRegex(SectorDataUnavailable, "刷新超时"):
            refs.snapshot(self.now)
        self.assertFalse(refs.manifest["ready"])

    def test_expected_day_is_checked_on_later_refresh_not_only_startup(self):
        refs = self.references(refresh_seconds=0,
            freshness=ReferenceFreshnessPolicy(date(2026, 9, 21)))
        self.sdk.data["fut_contract"]["date"] = "2026-09-18"
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-18"
        with self.assertRaisesRegex(SectorDataUnavailable, "来源日不符"):
            refs.snapshot(self.now)
        self.assertFalse(refs.manifest["ready"])

    def test_publication_gate_and_source_age_are_both_enforced(self):
        self.sdk.data["fut_basic"]["available_ns"] = BASE + 10
        refs = self.references(freshness=ReferenceFreshnessPolicy(date(2026, 9, 21)))
        self.assertFalse(refs.manifest["ready"])
        with self.assertRaises(SectorDataUnavailable):
            refs.snapshot(BASE)
        self.now += 10
        refs.snapshot(self.now)
        self.assertTrue(refs.manifest["ready"])

    def test_export_rejects_missing_publication_without_inventing_history(self):
        with self.assertRaisesRegex(ValueError, "available_ns"):
            export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        self.assertFalse(self.path.exists())

    def test_observed_export_cannot_leak_to_earlier_events(self):
        export_reference_bundle(self.source, self.queries, self.path,
            visibility="observed", clock_ns=lambda: BASE)
        with ExportedReferenceSource(self.path, as_of_ns=BASE - 1) as local:
            self.assertFalse(local.contract_structure(self.queries[Dataset.CONTRACT_STRUCTURE]).rows)
            self.assertEqual(len(local.read_as_of(Dataset.CONTRACT_STRUCTURE,
                self.queries[Dataset.CONTRACT_STRUCTURE], BASE).rows), 1)
            self.assertEqual(local.manifest["publication_evidence"], "observation_lower_bound_only")

    def test_roundtrip_preserves_types_scope_versions_and_online_snapshot(self):
        self.explicit()
        self.sdk.data["fut_basic"]["extra_decimal"] = Decimal("1.2300")
        self.sdk.data["fut_basic"]["extra_timestamp"] = pd.Timestamp("2026-09-21T12:00:00.123456789+08:00")
        online = LiveFuturesReferences(self.source, products=("RB",), trading_day=DAY, started_ns=BASE,
            factor_date_basis="source", factor_availability="explicit", clock_ns=lambda: BASE)
        envelope = export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        self.assertNotIn("never-export-this-secret", self.path.read_text())
        with patch("bomber.framework.dataprep.sources.dolphindb._session_factory", side_effect=AssertionError("离线不允许SDK")):
            with ExportedReferenceSource(self.path, as_of_ns=BASE) as local:
                for dataset, query in self.queries.items():
                    self.assertEqual(local.read(dataset, query).fingerprint, self.source.read(dataset, query).fingerprint)
                offline = LiveFuturesReferences(local, products=("RB",), trading_day=DAY, started_ns=BASE,
                    factor_date_basis="source", factor_availability="explicit", clock_ns=lambda: BASE)
                self.assertEqual(offline.snapshot(BASE), online.snapshot(BASE))
                audit = local.manifest["datasets"][Dataset.ADJUSTMENT_FACTORS.value]
                self.assertEqual(audit["source_versions"], ["fixture-v1"])
                self.assertEqual(audit["query"]["products"], ["RB"])
        self.assertEqual(envelope["payload"]["manifest"]["source_identity"]["host"], "fake")

    def test_later_revision_is_visible_only_after_actual_publication(self):
        self.explicit()
        old = self.sdk.data["fut_adjustment_factors"]
        new = old.assign(available_ns=BASE + 10, revision=2, pcr_cumfactor="3", source_version="fixture-v2")
        self.sdk.data["fut_adjustment_factors"] = pd.concat([old, new], ignore_index=True)
        export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        with ExportedReferenceSource(self.path, as_of_ns=BASE) as local:
            before = local.adjustment_factors(self.queries[Dataset.ADJUSTMENT_FACTORS])
            after = local.read_as_of(Dataset.ADJUSTMENT_FACTORS, self.queries[Dataset.ADJUSTMENT_FACTORS], BASE + 10)
            self.assertEqual((before.rows[0]["pcr_cumfactor"], after.rows[0]["pcr_cumfactor"]), ("2", "3"))

    def test_export_fails_on_unstable_reads_and_does_not_publish(self):
        self.explicit()
        def mutate(table):
            if table == "fut_adjustment_factors":
                self.sdk.data[table]["pcr_cumfactor"] = "3"
        self.sdk.on_read = mutate
        with self.assertRaises(ReferenceSourceError):
            export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        self.assertFalse(self.path.exists())

    def test_conflicting_same_revision_and_duplicate_output_are_rejected(self):
        self.explicit()
        old = self.sdk.data["fut_adjustment_factors"]
        self.sdk.data["fut_adjustment_factors"] = pd.concat([old, old.assign(pcr_cumfactor="3")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "冲突"):
            export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        self.sdk.data["fut_adjustment_factors"] = old
        export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        self.assertEqual(self.path.read_bytes(), original)

    def test_tampered_or_incompatible_bundle_is_rejected(self):
        self.explicit()
        export_reference_bundle(self.source, self.queries, self.path, clock_ns=lambda: BASE)
        data = json.loads(self.path.read_text())
        data["payload"]["manifest"]["schema_version"] = 2
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "校验"):
            ExportedReferenceSource(self.path, as_of_ns=BASE).open()

    def test_local_export_source_cannot_be_used_as_live_reference(self):
        with self.assertRaises(ValueError):
            ExportedReferenceSource(self.path, as_of_ns=BASE, purpose=Purpose.LIVE_REFERENCE)

    def test_export_verification_cli_is_offline_and_observed_time_is_respected(self):
        from scripts.integration.reference_data.verify_export import main
        export_reference_bundle(self.source, self.queries, self.path, visibility="observed", clock_ns=lambda: BASE)
        base = ["--bundle", str(self.path), "--trading-day", "20260922",
            "--expected-source-day", "20260921", "--products", "RB"]
        with patch("bomber.framework.dataprep.sources.dolphindb._session_factory", side_effect=AssertionError("禁止连接")):
            result = main(base)
            self.assertEqual((result["status"], result["source_day"]), ("passed", "2026-09-21"))
            self.assertEqual(result["cumulative_factors"]["RB"]["main"], "2")
            with self.assertRaises((SectorDataUnavailable, ValueError)):
                main([*base, "--as-of-ns", str(BASE - 1)])


if __name__ == "__main__":
    unittest.main()
