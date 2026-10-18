"""阶段一契约夹具；本次仅生成，须在具备 pandas/pyarrow 的环境执行。"""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from bomber.framework.dataprep import BarFileKey, BarReadSpec, InputError, InputSession, resolve_paths
from bomber.framework.dataprep.bars import read_bar_frame, PreparedFrameReader
from bomber.framework.dataprep.catalog import scan_bar_files, select_bar_files
from bomber.framework.dataprep.calendar import load_calendar, infer_calendar_from_inventory
from bomber.framework.dataprep.metadata import load_futures_basic, load_options_basic
from bomber.framework.dataprep.references import load_role_assignments
from bomber.framework.dataprep.scenarios import plan_fixed_contracts, prepare_fixed_contracts, prepare_option_chain, plan_option_chain

DAY = date(2026, 9, 11)


class InputContractsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def file(self, name="rb2610_20260911.feather"):
        path = self.root / name
        path.touch()
        return path

    def frame(self, **values):
        return pd.DataFrame([dict(datetime="2026-09-11 09:30", open=100, high=101,
            low=99, close=100, volume=1, **values)])

    def test_explicit_paths_override_environment_and_root(self):
        paths = resolve_paths({"bars_dir": self.root / "explicit", "data_root": self.root / "data"},
            env={"FUT_KLINE_DATA_DIR": str(self.root / "env")}, validate=False)
        self.assertEqual(paths.fut, self.root / "explicit")
        self.assertEqual(paths.sources["fut"], "explicit:bars_dir")

    def test_new_kline_parent_overrides_root_legacy_does_not(self):
        parent = self.root / "new"
        (parent / "opt").mkdir(parents=True)
        paths = resolve_paths({"data_root": self.root / "data"},
            env={"KLINE_DATA_DIR": str(parent)}, validate=False)
        self.assertEqual(paths.opt, parent / "opt")
        paths = resolve_paths({"data_root": self.root / "data"},
            env={"KLINE_DIR": str(parent)}, validate=False)
        self.assertEqual(paths.opt, self.root / "data" / "kline" / "opt")

    def test_conflicting_aliases_are_rejected(self):
        with self.assertRaisesRegex(InputError, "CONFLICTING_PATH"):
            resolve_paths({"fut_dir": self.root / "a", "bars_dir": self.root / "b"}, env={}, validate=False)

    def test_only_requested_input_paths_are_required(self):
        opt = self.root / "opt"
        opt.mkdir()
        paths = resolve_paths({"opt_dir": opt}, env={}, required_kinds=("option",))
        self.assertIsNone(paths.fut)

    def test_two_role_default_files_require_an_explicit_choice(self):
        for name in ("fut_contract.feather", "fut_contract_data.feather"):
            self.file(name)
        with self.assertRaisesRegex(InputError, "CONFLICTING_PATH"):
            resolve_paths({"role_dir": self.root}, env={}, required_files=("contract_struct",))
        paths = resolve_paths({"role_dir": self.root, "contract_struct": self.root / "fut_contract.feather"},
            env={}, required_files=("contract_struct",))
        self.assertEqual(paths.contract_struct.name, "fut_contract.feather")

    def test_inventory_normalizes_identity_and_scans_once(self):
        path = self.file("RB2610_20260911.feather")
        with InputSession() as session:
            index = scan_bar_files(self.root)
            self.assertEqual(index[BarFileKey("future", "rb2610.SHFE", DAY)], path)
            scan_bar_files(self.root, start_day=DAY)
            self.assertEqual(session.counters["directory_scans"], 1)

    def test_duplicate_file_and_directory_identity_are_hard_errors(self):
        self.file()
        other = self.root / "copy"
        other.mkdir()
        (other / "RB2610_20260911.feather").touch()
        with self.assertRaisesRegex(InputError, "DUPLICATE_FILE"):
            scan_bar_files(self.root)

    def test_filename_and_dated_directory_must_agree(self):
        folder = self.root / "20260910"
        folder.mkdir()
        (folder / "rb2610_20260911.feather").touch()
        with self.assertRaisesRegex(InputError, "IDENTITY_MISMATCH"):
            scan_bar_files(self.root)

    def test_required_missing_fails_optional_missing_is_reported(self):
        present = BarFileKey("future", "RB2610", DAY)
        absent = BarFileKey("future", "RB2701", DAY)
        index = {present: self.file()}
        _, coverage = select_bar_files(index, (present,), (absent,))
        self.assertEqual(coverage.optional_missing, (absent,))
        with self.assertRaisesRegex(InputError, "MISSING_REQUIRED_BAR"):
            select_bar_files(index, (absent,))

    def test_start_is_shifted_once_end_is_unchanged_and_cache_is_isolated(self):
        path = self.file()
        raw = self.frame()
        with InputSession() as session, patch("pandas.read_feather", return_value=raw) as reader:
            started = read_bar_frame(path, BarReadSpec(timestamp_label="start"))
            ended = read_bar_frame(path, BarReadSpec(timestamp_label="end"))
            self.assertEqual(started.first_ns - ended.first_ns, 60_000_000_000)
            self.assertEqual(started.frame.source_timestamp.iloc[0].minute, 30)
            started.frame.loc[0, "close"] = 777
            self.assertEqual(read_bar_frame(path, BarReadSpec(timestamp_label="start")).frame.close.iloc[0], 100)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(session.counters["feather_reads"], 1)
        self.assertEqual(session.cache, {})

    def test_utc_and_microsecond_source_yield_the_same_nanoseconds(self):
        raw = self.frame()
        raw["datetime"] = pd.to_datetime(["2026-09-11 01:30Z"]).as_unit("us")
        with patch("pandas.read_feather", return_value=raw):
            result = read_bar_frame(self.file())
        self.assertEqual(result.first_ns, pd.Timestamp("2026-09-11 09:30", tz="Asia/Shanghai").value)

    def test_available_time_is_preserved_and_pre_completion_is_rejected(self):
        spec = BarReadSpec(available_column="published")
        raw = self.frame(published="2026-09-11 09:31")
        with patch("pandas.read_feather", return_value=raw):
            result = read_bar_frame(self.file(), spec)
        self.assertEqual(result.frame.available_ns.iloc[0] - result.first_ns, 60_000_000_000)
        raw["published"] = "2026-09-11 09:29"
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "INVALID_TIMESTAMP"):
            read_bar_frame(self.file(), spec)

    def test_arrival_that_reverses_event_time_is_rejected(self):
        raw = pd.concat([self.frame(published="2026-09-11 09:32"),
            self.frame(published="2026-09-11 09:31")], ignore_index=True)
        raw.loc[1, "datetime"] = "2026-09-11 09:31"
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "UNSUPPORTED_CAPABILITY"):
            read_bar_frame(self.file(), BarReadSpec(available_column="published"))

    def test_exchange_night_session_uses_declared_trading_day(self):
        raw = self.frame()
        raw["datetime"] = "2026-09-10 21:01"
        with patch("pandas.read_feather", return_value=raw):
            result = read_bar_frame(self.file())
        self.assertEqual(result.frame.trading_day.iloc[0], DAY)
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "IDENTITY_MISMATCH"):
            read_bar_frame(self.file(), BarReadSpec(trading_day_policy="day_session"))

    def test_day_session_shift_cannot_cross_date(self):
        raw = self.frame()
        raw["datetime"] = "2026-09-11 23:59"
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "跨日期"):
            read_bar_frame(self.file(), BarReadSpec(timestamp_label="start", trading_day_policy="day_session"))

    def test_invalid_execution_values_reject_but_research_values_survive_with_audit(self):
        raw = self.frame()
        raw["close"] = "bad"
        path = self.file()
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "INVALID_BAR_FIELDS"):
            read_bar_frame(path)
        raw["open_interest"] = -1
        spec = BarReadSpec(required_fields=("close", "volume", "open_interest"), value_policy="research_audited")
        with patch("pandas.read_feather", return_value=raw):
            result = read_bar_frame(path, spec)
        self.assertTrue(pd.isna(result.frame.close.iloc[0]))
        self.assertEqual(len(result.issues), 2)
        self.assertEqual(result.frame.input_quality.iloc[0], "INVALID_BAR_FIELDS")

    def test_duplicate_and_unaligned_minutes_are_rejected(self):
        raw = pd.concat([self.frame(), self.frame()], ignore_index=True)
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "DUPLICATE_BAR"):
            read_bar_frame(self.file())
        raw = self.frame()
        raw["datetime"] = "2026-09-11 09:30:01"
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "INVALID_TIMESTAMP"):
            read_bar_frame(self.file())

    def test_volume_semantics_and_ohlc_are_strict(self):
        with self.assertRaises(ValueError):
            BarReadSpec(volume_semantics="cumulative")
        for field, value in (("volume", .5), ("volume", -1), ("high", 90), ("low", 110)):
            raw = self.frame()
            raw[field] = value
            with self.subTest(field=field, value=value), patch("pandas.read_feather", return_value=raw), self.assertRaises(InputError):
                read_bar_frame(self.file())

    def test_input_change_during_session_is_rejected(self):
        path = self.file()
        with InputSession(), patch("pandas.read_feather", return_value=self.frame()):
            read_bar_frame(path)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(InputError, "SOURCE_CHANGED"):
                read_bar_frame(path)

    def test_fixed_bundle_consumes_prepared_rows_without_second_shift(self):
        path = self.file()
        key = BarFileKey("future", "RB2610", DAY)
        with InputSession(), patch("pandas.read_feather", return_value=self.frame()):
            bundle = prepare_fixed_contracts(self.root, plan_fixed_contracts((key,), requested=(DAY, DAY)),
                spec=BarReadSpec(timestamp_label="start"))
        rows = list(bundle.sources[0].reader().read(path))
        self.assertEqual(rows[0][1]["datetime"].minute, 31)
        self.assertEqual(bundle.sources[0].result.frame.source_timestamp.iloc[0].minute, 30)

    def test_research_option_bundle_does_not_create_execution_or_change_candidates(self):
        path = self.file("MO2610-C-7000_20260911.feather")
        key = BarFileKey("option", "MO2610-C-7000", DAY)
        raw = pd.DataFrame([dict(datetime="2026-09-11 13:58", close=-1, open_interest=10, volume=0)])
        spec = BarReadSpec(required_fields=("close", "open_interest", "volume"),
            value_policy="research_audited", trading_day_policy="day_session")
        with patch("pandas.read_feather", return_value=raw):
            bundle = prepare_option_chain({"option": self.root}, plan_option_chain((key,)), specs={"option": spec})
        self.assertEqual(len(bundle.sources), 1)
        self.assertEqual(bundle.sources[0].purpose, "research")
        self.assertEqual(bundle.sources[0].result.frame.close.iloc[0], -1)
        self.assertEqual(bundle.context.instruments, {})

    def test_role_table_requires_only_declared_roles_and_rejects_conflicting_versions(self):
        raw = pd.DataFrame([dict(trade_date="2026-09-10", code="RB", main="rb2610")])
        with patch("pandas.read_feather", return_value=raw):
            result = load_role_assignments("unused", ("RB",), ("main",))
        self.assertEqual(result.main.iloc[0], "rb2610")
        raw = pd.concat([raw, raw.assign(main="rb2701")], ignore_index=True)
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "CONFLICTING_METADATA"):
            load_role_assignments("unused", ("RB",), ("main",))

    def test_contract_tick_multiplier_and_lifecycle_are_validated(self):
        raw = pd.DataFrame([dict(symbol="rb2610", code="RB", exchangeCD="XSGE",
            minChgPriceNum=1, contMultNum=10, listDate="2026-01-01", lastTradeDate="2026-10-15")])
        with patch("pandas.read_feather", return_value=raw):
            specs = load_futures_basic("unused", ("RB",))
        self.assertEqual(specs["RB2610"].venue, "SHFE")
        raw["contMultNum"] = 0
        with patch("pandas.read_feather", return_value=raw), self.assertRaises(ValueError):
            load_futures_basic("unused", ("RB",))

    def test_static_option_code_alias_and_conflicting_strike(self):
        raw = pd.DataFrame([dict(Code="MO2610-C-7000", contractType="CO", strikePrice=7000,
            contMultNum=100, varTicker="000852", exchangeCD="CCFX", listDate="2026-08-01",
            lastTradeDate="2026-10-16", expDate="2026-10-16")])
        with patch("pandas.read_feather", return_value=raw):
            specs = load_options_basic("unused", "MO", "000852", .2)
        self.assertEqual(specs["MO2610-C-7000"].month, "202610")
        raw["strikePrice"] = 7100
        with patch("pandas.read_feather", return_value=raw), self.assertRaisesRegex(InputError, "CONFLICTING_METADATA"):
            load_options_basic("unused", "MO", "000852", .2)

    def test_calendar_scope_and_inference_assumptions(self):
        path = self.root / "calendar.csv"
        path.write_text("date,is_trading_day\n2026-09-11,True\n2026-09-12,False\n", encoding="utf-8")
        self.assertEqual(load_calendar(path, DAY, date(2026, 9, 12)), (DAY,))
        with self.assertRaisesRegex(InputError, "INSUFFICIENT_CALENDAR"):
            load_calendar(path, date(2026, 9, 10), DAY)
        with self.assertRaisesRegex(InputError, "INVALID_CALENDAR"):
            infer_calendar_from_inventory({}, "")

    def test_reports_distinguish_requested_days_from_reference_dependencies(self):
        path = self.file()
        with InputSession() as session, patch("pandas.read_feather", return_value=self.frame()):
            session.coverage.requested = (date(2026, 9, 14), date(2026, 9, 15))
            read_bar_frame(path)
            session.write_reports(self.root / "reports")
        coverage = json.loads((self.root / "reports" / "input_coverage.json").read_text())
        self.assertEqual(coverage["actual_days"], [])
        self.assertEqual(coverage["dependency_days"], ["2026-09-11"])
        self.assertTrue((self.root / "reports" / "input_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
