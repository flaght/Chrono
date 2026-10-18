from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bomber.framework.calendars import NativeCalendarProvider, ExportedCalendarProvider, export_calendar
from bomber.framework.dataprep.calendar import load_calendar


class TradingCalendarTests(unittest.TestCase):
    def setUp(self):
        self.provider = NativeCalendarProvider()
        self.snapshot = self.provider.snapshot(date(2026, 10, 1), date(2026, 10, 30))

    def test_existing_native_rules_and_count(self):
        self.assertFalse(self.snapshot.is_trading_day(date(2026, 10, 1)))
        self.assertTrue(self.snapshot.is_trading_day(date(2026, 10, 12)))
        self.assertEqual(self.snapshot.remaining_days(date(2026, 10, 12), date(2026, 10, 16)), 4)
        self.assertEqual(self.snapshot.manifest["backend"], "compiled")

    def test_unknown_year_and_unsupported_calendar_fail(self):
        with self.assertRaisesRegex(ValueError, "覆盖不足"):
            self.provider.snapshot(date(2027, 1, 1), date(2027, 12, 31))
        with self.assertRaises(ValueError):
            NativeCalendarProvider("china.hkex")
        with self.assertRaises(ValueError):
            self.snapshot.remaining_days(date(2026, 10, 12), date(2026, 11, 20))

    def test_no_native_extension_never_falls_back_to_weekdays(self):
        with patch.dict("sys.modules", {"bomber.framework.calendars.api": None}):
            with self.assertRaisesRegex(RuntimeError, "当前Python"):
                self.provider.snapshot(date(2026, 10, 12), date(2026, 10, 16))

    def test_file_export_has_same_snapshot_and_backtest_interface(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"calendar.csv"
            export_calendar(self.snapshot, path)
            restored = ExportedCalendarProvider(path).snapshot(date(2026, 10, 12), date(2026, 10, 16))
            self.assertEqual(restored.fingerprint, self.snapshot.fingerprint)
            self.assertEqual(restored.days, self.snapshot.days)
            self.assertEqual(restored.backend, "file")
            self.assertEqual(load_calendar(path, date(2026, 10, 12), date(2026, 10, 16)), self.snapshot.days)
            with self.assertRaises(ValueError):
                export_calendar(self.snapshot, path)
            path.write_text(path.read_text().replace("2026-10-12,1", "2026-10-12,0"))
            with self.assertRaisesRegex(ValueError, "摘要"):
                load_calendar(path)

    def test_exported_metadata_identity_cannot_change(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"calendar.csv"
            export_calendar(self.snapshot, path)
            manifest = Path(str(path)+".manifest.json")
            values = json.loads(manifest.read_text())
            values["rules_version"] = "another-rules-version"
            manifest.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "身份"):
                load_calendar(path)


if __name__ == "__main__":
    unittest.main()
