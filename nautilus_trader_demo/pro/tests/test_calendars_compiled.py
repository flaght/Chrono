"""Run after building calendars with the same Python interpreter."""

import importlib
import importlib.machinery
import pickle
import unittest
from datetime import date

from bomber.framework.calendars.api import (
    advanceDateByCalendar, bizDatesList, isBizDay, makeSchedule,
)
from bomber.framework.calendars.DateUtilities import Calendar, Date, Period


class CompiledCalendarTests(unittest.TestCase):
    def test_all_extensions_load_from_current_namespace(self):
        names = (
            "DateUtilities.Calendar", "DateUtilities.Date", "DateUtilities.Period",
            "DateUtilities.Schedule", "Enums.BizDayConventions", "Enums.DateGeneration",
            "Enums.Months", "Enums.NormalizingType", "Enums.OptionType",
            "Enums.TimeUnits", "Enums.Weekdays",
        )
        for name in names:
            with self.subTest(name=name):
                module = importlib.import_module(f"bomber.framework.calendars.{name}")
                self.assertTrue(any(module.__file__.endswith(suffix)
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES))

    def test_cffex_uses_existing_holiday_and_weekend_rules(self):
        self.assertFalse(isBizDay("china.cffex", date(2026, 1, 1)))
        self.assertFalse(isBizDay("china.cffex", date(2026, 1, 10)))
        self.assertTrue(isBizDay("china.cffex", date(2026, 1, 12)))

    def test_public_list_and_remaining_day_count_agree(self):
        days = bizDatesList("china.cffex", date(2026, 1, 12), date(2026, 1, 16))
        self.assertEqual([d.date() for d in days],
            [date(2026, 1, day) for day in range(12, 17)])
        # 08 counts after today, including the last trading day.
        self.assertEqual(Calendar("china.cffex").bizDaysBetween(
            Date(2026, 1, 12), Date(2026, 1, 16), False, True), 4)

    def test_business_day_advance_skips_weekend(self):
        self.assertEqual(advanceDateByCalendar(
            "china.cffex", date(2026, 1, 9), "1b").date(), date(2026, 1, 12))

    def test_schedule_and_settings_dependencies_load(self):
        from bomber.framework.calendars.Env import Settings
        original = Settings.evaluationDate
        try:
            Settings.evaluationDate = Date(2026, 1, 12)
            days = makeSchedule(date(2026, 1, 12), date(2026, 3, 12),
                "1m", calendar="china.cffex")
            self.assertEqual([d.date() for d in days],
                [date(2026, 1, 12), date(2026, 2, 12), date(2026, 3, 12)])
        finally:
            Settings.evaluationDate = original

    def test_date_period_and_calendar_pickle_use_new_namespace(self):
        for value in (Date(2026, 1, 12), Period("1m"), Calendar("china.cffex")):
            with self.subTest(value=value):
                self.assertEqual(pickle.loads(pickle.dumps(value)), value)


if __name__ == "__main__":
    unittest.main()
