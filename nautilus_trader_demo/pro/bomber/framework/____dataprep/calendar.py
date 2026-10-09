"""交易日历数据，明确区分外部提供与根据文件推断的日期。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .catalog import scan_bar_files
from .session import current_session, fail, read_feather


@dataclass(frozen=True)
class TradingCalendar:
    days: tuple[date, ...]
    source: str
    inferred: bool = False

    def previous(self, day):
        before = [candidate for candidate in self.days if candidate < day]
        if not before:
            fail("INSUFFICIENT_CALENDAR", f"No previous trading day for {day}")
        return before[-1]

    def require_range(self, start_day, end_day):
        if not self.days or start_day < self.days[0] or end_day > self.days[-1]:
            fail("INSUFFICIENT_CALENDAR", f"Calendar does not cover {start_day}..{end_day}")


def load_calendar(path, start_day=None, end_day=None):
    import pandas as pd
    session = current_session()
    if Path(path).suffix.lower() == ".csv":
        cache_key = ("calendar_csv", session.fingerprint(path)) if session is not None else None
        if session is not None and cache_key in session.cache:
            frame = session.cache[cache_key].copy(deep=True)
        else:
            frame = pd.read_csv(path)
            if session is not None:
                session.cache[cache_key] = frame.copy(deep=True)
    else:
        frame = read_feather(path)
    if frame.empty or not frame.columns.is_unique or not {"date", "is_trading_day"} <= set(frame):
        fail("MISSING_FIELD", "Calendar needs unique date and is_trading_day columns", source=path)
    flags = frame.is_trading_day.astype(str).str.lower()
    if not flags.isin(("true", "false", "1", "0")).all():
        fail("INVALID_CALENDAR", "Trading-day flag must be True/False or 1/0", source=path)
    stamps = pd.to_datetime(frame.date, errors="raise")
    dates = stamps.dt.date
    if stamps.isna().any() or dates.duplicated().any():
        fail("INVALID_CALENDAR", "Null or duplicate calendar dates", source=path)
    if start_day is not None and start_day < dates.min() or end_day is not None and end_day > dates.max():
        fail("INSUFFICIENT_CALENDAR", "Calendar does not cover requested range", source=path)
    days = tuple(sorted(dates.loc[flags.isin(("true", "1"))]))
    if not days:
        fail("INVALID_CALENDAR", "Calendar has no trading days", source=path)
    session = current_session()
    if session is not None:
        session.coverage.calendar_source = str(Path(path).resolve())
    return days


def infer_market_calendar(roots, start_day, end_day):
    # 兼容模式信任上游数据完整性；缺文件
    # 不能证明交易所休市，此假设会写入报告。
    days = set()
    for root in roots:
        days.update(key.trading_day for key in scan_bar_files(root))
    calendar = tuple(sorted(days))
    if not calendar:
        fail("INSUFFICIENT_CALENDAR", "No market dates from which to infer calendar")
    if start_day < calendar[0] or end_day > calendar[-1]:
        fail("INSUFFICIENT_CALENDAR", f"请求区间{start_day}..{end_day}超出行情日期范围{calendar[0]}..{calendar[-1]}")
    session = current_session()
    if session is not None:
        session.coverage.calendar_source = "INFERRED_MARKET_DATES"
        session.coverage.assumptions += ("inferred_calendar_trusts_upstream_market_date_completeness",)
    return calendar


def infer_calendar_from_inventory(index, coverage_assumptions):
    if not coverage_assumptions:
        fail("INVALID_CALENDAR", "State completeness assumptions for inferred calendars")
    return TradingCalendar(tuple(sorted({k.trading_day for k in index})),
                           "INFERRED_INVENTORY:" + str(coverage_assumptions), True)
