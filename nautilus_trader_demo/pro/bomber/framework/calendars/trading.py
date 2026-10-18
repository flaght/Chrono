"""Versioned trading-date snapshots; Live uses compiled rules, backtests use exports."""
from dataclasses import dataclass
from datetime import date, timedelta
import csv
import hashlib
import json
from pathlib import Path
from typing import Protocol


RULES_VERSION = "bundled-sse-2026-v1"
RULES_START = date(1991, 1, 1)
RULES_END = date(2026, 12, 31)


@dataclass(frozen=True)
class CalendarSnapshot:
    calendar_id: str
    rules_version: str
    start: date
    end: date
    days: tuple[date, ...]
    backend: str

    def __post_init__(self):
        if not self.calendar_id or not self.rules_version or self.start > self.end:
            raise ValueError("日历身份、版本或覆盖范围无效")
        if not self.days or tuple(sorted(set(self.days))) != self.days or any(
                not self.start <= day <= self.end for day in self.days):
            raise ValueError("交易日须非空、有序、唯一并属于覆盖范围")

    @property
    def fingerprint(self):
        data = {"calendar_id": self.calendar_id, "rules_version": self.rules_version,
            "start": str(self.start), "end": str(self.end), "days": list(map(str, self.days))}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @property
    def manifest(self):
        return {"calendar_id": self.calendar_id, "rules_version": self.rules_version,
            "coverage_start": str(self.start), "coverage_end": str(self.end),
            "fingerprint": self.fingerprint, "backend": self.backend, "trading_days": len(self.days)}

    def require_range(self, start, end):
        if start > end or start < self.start or end > self.end:
            raise ValueError(f"日历覆盖不足: {start}..{end}; 已维护{self.start}..{self.end}")

    def is_trading_day(self, day):
        self.require_range(day, day)
        return day in self.days

    def remaining_days(self, day, last_day):
        self.require_range(day, last_day)
        if last_day not in self.days:
            raise ValueError("合约最后交易日不属于交易日历")
        return sum(day < candidate <= last_day for candidate in self.days)


class CalendarProvider(Protocol):
    def snapshot(self, start: date, end: date) -> CalendarSnapshot: ...


class NativeCalendarProvider:
    """Explicitly maintained range; the low-level library cannot guess future holidays."""
    def __init__(self, calendar_id="china.cffex"):
        self.calendar_id = calendar_id.lower()
        if self.calendar_id not in {"china.cffex", "china.sse"}:
            raise ValueError("当前有明确维护范围的日历为china.cffex、china.sse")

    def snapshot(self, start=RULES_START, end=RULES_END):
        if start > end or start < RULES_START or end > RULES_END:
            raise ValueError(f"编译日历覆盖不足: {start}..{end}; 已维护{RULES_START}..{RULES_END}")
        try:
            from .api import bizDatesList
        except ImportError as error:
            raise RuntimeError("交易日历扩展不可用，请用当前Python运行calendars/setup.py build_ext --inplace") from error
        days = tuple(item.date() for item in bizDatesList(self.calendar_id, start, end))
        return CalendarSnapshot(self.calendar_id, RULES_VERSION, start, end, days, "compiled")


def export_calendar(snapshot, path):
    """Export all dates and a completion manifest; refuse to overwrite historical inputs."""
    path = Path(path)
    manifest_path = Path(str(path) + ".manifest.json")
    if path.suffix.lower() != ".csv" or path.exists() or manifest_path.exists():
        raise ValueError("日历导出须使用新的CSV路径")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "is_trading_day"))
        current, days = snapshot.start, set(snapshot.days)
        while current <= snapshot.end:
            writer.writerow((str(current), int(current in days)))
            current += timedelta(days=1)
    manifest = {"format_version": 1, **snapshot.manifest,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


class ExportedCalendarProvider:
    def __init__(self, path):
        self.path = Path(path)

    def snapshot(self, start, end):
        metadata = json.loads(Path(str(self.path) + ".manifest.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("sha256") != hashlib.sha256(self.path.read_bytes()).hexdigest():
            raise ValueError("日历导出版本或文件摘要不符")
        first, last = date.fromisoformat(metadata["coverage_start"]), date.fromisoformat(metadata["coverage_end"])
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["date", "is_trading_day"]:
                raise ValueError("日历导出列不符")
            rows = list(reader)
        expected = [first + timedelta(days=n) for n in range((last-first).days+1)]
        if [date.fromisoformat(row["date"]) for row in rows] != expected or any(
                row["is_trading_day"] not in {"0", "1"} for row in rows):
            raise ValueError("日历导出日期覆盖不完整或标志无效")
        snapshot = CalendarSnapshot(metadata["calendar_id"], metadata["rules_version"], first, last,
            tuple(day for day, row in zip(expected, rows) if row["is_trading_day"] == "1"), "file")
        if snapshot.fingerprint != metadata["fingerprint"] or len(snapshot.days) != metadata["trading_days"]:
            raise ValueError("日历快照身份或内容不符")
        snapshot.require_range(start, end)
        return snapshot
