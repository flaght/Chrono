"""配置驱动的交易日历与完整分钟枚举；算法不按品种选择。"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo
import json
from pathlib import Path

SHANGHAI = ZoneInfo("Asia/Shanghai")
MINUTE_NS = 60_000_000_000


@dataclass(frozen=True)
class TradingWindow:
    trading_day: str
    start: datetime
    end: datetime


class SessionCalendar:
    """读取显式时段、交易日期及覆盖；未知日期/归属不能猜测。"""

    def __init__(self, path, product=None):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1:
            raise ValueError("交易日历须声明schema_version=1和session_segments，旧配置须迁移")
        self.product = data["product"].upper()
        if product is not None and self.product != product.upper():
            raise ValueError(f"日历品种{self.product}与请求{product}不一致")
        self.zone = ZoneInfo(data["timezone"])
        self.start, self.end = date.fromisoformat(data["start"]), date.fromisoformat(data["end"])
        if self.start > self.end:
            raise ValueError("日历覆盖范围不合法")
        self.provenance = {"source": data.get("source"), "schema_version": 1,
            "sha256": sha256(raw).hexdigest(), "path": str(Path(path).resolve()),
            "evidence": "configured_snapshot_not_historical_revision_asof"}
        self.holidays = self._dates(data.get("holidays", []))
        self.no_night = self._dates(data.get("no_night", []))
        if "trading_days" in data:
            if "weekdays" in data or self.holidays:
                raise ValueError("显式交易日期不能同时声明weekdays/holidays")
            self.days = self._dates(data["trading_days"])
        else:
            weekdays = data["weekdays"]
            if (not isinstance(weekdays, list) or not weekdays or len(set(weekdays)) != len(weekdays)
                    or any(type(day) is not int or not 0 <= day <= 6 for day in weekdays)):
                raise ValueError("weekdays须显式声明0至6且不重复")
            self.days = set()
            day = self.start
            while day <= self.end:
                if day.weekday() in weekdays and day not in self.holidays:
                    self.days.add(day)
                day += timedelta(days=1)
        segments = self._segments(data["session_segments"])
        overrides = {date.fromisoformat(day): self._segments(rows, allow_empty=True)
            for day, rows in data.get("overrides", {}).items()}
        if any(not self.start <= day <= self.end for day in overrides):
            raise ValueError("特殊日期覆盖超出日历范围")
        ordered_days = sorted(self.days)
        # 区间只展开一次；所有消费者查询同一配置快照。
        self._intervals = []
        for index, day in enumerate(ordered_days):
            for begin, end, offset, rule in overrides.get(day, segments):
                if rule == "next_trading_day" and day in self.no_night and day not in overrides:
                    continue
                owner = day if rule == "same_day" else (
                    ordered_days[index + 1] if index + 1 < len(ordered_days) else None)
                left = datetime.combine(day, begin, self.zone)
                right = datetime.combine(day + timedelta(days=offset), end, self.zone)
                if right <= left:
                    raise ValueError("会话结束须晚于开始，跨午夜须声明end_day_offset=1")
                self._intervals.append((owner, left, right))
        self._intervals.sort(key=lambda item: item[1])
        if any(left[2] > right[1] for left, right in zip(self._intervals, self._intervals[1:])):
            raise ValueError("会话区间重叠或冲突")

    def _dates(self, values):
        days = [date.fromisoformat(value) for value in values]
        if len(set(days)) != len(days) or any(not self.start <= day <= self.end for day in days):
            raise ValueError("日历日期重复或超出覆盖范围")
        return set(days)

    @staticmethod
    def _segments(rows, *, allow_empty=False):
        if not isinstance(rows, list) or (not rows and not allow_empty):
            raise ValueError("session_segments须显式声明非空会话时段")
        result = []
        for row in rows:
            if not row.keys() <= {"start", "end", "end_day_offset", "trading_day_rule"}:
                raise ValueError("未知会话时段字段")
            begin, end = time.fromisoformat(row["start"]), time.fromisoformat(row["end"])
            offset, rule = row.get("end_day_offset", 0), row["trading_day_rule"]
            if (type(offset) is not int or offset not in {0, 1}
                    or rule not in {"same_day", "next_trading_day"}
                    or any(value.second or value.microsecond or value.tzinfo is not None for value in (begin, end))):
                raise ValueError("时段须按完整分钟声明时间、0/1日偏移及明确交易日归属规则")
            result.append((begin, end, offset, rule))
        return tuple(result)

    def _require_day(self, day):
        if not self.start <= day <= self.end:
            raise RuntimeError("交易日历未覆盖日期，须补充权威日历")

    def is_trading_day(self, day):
        self._require_day(day)
        return day in self.days

    def previous_trading_day(self, day):
        day = datetime.strptime(day, "%Y%m%d").date()
        self._require_day(day)
        previous = [candidate for candidate in self.days if candidate < day]
        if not previous:
            raise RuntimeError("日历未覆盖此前来源交易日")
        return max(previous).strftime("%Y%m%d")

    def window(self, now):
        if now.tzinfo is None:
            raise ValueError("日历时间必须携带时区")
        now = now.astimezone(self.zone)
        self._require_day(now.date())
        for owner, left, right in self._intervals:
            if left <= now < right:
                if owner is None:
                    raise RuntimeError("日历未覆盖夜盘所属的下一交易日")
                return TradingWindow(owner.strftime("%Y%m%d"), left, right)
        return None

    def _minutes(self, before_ns, after_ns=None):
        self._require_day(datetime.fromtimestamp((before_ns - 1) // 10**9, self.zone).date())
        if after_ns is not None:
            self._require_day(datetime.fromtimestamp(after_ns // 10**9, self.zone).date())
        for owner, left, right in self._intervals:
            start = int(left.timestamp()) * 10**9
            end = min(int(right.timestamp()) * 10**9, before_ns)
            if after_ns is not None:
                start = max(start, ((after_ns + 1) // MINUTE_NS) * MINUTE_NS)
            for stamp in range(start + MINUTE_NS - 1, end, MINUTE_NS):
                if after_ns is not None and stamp <= after_ns:
                    continue
                if owner is None:
                    raise RuntimeError("日历未覆盖该分钟所属的下一交易日")
                yield stamp

    def missing_minutes(self, after_ns, before_ns):
        return tuple(self._minutes(before_ns, after_ns))

    def last_minutes(self, before_ns, count):
        if type(count) is not int or not 0 <= count <= 10000:
            raise ValueError("历史分钟窗口须为0至10000整数")
        if not count:
            return ()
        stamps = tuple(self._minutes(before_ns))
        if len(stamps) < count:
            raise RuntimeError("日历覆盖范围内的完整交易分钟不足")
        return stamps[-count:]


# 保留旧导入名，均使用相同算法；没有独立品种类或内置市场时间。
CopperSessions = SessionCalendar
IndexSessions = SessionCalendar


def load_sessions(path, product):
    """产品只用于核对配置身份，不决定算法、时段或交易能力。"""
    return SessionCalendar(path, product)
