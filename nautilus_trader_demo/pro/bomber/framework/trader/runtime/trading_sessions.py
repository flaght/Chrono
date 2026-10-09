"""有日期覆盖范围的SHFE CU日夜盘日历；未知日期禁止猜测开盘。"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
from pathlib import Path

SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class TradingWindow:
    trading_day: str
    start: datetime
    end: datetime


class CopperSessions:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        if data.get("product") != "CU" or data.get("timezone") != "Asia/Shanghai":
            raise ValueError("首期持续会话仅支持北京时间SHFE CU日历")
        self.start = date.fromisoformat(data["start"])
        self.end = date.fromisoformat(data["end"])
        self.holidays = set(date.fromisoformat(item) for item in data["holidays"])
        self.no_night = set(date.fromisoformat(item) for item in data["no_night"])
        if self.start >= self.end or any(not self.start <= d <= self.end for d in self.holidays | self.no_night):
            raise ValueError("日历覆盖范围不合法")

    def is_trading_day(self, day):
        return self.start <= day <= self.end and day.weekday() < 5 and day not in self.holidays

    def previous_trading_day(self, day):
        candidate = datetime.strptime(day, "%Y%m%d").date() - timedelta(days=1)
        while candidate >= self.start:
            if self.is_trading_day(candidate):
                return candidate.strftime("%Y%m%d")
            candidate -= timedelta(days=1)
        raise RuntimeError("日历未覆盖此前来源交易日")

    def window(self, now):
        if now.tzinfo is None:
            raise ValueError("日历时间必须携带时区")
        now = now.astimezone(SHANGHAI)
        day = now.date()
        if not self.start <= day <= self.end:
            raise RuntimeError("交易日历已过期或未覆盖当前日期，须更新日历后启动")
        if self.is_trading_day(day):
            for begin, end in ((time(9), time(10, 15)), (time(10, 30), time(11, 30)),
                               (time(13, 30), time(15))):
                left = datetime.combine(day, begin, SHANGHAI)
                right = datetime.combine(day, end, SHANGHAI)
                if left <= now < right:
                    return TradingWindow(day.strftime("%Y%m%d"), left, right)
        anchor = day if now.time() >= time(21) else day - timedelta(days=1)
        following = anchor + timedelta(days=1)
        while following.weekday() >= 5:
            following += timedelta(days=1)
        if (self.is_trading_day(anchor) and anchor not in self.no_night
                and self.is_trading_day(following)):
            left = datetime.combine(anchor, time(21), SHANGHAI)
            right = datetime.combine(anchor + timedelta(days=1), time(1), SHANGHAI)
            if left <= now < right:
                return TradingWindow(following.strftime("%Y%m%d"), left, right)
        return None

    def missing_minutes(self, after_ns, before_ns):
        """只枚举交易窗口内已经完整结束的分钟；不补造无成交行情。"""
        minute = 60_000_000_000
        first = ((after_ns + 1) // minute) * minute
        for stamp in range(first, before_ns // minute * minute, minute):
            moment = datetime.fromtimestamp(stamp / 1e9, SHANGHAI)
            if self.window(moment) is not None:
                yield stamp + minute - 1

    def last_minutes(self, before_ns, count):
        """最近count根完整交易分钟；日历范围不足时不猜测未知交易日。"""
        if type(count) is not int or not 0 <= count <= 10000:
            raise ValueError("历史分钟窗口须为0至10000整数")
        result = []
        minute = 60_000_000_000
        stamp = before_ns // minute * minute - minute
        while len(result) < count:
            moment = datetime.fromtimestamp(stamp / 1e9, SHANGHAI)
            if self.window(moment) is not None:
                result.append(stamp + minute - 1)
            stamp -= minute
        return tuple(reversed(result))


class IndexSessions:
    """用户声明覆盖区间及全部IM交易日期；不推断未知节假日。"""
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        if data.get("product") != "IM" or data.get("timezone") != "Asia/Shanghai":
            raise ValueError("IM日历须声明product=IM及Asia/Shanghai")
        self.start, self.end = date.fromisoformat(data["start"]), date.fromisoformat(data["end"])
        days = [date.fromisoformat(item) for item in data["trading_days"]]
        if (self.start > self.end or not days or len(set(days)) != len(days)
                or any(not self.start <= day <= self.end or day.weekday() >= 5 for day in days)):
            raise ValueError("IM日历范围或显式交易日期无效")
        self.days = set(days)

    def is_trading_day(self, day):
        if not self.start <= day <= self.end:
            raise RuntimeError("IM日历未覆盖日期，须补充权威日历")
        return day in self.days

    def previous_trading_day(self, day):
        day = datetime.strptime(day, "%Y%m%d").date()
        self.is_trading_day(day)
        previous = [candidate for candidate in self.days if candidate < day]
        if not previous:
            raise RuntimeError("IM日历未覆盖此前来源交易日")
        return max(previous).strftime("%Y%m%d")

    def window(self, now):
        if now.tzinfo is None:
            raise ValueError("日历时间必须携带时区")
        now = now.astimezone(SHANGHAI)
        if self.is_trading_day(now.date()):
            for begin, end in ((time(9, 30), time(11, 30)), (time(13), time(15))):
                left, right = (datetime.combine(now.date(), value, SHANGHAI) for value in (begin, end))
                if left <= now < right:
                    return TradingWindow(now.strftime("%Y%m%d"), left, right)
        return None

    def missing_minutes(self, after_ns, before_ns):
        return tuple(self._minutes(after_ns, before_ns))

    def _minutes(self, after_ns, before_ns):
        minute = 60_000_000_000
        for stamp in range(((after_ns + 1) // minute + 1) * minute - 1, before_ns, minute):
            moment = datetime.fromtimestamp((stamp // minute * minute) // 10**9, SHANGHAI)
            if self.window(moment) is not None:
                yield stamp

    def last_minutes(self, before_ns, count):
        if type(count) is not int or not 0 <= count <= 10000:
            raise ValueError("IM历史分钟须为0至10000")
        if not count:
            return ()
        minute = 60_000_000_000
        stamp = (before_ns // minute) * minute - 1
        result = []
        while len(result) < count:
            moment = datetime.fromtimestamp((stamp // minute * minute) // 10**9, SHANGHAI)
            if self.window(moment) is not None:
                result.append(stamp)
            stamp -= minute
        return tuple(reversed(result))


def load_sessions(path, product):
    """按品种选择会话日历；新增品种不沿用CU开盘时间。"""
    if product.upper() == "CU":
        return CopperSessions(path)
    if product.upper() == "IM":
        return IndexSessions(path)
    raise ValueError("历史日历当前仅支持CU或IM")
