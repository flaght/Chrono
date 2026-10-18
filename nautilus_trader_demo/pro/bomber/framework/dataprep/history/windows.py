"""用户指定IM交易日的分钟窗口；不是自动推断节假日的全年日历。"""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from .base import MINUTE_NS, HistoryError


class ImDayWindow:
    def __init__(self, trading_day):
        if type(trading_day) is not date or trading_day.weekday() >= 5:
            raise HistoryError("须声明工作日IM交易日期；真实休市日由权威数据核验")
        self.day = trading_day
        zone = ZoneInfo("Asia/Shanghai")
        stamps = []
        for start, end in ((time(9, 30), time(11, 30)), (time(13), time(15))):
            left = int(datetime.combine(trading_day, start, zone).timestamp()) * 10**9
            right = int(datetime.combine(trading_day, end, zone).timestamp()) * 10**9
            stamps.extend(stamp + MINUTE_NS - 1 for stamp in range(left, right, MINUTE_NS))
        self.stamps = tuple(stamps)
        self.end_ns = self.stamps[-1] + 1

    def last_minutes(self, before_ns, count):
        if type(count) is not int or not 0 <= count <= 240:
            raise HistoryError("单指定交易日窗口须为0至240分钟")
        available = tuple(stamp for stamp in self.stamps if stamp < before_ns)
        if len(available) < count:
            raise HistoryError("指定交易日截止前的完整分钟不足，不能推断此前日期")
        return available[-count:] if count else ()

    def missing_minutes(self, after_ns, before_ns):
        return tuple(stamp for stamp in self.stamps if after_ns < stamp < before_ns)
