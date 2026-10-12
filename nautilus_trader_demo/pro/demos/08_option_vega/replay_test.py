"""08专用Replay工程时间映射；原始Feed及策略算法保持原时间契约。"""
from datetime import date, datetime
import logging
from threading import RLock
from zoneinfo import ZoneInfo

MINUTE_NS = 60_000_000_000
ZONE = ZoneInfo("Asia/Shanghai")
logger = logging.getLogger("OptionReplayTest")


def parse_minute(value):
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
        if parsed.strftime("%H:%M") != value:
            raise ValueError()
        return parsed
    except (TypeError, ValueError):
        raise ValueError("Replay测试时刻须为HH:MM") from None


class ReplayTestTimeMap:
    """以第一份完成Bar的自然日建立唯一偏移，所有合约共享；不修改原Bar。"""
    def __init__(self, valuation_day, target_time, source_time=None):
        self.day = date.fromisoformat(valuation_day) if isinstance(valuation_day, str) else valuation_day
        self.target_time = parse_minute(target_time)
        self.source_time = None if source_time is None else parse_minute(source_time)
        self._lock = RLock()
        self._source_day = self._offset_ns = self._source_anchor_ns = None
        self._samples = []
        self._count = 0

    @property
    def identity(self):
        return {"kind": "replay_test_mapped", "valuation_day": self.day.isoformat(),
            "target_time": self.target_time.strftime("%H:%M"),
            "source_time": None if self.source_time is None else self.source_time.strftime("%H:%M"),
            "anchor": "first_completed_bar" if self.source_time is None else "explicit_source_time",
            "scope": "engineering_chain"}

    @property
    def snapshot(self):
        with self._lock:
            return {**self.identity, "source_anchor_ns": self._source_anchor_ns,
                "source_day": None if self._source_day is None else self._source_day.isoformat(),
                "offset_ns": self._offset_ns, "bars_mapped": self._count,
                "samples": [dict(row) for row in self._samples]}

    def map_bar(self, bar):
        from bomber.framework.market.basic.base import Bar
        with self._lock:
            source = bar.ts_event
            if (source + 1) % MINUTE_NS:
                raise ValueError("Replay测试只映射CTP已完成分钟Bar")
            source_day = datetime.fromtimestamp(source // 1_000_000_000, ZONE).date()
            if self._offset_ns is None:
                target = int(datetime.combine(self.day, self.target_time, ZONE).timestamp()) * 1_000_000_000 - 1
                anchor = source if self.source_time is None else int(datetime.combine(
                    source_day, self.source_time, ZONE).timestamp()) * 1_000_000_000 - 1
                self._source_day, self._source_anchor_ns, self._offset_ns = source_day, anchor, target - anchor
                logger.info("工程时间映射已建立: source=%s target=%s offset_minutes=%s",
                    datetime.fromtimestamp((anchor+1)//1_000_000_000, ZONE).isoformat(),
                    datetime.fromtimestamp((target+1)//1_000_000_000, ZONE).isoformat(), self._offset_ns // MINUTE_NS)
            if source_day != self._source_day:
                raise ValueError("Replay测试原始事件跨日，须停止并重新建立时间映射")
            mapped = source + self._offset_ns
            if datetime.fromtimestamp(mapped // 1_000_000_000, ZONE).date() != self.day:
                raise ValueError("Replay测试映射结果跨出估值日")
            result = Bar(bar_type=bar.bar_type, open=bar.open, high=bar.high, low=bar.low,
                close=bar.close, volume=bar.volume, ts_event=mapped, ts_init=bar.ts_init)
            self._count += 1
            self._samples.append({"instrument": str(bar.bar_type.instrument_id),
                "original_event_ns": source, "mapped_event_ns": mapped, "ts_init": bar.ts_init})
            del self._samples[:-32]
            return result


def mapped_bar_feed(source_id, upstream, *, price_type, time_map):
    """复用公共聚合器，只在08向下游发出完成Bar时平移业务时间。"""
    from bomber.framework.market.stream.aggregation import TradeTickBarFeed, QuoteMidBarFeed
    base = {"last": TradeTickBarFeed, "mid": QuoteMidBarFeed}[price_type]

    class ReplayTestBarFeed(base):
        def _emit_bar(self, bar):
            super()._emit_bar(self.time_map.map_bar(bar))

    result = ReplayTestBarFeed(source_id, upstream)
    result.time_map = time_map
    return result
