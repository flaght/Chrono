"""来源无关的已收盘分钟帧；估值时间、接收时间和执行请求时间独立保存。"""
from dataclasses import dataclass
from decimal import Decimal
import math
from types import MappingProxyType

MINUTE_NS = 60_000_000_000


def minute_close_ns(timestamp, label="end"):
    """end为收盘边界，inclusive-end为边界前1ns，start为分钟开始标签。"""
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("分钟时间须为非负整数纳秒")
    if label == "inclusive-end":
        if (timestamp + 1) % MINUTE_NS:
            raise ValueError("inclusive-end须为分钟边界前1ns")
        return timestamp + 1
    if label not in {"start", "end"} or timestamp % MINUTE_NS:
        raise ValueError("Bar标签须为明确的分钟边界")
    return timestamp + MINUTE_NS if label == "start" else timestamp


@dataclass(frozen=True)
class FrameObservation:
    price: Decimal
    source_id: str
    source_ns: int
    received_ns: int


@dataclass(frozen=True)
class MinuteFrame:
    valuation_ns: int
    request_ns: int
    observations: object

    def __post_init__(self):
        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))


class MinuteFrameBuffer:
    """给各来源等待同一分钟的接收宽限期；封闭后的迟到数据不会追溯决策。

    输入必须是已收盘Bar。截止时间从本机接收时刻计算，与历史估值日期无关。
    调用方负责串行调用；不前填候选价格，不制造零成交Bar。
    """
    def __init__(self, sources, *, grace_seconds=2, max_pending_frames=32):
        if not sources or any(not key or not value for key, value in sources.items()):
            raise ValueError("行情键须明确绑定唯一来源")
        if not math.isfinite(grace_seconds) or not 0 <= grace_seconds <= 30 or max_pending_frames < 1:
            raise ValueError("分钟等待须在[0,30]秒且缓冲上限为正")
        self.sources = dict(sources)
        self.grace_ns = int(grace_seconds * 1e9)
        self.max_pending_frames = max_pending_frames
        self.pending = {}
        self.last_closed_ns = -1
        self.late_count = 0

    def add(self, key, price, valuation_ns, received_ns, *, source_id, source_ns=None):
        if key not in self.sources or source_id != self.sources[key]:
            raise ValueError("行情来源与绑定不一致")
        if valuation_ns % MINUTE_NS or valuation_ns < 0 or received_ns < 0:
            raise ValueError("行情帧时间无效")
        if valuation_ns <= self.last_closed_ns:
            self.late_count += 1
            return False
        price = Decimal(str(price))
        if not price.is_finite() or price <= 0:
            raise ValueError("行情价格须为有限正数")
        if valuation_ns not in self.pending:
            if len(self.pending) >= self.max_pending_frames:
                raise RuntimeError("行情帧缓冲已满，来源时间可能不一致")
            self.pending[valuation_ns] = [received_ns + self.grace_ns, {}]
        elif received_ns > self.pending[valuation_ns][0]:
            # 控制线程尚未flush也不能延长接收期限；边界时刻允许到达。
            self.late_count += 1
            return False
        row = FrameObservation(price, source_id, valuation_ns if source_ns is None else source_ns, received_ns)
        observations = self.pending[valuation_ns][1]
        if key in observations:
            if row.price != observations[key].price or row.source_ns != observations[key].source_ns:
                raise ValueError("同一来源同一分钟出现冲突Bar")
            return False
        observations[key] = row
        return True

    def close_due(self, now_ns):
        frames = []
        for timestamp in sorted(tuple(self.pending)):
            deadline, rows = self.pending[timestamp]
            if now_ns < deadline:
                break
            frames.append(MinuteFrame(timestamp, now_ns, rows))
            del self.pending[timestamp]
            self.last_closed_ns = timestamp
        return tuple(frames)
