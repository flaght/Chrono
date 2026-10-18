"""历史行情请求及Provider契约；策略不直接连接存储后端。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

MINUTE_NS = 60_000_000_000


class HistoryError(ValueError):
    pass


class HistoryUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryRequest:
    instrument_id: str
    start_ns: int
    end_ns: int
    max_rows: int = 10000

    def __post_init__(self):
        if (not self.instrument_id or self.start_ns < 0 or self.end_ns <= self.start_ns
                or not 1 <= self.max_rows <= 1000000):
            raise HistoryError("历史请求合约、区间或行数上限无效")


@dataclass(frozen=True)
class HistoryBar:
    instrument_id: str
    ts_event: int
    close: Decimal | None = None
    adjusted_close: Decimal | None = None
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    volume: Decimal | None = None
    cumulative_factor: Decimal | None = None

    def __post_init__(self):
        if not self.instrument_id or type(self.ts_event) is not int or (self.ts_event + 1) % MINUTE_NS:
            raise HistoryError("历史分钟须使用完整分钟末最后1ns及真实合约标识")
        if self.close is None and self.adjusted_close is None:
            raise HistoryError("历史分钟缺少收盘价")
        for name in ("open", "high", "low", "close", "adjusted_close", "volume", "cumulative_factor"):
            value = getattr(self, name)
            if value is not None:
                value = Decimal(str(value))
                if not value.is_finite() or (value < 0 if name == "volume" else value <= 0):
                    raise HistoryError(f"历史分钟{name}无效")
                object.__setattr__(self, name, value)
        if self.adjusted_close is None and self.close is not None and self.cumulative_factor is not None:
            object.__setattr__(self, "adjusted_close", self.close * self.cumulative_factor)
        if self.high is not None and self.low is not None and self.high < self.low:
            raise HistoryError("历史分钟最高价低于最低价")
        for value in (self.open, self.close):
            if value is not None and ((self.high is not None and value > self.high)
                    or (self.low is not None and value < self.low)):
                raise HistoryError("历史分钟OHLC范围不一致")


class HistoryProvider(ABC):
    storage_kind = "database"
    @abstractmethod
    def open(self): ...
    @abstractmethod
    def read(self, request: HistoryRequest) -> tuple[HistoryBar, ...]: ...
    @abstractmethod
    def close(self): ...


def select_bars(rows, request):
    """同键不同内容失败，完全重复去重；只接受请求范围内完整分钟。"""
    selected = {}
    for bar in rows:
        if bar.instrument_id != request.instrument_id or not request.start_ns <= bar.ts_event < request.end_ns:
            continue
        previous = selected.get(bar.ts_event)
        if previous is not None and previous != bar:
            raise HistoryError("同一分钟存在内容冲突，须按上游修订口径处理")
        selected[bar.ts_event] = bar
    if len(selected) > request.max_rows:
        raise HistoryError("历史结果超出声明行数上限")
    return tuple(selected[key] for key in sorted(selected))
