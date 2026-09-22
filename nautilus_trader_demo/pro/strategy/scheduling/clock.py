"""可替换的时钟 Feed；历史回放与在线外部时钟使用同一策略回调。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from market.basic.base import Bar, CustomBar, MarketDataFeed, QuoteTick, SubscriptionRequest, TradeTick
from market.replay.base import FileReplayFeed, ReplaySummary


class ManualClockFeed(MarketDataFeed):
    """在线时钟端口；由外部调度器调用 emit_time，自己不读取 Bar。"""

    def __init__(self, source_id: str = "MANUAL_CLOCK") -> None:
        super().__init__(source_id=source_id)
        self._time_handlers: list[Callable[[int], None]] = []
        self._last_ns = -1

    def register_time_handler(self, handler: Callable[[int], None]) -> None:
        self._time_handlers.append(handler)

    def subscribe(self, *args: object, **kwargs: object) -> None:
        raise TypeError("ManualClockFeed仅提供时钟事件，不接受行情订阅")

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        raise TypeError("ManualClockFeed仅提供时钟事件，不接受行情订阅")

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        raise TypeError("ManualClockFeed没有可取消的行情订阅")

    def emit_time(self, ts_event: int) -> None:
        if not self._is_connected:
            raise RuntimeError("时钟尚未连接")
        if ts_event <= self._last_ns:
            raise ValueError("时钟时间必须严格递增")
        self._last_ns = ts_event
        for handler in tuple(self._time_handlers):
            handler(ts_event)


class TimedFileReplayFeed(FileReplayFeed):
    """合并文件行情与计划时点；同一时刻先推行情，后触发目标。"""

    def __init__(self, slots: Iterable[int], source_id: str = "TIMED_FILE_REPLAY") -> None:
        super().__init__(source_id)
        self._slots = tuple(sorted(set(slots)))
        if any(slot <= 0 for slot in self._slots):
            raise ValueError("计划时点必须为正")
        self._time_handlers: list[Callable[[int], None]] = []

    def register_time_handler(self, handler: Callable[[int], None]) -> None:
        self._time_handlers.append(handler)

    def _emit_time(self, slot: int) -> None:
        for handler in tuple(self._time_handlers):
            handler(slot)

    def replay(self) -> ReplaySummary:
        trades = quotes = bars = custom_bars = 0
        slot_index = 0
        events = self.load_events()
        for index, event in enumerate(events):
            # 当前事件之前的计划必须先触发；相同时间戳的一组行情则全部先处理。
            while slot_index < len(self._slots) and self._slots[slot_index] < event.ts_init:
                self._emit_time(self._slots[slot_index])
                slot_index += 1
            if isinstance(event, TradeTick):
                self._emit_trade_tick(event)
                trades += 1
            elif isinstance(event, QuoteTick):
                self._emit_quote_tick(event)
                quotes += 1
            elif isinstance(event, CustomBar):
                self._emit_custom_bar(event)
                custom_bars += 1
            elif isinstance(event, Bar):
                self._emit_bar(event)
                bars += 1
            next_ns = events[index + 1].ts_init if index + 1 < len(events) else None
            if next_ns != event.ts_init:
                while slot_index < len(self._slots) and self._slots[slot_index] == event.ts_init:
                    self._emit_time(self._slots[slot_index])
                    slot_index += 1
        # 回放结束之后的计划不能假装已执行；策略会将其审计为未到达。
        return ReplaySummary(trades, quotes, bars, custom_bars)
