"""将统一行情Feed桥接到需要逐事件推进的执行Backend。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from market.basic.base import (
    Bar,
    DataType,
    InstrumentId,
    MarketDataFeed,
    QuoteTick,
    SubscriptionRequest,
    TradeTick,
)
from strategy.execution.ports import SimExecutionBackendPort


_SUPPORTED_DATA_TYPES = frozenset(
    {DataType.QUOTE_TICK, DataType.TRADE_TICK, DataType.BAR},
)


@dataclass(frozen=True)
class MarketStreamBinding:
    """声明一个Feed事件到模拟Backend的行情绑定。

    E8只允许Nautilus原生认识的QuoteTick、TradeTick和Bar。CustomBar中的
    因子仍由策略链路消费，不能作为第二份行情再次推进撮合时钟。
    """

    instrument_id: InstrumentId | str
    data_type: DataType
    bar_spec: str | None = None
    fields: tuple[str, ...] = ()
    extra_params: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        instrument_id = self.instrument_id
        if isinstance(instrument_id, str):
            instrument_id = InstrumentId.from_str(instrument_id)
        if self.data_type not in _SUPPORTED_DATA_TYPES:
            raise ValueError(
                "Nautilus行情桥只支持QUOTE_TICK、TRADE_TICK和BAR，"
                f"不支持{self.data_type.name}",
            )
        bar_spec = self.bar_spec.strip().upper() if self.bar_spec else None
        if self.data_type is DataType.BAR and not bar_spec:
            raise ValueError("BAR绑定必须提供bar_spec，例如1-MINUTE")
        if self.data_type is not DataType.BAR and bar_spec is not None:
            raise ValueError("只有BAR绑定可以提供bar_spec")
        object.__setattr__(self, "instrument_id", instrument_id)
        object.__setattr__(self, "bar_spec", bar_spec)
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "extra_params", dict(self.extra_params))

    def to_subscription(self) -> SubscriptionRequest:
        return SubscriptionRequest(
            instrument_id=self.instrument_id,
            data_type=self.data_type,
            bar_spec=self.bar_spec,
            fields=self.fields,
            extra_params=self.extra_params,
        )


class NautilusMarketFeedAdapter:
    """把MarketDataFeed标准事件串行送入模拟Backend。

    网络线程只负责把数据放入Feed队列；Feed分派线程调用本适配器；本适配器再用
    一把锁保证BacktestEngine不会被多个行情回调并发推进。策略是否同时订阅同一
    Feed由组合层决定，本类不包含任何策略逻辑和订单逻辑。
    """

    def __init__(
        self,
        adapter_id: str,
        feed: MarketDataFeed,
        backend: SimExecutionBackendPort,
        bindings: Sequence[MarketStreamBinding],
    ) -> None:
        if not adapter_id.strip():
            raise ValueError("adapter_id不能为空")
        normalized = tuple(bindings)
        if not normalized:
            raise ValueError("bindings不能为空")
        if len(set(normalized)) != len(normalized):
            raise ValueError("bindings不能重复")

        self.adapter_id = adapter_id
        self.feed = feed
        self.backend = backend
        self.bindings = normalized
        self._lifecycle_lock = threading.RLock()
        self._process_lock = threading.RLock()
        self._error_handlers: list[Callable[[Exception, Any], None]] = []
        self._handlers_attached = False
        self._started = False
        self._stopped = False
        self._events_processed = 0
        self._reports_received = 0
        self._last_error: Exception | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def reports_received(self) -> int:
        return self._reports_received

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def register_error_handler(
        self,
        handler: Callable[[Exception, Any], None],
    ) -> None:
        if handler not in self._error_handlers:
            self._error_handlers.append(handler)

    def start(self) -> None:
        """完成订阅、启动Backend并连接Feed；重复启动安全。"""

        with self._lifecycle_lock:
            if self._started:
                return
            if self._stopped:
                raise RuntimeError("已停止的行情适配器不能重新启动")
            self._attach_handlers()
            for binding in self.bindings:
                self.feed.subscribe(
                    binding.instrument_id,
                    binding.data_type,
                    bar_spec=binding.bar_spec,
                    fields=binding.fields,
                    **dict(binding.extra_params),
                )
            self.backend.start()
            # 必须在connect前开放事件入口。部分Feed会在建立连接或恢复订阅时立刻
            # 收到首包行情，若connect返回后才置位会无声丢失第一份事件。
            self._started = True
            try:
                self.feed.connect()
            except Exception:
                self._started = False
                self.backend.stop()
                self._stopped = True
                raise

    def stop(self) -> None:
        """先停止行情输入，再结束Backend；重复调用安全。"""

        with self._lifecycle_lock:
            if self._stopped:
                return
            if not self._started:
                return
            # 先关闭事件闸门，但不能持锁等待Feed分派线程退出，否则分派线程若正
            # 准备进入回调会与disconnect.join形成死锁。
            self._started = False
        try:
            self.feed.disconnect()
        finally:
            try:
                # 等待已经进入的单个行情推进完成，再安全释放Backend。
                with self._process_lock:
                    self.backend.stop()
            finally:
                with self._lifecycle_lock:
                    self._stopped = True

    def _attach_handlers(self) -> None:
        if self._handlers_attached:
            return
        requested = {binding.data_type for binding in self.bindings}
        if DataType.QUOTE_TICK in requested:
            self.feed.register_quote_tick_handler(self._on_market_event)
        if DataType.TRADE_TICK in requested:
            self.feed.register_trade_tick_handler(self._on_market_event)
        if DataType.BAR in requested:
            self.feed.register_bar_handler(self._on_market_event)
        self._handlers_attached = True

    def _on_market_event(self, event: QuoteTick | TradeTick | Bar) -> None:
        if not self._matches_binding(event):
            return
        with self._process_lock:
            if not self._started:
                return
            try:
                reports = self.backend.process_market_event(event)
            except Exception as exc:
                self._last_error = exc
                for handler in tuple(self._error_handlers):
                    handler(exc, event)
                raise
            self._events_processed += 1
            self._reports_received += len(tuple(reports))

    def _matches_binding(self, event: QuoteTick | TradeTick | Bar) -> bool:
        if isinstance(event, QuoteTick):
            event_type = DataType.QUOTE_TICK
            instrument_id = event.instrument_id
            bar_spec = None
        elif isinstance(event, TradeTick):
            event_type = DataType.TRADE_TICK
            instrument_id = event.instrument_id
            bar_spec = None
        elif isinstance(event, Bar):
            event_type = DataType.BAR
            instrument_id = event.bar_type.instrument_id
            bar_type_text = str(event.bar_type).upper()
            return any(
                binding.data_type is event_type
                and binding.instrument_id == instrument_id
                and f"-{binding.bar_spec}-" in bar_type_text
                for binding in self.bindings
            )
        else:
            return False
        return any(
            binding.data_type is event_type
            and binding.instrument_id == instrument_id
            and binding.bar_spec is bar_spec
            for binding in self.bindings
        )
