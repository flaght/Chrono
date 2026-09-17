import logging
import queue
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Protocol

from market.basic.base import (
    Bar,
    CustomBar,
    InstrumentId,
    QuoteTick,
    TradeTick,
    make_bar,
    make_custom_bar,
    make_custom_bar_all_in_one,
    make_quote_tick,
    make_trade_tick,
)
from market.basic.base import  MarketDataFeed

logger = logging.getLogger("StreamDataFeed")


class StreamDataFeed(MarketDataFeed):
    """B2类：实时流式行情数据源基类。

    提供线程安全缓冲队列（解耦网络 IO 线程与策略消费）与后台事件消费分派循环。
    """

    def __init__(self, source_id: str, queue_size: int = 100_000) -> None:
        super().__init__(source_id=source_id)
        self._event_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def connect(self) -> None:
        if self._is_connected:
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._dispatch_loop,
            name=f"{self.source_id}-Dispatcher",
            daemon=True,
        )
        self._worker_thread.start()
        self._start_network_client()
        self._is_connected = True
        logger.info(f"[{self.source_id}] 实时行情数据源已连接。")

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        self._stop_event.set()
        self._stop_network_client()
        if self._worker_thread and self._worker_thread.is_alive():
            self._event_queue.put(None)
            self._worker_thread.join(timeout=2.0)
        self._is_connected = False
        logger.info(f"[{self.source_id}] 实时行情数据源已安全断开。")

    def enqueue_event(self, event: Any) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"[{self.source_id}] 队列满，丢弃历史事件！")

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._event_queue.get(timeout=0.2)
                if item is None:
                    break

                if isinstance(item, TradeTick):
                    self._emit_trade_tick(item)
                elif isinstance(item, QuoteTick):
                    self._emit_quote_tick(item)
                elif isinstance(item, CustomBar):
                    self._emit_custom_bar(item)
                elif isinstance(item, Bar):
                    self._emit_bar(item)

                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"[{self.source_id}] 事件分派异常: {exc}", exc_info=True)

    def _start_network_client(self) -> None:
        raise NotImplementedError

    def _stop_network_client(self) -> None:
        raise NotImplementedError