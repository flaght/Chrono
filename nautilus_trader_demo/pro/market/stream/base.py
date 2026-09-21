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
from market.stream.health import (
    HealthHandler,
    MarketHealthReason,
    MarketHealthSnapshot,
    MarketHealthState,
    StreamHealthConfig,
    StreamHealthMonitor,
)

logger = logging.getLogger("StreamDataFeed")


class StreamDataFeed(MarketDataFeed):
    """B2类：实时流式行情数据源基类。

    提供线程安全缓冲队列（解耦网络 IO 线程与策略消费）与后台事件消费分派循环。
    """

    def __init__(
        self,
        source_id: str,
        queue_size: int = 100_000,
        health_config: StreamHealthConfig | None = None,
    ) -> None:
        super().__init__(source_id=source_id)
        self._event_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._health_monitor = StreamHealthMonitor(source_id, health_config)

    @property
    def health_state(self) -> MarketHealthState:
        return self._health_monitor.snapshot.state

    @property
    def health_snapshot(self) -> MarketHealthSnapshot:
        return self._health_monitor.snapshot

    def register_health_handler(self, handler: HealthHandler) -> None:
        self._health_monitor.register_handler(handler)

    def acknowledge_health_degradation(self) -> MarketHealthSnapshot:
        return self._health_monitor.acknowledge_degradation()

    def report_stream_interruption(self, detail: str) -> None:
        """供具体网络驱动在断线/异常回调中上报流中断。"""
        if self._stop_event.is_set():
            return
        self._health_monitor.on_stream_interrupted(detail)

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
        self._health_monitor.on_connected()
        try:
            self._start_network_client()
        except Exception as exc:
            self._stop_event.set()
            self._signal_dispatch_stop()
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
            self._health_monitor.on_disconnected(
                MarketHealthReason.CONNECTION_ERROR,
                f"行情连接失败: {exc}",
            )
            raise
        self._is_connected = True
        logger.info(f"[{self.source_id}] 实时行情数据源已连接。")

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        self._stop_event.set()
        try:
            self._stop_network_client()
        finally:
            if self._worker_thread and self._worker_thread.is_alive():
                self._signal_dispatch_stop()
                self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
            self._is_connected = False
            self._health_monitor.on_disconnected()
            logger.info(f"[{self.source_id}] 实时行情数据源已安全断开。")

    def enqueue_event(self, event: Any) -> None:
        if not self._health_monitor.on_event(event):
            logger.warning(
                "[%s] 丢弃时间戳回退事件: %s",
                self.source_id,
                self._health_monitor.snapshot.detail,
            )
            return
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            self._health_monitor.on_queue_overflow()
            logger.warning(f"[{self.source_id}] 队列满，丢弃实时事件！")

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                self._health_monitor.check_timeout()
                continue
            if item is None:
                self._event_queue.task_done()
                break
            try:
                if isinstance(item, TradeTick):
                    self._emit_trade_tick(item)
                elif isinstance(item, QuoteTick):
                    self._emit_quote_tick(item)
                elif isinstance(item, CustomBar):
                    self._emit_custom_bar(item)
                elif isinstance(item, Bar):
                    self._emit_bar(item)
            except Exception as exc:
                self._health_monitor.on_dispatch_error(exc)
                logger.error(f"[{self.source_id}] 事件分派异常: {exc}", exc_info=True)
            finally:
                self._event_queue.task_done()

    def _signal_dispatch_stop(self) -> None:
        try:
            self._event_queue.put_nowait(None)
        except queue.Full:
            # stop_event已置位，消费者结束当前回调后会自行退出；不能在这里阻塞。
            pass

    def _start_network_client(self) -> None:
        raise NotImplementedError

    def _stop_network_client(self) -> None:
        raise NotImplementedError
