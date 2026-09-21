import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from market.basic.base import (
    DataType,
    InstrumentId,
    SubscriptionRequest,
    make_quote_tick,
    make_trade_tick,
)
from market.stream.base import StreamDataFeed
from market.stream.health import StreamHealthConfig

logger = logging.getLogger("BNWSStreamDataFeed")


@dataclass
class BNWSConfig:
    ws_base_url: str = "wss://stream.binance.com:9443"
    api_key: str | None = None
    secret_key: str | None = None


class BNWSStreamDataFeed(StreamDataFeed):
    """web socket 实时行情数据源。"""

    def __init__(
        self,
        config: BNWSConfig | None = None,
        source_id: str = "BINANCE_LIVE_SOURCE",
        queue_size: int = 100_000,
        health_config: StreamHealthConfig | None = None,
    ) -> None:
        super().__init__(
            source_id=source_id,
            queue_size=queue_size,
            health_config=health_config,
        )
        self.config = config or BNWSConfig()
        self._active_streams: set[str] = set()
        self._ws_app: Any = None
        self._net_thread: threading.Thread | None = None
        self._last_ws_error: str | None = None

    @property
    def last_ws_error(self) -> str | None:
        """返回最近一次 WebSocket 错误，供监控和测试诊断。"""
        return self._last_ws_error

    def _start_network_client(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError(
                "Binance 实时行情需要 websocket-client；"
                "请执行 `uv pip install --link-mode=copy websocket-client`",
            ) from exc

        # Runner 会先声明订阅、再连接 Feed。未连接时 MarketDataFeed 只保存
        # SubscriptionRequest，不调用网络钩子，因此启动时必须从声明式订阅
        # 恢复组合流名称。
        configured_streams = set(self._active_streams)
        for requests in self._subscriptions.values():
            configured_streams.update(self._stream_name(request) for request in requests)
        self._active_streams = configured_streams
        streams = sorted(configured_streams)
        if not streams:
            raise RuntimeError("连接 Binance 前至少需要一个行情订阅")
        stream_path = "/".join(streams)
        ws_url = f"{self.config.ws_base_url.rstrip('/')}/stream?streams={stream_path}"
        logger.info("正在连接 Binance WebSocket 流: %s", ws_url)

        def on_open(ws: Any) -> None:
            del ws
            logger.info("Binance WebSocket 已连接")

        def on_message(ws: Any, message: str) -> None:
            del ws
            try:
                data = json.loads(message)
                self.on_ws_message(data.get("data", data))
            except Exception:
                logger.exception("Binance WebSocket 报文处理失败")

        def on_error(ws: Any, error: Any) -> None:
            del ws
            self._last_ws_error = str(error)
            self.report_stream_interruption(f"Binance WebSocket异常: {error}")
            logger.error("Binance WebSocket 异常: %s", error)

        def on_close(ws: Any, status: Any, message: Any) -> None:
            del ws
            if not self._stop_event.is_set():
                self.report_stream_interruption(
                    f"Binance WebSocket关闭: status={status} message={message}",
                )
            logger.info(
                "Binance WebSocket 已关闭: status=%s message=%s",
                status,
                message,
            )

        self._last_ws_error = None
        self._ws_app = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._net_thread = threading.Thread(
            target=self._ws_app.run_forever,
            name="Binance-NetThread",
            daemon=True,
        )
        self._net_thread.start()

    def _stop_network_client(self) -> None:
        ws_app = self._ws_app
        if ws_app is not None:
            ws_app.close()
        thread = self._net_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._ws_app = None
        self._net_thread = None
        logger.info("Binance WebSocket 已安全断开")

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self._active_streams.add(self._stream_name(request))

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        self._active_streams.discard(self._stream_name(request))

    @staticmethod
    def _stream_name(request: SubscriptionRequest) -> str:
        raw_symbol = str(request.instrument_id).split(".")[0].lower()
        if request.data_type == DataType.TRADE_TICK:
            return f"{raw_symbol}@trade"
        if request.data_type == DataType.QUOTE_TICK:
            return f"{raw_symbol}@bookTicker"
        return f"{raw_symbol}@kline_1m"

    def on_ws_message(self, message: str | dict[str, Any]) -> None:
        if isinstance(message, str):
            data = json.loads(message)
        else:
            data = message
        event_type = data.get("e")

        # 1. 逐笔成交帧 (trade)
        if event_type == "trade":
            symbol = data["s"].upper()
            inst_id = InstrumentId.from_str(f"{symbol}.BINANCE")
            trade_time_ms = int(data["T"])
            ts_ns = trade_time_ms * 1_000_000
            meta = self.get_instrument_meta(inst_id)

            tick = make_trade_tick(
                instrument_id=inst_id,
                price=float(data["p"]),
                size=float(data["q"]),
                trade_id=str(data["t"]),
                ts_event=ts_ns,
                ts_init=ts_ns,
                meta=meta,
            )
            self.enqueue_event(tick)

        # 2. 最优挂单帧 (bookTicker)
        elif "b" in data and "a" in data and "s" in data:
            symbol = data["s"].upper()
            inst_id = InstrumentId.from_str(f"{symbol}.BINANCE")
            event_time_ms = data.get("E")
            ts_ns = (
                int(event_time_ms) * 1_000_000
                if event_time_ms is not None
                else time.time_ns()
            )
            meta = self.get_instrument_meta(inst_id)
            quote = make_quote_tick(
                instrument_id=inst_id,
                bid_price=float(data["b"]),
                ask_price=float(data["a"]),
                bid_size=float(data["B"]),
                ask_size=float(data["A"]),
                ts_event=ts_ns,
                ts_init=ts_ns,
                meta=meta,
            )
            self.enqueue_event(quote)
