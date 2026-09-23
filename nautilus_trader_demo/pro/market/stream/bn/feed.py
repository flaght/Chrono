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
    make_bar,
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
    market_type: str = "spot"

    def __post_init__(self) -> None:
        normalized = self.market_type.strip().lower()
        if normalized not in {"spot", "futures"}:
            raise ValueError("market_type必须是spot或futures")
        self.market_type = normalized


_BAR_SPEC_TO_INTERVAL = {
    "1-MINUTE": "1m",
    "3-MINUTE": "3m",
    "5-MINUTE": "5m",
    "15-MINUTE": "15m",
    "30-MINUTE": "30m",
    "1-HOUR": "1h",
    "2-HOUR": "2h",
    "4-HOUR": "4h",
    "6-HOUR": "6h",
    "8-HOUR": "8h",
    "12-HOUR": "12h",
    "1-DAY": "1d",
}
_INTERVAL_TO_BAR_SPEC = {value: key for key, value in _BAR_SPEC_TO_INTERVAL.items()}


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
        self._diagnostic_lock = threading.Lock()
        self._ws_open = False
        self._ws_messages = 0
        self._kline_messages = 0
        self._closed_kline_messages = 0
        self._message_errors = 0
        self._last_message_error: str | None = None

    @property
    def last_ws_error(self) -> str | None:
        """返回最近一次 WebSocket 错误，供监控和测试诊断。"""
        return self._last_ws_error

    @property
    def message_diagnostics(self) -> dict[str, Any]:
        """只暴露计数和错误类型，不输出账户数据或完整行情报文。"""
        with self._diagnostic_lock:
            return {
                "ws_open": self._ws_open,
                "ws_messages": self._ws_messages,
                "kline_messages": self._kline_messages,
                "closed_kline_messages": self._closed_kline_messages,
                "message_errors": self._message_errors,
                "last_message_error": self._last_message_error,
            }

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
            with self._diagnostic_lock:
                self._ws_open = True
            logger.info("Binance WebSocket 已连接")

        def on_message(ws: Any, message: str) -> None:
            del ws
            with self._diagnostic_lock:
                self._ws_messages += 1
            try:
                data = json.loads(message)
                self.on_ws_message(data.get("data", data))
            except Exception as exc:
                with self._diagnostic_lock:
                    self._message_errors += 1
                    self._last_message_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Binance WebSocket 报文处理失败")

        def on_error(ws: Any, error: Any) -> None:
            del ws
            self._last_ws_error = str(error)
            self.report_stream_interruption(f"Binance WebSocket异常: {error}")
            logger.error("Binance WebSocket 异常: %s", error)

        def on_close(ws: Any, status: Any, message: Any) -> None:
            del ws
            with self._diagnostic_lock:
                self._ws_open = False
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
        with self._diagnostic_lock:
            self._ws_open = False
            self._ws_messages = 0
            self._kline_messages = 0
            self._closed_kline_messages = 0
            self._message_errors = 0
            self._last_message_error = None
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
        with self._diagnostic_lock:
            self._ws_open = False
        logger.info("Binance WebSocket 已安全断开")

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self._active_streams.add(self._stream_name(request))

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        self._active_streams.discard(self._stream_name(request))

    @staticmethod
    def _raw_symbol(instrument_id: InstrumentId) -> str:
        raw_symbol = str(instrument_id).split(".")[0]
        return raw_symbol.removesuffix("-PERP").lower()

    def _stream_name(self, request: SubscriptionRequest) -> str:
        instrument_symbol = str(request.instrument_id).split(".")[0].upper()
        is_perpetual = instrument_symbol.endswith("-PERP")
        if self.config.market_type == "futures" and not is_perpetual:
            raise ValueError(
                "futures Feed必须订阅*-PERP.BINANCE，避免与现货标识冲突",
            )
        if self.config.market_type == "spot" and is_perpetual:
            raise ValueError("spot Feed不能订阅永续合约标识")
        raw_symbol = self._raw_symbol(request.instrument_id)
        if request.data_type == DataType.TRADE_TICK:
            return f"{raw_symbol}@trade"
        if request.data_type == DataType.QUOTE_TICK:
            return f"{raw_symbol}@bookTicker"
        if request.data_type == DataType.BAR:
            bar_spec = (request.bar_spec or "1-MINUTE").strip().upper()
            try:
                interval = _BAR_SPEC_TO_INTERVAL[bar_spec]
            except KeyError as exc:
                raise ValueError(f"Binance实时Kline不支持周期: {bar_spec}") from exc
            return f"{raw_symbol}@kline_{interval}"
        raise ValueError(f"Binance实时行情不支持数据类型: {request.data_type.name}")

    def _instrument_id(self, symbol: str) -> InstrumentId:
        normalized = symbol.strip().upper()
        if self.config.market_type == "futures":
            normalized = f"{normalized.removesuffix('-PERP')}-PERP"
        return InstrumentId.from_str(f"{normalized}.BINANCE")

    def on_ws_message(self, message: str | dict[str, Any]) -> None:
        if isinstance(message, str):
            data = json.loads(message)
        else:
            data = message
        # 同时接受原始事件和Binance组合流外层结构。
        data = data.get("data", data)
        event_type = data.get("e")

        # 1. 逐笔成交帧 (trade)
        if event_type == "trade":
            symbol = data["s"].upper()
            inst_id = self._instrument_id(symbol)
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
            inst_id = self._instrument_id(symbol)
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

        # 3. 已收盘Kline。进行中的x=false帧会不断覆盖，不能送入策略。
        elif event_type == "kline":
            with self._diagnostic_lock:
                self._kline_messages += 1
            kline = data.get("k") or {}
            if not kline.get("x", False):
                return
            with self._diagnostic_lock:
                self._closed_kline_messages += 1
            symbol = str(kline.get("s") or data.get("s") or "").upper()
            if not symbol:
                raise ValueError("Binance Kline缺少symbol")
            interval = str(kline.get("i") or "")
            try:
                bar_spec = _INTERVAL_TO_BAR_SPEC[interval]
            except KeyError as exc:
                raise ValueError(f"Binance Kline周期不受支持: {interval!r}") from exc
            inst_id = self._instrument_id(symbol)
            meta = self.get_instrument_meta(inst_id)
            if meta is None:
                raise ValueError(f"Binance Kline合约尚未注册: {inst_id}")
            close_time_ms = int(kline["T"])
            event_time_ms = int(data.get("E", close_time_ms))
            bar = make_bar(
                instrument_id=inst_id,
                open=kline["o"],
                high=kline["h"],
                low=kline["l"],
                close=kline["c"],
                volume=kline["v"],
                ts_event=close_time_ms * 1_000_000,
                ts_init=event_time_ms * 1_000_000,
                meta=meta,
                bar_type=bar_spec,
            )
            self.enqueue_event(bar)
