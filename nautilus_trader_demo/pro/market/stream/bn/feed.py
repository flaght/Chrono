import json, pdb
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Protocol

from market.basic.base import (
    Bar,
    CustomBar,
    DataType,
    InstrumentId,
    QuoteTick,
    SubscriptionRequest,
    TradeTick,
    make_bar,
    make_custom_bar,
    make_custom_bar_all_in_one,
    make_quote_tick,
    make_trade_tick,
)
from market.stream.base import StreamDataFeed

logger = logging.getLogger("BNWSStreamDataFeed")


@dataclass
class BNWSConfig:
    ws_base_url: str = "wss://stream.binance.com:9443"
    api_key: str | None = None
    secret_key: str | None = None


class BNWSStreamDataFeed(StreamDataFeed):
    """web socket 实时行情数据源。"""

    def __init__(self, config: BNWSConfig | None = None, source_id: str = "BINANCE_LIVE_SOURCE") -> None:
        super().__init__(source_id=source_id)
        self.config = config or BNWSConfig()
        self._active_streams: set[str] = set()

    def _start_network_client(self) -> None:
        logger.info(f"建立 WebSocket 长连接: {self.config.ws_base_url}")

    def _stop_network_client(self) -> None:
        logger.info("关闭 WebSocket 连接。")

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        raw_symbol = str(request.instrument_id).split(".")[0].lower()
        if request.data_type == DataType.TRADE_TICK:
            stream_name = f"{raw_symbol}@trade"
        elif request.data_type == DataType.QUOTE_TICK:
            stream_name = f"{raw_symbol}@bookTicker"
        else:
            stream_name = f"{raw_symbol}@kline_1m"

        self._active_streams.add(stream_name)

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        raw_symbol = str(request.instrument_id).split(".")[0].lower()
        stream_name = f"{raw_symbol}@trade"
        self._active_streams.discard(stream_name)

    def on_ws_message(self, message: str | dict[str, Any]) -> None:
        if isinstance(message, str):
            data = json.loads(message)
        else:
            data = message
        print(data)
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

        # # 2. 最优挂单帧 (bookTicker)
        elif "b" in data and "a" in data and "s" in data:
            symbol = data["s"].upper()
            inst_id = InstrumentId.from_str(f"{symbol}.BINANCE")
            ts_ns = int(time.time() * 1_000_000_000)
            meta = self.get_instrument_meta(inst_id)
            pdb.set_trace()
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
