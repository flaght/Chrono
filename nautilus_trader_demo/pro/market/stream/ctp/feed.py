"""CTP adapter implementing the same ``market.basic`` contract as Binance."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from market.basic.base import (
    DataType,
    InstrumentId,
    InstrumentMeta,
    QuoteTick,
    SubscriptionRequest,
)
from market.stream.base import StreamDataFeed

from market.native.ctp.driver import CtpMdDriver, create_native_driver
from market.stream.ctp.converter import CtpTickConverter


logger = logging.getLogger("CtpLiveDataFeed")
DriverFactory = Callable[["CtpLiveDataFeed"], CtpMdDriver]
_TICK_TYPES = frozenset((DataType.QUOTE_TICK, DataType.TRADE_TICK))


@dataclass(frozen=True)
class CtpMdConfig:
    front: str
    broker_id: str
    user_id: str
    password: str = field(repr=False)
    flow_path: str = "/tmp/bomber-ctp-md"
    production_mode: bool = False
    timezone: str = "Asia/Shanghai"


class CtpLiveDataFeed(StreamDataFeed):
    """Map CTP depth snapshots to standard QuoteTick/TradeTick events."""

    def __init__(
        self,
        config: CtpMdConfig,
        source_id: str = "CTP_LIVE_SOURCE",
        queue_size: int = 100_000,
        driver_factory: DriverFactory = create_native_driver,
    ) -> None:
        super().__init__(source_id=source_id, queue_size=queue_size)
        self.config = config
        self._driver_factory = driver_factory
        self._driver: CtpMdDriver | None = None
        self._converter = CtpTickConverter(config.timezone)
        self._symbol_to_instrument: dict[str, InstrumentId] = {}
        self._subscribed_symbols: set[str] = set()
        self._ready = threading.Event()
        self._lock = threading.RLock()

    def register_instrument(self, meta: InstrumentMeta) -> None:
        super().register_instrument(meta)
        self._symbol_to_instrument[_symbol(meta.instrument_id)] = meta.instrument_id

    def subscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType = DataType.TRADE_TICK,
        bar_spec: str | None = None,
        fields: Sequence[str] = (),
        **extra: Any,
    ) -> None:
        resolved = InstrumentId.from_str(instrument_id) if isinstance(instrument_id, str) else instrument_id
        if data_type not in _TICK_TYPES:
            raise ValueError("CTP 实时行情仅支持 QuoteTick 和 TradeTick")
        if self.get_instrument_meta(resolved) is None:
            raise ValueError(f"合约尚未注册: {resolved}")
        super().subscribe(resolved, data_type, bar_spec, fields, **extra)

    def connect(self) -> None:
        if not self._desired_symbols():
            raise RuntimeError("连接前至少订阅一个 CTP QuoteTick 或 TradeTick")
        super().connect()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait until login succeeds and all desired symbols are submitted."""
        return self._ready.wait(timeout)

    def _start_network_client(self) -> None:
        driver = self._driver_factory(self)
        self._driver = driver
        try:
            driver.start(
                front=self.config.front,
                flow_path=Path(self.config.flow_path).expanduser().resolve(),
                production_mode=self.config.production_mode,
            )
        except Exception:
            try:
                driver.stop()
            except Exception:
                logger.exception("CTP 启动失败后的资源清理也发生异常")
            self._driver = None
            raise

    def _stop_network_client(self) -> None:
        self._ready.clear()
        driver = self._driver
        if driver is not None:
            driver.stop()
        self._driver = None
        self._converter.reset()
        with self._lock:
            self._subscribed_symbols.clear()

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self._validate_request(request)
        if self._ready.is_set():
            self._subscribe_symbol(_symbol(request.instrument_id))

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        symbol = _symbol(request.instrument_id)
        if self._ready.is_set() and symbol not in self._desired_symbols():
            self._unsubscribe_symbol(symbol)

    def _validate_request(self, request: SubscriptionRequest) -> None:
        if request.data_type not in _TICK_TYPES:
            raise ValueError("CTP 实时行情仅支持 QuoteTick 和 TradeTick")
        if self.get_instrument_meta(request.instrument_id) is None:
            raise ValueError(f"合约尚未注册: {request.instrument_id}")

    # These callbacks execute on the native CTP callback worker.
    def on_front_connected(self) -> None:
        logger.info("CTP 行情前置已连接，正在登录")
        self._require_driver().login(
            broker_id=self.config.broker_id,
            user_id=self.config.user_id,
            password=self.config.password,
        )

    def on_front_disconnected(self, reason: int) -> None:
        self._ready.clear()
        with self._lock:
            self._subscribed_symbols.clear()
        logger.warning("CTP 行情前置断开: reason=%s；等待底层 API 自动重连", reason)

    def on_login_response(self, error: Mapping[str, Any] | None) -> None:
        if _error_id(error):
            self._ready.clear()
            logger.error("CTP 行情登录失败: %s", _error_text(error))
            return
        logger.info("CTP 行情登录成功，恢复订阅")
        try:
            for symbol in sorted(self._desired_symbols()):
                self._subscribe_symbol(symbol)
        except Exception:
            self._ready.clear()
            logger.exception("CTP 恢复订阅失败")
            return
        self._ready.set()

    def on_subscribe_response(
        self,
        data: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        symbol = str((data or {}).get("InstrumentID") or "")
        if _error_id(error):
            self._ready.clear()
            with self._lock:
                self._subscribed_symbols.discard(symbol)
            logger.error("CTP 行情订阅失败: symbol=%s %s", symbol, _error_text(error))
        else:
            logger.info("CTP 行情订阅成功: %s", symbol)

    def on_unsubscribe_response(
        self,
        data: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        symbol = str((data or {}).get("InstrumentID") or "")
        if _error_id(error):
            logger.error("CTP 行情退订失败: symbol=%s %s", symbol, _error_text(error))
        else:
            logger.info("CTP 行情退订成功: %s", symbol)

    def on_api_error(self, error: Mapping[str, Any] | None) -> None:
        logger.error("CTP 行情接口错误: %s", _error_text(error))

    def on_depth_market_data(self, data: Mapping[str, Any]) -> None:
        symbol = str(data.get("InstrumentID") or "").strip()
        instrument_id = self._symbol_to_instrument.get(symbol)
        if instrument_id is None:
            logger.warning("忽略未注册合约的 CTP 行情: %s", symbol)
            return
        meta = self.get_instrument_meta(instrument_id)
        if meta is None:
            logger.warning("忽略缺少 InstrumentMeta 的 CTP 行情: %s", instrument_id)
            return
        try:
            for event in self._converter.convert(data, instrument_id, meta):
                if self._event_is_subscribed(instrument_id, event):
                    self.enqueue_event(event)
        except Exception:
            logger.exception("CTP 行情转换失败: symbol=%s", symbol)

    def _desired_symbols(self) -> set[str]:
        return {
            _symbol(instrument_id)
            for instrument_id, requests in self._subscriptions.items()
            if any(request.data_type in _TICK_TYPES for request in requests)
        }

    def _subscribe_symbol(self, symbol: str) -> None:
        with self._lock:
            if symbol in self._subscribed_symbols:
                return
            self._require_driver().subscribe(symbol)
            self._subscribed_symbols.add(symbol)

    def _unsubscribe_symbol(self, symbol: str) -> None:
        with self._lock:
            if symbol not in self._subscribed_symbols:
                return
            self._require_driver().unsubscribe(symbol)
            self._subscribed_symbols.discard(symbol)

    def _require_driver(self) -> CtpMdDriver:
        if self._driver is None:
            raise RuntimeError("CTP 行情驱动尚未启动")
        return self._driver

    def _event_is_subscribed(self, instrument_id: InstrumentId, event: Any) -> bool:
        data_type = DataType.QUOTE_TICK if isinstance(event, QuoteTick) else DataType.TRADE_TICK
        return any(
            request.data_type is data_type
            for request in self._subscriptions.get(instrument_id, set())
        )


def _symbol(instrument_id: InstrumentId) -> str:
    return str(instrument_id).rsplit(".", 1)[0]


def _error_id(error: Mapping[str, Any] | None) -> int:
    return int((error or {}).get("ErrorID", 0) or 0)


def _error_text(error: Mapping[str, Any] | None) -> str:
    error = error or {}
    return f"{_error_id(error)} {error.get('ErrorMsg', '')}".strip()
