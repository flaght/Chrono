"""DolphinDB stream-table adapter using the common StreamDataFeed contract."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from market.basic.base import DataType, InstrumentId, InstrumentMeta, SubscriptionRequest
from market.stream.base import StreamDataFeed

from market.native.dolphin import (
    DolphinDbDriver,
    NativeDolphinSubscription,
    create_native_driver,
)
from market.stream.dolphin.config import DolphinDbConfig, DolphinDbStreamSpec
from market.stream.dolphin.converter import ConvertedEvent, DolphinDbMarketConverter


logger = logging.getLogger("DolphinDbLiveDataFeed")
DriverFactory = Callable[[], DolphinDbDriver]
_SUPPORTED_TYPES = frozenset(
    (DataType.QUOTE_TICK, DataType.TRADE_TICK, DataType.BAR, DataType.CUSTOM_BAR),
)


class DolphinDbLiveDataFeed(StreamDataFeed):
    def __init__(
        self,
        config: DolphinDbConfig,
        source_id: str = "DOLPHINDB_LIVE_SOURCE",
        queue_size: int = 100_000,
        driver_factory: DriverFactory = create_native_driver,
    ) -> None:
        super().__init__(source_id=source_id, queue_size=queue_size)
        self.config = config
        self._driver_factory = driver_factory
        self._driver: DolphinDbDriver | None = None
        self._converter = DolphinDbMarketConverter()
        self._symbol_to_instrument: dict[str, InstrumentId] = {}
        self._ready = threading.Event()

    def register_instrument(self, meta: InstrumentMeta) -> None:
        super().register_instrument(meta)
        symbol = str(meta.instrument_id).rsplit(".", 1)[0]
        self._symbol_to_instrument[symbol.casefold()] = meta.instrument_id

    def subscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType = DataType.TRADE_TICK,
        bar_spec: str | None = None,
        fields: Sequence[str] = (),
        **extra: Any,
    ) -> None:
        resolved = InstrumentId.from_str(instrument_id) if isinstance(instrument_id, str) else instrument_id
        if data_type not in _SUPPORTED_TYPES:
            raise ValueError(f"DolphinDB market stream does not support {data_type}")
        if self.get_instrument_meta(resolved) is None:
            raise ValueError(f"instrument is not registered: {resolved}")
        if data_type in {DataType.BAR, DataType.CUSTOM_BAR} and not bar_spec:
            raise ValueError("bar_spec is required for DolphinDB Bar subscriptions")
        super().subscribe(resolved, data_type, bar_spec, fields, **extra)

    def connect(self) -> None:
        if not self._subscriptions:
            raise RuntimeError("register at least one DolphinDB market subscription before connect")
        super().connect()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    def _start_network_client(self) -> None:
        driver = self._driver_factory()
        self._driver = driver
        try:
            driver.start(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                streaming_port=self.config.streaming_port,
                keep_alive_seconds=self.config.keep_alive_seconds,
            )
            for spec in self.config.streams:
                native = NativeDolphinSubscription(
                    table_name=spec.table_name,
                    action_name=spec.action_name,
                    columns=spec.columns,
                    offset=spec.offset,
                    resub=spec.resub,
                    batch_size=spec.batch_size,
                    throttle=spec.throttle,
                    filter=spec.filter,
                )
                driver.subscribe(
                    native,
                    lambda row, stream=spec: self.on_stream_row(stream, row),
                )
        except Exception:
            try:
                driver.stop()
            except Exception:
                logger.exception("DolphinDB start cleanup failed")
            self._driver = None
            raise
        self._ready.set()

    def _stop_network_client(self) -> None:
        self._ready.clear()
        driver = self._driver
        if driver is not None:
            driver.stop()
        self._driver = None
        self._converter.reset()

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self._validate_request(request)

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        # DolphinDB topics are table-level subscriptions. Instrument filtering is
        # local, so changing one logical subscription does not cancel the topic.
        return None

    def _validate_request(self, request: SubscriptionRequest) -> None:
        if request.data_type not in _SUPPORTED_TYPES:
            raise ValueError(f"unsupported DolphinDB data type: {request.data_type}")

    def on_stream_row(
        self,
        spec: DolphinDbStreamSpec,
        row: Mapping[str, Any],
    ) -> None:
        try:
            symbol = str(row.get("Code") or "").strip()
            if not symbol:
                raise ValueError("DolphinDB row has no Code")
            instrument_id = self._symbol_to_instrument.get(symbol.casefold())
            if instrument_id is None:
                logger.debug("ignore unregistered DolphinDB instrument: %s", symbol)
                return
            meta = self.get_instrument_meta(instrument_id)
            if meta is None:
                raise ValueError(f"missing InstrumentMeta: {instrument_id}")

            if spec.kind == "tick":
                converted = self._converter.convert_tick(row, instrument_id, meta)
            else:
                converted = self._converter.convert_bar(row, instrument_id, meta, spec.bar_spec)
            for item in converted:
                if self._is_subscribed(instrument_id, item):
                    self.enqueue_event(item.event)
        except Exception:
            logger.exception(
                "DolphinDB market row conversion failed: table=%s kind=%s",
                spec.table_name,
                spec.kind,
            )

    def _is_subscribed(
        self,
        instrument_id: InstrumentId,
        item: ConvertedEvent,
    ) -> bool:
        for request in self._subscriptions.get(instrument_id, set()):
            if request.data_type is not item.data_type:
                continue
            if item.data_type in {DataType.BAR, DataType.CUSTOM_BAR}:
                return (request.bar_spec or "").upper() == (item.bar_spec or "").upper()
            return True
        return False
