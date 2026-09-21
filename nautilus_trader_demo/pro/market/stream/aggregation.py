"""实时标准行情流转换组件。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from market.basic.base import (
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    TradeTick,
    make_bar,
)


_BAR_SPEC_TO_NS = {
    "1-MINUTE": 60_000_000_000,
}


@dataclass
class _TradeBarState:
    bucket: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class TradeTickBarFeed(MarketDataFeed):
    """把上游TradeTick聚合成已收盘的标准时间Bar。

    当前只开放1分钟周期。只有看到下一分钟的首个成交后，上一分钟Bar才会
    发出，因此不会把尚未结束的实时分钟误当成完整Bar。没有成交的分钟不会
    人工补零；是否补齐交易日时间轴应由更高层的数据质量策略决定。
    """

    def __init__(
        self,
        source_id: str,
        upstream: MarketDataFeed,
        *,
        bar_spec: str = "1-MINUTE",
    ) -> None:
        super().__init__(source_id)
        normalized = bar_spec.strip().upper()
        if normalized not in _BAR_SPEC_TO_NS:
            raise ValueError(f"暂不支持实时聚合周期: {bar_spec}")
        self.upstream = upstream
        self.bar_spec = normalized
        self._interval_ns = _BAR_SPEC_TO_NS[normalized]
        self._states: dict[InstrumentId, _TradeBarState] = {}
        self._handler_attached = False

    @property
    def health_snapshot(self):
        """沿用真实网络上游的健康状态，供Runner行情闸门判断。"""
        return getattr(self.upstream, "health_snapshot", None)

    def register_health_handler(self, handler) -> None:
        register = getattr(self.upstream, "register_health_handler", None)
        if callable(register):
            register(handler)

    def acknowledge_health_degradation(self):
        acknowledge = getattr(self.upstream, "acknowledge_health_degradation", None)
        if not callable(acknowledge):
            raise RuntimeError("上游Feed不支持行情健康恢复确认")
        return acknowledge()

    def register_instrument(self, meta: InstrumentMeta) -> None:
        super().register_instrument(meta)
        if self.upstream.get_instrument_meta(meta.instrument_id) is None:
            self.upstream.register_instrument(meta)

    def subscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType = DataType.BAR,
        bar_spec: str | None = None,
        fields=(),
        **extra,
    ) -> None:
        if data_type is not DataType.BAR:
            raise ValueError("TradeTickBarFeed只提供标准Bar")
        requested_spec = (bar_spec or self.bar_spec).strip().upper()
        if requested_spec != self.bar_spec:
            raise ValueError(
                f"聚合器周期为{self.bar_spec}，不能订阅{requested_spec}",
            )
        super().subscribe(instrument_id, data_type, requested_spec, fields, **extra)

    def connect(self) -> None:
        if self._is_connected:
            return
        if not self._subscriptions:
            raise RuntimeError("连接实时Bar聚合Feed前至少需要一个Bar订阅")
        if not self._handler_attached:
            self.upstream.register_trade_tick_handler(self._on_trade_tick)
            self._handler_attached = True
        for instrument_id in self._subscriptions:
            self.upstream.subscribe(instrument_id, DataType.TRADE_TICK)
        self.upstream.connect()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        try:
            self.upstream.disconnect()
        finally:
            # 最后一根尚未收盘的Bar不能在停机时伪装成完整Bar。
            self._states.clear()
            self._is_connected = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        if self._is_connected:
            self.upstream.subscribe(request.instrument_id, DataType.TRADE_TICK)

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        if self._is_connected and request.instrument_id not in self._subscriptions:
            self.upstream.unsubscribe(request.instrument_id, DataType.TRADE_TICK)
        self._states.pop(request.instrument_id, None)

    def _on_trade_tick(self, tick: TradeTick) -> None:
        instrument_id = tick.instrument_id
        if instrument_id not in self._subscriptions:
            return
        price = tick.price.as_decimal()
        size = tick.size.as_decimal()
        bucket = tick.ts_event // self._interval_ns
        state = self._states.get(instrument_id)
        if state is None:
            self._states[instrument_id] = _TradeBarState(
                bucket=bucket,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
            )
            return
        if bucket < state.bucket:
            # Stream健康监控会报告时间戳回退；聚合器本身不能污染已开始的Bar。
            return
        if bucket == state.bucket:
            state.high = max(state.high, price)
            state.low = min(state.low, price)
            state.close = price
            state.volume += size
            return

        meta = self.get_instrument_meta(instrument_id)
        if meta is None:
            raise RuntimeError(f"实时聚合缺少合约元数据: {instrument_id}")
        closed = make_bar(
            instrument_id=instrument_id,
            open=state.open,
            high=state.high,
            low=state.low,
            close=state.close,
            volume=state.volume,
            ts_event=(state.bucket + 1) * self._interval_ns - 1,
            ts_init=tick.ts_init,
            meta=meta,
            bar_type=self.bar_spec,
        )
        self._states[instrument_id] = _TradeBarState(
            bucket=bucket,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
        )
        self._emit_bar(closed)
