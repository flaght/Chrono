"""实时标准行情流转换组件。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import time

from bomber.framework.market.basic.base import (
    BarType,
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    QuoteTick,
    TradeTick,
    make_bar,
    make_trade_tick,
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


class ReceiveTimeTradeTickBarFeed(TradeTickBarFeed):
    """显式工程联调：用Tick接收时间聚合，不修改上游原始Tick。

    适用于持续推送旧日期行情的模拟柜台。Bar代表本次接收期间的价格流，
    不代表原交易日分钟，不可用于历史回测或解释历史收益。默认聚合组件仍
    使用事件时间；本类只在调用方显式选择时启用。健康门控仍来自原上游。
    """

    def __init__(self, *args, clock_ns=None, max_receive_age_ns=10_000_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        if max_receive_age_ns <= 0:
            raise ValueError("接收时间时效须大于零")
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        self._max_receive_age_ns = max_receive_age_ns
        self._last_receive_ns = {}
        self._timing = {"bar_time_basis": "receive", "ticks_used": 0,
                        "first_source_event_ns": None, "last_source_event_ns": None,
                        "first_receive_ns": None, "last_receive_ns": None}

    @property
    def timing_snapshot(self):
        return dict(self._timing)

    def disconnect(self) -> None:
        try:
            super().disconnect()
        finally:
            self._last_receive_ns.clear()

    def _on_trade_tick(self, tick: TradeTick) -> None:
        if tick.instrument_id not in self._subscriptions:
            return
        received = tick.ts_init
        age = self._clock_ns() - received
        if received <= 0 or not 0 <= age <= self._max_receive_age_ns:
            raise RuntimeError("联调Tick接收时间缺失、来自未来或已过期")
        previous = self._last_receive_ns.get(tick.instrument_id)
        if previous is not None and received < previous:
            raise RuntimeError("联调Tick接收时间倒退")
        meta = self.get_instrument_meta(tick.instrument_id)
        if meta is None:
            raise RuntimeError(f"接收时间聚合缺少合约元数据: {tick.instrument_id}")
        # 只为聚合器构造副本；上游标准Tick和其他订阅者保留原事件时间。
        current = make_trade_tick(
            instrument_id=tick.instrument_id, price=tick.price.as_decimal(),
            size=tick.size.as_decimal(), trade_id=str(tick.trade_id),
            ts_event=received, ts_init=received, meta=meta)
        self._last_receive_ns[tick.instrument_id] = received
        if not self._timing["ticks_used"]:
            self._timing.update(first_source_event_ns=tick.ts_event, first_receive_ns=received)
        self._timing.update(ticks_used=self._timing["ticks_used"] + 1,
                            last_source_event_ns=tick.ts_event, last_receive_ns=received)
        super()._on_trade_tick(current)


class QuoteMidBarFeed(TradeTickBarFeed):
    """从一档QuoteTick的(bid+ask)/2生成已收盘分钟MID Bar。

    每路单独聚合；无Quote分钟不补Bar。首个下一分钟Quote到来时才发出
    上一分钟Bar，停机时不发尚未确认收盘的Bar。Quote不包含成交量，Bar
    volume明确置0，不能伪装成成交量。原始Quote也原样转发给模拟Backend
    形成可撮合盘口，但截面策略只订阅已收盘的MID Bar。上游若支持replay，
    本Feed也可回放。
    """

    def subscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType = DataType.BAR,
        bar_spec: str | None = None,
        fields=(),
        **extra,
    ) -> None:
        if data_type is DataType.QUOTE_TICK:
            if bar_spec is not None:
                raise ValueError("QuoteTick订阅不能携带bar_spec")
            MarketDataFeed.subscribe(self, instrument_id, data_type, None, fields, **extra)
            return
        super().subscribe(instrument_id, data_type, bar_spec, fields, **extra)

    def connect(self) -> None:
        if self._is_connected:
            return
        if not self._subscriptions:
            raise RuntimeError("连接MID Bar聚合Feed前至少需要一个Bar订阅")
        if not self._handler_attached:
            self.upstream.register_quote_tick_handler(self._on_quote_tick)
            self._handler_attached = True
        for instrument_id in self._subscriptions:
            self.upstream.subscribe(instrument_id, DataType.QUOTE_TICK)
        self.upstream.connect()
        self._is_connected = True

    def replay(self):
        if not self._is_connected:
            raise RuntimeError("MID Bar聚合Feed尚未连接")
        replay = getattr(self.upstream, "replay", None)
        if not callable(replay):
            raise RuntimeError("MID Bar上游不支持离线回放")
        return replay()

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        if self._is_connected:
            self.upstream.subscribe(request.instrument_id, DataType.QUOTE_TICK)

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        if self._is_connected and request.instrument_id not in self._subscriptions:
            self.upstream.unsubscribe(request.instrument_id, DataType.QUOTE_TICK)
        self._states.pop(request.instrument_id, None)

    def _on_quote_tick(self, tick: QuoteTick) -> None:
        instrument_id = tick.instrument_id
        if instrument_id not in self._subscriptions:
            return
        bid = tick.bid_price.as_decimal()
        ask = tick.ask_price.as_decimal()
        if bid <= 0 or ask <= 0 or bid > ask:
            raise ValueError(f"无效一档报价: {instrument_id}")
        # 当前Quote先推进模拟盘口/旧订单，再用它确认上一分钟的MID Bar收盘。
        self._emit_quote_tick(tick)
        midpoint = (bid + ask) / 2
        bucket = tick.ts_event // self._interval_ns
        state = self._states.get(instrument_id)
        if state is None:
            self._states[instrument_id] = _TradeBarState(
                bucket, midpoint, midpoint, midpoint, midpoint, Decimal(0),
            )
            return
        if bucket < state.bucket:
            raise ValueError(f"Quote时间回退: {instrument_id}")
        if bucket == state.bucket:
            state.high = max(state.high, midpoint)
            state.low = min(state.low, midpoint)
            state.close = midpoint
            return
        meta = self.get_instrument_meta(instrument_id)
        if meta is None:
            raise RuntimeError(f"MID聚合缺少合约元数据: {instrument_id}")
        # 合法盘口中点可能落在半个最小报价单位上（例如100和101的中点
        # 100.5）。不能沿用交易合约的整数价格精度把MID悄悄取整。
        mid_meta = InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=meta.price_precision + 1,
            size_precision=meta.size_precision,
            price_increment=meta.price_increment / 2,
            multiplier=meta.multiplier,
            currency=meta.currency,
            exchange=meta.exchange,
        )
        closed = make_bar(
            instrument_id=instrument_id,
            open=state.open,
            high=state.high,
            low=state.low,
            close=state.close,
            volume=0,
            ts_event=(state.bucket + 1) * self._interval_ns - 1,
            ts_init=tick.ts_init,
            meta=mid_meta,
            bar_type=BarType.from_str(f"{instrument_id}-{self.bar_spec}-MID-EXTERNAL"),
        )
        self._states[instrument_id] = _TradeBarState(
            bucket, midpoint, midpoint, midpoint, midpoint, Decimal(0),
        )
        self._emit_bar(closed)
