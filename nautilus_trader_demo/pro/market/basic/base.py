import abc
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Mapping, Sequence
import numpy as np 

from fixes import Bar, BarSpecification, BarType, QuoteTick, TradeTick
from fixes import AggressorSide, BarAggregation, PriceType
from fixes import InstrumentId, Symbol, TradeId, Venue
from fixes import Price, Quantity

from market.basic.fast_factory import (  # type: ignore
        fast_make_trade_tick,
        fast_make_quote_tick,
        fast_make_trade_ticks_from_arrays,
    )

from market.basic.custom_bar import CustomBar


class DataType(Enum):
    """订阅的数据流类型枚举。"""
    TRADE_TICK = auto()   # 逐笔成交
    QUOTE_TICK = auto()   # 买卖盘口
    ORDER_BOOK = auto()   # 深度 L2
    BAR = auto()          # 标准 K线聚合 (纯 OHLCV)
    CUSTOM_BAR = auto()   # 复合 K线 (基础 OHLCV + DolphinDB 伴生指标/因子)
    CUSTOM_DATA = auto()  # 纯自定义截面因子/事件流

    
@dataclass(frozen=True)
class SubscriptionRequest:
    """订阅请求结构体。"""
    instrument_id: InstrumentId
    data_type: DataType = DataType.TRADE_TICK
    bar_spec: str | None = None
    fields: tuple[str, ...] = ()  # 显式声明需要提取的自定义指标列（如 vwap, rsi_14）
    extra_params: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class InstrumentMeta:
    """合约主数据与精度配置（指导 Cython 紧凑数值构造）。"""
    instrument_id: InstrumentId
    price_precision: int = 2
    size_precision: int = 0
    price_increment: Decimal = Decimal("0.01")
    multiplier: Decimal = Decimal("1")
    currency: str = "CNY"
    exchange: str = "SHFE"


def make_trade_tick(
    instrument_id: InstrumentId,
    price: Decimal | float | str,
    size: Decimal | float | str,
    trade_id: str,
    ts_event: int,
    ts_init: int | None = None,
    aggressor_side: AggressorSide | str = AggressorSide.NO_AGGRESSOR,
    meta: InstrumentMeta | None = None,
) -> TradeTick:
    ts_init = ts_event if ts_init is None else ts_init
    price_prec = meta.price_precision if meta else 2
    size_prec = meta.size_precision if meta else 0
    side_val = 0
    if aggressor_side == AggressorSide.BUYER or aggressor_side == "BUYER":
        side_val = 1
    elif aggressor_side == AggressorSide.SELLER or aggressor_side == "SELLER":
        side_val = 2

    return fast_make_trade_tick(
            instrument_id,
            float(price),
            float(size),
            price_prec,
            size_prec,
            trade_id,
            ts_event,
            ts_init,
            side_val,
        )


def make_quote_tick(
    instrument_id: InstrumentId,
    bid_price: Decimal | float | str,
    ask_price: Decimal | float | str,
    bid_size: Decimal | float | str,
    ask_size: Decimal | float | str,
    ts_event: int,
    ts_init: int | None = None,
    meta: InstrumentMeta | None = None,
) -> QuoteTick:
    """构造兼容 Cython 原生 C 扩展的 QuoteTick 实例。"""
    ts_init = ts_event if ts_init is None else ts_init
    price_prec = meta.price_precision if meta else 2
    size_prec = meta.size_precision if meta else 0

    return fast_make_quote_tick(
            instrument_id,
            float(bid_price),
            float(ask_price),
            float(bid_size),
            float(ask_size),
            price_prec,
            size_prec,
            ts_event,
            ts_init,
        )


def make_bar(
    instrument_id: InstrumentId,
    open: Decimal | float | str,
    high: Decimal | float | str,
    low: Decimal | float | str,
    close: Decimal | float | str,
    volume: Decimal | float | str,
    ts_event: int,
    ts_init: int | None = None,
    meta: InstrumentMeta | None = None,
    bar_type: BarType | str | None = None,
) -> Bar:
    """构造标准 Bar 实例。"""
    ts_init = ts_event if ts_init is None else ts_init
    price_prec = meta.price_precision if meta else 2
    size_prec = meta.size_precision if meta else 0

    if bar_type is None:
        resolved_bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")
    elif isinstance(bar_type, str):
        if "-" in bar_type and len(bar_type.split("-")) >= 4:
            resolved_bar_type = BarType.from_str(bar_type)
        else:
            resolved_bar_type = BarType.from_str(f"{instrument_id}-{bar_type}-LAST-EXTERNAL")
    else:
        resolved_bar_type = bar_type

    o_str = f"{float(open):.{price_prec}f}"
    h_str = f"{float(high):.{price_prec}f}"
    l_str = f"{float(low):.{price_prec}f}"
    c_str = f"{float(close):.{price_prec}f}"
    v_str = f"{float(volume):.{size_prec}f}"

    return Bar(
        bar_type=resolved_bar_type,
        open=Price.from_str(o_str),
        high=Price.from_str(h_str),
        low=Price.from_str(l_str),
        close=Price.from_str(c_str),
        volume=Quantity.from_str(v_str),
        ts_event=ts_event,
        ts_init=ts_init,
    )


def make_bars_from_arrays(
    bar_type: BarType,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    ts_events: Sequence[int],
    ts_inits: Sequence[int],
    price_prec: int = 2,
    size_prec: int = 0,
) -> list[Bar]:
    """批量从连续数组构造 Bar 列表。"""
    bars = []
    inst_id = bar_type.instrument_id
    for i in range(len(opens)):
        bars.append(
            make_bar(
                instrument_id=inst_id,
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=volumes[i],
                ts_event=ts_events[i],
                ts_init=ts_inits[i],
                bar_type=bar_type,
            )
        )
    return bars


def make_custom_bar(
    bar: Bar,
    factors: Mapping[str, float],
    ts_event: int | None = None,
    ts_init: int | None = None,
) -> CustomBar:
    """构造复合 CustomBar 实体。"""
    ts_ev = bar.ts_event if ts_event is None else ts_event
    ts_in = bar.ts_init if ts_init is None else ts_init
    return CustomBar(
        bar=bar,
        factors=dict(factors),
        ts_event=ts_ev,
        ts_init=ts_in,
    )


def make_custom_bar_all_in_one(
    instrument_id: InstrumentId,
    open: Decimal | float | str,
    high: Decimal | float | str,
    low: Decimal | float | str,
    close: Decimal | float | str,
    volume: Decimal | float | str,
    factors: Mapping[str, float],
    ts_event: int,
    ts_init: int | None = None,
    meta: InstrumentMeta | None = None,
    bar_type: BarType | str | None = None,
) -> CustomBar:
    """一步到位构造复合 CustomBar 实体。"""
    bar = make_bar(
        instrument_id=instrument_id,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        ts_event=ts_event,
        ts_init=ts_init,
        meta=meta,
        bar_type=bar_type,
    )
    return make_custom_bar(bar, factors, ts_event, ts_init)


### 行情订阅抽象类
class MarketDataFeed(abc.ABC):
    """行情数据源抽象基类。"""
    
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._subscriptions: dict[InstrumentId, set[SubscriptionRequest]] = {}
        self._instruments: dict[InstrumentId, InstrumentMeta] = {}
        self._trade_tick_handlers: list[Callable[[TradeTick], None]] = []
        self._quote_tick_handlers: list[Callable[[QuoteTick], None]] = []
        self._bar_handlers: list[Callable[[Bar], None]] = []
        self._custom_bar_handlers: list[Callable[[CustomBar], None]] = []
        self._is_connected: bool = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def register_instrument(self, meta: InstrumentMeta) -> None:
        self._instruments[meta.instrument_id] = meta

    def get_instrument_meta(self, instrument_id: InstrumentId) -> InstrumentMeta | None:
        return self._instruments.get(instrument_id)

    def subscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType = DataType.TRADE_TICK,
        bar_spec: str | None = None,
        fields: Sequence[str] = (),
        **extra: Any,
    ) -> None:
        """声明式订阅指定标的行情。"""
        inst_id = InstrumentId.from_str(instrument_id) if isinstance(instrument_id, str) else instrument_id
        req = SubscriptionRequest(
            instrument_id=inst_id,
            data_type=data_type,
            bar_spec=bar_spec,
            fields=tuple(fields),
            extra_params=extra,
        )
        if inst_id not in self._subscriptions:
            self._subscriptions[inst_id] = set()
        self._subscriptions[inst_id].add(req)

        if self._is_connected:
            self._on_subscription_added(req)

    def unsubscribe(
        self,
        instrument_id: InstrumentId | str,
        data_type: DataType | None = None,
    ) -> None:
        inst_id = InstrumentId.from_str(instrument_id) if isinstance(instrument_id, str) else instrument_id
        if inst_id in self._subscriptions:
            if data_type is None:
                removed = self._subscriptions.pop(inst_id, set())
                for req in removed:
                    if self._is_connected:
                        self._on_subscription_removed(req)
            else:
                to_remove = {r for r in self._subscriptions[inst_id] if r.data_type == data_type}
                self._subscriptions[inst_id] -= to_remove
                for req in to_remove:
                    if self._is_connected:
                        self._on_subscription_removed(req)
                if not self._subscriptions[inst_id]:
                    del self._subscriptions[inst_id]

    def get_subscribed_instruments(self) -> frozenset[InstrumentId]:
        return frozenset(self._subscriptions.keys())

    # --- 回调注册 ---
    # --- 核心生命周期钩子 (由具体实现驱动) ---

    def register_trade_tick_handler(self, handler: Callable[[TradeTick], None]) -> None:
        self._trade_tick_handlers.append(handler)

    def register_quote_tick_handler(self, handler: Callable[[QuoteTick], None]) -> None:
        self._quote_tick_handlers.append(handler)

    def register_bar_handler(self, handler: Callable[[Bar], None]) -> None:
        self._bar_handlers.append(handler)

    def register_custom_bar_handler(self, handler: Callable[[CustomBar], None]) -> None:
        self._custom_bar_handlers.append(handler)


    # --- 内部事件分发 ---

    def _emit_trade_tick(self, tick: TradeTick) -> None:
        for handler in self._trade_tick_handlers:
            handler(tick)

    def _emit_quote_tick(self, tick: QuoteTick) -> None:
        for handler in self._quote_tick_handlers:
            handler(tick)

    def _emit_bar(self, bar: Bar) -> None:
        for handler in self._bar_handlers:
            handler(bar)

    def _emit_custom_bar(self, custom_bar: CustomBar) -> None:
        for handler in self._custom_bar_handlers:
            handler(custom_bar)

    # --- 抽象生命周期 ---

    @abc.abstractmethod
    def connect(self) -> None:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        pass

    @abc.abstractmethod
    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        pass