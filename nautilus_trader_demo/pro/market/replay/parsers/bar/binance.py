"""Binance 现货与期货 K 线 CSV 解析器。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping

from market.basic.base import CustomBar, DataType, InstrumentId, make_bar, make_custom_bar
from market.replay.parsers.base import (
    ParsedEvent,
    ParserContext,
    integer_nonnegative,
    nonnegative,
    positive,
    required,
)


class BinanceMarketType(str, Enum):
    """支持的 Binance 离线市场类型。"""

    SPOT = "spot"
    FUTURES = "futures"


_INTERVAL_TO_BAR_SPEC = {
    "1m": "1-MINUTE",
    "3m": "3-MINUTE",
    "5m": "5-MINUTE",
    "15m": "15-MINUTE",
    "30m": "30-MINUTE",
    "1h": "1-HOUR",
    "2h": "2-HOUR",
    "4h": "4-HOUR",
    "6h": "6-HOUR",
    "8h": "8-HOUR",
    "12h": "12-HOUR",
    "1d": "1-DAY",
    "3d": "3-DAY",
    "1w": "1-WEEK",
}


def binance_instrument_id(
    symbol: str,
    market_type: BinanceMarketType | str,
) -> InstrumentId:
    """为 Binance 现货或 USDT 永续数据返回无冲突标识。"""

    market = BinanceMarketType(market_type)
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Binance symbol cannot be empty")
    if market is BinanceMarketType.FUTURES and not normalized.endswith("-PERP"):
        normalized = f"{normalized}-PERP"
    return InstrumentId.from_str(f"{normalized}.BINANCE")


class BinanceKlineParser:
    """将 Binance K 线 CSV 数据源转换为标准 Bar 与自定义 Bar。

    现货和期货文件使用相同列；market_type 与
    interval 必须由数据源元信息明确提供，不能根据行内数据
    猜测。标准 Bar 包含 OHLCV；可选的 CustomBar 包含
    Binance 特有的成交量与成交笔数字段。
    """

    def __init__(
        self,
        symbol: str,
        market_type: BinanceMarketType | str,
        interval: str = "1m",
        include_factors: bool = True,
    ) -> None:
        try:
            self.market_type = BinanceMarketType(market_type)
        except ValueError as exc:
            choices = ", ".join(item.value for item in BinanceMarketType)
            raise ValueError(f"market_type must be one of: {choices}") from exc
        try:
            self.bar_spec = _INTERVAL_TO_BAR_SPEC[interval]
        except KeyError as exc:
            raise ValueError(f"unsupported Binance kline interval: {interval!r}") from exc
        self.interval = interval
        self.instrument_id = binance_instrument_id(symbol, self.market_type)
        self.include_factors = include_factors

    def reset(self) -> None:
        pass

    def parse(
        self,
        row: Mapping[str, Any],
        context: ParserContext,
    ) -> Iterable[ParsedEvent]:
        meta = context.require_meta(self.instrument_id)
        open_price = positive(row, "open", context)
        high_price = positive(row, "high", context)
        low_price = positive(row, "low", context)
        close_price = positive(row, "close", context)
        volume = nonnegative(row, "volume", context)
        if high_price < max(open_price, low_price, close_price):
            raise context.error("high is below an OHLC component")
        if low_price > min(open_price, high_price, close_price):
            raise context.error("low is above an OHLC component")

        open_ns = _unix_timestamp_to_ns(required(row, "open_time", context), context)
        close_ns = _unix_timestamp_to_ns(required(row, "close_time", context), context)
        if close_ns < open_ns:
            raise context.error("close_time is before open_time")

        # 外部聚合 Bar 在对应周期结束后才可见。
        bar = make_bar(
            instrument_id=self.instrument_id,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            ts_event=close_ns,
            ts_init=close_ns,
            meta=meta,
            bar_type=self.bar_spec,
        )
        events = [ParsedEvent(DataType.BAR, self.instrument_id, bar, self.bar_spec)]
        if not self.include_factors:
            return events

        factors = {
            "quote_volume": nonnegative(row, "quote_volume", context),
            "trade_count": float(integer_nonnegative(row, "count", context)),
            "taker_buy_volume": nonnegative(row, "taker_buy_volume", context),
            "taker_buy_quote_volume": nonnegative(
                row,
                "taker_buy_quote_volume",
                context,
            ),
        }
        custom_bar: CustomBar = make_custom_bar(bar, factors)
        events.append(
            ParsedEvent(
                DataType.CUSTOM_BAR,
                self.instrument_id,
                custom_bar,
                self.bar_spec,
            ),
        )
        return events


def _unix_timestamp_to_ns(value: str, context: ParserContext) -> int:
    """将秒、毫秒、微秒或纳秒时间统一转换为纳秒。"""

    try:
        timestamp = int(value)
    except (TypeError, ValueError) as exc:
        raise context.error(f"invalid Unix timestamp: {value!r}") from exc
    if timestamp < 0:
        raise context.error(f"Unix timestamp must be nonnegative: {timestamp}")

    digits = len(str(timestamp))
    if digits <= 10:
        return timestamp * 1_000_000_000
    if digits <= 13:
        return timestamp * 1_000_000
    if digits <= 16:
        return timestamp * 1_000
    if digits <= 19:
        return timestamp
    raise context.error(f"unsupported Unix timestamp precision: {value!r}")
