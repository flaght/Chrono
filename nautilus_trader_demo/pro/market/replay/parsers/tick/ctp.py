"""CTP depth-market CSV parser."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from market.basic.base import (
    AggressorSide,
    DataType,
    InstrumentId,
    Price,
    Quantity,
    QuoteTick,
    TradeId,
    TradeTick,
)
from market.replay.parsers.base import (
    ParsedEvent,
    ParserContext,
    datetime_to_ns,
    integer_nonnegative,
    nonnegative,
    positive,
    required,
)


class CtpTickParser:
    """Convert CTP depth snapshots into quotes and inferred trades."""

    def __init__(
        self,
        exchange: str,
        timezone: str = "Asia/Shanghai",
        night_session_action_day: str | None = None,
    ) -> None:
        self.exchange = exchange.upper()
        self.timezone = ZoneInfo(timezone)
        self.night_session_action_day = night_session_action_day
        self._last_volume: dict[InstrumentId, int] = {}

    def reset(self) -> None:
        self._last_volume.clear()

    def parse(
        self,
        row: Mapping[str, Any],
        context: ParserContext,
    ) -> Iterable[ParsedEvent]:
        symbol = required(row, "InstrumentID", context)
        instrument_id = InstrumentId.from_str(f"{symbol}.{self.exchange}")
        meta = context.require_meta(instrument_id)
        ts_event = self._timestamp(row, context)

        bid_price = positive(row, "BidPrice1", context)
        ask_price = positive(row, "AskPrice1", context)
        if bid_price > ask_price:
            raise context.error("BidPrice1 exceeds AskPrice1")

        quote = QuoteTick(
            instrument_id,
            Price(bid_price, meta.price_precision),
            Price(ask_price, meta.price_precision),
            Quantity(nonnegative(row, "BidVolume1", context), meta.size_precision),
            Quantity(nonnegative(row, "AskVolume1", context), meta.size_precision),
            ts_event,
            ts_event,
        )
        events: list[ParsedEvent] = [
            ParsedEvent(DataType.QUOTE_TICK, instrument_id, quote),
        ]

        cumulative = integer_nonnegative(row, "Volume", context)
        previous = self._last_volume.get(instrument_id)
        self._last_volume[instrument_id] = cumulative
        if previous is not None and cumulative > previous:
            trade = TradeTick(
                instrument_id,
                Price(positive(row, "LastPrice", context), meta.price_precision),
                Quantity(cumulative - previous, meta.size_precision),
                AggressorSide.NO_AGGRESSOR,
                TradeId(_synthetic_trade_id(symbol, ts_event, cumulative)),
                ts_event,
                ts_event,
            )
            events.append(ParsedEvent(DataType.TRADE_TICK, instrument_id, trade))
        return events

    def _timestamp(self, row: Mapping[str, Any], context: ParserContext) -> int:
        day = required(row, "TradingDay", context)
        clock = required(row, "UpdateTime", context)
        # CTP的TradingDay是交易日，不一定等于夜盘所在自然日。源文件没有
        # ActionDay时不能可靠推导节假日，因此由调用方显式传入夜盘自然日。
        if self.night_session_action_day is not None and clock >= "18:00:00":
            day = self.night_session_action_day
        millisec = integer_nonnegative(row, "UpdateMillisec", context)
        if millisec > 999:
            raise context.error("UpdateMillisec must be between 0 and 999")
        try:
            timestamp = datetime.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S")
        except ValueError as exc:
            raise context.error("invalid TradingDay or UpdateTime") from exc
        timestamp = timestamp.replace(tzinfo=self.timezone, microsecond=millisec * 1_000)
        return datetime_to_ns(timestamp)


def _synthetic_trade_id(symbol: str, ts_event: int, cumulative: int) -> str:
    """Return a stable 36-character ID for a trade inferred from a snapshot."""
    source = f"{symbol}|{ts_event}|{cumulative}".encode()
    return "ctp-" + hashlib.blake2b(source, digest_size=16).hexdigest()
