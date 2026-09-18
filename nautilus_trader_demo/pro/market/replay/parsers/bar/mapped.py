"""Explicitly mapped vendor Bar parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from market.basic.base import CustomBar, DataType, InstrumentId, make_bar, make_custom_bar
from market.replay.parsers.base import (
    ParsedEvent,
    ParserContext,
    nonnegative,
    parse_iso_timestamp,
    positive,
    required,
)


@dataclass(frozen=True)
class BarColumns:
    symbol: str
    exchange: str
    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    value: str | None = None
    open_interest: str | None = None
    vwap: str | None = None
    bar_spec: str | None = None


class MappedBarParser:
    """Map explicitly configured vendor columns into standard Bars."""

    def __init__(
        self,
        columns: BarColumns,
        bar_spec: str = "1-MINUTE",
        timezone: str = "Asia/Shanghai",
    ) -> None:
        self.columns = columns
        self.bar_spec = bar_spec.upper()
        self.timezone = ZoneInfo(timezone)

    def reset(self) -> None:
        pass

    def parse(
        self,
        row: Mapping[str, Any],
        context: ParserContext,
    ) -> Iterable[ParsedEvent]:
        c = self.columns
        instrument_id = InstrumentId.from_str(
            f"{required(row, c.symbol, context)}.{required(row, c.exchange, context).upper()}",
        )
        meta = context.require_meta(instrument_id)
        bar_spec = (
            required(row, c.bar_spec, context).upper()
            if c.bar_spec is not None
            else self.bar_spec
        )
        ts_event = parse_iso_timestamp(
            required(row, c.timestamp, context),
            context,
            self.timezone,
        )
        open_price = positive(row, c.open, context)
        high_price = positive(row, c.high, context)
        low_price = positive(row, c.low, context)
        close_price = positive(row, c.close, context)
        volume = nonnegative(row, c.volume, context)
        if high_price < max(open_price, low_price, close_price):
            raise context.error("high is below an OHLC component")
        if low_price > min(open_price, high_price, close_price):
            raise context.error("low is above an OHLC component")

        bar = make_bar(
            instrument_id=instrument_id,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            ts_event=ts_event,
            ts_init=ts_event,
            meta=meta,
            bar_type=bar_spec,
        )
        events = [ParsedEvent(DataType.BAR, instrument_id, bar, bar_spec)]
        factors: dict[str, float] = {}
        if c.value is not None:
            factors["value"] = nonnegative(row, c.value, context)
        if c.open_interest is not None:
            factors["open_interest"] = nonnegative(row, c.open_interest, context)
        if c.vwap is not None:
            factors["vwap"] = positive(row, c.vwap, context)
        if factors:
            custom_bar: CustomBar = make_custom_bar(bar, factors)
            events.append(
                ParsedEvent(DataType.CUSTOM_BAR, instrument_id, custom_bar, bar_spec),
            )
        return events
