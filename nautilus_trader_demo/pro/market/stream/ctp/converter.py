"""Convert CTP depth snapshots into standard QuoteTick and TradeTick events."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from market.basic.base import (
    InstrumentId,
    InstrumentMeta,
    QuoteTick,
    TradeTick,
    make_quote_tick,
    make_trade_tick,
)


@dataclass
class _InstrumentState:
    cumulative_volume: int | None = None
    quote_key: tuple[float, float, float, float] | None = None


class CtpTickConverter:
    """Stateful converter for CTP ``DepthMarketData`` dictionaries."""

    def __init__(self, timezone: str = "Asia/Shanghai") -> None:
        self.timezone = _load_timezone(timezone)
        self._states: dict[InstrumentId, _InstrumentState] = {}

    def reset(self) -> None:
        self._states.clear()

    def convert(
        self,
        data: Mapping[str, Any],
        instrument_id: InstrumentId,
        meta: InstrumentMeta,
    ) -> list[QuoteTick | TradeTick]:
        ts_event = self._timestamp_ns(data, meta.exchange)
        ts_init = _now_ns()
        state = self._states.setdefault(instrument_id, _InstrumentState())
        events: list[QuoteTick | TradeTick] = []

        bid_price = _ctp_number(data.get("BidPrice1"))
        ask_price = _ctp_number(data.get("AskPrice1"))
        bid_size = _nonnegative_number(data.get("BidVolume1"))
        ask_size = _nonnegative_number(data.get("AskVolume1"))
        if (
            bid_price is not None
            and ask_price is not None
            and bid_size is not None
            and ask_size is not None
            and bid_price <= ask_price
        ):
            quote_key = (bid_price, ask_price, bid_size, ask_size)
            if quote_key != state.quote_key:
                events.append(
                    make_quote_tick(
                        instrument_id=instrument_id,
                        bid_price=bid_price,
                        ask_price=ask_price,
                        bid_size=bid_size,
                        ask_size=ask_size,
                        ts_event=ts_event,
                        ts_init=ts_init,
                        meta=meta,
                    ),
                )
                state.quote_key = quote_key

        cumulative = _optional_nonnegative_integer(data.get("Volume"))
        if cumulative is not None:
            previous = state.cumulative_volume
            state.cumulative_volume = cumulative
            if previous is not None and cumulative > previous:
                last_price = _ctp_number(data.get("LastPrice"))
                if last_price is not None:
                    events.append(
                        make_trade_tick(
                            instrument_id=instrument_id,
                            price=last_price,
                            size=cumulative - previous,
                            trade_id=_synthetic_trade_id(instrument_id, ts_event, cumulative),
                            ts_event=ts_event,
                            ts_init=ts_init,
                            meta=meta,
                        ),
                    )
        return events

    def _timestamp_ns(self, data: Mapping[str, Any], exchange: str) -> int:
        clock = str(data.get("UpdateTime") or "").strip()
        if not clock:
            raise ValueError("CTP snapshot has no UpdateTime")

        exchange = exchange.upper()
        action_day = _day_string(data.get("ActionDay"))
        trading_day = _day_string(data.get("TradingDay"))
        if exchange in {"DCE", "XDCE"} and not action_day:
            day = datetime.now(self.timezone).strftime("%Y%m%d")
        else:
            day = action_day or trading_day
        if not day:
            raise ValueError("CTP snapshot has neither ActionDay nor TradingDay")

        millisec = _nonnegative_integer(data.get("UpdateMillisec", 0))
        if millisec > 999:
            raise ValueError("UpdateMillisec must be between 0 and 999")
        timestamp = datetime.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S")
        timestamp = timestamp.replace(
            tzinfo=self.timezone,
            microsecond=millisec * 1_000,
        )
        return _datetime_to_ns(timestamp)


def _ctp_number(value: Any) -> float | None:
    """Return a positive finite CTP number, filtering DBL_MAX sentinels."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0 or result >= 1e100:
        return None
    return result


def _nonnegative_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0 or result >= 1e100:
        return None
    return result


def _nonnegative_integer(value: Any) -> int:
    result = _nonnegative_number(value)
    if result is None or not result.is_integer():
        raise ValueError(f"expected a non-negative integer, got {value!r}")
    return int(result)


def _optional_nonnegative_integer(value: Any) -> int | None:
    result = _nonnegative_number(value)
    if result is None or not result.is_integer():
        return None
    return int(result)


def _day_string(value: Any) -> str:
    result = str(value or "").strip()
    return result if len(result) == 8 and result.isdigit() else ""


def _load_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name == "Asia/Shanghai":
            # CTP instruments use China Standard Time; this also supports slim
            # Linux images which omit the system tz database and tzdata wheel.
            return timezone(timedelta(hours=8), name="Asia/Shanghai")
        raise


def _datetime_to_ns(timestamp: datetime) -> int:
    utc_timestamp = timestamp.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_timestamp - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _now_ns() -> int:
    return time.time_ns()


def _synthetic_trade_id(
    instrument_id: InstrumentId,
    ts_event: int,
    cumulative: int,
) -> str:
    source = f"{instrument_id}|{ts_event}|{cumulative}".encode()
    return "ctp-" + hashlib.blake2b(source, digest_size=16).hexdigest()
