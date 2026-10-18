"""Parser contracts, error context and common scalar conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from market.basic.base import (
    Bar,
    CustomBar,
    DataType,
    InstrumentId,
    InstrumentMeta,
    QuoteTick,
    TradeTick,
)


MarketEvent = TradeTick | QuoteTick | Bar | CustomBar


class DataLoadError(ValueError):
    """A file row cannot be converted into market data."""


@dataclass(frozen=True)
class ParsedEvent:
    data_type: DataType
    instrument_id: InstrumentId
    payload: MarketEvent
    bar_spec: str | None = None


@dataclass(frozen=True)
class ParserContext:
    path: Path
    line: int
    get_meta: Callable[[InstrumentId], InstrumentMeta | None]

    def require_meta(self, instrument_id: InstrumentId) -> InstrumentMeta:
        meta = self.get_meta(instrument_id)
        if meta is None:
            raise self.error(f"instrument is not registered: {instrument_id}")
        return meta

    def error(self, message: str) -> DataLoadError:
        return DataLoadError(f"{self.path}:{self.line}: {message}")


class RowParser(Protocol):
    """Convert source-specific rows into normalized market events."""

    def reset(self) -> None: ...

    def parse(
        self,
        row: Mapping[str, Any],
        context: ParserContext,
    ) -> Iterable[ParsedEvent]: ...


def required(row: Mapping[str, Any], column: str, context: ParserContext) -> str:
    result = row.get(column)
    if result is None or (isinstance(result, str) and not result.strip()):
        raise context.error(f"missing column value: {column}")
    return str(result).strip()


def number(row: Mapping[str, Any], column: str, context: ParserContext) -> float:
    raw = row.get(column)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise context.error(f"missing column value: {column}")
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise context.error(f"{column} is not numeric: {raw!r}") from exc
    if not math.isfinite(result):
        raise context.error(f"{column} must be finite")
    return result


def positive(row: Mapping[str, Any], column: str, context: ParserContext) -> float:
    result = number(row, column, context)
    if result <= 0:
        raise context.error(f"{column} must be positive")
    return result


def nonnegative(row: Mapping[str, Any], column: str, context: ParserContext) -> float:
    result = number(row, column, context)
    if result < 0:
        raise context.error(f"{column} must be nonnegative")
    return result


def integer_nonnegative(
    row: Mapping[str, Any],
    column: str,
    context: ParserContext,
) -> int:
    result = nonnegative(row, column, context)
    if not result.is_integer():
        raise context.error(f"{column} must be an integer")
    return int(result)


def parse_iso_timestamp(
    value: str,
    context: ParserContext,
    default_timezone: tzinfo | None = None,
) -> int:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise context.error(f"invalid ISO timestamp: {value!r}") from exc
    if timestamp.tzinfo is None:
        if default_timezone is None:
            raise context.error("bar timestamp must include a timezone")
        timestamp = timestamp.replace(tzinfo=default_timezone)
    return datetime_to_ns(timestamp)


def datetime_to_ns(timestamp: datetime) -> int:
    utc_timestamp = timestamp.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_timestamp - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )
