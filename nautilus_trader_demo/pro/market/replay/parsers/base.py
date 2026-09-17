import math
from typing import Mapping, Any
from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo
from market.basic.base import InstrumentId, DataType, QuoteTick, TradeTick
from market.replay.base import ParsedEvent, ParserContext

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
