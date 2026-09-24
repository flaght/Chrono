"""Configuration and exact schemas for DolphinDB market stream tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StreamKind = Literal["tick", "bar"]

TICK_COLUMNS = (
    "date",
    "Code",
    "TradingDay",
    "ActionDay",
    "tradeTime",
    "OpenPrice",
    "HighestPrice",
    "LowestPrice",
    "LastPrice",
    "AveragePrice",
    "Volume",
    "Turnover",
    "BidPrice1",
    "BidVolume1",
    "AskPrice1",
    "AskVolume1",
    "PreSettlementPrice",
    "PreClosePrice",
    "PreOpenInterest",
    "OpenInterest",
    "UpperLimitPrice",
    "LowerLimitPrice",
    "UpdateTime",
    "UpdateMillisec",
    "receivedTime",
    "perPenetrationTime",
)

BAR_COLUMNS = (
    "minTime",
    "date",
    "Code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "prev_settlement",
    "volume",
    "turnover",
    "pct_change",
    "open_interest",
    "prev_open_interest",
    "updateTime",
)


@dataclass(frozen=True)
class DolphinDbStreamSpec:
    kind: StreamKind
    table_name: str
    action_name: str
    columns: tuple[str, ...]
    offset: int = -1
    resub: bool = True
    batch_size: int = 0
    throttle: float = 0.01
    filter: Any = field(default=None, compare=False, hash=False, repr=False)
    bar_spec: str = "1-MINUTE"

    def __post_init__(self) -> None:
        if self.kind not in {"tick", "bar"}:
            raise ValueError(f"unsupported DolphinDB stream kind: {self.kind}")
        if not self.table_name or not self.action_name:
            raise ValueError("table_name and action_name are required")
        expected = TICK_COLUMNS if self.kind == "tick" else BAR_COLUMNS
        if self.columns != expected:
            raise ValueError(
                f"{self.kind} columns do not match the exp5 schema; "
                f"expected {expected!r}",
            )
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.throttle < 0:
            raise ValueError("throttle must be non-negative")

    @classmethod
    def tick(
        cls,
        table_name: str,
        action_name: str = "bomberTick",
        **options: Any,
    ) -> "DolphinDbStreamSpec":
        return cls("tick", table_name, action_name, TICK_COLUMNS, **options)

    @classmethod
    def bar(
        cls,
        table_name: str,
        action_name: str = "bomberBar",
        **options: Any,
    ) -> "DolphinDbStreamSpec":
        return cls("bar", table_name, action_name, BAR_COLUMNS, **options)


@dataclass(frozen=True)
class DolphinDbConfig:
    host: str
    port: int
    username: str
    password: str = field(repr=False)
    streams: tuple[DolphinDbStreamSpec, ...] = ()
    streaming_port: int = 0
    keep_alive_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("DolphinDB host is required")
        if not 0 < self.port <= 65535:
            raise ValueError("DolphinDB port must be between 1 and 65535")
        if not self.streams:
            raise ValueError("at least one DolphinDB stream is required")
        topics = [(stream.table_name, stream.action_name) for stream in self.streams]
        if len(topics) != len(set(topics)):
            raise ValueError("DolphinDB table/action subscription topics must be unique")
