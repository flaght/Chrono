"""公共输入契约，不导入 pandas、引擎、账户或网关。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

# 包重命名后，序列化诊断结构仍保持兼容。
SCHEMA_VERSION = "inputs-v1"
ASSET_KINDS = frozenset({"future", "option", "index"})


@dataclass(frozen=True)
class InputIssue:
    code: str
    message: str
    severity: str = "ERROR"
    source: str | None = None
    symbol: str | None = None
    trading_day: date | None = None
    field: str | None = None
    row: int | None = None


class InputError(ValueError):
    def __init__(self, issue: InputIssue):
        self.issue = issue
        super().__init__(f"{issue.code}: {issue.message}" +
                         (f" [{issue.source}]" if issue.source else ""))


@dataclass(frozen=True)
class DataPaths:
    role: Path | None = None
    fut: Path | None = None
    opt: Path | None = None
    index: Path | None = None
    contract_struct: Path | None = None
    fut_basic: Path | None = None
    opt_basic: Path | None = None
    calendar: Path | None = None
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


@dataclass(frozen=True, order=True)
class BarFileKey:
    asset_kind: str
    symbol: str
    trading_day: date
    venue: str = ""

    def __post_init__(self):
        if self.asset_kind not in ASSET_KINDS or not self.symbol:
            raise ValueError("Invalid bar file identity")
        object.__setattr__(self, "symbol", self.symbol.strip().split(".")[0].upper())


@dataclass(frozen=True)
class BarReadSpec:
    required_fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    timestamp_column: str | None = None
    timezone: str = "Asia/Shanghai"
    timestamp_label: str = "end"
    interval_seconds: int = 60
    trading_day_policy: str = "exchange"
    value_policy: str = "execution_strict"
    available_column: str | None = None
    volume_semantics: str = "incremental"
    require_integer_volume: bool = True
    require_minute_alignment: bool = True
    field_mapping: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.timestamp_label not in {"start", "end"} or self.interval_seconds < 1:
            raise ValueError("Invalid completed-bar time specification")
        if self.trading_day_policy not in {"exchange", "day_session"}:
            raise ValueError("Unknown trading-day policy")
        if self.value_policy not in {"execution_strict", "research_audited", "close_strict"}:
            raise ValueError("Unknown value policy")
        if self.volume_semantics != "incremental":
            raise ValueError("Export cumulative volume to increments with an explicit reset policy first")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("Duplicate required fields")
        if self.value_policy == "execution_strict" and not {"open", "high", "low", "close", "volume"} <= set(self.required_fields):
            raise ValueError("Execution bars require OHLCV")


@dataclass(frozen=True)
class DataRequirements:
    assets: tuple[str, ...]
    products: tuple[str, ...] = ()
    contracts: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    fields: tuple[str, ...] = ("close",)
    bar_spec: str = "1-MINUTE"
    reference_services: tuple[str, ...] = ()
    synchronization: str = "complete_frame"

    def __post_init__(self):
        if not self.assets or not set(self.assets) <= ASSET_KINDS:
            raise ValueError("Invalid required assets")


@dataclass(frozen=True)
class InstrumentSpec:
    asset_kind: str
    symbol: str
    product: str
    venue: str
    currency: str
    listed: date
    last_trade: date
    tick: Decimal | None = None
    multiplier: Decimal | None = None
    expiry: date | None = None
    option_kind: str | None = None
    strike: Decimal | None = None
    underlying: str | None = None
    month: str | None = None
    available_ns: int | None = None
    source_version: str | None = None


@dataclass(frozen=True)
class BarLoadResult:
    key: BarFileKey
    path: Path
    frame: Any
    spec: BarReadSpec
    issues: tuple[InputIssue, ...] = ()

    @property
    def first_ns(self):
        return int(self.frame.event_ns.min())

    @property
    def last_ns(self):
        return int(self.frame.event_ns.max())


@dataclass(frozen=True)
class BindingSpec:
    data_key: str
    instrument_key: str
    data_type: str = "BAR"
    bar_spec: str = "1-MINUTE"


@dataclass
class CoverageReport:
    requested: tuple[date | None, date | None] = (None, None)
    actual_days: tuple[date, ...] = ()
    dependency_days: tuple[date, ...] = ()
    required_missing: tuple[BarFileKey, ...] = ()
    optional_missing: tuple[BarFileKey, ...] = ()
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ("completed_bar_available_at_end_unless_available_column_is_declared",)
    calendar_source: str | None = None


@dataclass(frozen=True)
class InputPlan:
    required: tuple[BarFileKey, ...]
    optional: tuple[BarFileKey, ...] = ()
    requested: tuple[date | None, date | None] = (None, None)
    bindings: tuple[BindingSpec, ...] = ()


@dataclass(frozen=True)
class InputContext:
    instruments: Mapping[str, InstrumentSpec] = field(default_factory=dict)
    references: Mapping[str, Any] = field(default_factory=dict)
    calendar: Any = None

    def __post_init__(self):
        object.__setattr__(self, "instruments", MappingProxyType(dict(self.instruments)))
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))


@dataclass(frozen=True)
class SourceSpec:
    result: BarLoadResult
    purpose: str = "execution"

    def reader(self):
        from .bars import PreparedFrameReader
        return PreparedFrameReader(self.result.frame)


@dataclass(frozen=True)
class HistoricalInputBundle:
    context: InputContext
    sources: tuple[SourceSpec, ...]
    bindings: tuple[BindingSpec, ...]
    coverage: CoverageReport
