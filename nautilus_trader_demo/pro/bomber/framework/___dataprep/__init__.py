"""可组合的数据准备能力；导入本包不会建立连接或启动运行。"""
from .contracts import (
    BarFileKey, BarLoadResult, BarReadSpec, BindingSpec, CoverageReport,
    DataPaths, DataRequirements, HistoricalInputBundle, InputContext,
    InputError, InputIssue, InputPlan, InstrumentSpec, SCHEMA_VERSION, SourceSpec,
)
from .paths import resolve_paths, resolve_futures_args, resolve_option_args
from .session import InputSession, input_session, write_input_reports

__all__ = [
    "BarFileKey", "BarLoadResult", "BarReadSpec", "BindingSpec", "CoverageReport",
    "DataPaths", "DataRequirements", "HistoricalInputBundle", "InputContext",
    "InputError", "InputIssue", "InputPlan", "InstrumentSpec", "SourceSpec",
    "SCHEMA_VERSION", "InputSession", "input_session", "write_input_reports",
    "resolve_paths", "resolve_futures_args", "resolve_option_args",
]
