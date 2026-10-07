"""统一标准化数据，再通过现有行情解析器回放已准备的行。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from math import isfinite

from .catalog import symbol
from .contracts import BarFileKey, BarLoadResult, BarReadSpec, InputIssue
from .session import current_session, fail, read_feather, record_issue


def timestamps(values, timezone="Asia/Shanghai"):
    import pandas as pd
    stamps = pd.to_datetime(values, errors="raise")
    if stamps.isna().any():
        fail("INVALID_TIMESTAMP", "Empty market timestamp")
    return (stamps.dt.tz_localize(timezone) if stamps.dt.tz is None
            else stamps.dt.tz_convert(timezone))


def timestamp_column(path):
    frame = read_feather(path)
    for name in ("datetime", "timestamp"):
        if name in frame.columns:
            return name
    fail("MISSING_FIELD", "Missing datetime/timestamp column", source=path)


def key_from_path(path, asset_kind="future"):
    from datetime import date
    code, separator, label = Path(path).stem.rpartition("_")
    if not separator or len(label) != 8 or not label.isdigit():
        fail("IDENTITY_MISMATCH", "Expected contract_YYYYMMDD.feather", source=path)
    day = date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")
    return BarFileKey(asset_kind, code, day)


def read_bar_frame(path, spec=None, key=None):
    import pandas as pd
    spec = spec or BarReadSpec()
    key = key or key_from_path(path)
    path = Path(path).expanduser().resolve()
    session = current_session()
    fingerprint = session.fingerprint(path) if session is not None else None
    cache_key = ("bar", fingerprint, key, spec)
    if session is not None and cache_key in session.cache:
        old = session.cache[cache_key]
        return replace(old, frame=old.frame.copy(deep=True))
    frame = read_feather(path)
    if not frame.columns.is_unique or frame.empty:
        fail("INVALID_SCHEMA", "Empty file or duplicate columns", source=path)
    frame = frame.reset_index(drop=True)
    mapping = dict(spec.field_mapping)
    frame = frame.rename(columns={raw: canonical for canonical, raw in mapping.items()})
    if not frame.columns.is_unique:
        fail("INVALID_SCHEMA", "Field mapping produces duplicate columns", source=path)
    time_key = spec.timestamp_column or next((n for n in ("datetime", "timestamp") if n in frame), None)
    needed = set(spec.required_fields) | ({time_key} if time_key else {"datetime"})
    if needed - set(frame):
        fail("MISSING_FIELD", f"Missing fields: {sorted(needed - set(frame))}", source=path)
    if "symbol" in frame and not frame.symbol.map(symbol).eq(key.symbol).all():
        fail("IDENTITY_MISMATCH", "Row symbol differs from file identity", source=path)
    if key.venue and "exchange" in frame:
        from .metadata import VENUES
        expected = key.venue.upper()
        if not frame.exchange.map(lambda value: VENUES.get(str(value).strip().upper(), str(value).strip().upper())).eq(expected).all():
            fail("IDENTITY_MISMATCH", "Row exchange differs from declared venue", source=path)
    if "trade_date" in frame and not pd.to_datetime(frame.trade_date, errors="raise").dt.date.eq(key.trading_day).all():
        fail("IDENTITY_MISMATCH", "trade_date differs from file trading day", source=path)
    raw = timestamps(frame[time_key], spec.timezone)
    if spec.trading_day_policy == "day_session" and not raw.dt.date.eq(key.trading_day).all():
        fail("IDENTITY_MISMATCH", "Day-session date differs from file", source=path)
    if spec.trading_day_policy == "exchange" and "trade_date" not in frame:
        # 以上游文件声明的身份为准；缺少交易所日历时，夜盘和节假日
        # 不能仅根据自然日期偏移重建交易日。
        if raw.dt.date.map(lambda d: d > key.trading_day).any():
            fail("IDENTITY_MISMATCH", "Timestamp is later than file trading day", source=path)
        if session is not None and "exchange_trading_day_trusted_from_file_identity" not in session.coverage.assumptions:
            session.coverage.assumptions += ("exchange_trading_day_trusted_from_file_identity",)
    event = raw + pd.Timedelta(seconds=spec.interval_seconds if spec.timestamp_label == "start" else 0)
    ns = event.map(lambda stamp: int(stamp.value))
    if spec.require_minute_alignment and (ns % (spec.interval_seconds * 1_000_000_000)).any():
        fail("INVALID_TIMESTAMP", "分钟快照时间必须对齐整分钟", source=path)
    if ns.duplicated().any():
        fail("DUPLICATE_BAR", "Duplicate completed bar timestamp", source=path)
    if spec.trading_day_policy == "day_session" and not event.dt.date.eq(key.trading_day).all():
        fail("IDENTITY_MISMATCH", "分钟时间转换后跨日期", source=path)
    if spec.available_column and spec.available_column not in frame:
        fail("MISSING_FIELD", f"Missing {spec.available_column}", source=path)
    available_values = frame[spec.available_column].copy() if spec.available_column else None
    frame["source_timestamp"] = raw
    frame["datetime"] = event
    frame["event_ns"] = ns
    frame["bar_ns"] = ns
    frame["trading_day"] = key.trading_day
    if spec.available_column:
        available = timestamps(available_values, spec.timezone)
        frame["available_ns"] = available.map(lambda stamp: int(stamp.value))
    else:
        frame["available_ns"] = ns
    if frame.available_ns.lt(ns).any():
        fail("INVALID_TIMESTAMP", "Availability precedes completed minute", source=path)
    frame["available_datetime"] = pd.to_datetime(frame.available_ns, utc=True).dt.tz_convert(spec.timezone)
    issues = []
    for name in spec.required_fields:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        finite = frame[name].map(lambda v: not pd.isna(v) and isfinite(float(v)))
        valid = finite & (frame[name].ge(0) if name in {"volume", "open_interest"} else frame[name].gt(0))
        if name == "volume" and spec.require_integer_volume:
            valid &= frame[name].mod(1).eq(0)
        for index in frame.index[~valid]:
            issue = InputIssue("INVALID_BAR_FIELDS", f"Invalid {name}",
                "WARNING" if spec.value_policy == "research_audited" else "ERROR",
                str(path), key.symbol, key.trading_day, name, int(index) + 1)
            if spec.value_policy != "research_audited":
                record_issue(issue)
                from .contracts import InputError
                raise InputError(issue)
            issues.append(issue)
    if {"open", "high", "low", "close"} <= set(spec.required_fields):
        invalid = frame.high.lt(frame[["open", "low", "close"]].max(axis=1)) | frame.low.gt(frame[["open", "high", "close"]].min(axis=1))
        if invalid.any():
            fail("INVALID_OHLC", "OHLC relationship invalid", source=path,
                symbol=key.symbol, trading_day=key.trading_day, row=int(frame.index[invalid][0]) + 1)
    if spec.value_policy == "research_audited":
        frame["input_quality"] = ""
        for issue in issues:
            frame.at[issue.row - 1, "input_quality"] = "INVALID_BAR_FIELDS"
            record_issue(issue)
    if "volume" in frame:
        frame["zero_volume_bar"] = frame.volume.eq(0)
    frame = frame.sort_values(["available_ns", "event_ns"], kind="stable").reset_index(drop=True)
    if not frame.event_ns.is_monotonic_increasing:
        fail("UNSUPPORTED_CAPABILITY", "Out-of-order arrival needs stage-two late-frame handling", source=path)
    result = BarLoadResult(key, path, frame, spec, tuple(issues))
    if session is not None:
        session.cache[cache_key] = result
        details = {"identity": str(key), "rows": len(frame),
            "first_ns": result.first_ns, "last_ns": result.last_ns,
            "timestamp_label": spec.timestamp_label, "value_policy": spec.value_policy,
            "available_policy": spec.available_column or "completed_at_end",
            "gap_check": "interval_gaps_not_equivalent_to_missing_trading_minutes"}
        configurations = session.coverage.files.get(str(path), {}).get("specifications", [])
        configuration = {"fields": list(spec.required_fields), "timezone": spec.timezone,
            "timestamp_label": spec.timestamp_label, "value_policy": spec.value_policy,
            "available_column": spec.available_column, "trading_day_policy": spec.trading_day_policy}
        if configuration not in configurations:
            configurations.append(configuration)
        details["specifications"] = configurations
        session.coverage.files[str(path)] = details
        start, end = session.coverage.requested
        field = "dependency_days" if ((start is not None and key.trading_day < start)
            or (end is not None and key.trading_day > end)) else "actual_days"
        setattr(session.coverage, field,
            tuple(sorted(set(getattr(session.coverage, field)) | {key.trading_day})))
    return replace(result, frame=frame.copy(deep=True))


class PreparedFrameReader:
    suffixes = frozenset({".feather"})

    def __init__(self, frame):
        self.frame = frame.copy(deep=True)

    def read(self, path):
        for index, row in enumerate(self.frame.to_dict("records"), 1):
            yield index, row


def add_bar_source(feed, path, instrument_id, *, spec=None, key=None):
    """将已准备的行注册到现有行情源，不创建或启动行情源。"""
    if key is None:
        key = key_from_path(path)
        key = replace(key, venue=str(instrument_id.venue))
    result = read_bar_frame(path, spec, key)
    add_prepared_bar_source(feed, result, instrument_id)
    return result


def add_prepared_bar_source(feed, result, instrument_id):
    """消费场景已准备的结果，不重复读取或调整时间。"""
    from bomber.framework.market.replay.parsers.bar import FixedInstrumentBarParser
    if result.spec.value_policy != "execution_strict":
        fail("INVALID_SOURCE_PURPOSE", "Standard execution Bar requires execution_strict OHLCV", source=result.path)
    if result.spec.interval_seconds != 60:
        fail("UNSUPPORTED_CAPABILITY", "Existing demo bindings require one-minute bars", source=result.path)
    if result.key.venue and result.key.venue != str(instrument_id.venue):
        fail("IDENTITY_MISMATCH", "Prepared source venue differs from instrument", source=result.path)
    if result.key.symbol != symbol(instrument_id.symbol.value):
        fail("IDENTITY_MISMATCH", "Prepared source symbol differs from instrument", source=result.path)
    if "exchange" in result.frame:
        from .metadata import VENUES
        expected = str(instrument_id.venue).upper()
        if not result.frame.exchange.map(lambda value: VENUES.get(str(value).strip().upper(), str(value).strip().upper())).eq(expected).all():
            fail("IDENTITY_MISMATCH", "Prepared rows differ from instrument venue", source=result.path)
    parser = FixedInstrumentBarParser(instrument_id, timestamp="datetime",
        available_nanoseconds="available_ns", timezone=result.spec.timezone)
    feed.add_source(result.path, PreparedFrameReader(result.frame), parser, "bar")


def first_ns(path, spec=None):
    spec = spec or BarReadSpec(required_fields=("close",), value_policy="close_strict")
    return read_bar_frame(path, spec).first_ns
