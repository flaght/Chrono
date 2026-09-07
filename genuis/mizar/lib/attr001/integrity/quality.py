from datetime import timedelta
from typing import Any, Mapping, Dict
from dataclasses import asdict, dataclass
import pandas as pd
import pdb

from lib.attr001.ftd001 import *

@dataclass(frozen=True)
class IntegrityMetrics:
    bar_count: int
    missing_bar_count: int
    missing_bar_ratio: float
    duplicate_bar_count: int
    nan_count: int
    invalid_price_count: int
    zero_volume_count: int
    zero_openint_count: int
    instrument_mismatch_count: int
    start_time: str
    end_time: str
    source_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_freq(freq: str) -> str:
    normalized = freq.strip().lower()
    if normalized in {"1min", "1m", "min", "minute"}:
        return "1min"
    raise ValueError(f"Unsupported frequency: {freq}")


def _count_invalid_prices(df, price_columns) -> int:
    available_price_columns = [
        column for column in price_columns if column in df.columns
    ]
    if not available_price_columns:
        return 0
    invalid_mask = pd.Series(False, index=df.index)
    for column in available_price_columns:
        invalid_mask |= df[column].isna() | (df[column] <= 0)

    return int(invalid_mask.sum())


def _count_zero_flow(df: pd.DataFrame, name) -> int:
    if name not in df.columns:
        return 0
    return int(df[name].fillna(0).eq(0).sum())


def _count_instrument_mismatch(df: pd.DataFrame) -> int:
    if "symbol" in df.columns:
        return int((df["symbol"].astype(str)
                    != df["symbol"].astype(str)).sum())
    return max(int(df["symbol"].nunique()) - 1, 0)


def _count_missing_bars(df: pd.DataFrame, freq: str, trading_sessions) -> int:
    return len(_find_missing_datetimes(df, freq, trading_sessions))


def _find_missing_datetimes(df: pd.DataFrame, freq: str,
                            trading_sessions) -> int:
    if len(df) <= 1:
        return 0

    normalized_freq = _normalize_freq(freq)

    observed = pd.DatetimeIndex(
        df["trade_time"].drop_duplicates().sort_values())
    expected = pd.DatetimeIndex([])

    unique_dates = pd.Index(observed.normalize().unique()).sort_values()
    for trade_date in unique_dates:
        for start_text, end_text in trading_sessions:
            session_start = pd.Timestamp(f"{trade_date.date()} {start_text}")
            session_end = pd.Timestamp(f"{trade_date.date()} {end_text}")

            if session_end < session_start:
                session_end += timedelta(days=1)

            expected = expected.union(
                pd.date_range(start=session_start,
                              end=session_end,
                              freq=normalized_freq))

    observed_in_sessions = observed.intersection(expected)
    return expected.difference(observed_in_sessions)


def judge_integrity(metrics) -> str:
    if isinstance(metrics, IntegrityMetrics):
        metrics = metrics.to_dict()
    judge = {}
    if metrics["missing_bar_ratio"] > 0.005:
        judge["missing_bar_ratio"] = ("FAIL", "Miss Bar Ratio")
    if metrics["duplicate_bar_count"] > 0:
        judge["missing_bar_ratio"] = ("FAIL", "Duplicate Bar Count")
    if metrics["invalid_price_count"] > 0:
        judge["missing_bar_ratio"] = ("FAIL", "Invalid Price Count")
    if metrics["missing_bar_ratio"] > 0.001:
        judge["missing_bar_ratio"] = ("WARN", "Miss Bar Ratio")
    return judge


def metrics(data,
            name,
            price_columns=['open', 'close', 'high', 'low', 'vwap'],
            flow_columns=['volume', 'openint'],
            trading_sessions=(
                ("21:00", "23:00"),
                ("09:00", "10:15"),
                ("10:30", "11:30"),
                ("13:30", "15:00"),
            )):

    prepared = filter_trading_time(data=data,
                                   trading_sessions=trading_sessions)
    bar_count = len(prepared)
    duplicate_bar_count = int(
        prepared.duplicated(subset=["trade_time", "code", "symbol"]).sum())
    nan_count = int(prepared.isna().sum().sum())

    invalid_price_count = _count_invalid_prices(df=prepared,
                                                price_columns=price_columns)
    zero_volume_count = _count_zero_flow(df=prepared, name=flow_columns[0])
    zero_openint_count = _count_zero_flow(df=prepared, name=flow_columns[1])

    instrument_mismatch_count = _count_instrument_mismatch(df=prepared)

    missing_bar_count = _count_missing_bars(df=prepared,
                                            freq='1m',
                                            trading_sessions=trading_sessions)
    missing_bar_ratio = missing_bar_count / (
        bar_count + missing_bar_count) if bar_count else 0.0

    start_time = None if prepared.empty else prepared["trade_time"].min(
    ).isoformat()
    end_time = None if prepared.empty else prepared["trade_time"].max(
    ).isoformat()

    return IntegrityMetrics(
        bar_count=bar_count,
        missing_bar_count=missing_bar_count,
        missing_bar_ratio=missing_bar_ratio,
        duplicate_bar_count=duplicate_bar_count,
        nan_count=nan_count,
        invalid_price_count=invalid_price_count,
        zero_openint_count=zero_openint_count,
        zero_volume_count=zero_volume_count,
        instrument_mismatch_count=instrument_mismatch_count,
        start_time=start_time,
        end_time=end_time,
        source_name=name)


def find_missing_bars(data, freq, trading_sessions):
    missing_datetimes = _find_missing_datetimes(data, freq, trading_sessions)
    missing_df = pd.DataFrame({"trade_time": missing_datetimes})
    missing_df["trading_day"] = missing_df["trade_time"].dt.strftime(
        "%Y-%m-%d")
    missing_df["session_hint"] = missing_df["trade_time"].dt.strftime("%H:%M")
    return missing_df


def find_duplicate_bars(data: Any) -> pd.DataFrame:
    duplicate_mask = data.duplicated(
        subset=["trade_time", "code", "symbol"],
        keep=False,
    )
    return data.loc[duplicate_mask].sort_values(
        ["trade_time", "code", "symbol"]).reset_index(drop=True)


def find_invalid_price(data, price_columns) -> pd.DataFrame:
    available_price_columns = [
        column for column in price_columns if column in data.columns
    ]
    if not available_price_columns:
        return data.iloc[0:0].copy()

    invalid_mask = pd.Series(False, index=data.index)
    for column in available_price_columns:
        invalid_mask |= data[column].isna() | (data[column] <= 0)

    invalid_rows = data.loc[invalid_mask].copy()
    invalid_rows["invalid_price_fields"] = invalid_rows[
        available_price_columns].apply(
            lambda row: [
                column for column in available_price_columns
                if pd.isna(row[column]) or row[column] <= 0
            ],
            axis=1,
        )
    return invalid_rows.reset_index(drop=True)


def find_zero_value(data: Any, flow_column: str) -> pd.DataFrame:
    if flow_column not in data.columns:
        return data.iloc[0:0].copy()
    mask = data[flow_column].fillna(0).eq(0)
    return data.loc[mask].reset_index(drop=True)


def diagnostics(data,
                name,
                price_columns=['open', 'close', 'high', 'low', 'vwap'],
                flow_columns=['volume', 'openint'],
                trading_sessions=(("21:00", "23:00"), ("09:00", "10:15"),
                                  ("10:30", "11:30"), ("13:30", "15:00"))):

    metrics1 = metrics(data=data,
                       name=name,
                       price_columns=price_columns,
                       flow_columns=flow_columns,
                       trading_sessions=trading_sessions)

    prepared = filter_trading_time(data=data,
                                   trading_sessions=trading_sessions)

    missing_bars = find_missing_bars(data=prepared,
                                     freq='1m',
                                     trading_sessions=trading_sessions)
    duplicate_bars = find_duplicate_bars(data=prepared)

    invalid_price = find_invalid_price(data=prepared,
                                       price_columns=price_columns)

    zero_volume = find_zero_value(data=prepared, flow_column=flow_columns[0])
    zero_openint = find_zero_value(data=prepared, flow_column=flow_columns[1])

    return {
        "summary": {
            **metrics1.to_dict(),
            "status": judge_integrity(metrics1),
        },
        "missing_bars": missing_bars,
        "duplicate_bars": duplicate_bars,
        "invalid_price": invalid_price,
        "zero_volume": zero_volume,
        "zero_openint": zero_openint,
    }
