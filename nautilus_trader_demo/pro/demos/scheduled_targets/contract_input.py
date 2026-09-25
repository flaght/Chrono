"""按目标键、fut_basic 和真实合约 Bar 装配 CTP 期货输入。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import re

import pandas as pd

from market.basic.base import InstrumentMeta
from strategy import CtpFuturesBasicProfile

VENUES = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE",
          "XZCE": "CZCE", "CZCE": "CZCE", "XSIE": "INE", "INE": "INE"}
DAY_NS = 86_400_000_000_000
CONTRACT = re.compile(r"^([A-Za-z]+)(\d+)$")
REQUIRED_BASIC = {"symbol", "code", "exchangeCD", "contMultNum", "minChgPriceNum",
                  "listDate", "lastTradeDate"}


def contract_rows(basic: pd.DataFrame, keys: set[str]) -> dict[str, pd.Series]:
    if REQUIRED_BASIC - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(REQUIRED_BASIC - set(basic.columns))}")
    result = {}
    for key in keys:
        symbol, separator, requested_venue = key.partition(".")
        match = CONTRACT.fullmatch(symbol)
        if match is None or (separator and not requested_venue):
            raise ValueError(f"目标键须是真实期货合约或合约.交易所: {key}")
        rows = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol.lower()]
        if len(rows) != 1:
            raise ValueError(f"fut_basic 需要唯一合约记录: {symbol}，实际 {len(rows)}")
        row = rows.iloc[0]
        venue = VENUES.get(str(row["exchangeCD"]).strip().upper())
        if venue is None or (separator and requested_venue.upper() != venue):
            raise ValueError(f"{key} 的交易所与 fut_basic 不匹配")
        if str(row["code"]).strip().upper() != match.group(1).upper():
            raise ValueError(f"{key} 的品种与 fut_basic 不匹配")
        result[key] = row
    return result


def venue_of(row: pd.Series) -> str:
    return VENUES[str(row["exchangeCD"]).strip().upper()]


def symbol_of(row: pd.Series) -> str:
    return str(row["symbol"]).strip()


def bar_files(root: Path, symbols: set[str], start_day: date,
              end_day: date) -> dict[str, tuple[Path, ...]]:
    found: dict[str, dict[date, Path]] = {symbol.lower(): {} for symbol in symbols}
    for path in root.rglob("*.feather"):
        symbol, separator, label = path.stem.rpartition("_")
        if not separator or symbol.lower() not in found or len(label) != 8 or not label.isdigit():
            continue
        day = date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")
        if start_day <= day <= end_day:
            key = symbol.lower()
            if day in found[key]:
                raise ValueError(f"{symbol}/{day} 重复 Bar 文件")
            found[key][day] = path
    missing = sorted(symbol for symbol, rows in found.items() if not rows)
    if missing:
        raise FileNotFoundError(f"{root} 在请求区间没有这些合约的 Bar: {missing}")
    return {symbol: tuple(rows[day] for day in sorted(rows)) for symbol, rows in found.items()}


def timestamp_column(path: Path) -> str:
    import pyarrow as pa
    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"{path} 缺少 datetime/timestamp 列")


def _day_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def make_instrument(row: pd.Series, profile: CtpFuturesBasicProfile,
                    margin_init: Decimal, margin_maint: Decimal):
    symbol = symbol_of(row)
    product = str(row["code"]).strip().upper()
    multiplier = Decimal(str(row["contMultNum"]))
    increment = Decimal(str(row["minChgPriceNum"]))
    if not multiplier.is_finite() or multiplier <= 0 or not increment.is_finite() or increment <= 0:
        raise ValueError(f"{symbol} 的乘数或最小价格变动无效")
    precision = max(0, -increment.normalize().as_tuple().exponent)
    contract = profile.make_instrument(
        symbol, underlying=product.lower(), price_precision=precision,
        price_increment=increment, multiplier=multiplier,
        activation_ns=_day_ns(row["listDate"]),
        expiration_ns=_day_ns(row["lastTradeDate"]) + DAY_NS,
        margin_init=margin_init, margin_maint=margin_maint,
    )
    meta = InstrumentMeta(
        contract.id, price_precision=precision, size_precision=0,
        price_increment=increment, multiplier=multiplier,
        currency="CNY", exchange=str(profile.venue),
    )
    return contract, meta, multiplier
