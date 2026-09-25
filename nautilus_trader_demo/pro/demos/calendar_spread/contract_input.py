"""单品种跨期回测所需的真实合约 Bar 与合约元数据装配。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from market.basic.base import InstrumentMeta
from strategy import CtpFuturesBasicProfile

VENUES = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE",
          "XZCE": "CZCE", "CZCE": "CZCE", "XSIE": "INE", "INE": "INE"}
DAY_NS = 86_400_000_000_000


def bar_paths(root: Path, required: set[tuple[date, str]],
              optional: set[tuple[date, str]] | None = None) -> dict[tuple[date, str], Path]:
    """必需的当日双腿缺失即报错；换月旧腿仅在存在时加载。"""
    optional = optional or set()
    expected = {f"{symbol.lower()}_{day:%Y%m%d}.feather": (day, symbol)
                for day, symbol in required | optional}
    found = {}
    for path in root.rglob("*.feather"):
        key = expected.get(path.name.lower())
        if key is None:
            continue
        if key in found:
            raise ValueError(f"重复真实合约 Bar: {found[key]}, {path}")
        found[key] = path
    missing = required - found.keys()
    if missing:
        day, symbol = min(missing)
        raise FileNotFoundError(f"{day}/{symbol} 缺少真实合约 Bar: {root}")
    return found


def timestamp_column(path: Path) -> str:
    import pyarrow as pa
    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"{path} 缺少 datetime/timestamp 列")


def venue(basic: pd.DataFrame, product: str) -> str:
    rows = basic.loc[basic["code"].astype(str).str.strip().str.upper() == product]
    if rows.empty:
        raise ValueError(f"fut_basic 没有 {product}")
    candidates = {VENUES.get(str(value).strip().upper()) for value in rows["exchangeCD"]}
    if None in candidates or len(candidates) != 1:
        raise ValueError(f"{product} 的交易所不受支持或不唯一: {candidates}")
    return candidates.pop()


def _day_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def instrument(basic: pd.DataFrame, profile: CtpFuturesBasicProfile,
               product: str, symbol: str, margin_init: Decimal, margin_maint: Decimal):
    rows = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol.lower()]
    if len(rows) != 1:
        raise ValueError(f"fut_basic 需要唯一合约记录: {symbol}，实际 {len(rows)}")
    row = rows.iloc[0]
    if str(row["code"]).strip().upper() != product or VENUES.get(
        str(row["exchangeCD"]).strip().upper()
    ) != str(profile.venue):
        raise ValueError(f"{symbol} 的品种或交易所不匹配")
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
